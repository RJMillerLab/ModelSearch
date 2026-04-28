"""
Simplified backend for ModelSearch Demo.

Data flow:
- write `job_meta.json`
- write `query2modelcard.json`
- write `card2tab2card_<search_type>.json`
- build API responses directly from those files

No QA.
No evaluation.
No compatibility layers.
"""

import atexit, html, json, math, os, random, re, string, subprocess, sys, threading, time
from datetime import datetime
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd
from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context
from flask_cors import CORS

from src.config import *
from src.demo.job_schema import JobMeta, JobPaths, Query2ModelCardFile, Query2Tab2CardFullMap
from src.integration.table_integration import GroupTableIntegrater, TableIntegrater
from src.search.ir_searcher import DenseSearcher, SparseSearcher
from src.search.query2modelcard import Query2ModelCardSearch
from src.search.query2tab2card import Query2Tab2CardSearch
from src.utils import _paths_for_resource_set, preview_from_local, read_csv_robust, resolve_table_path


MAX_TABLE_PAGE_ROWS = 25000
MAX_TABLE_PAGE_COLS = 200
MAX_TABLE_PREVIEW_JSON_ROWS = 500
MAX_TABLE_PREVIEW_JSON_COLS = 50

TABLE_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{{ title }}</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; background: #f5f5f5; color: #212529; }
    h1 { font-size: 1.15rem; margin: 0 0 8px 0; color: #333; }
    .path { font-size: 12px; color: #666; word-break: break-all; margin-bottom: 12px; font-family: ui-monospace, monospace; }
    .note { font-size: 12px; color: #856404; background: #fff3cd; border: 1px solid #ffc107; padding: 8px 10px; border-radius: 6px; margin-bottom: 12px; }
    .wrap { overflow: auto; max-height: calc(100vh - 140px); border: 1px solid #dee2e6; border-radius: 6px; background: #fff; }
    table { border-collapse: collapse; font-size: 12px; }
    th, td { border: 1px solid #dee2e6; padding: 6px 8px; text-align: left; vertical-align: top; }
    thead th { position: sticky; top: 0; background: #f8f9fa; z-index: 2; box-shadow: 0 1px 0 #dee2e6; white-space: nowrap; }
    td { max-width: 28rem; white-space: pre-wrap; word-break: break-word; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="path">{{ path_display }}</div>
  {% if note %}<div class="note">{{ note }}</div>{% endif %}
  <div class="wrap">{{ table_html|safe }}</div>
</body>
</html>"""

MARKDOWN_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{{ title }}</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f8fa; color: #24292f; }
    .wrap { max-width: 1400px; margin: 0 auto; padding: 24px; }
    .meta { margin-bottom: 16px; padding: 12px 14px; background: #fff; border: 1px solid #d0d7de; border-radius: 8px; font-size: 13px; }
    .doc { background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 24px; overflow-x: auto; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 12px; }
    th, td { border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; vertical-align: top; }
    thead th { background: #f6f8fa; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    pre { background: #f6f8fa; padding: 12px; border-radius: 6px; overflow: auto; }
    a { color: #0969da; text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="meta">
      <div><strong>{{ title }}</strong></div>
      <div style="margin-top:4px;">Saved markdown: <code>{{ path_display }}</code></div>
    </div>
    <div class="doc">{{ content_html|safe }}</div>
  </div>
</body>
</html>"""

app = Flask(__name__)
CORS(app)

jobs: Dict[str, Dict[str, Any]] = {}
search_runtime: Optional[Dict[str, Any]] = None
evaluation_runs: Dict[str, Dict[str, Any]] = {}
DEFAULT_EXTRA_PRESET_PATH = os.path.join(OUTPUT_DIR, "query", "query_rewrite_polished.jsonl")


def init_search_runtime() -> None:
    global search_runtime
    if search_runtime is not None:
        return
    _, sparse_index_full, _ = _paths_for_resource_set(["hugging", "github", "arxiv"])
    _, _, table_db_path = _paths_for_resource_set(["hugging"])
    search_runtime = {
        "table_resources": ["hugging"],
        "con_data": duckdb.connect(os.path.abspath(str(table_db_path)), read_only=True),
        "dense_full": DenseSearcher(emb_npz_path=EMB_NPZ),
        "dense_wtable": DenseSearcher(emb_npz_path=EMB_NPZ_HUGGING),
        "sparse_full": SparseSearcher(index_path=sparse_index_full),
    }


def ensure_search_runtime() -> Dict[str, Any]:
    init_search_runtime()
    return search_runtime

def close_search_runtime() -> None:
    global search_runtime
    if search_runtime is None:
        return
    search_runtime["con_data"].close()

atexit.register(close_search_runtime)


def generate_job_id() -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{ts}_{suffix}"


def _job_paths(job_id: str) -> JobPaths:
    return JobPaths(JOBS_DIR, job_id)


def _job_exists_on_disk(job_id: str) -> bool:
    paths = _job_paths(job_id)
    return os.path.isdir(paths.job_dir) and os.path.isfile(paths.job_meta_path)


def _read_pipeline_log_items(job_id: str) -> List[Dict[str, Any]]:
    paths = _job_paths(job_id)
    log_path = os.path.join(paths.job_dir, "pipeline_run.log")
    if not os.path.isfile(log_path):
        return []
    items: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            msg = line.rstrip("\n")
            if not msg:
                continue
            items.append({"timestamp": None, "message": msg})
    return items


def ordered_unique(items: List[Any]) -> List[str]:
    return list(dict.fromkeys(str(x).strip() for x in items if str(x).strip()))


def local_table_paths(items: List[str]) -> List[str]:
    return [resolve_table_path(name) for name in ordered_unique(items)]


def ordered_unique_paths(items: List[str]) -> List[str]:
    return ordered_unique(items)


def _resolve_required_table_path(path_q: str) -> str:
    raw = str(path_q or "").strip()
    resolved = resolve_table_path(raw) if raw else ""
    if not resolved or not os.path.isfile(resolved):
        raise FileNotFoundError(f"Could not resolve table path: {raw}")
    return resolved


def _resolve_grouped_query_mapping(query_to_retrieved: Dict[str, List[str]]) -> Dict[str, List[str]]:
    resolved_mapping: Dict[str, List[str]] = {}
    for query_table_path, retrieved_table_paths in (query_to_retrieved or {}).items():
        query_path = str(query_table_path or "").strip()
        if not query_path:
            continue
        resolved_mapping[_resolve_required_table_path(query_path)] = [
            _resolve_required_table_path(path)
            for path in (retrieved_table_paths or [])
            if str(path or "").strip()
        ]
    return resolved_mapping


def _table_group_output_prefix(*, integration_type: str, search_type: str, tables_source: str) -> str:
    return f"integrated_table_search_{integration_type}_{search_type}_{tables_source}"


def _serialize_grouped_integrated_tables(
    grouped_results: List[Dict[str, Any]],
    *,
    job_dir: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "query_table_path": result["query_table_path"],
            "retrieved_table_paths": list(result["retrieved_table_paths"]),
            "integration_input_table_paths": list(result["integration_input_table_paths"]),
            "saved_path": os.path.relpath(str(result["saved_path"]), job_dir),
            "integrated_table": _table_payload_from_dataframe(
                result["integrated_df"],
                input_tables=len(result["integration_input_table_paths"]),
            ),
        }
        for result in grouped_results
    ]


def _table_payload_from_dataframe(df: pd.DataFrame, *, input_tables: int) -> Dict[str, Any]:
    return {
        "columns": list(df.columns),
        "data": sanitize_for_json(df.values.tolist()),
        "stats": {
            "input_tables": int(input_tables),
            "output_rows": int(len(df)),
            "output_columns": int(df.shape[1]),
        },
    }


def _table_payload_from_path(path_q: str) -> Dict[str, Any]:
    resolved = resolve_table_path(path_q) or str(path_q)
    df = read_csv_robust(resolved)
    return {
        "columns": list(df.columns),
        "data": sanitize_for_json(df.values.tolist()),
        "path": resolved,
        "stats": {
            "input_tables": 1,
            "input_rows": int(len(df)),
            "output_rows": int(len(df)),
            "output_columns": int(df.shape[1]),
            "total_unique_tables": 1,
        },
    }


def _attach_single_table_preview(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    table_paths = [p for p in (out.get("table_paths") or []) if str(p).strip()]
    if len(table_paths) == 1:
        try:
            out["single_table_preview"] = _table_payload_from_path(table_paths[0])
        except Exception:
            pass
    return out


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_query_items_from_path(path: str, *, source_tag: str) -> List[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return []
    out: List[Dict[str, Any]] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                query = str(obj.get("query") or obj.get("rewritten_query") or "").strip()
                if not query:
                    resp_text = obj.get("response_text")
                    if isinstance(resp_text, str):
                        try:
                            parsed = json.loads(resp_text)
                            if isinstance(parsed, dict):
                                query = str(parsed.get("query", "")).strip()
                        except Exception:
                            pass
                if not query:
                    continue
                qid = str(obj.get("id") or obj.get("custom_id") or f"{source_tag}_{i}").strip()
                title = str(obj.get("title") or obj.get("id") or qid).strip() or qid
                out.append({"id": qid, "title": title, "query": query, "source": source_tag})
        return out

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("queries", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    for i, obj in enumerate(items):
        if not isinstance(obj, dict):
            continue
        query = str(obj.get("query", "")).strip()
        if not query:
            continue
        qid = str(obj.get("id") or f"{source_tag}_{i}").strip()
        title = str(obj.get("title") or qid).strip() or qid
        out.append({"id": qid, "title": title, "query": query, "source": source_tag})
    return out


def _evaluation_paths(job_id: str) -> Dict[str, str]:
    base_dir = os.path.join(OUTPUT_DIR, "evaluate", "pipeline", job_id)
    return {
        "base_dir": base_dir,
        "markdown_path": os.path.join(base_dir, "pipeline_match_log.md"),
        "summary_path": os.path.join(base_dir, "pipeline_summary.json"),
    }


def _load_evaluation_summary_payload(job_id: str) -> Dict[str, Any]:
    paths = _evaluation_paths(job_id)
    summary_path = paths["summary_path"]
    md_path = paths["markdown_path"]
    if not os.path.isfile(summary_path):
        return {"status": "missing", "job_id": job_id, "available": False}
    with open(summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    clusters = payload.get("clusters", []) if isinstance(payload, dict) else []
    cluster = clusters[0] if isinstance(clusters, list) and clusters else {}
    query = str(cluster.get("query", "")).strip()
    headers = cluster.get("query_headers", []) if isinstance(cluster.get("query_headers"), list) else []
    methods = []
    for row in cluster.get("query_method_counts", []) if isinstance(cluster.get("query_method_counts"), list) else []:
        if not isinstance(row, dict):
            continue
        methods.append(
            {
                "method": str(row.get("method", "")).strip(),
                "model_count": len(row.get("models", []) or []),
                "original_sum": int(row.get("raw_rows", 0) or 0),
                "original_dedup": int(row.get("raw_dedup_count", 0) or 0),
                "filter_sum": int(row.get("matched_rows", 0) or 0),
                "filter_dedup": int(row.get("matched_dedup_count", 0) or 0),
                "nugget_csv_path": str(row.get("nugget_csv_path", "")).strip(),
            }
        )
    return {
        "status": "success",
        "available": True,
        "job_id": job_id,
        "query": query,
        "headers": headers,
        "methods": methods,
        "markdown_path": md_path if os.path.isfile(md_path) else "",
        "summary_path": summary_path,
    }


def _evaluation_run_status_payload(job_id: str) -> Dict[str, Any]:
    run = evaluation_runs.get(job_id) or {}
    status = str(run.get("status", "idle"))
    payload: Dict[str, Any] = {
        "status": "success",
        "job_id": job_id,
        "run_status": status,
        "message": str(run.get("message", "")),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "log_path": str(run.get("log_path", "")),
    }
    if status == "completed":
        payload["summary"] = _load_evaluation_summary_payload(job_id)
    return payload


def _manual_eval_jobs_json_path(job_id: str) -> str:
    batch_dir = os.path.join(JOBS_DIR, "batch_runs")
    os.makedirs(batch_dir, exist_ok=True)
    return os.path.join(batch_dir, f"manual_eval_{job_id}.json")


def _build_manual_eval_jobs_json(job_id: str) -> str:
    paths = JobPaths(JOBS_DIR, job_id)
    if not os.path.isfile(paths.job_meta_path):
        raise FileNotFoundError(f"Unknown job_id: {job_id}")
    meta = JobMeta.load(paths.job_meta_path)
    out_path = _manual_eval_jobs_json_path(job_id)
    payload = [
        {
            "job_id": job_id,
            "query": meta.query,
            "search_response": {
                "folder_path": paths.job_dir,
                "model_top_k": int(meta.model_top_k),
            },
        }
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def _run_wrap_eval_background(job_id: str, llm_mode: str = "iter") -> None:
    run_state = evaluation_runs.setdefault(job_id, {})
    run_state["status"] = "running"
    run_state["message"] = "Running wrap_card_query_eval..."
    run_state["started_at"] = datetime.now().isoformat()
    run_state["finished_at"] = None
    paths = JobPaths(JOBS_DIR, job_id)
    log_path = os.path.join(paths.job_dir, "wrap_eval.log")
    run_state["log_path"] = log_path

    try:
        jobs_json_path = _build_manual_eval_jobs_json(job_id)
        wrap_llm_mode = str(llm_mode or "iter").strip().lower()
        cmd = [
            sys.executable,
            "-m",
            "src.evaluate.wrap_card_query_eval",
            "--jobs-json",
            jobs_json_path,
            "--job-id",
            job_id,
            "--llm-mode",
            wrap_llm_mode,
        ]
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"$ {' '.join(cmd)}\n\n")
            proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            run_state["status"] = "failed"
            tail = ""
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                if lines:
                    tail = "".join(lines[-12:]).strip()
            except Exception:
                tail = ""
            run_state["message"] = (
                f"wrap_card_query_eval failed (exit={proc.returncode})"
                + (f" | {tail}" if tail else "")
            )
        else:
            run_state["status"] = "completed"
            run_state["message"] = "Evaluation generated."
    except Exception as exc:
        run_state["status"] = "failed"
        run_state["message"] = f"Failed to run evaluation: {exc}"
    finally:
        run_state["finished_at"] = datetime.now().isoformat()


def _load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _csv_table_payload(csv_path: str) -> Dict[str, Any]:
    df = read_csv_robust(csv_path)
    return {
        "columns": list(df.columns),
        "data": sanitize_for_json(df.values.tolist()),
        "stats": {
            "output_rows": int(len(df)),
            "output_columns": int(df.shape[1]),
        },
    }


def _build_model_run_from_csv(job_id: str, csv_path: str) -> Optional[Dict[str, Any]]:
    basename = os.path.basename(csv_path)
    prefix = "integrated_model_search_"
    suffix = ".csv"
    if not (basename.startswith(prefix) and basename.endswith(suffix)):
        return None
    stem = basename[len(prefix):-len(suffix)]
    integration_type = "alite"
    retrieval_mode = stem[len(integration_type) + 1:] if stem.startswith(f"{integration_type}_") else "dense"
    if retrieval_mode not in QUERY2MODELCARD_RETRIEVAL_MODES:
        retrieval_mode = "dense"
    payload = {
        "status": "success",
        "integration_type": integration_type,
        "query2modelcard_retrieval_mode": retrieval_mode,
        "integrated_table": _csv_table_payload(csv_path),
        "saved_path": f"jobs_251117/{job_id}/{basename}",
    }
    json_path = os.path.join(JOBS_DIR, job_id, f"integration_model_search_{integration_type}_{retrieval_mode}.json")
    saved_json = _load_json_if_exists(json_path) or {}
    if saved_json:
        payload.update(saved_json)
        payload["integrated_table"] = payload.get("integrated_table") or _csv_table_payload(csv_path)
        payload["saved_path"] = payload.get("saved_path") or f"jobs_251117/{job_id}/{basename}"
    return payload


def _build_table_run_from_csv(job_id: str, csv_path: str) -> Optional[Dict[str, Any]]:
    basename = os.path.basename(csv_path)
    prefix = "integrated_table_search_"
    suffix = ".csv"
    if not (basename.startswith(prefix) and basename.endswith(suffix)):
        return None
    stem = basename[len(prefix):-len(suffix)]
    integration_type = "alite"
    if not stem.startswith(integration_type):
        return None
    remainder = stem[len(integration_type):].lstrip("_")
    tables_source = "intermediate"
    matched_source = next((src for src in ("all_from_modelcards", "intermediate") if remainder.endswith(f"_{src}")), None)
    if matched_source:
        tables_source = matched_source
        remainder = remainder[:-(len(matched_source) + 1)]
    search_type = remainder
    if search_type not in CARD2TAB2CARD_TYPES:
        return None
    payload = {
        "status": "success",
        "integration_type": integration_type,
        "search_type": search_type,
        "tables_source": tables_source,
        "grouped_integrated_tables": [
            {
                "query_table_path": "",
                "retrieved_table_paths": [],
                "integration_input_table_paths": [],
                "saved_path": f"jobs_251117/{job_id}/{basename}",
                "integrated_table": _csv_table_payload(csv_path),
            }
        ],
        "saved_path": f"jobs_251117/{job_id}/{basename}",
    }
    json_path = os.path.join(JOBS_DIR, job_id, f"integration_table_search_{integration_type}_{search_type}_{tables_source}.json")
    saved_json = _load_json_if_exists(json_path) or {}
    if saved_json:
        payload.update(saved_json)
        if not payload.get("grouped_integrated_tables"):
            payload["grouped_integrated_tables"] = [
                {
                    "query_table_path": "",
                    "retrieved_table_paths": [],
                    "integration_input_table_paths": [],
                    "saved_path": payload.get("saved_path") or f"jobs_251117/{job_id}/{basename}",
                    "integrated_table": _csv_table_payload(csv_path),
                }
            ]
        payload["saved_path"] = payload.get("saved_path") or f"jobs_251117/{job_id}/{basename}"
    return payload


def _build_table_run_from_json(job_id: str, json_path: str) -> Optional[Dict[str, Any]]:
    basename = os.path.basename(json_path)
    prefix = "integration_table_search_"
    suffix = ".json"
    if not (basename.startswith(prefix) and basename.endswith(suffix)):
        return None
    payload = _load_json_if_exists(json_path)
    if not payload:
        return None
    if payload.get("status") != "success":
        return None
    if str(payload.get("integration_type") or "").strip().lower() != "alite":
        return None
    search_type = str(payload.get("search_type") or "").strip()
    if search_type not in CARD2TAB2CARD_TYPES:
        return None
    payload["tables_source"] = str(payload.get("tables_source") or "intermediate").strip() or "intermediate"
    return payload


def _enrich_model_run_from_job(job_id: str, run: Dict[str, Any]) -> Dict[str, Any]:
    paths = JobPaths(JOBS_DIR, job_id)
    try:
        job_meta = JobMeta.load(paths.job_meta_path)
        q2m_file = Query2ModelCardFile.load(paths.query2modelcard_path)
        retrieval_mode = str(run.get("query2modelcard_retrieval_mode") or "dense").strip() or "dense"
        preview = q2m_file.build_preview(
            query=job_meta.query,
            table_resources=job_meta.table_resources,
            mode=retrieval_mode,
            max_models=int(job_meta.model_top_k),
        )
        out = {**preview, **run}
        stats = dict(preview.get("stats") or {})
        integrated_stats = ((run.get("integrated_table") or {}).get("stats") or {})
        stats.update(integrated_stats)
        stats["input_tables"] = int(len(preview.get("table_paths") or []))
        out["stats"] = stats
        return out
    except Exception:
        return run


def _enrich_table_run_from_job(job_id: str, run: Dict[str, Any]) -> Dict[str, Any]:
    paths = JobPaths(JOBS_DIR, job_id)
    try:
        job_meta = JobMeta.load(paths.job_meta_path)
        search_type = str(run.get("search_type") or "").strip()
        tables_source = str(run.get("tables_source") or "intermediate").strip() or "intermediate"
        preview = Query2Tab2CardFullMap(paths.card2tab2card_path(search_type)).build_preview(
            search_type=search_type,
            max_models=int(job_meta.model_top_k),
            tables_source=tables_source,
        )
        out = {**preview, **run}
        stats = dict(preview.get("stats") or {})
        integration_input_paths = run.get("integration_input_table_paths") or []
        if integration_input_paths:
            stats["input_tables"] = int(len(integration_input_paths))
        else:
            stats["input_tables"] = int(len(preview.get("query_tables") or [])) + int(len(preview.get("table_paths") or []))
        out["stats"] = stats
        return out
    except Exception:
        return run


def ensure_job_state(job_id: str) -> Dict[str, Any]:
    if job_id not in jobs:
        jobs[job_id] = {"status": "completed" if _job_exists_on_disk(job_id) else "pending", "logs": [], "results": None}
    jobs[job_id].setdefault("logs", [])
    return jobs[job_id]


def append_log(job_id: str, message: str) -> None:
    paths = JobPaths(JOBS_DIR, job_id)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{ts}] [{job_id}] {message}"
    ensure_job_state(job_id)["logs"].append({"timestamp": datetime.now().isoformat(), "message": message})
    print(line, flush=True)
    os.makedirs(paths.job_dir, exist_ok=True)
    with open(os.path.join(paths.job_dir, "pipeline_run.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def api_error(message: str, status: int = 500):
    return jsonify({"status": "error", "message": message}), status


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    return obj


def dataframe_to_html_table(df: pd.DataFrame) -> str:
    def cell(v: Any) -> str:
        if pd.isna(v) or v is None:
            return ""
        return html.escape(str(v))

    header = "".join(f"<th>{cell(c)}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        rows.append("<tr>" + "".join(f"<td>{cell(v)}</td>" for v in row) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def make_table_page_response(path_q: str) -> Response:
    df, source, basename = preview_from_local(path_q, max_rows=MAX_TABLE_PAGE_ROWS, max_cols=MAX_TABLE_PAGE_COLS)
    total_cols = df.shape[1]
    note = ""
    if total_cols > MAX_TABLE_PAGE_COLS:
        note += f" Showing first {MAX_TABLE_PAGE_COLS} of {total_cols} columns."
        df = df.iloc[:, :MAX_TABLE_PAGE_COLS]
    if len(df) >= MAX_TABLE_PAGE_ROWS:
        note = f"Showing first {MAX_TABLE_PAGE_ROWS} rows." + note
    body = render_template_string(TABLE_PAGE_HTML, title=basename, path_display=source, note=note.strip(), table_html=dataframe_to_html_table(df))
    return Response(body, mimetype="text/html; charset=utf-8")


def markdown_to_html(md_text: str) -> str:
    try:
        import markdown  # type: ignore
        return markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    except Exception:
        return _simple_markdown_to_html(md_text)


def _inline_md_to_html(text: str) -> str:
    out = html.escape(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>', out)
    return out


def _simple_markdown_to_html(md_text: str) -> str:
    lines = md_text.splitlines()
    chunks: List[str] = []
    list_items: List[str] = []
    in_code = False
    code_lines: List[str] = []
    raw_html_block: List[str] = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            chunks.append("<ul>" + "".join(f"<li>{item}</li>" for item in list_items) + "</ul>")
            list_items = []

    def flush_code() -> None:
        nonlocal code_lines
        if code_lines:
            chunks.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
            code_lines = []

    def flush_raw_html() -> None:
        nonlocal raw_html_block
        if raw_html_block:
            chunks.append("\n".join(raw_html_block))
            raw_html_block = []

    for line in lines:
        stripped = line.strip()

        if in_code:
            if stripped.startswith("```"):
                flush_code()
                in_code = False
            else:
                code_lines.append(line)
            continue

        if stripped.startswith("```"):
            flush_list()
            flush_raw_html()
            in_code = True
            code_lines = []
            continue

        if stripped.startswith("<") and stripped.endswith(">"):
            flush_list()
            raw_html_block.append(line)
            continue
        flush_raw_html()

        if not stripped:
            flush_list()
            chunks.append("")
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush_list()
            level = len(m.group(1))
            chunks.append(f"<h{level}>{_inline_md_to_html(m.group(2))}</h{level}>")
            continue

        m = re.match(r"^-\s+(.*)$", stripped)
        if m:
            list_items.append(_inline_md_to_html(m.group(1)))
            continue

        chunks.append(f"<p>{_inline_md_to_html(stripped)}</p>")

    flush_list()
    flush_raw_html()
    if in_code:
        flush_code()
    return "\n".join(part for part in chunks if part != "")


def run_search_job(job_id: str, query: str, top_k: int, table_search_k: int, model_top_k: int, use_by_type: bool, runtime) -> None:
    paths = JobPaths(JOBS_DIR, job_id)
    jobs[job_id]["status"] = "running"
    os.makedirs(paths.job_dir, exist_ok=True)
    start = time.time()
    rt = runtime
    q2m = Query2ModelCardSearch(query=query, top_k=min(200, 5 * top_k), job_id=job_id)
    q2m.search_dense(top_k=q2m.top_k, dense=rt["dense_full"])
    q2m.search_sparse(top_k=q2m.top_k, sparse=rt["sparse_full"])
    q2m.search_hybrid(top_k=q2m.top_k, sparse=rt["sparse_full"], dense=rt["dense_full"], candidate_factor=10)
    q2m.save_to_json(paths.query2modelcard_path)
    append_log(job_id, "query2modelcard finished")
    q2t2c = Query2Tab2CardSearch()
    for search_type in CARD2TAB2CARD_TYPES:
        q2t2c.pipeline_w_query_reranker(query, rt["con_data"], rt["dense_full"], rt["dense_wtable"], search_type=search_type, table_top_k=table_search_k, table_resources=rt["table_resources"], use_tab2tab_aug=bool(USE_TAB2TAB_AUG), apply_query_rerank=True, model_top_k=model_top_k, q2m_top_k=1)
        q2t2c.save_full_json(paths.card2tab2card_path(search_type))
        append_log(job_id, f"card2tab2card {search_type} finished")
    JobMeta.save_for_job(jobs_dir=JOBS_DIR, job_id=job_id, query=query, top_k=top_k, model_top_k=model_top_k, table_search_k=table_search_k, table_resources=rt["table_resources"], use_by_type=use_by_type, running_time_seconds=time.time() - start)
    jobs[job_id]["status"] = "completed"

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/preset-queries", methods=["GET"])
def preset_queries():
    source = str(request.args.get("source", "default")).strip().lower() or "default"
    extra_path = str(request.args.get("extra_path", "")).strip() or os.environ.get("MODELSEARCH_EXTRA_PRESET_QUERIES_PATH", DEFAULT_EXTRA_PRESET_PATH)
    default_items = _load_query_items_from_path(PRESET_QUERIES_PATH, source_tag="default")
    extra_items = _load_query_items_from_path(extra_path, source_tag="extra")
    if source == "extra":
        queries = extra_items
    elif source == "all":
        queries = default_items + extra_items
    else:
        queries = default_items
    available_sources = ["default"]
    if extra_items:
        available_sources.append("extra")
        available_sources.append("all")
    return jsonify(
        {
            "status": "success",
            "queries": queries,
            "source": source,
            "available_sources": available_sources,
            "extra_path": extra_path if extra_items else "",
        }
    )


@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json() or {}
    query = str(data["query"]).strip()
    top_k = int(data.get("top_k", 100))
    table_search_k = int(data.get("table_search_k", 3))
    model_top_k = int(data.get("model_top_k", 5))
    use_by_type = bool(data.get("use_by_type", False))
    job_id = generate_job_id()
    jobs[job_id] = {"status": "pending", "logs": [], "results": None}
    threading.Thread(target=run_search_job, args=(job_id, query, top_k, table_search_k, model_top_k, use_by_type, ensure_search_runtime()), daemon=True).start()
    return jsonify({"status": "started", "job_id": job_id})

@app.route("/api/status/<job_id>", methods=["GET"])
def status(job_id: str):
    if job_id in jobs:
        return jsonify({"job_id": job_id, "status": jobs[job_id]["status"], "logs": jobs[job_id]["logs"]})
    if _job_exists_on_disk(job_id):
        return jsonify({"job_id": job_id, "status": "completed", "logs": _read_pipeline_log_items(job_id)})
    return api_error(f"Unknown job_id: {job_id}", 404)


@app.route("/api/results/<job_id>", methods=["GET"])
def results(job_id: str):
    if job_id in jobs and jobs[job_id]["status"] != "completed":
        return jsonify({"status": jobs[job_id]["status"]}), 202
    paths = _job_paths(job_id)
    if not os.path.isfile(paths.job_meta_path):
        return api_error(f"Unknown job_id: {job_id}", 404)
    meta = JobMeta.load(paths.job_meta_path)
    return jsonify({"status": "success", "job_id": job_id, "results": {**meta.to_dict(), "folder_path": paths.job_dir, "run_log_path": os.path.join(paths.job_dir, "pipeline_run.log")}})


@app.route("/api/logs/<job_id>", methods=["GET"])
def logs(job_id: str):
    def generate():
        if job_id not in jobs:
            for item in _read_pipeline_log_items(job_id):
                yield f"data: {json.dumps(item)}\n\n"
            yield f"data: {json.dumps({'status': 'completed'})}\n\n"
            return
        last = 0
        while jobs[job_id]["status"] in ("pending", "running"):
            items = jobs[job_id]["logs"]
            for item in items[last:]:
                yield f"data: {json.dumps(item)}\n\n"
            last = len(items)
            time.sleep(0.5)
        items = jobs[job_id]["logs"]
        for item in items[last:]:
            yield f"data: {json.dumps(item)}\n\n"
        yield f"data: {json.dumps({'status': 'completed'})}\n\n"
    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/api/saved-searches", methods=["GET"])
def saved_searches():
    rows = []
    for name in sorted(os.listdir(JOBS_DIR), reverse=True):
        paths = JobPaths(JOBS_DIR, name)
        if not os.path.isdir(paths.job_dir):
            continue
        if not os.path.isfile(paths.job_meta_path):
            continue
        meta = JobMeta.load(paths.job_meta_path)
        q2m_file = Query2ModelCardFile.load(paths.query2modelcard_path)
        rows.append(
            {
                "folder_name": name,
                "path": paths.job_dir,
                "query": meta.query,
                "model_id": q2m_file.seed_model_id,
                "timestamp_str": datetime.fromisoformat(meta.timestamp).strftime("%Y-%m-%d %H:%M"),
                "top_k": meta.top_k,
                "model_top_k": meta.model_top_k,
                "use_by_type": meta.use_by_type,
                "table_search_k": meta.table_search_k,
            }
        )
    return jsonify({"status": "success", "searches": rows[:50]})


@app.route("/api/table-preview", methods=["GET"])
def table_preview():
    path_q = request.args["path"].strip()
    df, source, basename = preview_from_local(path_q, max_rows=MAX_TABLE_PREVIEW_JSON_ROWS, max_cols=MAX_TABLE_PREVIEW_JSON_COLS)
    preview = df.head(min(120, len(df)))
    return jsonify({"status": "success", "html": dataframe_to_html_table(preview), "rows": int(len(df)), "columns": int(df.shape[1]), "source": source, "table": basename})


@app.route("/api/table-page", methods=["GET"])
def table_page():
    return make_table_page_response(request.args["path"])


@app.route("/api/integration-review-page/<job_id>", methods=["GET"])
def integration_review_page(job_id: str):
    paths = _job_paths(job_id)
    if not os.path.isfile(paths.job_meta_path):
        return api_error(f"Unknown job_id: {job_id}", 404)
    try:
        from src.utils.check_retrieval_integration_consistency import write_job_markdown

        md_path = write_job_markdown(
            job_dir=paths.job_dir,
            search_types=None,
            integration_type="alite",
            preview_max_rows=None,
            preview_max_cols=None,
            output_path=os.path.join(paths.job_dir, "integration_review.md"),
        )
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()
        body = render_template_string(
            MARKDOWN_PAGE_HTML,
            title=f"Integration Review: {job_id}",
            path_display=md_path,
            content_html=markdown_to_html(md_text),
        )
        return Response(body, mimetype="text/html; charset=utf-8")
    except Exception as exc:
        return api_error(f"Failed to build integration review markdown: {exc}", 500)


@app.route("/api/evaluation-summary/<job_id>", methods=["GET"])
def evaluation_summary(job_id: str):
    try:
        return jsonify(_load_evaluation_summary_payload(job_id))
    except Exception as exc:
        return api_error(f"Failed to load evaluation summary: {exc}", 500)


@app.route("/api/evaluation-run/<job_id>", methods=["GET"])
def evaluation_run_status(job_id: str):
    try:
        return jsonify(_evaluation_run_status_payload(job_id))
    except Exception as exc:
        return api_error(f"Failed to load evaluation run status: {exc}", 500)


@app.route("/api/evaluation-run", methods=["POST"])
def evaluation_run_start():
    data = request.get_json() or {}
    job_id = str(data.get("job_id", "")).strip()
    llm_mode = str(data.get("llm_mode", "iter")).strip().lower() or "iter"
    if not job_id:
        return api_error("Missing job_id", 400)
    if llm_mode not in ("iter", "batch"):
        return api_error("llm_mode must be one of: iter, batch", 400)
    paths = JobPaths(JOBS_DIR, job_id)
    if not os.path.isfile(paths.job_meta_path):
        return api_error(f"Unknown job_id: {job_id}", 404)

    run = evaluation_runs.get(job_id, {})
    if str(run.get("status")) == "running":
        return jsonify({"status": "success", "job_id": job_id, "run_status": "running", "message": "Evaluation already running."})

    t = threading.Thread(target=_run_wrap_eval_background, args=(job_id, llm_mode), daemon=True)
    t.start()
    return jsonify({"status": "success", "job_id": job_id, "run_status": "running", "message": "Evaluation started."})


@app.route("/api/evaluation-page/<job_id>", methods=["GET"])
def evaluation_page(job_id: str):
    payload = _load_evaluation_summary_payload(job_id)
    if not payload.get("available"):
        return api_error(f"No evaluation markdown found for job_id: {job_id}", 404)
    md_path = str(payload.get("markdown_path", "")).strip()
    if not md_path or not os.path.isfile(md_path):
        return api_error(f"No evaluation markdown found for job_id: {job_id}", 404)
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    body = render_template_string(
        MARKDOWN_PAGE_HTML,
        title=f"Evaluation: {job_id}",
        path_display=md_path,
        content_html=markdown_to_html(md_text),
    )
    return Response(body, mimetype="text/html; charset=utf-8")


@app.route("/api/model-search-preview", methods=["POST"])
def model_search_preview():
    data = request.get_json() or {}
    job_id = str(data["job_id"]).strip()
    paths = JobPaths(JOBS_DIR, job_id)
    job_meta = JobMeta.load(paths.job_meta_path)
    q2m_file = Query2ModelCardFile.load(paths.query2modelcard_path)
    mode = data.get("query2modelcard_retrieval_mode") or "dense"
    if mode in ("integration", "saved", "default"):
        mode = "dense"
    payload = q2m_file.build_preview(query=job_meta.query, table_resources=job_meta.table_resources, mode=mode, max_models=int(data.get("max_models", job_meta.model_top_k)))
    payload = _attach_single_table_preview(payload)
    return jsonify({"status": "success", **payload})


@app.route("/api/table-search-preview", methods=["POST"])
def table_search_preview():
    data = request.get_json() or {}
    job_id = str(data["job_id"]).strip()
    paths = JobPaths(JOBS_DIR, job_id)
    # `search_type` here refers to the table-retrieval mode from Query2Tab2Card
    # (for example: keyword / single_column / unionable). It is unrelated to the
    # internal keyword-based transpose recognition now embedded in TableIntegrater.
    search_type = str(data.get("search_type", "single_column")).strip()
    tables_source = str(data.get("tables_source", "intermediate")).strip()
    payload = Query2Tab2CardFullMap(paths.card2tab2card_path(search_type)).build_preview(search_type=search_type, tables_source=tables_source)
    payload = _attach_single_table_preview(payload)
    return jsonify({"status": "success", **payload})


@app.route("/api/integrate-model-search", methods=["POST"])
def integrate_model_search():
    data = request.get_json() or {}
    required = ["job_id", "integration_type", "query2modelcard_retrieval_mode", "k", "max_models"]
    missing = [key for key in required if key not in data]
    if missing:
        return api_error(f"Missing required fields for /api/integrate-model-search: {', '.join(missing)}", 400)

    job_id = str(data["job_id"]).strip()
    paths = JobPaths(JOBS_DIR, job_id)
    integration_type = str(data["integration_type"]).strip().lower()
    if integration_type != "alite":
        return api_error("Only integration_type=alite is supported.", 400)
    retrieval_mode = str(data["query2modelcard_retrieval_mode"]).strip()
    k = int(data["k"])
    max_models = int(data["max_models"])
    job_meta = JobMeta.load(paths.job_meta_path)
    q2m_file = Query2ModelCardFile.load(paths.query2modelcard_path)
    append_log(
        job_id,
        f"integrate-model-search called integration_type={integration_type} retrieval_mode={retrieval_mode} k={k} max_models={max_models}",
    )
    payload = q2m_file.build_preview(query=job_meta.query, table_resources=job_meta.table_resources, mode=retrieval_mode, max_models=max_models)
    payload = _attach_single_table_preview(payload)
    table_paths = payload["table_paths"][:k]
    df = TableIntegrater().run(local_table_paths(table_paths), mode=integration_type)
    csv_name = f"integrated_model_search_{integration_type}_{retrieval_mode}.csv"
    csv_path = os.path.join(paths.job_dir, csv_name)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    response_payload = {"status": "success", "integration_type": integration_type, "query2modelcard_retrieval_mode": retrieval_mode, "k": k, "max_models": max_models, "integrated_table": {"columns": list(df.columns), "data": sanitize_for_json(df.values.tolist())}, "saved_path": f"jobs_251117/{job_id}/{csv_name}", **payload}
    _save_json(os.path.join(paths.job_dir, f"integration_model_search_{integration_type}_{retrieval_mode}.json"), response_payload)
    return jsonify(response_payload)


@app.route("/api/integrate", methods=["POST"])
def integrate():
    data = request.get_json() or {}
    required = ["job_id", "search_type", "integration_type", "tables_source", "k", "max_models"]
    missing = [key for key in required if key not in data]
    if missing:
        return api_error(f"Missing required fields for /api/integrate: {', '.join(missing)}", 400)

    job_id = str(data["job_id"]).strip()
    paths = JobPaths(JOBS_DIR, job_id)
    # This `search_type` is still the retrieval bucket name used to load the saved
    # Query2Tab2Card results. Orientation detection is no longer driven by backend
    # parameters; it is handled internally inside TableIntegrater.
    search_type = str(data["search_type"]).strip()
    integration_type = str(data["integration_type"]).strip().lower()
    if integration_type != "alite":
        return api_error("Only integration_type=alite is supported.", 400)
    tables_source = str(data["tables_source"]).strip()
    k = int(data["k"])
    max_models = int(data["max_models"])
    append_log(
        job_id,
        f"integrate called search_type={search_type} integration_type={integration_type} tables_source={tables_source} k={k} max_models={max_models}",
    )
    payload = Query2Tab2CardFullMap(paths.card2tab2card_path(search_type)).build_preview(search_type=search_type, max_models=max_models, tables_source=tables_source)
    payload = _attach_single_table_preview(payload)
    grouped_integrated_tables: List[Dict[str, Any]] = []
    integration_input_table_paths: List[str] = []
    group_output_dir = os.path.join(paths.job_dir, "table_integration_groups")
    group_output_prefix = _table_group_output_prefix(
        integration_type=integration_type,
        search_type=search_type,
        tables_source=tables_source,
    )

    if tables_source == "intermediate":
        query_to_retrieved = {
            str(query_path).strip(): [str(path).strip() for path in (retrieved_paths or []) if str(path).strip()]
            for query_path, retrieved_paths in (payload.get("query_table_to_retrieved_table_paths") or {}).items()
            if str(query_path).strip()
        }
        original_groups = list(query_to_retrieved.items())
        group_integrater = GroupTableIntegrater()
        grouped_results = group_integrater.run(
            _resolve_grouped_query_mapping(query_to_retrieved),
            mode=integration_type,
            k=k,
        )
        for result, (query_path, retrieved_paths) in zip(grouped_results, original_groups):
            limited_retrieved_paths = list(retrieved_paths[:k])
            result["query_table_path"] = query_path
            result["retrieved_table_paths"] = limited_retrieved_paths
            result["integration_input_table_paths"] = [query_path] + limited_retrieved_paths
        saved_grouped_results = group_integrater.save(
            grouped_results,
            output_dir=group_output_dir,
            output_name_prefix=group_output_prefix,
        )
        integration_input_table_paths = ordered_unique_paths(
            path
            for result in saved_grouped_results
            for path in result["integration_input_table_paths"]
        )
        grouped_integrated_tables = _serialize_grouped_integrated_tables(
            saved_grouped_results,
            job_dir=paths.job_dir,
        )
    else:
        retrieved_table_paths = [str(p).strip() for p in (payload.get("table_paths") or []) if str(p).strip()]
        integration_input_table_paths = ordered_unique_paths(retrieved_table_paths[:k])
        df = TableIntegrater().run(local_table_paths(integration_input_table_paths), mode=integration_type)
        if df is None:
            raise ValueError("Table integration returned no output.")
        flat_grouped_results = [
            {
                "query_table_path": "",
                "retrieved_table_paths": retrieved_table_paths[:k],
                "integration_input_table_paths": integration_input_table_paths,
                "integrated_df": df,
            }
        ]
        saved_grouped_results = GroupTableIntegrater().save(
            flat_grouped_results,
            output_dir=group_output_dir,
            output_name_prefix=group_output_prefix,
        )
        grouped_integrated_tables = _serialize_grouped_integrated_tables(
            saved_grouped_results,
            job_dir=paths.job_dir,
        )

    response_payload = {
        "status": "success",
        "integration_type": integration_type,
        "search_type": search_type,
        "k": k,
        "max_models": max_models,
        "tables_source": payload["tables_source"],
        "integration_input_table_paths": integration_input_table_paths,
        "grouped_integrated_tables": grouped_integrated_tables,
        **payload,
    }
    _save_json(os.path.join(paths.job_dir, f"integration_table_search_{integration_type}_{search_type}_{tables_source}.json"), response_payload)
    return jsonify(response_payload)


@app.route("/api/integration-runs/<job_id>", methods=["GET"])
def integration_runs(job_id: str):
    paths = JobPaths(JOBS_DIR, job_id)
    if not os.path.isdir(paths.job_dir):
        return api_error(f"Unknown job_id: {job_id}", 404)

    model_runs_by_key: Dict[str, Dict[str, Any]] = {}
    table_runs_by_key: Dict[str, Dict[str, Any]] = {}

    for basename in sorted(os.listdir(paths.job_dir)):
        full_path = os.path.join(paths.job_dir, basename)
        if not os.path.isfile(full_path):
            continue
        if basename.endswith(".csv") and basename.startswith("integrated_model_search_"):
            payload = _build_model_run_from_csv(job_id, full_path)
            if payload:
                key = f"{payload['integration_type']}::{payload['query2modelcard_retrieval_mode']}"
                model_runs_by_key[key] = payload
        elif basename.endswith(".csv") and basename.startswith("integrated_table_search_"):
            payload = _build_table_run_from_csv(job_id, full_path)
            if payload:
                key = f"{payload['integration_type']}::{payload['search_type']}::{payload['tables_source']}"
                table_runs_by_key[key] = payload
        elif basename.endswith(".json") and basename.startswith("integration_table_search_"):
            payload = _build_table_run_from_json(job_id, full_path)
            if payload:
                key = f"{payload['integration_type']}::{payload['search_type']}::{payload['tables_source']}"
                table_runs_by_key[key] = payload

    model_runs = [_enrich_model_run_from_job(job_id, run) for run in model_runs_by_key.values()]
    table_runs = [_enrich_table_run_from_job(job_id, run) for run in table_runs_by_key.values()]
    return jsonify({"status": "success", "job_id": job_id, "model_search_runs": model_runs, "table_search_runs": table_runs})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simplified ModelSearch backend")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    init_search_runtime()
    port = args.port if args.port is not None else int(os.environ.get("PORT", "5002"))
    app.run(host="0.0.0.0", port=port, debug=False)
