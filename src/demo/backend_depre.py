"""
Backend API for ModelSearch Demo.

Runs search in-process.
All job outputs (search results, integration, evaluation, QA) go under JOBS_DIR from config (e.g. data_<tag>/jobs_<tag>/<job_id>).
Minimal imports for fast startup.
"""

import os, sys, json, random, string, threading, time, math, re, html, atexit
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify, Response, stream_with_context, render_template_string, send_from_directory
from flask_cors import CORS
from datetime import datetime
#from src.config import REPO_ROOT, JOBS_DIR, CARD2TAB2CARD_TIMEOUT, USE_BY_TYPE, QUERY2MODELCARD_RETRIEVAL_MODES, CARD2TAB2CARD_TYPES, CARD2TAB2CARD_OUTPUT_JSON, VALID_MODEL_IDS_TXT, CLASSIFICATION_JSON, TABLE_RESOURCE_ALLOWLIST, RELATIONSHIP_PARQUET, PRESET_QUERIES_PATH
from src.config import *
from src.utils import preview_from_local, _paths_for_resource_set, _get_models_to_tables_batch_sql, resolve_table_path
#from src.utils import filter_results_by_classify_results
from src.search.query2tab2card import Query2Tab2CardSearch
from src.search.query2modelcard import Query2ModelCardSearch
from src.search.ir_searcher import DenseSearcher, SparseSearcher
import duckdb

# --- Search runtime (Dense/Sparse + paths). Initialized once in init_search_runtime() from ``if __name__``.
_search_runtime: Optional[Dict[str, Any]] = None


def init_search_runtime() -> None:
    t1 = time.time()
    print(f"Initializing search runtime at {t1}")
    """Load indexes once: full-corpus dense + table dense + full sparse; read-only DuckDB (hugging table lake)."""
    global _search_runtime
    if _search_runtime is not None:
        return
    table_resources = ["hugging"]
    # Returns (emb_npz, sparse_lucene_dir, duckdb_path); full-corpus sparse index for query2modelcard sparse/hybrid.
    _, sparse_index_full, _ = _paths_for_resource_set(["hugging", "github", "arxiv"])
    _, _, table_db_path = _paths_for_resource_set(["hugging"])
    db_key = os.path.abspath(str(table_db_path))
    con_data = duckdb.connect(db_key, read_only=True)
    _search_runtime = {
        "table_resources": table_resources,
        "con_data": con_data,
        "dense_full": DenseSearcher(emb_npz_path=EMB_NPZ),
        "dense_wtable": DenseSearcher(emb_npz_path=EMB_NPZ_HUGGING),
        "sparse_full": SparseSearcher(index_path=sparse_index_full),
    }
    print(f"Search runtime initialized at {time.time() - t1}")


def _atexit_close_search_duckdb() -> None:
    global _search_runtime
    if not _search_runtime:
        return
    conn = _search_runtime.get("con_data")
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass

atexit.register(_atexit_close_search_duckdb)

def _ensure_search_runtime() -> Dict[str, Any]:
    if _search_runtime is None:
        init_search_runtime()
    assert _search_runtime is not None
    return _search_runtime


def _append_pipeline_log(job_dir: str, message: str) -> None:
    """Append one line to job_dir/pipeline_run.log (e.g. integration timing after /api/search)."""
    path = os.path.join(job_dir, "pipeline_run.log")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    jid = os.path.basename(os.path.abspath(job_dir))
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] [{jid}] {message}\n")


def _sanitize_for_json(obj: Any) -> Any:
    """Replace float('nan') with None so JSON serialization produces null."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    return obj

# Paths from config.py (relative to repo root)

def _generate_job_id() -> str:
    """Generate human-readable job ID: YYYY-MM-DD_HH-MM-SS_xxxx (time + 4-char suffix for uniqueness)."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{ts}_{suffix}"


def _model_search_key(integration_type: str) -> str:
    """Slug for Model Search artifacts: integration method only (neighbors always from query2modelcard_results / dense)."""
    p = integration_type or "union"
    return re.sub(r"[^a-z0-9_]", "_", str(p).lower().strip())


def _table_search_key(integration_type: str, search_type: str, tables_source: str = "intermediate") -> str:
    """Slug for Table Search: e.g. alite_single_column_intermediate."""
    parts = [integration_type or "union", search_type or "single_column", (tables_source or "intermediate").replace("-", "_")]
    return "_".join(re.sub(r"[^a-z0-9_]", "_", (p or "").lower().strip()) for p in parts)

def _sanitize_for_js_template(obj: Any) -> Any:
    """Replace chars that break JS template literals (\\ ` $ { }) so LLM output cannot cause SyntaxError."""
    if isinstance(obj, str):
        for c in ("\\", "`", "$", "{", "}"):
            obj = obj.replace(c, " ")
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_js_template(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_js_template(v) for v in obj]
    return obj

def _api_error(message: str, status: int = 500):
    """Uniform API error response."""
    return jsonify({"status": "error", "message": message}), status

def _dataframe_to_html_table(df: pd.DataFrame) -> str:
    def _html_table_cell(v: Any) -> str:
        if pd.isna(v):
            return ""
        if v is None:
            return ""
        return html.escape(str(v))
    col_html = "".join(f"<th>{_html_table_cell(c)}</th>" for c in df.columns)
    body_rows: List[str] = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{_html_table_cell(v)}</td>" for v in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        "<table style=\"border-collapse:collapse;font-size:12px;min-width:100%;\">"
        f"<thead><tr style=\"background:#f8f9fa;\">{col_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )

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


app = Flask(__name__)
CORS(app)

jobs: Dict[str, "JobLogger"] = {}

class JobLogger:
    """Thread-safe logger for job progress. When log_file is set, also writes to file."""
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.logs: List[Dict] = []
        self.lock = threading.Lock()
        self.status = "pending"
        self.results: Optional[Dict] = None
        self._log_file_path: Optional[str] = None

    def set_log_file(self, path: str):
        """Enable writing logs to file (e.g. job_dir/pipeline_run.log)."""
        with self.lock:
            self._log_file_path = path

    def log(self, message: str):
        with self.lock:
            now = datetime.now()
            ts = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.logs.append({"timestamp": now.isoformat(), "message": message})
            line = f"[{ts}] [{self.job_id}] {message}"
            print(line, flush=True)
            if self._log_file_path:
                with open(self._log_file_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")

    def get_logs(self) -> List[Dict]:
        with self.lock:
            return self.logs.copy()

    def set_status(self, status: str):
        with self.lock:
            self.status = status

    def set_results(self, results: Dict):
        with self.lock:
            self.results = results


def _read_json(path: str) -> Optional[Dict]:
    """Read JSON file if it exists; else return None."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_json_job(job_id: str, filename:str) -> Optional[Dict]:
    """Read JSON file from job directory."""
    return _read_json(os.path.join(JOBS_DIR, job_id, filename))


JOB_META_FILENAME = "job_meta.json"

def _resolve_eval_qa_query(data: Dict[str, Any], sr: Optional[Dict]) -> Optional[str]:
    """Non-empty query: JSON body `query` if set, else assembled job payload field `query`."""
    raw = data.get("query")
    if raw is not None:
        s = str(raw).strip()
        if s:
            return s
    if not isinstance(sr, dict):
        return None
    raw = sr.get("query")
    if raw is None:
        return None
    s = str(raw).strip()
    return s if s else None

def _write_json_job(job_id: str, filename: str, data: Dict):
    """Write JSON data to job directory."""
    with open(os.path.join(JOBS_DIR, job_id, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _save_job_meta_json(job_id: str, data: Dict) -> None:
    """Persist lightweight job metadata only. Raw retrieval results stay in their own JSON files."""
    _write_json_job(job_id, JOB_META_FILENAME, data)


def _read_job_meta_json(job_id: str) -> Optional[Dict]:
    return _read_json_job(job_id, JOB_META_FILENAME)


def _extract_model_id(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("model_id") or item.get("id") or "").strip()
    return str(item).strip()

def _read_query2modelcard_json(job_id: str) -> Optional[Dict]:
    return _read_json_job(job_id, "query2modelcard.json")

def _query2modelcard_neighbors_by_mode(q2m_json: Dict[str, Any], model_top_k: int) -> tuple[str, Dict[str, List[str]]]:
    results = q2m_json["results"]
    dense = results["dense"]
    seed_model_id = _extract_model_id(dense[0]) if dense else ""
    neighbors_by_mode: Dict[str, List[str]] = {}
    for mode in QUERY2MODELCARD_RETRIEVAL_MODES:
        raw_items = results[mode]
        neighbors: List[str] = []
        for item in raw_items:
            mid = _extract_model_id(item)
            if not mid or mid == seed_model_id:
                continue
            neighbors.append(mid)
            if len(neighbors) >= int(model_top_k):
                break
        neighbors_by_mode[mode] = neighbors
    return seed_model_id, neighbors_by_mode


def _assembled_card2tab2card_results(job_id: str) -> Dict[str, Any]:
    from src.integration.pipeline_preview import Query2Tab2CardFullMap

    out: Dict[str, Any] = {}
    for search_type in CARD2TAB2CARD_TYPES:
        map_path = os.path.join(JOBS_DIR, job_id, f"card2tab2card_{search_type}.json")
        out[search_type] = Query2Tab2CardFullMap(map_path).build_preview(search_type=search_type)
    return out


def _read_search_results_json(job_id: str) -> Optional[Dict]:
    """Assemble the legacy search-results payload from job_meta + raw retrieval JSON files."""
    meta = _read_job_meta_json(job_id)
    q2m_json = _read_query2modelcard_json(job_id)
    query = q2m_json["query"]
    model_top_k = meta["model_top_k"]
    seed_model_id, neighbors_by_mode = _query2modelcard_neighbors_by_mode(q2m_json, model_top_k)
    card2tab2card_results = _assembled_card2tab2card_results(job_id)

    right_max_models = 0
    for payload in card2tab2card_results.values():
        mids = payload.get("model_ids", []) if isinstance(payload, dict) else []
        if isinstance(mids, list):
            right_max_models = max(right_max_models, len(mids))
    effective_model_top_k = min(model_top_k, right_max_models) if right_max_models > 0 else model_top_k
    for mode in QUERY2MODELCARD_RETRIEVAL_MODES:
        neighbors_by_mode[mode] = neighbors_by_mode.get(mode, [])[:effective_model_top_k]

    job_dir = os.path.join(JOBS_DIR, job_id)
    results = {
        "job_id": job_id,
        "query": query,
        "model_id": seed_model_id,
        "table_search_seed_model_id": seed_model_id,
        "top_k": meta.get("top_k"),
        "model_top_k": model_top_k,
        "effective_model_top_k": effective_model_top_k,
        "right_max_models": right_max_models,
        "table_search_k": meta.get("table_search_k"),
        "table_resources": meta.get("table_resources", []),
        "use_by_type": bool(meta.get("use_by_type", False)),
        "query2modelcard_results": list(neighbors_by_mode.get("dense", [])),
        "query2modelcard_all_modes": neighbors_by_mode,
        "card2tab2card_results": card2tab2card_results,
        "timestamp": meta.get("timestamp"),
        "folder_path": job_dir,
        "run_log_path": os.path.join(job_dir, "pipeline_run.log"),
        "running_time_seconds": meta.get("running_time_seconds"),
        "query2modelcard_full_json": "query2modelcard.json",
    }
    if meta.get("table_search_reason"):
        results["table_search_reason"] = meta["table_search_reason"]
    return results


def _table_resources_from_search_results(search_results: Dict[str, Any]) -> Optional[List[str]]:
    """Parquet column scope for model→table lookup. job_meta must carry an explicit resource list."""
    tr = search_results.get("table_resources")
    if not isinstance(tr, list) or not tr:
        raise ValueError("job_meta.json is missing a non-empty `table_resources` list")
    out = [
        str(x).strip().lower()
        for x in tr
        if str(x).strip() and str(x).strip().lower() in ("hugging", "github", "arxiv", "llm")
    ]
    if not out:
        raise ValueError("job_meta.json has no valid `table_resources` values")
    return list(dict.fromkeys(out))

def _ordered_unique_strs(items: List[Any]) -> List[str]:
    return list(dict.fromkeys(str(x).strip() for x in items if str(x).strip()))


def _local_table_basenames(items: List[Any]) -> List[str]:
    out: List[str] = []
    for item in items:
        name = str(item).strip()
        if not name:
            continue
        if resolve_table_path(name):
            out.append(name)
    return _ordered_unique_strs(out)


def _local_table_paths(items: List[Any]) -> List[str]:
    out: List[str] = []
    for item in items:
        path = resolve_table_path(str(item).strip())
        if path:
            out.append(path)
    return _ordered_unique_strs(out)


def _table_resources_for_request(data: Dict[str, Any], search_results: Dict[str, Any]) -> List[str]:
    override = data.get("table_resources")
    if isinstance(override, list) and override:
        out = [
            str(x).strip().lower()
            for x in override
            if str(x).strip() and str(x).strip().lower() in ("hugging", "github", "arxiv", "llm")
        ]
        if not out:
            raise ValueError("request `table_resources` must contain at least one valid resource")
        return list(dict.fromkeys(out))
    return _table_resources_from_search_results(search_results)


def _query2modelcard_mode_from_request(raw_mode: Any) -> str:
    mode_s = str(raw_mode).strip().lower() if raw_mode is not None else ""
    if raw_mode is None or mode_s in ("", "integration", "saved", "default"):
        return "dense"
    if mode_s not in QUERY2MODELCARD_RETRIEVAL_MODES:
        raise ValueError(f"Unsupported query2modelcard_retrieval_mode: {raw_mode}")
    return mode_s


def _query2modelcard_model_ids_from_json(q2m_json: Dict[str, Any], *, retrieval_mode: Any = None, max_models: Optional[int] = None) -> tuple[str, List[str]]:
    mode = _query2modelcard_mode_from_request(retrieval_mode)
    results = q2m_json["results"]
    raw_items = results[mode]
    if not raw_items:
        return mode, []
    seed_model_id = _extract_model_id(raw_items[0])
    model_ids: List[str] = []
    for item in raw_items:
        mid = _extract_model_id(item)
        if not mid or mid == seed_model_id:
            continue
        model_ids.append(mid)
    model_ids = _ordered_unique_strs(model_ids)
    if max_models is not None:
        model_ids = model_ids[:max(0, int(max_models))]
    return mode, model_ids


def _build_model_search_preview_payload(job_meta: Dict[str, Any], q2m_json: Dict[str, Any], *, resources: List[str], retrieval_mode: Any = None, max_models: Optional[int] = None) -> Dict[str, Any]:
    mode, model_ids = _query2modelcard_model_ids_from_json(q2m_json,retrieval_mode=retrieval_mode,max_models=max_models)
    model_to_all_tables = _get_models_to_tables_batch_sql(model_ids, resources=resources)
    model_to_table_paths = {mid: _local_table_basenames(model_to_all_tables.get(mid, [])) for mid in model_ids}
    model_to_table_paths = {mid: paths for mid, paths in model_to_table_paths.items() if paths}
    models_with_tables = list(model_to_table_paths.keys())
    table_paths = _ordered_unique_strs([p for paths in model_to_table_paths.values() for p in paths])
    return {
        "preview_format_version": 1,
        "query2modelcard_retrieval_mode": mode,
        "models_with_tables": models_with_tables,
        "model_ids": models_with_tables,
        "table_paths": table_paths,
        "model_to_table_paths": model_to_table_paths,
        "job_context": {
            "query": str(job_meta.get("query") or q2m_json.get("query") or "").strip(),
            "table_search_seed_model_id": _extract_model_id(q2m_json["results"]["dense"][0]),
        },
        "stats": {
            "models_with_tables": len(models_with_tables),
            "total_unique_tables": len(table_paths),
        },
    }


def _build_table_search_preview_payload(*, job_dir: str, search_type: str, search_results: Optional[Dict[str, Any]] = None, max_models: Optional[int] = None) -> Dict[str, Any]:
    map_path = os.path.join(job_dir, f"card2tab2card_{search_type}.json")
    from src.integration.pipeline_preview import Query2Tab2CardFullMap
    preview_obj = Query2Tab2CardFullMap(map_path)
    payload = preview_obj.build_preview(search_type=search_type, max_models=max_models)
    return {
        "preview_format_version": 1,
        "search_type": search_type,
        "tables_source": payload["preview_meta"]["tables_source"],
        "query_tables": payload["query_tables"],
        "table_paths": payload["table_paths"],
        "model_to_table_paths": payload["model_to_table_paths"],
        "models_with_tables": payload["models_with_tables"],
        "pipeline_trace": payload["pipeline_trace"],
        "tab2tab_trace_rows": payload["tab2tab_trace_rows"],
        "after_model_cap_trace_rows": payload["after_model_cap_trace_rows"],
        "retrieved_table_model_rows": payload["retrieved_table_model_rows"],
        "preview_meta": payload["preview_meta"],
        "job_context": {
            "query": payload["query"],
            "table_search_seed_model_id": payload["query_seed_model_id"],
        },
        "stats": payload["stats"],
    }


def run_search_pipeline(job_id: str, query: Optional[str] = None, top_k: int = 20, model_id: Optional[str] = None, table_search_k: Optional[int] = None, use_by_type: bool = False, model_top_k: int = 5, table_resource_allowlist: Optional[List[str]] = None):
    """Run pipeline by calling CLI commands (build_index.md). All outputs under job_dir."""
    logger = jobs.get(job_id)
    if not logger:
        return

    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    start_time = time.time()

    _run_pipeline_body(logger, job_id, job_dir, start_time, query, top_k, model_id, table_search_k, use_by_type, model_top_k=model_top_k, table_resource_allowlist=table_resource_allowlist)


def _run_pipeline_body(logger: "JobLogger", job_id: str, job_dir: str, start_time: float, query: Optional[str], top_k: int, model_id: Optional[str], table_search_k: Optional[int], use_by_type: bool = False, *, model_top_k: int = 5, table_resource_allowlist: Optional[List[str]] = None):
    run_log_path = os.path.join(job_dir, "pipeline_run.log")
    logger.set_log_file(run_log_path)
    if not query or not str(query).strip():
        logger.set_status("error")
        logger.set_results(
            {
                "error": "query required",
                "model_id": None,
                "query2modelcard_results": [],
                "card2tab2card_results": {},
                "folder_path": job_dir,
                "run_log_path": run_log_path,
            }
        )
        return
    query = str(query).strip()
    logger.set_status("running")
    rt = _ensure_search_runtime()
    table_resources = list(rt["table_resources"])
    con_data = rt["con_data"]

    q2m_full_path = os.path.join(job_dir, "query2modelcard.json")
    q2m_top_k_here = min(200, 5 * int(top_k))

    q2m = Query2ModelCardSearch(query=query, top_k=q2m_top_k_here)
    q2m.search_dense(top_k=q2m_top_k_here, dense=rt["dense_full"])
    q2m.search_sparse(top_k=q2m_top_k_here, sparse=rt["sparse_full"])
    q2m.search_hybrid(top_k=q2m_top_k_here, sparse=rt["sparse_full"], dense=rt["dense_full"], candidate_factor=10)
    q2m.save_to_json(q2m_full_path)

    results_list = list(q2m.results['dense'])
    if not results_list:
        logger.set_status("error")
        logger.set_results(
            {
                "error": "No model from query",
                "model_id": None,
                "query2modelcard_results": [],
                "card2tab2card_results": {},
                "folder_path": job_dir,
                "run_log_path": run_log_path,
            }
        )
        return

    first = results_list[0]
    raw_top = first if isinstance(first, str) else (first.get("model_id") if isinstance(first, dict) else str(first))
    query_seed_model_id = str(raw_top).strip() if raw_top else ""
    if not query_seed_model_id:
        logger.set_status("error")
        logger.set_results(
            {
                "error": "Empty model_id from query",
                "model_id": None,
                "query2modelcard_results": [],
                "card2tab2card_results": {},
                "folder_path": job_dir,
                "run_log_path": run_log_path,
            }
        )
        return

    table_search_seed_model_id = str(query_seed_model_id).strip()
    table_search_k_input = max(1, int(table_search_k or 1))
    k_table = table_search_k_input

    right_max_models = 0
    table_search_empty_reason: Optional[str] = None

    q2t2c = Query2Tab2CardSearch()
    for st in CARD2TAB2CARD_TYPES:
        out_path = os.path.join(job_dir, f"card2tab2card_{st}.json")
        q2t2c.pipeline_w_query_reranker(query, con_data, rt["dense_full"], rt["dense_wtable"], search_type=st, table_top_k=k_table, table_resources=table_resources, use_tab2tab_aug=bool(USE_TAB2TAB_AUG), apply_query_rerank=True, model_top_k=int(model_top_k), q2m_top_k=1)
        q2t2c.save_full_json(out_path)
        fm = q2t2c.get_full_map()
        reranked = fm["model_rerank_map"]
        right_max_models = max(right_max_models, len(reranked))
        card2tab = fm["card2tab_map"]
        query_tables = card2tab[table_search_seed_model_id]
        if len(reranked) == 0 and len(query_tables) == 0 and table_search_empty_reason is None:
            table_search_empty_reason = (
                f"Table-search seed «{table_search_seed_model_id}» has no tables in the dataset: it is not in "
                f"{RELATIONSHIP_PARQUET} or has no csv_basename. "
                "Try another query whose top result has linked tables, or check the parquet has column modelId and rows for this model."
            )

    elapsed_total = time.time() - start_time
    job_meta = {
        "job_id": job_id,
        "query": query,
        "top_k": top_k,
        "model_top_k": model_top_k,
        "table_search_k": table_search_k_input,
        "table_resources": table_resources,
        "use_by_type": use_by_type,
        "timestamp": datetime.fromtimestamp(start_time).isoformat(),
        "running_time_seconds": round(elapsed_total, 3),
    }
    if table_search_empty_reason:
        job_meta["table_search_reason"] = table_search_empty_reason

    _save_job_meta_json(job_id, job_meta)
    logger.set_results(_read_search_results_json(job_id) or job_meta)
    logger.set_status("completed")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ----- Optional UI (single-server deploy, e.g. Hugging Face Spaces) -----
# When SERVE_UI=1 or running on port 7860, backend serves the frontend so one port is enough.
def _serve_ui():
    """Serve index and static assets with BACKEND_URL='' (same-origin API)."""
    from src.demo.frontend import RAW_HTML_TEMPLATE
    html = RAW_HTML_TEMPLATE.replace("{{BACKEND_URL}}", "")
    return render_template_string(html)


@app.route("/")
def index():
    if os.environ.get("SERVE_UI", "").strip().lower() in ("1", "true", "yes"):
        return _serve_ui()
    return jsonify({"message": "ModelSearch API. Set SERVE_UI=1 to serve the demo UI.", "docs": "/api/health"})

@app.route("/static/app.js")
def serve_app_js():
    if os.environ.get("SERVE_UI", "").strip().lower() not in ("1", "true", "yes"):
        return jsonify({"error": "Set SERVE_UI=1 to serve UI"}), 404
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app_js_path = os.path.join(static_dir, "app.js")
    if not os.path.isfile(app_js_path):
        return jsonify({"error": "app.js not found"}), 404
    with open(app_js_path, "r", encoding="utf-8") as f:
        content = f.read().replace("{{BACKEND_URL}}", "")
    return Response(content, mimetype="application/javascript")

@app.route("/static/fig/<path:filename>")
def serve_fig(filename):
    if os.environ.get("SERVE_UI", "").strip().lower() not in ("1", "true", "yes"):
        return jsonify({"error": "Set SERVE_UI=1 to serve UI"}), 404
    fig_dir = os.path.join(REPO_ROOT, "fig")
    if not os.path.isdir(fig_dir):
        return jsonify({"error": "fig not found"}), 404
    file_path = os.path.join(fig_dir, filename)
    if not os.path.isfile(file_path):
        return jsonify({"error": "file not found"}), 404
    return send_from_directory(fig_dir, filename)

@app.route("/api/search", methods=["POST"])
def search():
    data = request.json or {}

    # Legacy "mimic" mode: load saved search results by folder/template instead of running pipeline.
    search_mode = data.get("search_mode", "new")
    if search_mode == "mimic":
        job_id = data.get("folder_name")
        if not job_id:
            return jsonify({"status": "error", "message": "job_id required for mimic"}), 400
        if job_id == "template":
            return jsonify({"status": "error", "message": "No past searching, please search!"}), 400
        saved = _read_search_results_json(job_id)
        if saved is None:
            return jsonify({"status": "error", "message": f"Saved results not found: {job_id}"}), 404

        jobs[job_id] = JobLogger(job_id)
        jobs[job_id].set_results(saved)
        jobs[job_id].set_status("completed")

        out = {"status": "completed", "job_id": job_id, "results": saved}

        extras = _load_job_extras(job_id)
        out.update(extras)
        # Sanitize evaluation text so it is safe for JS templates on the frontend.
        if isinstance(out.get("evaluation_results"), dict) and "evaluation" in out["evaluation_results"]:
            out["evaluation_results"]["evaluation"] = _sanitize_for_js_template(out["evaluation_results"]["evaluation"])

        return jsonify(out)

    # Default "new" search: run the full pipeline (query-only; direct model-id entry removed).
    top_k = int(data.get("top_k", 20))
    table_search_k = data.get("table_search_k")
    model_top_k = int(data.get("model_top_k", 5))
    use_by_type = bool(data.get("use_by_type", False))

    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"status": "error", "message": "query required"}), 400

    job_id = _generate_job_id()
    jobs[job_id] = JobLogger(job_id)

    thread = threading.Thread(
        target=run_search_pipeline,
        kwargs={
            "job_id": job_id,
            "query": query,
            "top_k": top_k,
            "model_id": None,
            "table_search_k": table_search_k,
            "use_by_type": use_by_type,
            "model_top_k": model_top_k,
        },
    )
    thread.daemon = True
    thread.start()
    return jsonify({"status": "started", "job_id": job_id, "message": "Search pipeline started"})

@app.route("/api/status/<job_id>", methods=["GET"])
def get_status(job_id: str):
    if job_id not in jobs:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    logger = jobs[job_id]
    return jsonify({"job_id": job_id, "status": logger.status, "logs": logger.get_logs()})

def _load_job_extras(job_id: str) -> dict:
    """Load model_search_runs, table_search_runs, evaluation, qa from job dir."""
    out = {}
    job_dir = os.path.join(JOBS_DIR, job_id)
    if not os.path.isdir(job_dir):
        return out
    for key, filename in [
        ("integration_model_search", "integration_model_search.json"),
        ("integration_table_search", "integration_table_search.json"),
        ("evaluation_results", "evaluation_results.json"),
        ("qa_results", "qa_results.json"),
    ]:
        p = os.path.join(job_dir, filename)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                out[key] = json.load(f)
    model_runs = []
    table_runs = []
    for fname in sorted(os.listdir(job_dir)):
        if fname.startswith("integration_model_search_") and fname.endswith(".json"):
            with open(os.path.join(job_dir, fname), "r", encoding="utf-8") as f:
                d = json.load(f)
            key = fname.replace("integration_model_search_", "").replace(".json", "")
            model_runs.append({"key": key, **d})
        elif fname.startswith("integration_table_search_") and fname.endswith(".json"):
            with open(os.path.join(job_dir, fname), "r", encoding="utf-8") as f:
                d = json.load(f)
            key = fname.replace("integration_table_search_", "").replace(".json", "")
            table_runs.append({"key": key, **d})
        elif fname.startswith("integration_run_") and fname.endswith(".json"):
            with open(os.path.join(job_dir, fname), "r", encoding="utf-8") as f:
                run = json.load(f)
            m, t = run.get("model_result"), run.get("table_result")
            if m and m.get("status") == "success":
                mk = _model_search_key(run.get("integration_type"))
                if not any(r["key"] == mk for r in model_runs):
                    model_runs.append({"key": mk, **m})
            if t and t.get("status") == "success":
                tk = _table_search_key(run.get("integration_type"), run.get("search_type"))
                if not any(r["key"] == tk for r in table_runs):
                    table_runs.append({"key": tk, **t})
    if model_runs:
        out["model_search_runs"] = model_runs
    elif out.get("integration_model_search") and out["integration_model_search"].get("status") == "success":
        m = out["integration_model_search"]
        out["model_search_runs"] = [{"key": _model_search_key(m.get("integration_type")), **m}]
    if table_runs:
        out["table_search_runs"] = table_runs
    elif out.get("integration_table_search") and out["integration_table_search"].get("status") == "success":
        t = out["integration_table_search"]
        out["table_search_runs"] = [{"key": _table_search_key(t.get("integration_type"), t.get("search_type")), **t}]
    return out


@app.route("/api/results/<job_id>", methods=["GET"])
def get_results(job_id: str):
    if job_id not in jobs:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    logger = jobs[job_id]
    if logger.status == "error" and logger.results is not None:
        # Preserve error status so failures are explicit for debugging; results payload may contain more detail.
        resp = {"status": "error", "job_id": job_id, "results": logger.results}
        resp.update(_load_job_extras(job_id))
        return jsonify(resp)
    if logger.status != "completed":
        return jsonify({"status": logger.status, "message": "Job not completed yet"}), 202
    resp = {"status": "success", "job_id": job_id, "results": logger.results}
    resp.update(_load_job_extras(job_id))
    return jsonify(resp)

@app.route("/api/logs/<job_id>", methods=["GET"])
def stream_logs(job_id: str):
    if job_id not in jobs:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    
    def generate():
        logger = jobs[job_id]
        last = 0
        while logger.status in ("pending", "running"):
            logs = logger.get_logs()
            for log in logs[last:]:
                yield f"data: {json.dumps(log)}\n\n"
            last = len(logs)
            time.sleep(0.5)
        logs = logger.get_logs()
        for log in logs[last:]:
            yield f"data: {json.dumps(log)}\n\n"
        yield f"data: {json.dumps({'status': 'completed'})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.route("/api/preset-queries", methods=["GET"])
def get_preset_queries():
    if not os.path.exists(PRESET_QUERIES_PATH):
        return jsonify({"status": "success", "queries": []})
    with open(PRESET_QUERIES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data.get("queries", data) if isinstance(data, dict) else data
    return jsonify({"status": "success", "queries": queries})


@app.route("/api/table-preview", methods=["GET"])
def get_table_preview():
    """JSON snippet for inline expand (integration / toggles); caps rows/cols for speed."""
    path_q = (request.args.get("path") or "").strip()
    if not path_q:
        return jsonify({"status": "error", "message": "path query parameter required"}), 400
    df, source, bn = preview_from_local(path_q, max_rows=MAX_TABLE_PREVIEW_JSON_ROWS, max_cols=MAX_TABLE_PREVIEW_JSON_COLS)
    if df is None:
        return jsonify({"status": "error", "message": "Table file not found in local CSV roots"}), 404
    ncols = int(df.shape[1])
    if ncols > MAX_TABLE_PREVIEW_JSON_COLS:
        df = df.iloc[:, :MAX_TABLE_PREVIEW_JSON_COLS]
    preview = df.head(min(120, len(df)))
    return jsonify(
        {
            "status": "success",
            "html": _dataframe_to_html_table(preview),
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "source": source,
            "table": bn or "",
        }
    )


def make_table_page_response(path_q: str) -> Response:
    """Build full-page HTML for one CSV (used by /api/table-page on API and UI servers)."""
    path_q = (path_q or "").strip()
    if not path_q:
        return Response("Missing path", status=400, mimetype="text/plain; charset=utf-8")
    df, source, bn = preview_from_local(path_q, max_rows=MAX_TABLE_PAGE_ROWS, max_cols=MAX_TABLE_PAGE_COLS)
    if df is None:
        return Response("<!DOCTYPE html><html><body><p>Table file not found in local CSV roots.</p></body></html>", status=404, mimetype="text/html; charset=utf-8")
    total_cols = int(df.shape[1])
    col_note = ""
    if total_cols > MAX_TABLE_PAGE_COLS:
        df = df.iloc[:, :MAX_TABLE_PAGE_COLS]
        col_note = f" Showing first {MAX_TABLE_PAGE_COLS} of {total_cols} columns."
    row_note = ""
    if len(df) >= MAX_TABLE_PAGE_ROWS:
        row_note = f" Showing first {MAX_TABLE_PAGE_ROWS} rows (file may contain more)."
    note = (row_note + col_note).strip()
    title = bn or os.path.basename(str(path_q))
    display_path = source or str(path_q)
    body = render_template_string(TABLE_PAGE_HTML, title=title, path_display=display_path, note=note, table_html=_dataframe_to_html_table(df))
    return Response(body, mimetype="text/html; charset=utf-8")


@app.route("/api/table-page", methods=["GET"])
def table_page():
    """Full-page HTML table view (new tab from retrieval results)."""
    return make_table_page_response(request.args.get("path") or "")


@app.route("/api/table-search-preview", methods=["POST"])
def table_search_preview():
    """Prepare query2tab2card relationship preview without running integration."""
    data = request.get_json() or {}
    job_id = data['job_id']
    search_type = str(data.get("search_type") or "single_column").strip()
    job_dir = os.path.join(JOBS_DIR, job_id)
    sr = _read_search_results_json(job_id)
    preview_payload = _build_table_search_preview_payload(job_dir=job_dir,search_type=search_type,search_results=sr)
    return jsonify({"status": "success", **preview_payload})


@app.route("/api/model-search-preview", methods=["POST"])
def model_search_preview():
    """Prepare Query2Card relationship preview directly from ``query2modelcard.json``."""
    data = request.get_json() or {}
    job_id = data['job_id']

    job_meta = _read_job_meta_json(job_id)
    q2m_json = _read_query2modelcard_json(job_id)
    if not isinstance(job_meta, dict) or not isinstance(q2m_json, dict):
        return _api_error(f"job artifacts not found for job {job_id}", 404)
    try:
        resources = _table_resources_for_request(data, job_meta)
        preview_payload = _build_model_search_preview_payload(
            job_meta,
            q2m_json,
            resources=resources,
            retrieval_mode=data.get("query2modelcard_retrieval_mode"),
        )
    except Exception as exc:
        return jsonify({"status": "no_result", "message": str(exc)})
    if not preview_payload["models_with_tables"]:
        return jsonify({"status": "no_result", "message": "No locally resolvable tables found for the selected model-search results", **preview_payload})
    return jsonify({"status": "success", **preview_payload})

@app.route("/api/saved-searches", methods=["GET"])
def list_saved_searches():
    candidates = []
    for name in os.listdir(JOBS_DIR):
        path = os.path.join(JOBS_DIR, name)
        meta_path = os.path.join(path, JOB_META_FILENAME)
        q2m_path = os.path.join(path, "query2modelcard.json")
        if os.path.isdir(path) and os.path.isfile(meta_path) and os.path.isfile(q2m_path):
            mtime = os.path.getmtime(meta_path)
            candidates.append((name, path, mtime))
    candidates.sort(key=lambda x: x[2], reverse=True)
    searches = []
    for name, path, _ in candidates[:50]:
        entry = {"folder_name": name, "path": path, "query": None, "model_id": None, "timestamp_str": "", "top_k": None, "model_top_k": None, "use_by_type": False, "table_search_k": None}
        with open(os.path.join(path, JOB_META_FILENAME), "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(os.path.join(path, "query2modelcard.json"), "r", encoding="utf-8") as f:
            q2m = json.load(f)
        q2m_results = q2m["results"]
        dense = q2m_results["dense"]
        seed_model_id = _extract_model_id(dense[0]) if dense else ""
        entry["query"] = meta.get("query") or q2m.get("query") or ""
        entry["model_id"] = seed_model_id
        ts = meta.get("timestamp")
        if ts:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            entry["timestamp_str"] = dt.strftime("%Y-%m-%d %H:%M")
        entry["top_k"] = meta.get("top_k")
        entry["model_top_k"] = meta.get("model_top_k")
        entry["use_by_type"] = bool(meta.get("use_by_type", False))
        entry["table_search_k"] = meta.get("table_search_k")
        searches.append(entry)
    return jsonify({"status": "success", "searches": searches})


# Stubs for integrate / evaluate / qa (use legacy backend for full support)
@app.route("/api/integrate", methods=["POST"])
def integrate():
    """Integrate tables from Card2Tab2Card search results"""
    t_integrate = time.time()
    data = request.get_json()
    job_id = data['job_id']
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    # Default table-search mode should prefer unionable.
    search_type = data.get("search_type", "unionable")
    # Default integration method should match the UI default (ALITE).
    integration_type = data.get("integration_type", "alite")
    k = int(data.get("k", 10))
    max_models = int(data.get("max_models", 10))
    tables_source = data.get("tables_source", "intermediate")
    run_key = _table_search_key(integration_type, search_type, tables_source)
    sr = _read_search_results_json(job_id)
    if not isinstance(sr, dict):
        return _api_error(f"job artifacts not found for job {job_id}", 404)

    try:
        preview_payload = _build_table_search_preview_payload(
            job_dir=job_dir,
            search_type=search_type,
            search_results=sr,
            max_models=max_models,
        )
        table_paths = preview_payload["table_paths"][:max(0, k)]
        resolved_table_paths = _local_table_paths(table_paths)
        if not resolved_table_paths:
            raise ValueError("No locally resolvable tables found for the selected Query2Tab2Card results")
        from src.integration.table_integration import TableIntegrater
        integrated_df = TableIntegrater().run(resolved_table_paths, mode=integration_type)
    except Exception as exc:
        result = {"success": False, "error": str(exc)}
    else:
        result = {
            "success": integrated_df is not None,
            "integrated_table": integrated_df,
            **preview_payload,
        }

    elapsed_s = round(time.time() - t_integrate, 4)
    if not result.get("success", False):
        save_payload = {
            "status": "no_result",
            "integration_type": integration_type,
            "search_type": search_type,
            "tables_source": tables_source,
            "k": k,
            "max_models": max_models,
            "integration_elapsed_s": elapsed_s,
            "error": result.get("error", "Integration failed"),
            "message": result.get("error", "Integration failed"),
        }
        json_path = os.path.join(job_dir, f"integration_table_search_{run_key}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(save_payload, f, ensure_ascii=False, indent=0)
        line = f"Integration (table search) FAILED after {elapsed_s:.2f}s — {save_payload['message']}"
        print(f"[integrate] job_id={job_id} {line}", flush=True)
        _append_pipeline_log(job_dir, line)
        return jsonify({"status": "no_result", "message": save_payload["message"], **save_payload})

    # Convert DataFrame to dict for JSON response (NaN -> null for valid JSON).
    integrated_df = result.get("integrated_table")
    saved_path = None
    if integrated_df is not None:
        raw_data = integrated_df.values.tolist()
        result["integrated_table"] = {"columns": list(integrated_df.columns), "data": _sanitize_for_json(raw_data)}
        csv_name = f"integrated_table_search_{run_key}.csv"
        save_path = os.path.join(job_dir, csv_name)
        integrated_df.to_csv(save_path, index=False, encoding="utf-8")
        saved_path = f"data_251117/jobs_251117/{job_id}/{csv_name}"
    if saved_path:
        result["saved_path"] = saved_path
    if "models_with_tables" not in result:
        result["models_with_tables"] = []
    elapsed_s = round(time.time() - t_integrate, 4)
    save_payload = {
        "status": "success",
        "integration_type": integration_type,
        "search_type": search_type,
        "tables_source": tables_source,
        "k": k,
        "max_models": max_models,
        "integration_elapsed_s": elapsed_s,
        **result,
    }
    json_path = os.path.join(job_dir, f"integration_table_search_{run_key}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, ensure_ascii=False, indent=0)
    with open(os.path.join(job_dir, "integration_table_search.json"), "w", encoding="utf-8") as f:
        json.dump(save_payload, f, ensure_ascii=False, indent=0)
    line = (
        f"Integration (table search) OK in {elapsed_s:.2f}s "
        f"(type={integration_type}, search_type={search_type}, tables_source={tables_source})"
    )
    print(f"[integrate] job_id={job_id} {line}", flush=True)
    _append_pipeline_log(job_dir, line)
    return jsonify({"status": "success", "integration_elapsed_s": elapsed_s, **result})



@app.route("/api/integrate-model-search", methods=["POST"])
def integrate_model_search():
    """Integrate tables from query2modelcard neighbor list (model search)."""
    t_integrate = time.time()
    data = request.get_json()
    job_id = data['job_id']
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    integration_type = data.get("integration_type", "alite")
    k = int(data.get("k", 10))
    max_models = int(data.get("max_models", 10))
    run_key = _model_search_key(integration_type)
    job_meta = _read_job_meta_json(job_id)
    q2m_json = _read_query2modelcard_json(job_id)
    if not isinstance(job_meta, dict) or not isinstance(q2m_json, dict):
        return _api_error(f"job artifacts not found for job {job_id}", 404)

    try:
        resources = _table_resources_for_request(data, job_meta)
        preview_payload = _build_model_search_preview_payload(
            job_meta,
            q2m_json,
            resources=resources,
            max_models=max_models,
        )
        table_paths = preview_payload["table_paths"][:max(0, k)]
        resolved_table_paths = _local_table_paths(table_paths)
        if not resolved_table_paths:
            raise ValueError("No locally resolvable tables found for the selected Query2Card results")
        from src.integration.table_integration import TableIntegrater
        integrated_df = TableIntegrater().run(resolved_table_paths, mode=integration_type)
    except Exception as exc:
        result = {"success": False, "error": str(exc)}
    else:
        result = {
            "success": integrated_df is not None,
            "integrated_table": integrated_df,
            **preview_payload,
        }

    elapsed_s = round(time.time() - t_integrate, 4)
    if not result.get("success", False):
        save_payload = {
            "status": "no_result",
            "integration_type": integration_type,
            "k": k,
            "max_models": max_models,
            "integration_elapsed_s": elapsed_s,
            "error": result.get("error", "Integration failed"),
            "message": result.get("error", "Integration failed"),
        }
        json_path = os.path.join(job_dir, f"integration_model_search_{run_key}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(save_payload, f, ensure_ascii=False, indent=0)
        line = f"Integration (model search) FAILED after {elapsed_s:.2f}s — {save_payload['message']}"
        print(f"[integrate-model-search] job_id={job_id} {line}", flush=True)
        _append_pipeline_log(job_dir, line)
        return jsonify({"status": "no_result", "message": save_payload["message"], **save_payload})

    # Convert DataFrame to dict for JSON response (NaN -> null for valid JSON).
    integrated_df = result.get("integrated_table")
    saved_path = None
    if integrated_df is not None:
        raw_data = integrated_df.values.tolist()
        result["integrated_table"] = {"columns": list(integrated_df.columns), "data": _sanitize_for_json(raw_data)}
        csv_name = f"integrated_model_search_{run_key}.csv"
        save_path = os.path.join(job_dir, csv_name)
        integrated_df.to_csv(save_path, index=False, encoding="utf-8")
        saved_path = f"data_251117/jobs_251117/{job_id}/{csv_name}"
    if saved_path:
        result["saved_path"] = saved_path
    elapsed_s = round(time.time() - t_integrate, 4)
    save_payload = {
        "status": "success",
        "integration_type": integration_type,
        "k": k,
        "max_models": max_models,
        "integration_elapsed_s": elapsed_s,
        **result,
    }
    json_path = os.path.join(job_dir, f"integration_model_search_{run_key}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, ensure_ascii=False, indent=0)
    with open(os.path.join(job_dir, "integration_model_search.json"), "w", encoding="utf-8") as f:
        json.dump(save_payload, f, ensure_ascii=False, indent=0)
    line = f"Integration (model search) OK in {elapsed_s:.2f}s (type={integration_type})"
    print(f"[integrate-model-search] job_id={job_id} {line}", flush=True)
    _append_pipeline_log(job_dir, line)
    return jsonify({"status": "success", "integration_elapsed_s": elapsed_s, **result})

def _load_integrated_table_from_json(job_dir: str, json_name: str) -> Optional[pd.DataFrame]:
    """Load integrated table from integration JSON (has integrated_table with columns + data)."""
    path = os.path.join(job_dir, json_name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tbl = data.get("integrated_table")
    if not tbl:
        return None
    cols = tbl.get("columns") or []
    rows = tbl.get("data") or []
    if not cols and not rows:
        return None
    return pd.DataFrame(rows, columns=cols) if cols else pd.DataFrame(rows)

def _load_tables_from_integration_run(job_dir: str, run_key: str):
    """Load (table1_df, table2_df) from integration_run_<key>.json. table1=Table Search, table2=Model Search."""
    safe_key = re.sub(r"[^a-z0-9_]", "_", (run_key or "").lower().strip()) or "run"
    path = os.path.join(job_dir, f"integration_run_{safe_key}.json")
    if not os.path.isfile(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        run_data = json.load(f)
    t_res = run_data.get("table_result") or {}
    m_res = run_data.get("model_result") or {}
    tbl1 = (t_res.get("integrated_table") or {})
    tbl2 = (m_res.get("integrated_table") or {})
    cols1, rows1 = tbl1.get("columns") or [], tbl1.get("data") or []
    cols2, rows2 = tbl2.get("columns") or [], tbl2.get("data") or []
    df1 = pd.DataFrame(rows1, columns=cols1) if cols1 or rows1 else pd.DataFrame()
    df2 = pd.DataFrame(rows2, columns=cols2) if cols2 or rows2 else pd.DataFrame()
    return (df1 if not df1.empty else None), (df2 if not df2.empty else None)

@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """Evaluate quality (relevance, coverage, diversity) between Table Search and Model Search integrated tables using LLM."""
    data = request.get_json()
    job_id = data['job_id']
    job_dir = os.path.join(JOBS_DIR, job_id)
    integration_run_key = data.get("integration_run_key")

    # Load integrated tables: table1 = Table Search, table2 = Model Search
    table1_df, table2_df = None, None
    if integration_run_key:
        table1_df, table2_df = _load_tables_from_integration_run(job_dir, integration_run_key)
    if table1_df is None or table2_df is None:
        table1_df = _load_integrated_table_from_json(job_dir, "integration_table_search.json")
        table2_df = _load_integrated_table_from_json(job_dir, "integration_model_search.json")

    if table1_df is None or table1_df.empty:
        return jsonify({"status": "error", "message": "Table Search integration not found. Please run Table Search integration first."}), 400
    if table2_df is None or table2_df.empty:
        return jsonify({"status": "error", "message": "Model Search integration not found. Please run Model Search integration first."}), 400

    sr = _read_search_results_json(job_id)
    query = _resolve_eval_qa_query(data, sr)
    if not query:
        return jsonify(
            {
                "status": "error",
                "message": "Missing search query: pass a non-empty `query` in the request body, or ensure this job has `job_meta.json` and `query2modelcard.json`.",
            }
        ), 400

    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    from evaluation.llm import evaluate_diversity_with_llm
    result = evaluate_diversity_with_llm(query=query, table1=table1_df, table2=table2_df, table1_source="Table Search Integration", table2_source="Model Search Integration")

    result = _sanitize_for_js_template(result)

    # Convert DataFrames to frontend format for optional display.
    # At this point, tables loaded from JSON/CSV are already in their
    # deterministic saved order; we do not reorder them again here.
    def _df_to_dict(df: pd.DataFrame) -> Optional[Dict]:
        if df is None or df.empty:
            return None
        return {"columns": list(df.columns), "data": _sanitize_for_json(df.values.tolist())}

    return jsonify({"status": "success", "evaluation": result, "table1": _df_to_dict(table1_df), "table2": _df_to_dict(table2_df)})


@app.route("/api/qa", methods=["POST"])
def qa():
    """Answer question based on integrated table using LLM."""
    data = request.get_json()
    job_id = data['job_id']
    job_dir = os.path.join(JOBS_DIR, job_id)
    use_table_search = bool(data.get("use_table_search", True))

    # Load the appropriate integrated table
    if use_table_search:
        table_df = _load_integrated_table_from_json(job_dir, "integration_table_search.json")
        table_source = "Table Search Integration"
        qa_mode = "card2tab2card"
    else:
        table_df = _load_integrated_table_from_json(job_dir, "integration_model_search.json")
        table_source = "Model Search Integration"
        qa_mode = "query2modelcard"

    if table_df is None:
        table_df = pd.DataFrame()

    sr = _read_search_results_json(job_id)
    query = _resolve_eval_qa_query(data, sr)
    if not query:
        return jsonify(
            {
                "status": "error",
                "message": "Missing search query: pass a non-empty `query` in the request body, or ensure this job has `job_meta.json` and `query2modelcard.json`.",
            }
        ), 400

    search_results_data = sr if isinstance(sr, dict) else None
    model_ids_to_rank = None
    if isinstance(sr, dict) and not use_table_search:
        mids = sr["query2modelcard_results"]
        if mids:
            model_ids_to_rank = list(mids)[:50]

    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    from evaluation.llm_qa import answer_question_with_llm
    result = answer_question_with_llm(query=query, table=table_df, table_source=table_source, qa_mode=qa_mode, model_ids_to_rank=model_ids_to_rank, search_results_data=search_results_data)

    qa_answer = result.get("answer")
    if not isinstance(qa_answer, dict):
        qa_answer = {"answer": str(qa_answer) if qa_answer else "No answer provided", "model_ranking": [], "summary": {}, "confidence": "medium", "limitations": []}

    return jsonify({"status": "success", "qa": qa_answer, "query": query})


@app.route("/api/integration-runs/<job_id>", methods=["GET"])
def get_integration_runs(job_id: str):
    """Load model_search_runs and table_search_runs for a job (separate storage)."""
    if not job_id or not job_id.strip():
        return jsonify({"status": "success", "job_id": job_id or "", "model_search_runs": [], "table_search_runs": []})
    out = _load_job_extras(job_id.strip())
    return jsonify({"status": "success", "job_id": job_id, "model_search_runs": out.get("model_search_runs", []), "table_search_runs": out.get("table_search_runs", [])})


@app.route("/api/save-integration-run", methods=["POST"])
def save_integration_run():
    """Save one integration run to job_dir as integration_run_<key>.json for tabs. Creates job_dir if missing."""
    data = request.get_json()
    job_id = data['job_id']
    key = data.get("key")
    if not key or (isinstance(key, str) and not key.strip()):
        return _api_error("job_id and key required", 400)
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    safe_key = re.sub(r"[^a-z0-9_]", "_", (key or "").lower().strip()) or "run"
    path = os.path.join(job_dir, f"integration_run_{safe_key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=0)
    return jsonify({"status": "success"})


@app.route("/api/save-evaluation", methods=["POST"])
def save_evaluation():
    """Save evaluation result to job_dir for load-previous restore. Creates job_dir if missing."""
    data = request.get_json()
    job_id = data['job_id']
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    path = os.path.join(job_dir, "evaluation_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=0)
    return jsonify({"status": "success"})


@app.route("/api/save-qa", methods=["POST"])
def save_qa():
    """Save QA result to job_dir for load-previous restore. Creates job_dir if missing."""
    data = request.get_json()
    job_id = data['job_id']
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    path = os.path.join(job_dir, "qa_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=0)
    return jsonify({"status": "success"})


if __name__ == "__main__":
    import argparse as _argparse

    parser = _argparse.ArgumentParser(description="ModelSearch Demo Backend")
    parser.add_argument("--port", type=int, default=None, help="Port (default: env PORT or 5002)")
    args, _ = parser.parse_known_args()

    init_search_runtime()
    port = args.port if args.port is not None else int(os.environ.get("PORT", "5002"))
    app.run(host="0.0.0.0", port=port, debug=False)
