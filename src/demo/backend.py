"""
Backend API for ModelSearch Demo.

Runs search in-process.
All job outputs (search results, integration, evaluation, QA) go under JOBS_DIR from config (e.g. data_<tag>/jobs_<tag>/<job_id>).
Minimal imports for fast startup.
"""

import os, sys, json, random, string, threading, time, math, re, html, shutil
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify, Response, stream_with_context, render_template_string, send_from_directory
from flask_cors import CORS
from datetime import datetime
#from src.config import REPO_ROOT, JOBS_DIR, CARD2TAB2CARD_TIMEOUT, USE_BY_TYPE, CARD2CARD_MODES, CARD2TAB2CARD_TYPES, CARD2TAB2CARD_OUTPUT_JSON, VALID_MODEL_IDS_TXT, CLASSIFICATION_JSON, TABLE_RESOURCE_ALLOWLIST, RELATIONSHIP_PARQUET, PRESET_QUERIES_PATH
from src.config import *
from src.utils import resolve_table_path, get_device
#from src.utils import filter_results_by_classify_results

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


def _model_search_key(integration_type: str, card2card_mode: str) -> str:
    """Slug for Model Search only: e.g. alite_dense. Params: integration_type, card2card_mode."""
    parts = [integration_type or "union", card2card_mode or "dense"]
    return "_".join(re.sub(r"[^a-z0-9_]", "_", (p or "").lower().strip()) for p in parts)


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


def _require_job_id(data: Optional[Dict]) -> tuple:
    """Returns (job_id, None) or (None, error_response). Caller should return error_response if not None. job_id is always returned as str."""
    safe = data if isinstance(data, dict) else {}
    job_id = safe.get("job_id")
    if not job_id or (isinstance(job_id, str) and not job_id.strip()):
        return None, _api_error("job_id required", 400)
    return (str(job_id).strip(), None)


def _require_results_file(job_id: str) -> tuple:
    """Returns (job_dir, results_file, None) or (None, None, error_response)."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    results_file = os.path.join(job_dir, "search_results.json")
    if not os.path.exists(results_file):
        return None, None, _api_error(f"Results file not found for job {job_id}", 404)
    return job_dir, results_file, None


def _require_job_dir(job_id: str) -> tuple:
    """Returns (job_dir, None) or (None, error_response)."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    if not os.path.isdir(job_dir):
        return None, _api_error(f"Job directory not found: {job_id}", 404)
    return job_dir, None


def _integration_saved_path_for_api(job_id: str, csv_name: str) -> str:
    """Repo-relative path for API/UI display only (fixed prefix; files still written under JOBS_DIR)."""
    return f"data_251117/jobs_251117/{job_id}/{csv_name}"


def _allowed_csv_roots() -> List[str]:
    """Only the three processed table index dirs (config) and job output dir — no wider ModelTables/data."""
    roots: List[str] = []
    for d in list(TABLE_BASE_DIRS) + [JOBS_DIR]:
        if not d:
            continue
        try:
            if os.path.isdir(d):
                roots.append(os.path.realpath(d))
        except OSError:
            continue
    return roots


def _path_is_under_roots(candidate: str, roots: List[str]) -> bool:
    try:
        cr = os.path.realpath(candidate)
    except OSError:
        return False
    for r in roots:
        if cr == r or cr.startswith(r + os.sep):
            return True
    return False


def _real_file_candidates(raw_path: str) -> List[str]:
    """Absolute realpaths to try for a path string from search results (may contain ../)."""
    if not raw_path or not isinstance(raw_path, str):
        return []
    s = raw_path.strip()
    if not s:
        return []
    seen = set()
    out: List[str] = []

    def push(p: str) -> None:
        try:
            rp = os.path.realpath(p)
            if os.path.isfile(rp) and rp not in seen:
                seen.add(rp)
                out.append(rp)
        except OSError:
            return

    if os.path.isabs(s):
        push(s)
    else:
        # Paths like ../Repo/ModelTables/data/processed/... are stored relative to job CWD
        # or repo root; anchoring to REPO_ROOT collapses .. correctly.
        push(os.path.join(str(REPO_ROOT), s))
    return out


def _resolve_table_file_for_preview(raw_path: str) -> Optional[str]:
    """Resolve a CSV path only under TABLE_BASE_DIRS (three processed dirs) or JOBS_DIR."""
    roots = _allowed_csv_roots()
    for cand in _real_file_candidates(raw_path):
        if roots and _path_is_under_roots(cand, roots):
            return cand
    s = (raw_path or "").strip()
    if not s:
        return None
    if not roots:
        return None
    r = resolve_table_path(s)
    if r and os.path.isfile(r):
        cand = os.path.realpath(r)
        if _path_is_under_roots(cand, roots):
            return cand
    rel = os.path.normpath(os.path.join(str(REPO_ROOT), s.lstrip("/\\")))
    if os.path.isfile(rel):
        cand = os.path.realpath(rel)
        if _path_is_under_roots(cand, roots):
            return cand
    return None


def _html_table_cell(v: Any) -> str:
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    if v is None:
        return ""
    return html.escape(str(v))


def _dataframe_to_html_table(df: pd.DataFrame) -> str:
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
    """Read JSON file if it exists; else return None. No try/except - caller handles."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_json_job(job_id: str, filename:str) -> Optional[Dict]:
    """Read JSON file from job directory."""
    return _read_json(os.path.join(JOBS_DIR, job_id, filename))

def _write_json_job(job_id: str, filename: str, data: Dict):
    """Write JSON data to job directory."""
    with open(os.path.join(JOBS_DIR, job_id, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _ordered_model_ids_from_q2m_results(results_list: Any) -> List[str]:
    """Stable de-dupe preserving query2modelcard rank order (str ids)."""
    if not isinstance(results_list, list):
        return []
    out: List[str] = []
    seen: set = set()
    for r in results_list:
        mid = r if isinstance(r, str) else (r.get("model_id") if isinstance(r, dict) else str(r))
        if not mid:
            continue
        mid_s = str(mid).strip()
        if not mid_s or mid_s in seen:
            continue
        seen.add(mid_s)
        out.append(mid_s)
    return out


def run_search_pipeline(
    job_id: str,
    query: Optional[str] = None,
    top_k: int = 20,
    model_id: Optional[str] = None,
    table_search_k: Optional[int] = None,
    card2card_retrieval_mode: str = "dense",
    require_seed_has_tables: bool = True,
    use_by_type: bool = False,
    model_top_k: int = 5,
    table_resource_allowlist: Optional[List[str]] = None,
):
    """Run pipeline by calling CLI commands (build_index.md). All outputs under job_dir."""
    logger = jobs.get(job_id)
    if not logger:
        return

    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    start_time = time.time()

    def _set_pipeline_error(msg: str, mid=None):
        logger.set_status("error")
        d = {"error": msg, "model_id": mid, "card2card_results": [], "card2tab2card_results": {}}
        d["folder_path"] = job_dir
        d["run_log_path"] = os.path.join(job_dir, "pipeline_run.log")
        logger.set_results(d)

    try:
        _run_pipeline_body(logger, job_id, job_dir, start_time, query, top_k, model_id, table_search_k, card2card_retrieval_mode, require_seed_has_tables, use_by_type, model_top_k=model_top_k, table_resource_allowlist=table_resource_allowlist)
    except Exception as e:
        logger.log(f"Pipeline crashed: {e}")
        _set_pipeline_error(f"Pipeline error: {e}", model_id)
        import traceback
        logger.log(traceback.format_exc())

def _run_pipeline_body(logger: "JobLogger", job_id: str, job_dir: str, start_time: float, query: Optional[str], top_k: int, model_id: Optional[str], table_search_k: Optional[int], card2card_retrieval_mode: str, require_seed_has_tables: bool = True, use_by_type: bool = False, *, model_top_k: int = 5, table_resource_allowlist: Optional[List[str]] = ['hugging']):
    # Write all logs to job_dir/pipeline_run.log for debugging
    run_log_path = os.path.join(job_dir, "pipeline_run.log")
    logger.set_log_file(run_log_path)
    logger.log("=" * 60)
    logger.log(f"Job directory (all outputs saved here): {os.path.abspath(job_dir)}")
    logger.log(f"Run log file: {os.path.abspath(run_log_path)}")
    logger.log("=" * 60)
    logger.log("Starting search pipeline (CLI)...")
    logger.log("Mode: Query → ModelCard → Search")
    if not query or not str(query).strip():
        logger.log("query is required (model-id-only pipeline removed)")
        logger.set_status("error")
        logger.set_results(
            {
                "error": "query required",
                "model_id": None,
                "card2card_results": [],
                "card2tab2card_results": {},
                "folder_path": job_dir,
                "run_log_path": run_log_path,
            }
        )
        return
    query = str(query).strip()
    logger.log(f"Query: {query}")
    logger.log(
        f"Run settings: top_k={top_k}, model_top_k={model_top_k}, per_table_search_k={table_search_k or 1}, "
        f"card2card={card2card_retrieval_mode}, "
        f"table type classification (by_type)={'ON' if use_by_type else 'OFF'}, "
        f"require_seed_has_tables={require_seed_has_tables}"
    )
    # Print runtime device once per pipeline run for easier debugging/profiling.
    dev = get_device()
    if dev == "cuda":
        try:
            import torch
            gpu_count = int(torch.cuda.device_count())
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "unknown"
            logger.log(f"Runtime device: {dev} (gpu_count={gpu_count}, gpu0={gpu_name})")
        except Exception as e:
            logger.log(f"Runtime device: {dev} (gpu details unavailable: {e})")
    else:
        logger.log(f"Runtime device: {dev}")
    logger.set_status("running")

    # Normalize resource selection once and reuse in in-process calls.
    raw_resources = list(table_resource_allowlist or TABLE_RESOURCE_ALLOWLIST)
    table_resources = [r for r in raw_resources if r in ("hugging", "github", "arxiv")]
    if not table_resources:
        table_resources = list(TABLE_RESOURCE_ALLOWLIST)
    # Query2Card / card2card search: full corpus (EMB_NPZ + full sparse when hybrid).
    model_resources_full = ["hugging", "github", "arxiv"]
    # Table-search seed uses hugging-only embeddings by current table_resources default.
    logger.log(f"Resources: Query2Card/card2card={model_resources_full}, table CLI={table_resources}")

    def _table_emb_npz_for_resources(resources: List[str]) -> str:
        rset = set(str(r).strip().lower() for r in (resources or []) if str(r).strip())
        if rset == {"hugging"}:
            return EMB_NPZ_HUGGING
        if rset == {"hugging", "github", "arxiv"}:
            return EMB_NPZ
        # Keep same behavior as query2tab2card._resource_paths fallback constraints.
        return EMB_NPZ_HUGGING

    from src.search.query2modelcard import get_query2modelcard_dense_runtime
    full_emb_npz = _table_emb_npz_for_resources(model_resources_full)
    logger.log(f"[Query2ModelCard-FULL] Preload dense runtime from npz: {full_emb_npz}")
    t_preload_full = time.time()
    dense_runtime_full = get_query2modelcard_dense_runtime(emb_npz_path=full_emb_npz)
    logger.log(f"[Query2ModelCard-FULL] Dense runtime ready in {time.time() - t_preload_full:.2f}s")

    table_emb_npz = _table_emb_npz_for_resources(table_resources)
    logger.log(f"[Query2Tab2Card] Preload dense runtime from npz: {table_emb_npz}")
    t_preload = time.time()
    dense_runtime_table = get_query2modelcard_dense_runtime(emb_npz_path=table_emb_npz)
    logger.log(f"[Query2Tab2Card] Dense runtime ready in {time.time() - t_preload:.2f}s")

    # Table Search now delegates full flow to query2tab2card:
    # query -> query2modelcard -> card2tab2card -> query-rerank.
    # `require_seed_has_tables` is kept for API compatibility but no longer gates execution here.
    if require_seed_has_tables:
        logger.log(
            "require_seed_has_tables=True is accepted for backward compatibility; "
            "table flow now always runs query2tab2card and lets that module own seed + mapping logic."
        )

    q2m_ordered_model_ids: Optional[List[str]] = None

    logger.log("Step 1a: query2modelcard (full embeddings / full sparse for hybrid, INPROC)...")
    logger.log(f"query2modelcard input query: {query!r}")
    q2m_full_path = os.path.join(job_dir, "query2modelcard_full.json")
    legacy_q2m_path = os.path.join(job_dir, "query2modelcard.json")

    q2m_top_k_relaxed = min(200, 5 * top_k)
    q2m_top_k_here = q2m_top_k_relaxed
    logger.log(
        f"query2modelcard FULL: top_k={q2m_top_k_here}, resources={model_resources_full} → {os.path.basename(q2m_full_path)}"
    )
    t0 = time.time()
    from src.search.query2modelcard import search_query2modelcard
    logger.log(
        f"[query2modelcard_full] INPROC: search_query2modelcard("
        f"query={query!r}, top_k={q2m_top_k_here}, retrieval_mode={card2card_retrieval_mode!r}, "
        f"sparse_index_path={SPARSE_INDEX!r})"
    )
    _ = search_query2modelcard(
        query=query,
        top_k=q2m_top_k_here,
        output_json=q2m_full_path,
        retrieval_mode=card2card_retrieval_mode,
        candidate_factor=10,
        emb_npz_path=EMB_NPZ,
        sparse_index_path=SPARSE_INDEX,
        dense_runtime=dense_runtime_full,
    )
    elapsed = time.time() - t0
    logger.log(f"[query2modelcard_full] SAVED: {q2m_full_path}")
    logger.log(f"[query2modelcard_full] ELAPSED: {elapsed:.2f}s")
    logger.log("[query2modelcard_full] EXIT: 0")
    data = _read_json(q2m_full_path)
    if not data or "results" not in data or not data["results"]:
        logger.log("query2modelcard (full) returned no results")
        logger.set_status("error")
        logger.set_results({"error": "No model from query", "model_id": None, "card2card_results": [], "card2tab2card_results": {}, "folder_path": job_dir, "run_log_path": run_log_path})
        return
    try:
        shutil.copyfile(q2m_full_path, legacy_q2m_path)
    except OSError as e:
        logger.log(f"warn: could not copy to query2modelcard.json: {e}")
    results_list = data["results"]
    stored_query = data.get("query", "")

    first = results_list[0]
    raw_top = first if isinstance(first, str) else (first.get("model_id") if isinstance(first, dict) else str(first))
    query_seed_model_id = str(raw_top).strip() if raw_top else ""
    if not query_seed_model_id:
        logger.log("query2modelcard (full) returned empty top-1 model_id")
        logger.set_status("error")
        logger.set_results({"error": "Empty model_id from query", "model_id": None, "card2card_results": [], "card2tab2card_results": {}, "folder_path": job_dir, "run_log_path": run_log_path})
        return
    logger.log(
        f"Query2Card seed (full index, top-1): {query_seed_model_id!r} "
        f"(query in JSON: {stored_query!r})"
    )

    table_search_seed_model_id = query_seed_model_id
    logger.log(
        "Step 1b removed: backend no longer selects table seed with query2modelcard_w_table; "
        "query2tab2card handles seed selection internally."
    )

    model_id = query_seed_model_id  # API / UI: primary seed from full q2m
    table_search_seed_model_id = str(table_search_seed_model_id).strip()
    logger.log(f"Seeds: Query2Card (full)={query_seed_model_id!r}, Query2Tab2Card=(internal to query2tab2card)")

    q2m_ordered_model_ids = _ordered_model_ids_from_q2m_results(results_list)
    # NOTE: This repo snapshot may not include legacy `scripts/check_model_in_index.py`.
    # We let downstream scripts produce empty results if the model id is missing.

    # Table Search: user `table_search_k` is the per-table top-k passed into tab2tab.
    # (card2tab2card does its own merging/dedup/capping downstream.)
    table_search_k_input = max(1, int(table_search_k))
    # IMPORTANT: keep true per-query-table top-k for Tab2Tab (no hidden multiplier).
    k_table = table_search_k_input

    # Card2Card: scaled top_k for left panel lists from query2modelcard ranking.
    card2card_top_k = max(100, int(model_top_k) * 20)

    def _normalize_card2tab2card_payload(payload: Any) -> Dict[str, Any]:
        """
        Normalize historical/variant card2tab2card output formats into a canonical schema:
          {
            "model_ids": [...],
            "query_tables": [...],
            "searched_tables": [...],
            "mappings": {
              "card_to_related_tables": {...},
              "query_table_to_retrieved_tables": {...},
              "retrieved_table_to_related_models": {...}
            },
            "intermediate": {...},
            "pipeline_trace": {...}
          }
        """
        def _canonicalize(d: Dict[str, Any]) -> Dict[str, Any]:
            model_ids = d.get("model_ids", [])
            if not isinstance(model_ids, list):
                model_ids = []
            query_tables = d.get("query_tables", [])
            if not isinstance(query_tables, list):
                query_tables = []
            searched_tables = d.get("searched_tables", [])
            if not isinstance(searched_tables, list):
                searched_tables = []

            mappings = d.get("mappings", {})
            if not isinstance(mappings, dict):
                mappings = {}
            m_card_to_related = mappings.get("card_to_related_tables", {})
            if not isinstance(m_card_to_related, dict):
                m_card_to_related = {}
            m_qt_to_rt = mappings.get("query_table_to_retrieved_tables", {})
            if not isinstance(m_qt_to_rt, dict):
                m_qt_to_rt = {}
            m_rt_to_models = mappings.get("retrieved_table_to_related_models", {})
            if not isinstance(m_rt_to_models, dict):
                m_rt_to_models = {}
            m_mid_to_tables = mappings.get("model_id_to_related_tables", {})
            if not isinstance(m_mid_to_tables, dict):
                m_mid_to_tables = {}
            m_tab2tab_rt_to_models = mappings.get("tab2tab_retrieved_table_to_related_models", {})
            if not isinstance(m_tab2tab_rt_to_models, dict):
                m_tab2tab_rt_to_models = {}

            inter = d.get("intermediate")
            if not isinstance(inter, dict):
                inter = {}
            inter_table_to_models = inter.get("table_to_models")
            if not isinstance(inter_table_to_models, dict):
                inter_table_to_models = m_rt_to_models
            inter_retrieved = inter.get("retrieved_table_filenames")
            if not isinstance(inter_retrieved, list):
                inter_retrieved = list(searched_tables)
            inter_qt_to_rt = inter.get("query_table_to_retrieved_tables")
            if not isinstance(inter_qt_to_rt, dict):
                inter_qt_to_rt = m_qt_to_rt

            inter["table_to_models"] = inter_table_to_models
            inter["retrieved_table_filenames"] = inter_retrieved
            inter["query_table_to_retrieved_tables"] = inter_qt_to_rt
            if not isinstance(inter.get("table_id_to_filename"), dict):
                inter["table_id_to_filename"] = {}

            out = dict(d)
            out["model_ids"] = model_ids
            out["query_tables"] = query_tables
            out["searched_tables"] = searched_tables
            out["mappings"] = {
                "card_to_related_tables": m_card_to_related,
                "query_table_to_retrieved_tables": m_qt_to_rt,
                "retrieved_table_to_related_models": m_rt_to_models if m_rt_to_models else inter_table_to_models,
                "model_id_to_related_tables": m_mid_to_tables,
                "tab2tab_retrieved_table_to_related_models": m_tab2tab_rt_to_models,
            }
            out["intermediate"] = inter
            if not isinstance(out.get("pipeline_trace"), dict):
                out["pipeline_trace"] = {}
            return out

        if payload is None:
            return _canonicalize({"model_ids": [], "intermediate": {}, "mappings": {}})
        # Current schema: dict with model_ids + intermediate
        if isinstance(payload, dict) and "model_ids" in payload:
            return _canonicalize(payload)
        # Variant: just a list of model_ids
        if isinstance(payload, list):
            return _canonicalize({"model_ids": payload, "intermediate": {}, "mappings": {}})
        # Historical batch schema: { queryKey: [...]} or { queryKey: {...} }
        if isinstance(payload, dict):
            for v in payload.values():
                if isinstance(v, dict) and "model_ids" in v:
                    return _canonicalize(v)
                if isinstance(v, list):
                    return _canonicalize({"model_ids": v, "intermediate": {}, "mappings": {}})
        # Fallback
        return _canonicalize({"model_ids": [], "intermediate": {}, "mappings": {}})

    logger.log("Step 2: Card2Card from query2modelcard rank + Card2Tab2Card (parallel where applicable)...")

    def run_card2tab2card(st: str) -> tuple:
        out_path = os.path.join(job_dir, f"card2tab2card_{st}.json")
        # In-process call avoids Python subprocess startup/import overhead.
        logger.log(
            f"[Card2Tab2Card-{st}] INPROC: search_query2tab2card("
            f"query={query!r}, search_type={st!r}, table_top_k={k_table}, "
            f"model_top_k={int(model_top_k)}, q2m_table_candidate_k=9, resources={table_resources})"
        )
        t0 = time.time()
        from src.search.query2tab2card import search_query2tab2card
        payload = search_query2tab2card(
            query=(query or ""),
            search_type=st,
            output_json=out_path,
            table_top_k=k_table,
            table_resources=table_resources,
            apply_query_rerank=True,
            model_top_k=int(model_top_k),
            q2m_table_candidate_k=9,
            dense_runtime=dense_runtime_table,
        )
        elapsed = time.time() - t0
        logger.log(f"[Card2Tab2Card-{st}] SAVED: {out_path}")
        logger.log(f"[Card2Tab2Card-{st}] ELAPSED: {elapsed:.2f}s")
        logger.log(f"[Card2Tab2Card-{st}] EXIT: 0")
        return (st, 0, out_path, payload, "", elapsed)

    # Add by_type if run on sub-lake (use_by_type currently only echoed in results; no extra CLI).
    card2tab2card_types = list(CARD2TAB2CARD_TYPES)
    futures = {}
    # NOTE:
    # query2tab2card now runs in-process. Running multiple search_type workers concurrently
    # can race during SentenceTransformer/PyTorch model initialization on CUDA and throw:
    # "Cannot copy out of meta tensor; no data!".
    # Keep card2tab2card workers serialized for stability.
    with ThreadPoolExecutor(max_workers=1) as ex:
        for st in card2tab2card_types:
            futures[ex.submit(run_card2tab2card, st)] = ("card2tab2card", st)

    card2card_all = {}
    card2tab2card_all = {}
    table_search_empty_reason: Optional[str] = None

    for fut in as_completed(futures):
        kind, name = futures[fut]
        res = fut.result()
        if kind == "card2tab2card":
            st, rc, out_path, payload_raw, err, elapsed = res
            logger.log(f"[Card2Tab2Card-{st}] Done in {elapsed:.2f}s")
            data = payload_raw if isinstance(payload_raw, dict) else _read_json(out_path)
            if data is not None:
                payload = _normalize_card2tab2card_payload(data)
                card2tab2card_all[st] = payload
                mid = payload.get("model_ids", [])
                lst = list(mid) if isinstance(mid, (list, np.ndarray)) else []
                qty = len(payload.get("query_tables", []))
                logger.log(f"[Card2Tab2Card-{st}] Read {len(lst)} model_ids, {qty} query_tables for seed model")
                if rc != 0:
                    logger.log(f"[Card2Tab2Card-{st}] Used JSON despite exit code {rc}")
                if len(lst) == 0 and qty == 0:
                    logger.log(f"[Card2Tab2Card-{st}] Seed model has no tables in relationship_parquet (model_id not in parquet or no csv_basename). Check {RELATIONSHIP_PARQUET} has column modelId and rows for this model.")
                    if table_search_empty_reason is None:
                        table_search_empty_reason = (
                            f"Table-search seed «{table_search_seed_model_id}» has no tables in the dataset: it is not in "
                            f"{RELATIONSHIP_PARQUET} or has no csv_basename. "
                            "Try another query whose top result has linked tables, or check the parquet has column modelId and rows for this model."
                        )
            else:
                if rc != 0:
                    logger.log(f"[Card2Tab2Card-{st}] Error (exit {rc}): {err}")
                # No valid JSON on disk: save empty so frontend can handle it
                card2tab2card_all[st] = {"model_ids": [], "intermediate": {}}
                logger.log(f"[Card2Tab2Card-{st}] No JSON at {out_path}")

    seed_s = str(query_seed_model_id).strip()
    if q2m_ordered_model_ids:
        neighbor_list = [m for m in q2m_ordered_model_ids if m != seed_s][:card2card_top_k]
        for mode in CARD2CARD_MODES:
            card2card_all[mode] = list(neighbor_list)
        logger.log(
            f"[Card2Card] {len(neighbor_list)} neighbor ids from query2modelcard order "
            f"(mirrored to dense/sparse/hybrid)."
        )
    else:
        for mode in CARD2CARD_MODES:
            card2card_all[mode] = []
        logger.log("[Card2Card] Missing query2modelcard ordering list; left panel empty.")

    # --- Cap for UI: min(requested model_top_k, max Card2Tab2Card list length) ---
    right_max_models = 0
    for st, payload in card2tab2card_all.items():
        if not isinstance(payload, dict):
            continue
        mids = payload.get("model_ids", [])
        if isinstance(mids, list):
            right_max_models = max(right_max_models, len(mids))

    effective_model_top_k = min(int(model_top_k), int(right_max_models)) if isinstance(model_top_k, int) else right_max_models
    if effective_model_top_k < 0:
        effective_model_top_k = 0
    logger.log(f"effective_model_top_k: input={model_top_k}, right_max_models={right_max_models} => cap={effective_model_top_k}")

    # Query2Tab2Card now owns query-driven dense rerank and emits mapping-rich payloads.
    # Here we only ensure pipeline_trace exists for legacy payloads.
    def _norm_table_to_models_json(t2m: Any) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        if not isinstance(t2m, dict):
            return out
        for k, v in t2m.items():
            if isinstance(v, list):
                out[str(k)] = [str(m) for m in v if m is not None]
            else:
                out[str(k)] = []
        return out

    for _st, _payload in card2tab2card_all.items():
        if not isinstance(_payload, dict):
            continue
        inter = _payload.get("intermediate")
        if not isinstance(inter, dict):
            inter = {}
            _payload["intermediate"] = inter
        pt = _payload.get("pipeline_trace")
        if not isinstance(pt, dict):
            st_list = _payload.get("searched_tables")
            if not isinstance(st_list, list):
                st_list = list(inter.get("retrieved_table_filenames") or [])
            t2m = inter.get("table_to_models") if isinstance(inter.get("table_to_models"), dict) else {}
            mids = _payload.get("model_ids")
            mids_list = [str(x) for x in (mids if isinstance(mids, list) else []) if x is not None]
            _payload["pipeline_trace"] = {
                "tab2tab": {
                    "searched_tables": [str(x) for x in st_list],
                    "retrieved_table_filenames": [str(x) for x in st_list],
                    "table_to_models": _norm_table_to_models_json(t2m),
                },
                "model_ids_before_dense_rerank": list(mids_list),
                "model_ids_after_dense_rerank": list(mids_list),
                "after_model_cap": {
                    "searched_tables": [str(x) for x in st_list],
                    "table_to_models": _norm_table_to_models_json(t2m),
                },
                "dense_rerank_applied": False,
            }

    # Cap left side (Card2Card)
    for mode in CARD2CARD_MODES:
        val = card2card_all.get(mode)
        if isinstance(val, list) and len(val) > effective_model_top_k:
            card2card_all[mode] = val[:effective_model_top_k]
            logger.log(f"[Card2Card-{mode.upper()}] Truncated to effective_model_top_k={effective_model_top_k}")

    # Primary Card2Card list is strictly the configured retrieval_mode ("dense" by default).
    primary = card2card_all.get("dense")
    if not isinstance(primary, list):
        logger.log("[Card2Card] Primary retrieval mode 'dense' missing or errored; returning empty primary list.")
        primary = []

    elapsed_total = time.time() - start_time
    logger.log(f"Step 3: Done. Total time: {elapsed_total:.2f}s")

    results_data = {
        "job_id": job_id,
        "query": query,
        "model_id": model_id,
        "table_search_seed_model_id": table_search_seed_model_id,
        "top_k": top_k,
        "model_top_k": model_top_k,
        "effective_model_top_k": effective_model_top_k,
        "right_max_models": right_max_models,
        "table_search_k": table_search_k_input,
        "table_resources": table_resources,
        "card2card_retrieval_mode": card2card_retrieval_mode,
        "use_by_type": use_by_type,
        "require_seed_has_tables": require_seed_has_tables,
        "card2card_results": primary,
        "card2card_all_modes": card2card_all,
        "card2tab2card_results": card2tab2card_all,
        "timestamp": datetime.fromtimestamp(start_time).isoformat(),
        "folder_path": job_dir,
        "run_log_path": run_log_path,
        "running_time_seconds": round(elapsed_total, 3),
        "query2modelcard_full_json": os.path.basename(q2m_full_path),
    }
    if table_search_empty_reason:
        results_data["table_search_reason"] = table_search_empty_reason

    _write_json_job(job_id, "search_results.json", results_data)
    logger.log(f"Results saved to {os.path.join(job_dir, 'search_results.json')}")
    logger.log(f"[FINAL] Job directory: {os.path.abspath(job_dir)} | Run log: {os.path.abspath(run_log_path)} | Total: {elapsed_total:.2f}s")
    logger.set_results(results_data)
    logger.set_status("completed")
    logger.log("Pipeline completed.")


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
        saved = _read_json_job(job_id, "search_results.json")
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
    card2card_retrieval_mode = data.get("card2card_retrieval_mode", "dense")
    require_seed_has_tables = bool(data.get("require_seed_has_tables", True))
    use_by_type = bool(data.get("use_by_type", False))

    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"status": "error", "message": "query required"}), 400

    job_id = _generate_job_id()
    jobs[job_id] = JobLogger(job_id)

    thread = threading.Thread(
        target=run_search_pipeline,
        args=(job_id, query, top_k, None, table_search_k, card2card_retrieval_mode, require_seed_has_tables, use_by_type, model_top_k),
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
                mk = _model_search_key(run.get("integration_type"), run.get("card2card_retrieval_mode"))
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
        out["model_search_runs"] = [{"key": _model_search_key(m.get("integration_type"), m.get("card2card_retrieval_mode")), **m}]
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
    resolved = _resolve_table_file_for_preview(path_q)
    if not resolved:
        return jsonify({"status": "error", "message": "Table file not found or access denied"}), 404
    try:
        df = pd.read_csv(resolved, nrows=MAX_TABLE_PREVIEW_JSON_ROWS)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to read CSV: {e}"}), 500
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
        }
    )


def make_table_page_response(path_q: str) -> Response:
    """Build full-page HTML for one CSV (used by /api/table-page on API and UI servers)."""
    path_q = (path_q or "").strip()
    if not path_q:
        return Response("Missing path", status=400, mimetype="text/plain; charset=utf-8")
    resolved = _resolve_table_file_for_preview(path_q)
    if not resolved:
        return Response(
            "<!DOCTYPE html><html><body><p>Table file not found or access denied.</p></body></html>",
            status=404,
            mimetype="text/html; charset=utf-8",
        )
    try:
        df = pd.read_csv(resolved, nrows=MAX_TABLE_PAGE_ROWS)
    except Exception as e:
        msg = html.escape(str(e))
        return Response(
            f"<!DOCTYPE html><html><body><p>Failed to read CSV: {msg}</p></body></html>",
            status=500,
            mimetype="text/html; charset=utf-8",
        )
    total_cols = int(df.shape[1])
    col_note = ""
    if total_cols > MAX_TABLE_PAGE_COLS:
        df = df.iloc[:, :MAX_TABLE_PAGE_COLS]
        col_note = f" Showing first {MAX_TABLE_PAGE_COLS} of {total_cols} columns."
    row_note = ""
    if len(df) >= MAX_TABLE_PAGE_ROWS:
        row_note = f" Showing first {MAX_TABLE_PAGE_ROWS} rows (file may contain more)."
    note = (row_note + col_note).strip()
    title = os.path.basename(resolved)
    body = render_template_string(
        TABLE_PAGE_HTML,
        title=title,
        path_display=resolved,
        note=note,
        table_html=_dataframe_to_html_table(df),
    )
    return Response(body, mimetype="text/html; charset=utf-8")


@app.route("/api/table-page", methods=["GET"])
def table_page():
    """Full-page HTML table view (new tab from retrieval results)."""
    return make_table_page_response(request.args.get("path") or "")


@app.route("/api/table-search-preview", methods=["POST"])
def table_search_preview():
    """Prepare query2tab2card relationship preview without running integration."""
    data = request.get_json() or {}
    job_id, err = _require_job_id(data)
    if err is not None:
        return err
    search_type = str(data.get("search_type") or "single_column").strip()
    tables_source = str(data.get("tables_source") or "intermediate").strip()
    tr_override = data.get("table_resources")
    tr_list = tr_override if isinstance(tr_override, list) else None

    sr = _read_json_job(job_id, "search_results.json")
    if not isinstance(sr, dict):
        return _api_error(f"search_results.json not found for job {job_id}", 404)
    try:
        from src.integration.pipeline_preview import prepare_query2tab2card_preview

        preview, prep_err = prepare_query2tab2card_preview(
            search_results=sr,
            search_type=search_type,
            tables_source=tables_source,
            table_resources=tr_list,
        )
        if prep_err:
            return jsonify({"status": "no_result", "message": prep_err, "search_type": search_type, "tables_source": tables_source})
        return jsonify(
            {
                "status": "success",
                "search_type": search_type,
                "tables_source": tables_source,
                "query_tables": preview.get("query_tables", []),
                "table_paths": preview.get("table_paths", []),
                "model_to_table_paths": preview.get("model_to_table_paths_ts", {}),
                "models_with_tables": preview.get("models_with_tables_list", []),
                "pipeline_trace": preview.get("pipeline_trace", {}),
                "tab2tab_trace_rows": preview.get("tab2tab_trace_rows", []),
                "after_model_cap_trace_rows": preview.get("after_model_cap_trace_rows", []),
                "retrieved_table_model_rows": preview.get("after_model_cap_trace_rows", []),
                "stats": {
                    "total_unique_tables": len(preview.get("table_paths", [])),
                    "models_with_tables": len(preview.get("models_with_tables_list", [])),
                    "tables_source": tables_source,
                },
            }
        )
    except Exception as e:
        return _api_error(f"preview failed: {e}", 500)


@app.route("/api/model-search-preview", methods=["POST"])
def model_search_preview():
    """Prepare query2card relationship preview without running integration."""
    data = request.get_json() or {}
    job_id, err = _require_job_id(data)
    if err is not None:
        return err
    card2card_mode = str(data.get("card2card_retrieval_mode") or "dense").strip().lower()
    tr_override = data.get("table_resources")
    tr_list = tr_override if isinstance(tr_override, list) else None

    sr = _read_json_job(job_id, "search_results.json")
    if not isinstance(sr, dict):
        return _api_error(f"search_results.json not found for job {job_id}", 404)

    modes = sr.get("card2card_all_modes") if isinstance(sr.get("card2card_all_modes"), dict) else {}
    mids_raw = modes.get(card2card_mode, [])
    model_ids = [str(x).strip() for x in (mids_raw if isinstance(mids_raw, list) else []) if str(x).strip()]
    if not model_ids:
        return jsonify(
            {
                "status": "no_result",
                "message": f"No model ids for retrieval mode {card2card_mode!r}",
                "card2card_retrieval_mode": card2card_mode,
            }
        )

    from src.integration.pipeline_preview import _resolve_table_paths_for_model_ids, _table_resources_for_integration

    resources = tr_list if isinstance(tr_list, list) and tr_list else _table_resources_for_integration(sr)
    table_paths, model_to_table_paths = _resolve_table_paths_for_model_ids(model_ids, resources=resources)
    return jsonify(
        {
            "status": "success",
            "card2card_retrieval_mode": card2card_mode,
            "models_with_tables": model_ids,
            "table_paths": table_paths,
            "model_to_table_paths": model_to_table_paths,
            "stats": {
                "models_with_tables": len(model_ids),
                "total_unique_tables": len(table_paths),
            },
        }
    )

@app.route("/api/saved-searches", methods=["GET"])
def list_saved_searches():
    candidates = []
    for name in os.listdir(JOBS_DIR):
        path = os.path.join(JOBS_DIR, name)
        json_path = os.path.join(path, "search_results.json")
        if os.path.isdir(path) and os.path.isfile(json_path):
            mtime = os.path.getmtime(json_path)
            candidates.append((name, path, mtime))
    candidates.sort(key=lambda x: x[2], reverse=True)
    searches = []
    for name, path, _ in candidates[:50]:
        json_path = os.path.join(path, "search_results.json")
        entry = {"folder_name": name, "path": path, "query": None, "model_id": None, "timestamp_str": "", "top_k": None, "model_top_k": None, "use_by_type": False, "require_seed_has_tables": False, "card2card_retrieval_mode": None, "table_search_k": None}
        with open(json_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        entry["query"] = saved.get("query") or ""
        entry["model_id"] = saved.get("model_id") or ""
        ts = saved.get("timestamp")
        if ts:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            entry["timestamp_str"] = dt.strftime("%Y-%m-%d %H:%M")
        entry["top_k"] = saved.get("top_k")
        entry["model_top_k"] = saved.get("model_top_k")
        entry["use_by_type"] = bool(saved.get("use_by_type", False))
        entry["require_seed_has_tables"] = bool(saved.get("require_seed_has_tables", False))
        entry["card2card_retrieval_mode"] = saved.get("card2card_retrieval_mode")
        entry["table_search_k"] = saved.get("table_search_k")
        searches.append(entry)
    return jsonify({"status": "success", "searches": searches})


# Stubs for integrate / evaluate / qa (use legacy backend for full support)
@app.route("/api/integrate", methods=["POST"])
def integrate():
    """Integrate tables from Card2Tab2Card search results"""
    data = request.get_json()
    job_id, err = _require_job_id(data)
    if err is not None:
        return err
    job_dir, results_file, err = _require_results_file(job_id)
    if err is not None:
        return err
    os.makedirs(job_dir, exist_ok=True)
    # Default table-search mode should prefer unionable.
    search_type = data.get("search_type", "unionable")
    # Default integration method should match the UI default (ALITE).
    integration_type = data.get("integration_type", "alite")
    k = int(data.get("k", 10))
    max_models = int(data.get("max_models", 10))
    tables_source = data.get("tables_source", "intermediate")
    tr_override = data.get("table_resources")
    tr_list = tr_override if isinstance(tr_override, list) else None
    from src.integration.table_integration import integrate_tables_from_card2tab2card
    result = integrate_tables_from_card2tab2card(
        search_results_json=results_file,
        search_type=search_type,
        integration_type=integration_type,
        k=k,
        tables_source=tables_source,
        table_resources=tr_list,
    )
    run_key = _table_search_key(integration_type, search_type, tables_source)

    if not result.get("success", False):
        save_payload = {"status": "no_result", "integration_type": integration_type, "search_type": search_type, "tables_source": tables_source, "k": k, "max_models": max_models, "error": result.get("error", "Integration failed"), "message": result.get("error", "Integration failed")}
        json_path = os.path.join(job_dir, f"integration_table_search_{run_key}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(save_payload, f, ensure_ascii=False, indent=0)
        return jsonify({"status": "no_result", "message": save_payload["message"], **save_payload})

    # Convert DataFrame to dict for JSON response (NaN -> null for valid JSON).
    # The integration layer already applies the deterministic column order; we just persist it.
    integrated_df = result.get("integrated_table")
    saved_path = None
    if integrated_df is not None:
        raw_data = integrated_df.values.tolist()
        result["integrated_table"] = {"columns": list(integrated_df.columns), "data": _sanitize_for_json(raw_data)}
        csv_name = f"integrated_table_search_{run_key}.csv"
        save_path = os.path.join(job_dir, csv_name)
        integrated_df.to_csv(save_path, index=False, encoding="utf-8")
        saved_path = _integration_saved_path_for_api(job_id, csv_name)
    if saved_path:
        result["saved_path"] = saved_path
    # Ensure models_with_tables is always present for Table Search (model IDs used in this integration; may differ from full retrieval list)
    if "models_with_tables" not in result:
        result["models_with_tables"] = []
    save_payload = {"status": "success", "integration_type": integration_type, "search_type": search_type, "tables_source": tables_source, "k": k, "max_models": max_models, **result}
    json_path = os.path.join(job_dir, f"integration_table_search_{run_key}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, ensure_ascii=False, indent=0)
    with open(os.path.join(job_dir, "integration_table_search.json"), "w", encoding="utf-8") as f:
        json.dump(save_payload, f, ensure_ascii=False, indent=0)
    return jsonify({"status": "success", **result})



@app.route("/api/integrate-model-search", methods=["POST"])
def integrate_model_search():
    """Integrate tables from Card2Card (model search) results"""
    data = request.get_json()
    job_id, err = _require_job_id(data)
    if err is not None:
        return err
    job_dir, results_file, err = _require_results_file(job_id)
    if err is not None:
        return err
    os.makedirs(job_dir, exist_ok=True)
    integration_type = data.get("integration_type", "alite")
    k = int(data.get("k", 10))
    max_models = int(data.get("max_models", 10))
    card2card_retrieval_mode = data.get("card2card_retrieval_mode") or None
    tr_override = data.get("table_resources")
    tr_list = tr_override if isinstance(tr_override, list) else None
    from src.integration.table_integration import integrate_tables_from_card2card
    
    result = integrate_tables_from_card2card(
        search_results_json=results_file,
        integration_type=integration_type,
        k=k,
        max_models=max_models,
        card2card_retrieval_mode=card2card_retrieval_mode,
        table_resources=tr_list,
    )
    run_key = _model_search_key(integration_type, card2card_retrieval_mode or "dense")

    if not result.get("success", False):
        save_payload = {"status": "no_result", "integration_type": integration_type, "card2card_retrieval_mode": card2card_retrieval_mode or "dense", "k": k, "max_models": max_models, "error": result.get("error", "Integration failed"), "message": result.get("error", "Integration failed")}
        json_path = os.path.join(job_dir, f"integration_model_search_{run_key}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(save_payload, f, ensure_ascii=False, indent=0)
        return jsonify({"status": "no_result", "message": save_payload["message"], **save_payload})

    # Convert DataFrame to dict for JSON response (NaN -> null for valid JSON).
    # The integration layer already applies the deterministic column order; we just persist it.
    integrated_df = result.get("integrated_table")
    saved_path = None
    if integrated_df is not None:
        raw_data = integrated_df.values.tolist()
        result["integrated_table"] = {"columns": list(integrated_df.columns), "data": _sanitize_for_json(raw_data)}
        csv_name = f"integrated_model_search_{run_key}.csv"
        save_path = os.path.join(job_dir, csv_name)
        integrated_df.to_csv(save_path, index=False, encoding="utf-8")
        saved_path = _integration_saved_path_for_api(job_id, csv_name)
    if saved_path:
        result["saved_path"] = saved_path
    save_payload = {"status": "success", "integration_type": integration_type, "card2card_retrieval_mode": card2card_retrieval_mode or "dense", "k": k, "max_models": max_models, **result}
    json_path = os.path.join(job_dir, f"integration_model_search_{run_key}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, ensure_ascii=False, indent=0)
    with open(os.path.join(job_dir, "integration_model_search.json"), "w", encoding="utf-8") as f:
        json.dump(save_payload, f, ensure_ascii=False, indent=0)
    return jsonify({"status": "success", **result})

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
    data = request.get_json() or {}
    job_id, err = _require_job_id(data)
    if err is not None:
        return err
    job_dir, err = _require_job_dir(job_id)
    if err is not None:
        return err
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

    # Load query from search results
    query = "model search query"
    sr = _read_json_job(job_id, "search_results.json")
    if sr:
        query = sr.get("query") or query

    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
        from evaluation.llm import evaluate_diversity_with_llm
        result = evaluate_diversity_with_llm(query=query, table1=table1_df, table2=table2_df, table1_source="Table Search Integration", table2_source="Model Search Integration")
    except ValueError as ve:
        return jsonify({"status": "error", "message": f"Evaluation failed: {str(ve)}"}), 500

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
    data = request.get_json() or {}
    job_id, err = _require_job_id(data)
    if err is not None:
        return err
    job_dir, err = _require_job_dir(job_id)
    if err is not None:
        return err
    use_table_search = bool(data.get("use_table_search", True))

    # Load the appropriate integrated table
    if use_table_search:
        table_df = _load_integrated_table_from_json(job_dir, "integration_table_search.json")
        table_source = "Table Search Integration"
        qa_mode = "card2tab2card"
    else:
        table_df = _load_integrated_table_from_json(job_dir, "integration_model_search.json")
        table_source = "Model Search Integration"
        qa_mode = "card2card"

    if table_df is None:
        table_df = pd.DataFrame()

    # Load query and search results from job
    results_file = os.path.join(job_dir, "search_results.json")
    query = "model search query"
    search_results_data = None
    model_ids_to_rank = None

    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            sr = json.load(f)
        query = sr.get("query") or query
        search_results_data = sr

        # Extract model_ids for ranking with strict schema:
        # - If we used Table Search, do not infer model_ids_to_rank from other sources (leave as None).
        # - Else, use only the configured Card2Card retrieval_mode list when available.
        if not use_table_search:
            modes = sr.get("card2card_all_modes")
            rmode = sr.get("card2card_retrieval_mode")
            if isinstance(modes, dict) and isinstance(rmode, str):
                mids = modes.get(rmode)
                if isinstance(mids, list) and mids:
                    model_ids_to_rank = list(mids)[:50]

    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
        from evaluation.llm_qa import answer_question_with_llm

        result = answer_question_with_llm(query=query, table=table_df, table_source=table_source, qa_mode=qa_mode, model_ids_to_rank=model_ids_to_rank, search_results_data=search_results_data)
    except ValueError as ve:
        return jsonify({"status": "error", "message": f"QA failed: {str(ve)}"}), 500

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
    data = request.get_json() or {}
    job_id, err = _require_job_id(data)
    if err is not None:
        return err
    key = data.get("key")
    if not key or (isinstance(key, str) and not key.strip()):
        return _api_error("job_id and key required", 400)
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    safe_key = re.sub(r"[^a-z0-9_]", "_", (key or "").lower().strip()) or "run"
    path = os.path.join(job_dir, f"integration_run_{safe_key}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=0)
        return jsonify({"status": "success"})
    except Exception as e:
        return _api_error(str(e), 500)


@app.route("/api/save-evaluation", methods=["POST"])
def save_evaluation():
    """Save evaluation result to job_dir for load-previous restore. Creates job_dir if missing."""
    data = request.get_json() or {}
    job_id, err = _require_job_id(data)
    if err is not None:
        return err
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    path = os.path.join(job_dir, "evaluation_results.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=0)
        return jsonify({"status": "success"})
    except Exception as e:
        return _api_error(str(e), 500)


@app.route("/api/save-qa", methods=["POST"])
def save_qa():
    """Save QA result to job_dir for load-previous restore. Creates job_dir if missing."""
    data = request.get_json() or {}
    job_id, err = _require_job_id(data)
    if err is not None:
        return err
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    path = os.path.join(job_dir, "qa_results.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=0)
        return jsonify({"status": "success"})
    except Exception as e:
        return _api_error(str(e), 500)


if __name__ == "__main__":
    import argparse as _argparse
    parser = _argparse.ArgumentParser(description="ModelSearch Demo Backend")
    parser.add_argument("--port", type=int, default=None, help="Port (default: env PORT or 5002)")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip loading NPZ + encoder at startup (first job pays cold start).",
    )
    args, _ = parser.parse_known_args()

    print("Backend (in-process) starting...", flush=True)
    if USE_BY_TYPE:
        print(f"  USE_BY_TYPE=1: card2tab2card by_type enabled", flush=True)
    if not args.no_warmup and os.environ.get("BACKEND_SKIP_WARMUP", "").strip().lower() not in ("1", "true", "yes"):
        try:
            from src.search.query2modelcard import warmup_dense_runtimes_for_backend

            warmup_dense_runtimes_for_backend(log=lambda m: print(m, flush=True))
        except Exception as e:
            print(f"[warmup] failed (server still starts; first job may be slow): {e}", flush=True)
    port = args.port if args.port is not None else int(os.environ.get("PORT", "5002"))
    app.run(host="0.0.0.0", port=port, debug=False)
