"""
Backend API for ModelSearch Demo (CLI-based)

Runs search via subprocess commands from docs/build_index.md.
All job outputs (search results, integration, evaluation, QA) go under data/jobs/<job_id>.
Minimal imports for fast startup.
"""

import os, sys, json, random, string, threading, subprocess, time, math, re, shutil, shlex, html
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, List, Optional, Any, Tuple
from flask import Flask, request, jsonify, Response, stream_with_context, render_template_string, send_from_directory
from flask_cors import CORS
from datetime import datetime
#from src.config import REPO_ROOT, JOBS_DIR, CARD2TAB2CARD_TIMEOUT, USE_BY_TYPE, CARD2CARD_MODES, CARD2TAB2CARD_TYPES, CARD2TAB2CARD_OUTPUT_JSON, VALID_MODEL_IDS_TXT, CLASSIFICATION_JSON, TABLE_RESOURCE_ALLOWLIST, RELATIONSHIP_PARQUET, PRESET_QUERIES_PATH
from src.config import *
from src.utils import model_id_has_resolvable_local_tables, resolve_table_path
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


def _integration_run_key(integration_type: str, search_type: str, card2card_mode: str) -> str:
    """Slug for combined integration run (legacy): e.g. union_single_column_dense."""
    parts = [integration_type or "union", search_type or "single_column", card2card_mode or "dense"]
    return "_".join(re.sub(r"[^a-z0-9_]", "_", (p or "").lower().strip()) for p in parts)


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


def _allowed_csv_roots() -> List[str]:
    roots: List[str] = []
    for d in list(TABLE_BASE_DIRS) + [JOBS_DIR, MODELTABLES_DATA]:
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


def _resolve_table_file_for_preview(raw_path: str) -> Optional[str]:
    """Resolve a CSV path from UI (absolute, basename, or repo-relative) to a readable file under allowed dirs."""
    if not raw_path or not isinstance(raw_path, str):
        return None
    s = raw_path.strip()
    if not s or ".." in s.split(os.sep):
        return None
    roots = _allowed_csv_roots()
    if not roots:
        return None
    if os.path.isabs(s) and os.path.isfile(s):
        cand = os.path.realpath(s)
        if _path_is_under_roots(cand, roots):
            return cand
    r = resolve_table_path(s)
    if r and os.path.isfile(r):
        cand = os.path.realpath(r)
        if _path_is_under_roots(cand, roots):
            return cand
    rel = os.path.normpath(os.path.join(REPO_ROOT, s.lstrip("/\\")))
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

    def log_cmd(self, step: str, cmd: List[str], out_path: Optional[str] = None, elapsed: Optional[float] = None, rc: Optional[int] = None):
        """Log command execution details (CMD, OUT, ELAPSED) for pipeline run log."""
        # shlex.join quotes argv pieces with spaces (subprocess still uses list; this is display-only).
        try:
            cmd_display = shlex.join(str(x) for x in cmd)
        except Exception:
            cmd_display = " ".join(str(x) for x in cmd)
        self.log(f"[{step}] CMD: {cmd_display}")
        if out_path:
            self.log(f"[{step}] SAVED: {out_path}")
        if elapsed is not None:
            self.log(f"[{step}] ELAPSED: {elapsed:.2f}s")
        if rc is not None:
            self.log(f"[{step}] EXIT: {rc}")

    def get_logs(self) -> List[Dict]:
        with self.lock:
            return self.logs.copy()

    def set_status(self, status: str):
        with self.lock:
            self.status = status

    def set_results(self, results: Dict):
        with self.lock:
            self.results = results


def _run_cmd(cmd: List[str], cwd: str, env: Optional[Dict] = None, timeout: Optional[int] = 300) -> tuple:
    """Run command; return (returncode, stdout, stderr). No try/except - caller checks returncode."""
    env = env or os.environ.copy()
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    return (r.returncode, r.stdout or "", r.stderr or "")


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

@lru_cache(maxsize=16)
def _cached_valid_model_ids_from_parquet(frozen_table_resources: Tuple[str, ...]) -> frozenset:
    """
    Model IDs with ≥1 table path in RELATIONSHIP_PARQUET for the given resource columns
    (same definition as card2tab2card --resources / load_modelid_to_csvlist).
    Cached per resource tuple for the backend process lifetime.
    """
    from src.utils import _load_modelid_to_csv_expand

    res_list = list(frozen_table_resources)
    df = _load_modelid_to_csv_expand(res_list if res_list else None)
    if df.empty or "modelId" not in df.columns:
        return frozenset()
    s = df["modelId"].dropna().astype(str).str.strip()
    return frozenset(x for x in s if x)


def _valid_model_ids_for_table_resources(table_resources: List[str]) -> set:
    """Align require_seed_has_tables with card2tab2card table source (not a broader txt built for all columns)."""
    tr = [str(x).strip().lower() for x in table_resources if str(x).strip()]
    key = tuple(sorted(tr))
    if not key:
        key = tuple(sorted(TABLE_RESOURCE_ALLOWLIST))
    return set(_cached_valid_model_ids_from_parquet(key))

def run_search_pipeline(
    job_id: str,
    query: Optional[str] = None,
    top_k: int = 20,
    model_id: Optional[str] = None,
    table_search_k: Optional[int] = None,
    tab2tab_mode: str = "search",
    tab2tab_json: Optional[str] = None,
    card2card_retrieval_mode: str = "dense",
    require_seed_has_tables: bool = False,
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
        _run_pipeline_body(logger, job_id, job_dir, start_time, query, top_k, model_id, table_search_k, tab2tab_mode, tab2tab_json, card2card_retrieval_mode, require_seed_has_tables, use_by_type, model_top_k=model_top_k, table_resource_allowlist=table_resource_allowlist)
    except Exception as e:
        logger.log(f"Pipeline crashed: {e}")
        _set_pipeline_error(f"Pipeline error: {e}", model_id)
        import traceback
        logger.log(traceback.format_exc())

def _run_pipeline_body(logger: "JobLogger", job_id: str, job_dir: str, start_time: float, query: Optional[str], top_k: int, model_id: Optional[str], table_search_k: Optional[int], tab2tab_mode: str, tab2tab_json: Optional[str], card2card_retrieval_mode: str, require_seed_has_tables: bool = False, use_by_type: bool = False, *, model_top_k: int = 5, table_resource_allowlist: Optional[List[str]] = ['hugging']):
    # Write all logs to job_dir/pipeline_run.log for debugging
    run_log_path = os.path.join(job_dir, "pipeline_run.log")
    logger.set_log_file(run_log_path)
    logger.log("=" * 60)
    logger.log(f"Job directory (all outputs saved here): {os.path.abspath(job_dir)}")
    logger.log(f"Run log file: {os.path.abspath(run_log_path)}")
    logger.log("=" * 60)
    logger.log("Starting search pipeline (CLI)...")
    logger.log("Mode: Query → ModelCard → Search" if query else "Mode: Model ID → Search")
    if query:
        logger.log(f"Query: {query}")
    else:
        logger.log(f"Model ID: {model_id}")
    logger.log(
        f"Run settings: top_k={top_k}, model_top_k={model_top_k}, per_table_search_k={table_search_k or 1}, "
        f"card2card={card2card_retrieval_mode}, "
        f"table type classification (by_type)={'ON' if use_by_type else 'OFF'}, "
        f"require_seed_has_tables={require_seed_has_tables}"
    )
    logger.set_status("running")

    # Normalize resource selection once and reuse in all subprocess commands.
    raw_resources = list(table_resource_allowlist or TABLE_RESOURCE_ALLOWLIST)
    model_resources = [r for r in raw_resources if r in ("hugging", "github", "arxiv")]
    table_resources = [r for r in raw_resources if r in ("hugging", "github", "arxiv")]
    if not model_resources:
        model_resources = list(TABLE_RESOURCE_ALLOWLIST)
    if not table_resources:
        table_resources = list(TABLE_RESOURCE_ALLOWLIST)
    logger.log(f"Resources: model={model_resources}, table={table_resources}")

    def _with_resources(cmd: List[str], resources: List[str]) -> List[str]:
        return cmd + ["--resources", *resources]

    def _build_q2m_cmd(*, q: str, q_top_k: int, out_path: str) -> List[str]:
        base = [
            sys.executable, "-m", "src.search.query2modelcard",
            "--query", q,
            "--top_k", str(q_top_k),
            "--retrieval_mode", card2card_retrieval_mode,
            "--output_json", out_path,
        ]
        return _with_resources(base, model_resources)

    def _build_card2card_cmd(*, mode: str, top_k_value: int, out_path: str) -> List[str]:
        base = [
            sys.executable, "-m", "src.search.card2card", "search",
            "--model_ids_file", card2card_model_ids_file,
            "--top_k", str(top_k_value),
            "--retrieval_mode", mode,
            "--output_json", out_path,
        ]
        return _with_resources(base, model_resources)

    def _build_card2tab2card_cmd(*, st: str, table_k_value: int, out_path: str) -> List[str]:
        base = [
            sys.executable, "-m", "src.search.card2tab2card",
            "--model_id", model_id,
            "--search_type", st,
            "--table_top_k", str(table_k_value),
            "--model_top_k", str(model_top_k),
            "--output_json", out_path,
        ]
        return _with_resources(base, table_resources)

    # Resolve model_id (query mode: from FAISS retrieval only; no default/hardcoded id)
    seed_no_tables_skip_table_search = False  # when True: use first model for Card2Card only, skip Card2Tab2Card and set reason
    valid_model_ids = None
    if require_seed_has_tables:
        valid_model_ids = _valid_model_ids_for_table_resources(table_resources)
        logger.log(
            f"Valid model IDs (parquet lists ≥1 path for table_resources={table_resources}) = {len(valid_model_ids)} "
            "(seed picker also requires local CSV under TABLE_BASE_DIRS)"
        )
    if query:
        logger.log("Step 1: Extracting model card from query (query2modelcard)...")
        logger.log(f"query2modelcard input query: {query!r}")
        q2m_out = os.path.join(job_dir, "query2modelcard.json")

        if require_seed_has_tables:
            assert valid_model_ids is not None
            logger.log(
                f"Narrow down: parquet allowlist size = {len(valid_model_ids)}; "
                "picking first hit that also has resolvable local table files"
            )
            _res_key = tuple(sorted(table_resources))

            q2m_top_k = min(200, 5 * top_k)
            logger.log(f"query2modelcard: top_k={q2m_top_k} (single call, then filter by parquet + local CSV)")
            cmd = _build_q2m_cmd(q=query, q_top_k=q2m_top_k, out_path=q2m_out)
            t0 = time.time()
            rc, out, err = _run_cmd(cmd, REPO_ROOT)
            elapsed = time.time() - t0
            logger.log_cmd("query2modelcard", cmd, q2m_out, elapsed, rc)
            if rc != 0:
                logger.log(f"query2modelcard failed (exit {rc}): {err or out}")
                logger.set_status("error")
                logger.set_results({"error": f"query2modelcard failed: {err or out}", "model_id": None, "card2card_results": [], "card2tab2card_results": {}, "folder_path": job_dir, "run_log_path": run_log_path})
                return
            data = _read_json(q2m_out)
            if not data or "results" not in data or not data["results"]:
                logger.log("query2modelcard returned no results")
                logger.set_status("error")
                logger.set_results({"error": "No model from query", "model_id": None, "card2card_results": [], "card2tab2card_results": {}, "folder_path": job_dir, "run_log_path": run_log_path})
                return
            # get mid from query2modelcard results
            results_list = data["results"]
            chosen = None
            for i, r in enumerate(results_list):
                mid = r if isinstance(r, str) else (r.get("model_id") if isinstance(r, dict) else str(r))
                if not mid:
                    continue
                mid_s = str(mid).strip()
                if mid_s not in valid_model_ids:
                    continue
                if not model_id_has_resolvable_local_tables(mid_s, _res_key):
                    logger.log(f"Narrow down: skip #{i+1} {mid_s!r} (in parquet allowlist but no local CSV under TABLE_BASE_DIRS)")
                    continue
                chosen = mid_s
                logger.log(f"Narrow down: first result with local tables is #{i+1}: {chosen}")
                break
            # chosen set: first hit in retrieval list that is in parquet allowlist AND has ≥1 local CSV under TABLE_BASE_DIRS
            if chosen is not None:
                model_id = chosen
                stored_query = data.get("query", "")
                logger.log(f"Extracted model (with tables): {model_id} (from query2modelcard JSON, query in file: {stored_query!r})")
            else:
                # Retrieval top-1 may still be a fine Card2Card seed; it just failed our table-search seed rules for every hit in this top_k window.
                first = results_list[0]
                raw_mid = first if isinstance(first, str) else (first.get("model_id") if isinstance(first, dict) else str(first))
                model_id = str(raw_mid).strip() if raw_mid else ""
                stored_query = data.get("query", "")
                seed_no_tables_skip_table_search = True
                logger.log(
                    f"Table Search Seed Model: none of top-{len(results_list)} pass parquet allowlist + resolvable local CSV. "
                    "Using retrieval top-1 for Card2Card only; Card2Tab2Card skipped."
                )
                logger.log(
                    f"Extracted model (fallback top-1): {model_id!r} (query2modelcard JSON, query in file: {stored_query!r})"
                )
        else: # no condition on whether this modelcard must contain table (for table search)
            cmd = _build_q2m_cmd(q=query, q_top_k=1, out_path=q2m_out)
            t0 = time.time()
            rc, out, err = _run_cmd(cmd, REPO_ROOT)
            elapsed = time.time() - t0
            logger.log_cmd("query2modelcard", cmd, q2m_out, elapsed, rc)
            if rc != 0:
                logger.log(f"query2modelcard failed (exit {rc}): {err or out}")
                logger.set_status("error")
                logger.set_results({"error": f"query2modelcard failed: {err or out}", "model_id": None, "card2card_results": [], "card2tab2card_results": {}, "folder_path": job_dir, "run_log_path": run_log_path})
                return
            data = _read_json(q2m_out)
            if not data or "results" not in data or not data["results"]:
                logger.log("query2modelcard returned no results")
                logger.set_status("error")
                logger.set_results({"error": "No model from query", "model_id": None, "card2card_results": [], "card2tab2card_results": {}})
                return
            results_list = data["results"]
            stored_query = data.get("query", "")

        if not require_seed_has_tables:
            first = results_list[0]
            raw_mid = first if isinstance(first, str) else (first.get("model_id") if isinstance(first, dict) else str(first))
            model_id = str(raw_mid).strip() if raw_mid else ""
            if not model_id:
                logger.log("query2modelcard returned empty model_id")
                logger.set_status("error")
                logger.set_results({"error": "Empty model_id from query", "model_id": None, "card2card_results": [], "card2tab2card_results": {}, "folder_path": job_dir, "run_log_path": run_log_path})
                return
            logger.log(f"Extracted model: {model_id} (from query2modelcard JSON, query in file: {stored_query!r})")
        if not model_id:
            logger.log("model_id is required in modelid mode")
            logger.set_status("error")
            logger.set_results({"error": "Model ID is required (empty input)", "model_id": None, "card2card_results": [], "card2tab2card_results": {}, "folder_path": job_dir, "run_log_path": run_log_path})
            return
        logger.log(f"Using model_id: {model_id}")
        # NOTE: This repo snapshot may not include legacy `scripts/check_model_in_index.py`.
        # We let downstream scripts produce empty results if the model id is missing.

    # Table Search: user `table_search_k` is the per-table top-k passed into tab2tab.
    # (card2tab2card does its own merging/dedup/capping downstream.)
    table_search_k_input = max(1, int(table_search_k))
    k_table = table_search_k_input * 20
    #k_table = table_search_k_input

    # Card2Card: scaled `--top_k` for first (and only) dense/sparse/hybrid subprocess.
    card2card_top_k = max(100, int(model_top_k) * 20)

    # card2card (src/search/card2card.py) expects `--model_ids_file` + `--output_json`.
    card2card_model_ids_file = os.path.join(job_dir, "card2card_model_ids.txt")
    with open(card2card_model_ids_file, "w", encoding="utf-8") as f:
        f.write(str(model_id).strip() + "\n")

    def _normalize_card2tab2card_payload(payload: Any) -> Dict[str, Any]:
        """
        Normalize historical/variant card2tab2card output formats into:
          { "model_ids": [...], "intermediate": {...}, "query_tables": [...]? }
        The frontend/integration expects at least `model_ids` + `intermediate`.
        """
        if payload is None:
            return {"model_ids": [], "intermediate": {}}
        # Current schema: dict with model_ids + intermediate
        if isinstance(payload, dict) and "model_ids" in payload:
            if "intermediate" not in payload or not isinstance(payload.get("intermediate"), dict):
                payload["intermediate"] = payload.get("intermediate") or {}
            return payload
        # Variant: just a list of model_ids
        if isinstance(payload, list):
            return {"model_ids": payload, "intermediate": {}}
        # Historical batch schema: { queryKey: [...]} or { queryKey: {...} }
        if isinstance(payload, dict):
            for v in payload.values():
                if isinstance(v, dict) and "model_ids" in v:
                    if "intermediate" not in v or not isinstance(v.get("intermediate"), dict):
                        v["intermediate"] = v.get("intermediate") or {}
                    return v
                if isinstance(v, list):
                    return {"model_ids": v, "intermediate": {}}
        # Fallback
        return {"model_ids": [], "intermediate": {}}

    # Step 2: Run Card2Card (dense, sparse, hybrid) and Card2Tab2Card in parallel via CLI
    logger.log("Step 2: Running Card2Card + Card2Tab2Card (parallel CLI)...")

    def run_card2card(mode: str) -> tuple:
        out_path = os.path.join(job_dir, f"card2card_{mode}.json")
        cmd = _build_card2card_cmd(mode=mode, top_k_value=card2card_top_k, out_path=out_path)
        t0 = time.time()
        rc, out, err = _run_cmd(cmd, REPO_ROOT)
        elapsed = time.time() - t0
        logger.log_cmd(f"Card2Card-{mode.upper()}", cmd, out_path, elapsed, rc)
        return (mode, rc, out_path, out, err, elapsed)

    def run_card2tab2card(st: str) -> tuple:
        out_path = os.path.join(job_dir, f"card2tab2card_{st}.json")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # Ensure print() goes to stdout immediately for pipeline log piping
        cmd = _build_card2tab2card_cmd(st=st, table_k_value=k_table, out_path=out_path)
        t0 = time.time()
        rc, out, err = _run_cmd(cmd, REPO_ROOT, env=env, timeout=CARD2TAB2CARD_TIMEOUT)
        elapsed = time.time() - t0
        logger.log_cmd(f"Card2Tab2Card-{st}", cmd, out_path, elapsed, rc)
        return (st, rc, out_path, out, err, elapsed)

    def run_card2tab2card_by_type() -> tuple:
        """Run card2tab2card by_type request.

        Note: simplified `src/search/card2tab2card.py` does not support `--mode by_type`.
        So we run the simplified pipeline and store results into the per-job output JSON.
        """
        st = "by_type"
        out_path = os.path.join(job_dir, f"card2tab2card_{st}.json")
        # Use simplified card2tab2card (no legacy depre).
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        cmd = _build_card2tab2card_cmd(st="keyword", table_k_value=k_table, out_path=out_path)
        t0 = time.time()
        rc, out, err = _run_cmd(cmd, REPO_ROOT, env=env, timeout=CARD2TAB2CARD_TIMEOUT)
        elapsed = time.time() - t0
        logger.log_cmd("Card2Tab2Card-by_type", cmd, out_path, elapsed, rc)
        return (st, rc, out_path, out, err, elapsed)

    # Add by_type if run on sub-lake
    #if use_by_type:
    #    card2tab2card_types = list(CARD2TAB2CARD_TYPES) + ["by_type"]
    #else:
    card2tab2card_types = list(CARD2TAB2CARD_TYPES)
    # EXECUTE CARD2CARD AND CARD2TAB2CARD IN PARALLEL
    futures = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for m in CARD2CARD_MODES:
            futures[ex.submit(run_card2card, m)] = ("card2card", m)
        if not seed_no_tables_skip_table_search:
            for st in card2tab2card_types:
                #if st == "by_type":
                #    futures[ex.submit(run_card2tab2card_by_type)] = ("card2tab2card", "by_type")
                #else:
                futures[ex.submit(run_card2tab2card, st)] = ("card2tab2card", st)

    card2card_all = {}
    card2tab2card_all = {}
    table_search_empty_reason: Optional[str] = None

    for fut in as_completed(futures):
        kind, name = futures[fut]
        res = fut.result()
        if kind == "card2card":
            mode, rc, out_path, out, err, elapsed = res
            logger.log(f"[Card2Card-{mode.upper()}] Done in {elapsed:.2f}s")
            if rc != 0:
                logger.log(f"[Card2Card-{mode.upper()}] Error (exit {rc}): {err or out}")
                card2card_all[mode] = {"error": err or out}
            else:
                data = _read_json(out_path)
                if data is not None:
                    # `src/search/card2card.py` CLI saves a dict like:
                    #   { "<seed_model_id>": ["neighbor1", "neighbor2", ...] }
                    # When we pass a file with a single model_id, we return that one neighbor list.
                    if isinstance(data, dict):
                        if "neighbors" in data:
                            card2card_all[mode] = data.get("neighbors", [])
                        elif "results" in data:
                            card2card_all[mode] = data.get("results", [])
                        else:
                            vals = list(data.values())
                            card2card_all[mode] = vals[0] if vals and isinstance(vals[0], list) else []
                    else:
                        card2card_all[mode] = []
                else:
                    card2card_all[mode] = []
        else:
            st, rc, out_path, out, err, elapsed = res
            logger.log(f"[Card2Tab2Card-{st}] Done in {elapsed:.2f}s")
            # Pipe subprocess stdout into pipeline log (card2tab2card prints go to stdout, not logger)
            combined = (out or "") + (err or "")
            if combined.strip():
                for line in combined.strip().split("\n"):
                    if line.strip():
                        logger.log(f"  | {line.strip()}")
            # Always try to read JSON from disk: CLI may write results then exit(1) e.g. due to get_device() in another process env
            data = _read_json(out_path)
            if data is not None:
                # CLI should write card2tab2card payload; normalize historical variants.
                payload = _normalize_card2tab2card_payload(data)
                card2tab2card_all[st] = payload
                mid = payload.get("model_ids", [])
                lst = list(mid) if isinstance(mid, (list, np.ndarray)) else []
                qty = len(payload.get("query_tables", []))
                logger.log(f"[Card2Tab2Card-{st}] Read {len(lst)} model_ids, {qty} query_tables for seed model")
                if rc != 0:
                    logger.log(f"[Card2Tab2Card-{st}] Used on-disk JSON despite exit code {rc} (CLI may have failed after writing)")
                if len(lst) == 0 and qty == 0:
                    logger.log(f"[Card2Tab2Card-{st}] Seed model has no tables in relationship_parquet (model_id not in parquet or no csv_basename). Check {RELATIONSHIP_PARQUET} has column modelId and rows for this model.")
                    if table_search_empty_reason is None:
                        table_search_empty_reason = (
                            f"Seed model «{model_id}» has no tables in the dataset: it is not in "
                            f"{RELATIONSHIP_PARQUET} or has no csv_basename. "
                            "Try another query whose top result has linked tables, or check the parquet has column modelId and rows for this model."
                        )
                    cli_out = (err or out or "").strip()
                    if cli_out and ("No tables" in cli_out or "Warning" in cli_out):
                        for line in cli_out.split("\n")[-3:]:
                            if line.strip():
                                logger.log(f"[Card2Tab2Card-{st}] CLI: {line.strip()}")
            else:
                if rc != 0:
                    logger.log(f"[Card2Tab2Card-{st}] Error (exit {rc}): {err or out}")
                # No valid JSON on disk: save empty so frontend can handle it
                card2tab2card_all[st] = {"model_ids": [], "intermediate": {}}
                if data is None:
                    logger.log(f"[Card2Tab2Card-{st}] No JSON at {out_path}")

    # When we skipped Table Search (require_seed_has_tables but none had tables), record reason only
    if seed_no_tables_skip_table_search and table_search_empty_reason is None:
        table_search_empty_reason = (
            "❌ None of the top-20 models from the query have tables in the dataset. "
            "❌ Table Search skipped. Select «Use top-1 result» for Table Search Seed Model to run with top-1 anyway."
        )

    # --- Enforce valid_model_ids on Card2Card lists (neighbors can be outside table-resource universe) ---
    # Card2Tab2Card: no payload filter here — basenames from the DuckDB index for a given resource are
    # disjoint from other resources' names in our pipeline, so reverse parquet lookup yields only models
    # that actually carry that file under the same resource family. If that assumption breaks, filter again
    # or pass resources into load_csvs_to_modelids.
    if require_seed_has_tables and valid_model_ids:
        # 1) Filter Card2Card model lists
        for mode in CARD2CARD_MODES:
            val = card2card_all.get(mode)
            if isinstance(val, list):
                before = len(val)
                card2card_all[mode] = [mid for mid in val if mid in valid_model_ids] # only return modelcard that has tables for fair comparison
                logger.log(f"[Card2Card-{mode.upper()}] Filtered valid models: {before} -> {len(card2card_all[mode])}")

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

    # --- Dense reranker: seed model's Card2Card dense neighbor order (in-memory reorder + truncate) ---
    # When Card2Tab2Card returns MORE model_ids than effective_model_top_k, we reorder candidates by
    # this order (ties / missing ids go last), then truncate. No subprocess.
    seed_dense_neighbor_order = card2card_all.get("dense") if isinstance(card2card_all.get("dense"), list) else []
    dense_reranker_rank = {mid: i for i, mid in enumerate(seed_dense_neighbor_order)}

    def _sync_payload_intermediate_to_model_ids(payload: Dict[str, Any], keep_model_ids: List[str]) -> None:
        """Ensure intermediate tables also correspond to kept model_ids."""
        intermediate = payload.get("intermediate")
        if not isinstance(intermediate, dict):
            return
        keep_set = set(str(m) for m in keep_model_ids if m is not None)
        if not keep_set:
            # Keep payload structure but drop tables.
            if isinstance(payload.get("searched_tables"), list):
                payload["searched_tables"] = []
            if isinstance(intermediate.get("retrieved_table_filenames"), list):
                intermediate["retrieved_table_filenames"] = []
            if isinstance(intermediate.get("table_to_models"), dict):
                intermediate["table_to_models"] = {}
            if isinstance(intermediate.get("table_id_to_filename"), dict):
                intermediate["table_id_to_filename"] = {}
            if isinstance(intermediate.get("retrieved_table_ids"), list):
                intermediate["retrieved_table_ids"] = []
            return

        table_to_models = intermediate.get("table_to_models")
        if not isinstance(table_to_models, dict):
            return

        new_table_to_models: Dict[str, List[str]] = {}
        for tname, model_list in table_to_models.items():
            if not isinstance(model_list, list):
                continue
            kept = [m for m in model_list if str(m) in keep_set]
            if kept:
                new_table_to_models[tname] = kept

        kept_tables = set(new_table_to_models.keys())
        intermediate["table_to_models"] = new_table_to_models

        if isinstance(payload.get("searched_tables"), list):
            payload["searched_tables"] = [t for t in payload["searched_tables"] if t in kept_tables]

        if isinstance(intermediate.get("retrieved_table_filenames"), list):
            intermediate["retrieved_table_filenames"] = [
                t for t in intermediate["retrieved_table_filenames"] if t in kept_tables
            ]

        tid_map = intermediate.get("table_id_to_filename")
        if isinstance(tid_map, dict):
            intermediate["table_id_to_filename"] = {str(tid): fname for tid, fname in tid_map.items() if fname in kept_tables}

        if isinstance(intermediate.get("retrieved_table_ids"), list) and isinstance(intermediate.get("table_id_to_filename"), dict):
            fname_map = intermediate.get("table_id_to_filename", {})
            kept_tids: List[int] = []
            for tid in intermediate["retrieved_table_ids"]:
                fname = fname_map.get(str(tid))
                if fname in kept_tables:
                    try:
                        kept_tids.append(int(tid))
                    except Exception:
                        pass
            intermediate["retrieved_table_ids"] = kept_tids

        payload["intermediate"] = intermediate

    # --- Dense reranker + cap: Card2Tab2Card side (in-memory only) ---
    for st, payload in card2tab2card_all.items():
        if not isinstance(payload, dict):
            continue
        mids = payload.get("model_ids", [])
        if not isinstance(mids, list):
            continue
        if len(mids) > effective_model_top_k and effective_model_top_k > 0:
            tail = len(dense_reranker_rank) + 100000
            dense_reranker_ordered = sorted(mids, key=lambda mid: dense_reranker_rank.get(mid, tail))
            payload["model_ids"] = dense_reranker_ordered[:effective_model_top_k]
            logger.log(
                f"[Card2Tab2Card-{st}] DENSE_RERANKER: {len(mids)} candidates -> top {effective_model_top_k} "
                f"(order from seed Card2Card dense neighbors; in-memory only)"
            )
        elif len(mids) > effective_model_top_k and effective_model_top_k == 0:
            payload["model_ids"] = []
        elif len(mids) > effective_model_top_k:
            # No seed dense list to rank by (e.g. empty); plain truncate.
            payload["model_ids"] = mids[:effective_model_top_k]
            logger.log(f"[Card2Tab2Card-{st}] truncated to {effective_model_top_k} (no dense_reranker order)")

        # Keep intermediate tables consistent with the truncated model_ids.
        final_mids = payload.get("model_ids", [])
        if isinstance(final_mids, list):
            _sync_payload_intermediate_to_model_ids(payload, final_mids)

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

    # Default "new" search: run the full pipeline.
    mode = data.get("mode", "query")
    top_k = int(data.get("top_k", 20))
    table_search_k = data.get("table_search_k")
    model_top_k = int(data.get("model_top_k", 5))
    tab2tab_mode = data.get("tab2tab_mode", "search")
    tab2tab_json = data.get("tab2tab_json")
    card2card_retrieval_mode = data.get("card2card_retrieval_mode", "dense")
    require_seed_has_tables = bool(data.get("require_seed_has_tables", False))
    use_by_type = bool(data.get("use_by_type", False))

    if mode == "query":
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"status": "error", "message": "query required"}), 400
        model_id = None
    elif mode == "modelid":
        model_id = (data.get("model_id") or "").strip()
        if not model_id:
            return jsonify({"status": "error", "message": "model_id required"}), 400
        query = None
    else:
        return jsonify({"status": "error", "message": "mode must be query or modelid"}), 400

    if tab2tab_mode == "load" and not tab2tab_json:
        return jsonify({"status": "error", "message": "tab2tab_json required when tab2tab_mode=load"}), 400

    job_id = _generate_job_id()
    jobs[job_id] = JobLogger(job_id)

    thread = threading.Thread(
        target=run_search_pipeline,
        args=(job_id, query, top_k, model_id, table_search_k, tab2tab_mode, tab2tab_json, card2card_retrieval_mode, require_seed_has_tables, use_by_type, model_top_k),
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


@app.route("/api/table-page", methods=["GET"])
def table_page():
    """Full-page HTML table view (new tab from retrieval results)."""
    path_q = (request.args.get("path") or "").strip()
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
    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
        from integration.table_integration import integrate_tables_from_card2tab2card
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
            saved_path = os.path.join("data", "jobs", job_id, csv_name)
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
    except Exception as e:
        import traceback
        traceback.print_exc()
        return _api_error(str(e), 500)

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
    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
        from integration.table_integration import integrate_tables_from_card2card
        
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
            saved_path = os.path.join("data", "jobs", job_id, csv_name)
        if saved_path:
            result["saved_path"] = saved_path
        save_payload = {"status": "success", "integration_type": integration_type, "card2card_retrieval_mode": card2card_retrieval_mode or "dense", "k": k, "max_models": max_models, **result}
        json_path = os.path.join(job_dir, f"integration_model_search_{run_key}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(save_payload, f, ensure_ascii=False, indent=0)
        with open(os.path.join(job_dir, "integration_model_search.json"), "w", encoding="utf-8") as f:
            json.dump(save_payload, f, ensure_ascii=False, indent=0)
        return jsonify({"status": "success", **result})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return _api_error(str(e), 500)


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
    if isinstance(qa_answer, dict):
        pass
    else:
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
    args, _ = parser.parse_known_args()

    print("Backend (CLI-based) starting...", flush=True)
    if USE_BY_TYPE:
        print(f"  USE_BY_TYPE=1: card2tab2card by_type enabled", flush=True)
    port = args.port if args.port is not None else int(os.environ.get("PORT", "5002"))
    app.run(host="0.0.0.0", port=port, debug=False)
