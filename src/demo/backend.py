"""
Backend API for ModelSearch Demo (CLI-based)

Runs search via subprocess commands from docs/build_index.md.
All job outputs (search results, integration, evaluation, QA) go under data/jobs/<job_id>.
Minimal imports for fast startup.
"""

import os, sys, json, random, string, threading, subprocess, time, math, re, shutil, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify, Response, stream_with_context, render_template_string, send_from_directory
from flask_cors import CORS
from datetime import datetime
#from src.config import REPO_ROOT, JOBS_DIR, CARD2TAB2CARD_TIMEOUT, USE_BY_TYPE, CARD2CARD_MODES, CARD2TAB2CARD_TYPES, CARD2TAB2CARD_OUTPUT_JSON, VALID_MODEL_IDS_TXT, CLASSIFICATION_JSON, TABLE_RESOURCE_ALLOWLIST, RELATIONSHIP_PARQUET, PRESET_QUERIES_PATH
from src.config import *
from src.utils import filter_results_by_classify_results

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
        self.log(f"[{step}] CMD: {' '.join(str(x) for x in cmd)}")
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

# Valid model IDs (have tables): built by scripts/build_valid_model_ids_txt.py (Part 1); inference only loads
def _get_valid_modelids_with_tables() -> set:
    """Cached set of model_id that have tables. Fast O(1) lookup; loads from txt once per backend lifetime."""
    with open(VALID_MODEL_IDS_TXT, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

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
        _run_pipeline_body(
            logger,
            job_id,
            job_dir,
            start_time,
            query,
            top_k,
            model_id,
            table_search_k,
            tab2tab_mode,
            tab2tab_json,
            card2card_retrieval_mode,
            require_seed_has_tables,
            use_by_type,
            model_top_k=model_top_k,
            table_resource_allowlist=table_resource_allowlist,
        )
    except Exception as e:
        logger.log(f"Pipeline crashed: {e}")
        _set_pipeline_error(f"Pipeline error: {e}", model_id)
        import traceback
        logger.log(traceback.format_exc())

def _run_pipeline_body(
    logger: "JobLogger",
    job_id: str,
    job_dir: str,
    start_time: float,
    query: Optional[str],
    top_k: int,
    model_id: Optional[str],
    table_search_k: Optional[int],
    tab2tab_mode: str,
    tab2tab_json: Optional[str],
    card2card_retrieval_mode: str,
    require_seed_has_tables: bool = False,
    use_by_type: bool = False,
    *,
    model_top_k: int = 5,
    table_resource_allowlist: Optional[List[str]] = None,
):
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

    # Resolve model_id (query mode: from FAISS retrieval only; no default/hardcoded id)
    seed_no_tables_skip_table_search = False  # when True: use first model for Card2Card only, skip Card2Tab2Card and set reason
    valid_model_ids = None
    if require_seed_has_tables:
        valid_model_ids = _get_valid_modelids_with_tables()
        logger.log(f"Valid model IDs (have tables) = {len(valid_model_ids)} (from {VALID_MODEL_IDS_TXT})")
    if query:
        logger.log("Step 1: Extracting model card from query (query2modelcard)...")
        logger.log(f"query2modelcard input query: {query!r}")
        q2m_out = os.path.join(job_dir, "query2modelcard.json")
        q2m_module = "src.search.query2modelcard"

        if require_seed_has_tables:
            assert valid_model_ids is not None
            logger.log(f"Narrow down: valid model IDs (have tables) = {len(valid_model_ids)} (from {VALID_MODEL_IDS_TXT})")
            # Two-phase: try 2× top_k first; if no valid model, run again with 5× (cap 200)
            phase1_k = min(100, 2 * top_k)
            phase2_k = min(200, 5 * top_k)
            for phase, q2m_top_k in enumerate([phase1_k, phase2_k], 1):
                if phase == 2 and phase2_k <= phase1_k:
                    break
                logger.log(f"query2modelcard phase {phase}: top_k={q2m_top_k}")
                cmd = [
                    sys.executable,
                    "-m",
                    q2m_module,
                    "--query",
                    query,
                    "--top_k",
                    str(q2m_top_k),
                    "--retrieval_mode",
                    card2card_retrieval_mode,
                    "--output_json",
                    q2m_out,
                ]
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
                results_list = data["results"]
                chosen = None
                for i, r in enumerate(results_list):
                    mid = r if isinstance(r, str) else (r.get("model_id") if isinstance(r, dict) else str(r))
                    if not mid:
                        continue
                    if str(mid).strip() in valid_model_ids:
                        chosen = str(mid).strip()
                        logger.log(f"Narrow down: first result in valid set is #{i+1}: {chosen} (phase {phase})")
                        break
                if chosen is not None:
                    model_id = chosen
                    stored_query = data.get("query", "")
                    logger.log(f"Extracted model (with tables): {model_id} (from query2modelcard JSON, query in file: {stored_query!r})")
                    break
            else:
                first = data["results"][0]
                model_id = first if isinstance(first, str) else (first.get("model_id") if isinstance(first, dict) else str(first))
                seed_no_tables_skip_table_search = True
                logger.log(f"Table Search Seed Model: none of top-{len(results_list)} in valid set. Using top-1 for Card2Card only; Table Search skipped.")
                logger.log(f"Extracted model (no tables): {model_id} (from query2modelcard JSON)")
        else:
            cmd = [
                sys.executable,
                "-m",
                q2m_module,
                "--query",
                query,
                "--top_k",
                "1",
                "--retrieval_mode",
                card2card_retrieval_mode,
                "--output_json",
                q2m_out,
            ]
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
            model_id = first if isinstance(first, str) else (first.get("model_id") if isinstance(first, dict) else str(first))
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

    # Table Search:
    # - user-provided `table_search_k` is the "final" per-table budget (default 1)
    # - during search we expand it by 20x to generate more candidates
    table_search_k_input = int(table_search_k) if table_search_k is not None else 1
    table_search_k_input = max(1, table_search_k_input)
    k_table = table_search_k_input * 20  # Per-table k passed into card2tab2card/tab2tab

    # Model Search:
    # - user-provided `model_top_k` is the "final" model budget
    # - during Card2Card retrieval we expand it by 100x to generate more candidates
    card2card_top_k = max(100, int(model_top_k) * 100)

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
        cmd = [
            sys.executable,
            "-m",
            "src.search.card2card",
            "search",
            "--model_ids_file",
            card2card_model_ids_file,
            "--top_k",
            str(card2card_top_k),
            "--retrieval_mode",
            mode,
            "--output_json",
            out_path,
        ]
        t0 = time.time()
        rc, out, err = _run_cmd(cmd, REPO_ROOT)
        elapsed = time.time() - t0
        logger.log_cmd(f"Card2Card-{mode.upper()}", cmd, out_path, elapsed, rc)
        return (mode, rc, out_path, out, err, elapsed)

    def run_card2tab2card(st: str) -> tuple:
        out_path = os.path.join(job_dir, f"card2tab2card_{st}.json")
        c2t2c_module = "src.search.card2tab2card"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # Ensure print() goes to stdout immediately for pipeline log piping
        cmd = [
            sys.executable,
            "-m",
            c2t2c_module,
            "--model_id",
            model_id,
            "--search_type",
            st,
            "--k",
            str(k_table),
            "--output_json",
            out_path,
        ]
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
        c2t2c_module = "src.search.card2tab2card"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [
            sys.executable,
            "-m",
            c2t2c_module,
            "--model_id",
            model_id,
            "--search_type",
            "keyword",
            "--k",
            str(k_table),
            "--output_json",
            out_path,
        ]
        t0 = time.time()
        rc, out, err = _run_cmd(cmd, REPO_ROOT, env=env, timeout=CARD2TAB2CARD_TIMEOUT)
        elapsed = time.time() - t0
        logger.log_cmd("Card2Tab2Card-by_type", cmd, out_path, elapsed, rc)
        return (st, rc, out_path, out, err, elapsed)

    card2card_modes = CARD2CARD_MODES
    card2tab2card_types = CARD2TAB2CARD_TYPES
    # Add by_type if requested
    if use_by_type:
        card2tab2card_types = list(card2tab2card_types) + ["by_type"]

    futures = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for m in card2card_modes:
            futures[ex.submit(run_card2card, m)] = ("card2card", m)
        if not seed_no_tables_skip_table_search:
            for st in card2tab2card_types:
                if st == "by_type":
                    futures[ex.submit(run_card2tab2card_by_type)] = ("card2tab2card", "by_type")
                else:
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
            "None of the top-20 models from the query have tables in the dataset. "
            "Table Search skipped. Select «Use top-1 result» for Table Search Seed Model to run with top-1 anyway."
        )

    # Optional: filter Card2Tab2Card(tab2tab) retrieved tables by resource origin
    # (based on ModelTables filename naming rules via classify_resource()).
    table_resource_allowlist = list(TABLE_RESOURCE_ALLOWLIST) if TABLE_RESOURCE_ALLOWLIST else None
    if table_resource_allowlist:
        logger.log(f"Table resource filtering (tab2tab outputs): allowlist={table_resource_allowlist}")

        def _filter_payload_tables_by_resource(payload: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(payload, dict):
                return payload

            intermediate = payload.get("intermediate", {})
            if not isinstance(intermediate, dict):
                return payload

            searched_tables = payload.get("searched_tables")
            if not isinstance(searched_tables, list):
                searched_tables = intermediate.get("retrieved_table_filenames")

            if not isinstance(searched_tables, list):
                return payload

            kept_tables = filter_results_by_classify_results(
                searched_tables,
                table_resource_allowlist,
            )
            # After filtering, we also cap to the user's final `table_search_k_input`.
            # card2tab2card internally multiplies `k` by number of query_tables, so we
            # mirror that behavior here to get a consistent "final" budget.
            query_tables = payload.get("query_tables", [])
            qn = len(query_tables) if isinstance(query_tables, list) and query_tables else 1
            table_limit = int(table_search_k_input) * max(1, qn)
            if table_limit > 0:
                kept_tables = kept_tables[:table_limit]
            kept_set = set(kept_tables)

            # Keep payload.searched_tables in sync
            if isinstance(payload.get("searched_tables"), list):
                payload["searched_tables"] = kept_tables

            # Filter intermediate.table_to_models and filename/ids lists
            if isinstance(intermediate.get("table_to_models"), dict):
                intermediate["table_to_models"] = {
                    fname: mids
                    for fname, mids in intermediate["table_to_models"].items()
                    if fname in kept_set
                }

            if isinstance(intermediate.get("retrieved_table_filenames"), list):
                intermediate["retrieved_table_filenames"] = [
                    t for t in intermediate["retrieved_table_filenames"] if t in kept_set
                ]

            tid_map = intermediate.get("table_id_to_filename")
            if isinstance(tid_map, dict):
                intermediate["table_id_to_filename"] = {
                    str(tid): fname
                    for tid, fname in tid_map.items()
                    if fname in kept_set
                }

            if isinstance(intermediate.get("retrieved_table_ids"), list) and isinstance(intermediate.get("table_id_to_filename"), dict):
                kept_tids: List[int] = []
                for tid in intermediate["retrieved_table_ids"]:
                    fname = intermediate["table_id_to_filename"].get(str(tid))
                    if fname in kept_set:
                        kept_tids.append(int(tid))
                intermediate["retrieved_table_ids"] = kept_tids

            # Update payload.model_ids based on surviving tables
            mids_set: set[str] = set()
            table_to_models = intermediate.get("table_to_models")
            if isinstance(table_to_models, dict):
                for mids in table_to_models.values():
                    if isinstance(mids, list):
                        for mid in mids:
                            if mid is not None:
                                mids_set.add(str(mid))

            if isinstance(payload.get("model_ids"), list):
                payload["model_ids"] = [m for m in payload["model_ids"] if str(m) in mids_set]

            payload["intermediate"] = intermediate
            return payload

        for st, payload in list(card2tab2card_all.items()):
            if not isinstance(payload, dict):
                continue
            before_models = payload.get("model_ids", [])
            before_tables = payload.get("intermediate", {}).get("retrieved_table_filenames", [])
            card2tab2card_all[st] = _filter_payload_tables_by_resource(payload)

            after_models = card2tab2card_all[st].get("model_ids", []) if isinstance(card2tab2card_all[st], dict) else []
            after_tables = card2tab2card_all[st].get("intermediate", {}).get("retrieved_table_filenames", []) if isinstance(card2tab2card_all[st], dict) else []
            logger.log(
                f"  [{st}] tab2tab tables: {len(before_tables) if isinstance(before_tables, list) else '?'} -> {len(after_tables) if isinstance(after_tables, list) else '?'}; "
                f"models: {len(before_models) if isinstance(before_models, list) else '?'} -> {len(after_models) if isinstance(after_models, list) else '?'}"
            )

    # --- Enforce valid_model_ids for left/right model lists (top-5 stability) ---
    # If we filtered seed correctly, but downstream Card2Card / Card2Tab2Card returns non-valid models,
    # the UI/table integration can look inconsistent or fail to load the intended tables.
    if require_seed_has_tables and valid_model_ids:
        # 1) Filter Card2Card model lists
        for mode in card2card_modes:
            val = card2card_all.get(mode)
            if isinstance(val, list):
                before = len(val)
                card2card_all[mode] = [mid for mid in val if mid in valid_model_ids]
                logger.log(f"[Card2Card-{mode.upper()}] Filtered valid models: {before} -> {len(card2card_all[mode])}")

        # 2) Filter Card2Tab2Card payloads + keep only tables that map to valid models
        #    If after filtering a payload has < model_top_k models, we re-run tab search with k=100
        #    to pull more candidate tables (so after mapping we can still get top-5 valid models).
        rerun_table_k = 100
        did_rerun_table_k = False

        def _filter_card2tab2card_payload(st: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(payload, dict):
                return payload
            mids = payload.get("model_ids", [])
            if not isinstance(mids, list):
                mids = []
            mids = [mid for mid in mids if mid in valid_model_ids]
            payload["model_ids"] = mids

            intermediate = payload.get("intermediate", {})
            if isinstance(intermediate, dict):
                table_to_models = intermediate.get("table_to_models", {})
                if isinstance(table_to_models, dict):
                    new_table_to_models: Dict[str, List[str]] = {}
                    for tname, model_list in table_to_models.items():
                        if not isinstance(model_list, list):
                            continue
                        kept = [m for m in model_list if m in valid_model_ids]
                        if kept:
                            new_table_to_models[tname] = kept
                    intermediate["table_to_models"] = new_table_to_models

                    # Update retrieved filenames/ids/table mapping to match surviving tables
                    kept_tables = set(new_table_to_models.keys())
                    for key in ("retrieved_table_filenames", "searched_tables"):
                        if key == "searched_tables":
                            if "searched_tables" in payload and isinstance(payload["searched_tables"], list):
                                payload["searched_tables"] = [x for x in payload["searched_tables"] if x in kept_tables]
                        else:
                            if key in intermediate and isinstance(intermediate[key], list):
                                intermediate[key] = [x for x in intermediate[key] if x in kept_tables]

                    if "table_id_to_filename" in intermediate and isinstance(intermediate["table_id_to_filename"], dict):
                        tid_map = intermediate["table_id_to_filename"]
                        intermediate["table_id_to_filename"] = {
                            str(tid): fname
                            for tid, fname in tid_map.items()
                            if fname in kept_tables
                        }

                    if "retrieved_table_ids" in intermediate and isinstance(intermediate["retrieved_table_ids"], list):
                        # Rebuild by filenames we kept; preserve ordering from original retrieved_table_ids.
                        fname_map = intermediate.get("table_id_to_filename", {})
                        intermediate["retrieved_table_ids"] = [
                            tid for tid in intermediate["retrieved_table_ids"]
                            if str(tid) in fname_map
                        ]
            payload["intermediate"] = intermediate
            return payload

        for st, payload in list(card2tab2card_all.items()):
            if not isinstance(payload, dict):
                continue
            card2tab2card_all[st] = _filter_card2tab2card_payload(st, payload)

        def _rerun_card2tab2card(st: str, k_override: int) -> Dict[str, Any]:
            out_path = os.path.join(job_dir, f"card2tab2card_{st}.json")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            cmd = [
                sys.executable,
                "-m",
                "src.search.card2tab2card",
                "--model_id",
                model_id,
                "--search_type",
                st,
                "--k",
                str(k_override),
                "--output_json",
                out_path,
            ]
            t0 = time.time()
            rc, out, err = _run_cmd(cmd, REPO_ROOT, env=env, timeout=CARD2TAB2CARD_TIMEOUT)
            elapsed = time.time() - t0
            logger.log_cmd(f"Card2Tab2Card-{st}-rerun_k", cmd, out_path, elapsed, rc)

            data = _read_json(out_path)
            if data is None:
                logger.log(f"[Card2Tab2Card-{st}-rerun_k] No JSON at {out_path}; returning empty payload")
                return {"model_ids": [], "intermediate": {}}
            payload = _normalize_card2tab2card_payload(data)
            # Re-apply resource filter + table budget after rerun, so subsequent
            # model selection stays consistent with the postprocess stage.
            if table_resource_allowlist:
                payload = _filter_payload_tables_by_resource(payload)
            payload = _filter_card2tab2card_payload(st, payload)
            return payload

        # Re-run tab search only for the short payloads.
        short_sts: List[str] = []
        for st, payload in card2tab2card_all.items():
            if isinstance(payload, dict):
                mids = payload.get("model_ids", [])
                if isinstance(mids, list) and len(mids) < model_top_k:
                    short_sts.append(st)

        if short_sts and k_table < rerun_table_k:
            did_rerun_table_k = True
            logger.log(f"Rescaling per-table k from {k_table} -> {rerun_table_k} for st(s)={short_sts} to recover top-{model_top_k} valid models")
            k_table = rerun_table_k
            for st in short_sts:
                card2tab2card_all[st] = _rerun_card2tab2card(st, k_override=k_table)

        if did_rerun_table_k:
            logger.log("Card2Tab2Card rerun complete; model_ids and intermediate have been re-filtered.")

        # 3) If dense list is still short after filtering, re-run Card2Card dense for more candidates.
        #    (Card2Card already runs with top_k=100; we only expand when filtering caused shortage.)
        dense_list = card2card_all.get("dense")
        if isinstance(dense_list, list) and len(dense_list) < model_top_k and card2card_top_k < 500:
            rerun_top_k = 500
            logger.log(f"Card2Card dense filtered too short ({len(dense_list)}<{model_top_k}); re-running Card2Card dense with top_k={rerun_top_k}")
            out_path = os.path.join(job_dir, "card2card_dense.json")
            cmd = [
                sys.executable,
                "-m",
                "src.search.card2card",
                "search",
                "--model_ids_file",
                card2card_model_ids_file,
                "--top_k",
                str(rerun_top_k),
                "--retrieval_mode",
                "dense",
                "--output_json",
                out_path,
            ]
            t0 = time.time()
            rc, out, err = _run_cmd(cmd, REPO_ROOT, env=None, timeout=CARD2TAB2CARD_TIMEOUT)
            elapsed = time.time() - t0
            logger.log_cmd("Card2Card-DENSE-rerun", cmd, out_path, elapsed, rc)

            data = _read_json(out_path)
            if data is not None:
                if isinstance(data, dict):
                    if "neighbors" in data:
                        card2card_all["dense"] = data.get("neighbors", [])
                    elif "results" in data:
                        card2card_all["dense"] = data.get("results", [])
                    else:
                        vals = list(data.values())
                        card2card_all["dense"] = vals[0] if vals and isinstance(vals[0], list) else []
                else:
                    card2card_all["dense"] = []
                card2card_all["dense"] = [mid for mid in card2card_all["dense"] if mid in valid_model_ids]

    # Effective model-topk:
    # - right-side table search (Card2Tab2Card) may return fewer models after filtering
    # - we expand search upstream, then finally return min(input_model_top_k, right_max_models)
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

    logger.log(f"Adaptive effective_model_top_k: input={model_top_k}, right_max_models={right_max_models} => effective={effective_model_top_k}")

    dense_order = card2card_all.get("dense") if isinstance(card2card_all.get("dense"), list) else []
    rank_map = {mid: i for i, mid in enumerate(dense_order)}

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

    # Rerank/cap right side (Card2Tab2Card)
    for st, payload in card2tab2card_all.items():
        if not isinstance(payload, dict):
            continue
        mids = payload.get("model_ids", [])
        if not isinstance(mids, list):
            continue
        if len(mids) > effective_model_top_k and effective_model_top_k > 0:
            big = len(rank_map) + 100000
            reranked = sorted(mids, key=lambda mid: rank_map.get(mid, big))
            payload["model_ids"] = reranked[:effective_model_top_k]
            logger.log(f"[Card2Tab2Card-{st}] Adaptive rerank: {len(mids)} -> {effective_model_top_k} (dense order)")
        elif len(mids) > effective_model_top_k and effective_model_top_k == 0:
            payload["model_ids"] = []
        elif len(mids) > effective_model_top_k:
            payload["model_ids"] = mids[:effective_model_top_k]

        # Keep intermediate tables consistent with the truncated model_ids.
        final_mids = payload.get("model_ids", [])
        if isinstance(final_mids, list):
            _sync_payload_intermediate_to_model_ids(payload, final_mids)

    # Cap left side (Card2Card)
    for mode in card2card_modes:
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

    results_data = {"job_id": job_id, "query": query, "model_id": model_id, "top_k": top_k, "model_top_k": model_top_k, "effective_model_top_k": effective_model_top_k, "right_max_models": right_max_models, "table_search_k": table_search_k_input, "card2card_retrieval_mode": card2card_retrieval_mode, "use_by_type": use_by_type, "require_seed_has_tables": require_seed_has_tables, "card2card_results": primary, "card2card_all_modes": card2card_all, "card2tab2card_results": card2tab2card_all, "timestamp": datetime.fromtimestamp(start_time).isoformat(), "folder_path": job_dir, "run_log_path": run_log_path, "running_time_seconds": round(elapsed_total, 3)}
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
        args=(
            job_id,
            query,
            top_k,
            model_id,
            table_search_k,
            tab2tab_mode,
            tab2tab_json,
            card2card_retrieval_mode,
            require_seed_has_tables,
            use_by_type,
            model_top_k,
        ),
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
    return jsonify({"status": "error", "message": "Use legacy backend for table preview"}), 501


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
    search_type = data.get("search_type", "keyword")
    integration_type = data.get("integration_type", "union")
    k = int(data.get("k", 10))
    max_models = int(data.get("max_models", 10))
    tables_source = data.get("tables_source", "intermediate")
    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
        from integration.table_integration import integrate_tables_from_card2tab2card
        result = integrate_tables_from_card2tab2card(search_results_json=results_file, search_type=search_type, integration_type=integration_type, k=k, tables_source=tables_source)
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
    integration_type = data.get("integration_type", "union")
    k = int(data.get("k", 10))
    max_models = int(data.get("max_models", 10))
    card2card_retrieval_mode = data.get("card2card_retrieval_mode") or None
    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
        from integration.table_integration import integrate_tables_from_card2card
        
        result = integrate_tables_from_card2card(search_results_json=results_file, integration_type=integration_type, k=k, max_models=max_models, card2card_retrieval_mode=card2card_retrieval_mode)
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
