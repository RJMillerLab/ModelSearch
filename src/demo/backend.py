"""
Backend API for ModelSearch Demo (CLI-based)

Runs search via subprocess commands from docs/build_index.md.
All job outputs (search results, integration, evaluation, QA) go under data/jobs/<job_id>.
Minimal imports for fast startup.
"""

import os
import sys
import json
import uuid
import threading
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from datetime import datetime
import math
import re
import numpy as np
import pandas as pd


def _sanitize_for_json(obj: Any) -> Any:
    """Replace float('nan') with None so JSON serialization produces null."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    return obj


# Repo root (backend lives in src/demo/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Paths from build_index.md (relative to repo root)
DEFAULT_EMB_NPZ = "data/card2card_embeddings.npz"
DEFAULT_FAISS_INDEX = "data/card2card.faiss"
DEFAULT_SPARSE_INDEX = "data/card2card_sparse_index"
DEFAULT_DB_PATH = "data/modellake.db"
# Card2Tab2Card needs this to map model_id -> tables (default from card2tab2card CLI)
DEFAULT_RELATIONSHIP_PARQUET = "data_citationlake/processed/modelcard_step3_dedup.parquet"

# All job outputs (search results, integration, evaluation, QA) live under data/jobs/<job_id>
JOBS_DIR = os.path.join(REPO_ROOT, "data", "jobs")


def _integration_run_key(integration_type: str, search_type: str, card2card_mode: str) -> str:
    """Slug for integration run filename: e.g. union_single_column_dense."""
    import re
    parts = [integration_type or "union", search_type or "single_column", card2card_mode or "dense"]
    return "_".join(re.sub(r"[^a-z0-9_]", "_", (p or "").lower().strip()) for p in parts)


app = Flask(__name__)
CORS(app)

jobs: Dict[str, "JobLogger"] = {}


class JobLogger:
    """Thread-safe logger for job progress"""
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.logs: List[Dict] = []
        self.lock = threading.Lock()
        self.status = "pending"
        self.results: Optional[Dict] = None
    
    def log(self, message: str):
        with self.lock:
            now = datetime.now()
            ts = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.logs.append({"timestamp": now.isoformat(), "message": message})
            print(f"[{ts}] [{self.job_id}] {message}", flush=True)
    
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
    r = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return (r.returncode, r.stdout or "", r.stderr or "")


def _read_json(path: str) -> Optional[Dict]:
    """Read JSON file if it exists; else return None. No try/except - caller handles."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Valid model IDs (have tables): built by scripts/build_valid_model_ids_txt.py (Part 1); inference only loads
VALID_MODEL_IDS_TXT = "data/valid_model_ids_with_tables.txt"


def _load_valid_model_ids_with_tables(txt_path: Optional[str] = None) -> set:
    """Load set of model_id that have tables. Txt is produced by scripts/build_valid_model_ids_txt.py (Part 1)."""
    path = txt_path or os.path.join(REPO_ROOT, VALID_MODEL_IDS_TXT)
    if not os.path.isfile(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except Exception:
        return set()


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
        logger.set_results({
            "error": msg,
            "model_id": mid,
            "card2card_results": [],
            "card2tab2card_results": {},
        })

    try:
        _run_pipeline_body(
            logger, job_id, job_dir, start_time,
            query, top_k, model_id, table_search_k, tab2tab_mode, tab2tab_json, card2card_retrieval_mode,
            require_seed_has_tables,
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
):
    logger.log("Starting search pipeline (CLI)...")
    logger.log("Mode: Query → ModelCard → Search" if query else "Mode: Model ID → Search")
    if query:
        logger.log(f"Query: {query}")
    else:
        logger.log(f"Model ID: {model_id}")
    logger.set_status("running")

    # Resolve model_id (query mode: from FAISS retrieval only; no default/hardcoded id)
    seed_no_tables_skip_table_search = False  # when True: use first model for Card2Card only, skip Card2Tab2Card and set reason
    if query:
        logger.log("Step 1: Extracting model card from query (query2modelcard)...")
        logger.log(f"query2modelcard input query: {query!r}")
        q2m_out = os.path.join(job_dir, "query2modelcard.json")
        # Narrow down: fetch 10× user top_k (cap 500), then postprocess to pick first with tables
        q2m_top_k = min(500, 10 * top_k) if require_seed_has_tables else 1
        if require_seed_has_tables:
            logger.log(f"Table Search Seed Model: Pick first model with tables. query2modelcard top_k={q2m_top_k} (10× of {top_k}), then pick first with tables.")
        # Run as script to avoid importing whole src.search (card2card, FAISS, etc.) and RuntimeWarning/segfault
        q2m_script = os.path.join(REPO_ROOT, "src", "search", "query2modelcard.py")
        cmd = [
            sys.executable, q2m_script,
            "--query", query,
            "--top_k", str(q2m_top_k),
            "--emb_npz", DEFAULT_EMB_NPZ,
            "--faiss_index", DEFAULT_FAISS_INDEX,
            "--output_json", q2m_out,
        ]
        rc, out, err = _run_cmd(cmd, REPO_ROOT)
        if rc != 0:
            logger.log(f"query2modelcard failed (exit {rc}): {err or out}")
            logger.set_status("error")
            logger.set_results({"error": f"query2modelcard failed: {err or out}", "model_id": None, "card2card_results": [], "card2tab2card_results": {}})
            return
        data = _read_json(q2m_out)
        if not data or "results" not in data or not data["results"]:
            logger.log("query2modelcard returned no results")
            logger.set_status("error")
            logger.set_results({"error": "No model from query", "model_id": None, "card2card_results": [], "card2tab2card_results": {}})
            return
        results_list = data["results"]
        stored_query = data.get("query", "")

        if require_seed_has_tables:
            # Valid = model_id that have tables (built by scripts/build_valid_model_ids_txt.py; inference only loads)
            valid_model_ids = _load_valid_model_ids_with_tables()
            logger.log(f"Narrow down: valid model IDs (have tables) = {len(valid_model_ids)} (from {VALID_MODEL_IDS_TXT})")
            chosen = None
            for i, r in enumerate(results_list):
                mid = r if isinstance(r, str) else (r.get("model_id") if isinstance(r, dict) else str(r))
                if not mid:
                    continue
                if str(mid).strip() in valid_model_ids:
                    chosen = str(mid).strip()
                    logger.log(f"Narrow down: first result in valid set is #{i+1}: {chosen}")
                    break
            if chosen is not None:
                model_id = chosen
                logger.log(f"Extracted model (with tables): {model_id} (from query2modelcard JSON, query in file: {stored_query!r})")
            else:
                # Cross: none of top-K are in valid set; use raw top-1 for Card2Card only, skip Table Search
                first = results_list[0]
                model_id = first if isinstance(first, str) else (first.get("model_id") if isinstance(first, dict) else str(first))
                seed_no_tables_skip_table_search = True
                logger.log(f"Table Search Seed Model: Pick first model with tables - none of top-{len(results_list)} in valid set (have tables). Using top-1 for Card2Card only; Table Search skipped.")
                logger.log(f"Extracted model (no tables): {model_id} (from query2modelcard JSON)")
        else:
            first = results_list[0]
            model_id = first if isinstance(first, str) else (first.get("model_id") if isinstance(first, dict) else str(first))
            if not model_id:
                logger.log("query2modelcard returned empty model_id")
                logger.set_status("error")
                logger.set_results({"error": "Empty model_id from query", "model_id": None, "card2card_results": [], "card2tab2card_results": {}})
                return
            logger.log(f"Extracted model: {model_id} (from query2modelcard JSON, query in file: {stored_query!r})")
    else:
        if not model_id:
            logger.log("model_id is required in modelid mode")
            logger.set_status("error")
            logger.set_results({"error": "Model ID is required (empty input)", "model_id": None, "card2card_results": [], "card2tab2card_results": {}})
            return
        logger.log(f"Using model_id: {model_id}")
        # User-provided ID must exist in our dataset; otherwise fail before any downstream
        check_cmd = [
            sys.executable, os.path.join(REPO_ROOT, "scripts", "check_model_in_index.py"),
            "--model_id", model_id,
            "--emb_npz", DEFAULT_EMB_NPZ,
        ]
        rc, out, err = _run_cmd(check_cmd, REPO_ROOT, timeout=60)
        if rc != 0:
            msg = (err or out or "Model ID not in dataset").strip()
            if "not in dataset" not in msg and "not in " not in msg:
                msg = f"Model ID '{model_id}' not found in dataset (not in card2card index). Cannot run downstream."
            logger.log(msg)
            logger.set_status("error")
            logger.set_results({"error": msg, "model_id": model_id, "card2card_results": [], "card2tab2card_results": {}})
            return
        logger.log("Model ID found in dataset, proceeding.")

    # Table search requires these files; fail early with clear message if missing
    rel_path = os.path.join(REPO_ROOT, DEFAULT_RELATIONSHIP_PARQUET)
    db_path_abs = os.path.join(REPO_ROOT, DEFAULT_DB_PATH)
    if not os.path.isfile(rel_path):
        logger.log(f"Table search unavailable: relationship_parquet not found at {rel_path}")
        logger.set_status("error")
        logger.set_results({
            "error": f"Table search failed: relationship_parquet not found at {DEFAULT_RELATIONSHIP_PARQUET}. Please add the file or symlink data_citationlake.",
            "model_id": model_id,
            "card2card_results": [],
            "card2tab2card_results": {},
        })
        return
    if not os.path.isfile(db_path_abs):
        logger.log(f"Table search unavailable: db not found at {db_path_abs}")
        logger.set_status("error")
        logger.set_results({
            "error": f"Table search failed: db not found at {DEFAULT_DB_PATH}. Please build or copy modellake.db.",
            "model_id": model_id,
            "card2card_results": [],
            "card2tab2card_results": {},
        })
        return

    k_table = table_search_k if table_search_k is not None else min(max(int(top_k * 1.5), 20), 20)

    # Step 2: Run Card2Card (dense, sparse, hybrid) and Card2Tab2Card in parallel via CLI
    logger.log("Step 2: Running Card2Card + Card2Tab2Card (parallel CLI)...")

    def run_card2card(mode: str) -> tuple:
        out_path = os.path.join(job_dir, f"card2card_{mode}.json")
        cmd = [
            sys.executable, "-m", "src.search.card2card", "search",
            "--model_id", model_id,
            "--retrieval_mode", mode,
            "--top_k", str(top_k),
            "--output_json", out_path,
        ]
        if mode == "dense":
            cmd.extend(["--emb_npz", DEFAULT_EMB_NPZ, "--faiss_index", DEFAULT_FAISS_INDEX])
        elif mode == "sparse":
            cmd.extend(["--sparse_index_path", DEFAULT_SPARSE_INDEX])
        else:
            cmd.extend(["--emb_npz", DEFAULT_EMB_NPZ, "--faiss_index", DEFAULT_FAISS_INDEX, "--sparse_index_path", DEFAULT_SPARSE_INDEX, "--hybrid_method", "rrf"])
        t0 = time.time()
        rc, out, err = _run_cmd(cmd, REPO_ROOT)
        elapsed = time.time() - t0
        return (mode, rc, out_path, out, err, elapsed)

    def run_card2tab2card(st: str) -> tuple:
        out_path = os.path.join(job_dir, f"card2tab2card_{st}.json")
        # Run as script to avoid RuntimeWarning (src.search pre-import) and unpredictable behaviour
        c2t2c_script = os.path.join(REPO_ROOT, "src", "search", "card2tab2card.py")
        cmd = [
            sys.executable, c2t2c_script,
            "--model_id", model_id,
            "--search_type", st,
            "--k", str(k_table),
            "--modelcard_k", "0",  # 0 = no limit: return all models that contain the retrieved tables
            "--db_path", DEFAULT_DB_PATH,
            "--relationship_parquet", DEFAULT_RELATIONSHIP_PARQUET,
            "--no_citationlake",
            "--output_json", out_path,
        ]
        t0 = time.time()
        rc, out, err = _run_cmd(cmd, REPO_ROOT)
        elapsed = time.time() - t0
        return (st, rc, out_path, out, err, elapsed)

    card2card_modes = ["dense", "sparse", "hybrid"]
    card2tab2card_types = ["keyword", "single_column"]  # CLI supports these without CSV

    futures = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for m in card2card_modes:
            futures[ex.submit(run_card2card, m)] = ("card2card", m)
        if not seed_no_tables_skip_table_search:
            for st in card2tab2card_types:
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
                    # CLI writes "neighbors" (list of model_id); legacy used "results"
                    card2card_all[mode] = data.get("neighbors", data.get("results", []))
                else:
                    card2card_all[mode] = []
        else:
            st, rc, out_path, out, err, elapsed = res
            logger.log(f"[Card2Tab2Card-{st}] Done in {elapsed:.2f}s")
            if rc != 0:
                logger.log(f"[Card2Tab2Card-{st}] Error (exit {rc}): {err or out}")
                # Save as empty result (not error object) so frontend can handle it gracefully
                card2tab2card_all[st] = {"model_ids": [], "intermediate": {}}
            else:
                data = _read_json(out_path)
                if data is not None:
                    # CLI writes {"model_ids": [...], "query_tables": [...], "intermediate": {...}}
                    # Save the full data object (not just model_ids) so frontend can access intermediate.table_to_models
                    if isinstance(data, dict):
                        card2tab2card_all[st] = data
                        mid = data.get("model_ids", [])
                        lst = list(mid) if isinstance(mid, (list, np.ndarray)) else []
                        qty = len(data.get("query_tables", []))
                    else:
                        # Legacy format: just a list/array of model_ids
                        lst = list(data) if isinstance(data, (list, np.ndarray)) else []
                        card2tab2card_all[st] = {"model_ids": lst, "intermediate": {}}
                        qty = 0
                    logger.log(f"[Card2Tab2Card-{st}] Read {len(lst)} model_ids, {qty} query_tables for seed model")
                    if len(lst) == 0 and qty == 0:
                        logger.log(f"[Card2Tab2Card-{st}] Seed model has no tables in relationship_parquet (model_id not in parquet or no csv_basename). Check {DEFAULT_RELATIONSHIP_PARQUET} has column modelId and rows for this model.")
                        if table_search_empty_reason is None:
                            table_search_empty_reason = (
                                f"Seed model «{model_id}» has no tables in the dataset: it is not in "
                                f"{DEFAULT_RELATIONSHIP_PARQUET} or has no csv_basename. "
                                "Try another query whose top result has linked tables, or check the parquet has column modelId and rows for this model."
                            )
                        cli_out = (err or out or "").strip()
                        if cli_out and ("No tables" in cli_out or "Warning" in cli_out):
                            for line in cli_out.split("\n")[-3:]:
                                if line.strip():
                                    logger.log(f"[Card2Tab2Card-{st}] CLI: {line.strip()}")
                else:
                    # Save as empty object (consistent format) so frontend can handle it
                    card2tab2card_all[st] = {"model_ids": [], "intermediate": {}}
                    logger.log(f"[Card2Tab2Card-{st}] No JSON at {out_path}")

    # When we skipped Table Search (require_seed_has_tables but none had tables), fill empty
    if seed_no_tables_skip_table_search:
        for st in card2tab2card_types:
            card2tab2card_all[st] = {"model_ids": [], "intermediate": {}}
        if table_search_empty_reason is None:
            table_search_empty_reason = (
                "None of the top-20 models from the query have tables in the dataset. "
                "Table Search skipped. Select «Use top-1 result» for Table Search Seed Model to run with top-1 anyway."
            )
    # Fill missing card2tab2card types with empty (no scan)
    for st in ["multi_column", "unionable", "complex", "correlation", "imputation", "augmentation",
               "dependent_data", "feature_for_ml", "multi_column_collinearity", "negative_example"]:
        if st not in card2tab2card_all:
            card2tab2card_all[st] = {"model_ids": [], "intermediate": {}}

    primary = card2card_all.get("dense", card2card_all.get(list(card2card_all.keys())[0] if card2card_all else "dense", []))
    if isinstance(primary, dict) and "error" in primary:
        primary = []

    elapsed_total = time.time() - start_time
    logger.log(f"Step 3: Done. Total time: {elapsed_total:.2f}s")

    results_data = {
        "job_id": job_id,
        "query": query,
        "model_id": model_id,
        "top_k": top_k,
        "table_search_k": k_table,
        "card2card_retrieval_mode": card2card_retrieval_mode,
        "card2card_results": primary,
        "card2card_all_modes": card2card_all,
        "card2tab2card_results": card2tab2card_all,
        "timestamp": datetime.fromtimestamp(start_time).isoformat(),
        "folder_path": job_dir,
        "running_time_seconds": round(elapsed_total, 3),
    }
    if table_search_empty_reason:
        results_data["table_search_reason"] = table_search_empty_reason

    out_file = os.path.join(job_dir, "search_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    logger.log(f"Results saved to {out_file}")
    logger.set_results(results_data)
    logger.set_status("completed")
    logger.log("Pipeline completed.")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/search", methods=["POST"])
def search():
    data = request.json or {}
    search_mode = data.get("search_mode", "new")

    if search_mode == "mimic":
        folder_name = data.get("folder_name")
        if not folder_name:
            return jsonify({"status": "error", "message": "folder_name required for mimic"}), 400
        if folder_name == "template":
            base_dir = os.path.join(REPO_ROOT, "config", "demo_template")
            search_path = os.path.join(base_dir, "search_results.json")
        else:
            base_dir = os.path.join(JOBS_DIR, folder_name)
            search_path = os.path.join(base_dir, "search_results.json")
        if not os.path.exists(search_path):
            return jsonify({"status": "error", "message": f"Saved results not found: {folder_name}"}), 404
        with open(search_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        # For saved searches under data/jobs/, use folder_name as job_id so integrate/eval/qa use same folder
        # For template (config/demo_template), use new uuid so we do not write into config
        if folder_name == "template":
            job_id = str(uuid.uuid4())
            base_dir = None  # no optional files for template
        else:
            job_id = folder_name
        jobs[job_id] = JobLogger(job_id)
        jobs[job_id].set_results(saved)
        jobs[job_id].set_status("completed")
        out = {"status": "completed", "job_id": job_id, "results": saved}
        if base_dir and os.path.isdir(base_dir):
            for key, filename in [
                ("integration_model_search", "integration_model_search.json"),
                ("integration_table_search", "integration_table_search.json"),
                ("evaluation_results", "evaluation_results.json"),
                ("qa_results", "qa_results.json"),
            ]:
                p = os.path.join(base_dir, filename)
                if os.path.isfile(p):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            out[key] = json.load(f)
                    except Exception:
                        pass
            integration_runs = []
            for fname in os.listdir(base_dir):
                if fname.startswith("integration_run_") and fname.endswith(".json"):
                    try:
                        with open(os.path.join(base_dir, fname), "r", encoding="utf-8") as f:
                            run_data = json.load(f)
                        integration_runs.append(run_data)
                    except Exception:
                        pass
            if integration_runs:
                out["integration_runs"] = integration_runs
            elif out.get("integration_model_search") and out.get("integration_table_search"):
                m, t = out["integration_model_search"], out["integration_table_search"]
                if m.get("status") == "success" and t.get("status") == "success":
                    out["integration_runs"] = [{
                        "key": _integration_run_key(
                            m.get("integration_type") or "union",
                            t.get("search_type") or "single_column",
                            m.get("card2card_retrieval_mode") or "dense",
                        ),
                        "integration_type": m.get("integration_type") or t.get("integration_type"),
                        "search_type": t.get("search_type"),
                        "card2card_retrieval_mode": m.get("card2card_retrieval_mode"),
                        "k": m.get("k") or t.get("k"),
                        "max_models": m.get("max_models") or t.get("max_models"),
                        "model_result": m,
                        "table_result": t,
                    }]
        return jsonify(out)

    mode = data.get("mode", "query")
    top_k = int(data.get("top_k", 20))
    table_search_k = data.get("table_search_k")
    tab2tab_mode = data.get("tab2tab_mode", "search")
    tab2tab_json = data.get("tab2tab_json")
    card2card_retrieval_mode = data.get("card2card_retrieval_mode", "dense")
    require_seed_has_tables = bool(data.get("require_seed_has_tables", False))

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

    job_id = str(uuid.uuid4())
    jobs[job_id] = JobLogger(job_id)
    thread = threading.Thread(
        target=run_search_pipeline,
        args=(job_id, query, top_k, model_id, table_search_k, tab2tab_mode, tab2tab_json, card2card_retrieval_mode, require_seed_has_tables),
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


@app.route("/api/results/<job_id>", methods=["GET"])
def get_results(job_id: str):
    if job_id not in jobs:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    logger = jobs[job_id]
    if logger.status == "error" and logger.results is not None:
        return jsonify({"status": "success", "job_id": job_id, "results": logger.results})
    if logger.status != "completed":
        return jsonify({"status": logger.status, "message": "Job not completed yet"}), 202
    return jsonify({"status": "success", "job_id": job_id, "results": logger.results})


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


PRESET_QUERIES_PATH = os.path.join(REPO_ROOT, "config", "preset_queries.json")


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
    jobs_parent = JOBS_DIR
    template_path = os.path.join(REPO_ROOT, "config", "demo_template", "search_results.json")
    template_available = os.path.isfile(template_path)
    if not os.path.isdir(jobs_parent):
        os.makedirs(jobs_parent, exist_ok=True)
        return jsonify({"status": "success", "searches": [], "template_available": template_available})
    searches = []
    for name in sorted(os.listdir(jobs_parent), reverse=True)[:50]:
        path = os.path.join(jobs_parent, name)
        json_path = os.path.join(path, "search_results.json")
        if os.path.isdir(path) and os.path.isfile(json_path):
            entry = {"folder_name": name, "path": path, "query": None, "model_id": None, "timestamp_str": "", "top_k": None}
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                entry["query"] = saved.get("query") or ""
                entry["model_id"] = saved.get("model_id") or ""
                ts = saved.get("timestamp")
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        entry["timestamp_str"] = dt.strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        entry["timestamp_str"] = str(ts)[:16]
                entry["top_k"] = saved.get("top_k")
            except Exception:
                pass
            searches.append(entry)
    return jsonify({"status": "success", "searches": searches, "template_available": template_available})


# Stubs for integrate / evaluate / qa (use legacy backend for full support)
@app.route("/api/integrate", methods=["POST"])
def integrate():
    """Integrate tables from Card2Tab2Card search results"""
    data = request.get_json()
    job_id = data.get("job_id")
    search_type = data.get("search_type", "keyword")
    integration_type = data.get("integration_type", "union")
    k = int(data.get("k", 10))
    max_models = int(data.get("max_models", 10))
    
    if not job_id:
        return jsonify({"status": "error", "message": "job_id required"}), 400
    
    job_dir = os.path.join(JOBS_DIR, job_id)
    results_file = os.path.join(job_dir, "search_results.json")
    if not os.path.exists(results_file):
        return jsonify({"status": "error", "message": f"Results file not found for job {job_id}"}), 404
    os.makedirs(job_dir, exist_ok=True)
    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
        from integration.table_integration import integrate_tables_from_search_results
        result = integrate_tables_from_search_results(
            search_results_json=results_file,
            search_type=search_type,
            integration_type=integration_type,
            k=k,
            db_path=DEFAULT_DB_PATH
        )
        
        if not result.get("success", False):
            return jsonify({"status": "error", "message": result.get("error", "Integration failed")}), 500
        
        # Convert DataFrame to dict for JSON response (NaN -> null for valid JSON)
        integrated_df = result.get("integrated_table")
        saved_path = None
        if integrated_df is not None:
            raw_data = integrated_df.values.tolist()
            result["integrated_table"] = {
                "columns": list(integrated_df.columns),
                "data": _sanitize_for_json(raw_data)
            }
            # Save integrated table to job dir for reference
            try:
                csv_name = "integrated_table_search.csv"
                save_path = os.path.join(job_dir, csv_name)
                integrated_df.to_csv(save_path, index=False, encoding="utf-8")
                saved_path = os.path.join("data", "jobs", job_id, csv_name)
            except Exception:
                pass
        if saved_path:
            result["saved_path"] = saved_path
        try:
            save_payload = {
                "status": "success",
                "integration_type": integration_type,
                "search_type": search_type,
                "k": k,
                "max_models": max_models,
                **result,
            }
            save_json = os.path.join(job_dir, "integration_table_search.json")
            with open(save_json, "w", encoding="utf-8") as f:
                json.dump(save_payload, f, ensure_ascii=False, indent=0)
        except Exception:
            pass
        return jsonify({"status": "success", **result})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/integrate-model-search", methods=["POST"])
def integrate_model_search():
    """Integrate tables from Card2Card (model search) results"""
    data = request.get_json()
    job_id = data.get("job_id")
    integration_type = data.get("integration_type", "union")
    k = int(data.get("k", 10))
    max_models = int(data.get("max_models", 10))
    card2card_retrieval_mode = data.get("card2card_retrieval_mode") or None  # dense, sparse, hybrid

    if not job_id:
        return jsonify({"status": "error", "message": "job_id required"}), 400
    job_dir = os.path.join(JOBS_DIR, job_id)
    results_file = os.path.join(job_dir, "search_results.json")
    if not os.path.exists(results_file):
        return jsonify({"status": "error", "message": f"Results file not found for job {job_id}"}), 404
    os.makedirs(job_dir, exist_ok=True)
    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
        from integration.table_integration import integrate_tables_from_model_search_results
        
        # For the demo backend we don't rely on live CitationLake.
        # Force integration to use the local relationship_parquet instead of CitationLake,
        # to avoid "CitationLake not available ... no fallback data found" errors.
        result = integrate_tables_from_model_search_results(
            search_results_json=results_file,
            integration_type=integration_type,
            k=k,
            max_models=max_models,
            db_path=DEFAULT_DB_PATH,
            relationship_parquet=DEFAULT_RELATIONSHIP_PARQUET,
            use_citationlake=False,
            card2card_retrieval_mode=card2card_retrieval_mode,
        )
        
        if not result.get("success", False):
            return jsonify({"status": "error", "message": result.get("error", "Integration failed")}), 500
        
        # Convert DataFrame to dict for JSON response (NaN -> null for valid JSON)
        integrated_df = result.get("integrated_table")
        saved_path = None
        if integrated_df is not None:
            raw_data = integrated_df.values.tolist()
            result["integrated_table"] = {
                "columns": list(integrated_df.columns),
                "data": _sanitize_for_json(raw_data)
            }
            try:
                csv_name = "integrated_model_search.csv"
                save_path = os.path.join(job_dir, csv_name)
                integrated_df.to_csv(save_path, index=False, encoding="utf-8")
                saved_path = os.path.join("data", "jobs", job_id, csv_name)
            except Exception:
                pass
        if saved_path:
            result["saved_path"] = saved_path
        try:
            save_payload = {
                "status": "success",
                "integration_type": integration_type,
                "card2card_retrieval_mode": card2card_retrieval_mode or "dense",
                "k": k,
                "max_models": max_models,
                **result,
            }
            save_json = os.path.join(job_dir, "integration_model_search.json")
            with open(save_json, "w", encoding="utf-8") as f:
                json.dump(save_payload, f, ensure_ascii=False, indent=0)
        except Exception:
            pass
        return jsonify({"status": "success", **result})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    return jsonify({"status": "error", "message": "Use legacy backend for evaluation"}), 501


@app.route("/api/qa", methods=["POST"])
def qa():
    return jsonify({"status": "error", "message": "Use legacy backend for QA"}), 501


@app.route("/api/save-integration-run", methods=["POST"])
def save_integration_run():
    """Save one integration run to job_dir as integration_run_<key>.json for tabs. Creates job_dir if missing."""
    data = request.get_json() or {}
    job_id = data.get("job_id")
    key = data.get("key")
    if not job_id or not key:
        return jsonify({"status": "error", "message": "job_id and key required"}), 400
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    safe_key = re.sub(r"[^a-z0-9_]", "_", (key or "").lower().strip()) or "run"
    path = os.path.join(job_dir, f"integration_run_{safe_key}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=0)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/save-evaluation", methods=["POST"])
def save_evaluation():
    """Save evaluation result to job_dir for load-previous restore. Creates job_dir if missing."""
    data = request.get_json() or {}
    job_id = data.get("job_id")
    if not job_id:
        return jsonify({"status": "error", "message": "job_id required"}), 400
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    path = os.path.join(job_dir, "evaluation_results.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=0)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/save-qa", methods=["POST"])
def save_qa():
    """Save QA result to job_dir for load-previous restore. Creates job_dir if missing."""
    data = request.get_json() or {}
    job_id = data.get("job_id")
    if not job_id:
        return jsonify({"status": "error", "message": "job_id required"}), 400
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    path = os.path.join(job_dir, "qa_results.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=0)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("Backend (CLI-based) starting...", flush=True)
    port = int(os.environ.get("PORT", "5002"))
    app.run(host="0.0.0.0", port=port, debug=False)
