"""
Backend API for ModelSearch Demo (CLI-based)

Runs search via subprocess commands from docs/build_index.md.
All outputs go to a single job dir; no scanning of JSON paths.
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

# Repo root (backend lives in src/demo/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Paths from build_index.md (relative to repo root)
DEFAULT_EMB_NPZ = "data/card2card_embeddings.npz"
DEFAULT_FAISS_INDEX = "data/card2card.faiss"
DEFAULT_SPARSE_INDEX = "data/card2card_sparse_index"
DEFAULT_DB_PATH = "data/modellake.db"

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


def run_search_pipeline(
    job_id: str,
    query: Optional[str] = None,
    top_k: int = 20,
    model_id: Optional[str] = None,
    table_search_k: Optional[int] = None,
    tab2tab_mode: str = "search",
    tab2tab_json: Optional[str] = None,
    card2card_retrieval_mode: str = "dense",
):
    """Run pipeline by calling CLI commands (build_index.md). All outputs under job_dir."""
    logger = jobs.get(job_id)
    if not logger:
        return

    job_dir = os.path.join(REPO_ROOT, "data", job_id)
    os.makedirs(job_dir, exist_ok=True)
    start_time = time.time()

    logger.log("Starting search pipeline (CLI)...")
    logger.log("Mode: Query → ModelCard → Search" if query else "Mode: Model ID → Search")
    if query:
        logger.log(f"Query: {query}")
    else:
        logger.log(f"Model ID: {model_id}")
    logger.set_status("running")

    # Resolve model_id
    if query:
        logger.log("Step 1: Extracting model card from query (query2modelcard)...")
        q2m_out = os.path.join(job_dir, "query2modelcard.json")
        cmd = [
            sys.executable, "-m", "src.search.query2modelcard",
            "--query", query,
            "--top_k", "1",
            "--emb_npz", DEFAULT_EMB_NPZ,
            "--faiss_index", DEFAULT_FAISS_INDEX,
            "--output_json", q2m_out,
        ]
        rc, out, err = _run_cmd(cmd, REPO_ROOT)
        if rc != 0:
            logger.log(f"query2modelcard failed (exit {rc}): {err or out}")
            logger.set_status("error")
            logger.set_results({"error": f"query2modelcard failed: {err or out}", "card2card_results": [], "card2tab2card_results": {}})
            return
        data = _read_json(q2m_out)
        if not data or "results" not in data or not data["results"]:
            logger.log("query2modelcard returned no results")
            logger.set_status("error")
            logger.set_results({"error": "No model from query", "card2card_results": [], "card2tab2card_results": {}})
            return
        model_id = data["results"][0]
        logger.log(f"Extracted model: {model_id}")
    else:
        if not model_id:
            logger.log("model_id is required in modelid mode")
            logger.set_status("error")
            return
        logger.log(f"Using model_id: {model_id}")

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
        cmd = [
            sys.executable, "-m", "src.search.card2tab2card",
            "--model_id", model_id,
            "--search_type", st,
            "--k", str(k_table),
            "--db_path", DEFAULT_DB_PATH,
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
        for st in card2tab2card_types:
            futures[ex.submit(run_card2tab2card, st)] = ("card2tab2card", st)

    card2card_all = {}
    card2tab2card_all = {}

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
                if data is not None and "results" in data:
                    card2card_all[mode] = data["results"]
                else:
                    card2card_all[mode] = []
        else:
            st, rc, out_path, out, err, elapsed = res
            logger.log(f"[Card2Tab2Card-{st}] Done in {elapsed:.2f}s")
            if rc != 0:
                logger.log(f"[Card2Tab2Card-{st}] Error (exit {rc}): {err or out}")
                card2tab2card_all[st] = {"error": err or out}
            else:
                data = _read_json(out_path)
                if data is not None:
                    card2tab2card_all[st] = data.get("model_ids", data) if isinstance(data.get("model_ids"), list) else data
                else:
                    card2tab2card_all[st] = []

    # Fill missing card2tab2card types with empty (no scan)
    for st in ["multi_column", "unionable", "complex", "correlation", "imputation", "augmentation",
               "dependent_data", "feature_for_ml", "multi_column_collinearity", "negative_example"]:
        if st not in card2tab2card_all:
            card2tab2card_all[st] = []

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
        path = os.path.join(REPO_ROOT, "data", "template", "search_results.json") if folder_name == "template" else os.path.join(REPO_ROOT, "data", folder_name, "search_results.json")
        if not os.path.exists(path):
            return jsonify({"status": "error", "message": f"Saved results not found: {folder_name}"}), 404
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        job_id = str(uuid.uuid4())
        jobs[job_id] = JobLogger(job_id)
        jobs[job_id].set_results(saved)
        jobs[job_id].set_status("completed")
        return jsonify({"status": "completed", "job_id": job_id, "results": saved})

    mode = data.get("mode", "query")
    top_k = int(data.get("top_k", 20))
    table_search_k = data.get("table_search_k")
    tab2tab_mode = data.get("tab2tab_mode", "search")
    tab2tab_json = data.get("tab2tab_json")
    card2card_retrieval_mode = data.get("card2card_retrieval_mode", "dense")

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
        args=(job_id, query, top_k, model_id, table_search_k, tab2tab_mode, tab2tab_json, card2card_retrieval_mode),
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


PRESET_QUERIES_PATH = os.path.join(REPO_ROOT, "data", "preset_queries.json")


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
    data_dir = os.path.join(REPO_ROOT, "data")
    if not os.path.isdir(data_dir):
        return jsonify({"status": "success", "folders": []})
    folders = []
    for name in sorted(os.listdir(data_dir), reverse=True)[:50]:
        path = os.path.join(data_dir, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "search_results.json")):
            folders.append({"name": name, "path": path})
    return jsonify({"status": "success", "folders": folders})


# Stubs for integrate / evaluate / qa (use legacy backend for full support)
@app.route("/api/integrate", methods=["POST"])
def integrate():
    return jsonify({"status": "error", "message": "Use legacy backend (bak/backend_before_cmd.py) for table integration"}), 501


@app.route("/api/integrate-model-search", methods=["POST"])
def integrate_model_search():
    return jsonify({"status": "error", "message": "Use legacy backend for model-search integration"}), 501


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    return jsonify({"status": "error", "message": "Use legacy backend for evaluation"}), 501


@app.route("/api/qa", methods=["POST"])
def qa():
    return jsonify({"status": "error", "message": "Use legacy backend for QA"}), 501


if __name__ == "__main__":
    print("Backend (CLI-based) starting...", flush=True)
    port = int(os.environ.get("PORT", "5002"))
    app.run(host="0.0.0.0", port=port, debug=False)
