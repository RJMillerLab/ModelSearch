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

import atexit, html, json, math, os, random, string, threading, time
from datetime import datetime
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd
from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context
from flask_cors import CORS

from src.config import *
from src.demo.job_schema import JobMeta, JobPaths, Query2ModelCardFile, Query2Tab2CardFullMap
from src.integration.table_integration import TableIntegrater
from src.search.ir_searcher import DenseSearcher, SparseSearcher
from src.search.query2modelcard import Query2ModelCardSearch
from src.search.query2tab2card import Query2Tab2CardSearch
from src.utils import _paths_for_resource_set, preview_from_local, resolve_table_path


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

jobs: Dict[str, Dict[str, Any]] = {}
search_runtime: Optional[Dict[str, Any]] = None


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


def _table_payload_from_path(path_q: str) -> Dict[str, Any]:
    resolved = resolve_table_path(path_q) or str(path_q)
    df = pd.read_csv(resolved)
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


def _load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    csv_name = f"integrated_model_search_{integration_type}.csv"
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
    table_paths = payload["table_paths"][:k]
    df = TableIntegrater().run(local_table_paths(table_paths), mode=integration_type)
    csv_name = f"integrated_table_search_{integration_type}_{search_type}.csv"
    csv_path = os.path.join(paths.job_dir, csv_name)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    response_payload = {"status": "success", "integration_type": integration_type, "search_type": search_type, "k": k, "max_models": max_models, "tables_source": payload["tables_source"], "integrated_table": {"columns": list(df.columns), "data": sanitize_for_json(df.values.tolist())}, "saved_path": f"jobs_251117/{job_id}/{csv_name}", **payload}
    _save_json(os.path.join(paths.job_dir, f"integration_table_search_{integration_type}_{search_type}_{tables_source}.json"), response_payload)
    return jsonify(response_payload)


@app.route("/api/integration-runs/<job_id>", methods=["GET"])
def integration_runs(job_id: str):
    paths = JobPaths(JOBS_DIR, job_id)
    if not os.path.isdir(paths.job_dir):
        return api_error(f"Unknown job_id: {job_id}", 404)

    model_runs: List[Dict[str, Any]] = []
    table_runs: List[Dict[str, Any]] = []

    for retrieval_mode in QUERY2MODELCARD_RETRIEVAL_MODES:
        payload = _load_json_if_exists(os.path.join(paths.job_dir, f"integration_model_search_alite_{retrieval_mode}.json"))
        if payload:
            model_runs.append(payload)

    for search_type in CARD2TAB2CARD_TYPES:
        for tables_source in ("intermediate", "all_from_modelcards"):
            payload = _load_json_if_exists(os.path.join(paths.job_dir, f"integration_table_search_alite_{search_type}_{tables_source}.json"))
            if payload:
                table_runs.append(payload)

    return jsonify({"status": "success", "job_id": job_id, "model_search_runs": model_runs, "table_search_runs": table_runs})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simplified ModelSearch backend")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    init_search_runtime()
    port = args.port if args.port is not None else int(os.environ.get("PORT", "5002"))
    app.run(host="0.0.0.0", port=port, debug=False)
