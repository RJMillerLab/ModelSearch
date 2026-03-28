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
from flask import Flask, Response, jsonify, render_template_string, request, send_from_directory, stream_with_context
from flask_cors import CORS

from src.config import *
from src.integration.pipeline_preview import Query2Tab2CardFullMap
from src.integration.table_integration import TableIntegrater
from src.search.ir_searcher import DenseSearcher, SparseSearcher
from src.search.query2modelcard import Query2ModelCardSearch
from src.search.query2tab2card import Query2Tab2CardSearch
from src.utils import _get_models_to_tables_batch_sql, _paths_for_resource_set, preview_from_local, resolve_table_path


JOB_META_FILENAME = "job_meta.json"
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

def close_search_runtime() -> None:
    global search_runtime
    if search_runtime is None:
        return
    search_runtime["con_data"].close()

atexit.register(close_search_runtime)

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def job_dir(job_id: str) -> str:
    return os.path.join(JOBS_DIR, job_id)


def job_meta_path(job_id: str) -> str:
    return os.path.join(job_dir(job_id), JOB_META_FILENAME)


def q2m_path(job_id: str) -> str:
    return os.path.join(job_dir(job_id), "query2modelcard.json")


def q2t2c_path(job_id: str, search_type: str) -> str:
    return os.path.join(job_dir(job_id), f"card2tab2card_{search_type}.json")


def generate_job_id() -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{ts}_{suffix}"


def extract_model_id(item: Any) -> str:
    if isinstance(item, dict):
        return str(item["model_id"]).strip()
    return str(item).strip()


def ordered_unique(items: List[Any]) -> List[str]:
    return list(dict.fromkeys(str(x).strip() for x in items if str(x).strip()))


def local_table_basenames(items: List[str]) -> List[str]:
    return [name for name in ordered_unique(items) if resolve_table_path(name)]


def local_table_paths(items: List[str]) -> List[str]:
    return [resolve_table_path(name) for name in ordered_unique(items)]


def model_neighbors_from_q2m(q2m_json: Dict[str, Any], mode: str, limit: int) -> List[str]:
    raw_items = q2m_json["results"][mode]
    seed_model_id = extract_model_id(q2m_json["results"]["dense"][0])
    model_ids = []
    for item in raw_items:
        model_id = extract_model_id(item)
        if model_id == seed_model_id:
            continue
        model_ids.append(model_id)
    return ordered_unique(model_ids)[:limit]


def model_search_preview_payload(job_meta: Dict[str, Any], q2m_json: Dict[str, Any], mode: str, max_models: Optional[int] = None) -> Dict[str, Any]:
    limit = int(max_models if max_models is not None else job_meta["model_top_k"])
    model_ids = model_neighbors_from_q2m(q2m_json, mode, limit)
    model_to_all_tables = _get_models_to_tables_batch_sql(model_ids, resources=job_meta["table_resources"])
    model_to_table_paths = {mid: local_table_basenames(model_to_all_tables[mid]) for mid in model_ids}
    model_to_table_paths = {mid: paths for mid, paths in model_to_table_paths.items() if paths}
    models_with_tables = list(model_to_table_paths.keys())
    table_paths = ordered_unique([path for paths in model_to_table_paths.values() for path in paths])
    return {
        "query2modelcard_retrieval_mode": mode,
        "models_with_tables": models_with_tables,
        "model_ids": models_with_tables,
        "model_to_table_paths": model_to_table_paths,
        "table_paths": table_paths,
        "job_context": {
            "query": job_meta["query"],
            "table_search_seed_model_id": extract_model_id(q2m_json["results"]["dense"][0]),
        },
        "stats": {
            "models_with_tables": len(models_with_tables),
            "total_unique_tables": len(table_paths),
        },
    }


def job_results(job_id: str) -> Dict[str, Any]:
    meta = read_json(job_meta_path(job_id))
    q2m_json = read_json(q2m_path(job_id))
    seed_model_id = extract_model_id(q2m_json["results"]["dense"][0])
    card2tab2card_results = {
        search_type: Query2Tab2CardFullMap(q2t2c_path(job_id, search_type)).build_backend_payload(search_type=search_type)
        for search_type in CARD2TAB2CARD_TYPES
    }
    right_max_models = max(len(card2tab2card_results[search_type]["model_ids"]) for search_type in CARD2TAB2CARD_TYPES)
    effective_model_top_k = min(meta["model_top_k"], right_max_models)
    query2modelcard_all_modes = {
        mode: model_neighbors_from_q2m(q2m_json, mode, effective_model_top_k)
        for mode in QUERY2MODELCARD_RETRIEVAL_MODES
    }
    return {
        "job_id": job_id,
        "query": meta["query"],
        "model_id": seed_model_id,
        "table_search_seed_model_id": seed_model_id,
        "top_k": meta["top_k"],
        "model_top_k": meta["model_top_k"],
        "effective_model_top_k": effective_model_top_k,
        "right_max_models": right_max_models,
        "table_search_k": meta["table_search_k"],
        "table_resources": meta["table_resources"],
        "use_by_type": meta["use_by_type"],
        "query2modelcard_results": query2modelcard_all_modes["dense"],
        "query2modelcard_all_modes": query2modelcard_all_modes,
        "card2tab2card_results": card2tab2card_results,
        "timestamp": meta["timestamp"],
        "folder_path": job_dir(job_id),
        "run_log_path": os.path.join(job_dir(job_id), "pipeline_run.log"),
        "running_time_seconds": meta["running_time_seconds"],
    }


def save_job_meta(job_id: str, *, query: str, top_k: int, model_top_k: int, table_search_k: int, table_resources: List[str], use_by_type: bool, running_time_seconds: float) -> None:
    write_json(
        job_meta_path(job_id),
        {
            "job_id": job_id,
            "query": query,
            "top_k": top_k,
            "model_top_k": model_top_k,
            "table_search_k": table_search_k,
            "table_resources": table_resources,
            "use_by_type": use_by_type,
            "timestamp": datetime.now().isoformat(),
            "running_time_seconds": round(running_time_seconds, 3),
        },
    )


def append_log(job_id: str, message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{ts}] [{job_id}] {message}"
    jobs[job_id]["logs"].append({"timestamp": datetime.now().isoformat(), "message": message})
    print(line, flush=True)
    with open(os.path.join(job_dir(job_id), "pipeline_run.log"), "a", encoding="utf-8") as f:
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
    jobs[job_id]["status"] = "running"
    os.makedirs(job_dir(job_id), exist_ok=True)
    start = time.time()
    rt = runtime
    q2m = Query2ModelCardSearch(query=query, top_k=min(200, 5 * top_k), job_id=job_id)
    q2m.search_dense(top_k=q2m.top_k, dense=rt["dense_full"])
    q2m.search_sparse(top_k=q2m.top_k, sparse=rt["sparse_full"])
    q2m.search_hybrid(top_k=q2m.top_k, sparse=rt["sparse_full"], dense=rt["dense_full"], candidate_factor=10)
    q2m.save_to_json(q2m_path(job_id))
    append_log(job_id, "query2modelcard finished")
    q2t2c = Query2Tab2CardSearch()
    for search_type in CARD2TAB2CARD_TYPES:
        q2t2c.pipeline_w_query_reranker(query, rt["con_data"], rt["dense_full"], rt["dense_wtable"], search_type=search_type, table_top_k=table_search_k, table_resources=rt["table_resources"], use_tab2tab_aug=bool(USE_TAB2TAB_AUG), apply_query_rerank=True, model_top_k=model_top_k, q2m_top_k=1)
        q2t2c.save_full_json(q2t2c_path(job_id, search_type))
        append_log(job_id, f"card2tab2card {search_type} finished")
    save_job_meta(job_id, query=query, top_k=top_k, model_top_k=model_top_k, table_search_k=table_search_k, table_resources=rt["table_resources"], use_by_type=use_by_type, running_time_seconds=time.time() - start)
    jobs[job_id]["results"] = job_results(job_id)
    jobs[job_id]["status"] = "completed"

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data["query"]
    top_k = data["top_k"]
    table_search_k = int(data["table_search_k"])
    model_top_k = int(data["model_top_k"])
    use_by_type = bool(data["use_by_type"])
    job_id = generate_job_id()
    jobs[job_id] = {"status": "pending", "logs": [], "results": None}
    threading.Thread(target=run_search_job, args=(job_id, query, top_k, table_search_k, model_top_k, use_by_type, search_runtime), daemon=True).start()
    return jsonify({"status": "started", "job_id": job_id})

@app.route("/api/status/<job_id>", methods=["GET"])
def status(job_id: str):
    return jsonify({"job_id": job_id, "status": jobs[job_id]["status"], "logs": jobs[job_id]["logs"]})


@app.route("/api/results/<job_id>", methods=["GET"])
def results(job_id: str):
    if jobs[job_id]["status"] != "completed":
        return jsonify({"status": jobs[job_id]["status"]}), 202
    return jsonify({"status": "success", "job_id": job_id, "results": jobs[job_id]["results"]})


@app.route("/api/logs/<job_id>", methods=["GET"])
def logs(job_id: str):
    def generate():
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
        path = job_dir(name)
        if not os.path.isdir(path):
            continue
        if not os.path.isfile(job_meta_path(name)):
            continue
        meta = read_json(job_meta_path(name))
        q2m_json = read_json(q2m_path(name))
        rows.append(
            {
                "folder_name": name,
                "path": path,
                "query": meta["query"],
                "model_id": extract_model_id(q2m_json["results"]["dense"][0]),
                "timestamp_str": datetime.fromisoformat(meta["timestamp"]).strftime("%Y-%m-%d %H:%M"),
                "top_k": meta["top_k"],
                "model_top_k": meta["model_top_k"],
                "use_by_type": meta["use_by_type"],
                "table_search_k": meta["table_search_k"],
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
    job_meta = read_json(job_meta_path(job_id))
    q2m_json = read_json(q2m_path(job_id))
    mode = data.get("query2modelcard_retrieval_mode") or "dense"
    if mode in ("integration", "saved", "default"):
        mode = "dense"
    payload = model_search_preview_payload(job_meta, q2m_json, mode, data.get("max_models"))
    return jsonify({"status": "success", **payload})


@app.route("/api/table-search-preview", methods=["POST"])
def table_search_preview():
    data = request.get_json() or {}
    job_id = str(data["job_id"]).strip()
    search_type = str(data.get("search_type", "single_column")).strip()
    payload = Query2Tab2CardFullMap(q2t2c_path(job_id, search_type)).build_backend_payload(search_type=search_type)
    return jsonify(
        {
            "status": "success",
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
    )


@app.route("/api/integrate-model-search", methods=["POST"])
def integrate_model_search():
    data = request.get_json() or {}
    job_id = str(data["job_id"]).strip()
    integration_type = str(data.get("integration_type", "alite")).strip()
    k = int(data.get("k", 10))
    max_models = int(data.get("max_models", 10))
    job_meta = read_json(job_meta_path(job_id))
    q2m_json = read_json(q2m_path(job_id))
    payload = model_search_preview_payload(job_meta, q2m_json, "dense", max_models)
    table_paths = payload["table_paths"][:k]
    df = TableIntegrater().run(local_table_paths(table_paths), mode=integration_type)
    csv_name = f"integrated_model_search_{integration_type}.csv"
    csv_path = os.path.join(job_dir(job_id), csv_name)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return jsonify({"status": "success", "integration_type": integration_type, "k": k, "max_models": max_models, "integrated_table": {"columns": list(df.columns), "data": sanitize_for_json(df.values.tolist())}, "saved_path": f"data_251117/jobs_251117/{job_id}/{csv_name}", **payload})


@app.route("/api/integrate", methods=["POST"])
def integrate():
    data = request.get_json() or {}
    job_id = str(data["job_id"]).strip()
    search_type = str(data.get("search_type", "unionable")).strip()
    integration_type = str(data.get("integration_type", "alite")).strip()
    k = int(data.get("k", 10))
    max_models = int(data.get("max_models", 10))
    payload = Query2Tab2CardFullMap(q2t2c_path(job_id, search_type)).build_backend_payload(search_type=search_type, max_models=max_models)
    table_paths = payload["table_paths"][:k]
    df = TableIntegrater().run(local_table_paths(table_paths), mode=integration_type)
    csv_name = f"integrated_table_search_{integration_type}_{search_type}.csv"
    csv_path = os.path.join(job_dir(job_id), csv_name)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return jsonify({"status": "success", "integration_type": integration_type, "search_type": search_type, "k": k, "max_models": max_models, "tables_source": payload["preview_meta"]["tables_source"], "integrated_table": {"columns": list(df.columns), "data": sanitize_for_json(df.values.tolist())}, "saved_path": f"data_251117/jobs_251117/{job_id}/{csv_name}", **payload})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simplified ModelSearch backend")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    search_runtime = init_search_runtime()
    port = args.port if args.port is not None else int(os.environ.get("PORT", "5002"))
    app.run(host="0.0.0.0", port=port, debug=False)
