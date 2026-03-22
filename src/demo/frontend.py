"""
Frontend for ModelSearch Demo

Simple web interface to compare Card2Card vs Card2Tab2Card search pipelines.
"""

import os
import sys
import json
import requests
from flask import (
    Flask,
    render_template_string,
    jsonify,
    send_from_directory,
    Response,
    request,
    stream_with_context,
)
from flask_cors import CORS

# static_folder=None: Flask's default /static/<path> would otherwise compete with our
# /static/app.js route and can serve raw app.js ({{BACKEND_URL}} never replaced).
app = Flask(__name__, static_folder=None)
CORS(app)

# Get the project root directory (assuming frontend.py is in src/demo/)
from src.config import REPO_ROOT

# Backend process URL (this app proxies /api/* here when MODELSEARCH_FRONTEND_API_PROXY is on).
API_UPSTREAM = os.environ.get("MODELSEARCH_API_UPSTREAM", "http://127.0.0.1:5002").rstrip("/")
_USE_API_PROXY = os.environ.get("MODELSEARCH_FRONTEND_API_PROXY", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
# Injected into HTML/JS: "" => same-origin /api/... (proxied). Full URL => browser calls API directly.
CLIENT_BACKEND_URL = "" if _USE_API_PROXY else API_UPSTREAM

# Raw template with {{BACKEND_URL}} placeholder (for single-server deploy; backend uses this with "").
RAW_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ModelSearch Demo</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1560px;
            margin: 0 auto;
            padding: 12px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 16px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .dashboard-layout {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            gap: 16px;
            align-items: start;
        }
        @media (max-width: 1100px) {
            .dashboard-layout {
                grid-template-columns: 1fr;
            }
        }
        .dashboard-col-scroll {
            max-height: calc(100vh - 88px);
            overflow-y: auto;
            overflow-x: hidden;
            min-width: 0;
            padding-right: 4px;
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
        }
        .input-section {
            margin-bottom: 12px;
        }
        .mode-selector {
            margin-bottom: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        .mode-option {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .mode-option input[type="radio"] {
            margin-right: 8px;
        }
        .mode-option label {
            margin: 0;
            font-weight: normal;
            cursor: pointer;
        }
        .mode-input {
            margin-top: 15px;
            display: none;
        }
        .mode-input.active {
            display: block;
        }
        .form-row {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 6px;
        }
        .form-row label {
            margin-bottom: 0;
            min-width: 140px;
        }
        .form-row .form-control {
            flex: 1;
            min-width: 160px;
            padding: 6px 10px;
            font-size: 13px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .form-row select.form-control { width: auto; max-width: 320px; }
        .form-row input[type="number"].form-control { width: 80px; flex: none; }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #555;
        }
        input[type="text"], input[type="number"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            box-sizing: border-box;
        }
        button {
            background: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .progress-section {
            margin-top: 12px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
            display: none;
        }
        .progress-section.active {
            display: block;
        }
        .log-container {
            max-height: 220px;
            overflow-y: auto;
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 10px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 11px;
            line-height: 1.5;
        }
        .log-entry {
            margin-bottom: 5px;
        }
        .log-timestamp {
            color: #858585;
            margin-right: 8px;
        }
        .log-message {
            color: #d4d4d4;
        }
        .results-section {
            margin-top: 12px;
            display: none;
        }
        .results-section.active {
            display: block;
        }
        .results-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            align-items: start;
        }
        .result-card {
            background: #f8f9fa;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
        }
        .result-card h3 {
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 14px;
            color: #495057;
        }
        .result-list {
            list-style: none;
            padding: 0;
        }
        .result-item {
            padding: 4px 6px;
            margin: 3px 0;
            background: white;
            border-radius: 3px;
            border-left: 3px solid #007bff;
            font-size: 13px;
            line-height: 1.45;
        }
        .result-item a {
            color: #0056b3;
            text-decoration: none;
        }
        .result-item a:hover {
            text-decoration: underline;
        }
        .comparison-section {
            margin-top: 20px;
            padding: 20px;
            background: #e7f3ff;
            border-radius: 4px;
        }
        .comparison-item {
            margin: 10px 0;
            padding: 10px;
            background: white;
            border-radius: 4px;
        }
        .expand-toggle {
            cursor: pointer;
            color: #007bff;
            font-weight: 600;
            padding: 4px 10px;
            margin-right: 6px;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            user-select: none;
            transition: transform 0.2s, background 0.15s;
            font-size: 12px;
            background: #e7f3ff;
            border: 1px solid #007bff;
            border-radius: 4px;
        }
        .expand-toggle:hover {
            color: #0056b3;
            background: #cce5ff;
        }
        .expand-toggle.expanded {
            transform: rotate(90deg);
        }
        .expand-toggle .expand-label { margin-left: 2px; }
        .collapsible-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }
        .collapsible-content.expanded {
            max-height: 5000px;
        }
        .search-type-section {
            margin: 4px 0;
            padding: 0;
            background: #fff;
            border-radius: 6px;
            border: 1px solid #e9ecef;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            overflow: hidden;
        }
        .search-type-header {
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 8px;
            background: linear-gradient(180deg, #fafbfc 0%, #f1f3f5 100%);
            border-bottom: 1px solid #e9ecef;
            margin-bottom: 0;
        }
        .search-type-header:hover {
            background: linear-gradient(180deg, #f1f3f5 0%, #e9ecef 100%);
        }
        .search-type-section .collapsible-content {
            padding: 8px;
        }
        .search-type-header::before {
            content: '▶';
            margin-right: 5px;
            transition: transform 0.2s;
            display: inline-block;
        }
        .search-type-header.expanded::before {
            transform: rotate(90deg);
        }
        .error {
            color: #dc3545;
            padding: 10px;
            background: #f8d7da;
            border-radius: 4px;
            margin-top: 10px;
        }
        .number-badge {
            display: inline-block;
            width: 20px;
            height: 20px;
            background: black;
            color: white;
            border-radius: 50%;
            text-align: center;
            line-height: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 5px;
            vertical-align: middle;
        }
        .pdf-section {
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .pdf-section img {
            height: 160px;
            width: auto;
            max-width: 100%;
            object-fit: contain;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            background: white;
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 style="margin: 0 0 12px 0; font-size: 22px; font-weight: 600; display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 12px;">
            <span>🔍 ModelSearch Demo</span>
            <span style="font-size: 13px; font-weight: normal; color: #555;">Compare <span class="number-badge">1</span> Card2Card vs <span class="number-badge">2</span> Card2Tab2Card</span>
        </h1>
        
        <div class="dashboard-layout">
        <div class="dashboard-col-scroll">
        <div class="input-section">
            <div style="margin-bottom: 8px; padding: 8px 10px; background: #f8f9fa; border-radius: 4px; border: 1px solid #ddd;">
                <label style="display: flex; align-items: center; cursor: pointer; gap: 6px; font-size: 13px;">
                    <input type="checkbox" id="load_previous_search" style="margin-right: 4px; width: 16px; height: 16px;">
                    <span style="font-weight: 500;">Load Previous Search</span>
                </label>
            </div>
            
            <div id="previous-search-section" style="display: none; margin-bottom: 10px;">
                <p style="margin-bottom: 4px; font-size: 12px; color: #666;">Pick a saved search</p>
                <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 4px;">
                    <select id="saved_search_select" class="form-control" style="min-width: 200px; max-width: 100%; height: 26px; font-size: 12px; padding: 2px 6px;">
                        <option value="">— select folder —</option>
                    </select>
                    <button id="loadMimicBtn" type="button" onclick="loadMimicSearch()" style="padding: 2px 10px; font-size: 12px; height: 26px;">Load Saved Results</button>
                </div>
                <div id="saved_searches_list" style="display: none;"></div>
            </div>
            
            <div id="new-search-inputs">
            <div id="diagram-section" class="pdf-section" style="margin-bottom: 10px; padding: 8px;">
                <img id="search-diagram" src="/static/fig/modelsearch.png" alt="ModelSearch Overview" style="height: 160px;" />
            </div>
            
            <div class="form-row" style="margin-top: 8px;">
                <label for="search_mode_select">Search Mode:</label>
                <select id="search_mode_select" class="form-control" onchange="toggleMode()">
                    <option value="query" selected>Query → ModelCard → Search</option>
                    <option value="modelid">ModelID → Search (direct)</option>
                </select>
            </div>
            
            <div class="mode-input active" id="query-input">
                <div class="form-row">
                    <label for="query">Query (preset / fill in):</label>
                    <select id="preset_query_select" onchange="onPresetQueryChange()" class="form-control" style="width: 200px; flex: none;">
                        <option value="">— custom —</option>
                    </select>
                    <input type="text" id="query" class="form-control" placeholder="Type or pick preset" value="Are there table foundation models that can handle small tables (≤100 rows/columns) with many missing values and produce column embeddings?">
                </div>
                <div class="form-row" id="require-seed-has-tables-row" style="margin-top: 8px;">
                    <label for="require_seed_has_tables">Table Search Seed Model (Model with Tables / Model without Tables):</label>
                    <select id="require_seed_has_tables" class="form-control" style="width: 200px; flex: none;" title="Only applies in Query mode: pick which seed model to use for Table Search.">
                        <option value="0">Use top-1 result</option>
                        <option value="1" selected>Pick first model with tables (from top-20)</option>
                    </select>
                    <span style="font-size: 11px; color: #666; margin-left: 8px;">Only in Query mode. If "Pick first model with tables" and none of top-20 have tables → Table Search skipped.</span>
                </div>
            </div>
            
            <div class="mode-input" id="modelid-input">
                <div class="form-row">
                    <label for="model_id">Model ID (direct):</label>
                    <input type="text" id="model_id" class="form-control" value="Salesforce/codet5-base" placeholder="HuggingFace model ID">
                </div>
            </div>
            
            <div class="form-row one-click-info" style="margin-bottom: 8px;">
                <details class="run-options-details" style="background: #f0f7ff; border: 2px solid #007bff; border-radius: 6px; overflow: hidden;">
                    <summary style="cursor: pointer; padding: 10px 12px; font-size: 13px; font-weight: 600; color: #004085; display: flex; align-items: center; gap: 8px; user-select: none;">
                        <span style="font-size: 16px;">ⓘ</span> Run options &amp; what gets logged
                    </summary>
                    <div style="padding: 10px 12px; font-size: 12px; color: #555; border-top: 1px solid #b8d4e8;">
                        <p style="margin: 0 0 6px 0;"><strong>One-click:</strong> Card2Card (Dense/Sparse/Hybrid), Card2Tab2Card (single_column, keyword, unionable).</p>
                        <p style="margin: 0 0 6px 0;"><strong>Log:</strong> Run settings (top_k, per_table_search_k, card2card mode, require_seed_has_tables) appear at start. Time ⏱️ per step.</p>
                        <p style="margin: 0;">When loading saved results, compare runs by their settings (e.g. K and retrieval mode).</p>
                    </div>
                </details>
            </div>
            
            <div class="form-row" style="display: flex; gap: 20px; align-items: flex-end; flex-wrap: wrap;">
                <!-- Model Card Top K: hidden - left pipeline aligns to right's max; we only control Table Top K -->
                <div style="flex: 1; min-width: 200px; display: none;">
                    <label for="top_k">Model Card Top K:</label>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <input type="range" id="top_k_slider" min="1" max="100" value="100" step="1" style="flex: 1; max-width: 200px;" oninput="updateTopKValue(this.value)">
                        <input type="number" id="top_k" class="form-control" value="100" min="1" max="100" oninput="updateTopKSlider(this.value)" style="width: 80px;">
                    </div>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <label for="table_search_k">Per-table search k:</label>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <input type="range" id="table_search_k_slider" min="1" max="5" value="3" step="1" style="flex: 1; max-width: 200px;" oninput="updateTableSearchKValue(this.value)">
                        <input type="number" id="table_search_k" class="form-control" value="3" min="1" max="5" oninput="updateTableSearchKSlider(this.value)" style="width: 80px;">
                    </div>
                    <span style="font-size: 11px; color: #666;">Top k for each individual table search (1–5). Results merged, filtered, capped at 20 tables; models capped at 50.</span>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <label for="model_top_k">Model top-k (dense reranker cap):</label>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <input type="range" id="model_top_k_slider" min="1" max="20" value="5" step="1" style="flex: 1; max-width: 200px;" oninput="updateModelTopKValue(this.value)">
                        <input type="number" id="model_top_k" class="form-control" value="5" min="1" max="20" oninput="updateModelTopKSlider(this.value)" style="width: 80px;">
                    </div>
                    <span style="font-size: 11px; color: #666;">Default 5. If Card2Tab2Card has more candidates, backend uses dense reranker (seed Card2Card order), then cap.</span>
                </div>
                <div>
                    <button id="searchBtn" onclick="startSearch()" style="padding: 10px 24px; font-size: 15px; font-weight: 600;">Start Search</button>
                </div>
            </div>
            </div>
        </div>
        
        <div id="progressSection" class="progress-section">
            <h3 style="margin-bottom: 6px; font-size: 14px;">Progress Logs</h3>
            <div id="logContainer" class="log-container"></div>
        </div>
        
        <div id="errorMsg" class="error" style="display: none;"></div>
        
        <div id="resultsSection" class="results-section">
            <div id="resultsMetaStrip" style="display: none; margin-bottom: 8px; padding: 6px 10px; line-height: 1.35; font-size: 12px; color: #495057; background: #fafafa; border: 1px solid #e9ecef; border-radius: 6px;"></div>
            <h3 style="margin-top: 0; margin-bottom: 8px; font-size: 14px; font-weight: bold;">Retrieval results</h3>
            <div id="resultsContent"></div>
        </div>
        </div>
        <div class="dashboard-col-scroll">
            <div id="integrationPanelMount" style="font-size: 12px; color: #888; padding: 8px 4px;">
                Table Integration will appear here after a search completes.
            </div>
        </div>
        </div>
    </div>
    
    <script src="/static/app.js"></script>

</body>
</html>
"""
HTML_TEMPLATE = RAW_HTML_TEMPLATE.replace("{{BACKEND_URL}}", CLIENT_BACKEND_URL)


if _USE_API_PROXY:

    @app.route("/api/table-page", methods=["GET"])
    def serve_table_page_on_ui_port():
        """
        Full CSV table HTML is served here (port 5001) using the same logic as the API app.
        If :5002 is an older process without GET /api/table-page, the browser would get Werkzeug 404;
        handling on the UI process avoids that mismatch when only the UI was restarted after a git pull.
        """
        try:
            from src.demo.backend import make_table_page_response
        except Exception as e:
            return jsonify(
                {"status": "error", "message": f"Could not load table-page handler: {e}"}
            ), 500
        return make_table_page_response(request.args.get("path") or "")

    def _upstream_api_url():
        q = request.query_string.decode("utf-8") if request.query_string else ""
        return API_UPSTREAM + request.path + ("?" + q if q else "")

    @app.route("/api", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    @app.route("/api/", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    @app.route("/api/<path:path>", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    def proxy_api(path=None):
        """Forward /api/* to the ModelSearch backend (default 127.0.0.1:5002)."""
        if request.method == "OPTIONS":
            return Response(status=204)
        url = _upstream_api_url()
        is_sse = request.method == "GET" and request.path.startswith("/api/logs/")

        if is_sse:

            def generate():
                try:
                    with requests.get(url, stream=True, timeout=(10, None)) as resp:
                        for chunk in resp.iter_content(chunk_size=2048):
                            if chunk:
                                yield chunk
                except requests.exceptions.RequestException:
                    line = json.dumps(
                        {
                            "status": "error",
                            "message": f"Cannot reach API at {API_UPSTREAM}. Start: python -m src.demo.backend",
                        }
                    )
                    yield f"data: {line}\n\n".encode("utf-8")

            return Response(stream_with_context(generate()), mimetype="text/event-stream")

        headers = {}
        if request.content_type:
            headers["Content-Type"] = request.content_type
        body = None
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            body = request.get_data()
        try:
            resp = requests.request(
                request.method,
                url,
                data=body,
                headers=headers,
                timeout=(15, 3600),
            )
        except requests.exceptions.ConnectionError:
            return jsonify(
                {
                    "status": "error",
                    "message": f"Cannot reach API at {API_UPSTREAM}. In another terminal run: python -m src.demo.backend",
                }
            ), 502
        except requests.exceptions.RequestException as e:
            return jsonify({"status": "error", "message": str(e)}), 502

        ct = resp.headers.get("Content-Type")
        return Response(resp.content, status=resp.status_code, content_type=ct)


@app.route('/')
def index():
    """Serve frontend HTML"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/static/app.js')
def serve_app_js():
    """Serve app script with BACKEND_URL injected (no \\u003c hacks; script is external so </ is safe)."""
    app_js_path = os.path.join(os.path.dirname(__file__), 'static', 'app.js')
    with open(app_js_path, 'r', encoding='utf-8') as f:
        content = f.read().replace("{{BACKEND_URL}}", CLIENT_BACKEND_URL)
    return Response(content, mimetype='application/javascript')


@app.route('/static/fig/<path:filename>')
def serve_fig(filename):
    """Serve files from fig directory (PDF, PNG, etc.)"""
    FIG_DIR = os.path.join(REPO_ROOT, 'fig')
    os.makedirs(FIG_DIR, exist_ok=True)
    
    file_path = os.path.join(FIG_DIR, filename)
    if os.path.exists(file_path):
        print(f"✅ Serving file from: {file_path}")
        return send_from_directory(FIG_DIR, filename)

    print(f"❌ File not found. Tried paths: {file_path}")
    return jsonify({"error": "File not found", "filename": filename}), 404


if __name__ == '__main__':
    print("Starting ModelSearch Frontend...")
    if _USE_API_PROXY:
        print(f"Proxying /api/* -> {API_UPSTREAM} (MODELSEARCH_FRONTEND_API_PROXY=0 to disable)")
    else:
        print(f"API proxy off — browser calls {API_UPSTREAM} directly")
    print("Open http://localhost:5001 in your browser")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
