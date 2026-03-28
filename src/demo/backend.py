"""
Current backend entrypoint.

Routes and runtime come from ``backend_simplified``.
The old large backend lives in ``backend_depre.py``.
"""

from src.demo.backend_simplified import app, init_search_runtime, make_table_page_response


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="ModelSearch backend")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    init_search_runtime()
    port = args.port if args.port is not None else int(os.environ.get("PORT", "5002"))
    app.run(host="0.0.0.0", port=port, debug=False)
