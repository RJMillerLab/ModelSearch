#!/usr/bin/env python3
"""
Batch runner for ModelSearch Demo preset queries.

Simulates the frontend behavior:
  1) Reads queries from docs/preset_queries.json
  2) For each query, calls /api/search on the backend
  3) Waits for the job to finish
  4) Optionally runs table-search + model-search integrations

Usage (from repo root):
  python scripts/batch_run_preset_queries.py \
    --backend_url http://localhost:5002 \
    --preset_path docs/preset_queries.json \
    --run_integration

Note:
  - Requires the backend server (src/demo/backend.py) to be running.
  - Jobs and outputs are stored under jobs_251117/<job_id> by the backend.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

import requests

from src.config import CARD2TAB2CARD_TYPES, JOBS_DIR, QUERY2MODELCARD_RETRIEVAL_MODES


def _load_preset_queries(preset_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(preset_path):
        raise FileNotFoundError(f"preset queries file not found: {preset_path}")
    with open(preset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # support both {"queries": [...]} and plain list
    if isinstance(data, dict):
        queries = data.get("queries", [])
    else:
        queries = data
    if not isinstance(queries, list):
        raise ValueError(f"Invalid preset queries format in {preset_path}")
    return queries


def _load_queries_from_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"queries file not found: {path}")
    if path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                query = str(obj.get("query") or obj.get("rewritten_query") or "").strip()
                if not query and isinstance(obj.get("response_text"), str):
                    try:
                        parsed = json.loads(obj["response_text"])
                        if isinstance(parsed, dict):
                            query = str(parsed.get("query", "")).strip()
                    except Exception:
                        pass
                if not query:
                    continue
                rows.append({"id": str(obj.get("id") or obj.get("custom_id") or f"q{i}"), "query": query})
        return rows
    return _load_preset_queries(path)


def _post_json(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _poll_results(backend_url: str, job_id: str, poll_interval: float = 5.0, timeout: int = 3600) -> Dict[str, Any]:
    """Poll /api/results/<job_id> until completed or error."""
    results_url = f"{backend_url}/api/results/{job_id}"
    t0 = time.time()
    while True:
        # Respect total timeout
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Polling results for job {job_id} timed out after {timeout} seconds")

        resp = requests.get(results_url, timeout=60)
        # 202 = still running; 200 = success or error payload
        if resp.status_code == 202:
            time.sleep(poll_interval)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"Unexpected status while polling results for job {job_id}: {resp.status_code} {resp.text}")

        data = resp.json()
        status = data.get("status")
        if status in ("success", "completed"):
            return data
        if status == "error":
            # Backend already includes message/error
            return data
        # Fallback: still not ready, wait and retry
        time.sleep(poll_interval)


def run_one_query(
    backend_url: str,
    query: str,
    mode: str = "query",
    top_k: int = 100,
    model_top_k: int = 3,
    table_search_k: int = 3,
    use_by_type: bool = False,
    run_integration: bool = False,
    integration_type: str = "alite",
    integration_k: int = 10,
    integration_max_models: Optional[int] = None,
    integration_model_modes: Optional[List[str]] = None,
    integration_search_types: Optional[List[str]] = None,
    integration_tables_sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run search (and optional integration) for a single query."""
    search_url = f"{backend_url}/api/search"

    # Build request body to match frontend's startSearch behavior (query mode).
    req_body: Dict[str, Any] = {
        "search_mode": "new",
        "mode": mode,
        "top_k": top_k,
        "model_top_k": model_top_k,
        "tab2tab_mode": "search",
        "table_search_k": table_search_k,
        "use_by_type": use_by_type,
    }
    if mode == "query":
        req_body["query"] = query
    else:
        # For completeness; not used when driving from docs/preset_queries.json
        req_body["model_id"] = query

    start = time.time()
    search_resp = _post_json(search_url, req_body)
    if search_resp.get("status") != "started":
        raise RuntimeError(f"Search did not start correctly for query={query!r}: {search_resp}")

    job_id = search_resp.get("job_id")
    if not job_id:
        raise RuntimeError(f"Backend did not return job_id for query={query!r}: {search_resp}")

    print(f"  → Started job_id={job_id}")
    result_envelope = _poll_results(backend_url, job_id)
    elapsed_search = time.time() - start

    results = result_envelope.get("results") or result_envelope
    error = results.get("error")
    if error:
        print(f"  ✖ Search error for job {job_id}: {error}")
    else:
        print(f"  ✓ Search finished for job {job_id} in {elapsed_search:.1f}s")

    outcome: Dict[str, Any] = {
        "job_id": job_id,
        "search_response": results,
        "search_elapsed_seconds": elapsed_search,
    }

    if not run_integration:
        return outcome

    # Run model-search integration (query2modelcard neighbors) and table-search integration (Card2Tab2Card),
    # mirroring frontend's runBothIntegrations behavior (simplified defaults).
    print("    ↳ Running integrations (model + table)...")
    model_int_url = f"{backend_url}/api/integrate-model-search"
    table_int_url = f"{backend_url}/api/integrate"
    effective_integration_max_models = integration_max_models
    if effective_integration_max_models is None:
        effective_integration_max_models = int(results.get("model_top_k", 5))
        print(f"      · integration max_models follows job model_top_k={effective_integration_max_models}")
    else:
        print(f"      · integration max_models override={effective_integration_max_models}")

    if not integration_model_modes:
        integration_model_modes = list(QUERY2MODELCARD_RETRIEVAL_MODES)

    model_int_res_by_mode: Dict[str, Any] = {}
    for retrieval_mode in integration_model_modes:
        model_payload = {
            "job_id": job_id,
            "integration_type": integration_type,
            "k": integration_k,
            "max_models": effective_integration_max_models,
            "query2modelcard_retrieval_mode": retrieval_mode,
        }
        try:
            res = _post_json(model_int_url, model_payload)
            model_int_res_by_mode[retrieval_mode] = res
            status = res.get("status")
            if status == "success":
                print(f"      ✓ Model Search integration success (mode={retrieval_mode})")
            else:
                print(f"      ⚠ Model Search integration status={status} (mode={retrieval_mode}) message={res.get('message')}")
        except Exception as e:
            model_int_res_by_mode[retrieval_mode] = {"status": "error", "message": str(e)}
            print(f"      ✖ Model Search integration failed (mode={retrieval_mode}): {e}")

    # Table Search integration
    if not integration_search_types:
        integration_search_types = list(CARD2TAB2CARD_TYPES)
    if not integration_tables_sources:
        integration_tables_sources = ["intermediate", "all_from_modelcards"]

    table_int_res_by_type: Dict[str, Any] = {}
    for st in integration_search_types:
        table_int_res_by_type[st] = {}
        for tables_source in integration_tables_sources:
            table_payload = {
                "job_id": job_id,
                "search_type": st,
                "integration_type": integration_type,
                "k": integration_k,
                "max_models": effective_integration_max_models,
                "tables_source": tables_source,
            }
            try:
                res = _post_json(table_int_url, table_payload)
                table_int_res_by_type[st][tables_source] = res
                status = res.get("status")
                if status == "success":
                    print(f"      ✓ Table Search integration success (search_type={st}, tables_source={tables_source})")
                else:
                    print(
                        f"      ⚠ Table Search integration status={status} "
                        f"(search_type={st}, tables_source={tables_source}) message={res.get('message')}"
                    )
            except Exception as e:
                table_int_res_by_type[st][tables_source] = {"status": "error", "message": str(e)}
                print(f"      ✖ Table Search integration failed (search_type={st}, tables_source={tables_source}): {e}")

    outcome["integration_model_search"] = model_int_res_by_mode
    outcome["integration_table_search"] = table_int_res_by_type
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-run ModelSearch Demo preset queries via backend API.",
    )
    parser.add_argument(
        "--backend_url",
        default="http://localhost:5002",
        help="Backend base URL (default: http://localhost:5002)",
    )
    parser.add_argument(
        "--preset_path",
        default=os.path.join("docs", "preset_queries.json"),
        help="Path to preset_queries.json (default: docs/preset_queries.json)",
    )
    parser.add_argument(
        "--extra_queries_path",
        default=os.path.join("data_251117", "query", "query_rewrite_polished.jsonl"),
        help="Optional extra queries file (json/jsonl). Default: data_251117/query/query_rewrite_polished.jsonl",
    )
    parser.add_argument(
        "--query_source",
        choices=["preset", "extra", "all"],
        default="preset",
        help="Which query source(s) to run: preset / extra / all.",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=0,
        help="Optional max number of queries to run (0 = all).",
    )
    parser.add_argument(
        "--query_offset",
        type=int,
        default=0,
        help="Skip the first N queries after source selection (default: 0). Useful for testing a specific query index like 596 for the 597th query.",
    )
    parser.add_argument(
        "--table_search_k",
        type=int,
        default=3,
        help="Per-table search k (frontend slider, default 2).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Model card top_k for Card2Card (frontend hidden slider, default 100).",
    )
    parser.add_argument(
        "--model_top_k",
        type=int,
        default=3,
        help="Final reranked model_top_k for query search jobs (default: 3).",
    )
    parser.add_argument(
        "--use_by_type",
        action="store_true",
        help="Enable Card2Tab2Card by_type run (frontend 'Use table type classification').",
    )
    parser.add_argument(
        "--run_integration",
        action="store_true",
        help="Also run model-search + table-search integrations for each query.",
    )
    parser.add_argument(
        "--integration_type",
        default="alite",
        choices=["union", "intersection", "alite", "outer_join"],
        help="Integration type (default: alite).",
    )
    parser.add_argument(
        "--integration_k",
        type=int,
        default=10,
        help="Integration k (row limit for integrated tables; default 10).",
    )
    parser.add_argument(
        "--integration_max_models",
        type=int,
        default=None,
        help="Optional override for integration max_models. Default: follow each job's model_top_k.",
    )
    parser.add_argument(
        "--integration_model_modes",
        nargs="+",
        default=[],
        help=(
            "Query2Card retrieval modes to integrate via /api/integrate-model-search. "
            f"Default: {' '.join(QUERY2MODELCARD_RETRIEVAL_MODES)}"
        ),
    )
    parser.add_argument(
        "--integration_search_types",
        nargs="+",
        default=[],
        help=(
            "Table search types to integrate via /api/integrate. "
            f"Default: {' '.join(CARD2TAB2CARD_TYPES)}"
        ),
    )
    parser.add_argument(
        "--integration_tables_sources",
        nargs="+",
        default=["intermediate", "all_from_modelcards"],
        help="tables_source values to integrate via /api/integrate (default: intermediate all_from_modelcards).",
    )
    parser.add_argument(
        "--run_wrap_eval",
        action="store_true",
        help="After batch search/integration, run python -m src.evaluate.wrap_card_query_eval on the saved batch summary JSON.",
    )
    parser.add_argument(
        "--wrap_llm_mode",
        choices=["batch", "iter"],
        default="iter",
        help="llm-mode passed to wrap_card_query_eval when --run_wrap_eval is enabled.",
    )

    args = parser.parse_args()

    backend_url = args.backend_url.rstrip("/")
    preset_path = args.preset_path
    extra_queries_path = args.extra_queries_path

    print(f"Backend URL: {backend_url}")
    print(f"Preset path: {preset_path}")
    print(f"Extra queries path: {extra_queries_path}")
    print(f"Query source: {args.query_source}")
    print(f"Run integration: {bool(args.run_integration)}")

    try:
        presets_default = _load_preset_queries(preset_path)
        presets_extra = _load_queries_from_json_or_jsonl(extra_queries_path) if os.path.exists(extra_queries_path) else []
        if args.query_source == "extra":
            presets = presets_extra
        elif args.query_source == "all":
            presets = list(presets_default) + list(presets_extra)
        else:
            presets = presets_default
    except Exception as e:
        print(f"Failed to load preset queries: {e}")
        return 1

    if args.query_offset and args.query_offset > 0:
        presets = presets[args.query_offset :]

    if args.max_queries and args.max_queries > 0:
        presets = presets[: args.max_queries]

    if not presets:
        print("No preset queries to run.")
        return 0

    print(f"Total queries to run: {len(presets)}\n")

    all_outcomes: List[Dict[str, Any]] = []
    for idx, q in enumerate(presets, 1):
        q_id = q.get("id") or f"q{idx}"
        q_text = q.get("query") or ""
        print(f"[{idx}/{len(presets)}] Running query id={q_id!r}: {q_text!r}")
        try:
            outcome = run_one_query(
                backend_url=backend_url,
                query=q_text,
                mode="query",
                top_k=args.top_k,
                model_top_k=args.model_top_k,
                table_search_k=args.table_search_k,
                use_by_type=bool(args.use_by_type),
                run_integration=bool(args.run_integration),
                integration_type=args.integration_type,
                integration_k=args.integration_k,
                integration_max_models=args.integration_max_models,
                integration_model_modes=(args.integration_model_modes or None),
                integration_search_types=(args.integration_search_types or None),
                integration_tables_sources=(args.integration_tables_sources or None),
            )
            all_outcomes.append({"id": q_id, "query": q_text, **outcome})
        except Exception as e:
            print(f"  ✖ Failed to run query {q_id!r}: {e}")
            all_outcomes.append({"id": q_id, "query": q_text, "error": str(e)})
        print("")

    # Optionally dump a summary JSON under jobs_251117/batch_runs for inspection.
    summary_path = ""
    try:
        summary_dir = os.path.join(JOBS_DIR, "batch_runs")
        os.makedirs(summary_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(summary_dir, f"batch_preset_queries_{ts}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_outcomes, f, ensure_ascii=False, indent=2)
        print(f"Batch summary saved to: {summary_path}")
    except Exception as e:
        print(f"Warning: failed to save batch summary: {e}")

    if args.run_wrap_eval and summary_path:
        try:
            cmd = [
                "python",
                "-m",
                "src.evaluate.wrap_card_query_eval",
                "--jobs-json",
                summary_path,
                "--all-job-ids",
                "--llm-mode",
                args.wrap_llm_mode,
            ]
            print("Running wrap_card_query_eval on saved batch summary...")
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Warning: wrap_card_query_eval failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
