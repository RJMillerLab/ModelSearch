#!/usr/bin/env python3
"""Experiment batch runner for query2modelcard -> nugget evaluation.

Two entry points are supported:

1. ``--queries-file``: run the backend query2modelcard/integration path first,
   save a reusable intermediate JSON, then evaluate it.
2. ``--jobs-json``: reuse an existing intermediate JSON and only run the nugget
   stages.

After the intermediate JSON exists, both paths share the same stages:

1. modelcard2nugget/card2nugget runs in batch for missing model cards only;
2. query2nugget runs in batch over all queries;
3. each query/method/top-k group filters its candidate nugget CSVs.

All outputs go into a new run directory so existing frontend/wrap artifacts are
not overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Literal

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.config import JOBS_DIR, OUTPUT_DIR
from src.evaluate.card2nugget_extraction import CARD2NUGGET_DIR, _safe_model_id, run_batch as run_card2nugget_batch
from src.evaluate.nugget_schema import NUGGET_SCHEMA_HEADERS
from src.evaluate.query2nugget_mapping import map_queries
from src.evaluate.query2nugget_mapping_batch import map_queries_batch
from src.evaluate.query2nugget_match import (
    _row_dict,
    build_qrels_and_run_llm_rerank,
    build_qrels_and_run_structured,
    count_csv_data_rows,
)
from src.utils.batch_run_preset_queries import (
    _load_queries_from_json_or_jsonl,
    _print_integration_batch_summary,
    run_one_query,
)

MODEL_METHODS = ("sparse", "dense", "hybrid")
TABLE_METHODS = ("keyword", "single_column", "unionable")
METHOD_ORDER = MODEL_METHODS + TABLE_METHODS
DEFAULT_TOP_K = (1, 3, 5, 10)


def _safe_name(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", (text or "").strip())
    return s.strip("_") or "item"


def _unique_keep(items: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _job_artifact_dir(item: dict[str, Any], jobs_root: Path) -> Path:
    search_resp = item.get("search_response") if isinstance(item.get("search_response"), dict) else {}
    folder_path = str(search_resp.get("folder_path", "")).strip()
    if folder_path:
        return jobs_root / Path(folder_path).name
    return jobs_root / str(item.get("job_id", "")).strip()


def _first_n(items: list[str], n: int) -> list[str]:
    return items[: max(0, n)] if n > 0 else items


def _ids_from_payload(payload: dict[str, Any], *, prefer_models_with_tables: bool = False) -> list[str]:
    if prefer_models_with_tables and isinstance(payload.get("models_with_tables"), list):
        return _unique_keep(payload.get("models_with_tables"))
    for key in ("model_ids", "models_with_tables", "model_rerank_map"):
        if isinstance(payload.get(key), list):
            return _unique_keep(payload.get(key))
    return []


def _table_ids_from_summary(item: dict[str, Any], method: str, preferred_sources: list[str]) -> tuple[list[str], str, str]:
    table_search = item.get("integration_table_search")
    if not isinstance(table_search, dict):
        return [], "", ""
    per_source = table_search.get(method)
    if not isinstance(per_source, dict):
        return [], "", ""
    source_order = list(preferred_sources) + [s for s in per_source if s not in preferred_sources]
    for source in source_order:
        payload = per_source.get(source)
        if not isinstance(payload, dict):
            continue
        ids = _ids_from_payload(payload, prefer_models_with_tables=True)
        if ids:
            status = str(payload.get("status", "")).strip() or "success"
            note = f"tables_source={source}"
            return ids, status, note
    return [], "", ""


def _table_ids_from_job_file(job_dir: Path, method: str) -> list[str]:
    payload = _load_json_if_exists(job_dir / f"card2tab2card_{method}.json")
    if isinstance(payload.get("model_rerank_map"), list):
        return _unique_keep(payload.get("model_rerank_map"))
    if isinstance(payload.get("tab2card_map"), dict):
        flat: list[str] = []
        for values in payload["tab2card_map"].values():
            if isinstance(values, list):
                flat.extend(values)
        return _unique_keep(flat)
    return []


def extract_method_model_sets(
    item: dict[str, Any],
    *,
    jobs_root: Path,
    preferred_table_sources: list[str],
    model_cap: int,
) -> list[dict[str, Any]]:
    job_dir = _job_artifact_dir(item, jobs_root)
    search_resp = item.get("search_response") if isinstance(item.get("search_response"), dict) else {}
    job_model_cap = int(search_resp.get("model_top_k", 3) or 3)
    effective_model_cap = max(0, max(job_model_cap, model_cap))
    out_by_method: dict[str, dict[str, Any]] = {}

    ims = item.get("integration_model_search")
    if isinstance(ims, dict):
        for method, payload in ims.items():
            if not isinstance(payload, dict):
                continue
            model_ids = _first_n(_ids_from_payload(payload), effective_model_cap)
            status = str(payload.get("status", "")).strip() or ("success" if model_ids else "")
            note = str(payload.get("message", "")).strip()
            method_name = str(method).strip()
            if method_name and (model_ids or status or note):
                out_by_method[method_name] = {"method": method_name, "model_ids": model_ids, "status": status, "note": note}

    q2mc = _load_json_if_exists(job_dir / "query2modelcard.json")
    q2mc_results = q2mc.get("results", {}) if isinstance(q2mc.get("results"), dict) else {}
    for method in MODEL_METHODS:
        if method not in out_by_method and isinstance(q2mc_results.get(method), list):
            ids = _first_n(_unique_keep(q2mc_results.get(method)), effective_model_cap)
            out_by_method[method] = {"method": method, "model_ids": ids, "status": "success" if ids else "", "note": ""}

    for method in TABLE_METHODS:
        ids, status, note = _table_ids_from_summary(item, method, preferred_table_sources)
        if not ids:
            ids = _first_n(_table_ids_from_job_file(job_dir, method), effective_model_cap)
            status = "success" if ids else ""
            note = "from_job_file" if ids else ""
        if ids:
            out_by_method[method] = {"method": method, "model_ids": _first_n(ids, effective_model_cap), "status": status, "note": note}

    return [out_by_method[m] for m in METHOD_ORDER if m in out_by_method]


def run_query2modelcard_backend(
    queries_file: Path,
    output_path: Path,
    *,
    backend_url: str,
    limit_jobs: int | None,
    backend_top_k: int,
    backend_model_top_k: int,
    backend_table_search_k: int,
    backend_use_by_type: bool,
    backend_run_integration: bool,
    backend_integration_type: str,
    backend_integration_k: int,
    backend_integration_max_models: int | None,
    backend_integration_model_modes: list[str],
    backend_integration_search_types: list[str],
    backend_integration_tables_sources: list[str],
) -> Path:
    queries = _load_queries_from_json_or_jsonl(str(queries_file))
    if limit_jobs is not None:
        queries = queries[: max(0, limit_jobs)]
    if not queries:
        raise RuntimeError(f"No valid queries found in {queries_file}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    backend_url = backend_url.rstrip("/")
    all_outcomes: list[dict[str, Any]] = []
    print(f"[backend] queries={len(queries)} from {queries_file}")
    print(f"[backend] url={backend_url} integration={backend_run_integration}")

    for idx, q in enumerate(queries, start=1):
        q_id = str(q.get("id") or f"q{idx}")
        q_text = str(q.get("query") or "").strip()
        print(f"[backend] {idx}/{len(queries)} id={q_id!r} query={q_text!r}")
        outcome = run_one_query(
            backend_url=backend_url,
            query=q_text,
            top_k=backend_top_k,
            model_top_k=backend_model_top_k,
            table_search_k=backend_table_search_k,
            use_by_type=backend_use_by_type,
            run_integration=backend_run_integration,
            integration_type=backend_integration_type,
            integration_k=backend_integration_k,
            integration_max_models=backend_integration_max_models,
            integration_model_modes=(backend_integration_model_modes or None),
            integration_search_types=(backend_integration_search_types or None),
            integration_tables_sources=(backend_integration_tables_sources or None),
        )
        all_outcomes.append({"id": q_id, "query": q_text, **outcome})
        output_path.write_text(json.dumps(all_outcomes, ensure_ascii=False, indent=2), encoding="utf-8")

    if backend_run_integration:
        _print_integration_batch_summary(all_outcomes)
    print(f"[backend] saved_intermediate={output_path.resolve()}")
    return output_path


def load_jobs(
    jobs_json: Path,
    *,
    limit_jobs: int | None,
    preferred_table_sources: list[str],
    model_cap: int,
    jobs_root: Path | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(jobs_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("queries", [])
    if not isinstance(payload, list):
        raise ValueError(f"Expected list or {{'queries': [...]}} in {jobs_json}")
    jobs_root = jobs_root or jobs_json.parent.parent
    jobs: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        job_id = str(item.get("job_id", "")).strip()
        query = str(item.get("query", "")).strip()
        if not job_id or not query:
            continue
        method_model_sets = extract_method_model_sets(
            item,
            jobs_root=jobs_root,
            preferred_table_sources=preferred_table_sources,
            model_cap=model_cap,
        )
        model_ids = _unique_keep(mid for method in method_model_sets for mid in method.get("model_ids", []))
        if model_ids:
            jobs.append({"job_id": job_id, "query": query, "method_model_sets": method_model_sets, "model_ids": model_ids})
        if limit_jobs and len(jobs) >= limit_jobs:
            break
    return jobs


def existing_card_output(model_id: str) -> dict[str, str] | None:
    csv_path = CARD2NUGGET_DIR / f"{_safe_model_id(model_id)}.csv"
    meta_path = CARD2NUGGET_DIR / f"{_safe_model_id(model_id)}_meta.yaml"
    if csv_path.is_file():
        return {"model_id": model_id, "csv_path": str(csv_path.resolve()), "meta_path": str(meta_path.resolve()), "note": "exists_skip"}
    return None


def ensure_card2nuggets(
    model_ids: list[str],
    *,
    llm_mode: Literal["batch", "iter"],
    max_new_cards: int | None,
) -> dict[str, dict[str, str]]:
    outputs: dict[str, dict[str, str]] = {}
    to_run: list[str] = []
    for model_id in model_ids:
        existing = existing_card_output(model_id)
        if existing:
            outputs[model_id] = existing
        else:
            to_run.append(model_id)
    if max_new_cards is not None:
        to_run = to_run[: max(0, max_new_cards)]
    if to_run:
        print(f"[card2nugget] creating {len(to_run)} missing model CSVs with llm_mode={llm_mode}")
        for row in run_card2nugget_batch(to_run, llm_mode=llm_mode):
            outputs[str(row["model_id"])] = row
    return outputs


def row_signature(cells: dict[str, str]) -> tuple[str, ...]:
    return tuple((cells.get(h, "") or "").strip() for h in NUGGET_SCHEMA_HEADERS)


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["doc_id", "table", "row_idx", *NUGGET_SCHEMA_HEADERS]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cells = row.get("cells", {}) if isinstance(row.get("cells"), dict) else {}
            out = {
                "doc_id": row.get("doc_id", ""),
                "table": row.get("table", ""),
                "row_idx": row.get("row_idx", ""),
            }
            out.update({h: cells.get(h, "") for h in NUGGET_SCHEMA_HEADERS})
            writer.writerow(out)


def write_card_nuggets_csv(path: Path, model_ids: list[str], csv_paths: list[Path]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    dedup: dict[tuple[str, ...], dict[str, str]] = {}
    provenance: dict[tuple[str, ...], list[str]] = {}
    for model_id, csv_path in zip(model_ids, csv_paths):
        if not csv_path.is_file():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cells = _row_dict(row)
                sig = row_signature(cells)
                if sig not in dedup:
                    dedup[sig] = {h: cells.get(h, "") for h in NUGGET_SCHEMA_HEADERS}
                    provenance[sig] = []
                if model_id not in provenance[sig]:
                    provenance[sig].append(model_id)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_model_ids", *NUGGET_SCHEMA_HEADERS])
        writer.writeheader()
        for sig, row in dedup.items():
            out = {"source_model_ids": " | ".join(provenance.get(sig, []))}
            out.update(row)
            writer.writerow(out)
    return len(dedup)


def score_method(
    query_map: dict[str, Any],
    csv_paths: list[Path],
    *,
    match_build: Literal["structured", "llm_rerank"],
    rerank_model: str | None,
) -> dict[str, Any]:
    builder = build_qrels_and_run_llm_rerank if match_build == "llm_rerank" else build_qrels_and_run_structured
    qrels_rows, run_rows, debug = builder([query_map], csv_paths, model=rerank_model, emit_match_report=False)
    hits = debug[0].get("hits", []) if debug else []
    signatures = {
        row_signature(h.get("cells", {}))
        for h in hits
        if isinstance(h, dict) and isinstance(h.get("cells"), dict)
    }
    return {
        "hit_rows": len(hits),
        "hit_dedup": len(signatures),
        "qrels_lines": len(qrels_rows),
        "run_lines": len(run_rows),
        "hits": hits,
    }


def rank_methods(scores: dict[str, dict[str, Any]], *, score_field: str) -> list[dict[str, Any]]:
    ordered = sorted(
        ({"method": method, "score": float(payload.get(score_field, 0))} for method, payload in scores.items()),
        key=lambda x: (-x["score"], METHOD_ORDER.index(x["method"]) if x["method"] in METHOD_ORDER else 999),
    )
    rank = 0
    prev_score: float | None = None
    for i, row in enumerate(ordered, start=1):
        if prev_score is None or row["score"] != prev_score:
            rank = i
            prev_score = row["score"]
        row["rank"] = rank
    return ordered


def aggregate_records(records: list[dict[str, Any]], *, top_k_values: list[int], score_field: str) -> dict[str, Any]:
    mean_scores: dict[str, dict[str, float]] = {method: {} for method in METHOD_ORDER}
    raw_card_rows_mean: dict[str, dict[str, float]] = {method: {} for method in METHOD_ORDER}
    win_tie_loss: dict[str, dict[str, dict[str, int]]] = {method: {} for method in TABLE_METHODS}
    method_rank_counts: dict[str, dict[str, dict[str, int]]] = {method: {} for method in METHOD_ORDER}

    for k in top_k_values:
        k_key = f"top_{k}"
        for method in METHOD_ORDER:
            vals = [
                float(r["method_scores"].get(method, {}).get(k_key, {}).get(score_field, 0))
                for r in records
            ]
            raw_vals = [
                float(r["method_scores"].get(method, {}).get(k_key, {}).get("card_rows", 0))
                for r in records
            ]
            mean_scores[method][k_key] = sum(vals) / len(vals) if vals else 0.0
            raw_card_rows_mean[method][k_key] = sum(raw_vals) / len(raw_vals) if raw_vals else 0.0
            method_rank_counts[method][k_key] = {}
        for r in records:
            best_semantic = max(
                float(r["method_scores"].get(method, {}).get(k_key, {}).get(score_field, 0))
                for method in MODEL_METHODS
            )
            for method in TABLE_METHODS:
                score = float(r["method_scores"].get(method, {}).get(k_key, {}).get(score_field, 0))
                label = "win" if score > best_semantic else "tie" if score == best_semantic else "loss"
                win_tie_loss[method].setdefault(k_key, {"win": 0, "tie": 0, "loss": 0})
                win_tie_loss[method][k_key][label] += 1
            for row in r["rankings"].get(k_key, []):
                rank_key = str(row["rank"])
                method_rank_counts[row["method"]][k_key][rank_key] = method_rank_counts[row["method"]][k_key].get(rank_key, 0) + 1

    return {
        "num_queries": len(records),
        "top_k_values": top_k_values,
        "score_field": score_field,
        "method_order": list(METHOD_ORDER),
        "model_methods": list(MODEL_METHODS),
        "table_methods": list(TABLE_METHODS),
        "mean_scores": mean_scores,
        "mean_card_rows": raw_card_rows_mean,
        "win_tie_loss_vs_best_semantic": win_tie_loss,
        "method_rank_counts": method_rank_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backend/query2modelcard -> nugget batch evaluation.")
    parser.add_argument("--queries-file", type=Path, default=None, help="JSON/JSONL queries file. Runs backend first and saves an intermediate JSON.")
    parser.add_argument("--jobs-json", type=Path, default=None, help="Existing query2modelcard intermediate JSON to reuse.")
    parser.add_argument("--output-root", type=Path, default=Path(OUTPUT_DIR) / "evaluate" / "query2modelcard_nugget_batch")
    parser.add_argument("--run-id", default=None, help="Run folder name. Default: timestamp.")
    parser.add_argument("--limit-jobs", type=int, default=None, help="Smoke-test only the first N jobs.")
    parser.add_argument("--top-k", type=int, nargs="+", default=list(DEFAULT_TOP_K), help="Top-k budgets to score.")
    parser.add_argument("--query-llm-mode", choices=["batch", "iter"], default="batch")
    parser.add_argument("--card-llm-mode", choices=["batch", "iter"], default="batch")
    parser.add_argument("--match-build", choices=["structured", "llm_rerank"], default="structured")
    parser.add_argument("--model", default=None, help="OpenAI model override for query2nugget/filter.")
    parser.add_argument("--max-new-card2nugget", type=int, default=None, help="Safety cap for newly generated card2nugget CSVs.")
    parser.add_argument("--score-field", choices=["hit_rows", "hit_dedup"], default="hit_dedup")
    parser.add_argument("--table-sources", nargs="+", default=["all_from_modelcards", "intermediate"])
    parser.add_argument("--no-method-csvs", action="store_true", help="Do not write per-job method nugget CSVs.")
    parser.add_argument("--backend-url", default="http://localhost:5002", help="Backend URL for --queries-file mode.")
    parser.add_argument("--backend-top-k", type=int, default=100, help="Backend /api/search top_k.")
    parser.add_argument("--backend-model-top-k", type=int, default=3, help="Backend /api/search model_top_k.")
    parser.add_argument("--backend-table-search-k", type=int, default=3, help="Backend /api/search table_search_k.")
    parser.add_argument("--backend-use-by-type", action="store_true", help="Enable backend table type classification.")
    parser.add_argument("--skip-backend-integration", action="store_true", help="Only run backend search, not integration.")
    parser.add_argument("--backend-integration-type", default="alite", choices=["union", "intersection", "alite", "outer_join"])
    parser.add_argument("--backend-integration-k", type=int, default=10)
    parser.add_argument("--backend-integration-max-models", type=int, default=None)
    parser.add_argument("--backend-integration-model-modes", nargs="+", default=[])
    parser.add_argument("--backend-integration-search-types", nargs="+", default=[])
    parser.add_argument("--backend-integration-tables-sources", nargs="+", default=["intermediate", "all_from_modelcards"])
    parser.add_argument("--backend-intermediate-output", type=Path, default=None, help="Optional path for saved backend intermediate JSON.")
    args = parser.parse_args()

    if bool(args.queries_file) == bool(args.jobs_json):
        parser.error("Provide exactly one of --queries-file or --jobs-json.")

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / run_id
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)

    jobs_root: Path | None = None
    requested_top_k_cap = max([int(k) for k in args.top_k if int(k) > 0] or [1])
    if args.queries_file:
        effective_backend_model_top_k = max(int(args.backend_model_top_k), requested_top_k_cap)
        if effective_backend_model_top_k != int(args.backend_model_top_k):
            print(
                f"[backend] bump backend_model_top_k {args.backend_model_top_k} -> {effective_backend_model_top_k} "
                f"to cover requested top-k={requested_top_k_cap}"
            )
        jobs_json = run_query2modelcard_backend(
            args.queries_file,
            args.backend_intermediate_output or (out_dir / "query2modelcard_backend_intermediate.json"),
            backend_url=args.backend_url,
            limit_jobs=args.limit_jobs,
            backend_top_k=args.backend_top_k,
            backend_model_top_k=effective_backend_model_top_k,
            backend_table_search_k=args.backend_table_search_k,
            backend_use_by_type=bool(args.backend_use_by_type),
            backend_run_integration=not bool(args.skip_backend_integration),
            backend_integration_type=args.backend_integration_type,
            backend_integration_k=args.backend_integration_k,
            backend_integration_max_models=args.backend_integration_max_models,
            backend_integration_model_modes=args.backend_integration_model_modes,
            backend_integration_search_types=args.backend_integration_search_types,
            backend_integration_tables_sources=args.backend_integration_tables_sources,
        )
        jobs_root = Path(JOBS_DIR)
    else:
        jobs_json = args.jobs_json

    jobs = load_jobs(
        jobs_json,
        limit_jobs=args.limit_jobs,
        preferred_table_sources=args.table_sources,
        model_cap=requested_top_k_cap,
        jobs_root=jobs_root,
    )
    if not jobs:
        raise RuntimeError(f"No valid jobs found in {jobs_json}")
    print(f"[load] jobs={len(jobs)} from {jobs_json}")

    queries = [j["query"] for j in jobs]
    query_maps = map_queries_batch(queries, model=args.model) if args.query_llm_mode == "batch" else map_queries(queries, model=args.model)
    query_maps_path = out_dir / "query_maps.json"
    query_maps_path.write_text(json.dumps({"queries": query_maps}, ensure_ascii=False, indent=2), encoding="utf-8")

    all_model_ids = _unique_keep(mid for job in jobs for mid in job["model_ids"])
    card_outputs = ensure_card2nuggets(
        all_model_ids,
        llm_mode=args.card_llm_mode,
        max_new_cards=args.max_new_card2nugget,
    )
    (out_dir / "card_outputs.json").write_text(json.dumps(card_outputs, ensure_ascii=False, indent=2), encoding="utf-8")

    top_k_values = sorted({int(k) for k in args.top_k if int(k) > 0})
    max_k = max(top_k_values)
    records: list[dict[str, Any]] = []
    per_query_path = out_dir / "per_query_method_scores.jsonl"
    with open(per_query_path, "w", encoding="utf-8") as f:
        for i, (job, query_map) in enumerate(zip(jobs, query_maps), start=1):
            job_id = job["job_id"]
            method_scores: dict[str, dict[str, Any]] = {}
            for method_info in job["method_model_sets"]:
                method = str(method_info["method"])
                if method not in METHOD_ORDER:
                    continue
                model_ids = _unique_keep(method_info.get("model_ids", []))
                method_scores.setdefault(method, {})
                for k in top_k_values:
                    selected_ids = model_ids[:k]
                    csv_paths = [
                        Path(card_outputs[mid]["csv_path"])
                        for mid in selected_ids
                        if mid in card_outputs and Path(card_outputs[mid]["csv_path"]).is_file()
                    ]
                    score = score_method(query_map, csv_paths, match_build=args.match_build, rerank_model=args.model)
                    score.update(
                        {
                            "model_ids": selected_ids,
                            "csv_paths": [str(p.resolve()) for p in csv_paths],
                            "model_count": len(selected_ids),
                            "csv_count": len(csv_paths),
                            "card_rows": sum(count_csv_data_rows(p) for p in csv_paths),
                        }
                    )
                    method_scores[method][f"top_{k}"] = {key: val for key, val in score.items() if key != "hits"}
                    if (not args.no_method_csvs) and k == max_k:
                        method_dir = out_dir / "method_nuggets" / _safe_name(job_id)
                        card_csv = method_dir / f"{method}_top{k}_card_nuggets.csv"
                        hit_csv = method_dir / f"{method}_top{k}_query_hits.csv"
                        write_card_nuggets_csv(card_csv, selected_ids, csv_paths)
                        write_rows_csv(hit_csv, score["hits"])
                        method_scores[method][f"top_{k}"]["card_nuggets_csv"] = str(card_csv.resolve())
                        method_scores[method][f"top_{k}"]["query_hits_csv"] = str(hit_csv.resolve())

            rankings = {
                f"top_{k}": rank_methods(
                    {method: scores.get(f"top_{k}", {}) for method, scores in method_scores.items()},
                    score_field=args.score_field,
                )
                for k in top_k_values
            }
            record = {
                "job_id": job_id,
                "query": job["query"],
                "query_map": query_map,
                "method_model_sets": job["method_model_sets"],
                "method_scores": method_scores,
                "rankings": rankings,
            }
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[score] {i}/{len(jobs)} job_id={job_id}")

    aggregate = aggregate_records(records, top_k_values=top_k_values, score_field=args.score_field)
    summary = {
        "config": {
            "queries_file": str(args.queries_file.resolve()) if args.queries_file else "",
            "jobs_json": str(jobs_json.resolve()),
            "jobs_root": str(jobs_root.resolve()) if jobs_root else "",
            "output_dir": str(out_dir.resolve()),
            "query_llm_mode": args.query_llm_mode,
            "card_llm_mode": args.card_llm_mode,
            "match_build": args.match_build,
            "score_field": args.score_field,
            "table_sources": args.table_sources,
            "backend_url": args.backend_url if args.queries_file else "",
            "backend_integration": bool(args.queries_file and not args.skip_backend_integration),
        },
        "aggregate": aggregate,
        "paths": {
            "query_maps": str(query_maps_path.resolve()),
            "card_outputs": str((out_dir / "card_outputs.json").resolve()),
            "per_query_method_scores": str(per_query_path.resolve()),
        },
    }
    summary_path = out_dir / "aggregate_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] summary={summary_path.resolve()}")


if __name__ == "__main__":
    main()
