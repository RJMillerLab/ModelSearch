#!/usr/bin/env python3
"""End-to-end wrapper: card2nugget -> query2nugget -> qrels/run for two clusters."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pyndeval
from src.config import OUTPUT_DIR
from src.evaluate.card2nugget_extraction import run_batch, _safe_model_id
from src.evaluate.evaluate_pyndeval import load_run, load_subtopic_qrels, mean
from src.evaluate.query2nugget_layer_mapping import map_queries_via_batch, build_qrels_and_run

EVAL_DIR = Path(OUTPUT_DIR) / "evaluate"
BATCH_DIR = EVAL_DIR / "batch"
PIPELINE_DIR = EVAL_DIR / "pipeline"


def _load_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _collect_queries(args: argparse.Namespace) -> list[str]:
    queries: list[str] = []
    if args.query:
        queries.append(args.query.strip())
    if args.queries_file:
        queries.extend(_load_lines(Path(args.queries_file)))
    return [q for q in queries if q]


def _collect_model_ids(inline: list[str] | None, file_path: str | None) -> list[str]:
    out: list[str] = []
    if inline:
        out.extend([x.strip() for x in inline if x and x.strip()])
    if file_path:
        out.extend(_load_lines(Path(file_path)))
    seen: set[str] = set()
    uniq: list[str] = []
    for m in out:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _safe_name(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", (text or "").strip())
    return s.strip("_") or "item"


def _write_qrels(path: Path, rows: list[tuple[str, str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, subtopic, doc_id, rel in rows:
            f.write(f"{qid} {subtopic} {doc_id} {rel}\n")


def _write_run(path: Path, rows: list[tuple[str, str, str, int, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, q0, doc_id, rank, score, tag in rows:
            f.write(f"{qid} {q0} {doc_id} {rank} {score} {tag}\n")


def _map_queries(queries: list[str], model: str | None) -> list[dict[str, Any]]:
    if len(queries) == 1:
        print("[query2nugget] single query still uses Batch API")
    return map_queries_via_batch(queries, model=model)


def _extract_job_sets(batch_json: Path, wanted_job_ids: set[str] | None) -> list[dict[str, Any]]:
    payload = json.loads(batch_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    sets: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        job_id = str(item.get("job_id", "")).strip()
        if wanted_job_ids and job_id not in wanted_job_ids:
            continue
        query = str(item.get("query", "")).strip()
        sparse = ((item.get("integration_model_search") or {}).get("sparse") or {})
        model_ids_raw = sparse.get("model_ids") or sparse.get("models_with_tables") or []
        if not isinstance(model_ids_raw, list):
            model_ids_raw = []
        model_ids = [str(x).strip() for x in model_ids_raw if str(x).strip()]
        if not query or not model_ids:
            continue
        sets.append({"job_id": job_id, "query": query, "model_ids": model_ids})
    return sets


def _split_existing_batch_models(model_ids: list[str]) -> tuple[list[str], list[dict[str, str]], list[str]]:
    to_run: list[str] = []
    reused: list[dict[str, str]] = []
    skipped_ids: list[str] = []
    for model_id in model_ids:
        csv_path = BATCH_DIR / f"{_safe_model_id(model_id)}.csv"
        meta_path = BATCH_DIR / f"{_safe_model_id(model_id)}_meta.yaml"
        if csv_path.is_file():
            reused.append(
                {
                    "model_id": model_id,
                    "csv_path": str(csv_path.resolve()),
                    "meta_path": str(meta_path.resolve()),
                    "note": "exists_skip",
                }
            )
            skipped_ids.append(model_id)
        else:
            to_run.append(model_id)
    return to_run, reused, skipped_ids


def _run_cluster(
    cluster_name: str,
    model_ids: list[str],
    query_maps: list[dict[str, Any]],
    out_dir: Path,
    subtopic: str,
) -> dict[str, Any]:
    print(f"[cluster:{cluster_name}] card2nugget candidates: {len(model_ids)} model(s)")
    to_run, reused_outputs, skipped_ids = _split_existing_batch_models(model_ids)
    print(
        f"[cluster:{cluster_name}] saved_model_ids={len(skipped_ids)} "
        f"to_create={len(to_run)} total={len(model_ids)}"
    )
    if skipped_ids:
        print(f"[cluster:{cluster_name}] skipped existing model ids: {skipped_ids}")

    new_outputs = run_batch(to_run) if to_run else []
    card_outputs = reused_outputs + new_outputs
    csv_paths = [Path(x["csv_path"]) for x in card_outputs if x.get("csv_path")]
    qrels_rows, run_rows, debug = build_qrels_and_run(query_maps, csv_paths, subtopic=subtopic)

    qrels_path = out_dir / f"{cluster_name}_real_subtopic.qrels"
    run_path = out_dir / f"{cluster_name}_real_initial.run"
    debug_path = out_dir / f"{cluster_name}_query_csv_match_debug.json"
    _write_qrels(qrels_path, qrels_rows)
    _write_run(run_path, run_rows)
    _save_json(debug_path, {"queries": debug, "csv_paths": [str(p) for p in csv_paths]})

    return {
        "cluster": cluster_name,
        "models": model_ids,
        "card_outputs": card_outputs,
        "qrels_path": str(qrels_path.resolve()),
        "run_path": str(run_path.resolve()),
        "debug_path": str(debug_path.resolve()),
        "qrels_lines": len(qrels_rows),
        "run_lines": len(run_rows),
        "existing_skipped": len(skipped_ids),
        "newly_created": len(new_outputs),
    }


def _evaluate_cluster(run_path: Path, qrels_path: Path, *, cutoff: int, alpha: float, per_query: bool) -> dict[str, Any]:
    run = load_run(run_path)
    qrels = load_subtopic_qrels(qrels_path)
    if not run or not qrels:
        return {
            "skipped": True,
            "reason": "empty_run_or_qrels",
            "run_rows": len(run),
            "qrels_rows": len(qrels),
        }

    measures = [f"alpha-nDCG@{cutoff}", f"strec@{cutoff}"]
    by_query = pyndeval.ndeval(qrels, run, measures=measures, alpha=alpha)
    alpha_key = f"alpha-nDCG@{cutoff}"
    strec_key = f"strec@{cutoff}"
    result: dict[str, Any] = {
        "skipped": False,
        "cutoff": cutoff,
        "alpha": alpha,
        "alpha_nDCG": mean(row[alpha_key] for row in by_query.values()),
        "strec": mean(row[strec_key] for row in by_query.values()),
    }
    if per_query:
        result["per_query"] = {
            qid: {"alpha_nDCG": row[alpha_key], "strec": row[strec_key]}
            for qid, row in sorted(by_query.items())
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wrap card2nugget + query2nugget (+qrels/run) for two model clusters.",
    )
    parser.add_argument("--query", default=None, help="Single query text.")
    parser.add_argument("--queries-file", default=None, help="One query per line.")
    parser.add_argument("--model", default=None, help="OpenAI model override for query2nugget.")
    parser.add_argument("--subtopic", default="1", help="Subtopic id written into qrels.")
    parser.add_argument("--eval-cutoff", type=int, default=20, help="Cutoff for evaluate_pyndeval metrics.")
    parser.add_argument("--eval-alpha", type=float, default=0.5, help="Alpha for alpha-nDCG.")
    parser.add_argument("--eval-per-query", action="store_true", help="Include per-query eval metrics.")
    parser.add_argument("--jobs-batch-json", default=None, help="batch_runs JSON containing {job_id, query, model_ids}.")
    parser.add_argument("--job-ids-file", default=None, help="Optional file: one job_id per line.")
    parser.add_argument("--max-job-sets", type=int, default=6, help="Max sets to run from jobs JSON (default: 6).")
    parser.add_argument(
        "--dedupe-model-sets",
        action="store_true",
        help="Skip repeated model-id sets when iterating job sets.",
    )

    parser.add_argument("--cluster-a-name", default="cluster_a", help="Label for cluster A outputs.")
    parser.add_argument("--cluster-a-model-ids", nargs="*", default=None, help="Cluster A model ids.")
    parser.add_argument("--cluster-a-model-ids-file", default=None, help="Text file for cluster A model ids.")

    parser.add_argument("--cluster-b-name", default="cluster_b", help="Label for cluster B outputs.")
    parser.add_argument("--cluster-b-model-ids", nargs="*", default=None, help="Cluster B model ids.")
    parser.add_argument("--cluster-b-model-ids-file", default=None, help="Text file for cluster B model ids.")

    parser.add_argument(
        "--output-dir",
        default=str(PIPELINE_DIR),
        help=f"Output directory (default: {PIPELINE_DIR})",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    if args.jobs_batch_json:
        jobs_path = Path(args.jobs_batch_json)
        if not jobs_path.is_file():
            parser.error(f"--jobs-batch-json not found: {jobs_path}")
        wanted: set[str] | None = None
        if args.job_ids_file:
            p = Path(args.job_ids_file)
            if not p.is_file():
                parser.error(f"--job-ids-file not found: {p}")
            wanted = set(_load_lines(p))

        job_sets = _extract_job_sets(jobs_path, wanted)
        if not job_sets:
            parser.error("No valid job sets found in --jobs-batch-json.")

        seen_signatures: set[str] = set()
        run_count = 0
        for item in job_sets:
            if run_count >= max(1, int(args.max_job_sets)):
                break
            query = item["query"]
            model_ids = item["model_ids"]
            job_id = item["job_id"] or f"job_{run_count:02d}"
            sig = "||".join(sorted(model_ids))
            if args.dedupe_model_sets and sig in seen_signatures:
                print(f"[jobs] skip duplicated model set for job_id={job_id}")
                continue
            seen_signatures.add(sig)

            run_count += 1
            cluster_name = _safe_name(f"jobset_{run_count:02d}_{job_id}")
            print(f"[jobs] running {cluster_name} | job_id={job_id} | models={len(model_ids)}")
            query_maps = _map_queries([query], args.model)
            cluster_out_dir = out_dir / cluster_name
            cluster_out_dir.mkdir(parents=True, exist_ok=True)
            summary = _run_cluster(cluster_name, model_ids, query_maps, cluster_out_dir, str(args.subtopic))
            summary["job_id"] = job_id
            summary["query"] = query
            summaries.append(summary)
    else:
        queries = _collect_queries(args)
        if not queries:
            parser.error("Provide --query and/or --queries-file.")

        cluster_a_ids = _collect_model_ids(args.cluster_a_model_ids, args.cluster_a_model_ids_file)
        cluster_b_ids = _collect_model_ids(args.cluster_b_model_ids, args.cluster_b_model_ids_file)
        if not cluster_a_ids or not cluster_b_ids:
            parser.error("Both cluster A and cluster B model id sets are required.")

        print(f"[query2nugget] mapping {len(queries)} query(ies)")
        query_maps = _map_queries(queries, args.model)
        query_map_path = out_dir / "query_header_keyword_mapping.json"
        _save_json(query_map_path, {"queries": query_maps} if len(query_maps) > 1 else query_maps[0])
        print(f"[query2nugget] saved mapping: {query_map_path.resolve()}")

        summaries.append(_run_cluster(args.cluster_a_name, cluster_a_ids, query_maps, out_dir, str(args.subtopic)))
        summaries.append(_run_cluster(args.cluster_b_name, cluster_b_ids, query_maps, out_dir, str(args.subtopic)))

    for s in summaries:
        eval_result = _evaluate_cluster(
            Path(s["run_path"]),
            Path(s["qrels_path"]),
            cutoff=args.eval_cutoff,
            alpha=args.eval_alpha,
            per_query=bool(args.eval_per_query),
        )
        s["evaluation"] = eval_result
        if eval_result.get("skipped"):
            print(
                f"[eval:{s['cluster']}] skipped ({eval_result.get('reason')}) "
                f"run_rows={eval_result.get('run_rows')} qrels_rows={eval_result.get('qrels_rows')}"
            )
        else:
            print(
                f"[eval:{s['cluster']}] alpha-nDCG@{args.eval_cutoff}={eval_result['alpha_nDCG']:.6f} "
                f"strec@{args.eval_cutoff}={eval_result['strec']:.6f}"
            )

    summary_path = out_dir / "pipeline_summary.json"
    _save_json(summary_path, {"clusters": summaries})
    print(f"[done] summary -> {summary_path.resolve()}")
    for s in summaries:
        print(
            f"[done] {s['cluster']}: qrels={s['qrels_lines']} run={s['run_lines']} "
            f"run_path={s['run_path']}"
        )


if __name__ == "__main__":
    main()
