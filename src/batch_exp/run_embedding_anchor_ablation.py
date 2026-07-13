#!/usr/bin/env python3
"""Run the dense table-anchor ablation and optionally redraw the paper figure."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SEARCH_TYPES = (
    ("keyword", "embedding_anchor_keyword"),
    ("single_column", "embedding_anchor_joinable"),
    ("unionable", "embedding_anchor_unionable"),
)


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run query -> dense table anchor -> tab2tab -> card nugget ablation."
    )
    parser.add_argument("--resources", nargs="+", default=["hugging"], choices=["hugging", "github", "arxiv"])
    parser.add_argument("--queries-file", default="data_251117/query/query_rewrite_polished.jsonl")
    parser.add_argument("--anchor-top-k", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10, help="Final table/model candidate budget for inference.")
    parser.add_argument("--eval-top-k", type=int, nargs="+", default=[10])
    parser.add_argument("--limit-jobs", type=int, default=0, help="Optional limit for inference/eval.")
    parser.add_argument("--jobs-dir", default="tmp/query2anc_table_embed_3tab2tab_jobs_top10")
    parser.add_argument("--jobs-json", default="tmp/query2anc_table_embed_3tab2tab_jobs_top10.json")
    parser.add_argument("--batch-output-prefix", default="tmp/card2tab2card")
    parser.add_argument("--run-id", default="embedding_anchor_3tab2tab_top10")
    parser.add_argument(
        "--query-maps-json",
        default="",
        help="Optional existing query_maps.json to reuse in nugget eval.",
    )
    parser.add_argument(
        "--base-summary-json",
        default="data_251117/evaluate/query2modelcard_nugget_batch/20260516_215540/aggregate_summary.json",
        help="Existing NL2Card/NL2Card2Tab2Card summary for combined plotting.",
    )
    parser.add_argument(
        "--figure-output",
        default="outputs/statistic.pdf",
        help="Combined figure output. Empty string skips plotting.",
    )
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    resources = list(args.resources)
    jobs_json = Path(args.jobs_json)
    jobs_dir = Path(args.jobs_dir)
    run_dir = Path("data_251117/evaluate/query2modelcard_nugget_batch") / args.run_id
    anchor_summary = run_dir / "aggregate_summary.json"
    anchor_per_query = run_dir / "per_query_method_scores.jsonl"

    if not args.skip_inference:
        for search_type, method_name in SEARCH_TYPES:
            output_json = f"{args.batch_output_prefix}_{method_name}_batch.json"
            cmd = [
                py,
                "-m",
                "src.search.query2anc_table_embed_inf_batch",
                "--resources",
                *resources,
                "--queries_file",
                args.queries_file,
                "--anchor_top_k",
                str(args.anchor_top_k),
                "--search_type",
                search_type,
                "--method_name",
                method_name,
                "--top_k",
                str(args.top_k),
                "--output_json",
                output_json,
                "--jobs_output_dir",
                str(jobs_dir),
                "--jobs_json",
                str(jobs_json),
            ]
            if args.limit_jobs > 0:
                cmd.extend(["--limit", str(args.limit_jobs)])
            _run(cmd, dry_run=args.dry_run)

    if not args.skip_eval:
        cmd = [
            py,
            "-m",
            "src.batch_exp.run_batch_eval",
            "--jobs-json",
            str(jobs_json),
            "--jobs-root",
            str(jobs_dir),
            "--top-k",
            *[str(k) for k in args.eval_top_k],
            "--run-id",
            args.run_id,
        ]
        if args.query_maps_json:
            cmd.extend(["--query-maps-json", args.query_maps_json])
        if args.limit_jobs > 0:
            cmd.extend(["--limit-jobs", str(args.limit_jobs)])
        _run(cmd, dry_run=args.dry_run)

    if args.figure_output and not args.skip_plot:
        _run(
            [
                py,
                "src/batch_exp/plot_distribution_eval.py",
                "--summary-json",
                args.base_summary_json,
                "--extra-per-query-jsonl",
                str(anchor_per_query),
                "--output",
                args.figure_output,
                "--compare-top-k",
                *[str(k) for k in args.eval_top_k],
            ],
            dry_run=args.dry_run,
        )
        print(f"[figure] base summary: {args.base_summary_json}", flush=True)
        print(f"[figure] anchor summary: {anchor_summary}", flush=True)


if __name__ == "__main__":
    main()
