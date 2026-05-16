#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

METHODS: List[str] = ["dense", "sparse", "hybrid", "keyword", "single_column", "unionable"]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict json: {path}")
    return data


def _extract_method_scores(summary: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
    clusters = summary.get("clusters") or []
    if not isinstance(clusters, list) or not clusters:
        raise ValueError("Invalid pipeline_summary: missing clusters")
    cluster = clusters[0] or {}
    job_id = str(cluster.get("job_id") or cluster.get("cluster") or "").strip()
    rows = cluster.get("query_method_counts") or []
    score_map: Dict[str, int] = {m: 0 for m in METHODS}
    for row in rows:
        method = str((row or {}).get("method", "")).strip()
        if method not in score_map:
            continue
        score_map[method] = int((row or {}).get("matched_dedup_count", 0) or 0)
    return job_id, score_map


def _winner_methods(score_map: Dict[str, int]) -> List[str]:
    max_score = max(score_map.values()) if score_map else 0
    return [m for m in METHODS if score_map.get(m, 0) == max_score]


def _collect_top1_counts(jobs_dir: str) -> Tuple[Counter, List[Dict[str, Any]], int]:
    counts: Counter = Counter({m: 0 for m in METHODS})
    details: List[Dict[str, Any]] = []
    total_jobs = 0

    for name in sorted(os.listdir(jobs_dir)):
        job_dir = os.path.join(jobs_dir, name)
        if not os.path.isdir(job_dir):
            continue
        summary_path = os.path.join(job_dir, "evaluate", "pipeline_summary.json")
        if not os.path.isfile(summary_path):
            continue
        total_jobs += 1
        summary = _load_json(summary_path)
        job_id, score_map = _extract_method_scores(summary)
        winners = _winner_methods(score_map)
        for method in winners:
            counts[method] += 1
        details.append(
            {
                "job_id": job_id or name,
                "winner_methods": winners,
                "winner_count": len(winners),
                **{f"score_{m}": score_map.get(m, 0) for m in METHODS},
            }
        )
    return counts, details, total_jobs


def _save_details_csv(path: str, details: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["job_id", "winner_methods", "winner_count"] + [f"score_{m}" for m in METHODS]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in details:
            out = dict(row)
            out["winner_methods"] = "|".join(row.get("winner_methods", []))
            writer.writerow(out)


def _plot_histogram(path: str, counts: Counter, total_jobs: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    methods = METHODS
    values = [int(counts.get(m, 0)) for m in methods]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(methods, values, color="#4C78A8")
    plt.title(f"Top1 Nugget Winner Count by Method (N={total_jobs} jobs)")
    plt.xlabel("Method")
    plt.ylabel("Top1 count")
    plt.xticks(rotation=20, ha="right")
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, str(v), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build histogram for top1 nugget winners across jobs. "
            "Top1 score is matched_dedup_count from evaluate/pipeline_summary.json."
        )
    )
    parser.add_argument("--jobs_dir", default="jobs_251117", help="Jobs root directory (default: jobs_251117)")
    parser.add_argument(
        "--out_png",
        default="jobs_251117/analysis/top1_nugget_winner_histogram.png",
        help="Output histogram png path",
    )
    parser.add_argument(
        "--out_csv",
        default="jobs_251117/analysis/top1_nugget_winner_details.csv",
        help="Output per-job winner details csv path",
    )
    args = parser.parse_args()

    counts, details, total_jobs = _collect_top1_counts(args.jobs_dir)
    if total_jobs == 0:
        print(f"No jobs with evaluate/pipeline_summary.json under: {args.jobs_dir}")
        return 1

    _save_details_csv(args.out_csv, details)
    _plot_histogram(args.out_png, counts, total_jobs)

    print(f"Analyzed jobs: {total_jobs}")
    print("Top1 winner counts:")
    for method in METHODS:
        print(f"  {method}: {int(counts.get(method, 0))}")
    print(f"Saved histogram: {args.out_png}")
    print(f"Saved details csv: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
