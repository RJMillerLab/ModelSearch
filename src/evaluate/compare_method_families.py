#!/usr/bin/env python3
"""Compare table-search methods vs model-search methods across many job ids.

Input is a batch jobs JSON (for example jobs_251117/batch_runs/batch_preset_queries_*.json).
For each job_id, this script loads:

  data_*/evaluate/pipeline/<job_id>/pipeline_summary.json

and computes method-level + family-level nugget statistics using the current
query2nugget header-nonempty matching logic.

Families:
  - model_search = sparse + dense + hybrid
  - table_search = keyword + single_column + unionable
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.config import OUTPUT_DIR
from src.evaluate.query2nugget_mapping import NUGGET_SCHEMA_HEADERS, _header_non_empty_for_row, _row_dict

MODEL_SEARCH_METHODS = ("sparse", "dense", "hybrid")
TABLE_SEARCH_METHODS = ("keyword", "single_column", "unionable")
ALL_METHODS = MODEL_SEARCH_METHODS + TABLE_SEARCH_METHODS


def _safe_stem_name(path: Path) -> str:
    return path.stem or "batch_jobs"


def _load_jobs(job_json_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(job_json_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("queries", [])
    if not isinstance(payload, list):
        return []
    out: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        job_id = str(item.get("job_id", "")).strip()
        query = str(item.get("query", "")).strip()
        if job_id:
            out.append({"job_id": job_id, "query": query})
    return out


def _row_signature(cells: dict[str, str]) -> tuple[str, ...]:
    return tuple((cells.get(h, "") or "").strip() for h in NUGGET_SCHEMA_HEADERS)


def _compute_stats_for_csvs(csv_paths: list[Path], headers: list[str]) -> dict[str, Any]:
    raw_count = 0
    filter_count = 0
    raw_signatures: set[tuple[str, ...]] = set()
    filter_signatures: set[tuple[str, ...]] = set()
    for csv_path in csv_paths:
        if not csv_path.is_file():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cells = _row_dict(row)
                sig = _row_signature(cells)
                raw_count += 1
                raw_signatures.add(sig)
                if headers and any(_header_non_empty_for_row(h, cells) for h in headers):
                    filter_count += 1
                    filter_signatures.add(sig)
    return {
        "original_sum": raw_count,
        "filter_sum": filter_count,
        "original_dedup": len(raw_signatures),
        "filter_dedup": len(filter_signatures),
    }


def _extract_cluster(summary_payload: dict[str, Any]) -> dict[str, Any]:
    clusters = summary_payload.get("clusters", [])
    if isinstance(clusters, list) and clusters:
        first = clusters[0]
        if isinstance(first, dict):
            return first
    return {}


def _compute_job_record(job_id: str, query: str, pipeline_root: Path) -> dict[str, Any] | None:
    summary_path = pipeline_root / job_id / "pipeline_summary.json"
    if not summary_path.is_file():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    cluster = _extract_cluster(payload)
    query_headers = cluster.get("query_headers", []) if isinstance(cluster.get("query_headers"), list) else []
    card_outputs = cluster.get("card_outputs", []) if isinstance(cluster.get("card_outputs"), list) else []
    by_model_id = {
        str(row.get("model_id", "")).strip(): Path(str(row.get("csv_path", "")))
        for row in card_outputs
        if isinstance(row, dict) and str(row.get("model_id", "")).strip()
    }
    method_model_sets = cluster.get("method_model_sets", []) if isinstance(cluster.get("method_model_sets"), list) else []
    method_to_ids = {
        str(row.get("method", "")).strip(): [
            str(mid).strip()
            for mid in (row.get("model_ids", []) if isinstance(row.get("model_ids"), list) else [])
            if str(mid).strip()
        ]
        for row in method_model_sets
        if isinstance(row, dict) and str(row.get("method", "")).strip()
    }

    method_stats: dict[str, dict[str, Any]] = {}
    for method in ALL_METHODS:
        mids = method_to_ids.get(method, [])
        csv_paths = [by_model_id[mid] for mid in mids if mid in by_model_id]
        stats = _compute_stats_for_csvs(csv_paths, query_headers)
        stats["model_count"] = len(mids)
        stats["model_ids"] = mids
        method_stats[method] = stats

    def family_stats(methods: tuple[str, ...]) -> dict[str, Any]:
        csv_paths: list[Path] = []
        model_ids: list[str] = []
        for method in methods:
            mids = method_to_ids.get(method, [])
            model_ids.extend(mids)
            csv_paths.extend(by_model_id[mid] for mid in mids if mid in by_model_id)
        stats = _compute_stats_for_csvs(csv_paths, query_headers)
        stats["model_count"] = len(model_ids)
        stats["method_count"] = len([m for m in methods if method_to_ids.get(m)])
        return stats

    model_search = family_stats(MODEL_SEARCH_METHODS)
    table_search = family_stats(TABLE_SEARCH_METHODS)
    metric = "filter_dedup"
    if table_search[metric] > model_search[metric]:
        winner = "table_search"
    elif table_search[metric] < model_search[metric]:
        winner = "model_search"
    else:
        winner = "tie"

    return {
        "job_id": job_id,
        "query": query or str(cluster.get("query", "")).strip(),
        "query_headers": query_headers,
        "summary_path": str(summary_path.resolve()),
        "method_stats": method_stats,
        "model_search": model_search,
        "table_search": table_search,
        "winner_filter_dedup": winner,
    }


def _to_markdown(batch_name: str, jobs_json: Path, rows: list[dict[str, Any]], missing_job_ids: list[str]) -> str:
    total = len(rows)
    table_wins = sum(1 for r in rows if r["winner_filter_dedup"] == "table_search")
    model_wins = sum(1 for r in rows if r["winner_filter_dedup"] == "model_search")
    ties = sum(1 for r in rows if r["winner_filter_dedup"] == "tie")

    lines = [
        f"# Method Family Comparison: `{batch_name}`",
        "",
        f"- Jobs JSON: `{jobs_json.resolve()}`",
        f"- Jobs analyzed: `{total}`",
        f"- Missing pipeline summaries: `{len(missing_job_ids)}`",
        "",
        "## Aggregate",
        "",
        "| comparison | count |",
        "| --- | ---: |",
        f"| `table_search filter_dedup > model_search filter_dedup` | {table_wins} |",
        f"| `model_search filter_dedup > table_search filter_dedup` | {model_wins} |",
        f"| `tie on filter_dedup` | {ties} |",
        "",
        "## Per Job",
        "",
        "| job_id | model_search_filter_dedup | table_search_filter_dedup | winner | query |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['job_id']}` | {row['model_search']['filter_dedup']} | {row['table_search']['filter_dedup']} | "
            f"`{row['winner_filter_dedup']}` | {str(row.get('query', '')).replace('|', '\\|')} |"
        )

    lines.extend(
        [
            "",
            "## Method Details",
            "",
            "| job_id | sparse | dense | hybrid | keyword | single_column | unionable | model_search(filter_dedup) | table_search(filter_dedup) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        ms = row["method_stats"]
        lines.append(
            f"| `{row['job_id']}` | "
            f"{ms['sparse']['filter_dedup']} | {ms['dense']['filter_dedup']} | {ms['hybrid']['filter_dedup']} | "
            f"{ms['keyword']['filter_dedup']} | {ms['single_column']['filter_dedup']} | {ms['unionable']['filter_dedup']} | "
            f"{row['model_search']['filter_dedup']} | {row['table_search']['filter_dedup']} |"
        )

    if missing_job_ids:
        lines.extend(["", "## Missing Jobs", ""])
        for job_id in missing_job_ids:
            lines.append(f"- `{job_id}`")

    lines.extend(
        [
            "",
            "_`model_search` = sparse + dense + hybrid; `table_search` = keyword + single_column + unionable._",
            "",
            "_Current comparison metric is `filter_dedup`: unique nugget rows remaining after query-header filtering._",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare keyword/single_column/unionable vs sparse/dense/hybrid across all job ids in a batch jobs JSON.",
    )
    parser.add_argument("--jobs-json", required=True, help="Batch jobs JSON (for example jobs_251117/batch_runs/batch_preset_queries_*.json).")
    parser.add_argument(
        "--pipeline-root",
        default=str(Path(OUTPUT_DIR) / "evaluate" / "pipeline"),
        help="Root directory containing per-job pipeline outputs (default: data_*/evaluate/pipeline).",
    )
    parser.add_argument("--output-md", default=None, help="Optional markdown output path.")
    parser.add_argument("--output-json", default=None, help="Optional json output path.")
    args = parser.parse_args()

    jobs_json = Path(args.jobs_json)
    if not jobs_json.is_file():
        raise SystemExit(f"--jobs-json not found: {jobs_json}")
    pipeline_root = Path(args.pipeline_root)
    batch_name = _safe_stem_name(jobs_json)

    jobs = _load_jobs(jobs_json)
    if not jobs:
        raise SystemExit(f"No valid job_id entries found in: {jobs_json}")

    rows: list[dict[str, Any]] = []
    missing_job_ids: list[str] = []
    for job in jobs:
        record = _compute_job_record(job["job_id"], job.get("query", ""), pipeline_root)
        if record is None:
            missing_job_ids.append(job["job_id"])
            continue
        rows.append(record)

    output_md = Path(args.output_md) if args.output_md else jobs_json.with_name(f"{batch_name}_method_family_compare.md")
    output_json = Path(args.output_json) if args.output_json else jobs_json.with_name(f"{batch_name}_method_family_compare.json")

    output_md.write_text(_to_markdown(batch_name, jobs_json, rows, missing_job_ids), encoding="utf-8")
    output_json.write_text(json.dumps({"jobs_json": str(jobs_json.resolve()), "rows": rows, "missing_job_ids": missing_job_ids}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved_markdown = {output_md.resolve()}")
    print(f"saved_json = {output_json.resolve()}")
    print(f"jobs_analyzed = {len(rows)}")
    print(f"missing_jobs = {len(missing_job_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
