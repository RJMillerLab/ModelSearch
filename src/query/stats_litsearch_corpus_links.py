#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _parse_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []
    return []


def _normalize_id(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def load_query_corpusids(query_jsonl: Path) -> list[tuple[str, set[str]]]:
    rows: list[tuple[str, set[str]]] = []
    with query_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            query = str(record.get("query") or "").strip()
            corpusids = {
                cid
                for cid in (_normalize_id(x) for x in (record.get("corpusids") or []))
                if cid
            }
            rows.append((query, corpusids))
    return rows


def count_model_link_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "hf_models" not in out.columns:
        out["hf_models"] = [[] for _ in range(len(out))]
    out["hf_models_list"] = out["hf_models"].apply(_parse_json_list)
    out["has_model_link"] = out["hf_models_list"].apply(lambda xs: len(xs) > 0)
    out["corpusid_norm"] = out["corpusid"].apply(_normalize_id) if "corpusid" in out.columns else None
    return out


def format_pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.00%"
    return f"{(numerator / denominator) * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick stats for LitSearch query ids against extracted corpus links.")
    parser.add_argument(
        "--parquet",
        required=True,
        type=Path,
        help="Path to corpus_hf_links parquet.",
    )
    parser.add_argument(
        "--query_jsonl",
        required=True,
        type=Path,
        help="Path to LitSearch query jsonl.",
    )
    parser.add_argument(
        "--top_k_examples",
        type=int,
        default=5,
        help="How many matched/unmatched examples to print.",
    )
    args = parser.parse_args()

    if not args.parquet.exists():
        raise SystemExit(f"Parquet not found: {args.parquet}")
    if not args.query_jsonl.exists():
        raise SystemExit(f"Query jsonl not found: {args.query_jsonl}")

    df = pd.read_parquet(args.parquet)
    if "corpusid" not in df.columns:
        raise SystemExit("Parquet must contain a `corpusid` column.")

    df = count_model_link_rows(df)

    total_rows = int(len(df))
    model_link_rows = int(df["has_model_link"].sum())
    unique_corpusids = int(df["corpusid_norm"].dropna().nunique()) if "corpusid_norm" in df.columns else 0
    unique_model_link_corpusids = int(df.loc[df["has_model_link"], "corpusid_norm"].dropna().nunique())

    query_rows = load_query_corpusids(args.query_jsonl)
    total_queries = len(query_rows)

    matched_queries: list[tuple[str, set[str]]] = []
    unmatched_queries: list[tuple[str, set[str]]] = []
    matched_query_corpusids: set[str] = set()
    all_query_corpusids: set[str] = set()

    model_link_corpusids = set(
        cid
        for cid in df.loc[df["has_model_link"], "corpusid_norm"].dropna().astype(str).tolist()
        if cid
    )

    for query, corpusids in query_rows:
        all_query_corpusids.update(corpusids)
        hit = corpusids & model_link_corpusids
        if hit:
            matched_queries.append((query, hit))
            matched_query_corpusids.update(hit)
        else:
            unmatched_queries.append((query, corpusids))

    matched_query_count = len(matched_queries)
    matched_unique_query_corpusids = len(matched_query_corpusids)
    total_unique_query_corpusids = len(all_query_corpusids)

    print("parquet_stats")
    print(f"  total_rows: {total_rows}")
    print(f"  rows_with_model_link: {model_link_rows} / {total_rows} ({format_pct(model_link_rows, total_rows)})")
    print(f"  unique_corpusids: {unique_corpusids}")
    print(f"  unique_corpusids_with_model_link: {unique_model_link_corpusids}")
    print()

    print("query_stats")
    print(f"  total_queries: {total_queries}")
    print(
        f"  queries_with_model_link_hits: {matched_query_count} / {total_queries} "
        f"({format_pct(matched_query_count, total_queries)})"
    )
    print(
        f"  unique_query_corpusids_with_model_link_hits: {matched_unique_query_corpusids} / "
        f"{total_unique_query_corpusids} ({format_pct(matched_unique_query_corpusids, total_unique_query_corpusids)})"
    )
    print()

    print("examples_matched")
    for idx, (query, hit) in enumerate(matched_queries[: max(0, args.top_k_examples)], start=1):
        print(f"  {idx}. hit_ids={sorted(hit)}")
        print(f"     query={query}")
    if not matched_queries:
        print("  (none)")
    print()

    print("examples_unmatched")
    for idx, (query, corpusids) in enumerate(unmatched_queries[: max(0, args.top_k_examples)], start=1):
        preview = sorted(corpusids)
        if len(preview) > 8:
            preview = preview[:8] + ["..."]
        print(f"  {idx}. corpusids={preview}")
        print(f"     query={query}")
    if not unmatched_queries:
        print("  (none)")


if __name__ == "__main__":
    main()
