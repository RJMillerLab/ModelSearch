#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def _is_nonempty_text(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return bool(str(value).strip())


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


def _first_nonempty_text(values: pd.Series) -> str | None:
    for value in values.tolist():
        if _is_nonempty_text(value):
            return str(value).strip()
    return None


def build_corpusid_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["corpusid_norm"] = out["corpusid"].apply(_normalize_id) if "corpusid" in out.columns else None
    if "title" not in out.columns:
        out["title"] = None
    if "hf_links" not in out.columns:
        out["hf_links"] = [[] for _ in range(len(out))]
    if "hf_models" not in out.columns:
        out["hf_models"] = [[] for _ in range(len(out))]

    out["title_text"] = out["title"].apply(lambda x: str(x).strip() if _is_nonempty_text(x) else None)
    out["hf_links_list"] = out["hf_links"].apply(_parse_json_list)
    out["hf_models_list"] = out["hf_models"].apply(_parse_json_list)
    out["has_hf_link"] = out["hf_links_list"].apply(lambda xs: len(xs) > 0)
    out["has_model_link"] = out["hf_models_list"].apply(lambda xs: len(xs) > 0)

    grouped = (
        out.dropna(subset=["corpusid_norm"])
        .groupby("corpusid_norm", sort=False)
        .agg(
            title=("title_text", _first_nonempty_text),
            has_title=("title_text", lambda s: any(_is_nonempty_text(x) for x in s)),
            has_hf_link=("has_hf_link", "any"),
            has_model_link=("has_model_link", "any"),
        )
        .reset_index()
    )
    grouped["valid_title"] = grouped["has_title"]
    grouped["unique_corpusid"] = grouped["corpusid_norm"]
    return grouped


def format_pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.00%"
    return f"{(numerator / denominator) * 100:.2f}%"


def render_mini_bar_chart(path: Path, labels: list[str], counts: list[int]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 2.6), dpi=160)
    colors = ["#2F6BFF", "#29A36A", "#F39C33", "#D84E4E"]
    bars = ax.bar(labels, counts, color=colors[: len(labels)], width=0.62)

    ax.set_title("Corpus Link Funnel", fontsize=10, pad=8)
    ax.set_ylabel("Unique corpusids", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    max_count = max(counts) if counts else 0
    ax.set_ylim(0, max_count * 1.18 + 1)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_count * 0.03 + 0.2,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument(
        "--plot_path",
        type=Path,
        default=None,
        help="Optional path to save a mini bar chart PNG.",
    )
    args = parser.parse_args()

    if not args.parquet.exists():
        raise SystemExit(f"Parquet not found: {args.parquet}")
    if not args.query_jsonl.exists():
        raise SystemExit(f"Query jsonl not found: {args.query_jsonl}")

    raw_df = pd.read_parquet(args.parquet)
    if "corpusid" not in raw_df.columns:
        raise SystemExit("Parquet must contain a `corpusid` column.")

    corpus_df = build_corpusid_summary(raw_df)

    unique_corpusids = int(len(corpus_df))
    valid_title_df = corpus_df[corpus_df["valid_title"]].copy()
    hf_link_df = valid_title_df[valid_title_df["has_hf_link"]].copy()
    hf_model_df = hf_link_df[hf_link_df["has_model_link"]].copy()

    query_rows = load_query_corpusids(args.query_jsonl)
    total_queries = len(query_rows)

    matched_queries: list[tuple[str, set[str]]] = []
    unmatched_queries: list[tuple[str, set[str]]] = []
    matched_query_corpusids: set[str] = set()
    all_query_corpusids: set[str] = set()

    model_link_corpusids = set(hf_model_df["corpusid_norm"].dropna().astype(str).tolist())

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

    print("corpusid_stats")
    print(f"  unique_corpusids: {unique_corpusids}")
    print(
        f"  valid_title: {len(valid_title_df)} / {unique_corpusids} "
        f"({format_pct(len(valid_title_df), unique_corpusids)})"
    )
    print(
        f"  with_hf_link: {len(hf_link_df)} / {len(valid_title_df)} "
        f"({format_pct(len(hf_link_df), len(valid_title_df))})"
    )
    print(
        f"  with_hf_model_link: {len(hf_model_df)} / {len(hf_link_df)} "
        f"({format_pct(len(hf_model_df), len(hf_link_df))})"
    )
    print(
        f"  with_hf_model_link_overall: {len(hf_model_df)} / {unique_corpusids} "
        f"({format_pct(len(hf_model_df), unique_corpusids)})"
    )
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

    if args.plot_path:
        render_mini_bar_chart(
            args.plot_path,
            labels=["unique", "title", "hf link", "hf model"],
            counts=[unique_corpusids, len(valid_title_df), len(hf_link_df), len(hf_model_df)],
        )
        print()
        print(f"saved_plot: {args.plot_path}")


if __name__ == "__main__":
    main()
