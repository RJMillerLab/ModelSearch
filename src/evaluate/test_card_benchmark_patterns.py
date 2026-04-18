"""
Quick pattern analysis for Hugging Face model cards.

Usage (from repo root):

    python -m src.evaluate.test_card_benchmark_patterns \
        --sample-size 20

This script:
1. Loads the card text parquet from ``config.CARD_CONTENT_RAW`` (or a custom path).
2. Checks whether ``tag`` / ``tags`` columns contain benchmark‑related keywords.
3. Strips markdown tables from ``card`` and searches the *non‑table* text for:
   - benchmark / dataset / evaluation‑score keywords
   - Hugging Face "official" metadata patterns such as ``model-index`` blocks
     and ``.eval_results`` references.

It prints a few sampled rows so you can inspect how often
important evaluation information appears outside tables.
"""

from __future__ import annotations

import argparse
import random
import re
from typing import Iterable, List, Optional

import pandas as pd

from src import config


BENCHMARK_KEYWORDS = [
    "benchmark",
    "leaderboard",
]

DATASET_KEYWORDS = [
    "dataset",
    "corpus",
    "dataset:",
]

EVAL_SCORE_KEYWORDS = [
    "evaluation",
    "eval score",
    "eval scores",
    "evaluation score",
    "evaluation scores",
    "metric",
    "metrics",
    "accuracy",
    "f1",
    "bleu",
    "rouge",
    "exact match",
]

# Patterns that correspond more directly to Hugging Face "official" structures.
OFFICIAL_METADATA_PATTERNS = [
    r"\bmodel-index\b",
    r"\beval_results\b",
    r"\.eval_results/",
    r"^---\s*$",  # YAML front‑matter separator
]


def _normalize_text(x: Optional[str]) -> str:
    if not isinstance(x, str):
        return ""
    return x.lower()


def _flatten_tag_value(tag_val) -> str:
    """
    Tags column can be string, list, or None. Normalize to a single string.
    """
    if tag_val is None:
        return ""
    if isinstance(tag_val, (list, tuple, set)):
        parts = []
        for v in tag_val:
            if v is None:
                continue
            parts.append(str(v))
        return " ".join(parts).lower()
    return str(tag_val).lower()


def remove_markdown_tables(md: str) -> str:
    """
    Remove simple markdown tables from card content.

    Heuristics:
    - Drop consecutive lines that contain at least one '|' *and* look like table rows.
    - Also drop typical header separator lines like '| --- | --- |' or '--- | ---'.
    """
    if not isinstance(md, str):
        return ""

    lines = md.splitlines()
    out: List[str] = []
    in_table_block = False

    def is_table_line(line: str) -> bool:
        stripped = line.strip()
        if "|" not in stripped:
            return False
        # Very short pipe‑containing lines are often not tables, but keep it simple.
        if re.match(r"^\s*\|?\s*:?-{3,}\s*(:?-{3,}\s*\|?)*\s*$", stripped):
            return True
        # If there are multiple cells separated by pipes, treat as table.
        segments = [s for s in stripped.split("|") if s.strip()]
        return len(segments) >= 2

    for line in lines:
        if is_table_line(line):
            in_table_block = True
            continue
        if in_table_block and not line.strip():
            # Blank line after table – end the table block.
            in_table_block = False
            continue
        if not in_table_block:
            out.append(line)

    return "\n".join(out)


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    t = _normalize_text(text)
    return any(k in t for k in keywords)


def detect_official_metadata_patterns(text: str) -> List[str]:
    """
    Return a list of which "official" patterns we saw in the non‑table text.
    """
    if not isinstance(text, str):
        return []
    hits: List[str] = []
    for pat in OFFICIAL_METADATA_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE):
            hits.append(pat)
    return hits


def sample_non_empty_cards(
    df: pd.DataFrame, sample_size: int, random_seed: int = 0
) -> pd.DataFrame:
    df_non_empty = df[df["card"].astype(str).str.len() > 0]
    if df_non_empty.empty:
        raise ValueError("No non‑empty 'card' rows found in dataset.")
    n = min(sample_size, len(df_non_empty))
    return df_non_empty.sample(n=n, random_state=random_seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--card-path",
        type=str,
        default=config.CARD_CONTENT_RAW,
        help="Path to parquet / JSONL with at least 'cmodelId' and 'card' columns.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="How many non‑empty cards to inspect.",
    )
    args = parser.parse_args()

    path = args.card_path
    print(f"Loading card content from: {path}")

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        # Fallback: assume JSON lines.
        df = pd.read_json(path, lines=True)

    required_cols = {"cmodelId", "card"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    # Optional tag columns.
    tag_col_name: Optional[str] = None
    for cand in ["tag", "tags", "Tags", "Tag"]:
        if cand in df.columns:
            tag_col_name = cand
            break

    if tag_col_name is None:
        print("Warning: no 'tag' / 'tags' column found; tag keyword checks will be empty.")

    df_sample = sample_non_empty_cards(df, sample_size=args.sample_size)

    rows = []
    for _, row in df_sample.iterrows():
        cid = row["cmodelId"]
        card_text = str(row["card"])

        tag_text = ""
        if tag_col_name is not None:
            tag_text = _flatten_tag_value(row[tag_col_name])

        non_table = remove_markdown_tables(card_text)

        has_benchmark_in_tag = contains_any(tag_text, BENCHMARK_KEYWORDS)
        has_dataset_in_tag = contains_any(tag_text, DATASET_KEYWORDS)
        has_evalscore_in_tag = contains_any(tag_text, EVAL_SCORE_KEYWORDS)

        has_benchmark_in_nontable = contains_any(non_table, BENCHMARK_KEYWORDS)
        has_dataset_in_nontable = contains_any(non_table, DATASET_KEYWORDS)
        has_evalscore_in_nontable = contains_any(non_table, EVAL_SCORE_KEYWORDS)

        official_hits = detect_official_metadata_patterns(non_table)

        # Grab a short snippet around the first benchmark/dataset/eval hit, if any.
        snippet = ""
        lowered = non_table.lower()
        for kw in BENCHMARK_KEYWORDS + DATASET_KEYWORDS + EVAL_SCORE_KEYWORDS:
            idx = lowered.find(kw)
            if idx != -1:
                start = max(0, idx - 80)
                end = min(len(non_table), idx + 80)
                snippet = non_table[start:end].replace("\n", " ")
                break

        rows.append(
            {
                "cmodelId": cid,
                "has_benchmark_in_tag": has_benchmark_in_tag,
                "has_dataset_in_tag": has_dataset_in_tag,
                "has_evalscore_in_tag": has_evalscore_in_tag,
                "has_benchmark_in_nontable": has_benchmark_in_nontable,
                "has_dataset_in_nontable": has_dataset_in_nontable,
                "has_evalscore_in_nontable": has_evalscore_in_nontable,
                "official_metadata_hits": official_hits,
                "snippet": snippet,
            }
        )

    print(f"\nSampled {len(rows)} cards with non‑empty text.\n")
    for r in rows:
        print("-" * 80)
        print(f"cmodelId: {r['cmodelId']}")
        print(
            f"tag keywords: "
            f"benchmark={r['has_benchmark_in_tag']}, "
            f"dataset={r['has_dataset_in_tag']}, "
            f"eval_score={r['has_evalscore_in_tag']}"
        )
        print(
            f"non‑table keywords: "
            f"benchmark={r['has_benchmark_in_nontable']}, "
            f"dataset={r['has_dataset_in_nontable']}, "
            f"eval_score={r['has_evalscore_in_nontable']}"
        )
        print(f"official metadata patterns: {r['official_metadata_hits']}")
        if r["snippet"]:
            print(f"snippet: {r['snippet']}")
        else:
            print("snippet: <no obvious benchmark/dataset/eval keyword in non‑table text>")


if __name__ == "__main__":
    # Use a non‑deterministic seed for human exploration; we don't need reproducibility here.
    random.seed()
    main()

