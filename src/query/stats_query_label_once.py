#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


SIX_LABELS = ["evidence-based", "comparison", "experience", "reason", "instruction", "debate"]
FOUR_LABELS = ["factual_retrieval", "comparison", "summarization", "causal_reasoning"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def format_pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.00%"
    return f"{(numerator / denominator) * 100:.2f}%"


def render_mini_bar_chart(path: Path, labels: list[str], counts: list[int], title: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 2.6), dpi=160)
    colors = ["#2F6BFF", "#29A36A", "#F39C33", "#D84E4E", "#6F42C1", "#17A2B8"]
    bars = ax.bar(labels, counts, color=colors[: len(labels)], width=0.62)

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_ylabel("Queries", fontsize=9)
    ax.tick_params(axis="x", labelsize=8, rotation=15)
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
    parser = argparse.ArgumentParser(description="Stats for query labels and optional distribution plot.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("src/query/data/query/query_label_once.jsonl"),
        help="Path to query_label_once JSONL.",
    )
    parser.add_argument(
        "--plot_path",
        type=Path,
        default=Path("src/query/data/query/query_label_once_distribution.png"),
        help="Path to save the bar chart.",
    )
    parser.add_argument(
        "--scheme",
        choices=["six", "four"],
        default="six",
        help="Label scheme to use when ordering categories.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    rows = load_jsonl(args.input)
    labels = SIX_LABELS if args.scheme == "six" else FOUR_LABELS

    counter = Counter()
    total = 0
    missing = 0
    for row in rows:
        total += 1
        label = str(row.get("label") or "").strip()
        if not label:
            missing += 1
            continue
        counter[label] += 1

    counts = [counter.get(label, 0) for label in labels]
    labeled_total = sum(counts)

    print(f"input={args.input}")
    print(f"rows={total}")
    print(f"labeled_rows={labeled_total}")
    print(f"missing_label_rows={missing}")
    for label in labels:
        count = counter.get(label, 0)
        print(f"{label}\t{count}\t{format_pct(count, labeled_total)}")

    render_mini_bar_chart(
        args.plot_path,
        labels=labels,
        counts=counts,
        title="Query Label Distribution" if args.scheme == "six" else "Query Label Distribution (4-class)",
    )
    print(f"saved_plot={args.plot_path}")


if __name__ == "__main__":
    main()
