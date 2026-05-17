#!/usr/bin/env python3
"""Plot nugget score distributions and rank share heatmaps from batch outputs.

This figure is intentionally separate from plot_batch_eval.py so the current
summary figure stays untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

METHOD_LABELS = {
    "sparse": "Sparse",
    "dense": "Dense",
    "hybrid": "Hybrid",
    "keyword": "Keyword",
    "single_column": "Single Column",
    "unionable": "Unionable",
}

METHOD_ORDER = ["sparse", "dense", "hybrid", "keyword", "single_column", "unionable"]
SEMANTIC_METHODS = ("sparse", "dense", "hybrid")
TABLE_METHODS = ("keyword", "single_column", "unionable")


def _load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    aggregate = data.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError(f"Missing aggregate object in {path}")
    return data


def _load_per_query_records(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def _safe_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _method_score(record: dict[str, Any], method: str, compare_key: str, score_field: str) -> float | None:
    method_scores = record.get("method_scores")
    if not isinstance(method_scores, dict):
        return None
    payload = method_scores.get(method)
    if not isinstance(payload, dict):
        return None
    bucket = payload.get(compare_key)
    if not isinstance(bucket, dict):
        return None
    return _safe_float(bucket.get(score_field))


def _collect_scores(
    records: list[dict[str, Any]],
    *,
    compare_key: str,
    score_field: str,
    integer_bins: bool,
) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {method: [] for method in METHOD_ORDER}
    for record in records:
        for method in METHOD_ORDER:
            score = _method_score(record, method, compare_key, score_field)
            if score is None:
                continue
            if integer_bins:
                score = float(int(round(score)))
            scores[method].append(score)
    return scores


def _collect_rank_counts(
    records: list[dict[str, Any]],
    *,
    compare_key: str,
    score_field: str,
) -> dict[str, list[int]]:
    rank_counts: dict[str, list[int]] = {method: [0] * len(METHOD_ORDER) for method in METHOD_ORDER}
    for record in records:
        scored: list[tuple[str, float]] = []
        for method in METHOD_ORDER:
            score = _method_score(record, method, compare_key, score_field)
            if score is None:
                continue
            scored.append((method, score))
        scored.sort(key=lambda item: (-item[1], METHOD_ORDER.index(item[0])))
        for rank_idx, (method, _) in enumerate(scored):
            if rank_idx < len(METHOD_ORDER):
                rank_counts[method][rank_idx] += 1
    return rank_counts


def _blue_palette() -> list[str]:
    return ["#eff6ff", "#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#2563eb"]


def _red_palette() -> list[str]:
    return ["#fef2f2", "#fee2e2", "#fecaca", "#fca5a5", "#f87171", "#b91c1c"]


def _red_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "rank_red",
        ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#de2d26", "#a50f15"],
    )


def plot(
    summary_path: Path,
    output_path: Path,
    *,
    compare_top_k: int | None,
    per_query_jsonl: Path | None,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))
    import matplotlib.pyplot as plt
    import numpy as np

    summary = _load_summary(summary_path)
    aggregate = summary["aggregate"]
    compare_k = compare_top_k or max(int(k) for k in aggregate["top_k_values"])
    compare_key = f"top_{compare_k}"
    score_field = str(aggregate.get("score_field", "hit_dedup"))

    summary_per_query_path = str(summary.get("paths", {}).get("per_query_method_scores", "") or "").strip()
    source_path = per_query_jsonl or (Path(summary_per_query_path) if summary_per_query_path else None)
    records = _load_per_query_records(source_path)
    if not records:
        raise RuntimeError(
            "No per-query records found. Provide --per-query-jsonl or ensure summary.paths.per_query_method_scores is set."
        )

    # Nugget counts are integers in the real evaluation output. We keep the
    # histogram bins aligned to whole counts so the demo and real runs share
    # the same layout.
    method_scores = _collect_scores(records, compare_key=compare_key, score_field=score_field, integer_bins=True)
    rank_counts = _collect_rank_counts(records, compare_key=compare_key, score_field=score_field)
    num_queries = len(records)

    # Layout: left 2x3 histogram grid, right stacked rank-share bars.
    fig = plt.figure(figsize=(13.6, 4.7), dpi=180)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.16)
    left = outer[0, 0].subgridspec(2, 3, hspace=0.14, wspace=0.16)
    right = outer[0, 1]
    ax_rank = fig.add_subplot(right)

    # Shared x range for the histogram family.
    all_scores = [v for vals in method_scores.values() for v in vals]
    if all_scores:
        lo = int(np.floor(min(all_scores)))
        hi = int(np.ceil(max(all_scores)))
    else:
        lo, hi = 0, 1
    lo = max(0, lo - 1)
    hi = hi + 1
    bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
    from matplotlib.colors import LinearSegmentedColormap

    blue_cmap = LinearSegmentedColormap.from_list("hist_blue", _blue_palette())
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    denom = max(1e-9, float(bin_centers.max() - bin_centers.min()))
    bin_colors = [blue_cmap((c - bin_centers.min()) / denom) for c in bin_centers]

    for idx, method in enumerate(METHOD_ORDER):
        ax = fig.add_subplot(left[idx // 3, idx % 3])
        vals = np.array(method_scores[method], dtype=float)
        mean_val = float(vals.mean()) if len(vals) else 0.0
        counts, _ = np.histogram(vals, bins=bins)
        ax.bar(
            bin_centers,
            counts,
            width=np.diff(bins),
            color=bin_colors,
            edgecolor="white",
            linewidth=0.7,
            align="center",
        )
        ax.axvline(mean_val, color="#7f0000", linestyle=(0, (4, 3)), linewidth=1.5)
        ymax = ax.get_ylim()[1]
        ax.text(
            mean_val,
            ymax * 0.96,
            f"Avg.={mean_val:.1f}",
            ha="center",
            va="top",
            fontsize=8.3,
            color="#7f0000",
            clip_on=True,
        )
        ax.set_title(METHOD_LABELS[method], fontsize=10.2, weight="bold", pad=6)
        ax.set_xlim(lo, hi)
        ax.grid(axis="y", color="#eeeeee", linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.tick_params(axis="both", labelsize=8.5)

    rank_matrix = np.array(
        [[rank_counts[method][rank_idx] / num_queries for rank_idx in range(len(METHOD_ORDER))] for method in METHOD_ORDER],
        dtype=float,
    )
    rank_colors = list(reversed(_red_palette()))
    y = np.arange(len(METHOD_ORDER))
    left_edge = np.zeros(len(METHOD_ORDER), dtype=float)
    for rank_idx in range(len(METHOD_ORDER)):
        vals = rank_matrix[:, rank_idx]
        color = rank_colors[rank_idx]
        ax_rank.barh(
            y,
            vals,
            left=left_edge,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            height=0.52,
            label=f"Rank {rank_idx + 1}",
        )
        for row_idx, val in enumerate(vals):
            pct = val * 100.0
            if pct >= 8.0:
                text_color = "white" if rank_idx >= 3 or pct >= 18 else "#1f2937"
                ax_rank.text(
                    left_edge[row_idx] + val / 2.0,
                    row_idx,
                    f"{pct:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8.3,
                    color=text_color,
                )
        left_edge += vals

    ax_rank.set_yticks(y, [METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=8.0)
    ax_rank.set_xlim(0.0, 1.0)
    ax_rank.set_xticks(np.linspace(0.0, 1.0, 6), [f"{int(v * 100)}%" for v in np.linspace(0.0, 1.0, 6)], fontsize=9)
    ax_rank.tick_params(axis="y", length=0, pad=2)
    ax_rank.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax_rank.invert_yaxis()
    ax_rank.grid(axis="x", color="#eeeeee", linestyle="--", linewidth=0.7)
    ax_rank.set_axisbelow(True)
    ax_rank.axhline(METHOD_ORDER.index("keyword") - 0.5, color="#6b6b6b", linestyle=(0, (4, 3)), linewidth=1.0)
    ax_rank.set_xlabel("")
    ax_rank.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.010),
        ncol=6,
        frameon=False,
        fontsize=8.4,
        handlelength=1.8,
        columnspacing=0.8,
        borderaxespad=0.0,
    )

    fig.text(
        0.055,
        0.955,
        "A. Nugget-Count Distributions by Method",
        ha="left",
        va="top",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.57,
        0.955,
        "B. Rank Share by Method (Top-1 to Top-6)",
        ha="left",
        va="top",
        fontsize=13,
        weight="bold",
    )
    fig.subplots_adjust(top=0.84, bottom=0.075, left=0.045, right=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot nugget score distributions and rank share heatmaps.")
    parser.add_argument("--summary-json", required=True, type=Path, help="aggregate_summary.json from run_batch_eval.py")
    parser.add_argument("--output", required=True, type=Path, help="Output image path, e.g. figure.png")
    parser.add_argument("--compare-top-k", type=int, default=None, help="Top-k budget for rank share panel. Default: largest top-k.")
    parser.add_argument("--per-query-jsonl", type=Path, default=None, help="Optional explicit per-query JSONL override.")
    args = parser.parse_args()
    plot(
        args.summary_json,
        args.output,
        compare_top_k=args.compare_top_k,
        per_query_jsonl=args.per_query_jsonl,
    )
    print(f"saved = {args.output.resolve()}")


if __name__ == "__main__":
    main()
