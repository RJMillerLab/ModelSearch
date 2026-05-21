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
    "single_column": "Joinable",
    "unionable": "Unionable",
}

METHOD_ORDER = ["sparse", "dense", "hybrid", "keyword", "single_column", "unionable"]
SEMANTIC_METHODS = ("sparse", "dense", "hybrid")
TABLE_METHODS = ("keyword", "single_column", "unionable")
DIST_GROUP_SEPARATOR_Y = 3.5
RANK_GROUP_SEPARATOR_Y = 2.5


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


def _resolve_per_query_path(summary_path: Path, summary_per_query_path: str, explicit_path: Path | None) -> Path | None:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    if summary_per_query_path.strip():
        raw = Path(summary_per_query_path.strip())
        candidates.append(raw)
        candidates.append(summary_path.parent / raw.name)
    candidates.append(summary_path.parent / "per_query_method_scores.jsonl")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


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
) -> tuple[dict[str, list[int]], dict[str, int]]:
    rank_counts: dict[str, list[int]] = {method: [0] * len(METHOD_ORDER) for method in METHOD_ORDER}
    availability: dict[str, int] = {method: 0 for method in METHOD_ORDER}
    for record in records:
        scored: list[tuple[str, float]] = []
        for method in METHOD_ORDER:
            score = _method_score(record, method, compare_key, score_field)
            if score is None:
                continue
            availability[method] += 1
            scored.append((method, score))
        scored.sort(key=lambda item: (-item[1], METHOD_ORDER.index(item[0])))
        for rank_idx, (method, _) in enumerate(scored):
            if rank_idx < len(METHOD_ORDER):
                rank_counts[method][rank_idx] += 1
    return rank_counts, availability


def _blue_palette() -> list[str]:
    return ["#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#2563eb", "#1d4ed8"]


def _red_palette() -> list[str]:
    return ["#fef2f2", "#fee2e2", "#fecaca", "#fca5a5", "#f87171", "#b91c1c"]


def _red_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "rank_red",
        ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#de2d26", "#a50f15"],
    )


def _render_distribution_panel(
    ax_dist,
    *,
    records: list[dict[str, Any]],
    compare_key: str,
    score_field: str,
    compare_k: int,
    show_method_labels: bool = True,
    show_xlabel: bool = True,
    show_title: bool = True,
) -> None:
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    # Nugget counts are integers in the real evaluation output. We keep the
    # histogram bins aligned to whole counts so the demo and real runs share
    # the same layout.
    method_scores = _collect_scores(records, compare_key=compare_key, score_field=score_field, integer_bins=True)

    all_scores = [v for vals in method_scores.values() for v in vals]
    if all_scores:
        lo = int(np.floor(min(all_scores)))
        hi = int(np.ceil(max(all_scores)))
    else:
        lo, hi = 0, 1
    lo = max(0, lo - 1)
    hi = hi + 1

    blue_cmap = LinearSegmentedColormap.from_list("hist_blue", _blue_palette())
    dist_data = [np.log1p(np.array(method_scores[method], dtype=float)) for method in METHOD_ORDER]

    positions = np.arange(1, len(METHOD_ORDER) + 1)
    parts = ax_dist.violinplot(
        dist_data,
        positions=positions,
        vert=False,
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        points=200,
        bw_method=0.28,
    )
    for idx, body in enumerate(parts["bodies"]):
        color = blue_cmap(0.25 + 0.55 * (idx / max(1, len(METHOD_ORDER) - 1)))
        body.set_facecolor(color)
        body.set_edgecolor("#93c5fd")
        body.set_alpha(0.96)
        body.set_linewidth(0.9)
    if "cmedians" in parts:
        parts["cmedians"].set_color("#1d4ed8")
        parts["cmedians"].set_linewidth(1.2)

    mean_vals = [float(np.mean(np.array(method_scores[method], dtype=float))) if len(method_scores[method]) else 0.0 for method in METHOD_ORDER]
    for pos, mean_val in zip(positions, mean_vals):
        log_mean = float(np.log1p(mean_val))
        ax_dist.plot([log_mean, log_mean], [pos - 0.18, pos + 0.18], color="#7f0000", linestyle=(0, (4, 3)), linewidth=1.5)
        ax_dist.annotate(
            f"Avg.={mean_val:.1f}",
            xy=(log_mean, pos),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.0,
            color="#7f0000",
            clip_on=False,
        )
    max_log = float(np.log1p(max(1.0, hi)))
    ax_dist.set_xlim(0.0, max(1.0, max_log * 1.08))
    tick_vals = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    tick_vals = [v for v in tick_vals if v <= hi]
    ax_dist.set_xticks([np.log1p(v) for v in tick_vals], [str(v) for v in tick_vals], fontsize=8.5)
    ax_dist.set_yticks(positions, [METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=8.0)
    if not show_method_labels:
        ax_dist.set_yticklabels([])
    ax_dist.set_xlabel("Nugget count (log1p scale)" if show_xlabel else "", fontsize=9)
    ax_dist.grid(axis="x", color="#eeeeee", linestyle="--", linewidth=0.6)
    ax_dist.set_axisbelow(True)
    ax_dist.tick_params(axis="y", pad=4)
    ax_dist.set_ylim(0.5, len(METHOD_ORDER) + 0.5)
    ax_dist.invert_yaxis()
    ax_dist.margins(y=0.06)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.axhline(DIST_GROUP_SEPARATOR_Y, color="#6b6b6b", linestyle=(0, (4, 3)), linewidth=1.0)
    if show_title:
        ax_dist.set_title(f"Top-{compare_k}", fontsize=11, weight="bold", pad=10)


def _render_rank_panel(
    ax_rank,
    *,
    records: list[dict[str, Any]],
    compare_key: str,
    score_field: str,
    compare_k: int,
    show_method_labels: bool = True,
    show_xlabel: bool = True,
    show_title: bool = True,
) -> tuple[list[Any], list[Any]]:
    import numpy as np

    _, availability = _collect_rank_counts(records, compare_key=compare_key, score_field=score_field)
    rank_counts, _ = _collect_rank_counts(records, compare_key=compare_key, score_field=score_field)

    rank_matrix = np.array(
        [
            [
                rank_counts[method][rank_idx] / max(1, availability[method])
                for rank_idx in range(len(METHOD_ORDER))
            ]
            for method in METHOD_ORDER
        ],
        dtype=float,
    )
    rank_colors = list(reversed(_red_palette()))
    y = np.arange(len(METHOD_ORDER))
    left_edge = np.zeros(len(METHOD_ORDER), dtype=float)
    containers = []
    for rank_idx in range(len(METHOD_ORDER)):
        vals = rank_matrix[:, rank_idx]
        color = rank_colors[rank_idx]
        bars = ax_rank.barh(
            y,
            vals,
            left=left_edge,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            height=0.52,
            label=f"Rank {rank_idx + 1}",
        )
        containers.append(bars)
        for row_idx, val in enumerate(vals):
            pct = val * 100.0
            if pct >= 8.0:
                ax_rank.text(
                    left_edge[row_idx] + val / 2.0,
                    row_idx,
                    f"{pct:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8.3,
                    color="#111111",
                )
        left_edge += vals

    ax_rank.set_yticks(y, [METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=8.0)
    if not show_method_labels:
        ax_rank.set_yticklabels([])
    ax_rank.set_xlim(0.0, 1.0)
    ax_rank.set_xticks(np.linspace(0.0, 1.0, 6), [f"{int(v * 100)}%" for v in np.linspace(0.0, 1.0, 6)], fontsize=9)
    ax_rank.tick_params(axis="y", length=0, pad=2)
    ax_rank.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    ax_rank.invert_yaxis()
    ax_rank.grid(axis="x", color="#eeeeee", linestyle="--", linewidth=0.7)
    ax_rank.set_axisbelow(True)
    ax_rank.axhline(RANK_GROUP_SEPARATOR_Y, color="#6b6b6b", linestyle=(0, (4, 3)), linewidth=1.0)
    ax_rank.set_xlabel(f"Top-{compare_k}" if show_xlabel else "", fontsize=9, labelpad=10)
    if show_title:
        ax_rank.set_title(f"Top-{compare_k}", fontsize=11, weight="bold", pad=10)
    return containers, [bars[0] for bars in containers]


def plot(
    summary_path: Path,
    output_path: Path,
    *,
    compare_top_ks: list[int] | None,
    per_query_jsonl: Path | None,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))
    import matplotlib.pyplot as plt

    summary = _load_summary(summary_path)
    aggregate = summary["aggregate"]
    available_top_ks = [int(k) for k in aggregate["top_k_values"]]
    if compare_top_ks:
        compare_ks = [k for k in compare_top_ks if k in available_top_ks]
        if not compare_ks:
            raise ValueError(f"None of the requested top-k values {compare_top_ks} exist in summary top_k_values={available_top_ks}")
    else:
        compare_ks = available_top_ks
    score_field = str(aggregate.get("score_field", "hit_dedup"))

    summary_per_query_path = str(summary.get("paths", {}).get("per_query_method_scores", "") or "").strip()
    source_path = _resolve_per_query_path(summary_path, summary_per_query_path, per_query_jsonl)
    records = _load_per_query_records(source_path)
    if not records:
        raise RuntimeError(
            "No per-query records found. Provide --per-query-jsonl or place per_query_method_scores.jsonl next to the summary JSON."
        )

    n_cols = len(compare_ks)
    fig_height = 6.0
    fig_width = max(13.2, 3.0 * n_cols)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=180)
    outer = fig.add_gridspec(2, n_cols, hspace=0.34, wspace=0.10)

    for col_idx, compare_k in enumerate(compare_ks):
        compare_key = f"top_{compare_k}"
        ax_dist = fig.add_subplot(outer[0, col_idx])
        ax_rank = fig.add_subplot(outer[1, col_idx])
        _render_distribution_panel(
            ax_dist,
            records=records,
            compare_key=compare_key,
            score_field=score_field,
            compare_k=compare_k,
            show_method_labels=(col_idx == 0),
            show_xlabel=False,
            show_title=False,
        )
        _render_rank_panel(
            ax_rank,
            records=records,
            compare_key=compare_key,
            score_field=score_field,
            compare_k=compare_k,
            show_method_labels=(col_idx == 0),
            show_xlabel=True,
            show_title=False,
        )
        if col_idx > 0:
            ax_dist.tick_params(axis="y", labelleft=False)
            ax_rank.tick_params(axis="y", labelleft=False)

    fig.text(0.5, 0.975, "Nugget Count Distribution and Query-Level Rank Outcomes by Top-k", ha="center", va="top", fontsize=14, weight="bold")
    fig.text(0.5, 0.915, "Distribution of nugget count", ha="center", va="center", fontsize=11, weight="bold")
    fig.text(0.5, 0.455, "Query-level rank outcomes", ha="center", va="center", fontsize=11, weight="bold")

    from matplotlib.patches import Patch

    legend_handles = [Patch(facecolor=color, edgecolor="white", label=f"Rank {idx + 1}") for idx, color in enumerate(reversed(_red_palette()))]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=6,
        frameon=False,
        fontsize=8.5,
        columnspacing=0.8,
        handlelength=1.8,
    )
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.085, right=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot nugget score distributions and rank share heatmaps.")
    parser.add_argument("--summary-json", required=True, type=Path, help="aggregate_summary.json from run_batch_eval.py")
    parser.add_argument("--output", required=True, type=Path, help="Output image path, e.g. figure.png")
    parser.add_argument(
        "--compare-top-k",
        type=int,
        nargs="*",
        default=None,
        help="Top-k budgets to plot. Default: all top-k values in the summary.",
    )
    parser.add_argument("--per-query-jsonl", type=Path, default=None, help="Optional explicit per-query JSONL override.")
    args = parser.parse_args()
    plot(
        args.summary_json,
        args.output,
        compare_top_ks=args.compare_top_k,
        per_query_jsonl=args.per_query_jsonl,
    )
    print(f"saved = {args.output.resolve()}")


if __name__ == "__main__":
    main()
