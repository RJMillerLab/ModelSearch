#!/usr/bin/env python3
"""Plot aggregate nugget evaluation figures from batch summary JSON."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

METHOD_LABELS = {
    "sparse": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid",
    "keyword": "Keyword",
    "single_column": "Joinable",
    "unionable": "Unionable",
}


def _load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    aggregate = data.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError(f"Missing aggregate object in {path}")
    return data


def plot(summary_path: Path, output_path: Path, *, compare_top_k: int | None) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))
    import matplotlib.pyplot as plt
    import numpy as np

    summary = _load_summary(summary_path)
    aggregate = summary["aggregate"]
    methods = [m for m in aggregate["method_order"] if m in METHOD_LABELS]
    top_k_values = [int(k) for k in aggregate["top_k_values"]]
    compare_k = compare_top_k or max(top_k_values)
    compare_key = f"top_{compare_k}"
    num_queries = int(aggregate.get("num_queries", 0))

    matrix = np.array(
        [
            [float(aggregate["mean_scores"].get(method, {}).get(f"top_{k}", 0.0)) for k in top_k_values]
            for method in methods
        ]
    )

    fig = plt.figure(figsize=(13.5, 6.4), dpi=180)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.18], wspace=0.33)
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_bar = fig.add_subplot(grid[0, 1])

    im = ax_heat.imshow(matrix, cmap="GnBu", vmin=0, vmax=max(1.0, float(matrix.max()) if matrix.size else 1.0))
    ax_heat.set_title(f"A. Mean nugget score over {num_queries} queries", fontsize=13, weight="bold", pad=14)
    ax_heat.set_xticks(range(len(top_k_values)), [f"top-{k}" for k in top_k_values], fontsize=10)
    ax_heat.set_yticks(range(len(methods)), [METHOD_LABELS[m] for m in methods], fontsize=10)
    ax_heat.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    for i in range(len(methods)):
        for j in range(len(top_k_values)):
            ax_heat.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="#162033", fontsize=10)
    ax_heat.set_xticks(np.arange(-0.5, len(top_k_values), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.4)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    if "hybrid" in methods and "keyword" in methods:
        cut = methods.index("keyword") - 0.5
        ax_heat.axhline(cut, color="#6b6b6b", linestyle=(0, (4, 3)), linewidth=1.0)
        ax_heat.text(-1.38, 1, "Semantic\nsearch", ha="center", va="center", fontsize=10)
        ax_heat.text(-1.38, 4, "Table\nsearch", ha="center", va="center", fontsize=10)
    cbar = fig.colorbar(im, ax=ax_heat, orientation="horizontal", fraction=0.08, pad=0.12)
    cbar.set_label("Mean nugget score (higher is better)", fontsize=9)

    table_methods = [m for m in aggregate["table_methods"] if m in METHOD_LABELS]
    y = np.arange(len(table_methods))
    colors = {"win": "#2f9b57", "tie": "#d8d8d8", "loss": "#f05a50"}
    left = np.zeros(len(table_methods))
    for label in ("win", "tie", "loss"):
        values = [
            int(aggregate["win_tie_loss_vs_best_semantic"].get(method, {}).get(compare_key, {}).get(label, 0))
            for method in table_methods
        ]
        ax_bar.barh(y, values, left=left, color=colors[label], edgecolor="white", label=label.capitalize())
        for yi, val, lft in zip(y, values, left):
            if val > 0:
                txt_color = "white" if label in {"win", "loss"} else "#222"
                ax_bar.text(lft + val / 2, yi, str(val), ha="center", va="center", color=txt_color, fontsize=10)
        left += np.array(values)
    ax_bar.set_title(
        f"B. Query-wise win count over {num_queries} queries\n(vs best semantic baseline, top-{compare_k})",
        fontsize=13,
        weight="bold",
        pad=14,
    )
    ax_bar.set_yticks(y, [METHOD_LABELS[m] for m in table_methods], fontsize=10)
    ax_bar.set_xlabel(f"Number of queries (out of {num_queries})", fontsize=10)
    ax_bar.set_xlim(0, max(num_queries, int(left.max()) if len(left) else 1))
    ax_bar.invert_yaxis()
    ax_bar.grid(axis="x", color="#dddddd", linestyle="--", linewidth=0.7)
    ax_bar.set_axisbelow(True)
    ax_bar.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False)

    fig.suptitle("Nugget-based comparison of model-search and table-search methods", fontsize=14, weight="bold", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot batch nugget evaluation summary.")
    parser.add_argument("--summary-json", required=True, type=Path, help="aggregate_summary.json from run_batch_eval.py")
    parser.add_argument("--output", required=True, type=Path, help="Output image path, e.g. figure.png")
    parser.add_argument("--compare-top-k", type=int, default=None, help="Top-k budget for win/tie/loss panel. Default: largest top-k.")
    args = parser.parse_args()
    plot(args.summary_json, args.output, compare_top_k=args.compare_top_k)
    print(f"saved = {args.output.resolve()}")


if __name__ == "__main__":
    main()
