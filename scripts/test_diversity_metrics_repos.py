"""
Quick sanity-check script for external diversity-metric repos.

Goal:
- Compare two ranked search result lists (e.g. two models' search outputs)
  using existing Python libraries, similar in spirit to scikit-learn.metrics.
- This script does NOT require ground-truth relevance; it uses:
    - RecTools: Intra-list diversity (item-feature-based).
    - Optionally prints notes about pyndeval / FairDiverse, which are mainly
      designed for TREC-style evaluation with qrels.

Usage (from repo root):
    python scripts/test_diversity_metrics_repos.py \\
        --results-a m1 m2 m3 m4 \\
        --results-b m3 m5 m6 \\
        --item-features-json path/to/item_features.json

Where item_features_json is an optional JSON mapping:
    {
      "m1": {"feat_a": 1, "feat_b": 0},
      "m2": {"feat_a": 0, "feat_b": 1},
      ...
    }
If omitted, we generate random binary features just to demonstrate the APIs.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _load_item_features(path: Optional[str], item_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Load item features for RecTools from JSON.

    If path is None, create random binary features to make the example runnable.
    """
    if path is None:
        _log("⚠️  No --item-features-json provided; generating random toy features.")
        rng = np.random.default_rng(0)
        features: Dict[str, Dict[str, float]] = {}
        for mid in item_ids:
            features[mid] = {
                "feat_1": float(rng.integers(0, 2)),
                "feat_2": float(rng.integers(0, 2)),
                "feat_3": float(rng.integers(0, 2)),
            }
        return features

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("item_features_json must be a JSON object mapping item_id -> feature dict")
    return {str(k): dict(v) for k, v in data.items()}


def run_rectools_intra_list_diversity(
    results_a: List[str],
    results_b: List[str],
    item_features: Dict[str, Dict[str, float]],
    k: int = 10,
) -> None:
    """
    Use RecTools' IntraListDiversity to compare two ranked lists.

    This is the closest existing thing to a "scikit-learn style" diversity
    metric for no-ground-truth settings.
    """
    try:
        from rectools import Columns
        from rectools.metrics import IntraListDiversity
        from rectools.metrics.distances import PairwiseHammingDistanceCalculator
        import pandas as pd
    except ImportError:
        _log("❌ RecTools not installed. Install with: pip install rectools")
        return

    all_items = sorted({*results_a, *results_b})
    # Build feature matrix for items
    rows = []
    for mid in all_items:
        feats = item_features.get(mid, {})
        row = {"item": mid}
        for kf, vf in feats.items():
            row[str(kf)] = float(vf)
        rows.append(row)
    if not rows:
        _log("No items to evaluate; skipping RecTools example.")
        return
    features_df = pd.DataFrame(rows).set_index("item")

    # Build recommendation DataFrame expected by RecTools:
    # Columns: user, item, rank. We'll treat the two result lists as from
    # two pseudo-users "A" and "B".
    reco_rows = []
    for u_label, res in [("A", results_a), ("B", results_b)]:
        for rank, mid in enumerate(res[:k], start=1):
            reco_rows.append(
                {
                    Columns.User: u_label,
                    Columns.Item: mid,
                    Columns.Rank: rank,
                }
            )
    reco_df = pd.DataFrame(reco_rows)

    # Distance calculator + metric
    dist_calc = PairwiseHammingDistanceCalculator(features_df)
    metric = IntraListDiversity(k=min(k, len(all_items)), distance_calculator=dist_calc)

    per_user = metric.calc_per_user(reco_df)
    overall = metric.calc(reco_df)

    _log("\n=== RecTools: IntraListDiversity ===")
    _log(f"Users: {list(per_user.index)}")
    for u, val in per_user.items():
        _log(f"  User {u}: ILD@{k} = {val:.4f}")
    _log(f"Overall mean ILD@{k}: {overall:.4f}")


# ---------------------------------------------------------------------------
# Helpers for adapting our pipeline's search_results.json format
# ---------------------------------------------------------------------------

def _load_search_results_json(path: str) -> Dict:
    """Load a search_results.json file produced by src/demo/backend.py."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("search_results.json must contain a JSON object")
    return data


def _normalize_model_id_list(raw_list) -> List[str]:
    """
    Normalize a list that may contain strings or dicts with model_id/modelId.
    """
    out: List[str] = []
    if not isinstance(raw_list, list):
        return out
    for item in raw_list:
        if isinstance(item, str):
            mid = item
        elif isinstance(item, dict):
            mid = item.get("model_id") or item.get("modelId") or ""
        else:
            mid = str(item)
        mid = str(mid).strip()
        if mid:
            out.append(mid)
    return out


def extract_query2modelcard_models(
    search_results: Dict,
    mode: str = "dense",
) -> List[str]:
    """
    Extract a ranked model list from query2modelcard neighbor lists in search_results.json.

    Args:
        search_results: Loaded JSON from search_results.json.
        mode: One of {"dense", "sparse", "hybrid"}.

    Returns:
        List of model IDs (strings) in ranked order.
    """
    mode = (mode or "dense").strip().lower()
    all_modes = search_results.get("query2modelcard_all_modes") or {}
    raw = all_modes.get(mode)
    if isinstance(raw, dict) and "error" in raw:
        raw = []
    models = _normalize_model_id_list(raw) if raw is not None else []
    if not models:
        primary = search_results.get("query2modelcard_results")
        models = _normalize_model_id_list(primary)
    return models


def extract_card2tab2card_models(
    search_results: Dict,
    search_type: str = "keyword",
) -> List[str]:
    """
    Extract a ranked model list from card2tab2card results.

    Args:
        search_results: Loaded JSON from search_results.json.
        search_type: One of the keys in card2tab2card_results, e.g.
                     "keyword", "single_column", "unionable", "by_type".

    Returns:
        List of model IDs (strings) in ranked order.
    """
    search_type = (search_type or "keyword").strip()
    c2t2c = search_results.get("card2tab2card_results") or {}
    stub = c2t2c.get(search_type)
    if isinstance(stub, dict):
        raw = stub.get("model_ids", [])
    elif isinstance(stub, list):
        raw = stub
    else:
        raw = []
    return _normalize_model_id_list(raw)


def load_results_from_search_results_json(
    path: str,
    system_a_source: str,
    system_b_source: str,
) -> Tuple[List[str], List[str]]:
    """
    Convert our pipeline's search_results.json into two ranked lists.

    This does NOT change the script's CLI. It is intended to be imported from
    Python or used in notebooks. Source strings are of the form:

        "query2modelcard:dense"
        "query2modelcard:sparse"
        "query2modelcard:hybrid"
        "card2tab2card:keyword"
        "card2tab2card:single_column"
        "card2tab2card:unionable"
        "card2tab2card:by_type"

    Example:
        from scripts.test_diversity_metrics_repos import (
            load_results_from_search_results_json,
            run_rectools_intra_list_diversity,
        )

        a, b = load_results_from_search_results_json(
            "data/jobs/<job_id>/search_results.json",
            system_a_source="query2modelcard:dense",
            system_b_source="card2tab2card:keyword",
        )
        # prepare item_features dict...
        run_rectools_intra_list_diversity(a, b, item_features, k=10)
    """
    data = _load_search_results_json(path)

    def _parse_source(src: str) -> Tuple[str, str]:
        parts = (src or "").split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid source spec '{src}'. Expected kind:arg, e.g. query2modelcard:dense")
        return parts[0].strip().lower(), parts[1].strip()

    kind_a, arg_a = _parse_source(system_a_source)
    kind_b, arg_b = _parse_source(system_b_source)

    if kind_a == "query2modelcard":
        res_a = extract_query2modelcard_models(data, mode=arg_a)
    elif kind_a == "card2tab2card":
        res_a = extract_card2tab2card_models(data, search_type=arg_a)
    else:
        raise ValueError(
            f"Unknown system_a_source kind '{kind_a}' (expected 'query2modelcard' or 'card2tab2card')"
        )

    if kind_b == "query2modelcard":
        res_b = extract_query2modelcard_models(data, mode=arg_b)
    elif kind_b == "card2tab2card":
        res_b = extract_card2tab2card_models(data, search_type=arg_b)
    else:
        raise ValueError(
            f"Unknown system_b_source kind '{kind_b}' (expected 'query2modelcard' or 'card2tab2card')"
        )

    return res_a, res_b


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Test external diversity-metric repos on two search result lists.")
    parser.add_argument(
        "--results-a",
        nargs="+",
        required=True,
        help="Ranked item IDs for search system A (e.g. model IDs).",
    )
    parser.add_argument(
        "--results-b",
        nargs="+",
        required=True,
        help="Ranked item IDs for search system B.",
    )
    parser.add_argument(
        "--item-features-json",
        type=str,
        default=None,
        help="Optional JSON file: {item_id: {feat_name: value, ...}}. "
        "If omitted, random features will be generated.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-k cutoff for diversity metrics.",
    )

    args = parser.parse_args(argv)

    results_a = [str(x) for x in args.results_a]
    results_b = [str(x) for x in args.results_b]

    _log(f"System A results (top {len(results_a)}): {results_a}")
    _log(f"System B results (top {len(results_b)}): {results_b}")

    item_ids = sorted({*results_a, *results_b})
    item_features = _load_item_features(args.item_features_json, item_ids)

    # 1) RecTools intra-list diversity
    run_rectools_intra_list_diversity(results_a, results_b, item_features, k=args.k)

    # 2) Optional: just note about other repos
    _log("\n=== Notes on other repos ===")
    try:
        import pyndeval  # type: ignore  # noqa: F401

        _log("✅ pyndeval is installed (TREC-style diversity metrics with qrels).")
        _log("   However, it requires qrels + runs files (ground truth), so it is not used in this no-qrels demo.")
    except ImportError:
        _log("ℹ️  pyndeval not installed. It targets TREC diversity metrics with qrels (not needed for this no-qrels demo).")

    try:
        import fairdiverse  # type: ignore  # noqa: F401

        _log("✅ FairDiverse is installed (fairness & diversity toolkit).")
        _log("   It is designed as a full experimental framework; integrating it into this tiny demo is overkill, ")
        _log("   but you can explore its docs for more advanced fairness/diversity experiments.")
    except ImportError:
        _log("ℹ️  FairDiverse not installed. It is a larger benchmark toolkit rather than a small metrics module.")


if __name__ == "__main__":
    main()

