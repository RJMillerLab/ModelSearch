"""
Evaluate diversity between two search result lists from a job folder.

This script is a thin wrapper around the diversity logic, focused only on
the *retrieved search results* (no integrated tables).

It reads:
    data/jobs/<job_id>/search_results.json

and extracts two ranked model lists, for example:
    - System A: query2modelcard dense neighbors (query2modelcard:dense)
    - System B: Card2Tab2Card keyword mode    (card2tab2card:keyword)

Then it uses RecTools' IntraListDiversity metric to compare their diversity.

Usage examples (from repo root):

    python scripts/eval_search_diversity_from_job.py \\
        --job-id 2025-01-01_12-00-00_abcd \\
        --system-a query2modelcard:dense \\
        --system-b card2tab2card:keyword \\
        --k 20

You can optionally pass a JSON with item features to get more meaningful
distances; otherwise random toy features are generated:

    python scripts/eval_search_diversity_from_job.py \\
        --job-id 2025-01-01_12-00-00_abcd \\
        --system-a query2modelcard:dense \\
        --system-b card2tab2card:keyword \\
        --item-features-json path/to/model_features.json

The features JSON should look like:

    {
      "model_id_1": {"feat_a": 1, "feat_b": 0},
      "model_id_2": {"feat_a": 0, "feat_b": 1},
      ...
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON at {path} must be an object")
    return data


def _normalize_model_id_list(raw) -> List[str]:
    """Normalize items that may be strings or dicts with model_id/modelId."""
    out: List[str] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
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


def extract_query2modelcard_models(search_results: Dict, mode: str = "dense") -> List[str]:
    """Extract ranked model_ids from query2modelcard neighbor lists."""
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


def extract_card2tab2card_models(search_results: Dict, search_type: str = "keyword") -> List[str]:
    """Extract ranked model_ids from card2tab2card results."""
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
    Convert search_results.json into two ranked model lists.

    system_*_source syntax:
        "query2modelcard:dense"
        "query2modelcard:sparse"
        "query2modelcard:hybrid"
        "card2tab2card:keyword"
        "card2tab2card:single_column"
        "card2tab2card:unionable"
        "card2tab2card:by_type"
    """
    data = _load_json(path)

    def _parse(src: str) -> Tuple[str, str]:
        parts = (src or "").split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid source '{src}'. Expected kind:arg, e.g. query2modelcard:dense")
        return parts[0].strip().lower(), parts[1].strip()

    kind_a, arg_a = _parse(system_a_source)
    kind_b, arg_b = _parse(system_b_source)

    if kind_a == "query2modelcard":
        a = extract_query2modelcard_models(data, mode=arg_a)
    elif kind_a == "card2tab2card":
        a = extract_card2tab2card_models(data, search_type=arg_a)
    else:
        raise ValueError(f"Unknown system_a kind '{kind_a}'")

    if kind_b == "query2modelcard":
        b = extract_query2modelcard_models(data, mode=arg_b)
    elif kind_b == "card2tab2card":
        b = extract_card2tab2card_models(data, search_type=arg_b)
    else:
        raise ValueError(f"Unknown system_b kind '{kind_b}'")

    return a, b


def _load_item_features(path: Optional[str], item_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Load item features JSON or generate random features if not provided.
    """
    if path is None:
        _log("⚠️  No --item-features-json provided; generating random toy features.")
        rng = np.random.default_rng(0)
        feats: Dict[str, Dict[str, float]] = {}
        for mid in item_ids:
            feats[mid] = {
                "feat_1": float(rng.integers(0, 2)),
                "feat_2": float(rng.integers(0, 2)),
                "feat_3": float(rng.integers(0, 2)),
            }
        return feats

    data = _load_json(path)
    return {str(k): dict(v) for k, v in data.items()}


def run_rectools_intra_list_diversity(
    results_a: List[str],
    results_b: List[str],
    item_features: Dict[str, Dict[str, float]],
    k: int = 10,
) -> None:
    """Use RecTools' IntraListDiversity to compare two ranked lists."""
    try:
        from rectools import Columns
        from rectools.metrics import IntraListDiversity
        from rectools.metrics.distances import PairwiseHammingDistanceCalculator
        import pandas as pd
    except ImportError:
        _log("❌ RecTools not installed. Install with: pip install rectools")
        return

    all_items = sorted({*results_a, *results_b})
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

    dist_calc = PairwiseHammingDistanceCalculator(features_df)
    metric = IntraListDiversity(k=min(k, len(all_items)), distance_calculator=dist_calc)

    per_user = metric.calc_per_user(reco_df)
    overall = metric.calc(reco_df)

    _log("\n=== RecTools: IntraListDiversity (from job search_results.json) ===")
    _log(f"Users: {list(per_user.index)}")
    for u, val in per_user.items():
        _log(f"  User {u}: ILD@{k} = {val:.4f}")
    _log(f"Overall mean ILD@{k}: {overall:.4f}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare diversity of two search result lists from a data/jobs/<job_id>/search_results.json."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--job-id",
        type=str,
        help="Job ID under data/jobs/<job_id>/search_results.json",
    )
    group.add_argument(
        "--job-dir",
        type=str,
        help="Full path to job directory that contains search_results.json",
    )
    parser.add_argument(
        "--system-a",
        required=True,
        help="Source spec for system A, e.g. query2modelcard:dense or card2tab2card:keyword",
    )
    parser.add_argument(
        "--system-b",
        required=True,
        help="Source spec for system B, e.g. query2modelcard:hybrid or card2tab2card:single_column",
    )
    parser.add_argument(
        "--item-features-json",
        type=str,
        default=None,
        help="Optional JSON with item features; if omitted, random features are used.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Top-k cutoff for ILD.",
    )

    args = parser.parse_args(argv)

    if args.job_dir:
        job_dir = args.job_dir
    else:
        job_dir = os.path.join("data", "jobs", args.job_id)
    search_path = os.path.join(job_dir, "search_results.json")

    if not os.path.isfile(search_path):
        raise FileNotFoundError(f"search_results.json not found at {search_path}")

    _log(f"Using search_results.json: {search_path}")
    _log(f"System A source: {args.system_a}")
    _log(f"System B source: {args.system_b}")

    res_a, res_b = load_results_from_search_results_json(
        search_path, system_a_source=args.system_a, system_b_source=args.system_b
    )
    _log(f"System A results (len={len(res_a)}): {res_a[:10]}{'...' if len(res_a) > 10 else ''}")
    _log(f"System B results (len={len(res_b)}): {res_b[:10]}{'...' if len(res_b) > 10 else ''}")

    item_ids = sorted({*res_a, *res_b})
    item_features = _load_item_features(args.item_features_json, item_ids)

    run_rectools_intra_list_diversity(res_a, res_b, item_features, k=args.k)


if __name__ == "__main__":
    main()

