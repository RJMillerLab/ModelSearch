"""
Table→table search, augmented with transpose.

Pipeline (always the same):

1. Resolve the query to one on-disk CSV path.
2. Build a transposed copy of that CSV (Blend transposed-corpus layout).
3. Call Blend ``search_table2table`` four times — every combination of
   (original vs transposed query) × (candidate table_type filter derived from ori/tr vs tr).
4. Merge the four ranked lists with RRF. ``*_t.csv`` / ``*_s.csv`` hits are
   folded to the same key as ``<group>.csv`` (Blend index ``table_group``).
5. Optionally write **one** JSON: four intermediate ranked lists (``lane_rankings``),
   RRF breakdown (``rrf_by_basename``), and the postprocessed list (``merged_ranking``).
   Per-lane files are not written (inner ``search_table2table(..., output_json=None)``).
6. Return ``merged_ranking`` (canonical ``*.csv`` basenames).  Compatible with tab2tab
   in the sense: same ``search_type`` / ``k`` / ``db_path`` semantics per lane; return
   type is still ``List[str]``, but names are canonicalized and RRF-merged across four runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.config import (
    BLEND_INTERNAL_REPO,
    MODELLAKE_DB_HUGGING,
    OUTPUT_DIR,
    TAB2TAB_AUG_OUTPUT_JSON,
)
from src.search.tab2tab import search_table2table
from src.utils import resolve_table_path

# Labels for the four Blend runs (also JSON keys under "lane_rankings").
LANE_RANKING_KEYS = (
    "original_query__original_db",
    "original_query__transposed_db",
    "transposed_query__original_db",
    "transposed_query__transposed_db",
)


def rrf_k() -> int:
    return int(os.environ.get("TAB2TAB_AUG_RRF_K", "60"))


def canonical_modellake_basename(basename: str) -> str:
    """
    ``foo_t.csv`` / ``foo_s.csv`` → ``foo.csv`` (same rule as Blend
    ``create_index_duckdb.get_group_and_type``).
    """
    bn = basename.strip()
    if bn.endswith("_t.csv"):
        return bn[:-6] + ".csv"
    if bn.endswith("_s.csv"):
        return bn[:-6] + ".csv"
    return bn


def resolve_query_table_path(query: str) -> str:
    """
    Same resolution rule as ``tab2tab.search_table2table`` (basename → TABLE_BASE_DIRS,
    else any path that exists).
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    q = query.strip()
    p = resolve_table_path(q) or (q if os.path.exists(q) else None)
    if not p:
        raise ValueError(f"No table path for query {query!r} (same rules as tab2tab).")
    return os.path.abspath(p)


def write_transposed_query_csv(source_csv: str, dest_csv: str) -> None:
    """Transpose ``source_csv`` into ``dest_csv`` (matches transposed-corpus indexing)."""
    df = pd.read_csv(source_csv)
    t = df.transpose().reset_index()
    t.columns = [str(c) if str(c).strip() else f"col_{i}" for i, c in enumerate(t.columns)]
    parent = os.path.dirname(os.path.abspath(dest_csv))
    if parent:
        os.makedirs(parent, exist_ok=True)
    t.to_csv(dest_csv, index=False)


def default_transposed_query_csv_path(source_csv: str) -> str:
    stem = os.path.splitext(os.path.basename(source_csv))[0]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:80]
    h = hashlib.sha256(os.path.abspath(source_csv).encode("utf-8")).hexdigest()[:16]
    out_dir = os.path.join(OUTPUT_DIR, "tab2tab_aug_transposed_queries")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{safe}_{h}_T.csv")


def rrf_merge_lane_rankings(
    lane_rankings: Dict[str, List[str]],
    *,
    exclude_basenames: Sequence[str],
    top_k: int,
    k_rrf: int,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    One RRF pass over the four lists.  Keys are canonical basenames; each lane
    contributes at most one rank per key (best rank if aliases appear).
    """
    raw_ex = {str(x).strip() for x in exclude_basenames if str(x).strip()}
    canon_ex = {canonical_modellake_basename(x) for x in raw_ex}

    best_rank_per_lane: Dict[str, Dict[str, int]] = defaultdict(dict)
    for lane_name, names in lane_rankings.items():
        for rank, name in enumerate(names, start=1):
            bn = str(name).strip()
            if not bn or bn in raw_ex:
                continue
            key = canonical_modellake_basename(bn)
            if key in canon_ex:
                continue
            lane_map = best_rank_per_lane[key]
            prev = lane_map.get(lane_name)
            if prev is None or rank < prev:
                lane_map[lane_name] = rank

    scored: List[Tuple[str, int, float, int]] = []
    rrf_detail: Dict[str, Any] = {}
    for basename, lane_to_rank in best_rank_per_lane.items():
        pairs = list(lane_to_rank.items())
        n_lanes = len(pairs)
        rrf = sum(1.0 / (k_rrf + r) for _, r in pairs)
        min_r = min(r for _, r in pairs)
        scored.append((basename, n_lanes, rrf, min_r))
        rrf_detail[basename] = {
            "lane_hits": n_lanes,
            "rrf": round(rrf, 6),
            "min_rank": min_r,
            "per_lane_rank": dict(pairs),
        }

    scored.sort(key=lambda x: (-x[1], -x[2], x[3]))
    final = [x[0] for x in scored[:top_k]]
    return final, rrf_detail


def search_tab2tab_aug(
    *,
    search_type: str,
    query: str,
    k: int,
    db_original: str,
    db_transposed: str,
    output_json: Optional[str] = None,
    transposed_query_csv: Optional[str] = None,
) -> List[str]:
    table_path = resolve_query_table_path(query)
    t0 = time.time()

    t_csv = transposed_query_csv or default_transposed_query_csv_path(table_path)
    write_transposed_query_csv(table_path, t_csv)
    query_bn = os.path.basename(table_path)
    transposed_query_bn = os.path.basename(t_csv)

    # We no longer switch to a separate "transposed DuckDB".
    # Instead, we filter candidate tables by `table_type` inside `db_original`
    # using `augmentation_types` (ori/tr/str).
    def lane_to_augmentation_types(lane_name: str) -> List[str]:
        return ["tr", "str"] if "transposed_db" in lane_name else ["ori", "str"]

    runs: List[Tuple[str, str]] = [
        (LANE_RANKING_KEYS[0], table_path),
        (LANE_RANKING_KEYS[1], table_path),
        (LANE_RANKING_KEYS[2], t_csv),
        (LANE_RANKING_KEYS[3], t_csv),
    ]
    lane_rankings: Dict[str, List[str]] = {}
    lane_augmentation_types: Dict[str, List[str]] = {}
    for label, q_path in runs:
        aug_types = lane_to_augmentation_types(label)
        lane_augmentation_types[label] = aug_types
        lane_rankings[label] = search_table2table(
            search_type=search_type,
            query=q_path,
            k=k,
            db_path=db_original,
            output_json=None,
            augmentation_types=aug_types,
        )

    final_ranking, rrf_by_basename = rrf_merge_lane_rankings(
        lane_rankings,
        exclude_basenames=(query_bn, transposed_query_bn),
        top_k=k,
        k_rrf=rrf_k(),
    )

    elapsed_s = round(time.time() - t0, 4)
    payload: Dict[str, Any] = {
        "version": 1,
        "search_type": search_type,
        "query_resolved": table_path,
        "transposed_query_csv": os.path.abspath(t_csv),
        "k_per_lane": k,
        "db_original": os.path.abspath(db_original),
        "db_transposed": os.path.abspath(db_transposed),
        "db_transposed_ignored": True,
        "lane_augmentation_types": lane_augmentation_types,
        "rrf_k": rrf_k(),
        "lane_rankings": lane_rankings,
        "rrf_by_basename": rrf_by_basename,
        "merged_ranking": final_ranking,
        "elapsed_s": elapsed_s,
    }

    if output_json:
        parent = os.path.dirname(os.path.abspath(output_json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return final_ranking


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Four tab2tab runs (query×DB transpose grid) + RRF merge → JSON."
    )
    parser.add_argument(
        "--search_type",
        choices=["single_column", "multi_column", "keyword", "unionable"],
        required=True,
    )
    parser.add_argument("--query", required=True, help="Basename (via TABLE_BASE_DIRS) or path to a CSV.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--resources",
        nargs="+",
        default=["hugging"],
        choices=["hugging"],
        help="Must be hugging; uses MODELLAKE_DB_HUGGING (candidate variants via augmentation_types).",
    )
    parser.add_argument(
        "--output_json",
        default="",
        help=f"Write full JSON. Default: {TAB2TAB_AUG_OUTPUT_JSON}",
    )
    parser.add_argument(
        "--transposed_query_csv",
        default="",
        help="Where to write the transposed query CSV (default: hashed file under OUTPUT_DIR).",
    )
    args = parser.parse_args()

    resources = [str(r).strip().lower() for r in args.resources if str(r).strip()]
    db_original = MODELLAKE_DB_HUGGING
    # Kept only for backward-compatible signature / payload; search ignores it.
    db_transposed = db_original

    out_path = args.output_json or TAB2TAB_AUG_OUTPUT_JSON
    t_wall = time.perf_counter()
    print(
        "[tab2tab_aug] "
        f"resources={resources!r} | db_original={os.path.abspath(db_original)} | "
        f"db_transposed={os.path.abspath(db_transposed)} (ignored) | blend={os.path.abspath(BLEND_INTERNAL_REPO or '')}",
        flush=True,
    )
    merged = search_tab2tab_aug(
        search_type=args.search_type,
        query=args.query,
        k=args.k,
        db_original=db_original,
        db_transposed=db_transposed,
        output_json=out_path,
        transposed_query_csv=(args.transposed_query_csv or None),
    )
    wall = time.perf_counter() - t_wall
    print(f"Saved: {out_path} | merged top-{args.k}: {len(merged)} | wall {wall:.4f}s", flush=True)


if __name__ == "__main__":
    main()
