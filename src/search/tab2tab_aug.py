"""
Augmented table-to-table search.

Design:
- Keep `src.search.tab2tab` as single-run retriever.
- Put augmentation merge logic here with clear class-based rerankers.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config import BLEND_INTERNAL_REPO, MODELLAKE_DB_HUGGING, TAB2TAB_AUG_OUTPUT_JSON
from src.search.tab2tab import search_table2table, search_table2table_with_scores

AUG_TYPES: Tuple[str, str, str] = ("ori", "tr", "str")


def canonical_modellake_basename(basename: str) -> str:
    bn = str(basename).strip()
    if bn.endswith("_t.csv"):
        return bn[:-6] + ".csv"
    if bn.endswith("_s.csv"):
        return bn[:-6] + ".csv"
    return bn


def _query_variant_basenames(query: str, query_aug_types: Sequence[str]) -> Dict[str, str]:
    q_bn = os.path.basename(str(query).strip())
    if not q_bn:
        raise ValueError("query must be non-empty")
    stem = canonical_modellake_basename(q_bn)[:-4] if canonical_modellake_basename(q_bn).endswith(".csv") else canonical_modellake_basename(q_bn)
    out: Dict[str, str] = {}
    for q_aug in query_aug_types:
        if q_aug == "ori":
            out[q_aug] = f"{stem}.csv"
        elif q_aug == "tr":
            out[q_aug] = f"{stem}_t.csv"
        elif q_aug == "str":
            out[q_aug] = f"{stem}_s.csv"
        else:
            raise ValueError(f"Unsupported query augmentation type: {q_aug!r}")
    return out


def _normalize_aug_types(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return list(AUG_TYPES)
    out = [str(v).strip().lower() for v in values if str(v).strip()]
    bad = [x for x in out if x not in AUG_TYPES]
    if bad:
        raise ValueError(f"Unsupported augmentation types: {bad!r}, expected one of {AUG_TYPES}")
    return out


@dataclass
class AugRetrievalOutput:
    lane_rankings: Dict[str, List[str]]
    lane_rankings_scored: Dict[str, List[Dict[str, Any]]]
    query_basenames: Dict[str, str]
    query_canon: str


class NineLaneRetriever:
    """
    Execute the 3x3 retrieval grid:
    query_aug (ori/tr/str) x candidate_aug (ori/tr/str).
    """

    def __init__(self, *, search_type: str, query: str, k: int, db_path: str, query_aug_types: Sequence[str], candidate_aug_types: Sequence[str]) -> None:
        self.search_type = search_type
        self.query = query
        self.k = k
        self.db_path = db_path
        self.query_aug_types = list(query_aug_types)
        self.candidate_aug_types = list(candidate_aug_types)
        self.query_basenames = _query_variant_basenames(query, self.query_aug_types)
        self.query_canon = canonical_modellake_basename(os.path.basename(str(query)))

    def run(self, *, with_scores: bool) -> AugRetrievalOutput:
        lane_rankings: Dict[str, List[str]] = {}
        lane_rankings_scored: Dict[str, List[Dict[str, Any]]] = {}

        for q_aug in self.query_aug_types:
            q_name = self.query_basenames[q_aug]
            for c_aug in self.candidate_aug_types:
                lane_id = f"q_{q_aug}__cand_{c_aug}"
                if with_scores:
                    lane_rankings_scored[lane_id] = search_table2table_with_scores(search_type=self.search_type, query=q_name, k=self.k, db_path=self.db_path, augmentation_types=[c_aug])
                else:
                    lane_rankings[lane_id] = search_table2table(search_type=self.search_type, query=q_name, k=self.k, db_path=self.db_path, augmentation_types=[c_aug])

        return AugRetrievalOutput(lane_rankings=lane_rankings, lane_rankings_scored=lane_rankings_scored, query_basenames=self.query_basenames, query_canon=self.query_canon)


class BaseAugReranker(ABC):
    @abstractmethod
    def rerank(self, *, lane_rankings: Dict[str, List[str]], lane_rankings_scored: Dict[str, List[Dict[str, Any]]], top_k: int, query_canon: str) -> Tuple[List[str], Dict[str, Any]]:
        raise NotImplementedError


class TableLevelHitCountReranker(BaseAugReranker):
    """No score required: rank by lane hit count, tie-break by best rank."""

    def rerank(self, *, lane_rankings: Dict[str, List[str]], lane_rankings_scored: Dict[str, List[Dict[str, Any]]], top_k: int, query_canon: str) -> Tuple[List[str], Dict[str, Any]]:
        hit_count: Dict[str, int] = defaultdict(int)
        min_rank: Dict[str, int] = {}
        for _lane, names in lane_rankings.items():
            seen: set[str] = set()
            for rank, n in enumerate(names, start=1):
                key = canonical_modellake_basename(os.path.basename(n))
                if not key or key == query_canon or key in seen:
                    continue
                seen.add(key)
                hit_count[key] += 1
                min_rank[key] = min(min_rank.get(key, 10**18), rank)
        scored = [(k, hit_count[k], min_rank.get(k, 10**18)) for k in hit_count]
        scored.sort(key=lambda x: (-x[1], x[2], x[0]))
        final = [x[0] for x in scored[:top_k]]
        stats = {k: {"hit_count": hc, "min_rank": mr} for k, hc, mr in scored[:top_k]}
        return final, stats


class ScoreMaxVariantReranker(BaseAugReranker):
    """
    Score-aware mode:
    for each canonical table, keep the best score among variants (`csv`, `csv_t`, `csv_s`),
    then rank by this max score.
    """

    def rerank(self, *, lane_rankings: Dict[str, List[str]], lane_rankings_scored: Dict[str, List[Dict[str, Any]]], top_k: int, query_canon: str) -> Tuple[List[str], Dict[str, Any]]:
        best_score: Dict[str, float] = defaultdict(lambda: float("-inf"))
        min_rank: Dict[str, int] = {}
        hit_count: Dict[str, int] = defaultdict(int)
        for _lane, items in lane_rankings_scored.items():
            seen: set[str] = set()
            for i, item in enumerate(items, start=1):
                fn = str(item.get("filename", "")).strip()
                if not fn:
                    continue
                key = canonical_modellake_basename(os.path.basename(fn))
                if not key or key == query_canon:
                    continue
                score = float(item.get("score", 0.0) or 0.0)
                rank = int(item.get("rank", i) or i)
                best_score[key] = max(best_score[key], score)
                min_rank[key] = min(min_rank.get(key, 10**18), rank)
                if key not in seen:
                    seen.add(key)
                    hit_count[key] += 1
        scored = [(k, best_score[k], hit_count.get(k, 0), min_rank.get(k, 10**18)) for k in best_score]
        scored.sort(key=lambda x: (-x[1], -x[2], x[3], x[0]))
        final = [x[0] for x in scored[:top_k]]
        stats = {k: {"max_score": s, "hit_count": hc, "min_rank": mr} for k, s, hc, mr in scored[:top_k]}
        return final, stats


class FlatNoScoreReranker(BaseAugReranker):
    """No score required: flatten all lanes in order, canonicalize, dedupe."""

    def rerank(self, *, lane_rankings: Dict[str, List[str]], lane_rankings_scored: Dict[str, List[Dict[str, Any]]], top_k: int, query_canon: str) -> Tuple[List[str], Dict[str, Any]]:
        out: List[str] = []
        seen: set[str] = set()
        for lane_id in sorted(lane_rankings.keys()):
            for name in lane_rankings.get(lane_id, []):
                key = canonical_modellake_basename(os.path.basename(name))
                if not key or key == query_canon or key in seen:
                    continue
                seen.add(key)
                out.append(key)
                if len(out) >= top_k:
                    return out, {"mode": "flat_noscore", "returned": len(out)}
        return out, {"mode": "flat_noscore", "returned": len(out)}


RERANKER_BY_MODE: Dict[str, BaseAugReranker] = {
    "table_level": TableLevelHitCountReranker(),
    "score_max": ScoreMaxVariantReranker(),
    "flat_noscore": FlatNoScoreReranker(),
}


def search_tab2tab_aug(*, search_type: str, query: str, k: int, db_original: str, db_transposed: str, output_json: Optional[str] = None, transposed_query_csv: Optional[str] = None, query_augmentation_types: Optional[List[str]] = None, candidate_augmentation_types: Optional[List[str]] = None, rerank_mode: str = "table_level") -> List[str]:
    del db_transposed, transposed_query_csv
    t0 = time.time()
    query_aug_types = _normalize_aug_types(query_augmentation_types)
    cand_aug_types = _normalize_aug_types(candidate_augmentation_types)

    rerank_mode_l = str(rerank_mode).strip().lower()
    if rerank_mode_l not in RERANKER_BY_MODE:
        raise ValueError(f"Unsupported rerank_mode={rerank_mode!r}. Expected one of: {sorted(RERANKER_BY_MODE.keys())}")

    retriever = NineLaneRetriever(search_type=search_type, query=query, k=k, db_path=db_original, query_aug_types=query_aug_types, candidate_aug_types=cand_aug_types)
    retrieved = retriever.run(with_scores=(rerank_mode_l == "score_max"))

    reranker = RERANKER_BY_MODE[rerank_mode_l]
    final, stats = reranker.rerank(lane_rankings=retrieved.lane_rankings, lane_rankings_scored=retrieved.lane_rankings_scored, top_k=k, query_canon=retrieved.query_canon)

    if output_json:
        payload = {"version": 3, "search_type": search_type, "query": query, "query_basenames": retrieved.query_basenames, "query_augmentation_types": query_aug_types, "candidate_augmentation_types": cand_aug_types, "rerank_mode": rerank_mode_l, "lane_rankings": retrieved.lane_rankings, "lane_rankings_scored": retrieved.lane_rankings_scored, "rerank_stats": stats, "merged_ranking": final, "elapsed_s": round(time.time() - t0, 4)}
        parent = os.path.dirname(os.path.abspath(output_json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="tab2tab augmentation with class-based rerankers.")
    parser.add_argument("--search_type", required=True, choices=["single_column", "multi_column", "keyword", "unionable"])
    parser.add_argument("--query", required=True, help="table basename or path; aug variant names are inferred from stem")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--query_augmentation_types", default="ori,tr,str")
    parser.add_argument("--candidate_augmentation_types", default="ori,tr,str")
    parser.add_argument("--rerank_mode", choices=sorted(RERANKER_BY_MODE.keys()), default="table_level")
    parser.add_argument("--output_json", default="", help=f"default: {TAB2TAB_AUG_OUTPUT_JSON}")
    args = parser.parse_args()

    db_original = MODELLAKE_DB_HUGGING
    out_path = args.output_json or TAB2TAB_AUG_OUTPUT_JSON
    t0 = time.perf_counter()
    print(f"[tab2tab_aug] db_original={os.path.abspath(db_original)} | blend={os.path.abspath(BLEND_INTERNAL_REPO or '')} | rerank_mode={args.rerank_mode}", flush=True)
    merged = search_tab2tab_aug(search_type=args.search_type, query=args.query, k=args.k, db_original=db_original, db_transposed=db_original, output_json=out_path, query_augmentation_types=[x.strip().lower() for x in args.query_augmentation_types.split(",") if x.strip()], candidate_augmentation_types=[x.strip().lower() for x in args.candidate_augmentation_types.split(",") if x.strip()], rerank_mode=args.rerank_mode)
    print(f"Saved: {out_path} | merged top-{args.k}: {len(merged)} | wall {time.perf_counter() - t0:.4f}s", flush=True)


if __name__ == "__main__":
    main()
