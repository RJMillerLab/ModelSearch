"""
Query -> Tab -> Card

Pipeline:
1) query2modelcard (get seed card from query)
2) card2tab2card (pure card->tab->card mapping, no model_top_k cap)
3) optional query-based dense rerank over returned model_ids (no cap by default)
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

from src.config import *
from src.search.card2tab2card import search_card2tab2card
from src.search.query2modelcard import dense_rerank_model_ids_by_query, search_query2modelcard


def _resource_paths(resources: List[str]) -> Tuple[str, str]:
    resource_set = set(resources)
    if resource_set == {"hugging"}:
        return MODELLAKE_DB_HUGGING, EMB_NPZ_HUGGING
    if resource_set == {"hugging", "github", "arxiv"}:
        return MODELLAKE_DB, EMB_NPZ
    raise NotImplementedError(
        f"Unsupported resource combination: {resource_set}. Must be one of: {'hugging', 'github', 'arxiv'}"
    )


def search_query2tab2card(
    query: str,
    *,
    search_type: str = "keyword",
    output_json: str = "",
    table_top_k: int = 10,
    table_resources: Optional[List[str]] = None,
    q2m_top_k: int = 20,
    seed_rank_index: int = 0,
    apply_query_rerank: bool = True,
) -> Dict[str, object]:
    q = str(query).strip()
    if not q:
        raise ValueError("query is required")
    resources = [str(r).strip().lower() for r in (table_resources or ["hugging"]) if str(r).strip()]
    db_path, emb_npz_path = _resource_paths(resources)

    q2m_ids = search_query2modelcard(
        query=q,
        top_k=max(1, int(q2m_top_k)),
        output_json=None,
        retrieval_mode="dense",
        emb_npz_path=emb_npz_path,
    )
    if not q2m_ids:
        payload: Dict[str, object] = {
            "query": q,
            "query_seed_model_id": "",
            "query2modelcard_model_ids": [],
            "query2tab2card_model_ids": [],
            "mappings": {
                "card_to_related_tables": {},
                "query_table_to_retrieved_tables": {},
                "retrieved_table_to_related_models": {},
            },
            "pipeline_trace": {
                "query2modelcard": {"model_ids": []},
                "card2tab2card": {},
                "query_dense_rerank": {"applied": False, "model_ids_after_dense_rerank": []},
            },
        }
        if output_json:
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload

    seed_idx = max(0, min(int(seed_rank_index), len(q2m_ids) - 1))
    seed_model = str(q2m_ids[seed_idx]).strip()
    card_payload = search_card2tab2card(
        model_id=seed_model,
        search_type=search_type,
        output_json="",
        db_path=db_path,
        table_top_k=table_top_k,
        table_resources=resources,
    )
    base_ids = list(card_payload.get("model_ids", [])) if isinstance(card_payload, dict) else []
    final_ids = base_ids
    rerank_applied = False
    if apply_query_rerank and base_ids:
        final_ids = dense_rerank_model_ids_by_query(q, base_ids, emb_npz_path=emb_npz_path)
        rerank_applied = True

    payload = {
        "query": q,
        "query_seed_model_id": seed_model,
        "query2modelcard_model_ids": q2m_ids,
        "query2tab2card_model_ids": final_ids,
        "query_tables": card_payload.get("query_tables", []),
        "searched_tables": card_payload.get("searched_tables", []),
        "model_ids": final_ids,
        "mappings": {
            "card_to_related_tables": card_payload.get("mappings", {}).get("card_to_related_tables", {}),
            "query_table_to_retrieved_tables": card_payload.get("mappings", {}).get("query_table_to_retrieved_tables", {}),
            "retrieved_table_to_related_models": card_payload.get("mappings", {}).get("retrieved_table_to_related_models", {}),
        },
        "intermediate": card_payload.get("intermediate", {}),
        "pipeline_trace": {
            "query2modelcard": {"model_ids": q2m_ids, "seed_rank_index": seed_idx, "seed_model_id": seed_model},
            "card2tab2card": card_payload.get("pipeline_trace", {}),
            "query_dense_rerank": {
                "applied": rerank_applied,
                "model_ids_before_dense_rerank": base_ids,
                "model_ids_after_dense_rerank": final_ids,
            },
        },
    }
    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Query -> Tab -> Card")
    parser.add_argument("--query", required=True, help="Original user query")
    parser.add_argument("--search_type", choices=["single_column", "multi_column", "keyword", "unionable"], default="keyword")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--resources", nargs="+", default=["hugging"], choices=["hugging", "github", "arxiv"], help="Table resource filter.")
    parser.add_argument("--table_top_k", type=int, default=10, help="Top-k retrieved tables per query table.")
    parser.add_argument("--q2m_top_k", type=int, default=20, help="Top-k candidates from query2modelcard for choosing seed model.")
    parser.add_argument("--seed_rank_index", type=int, default=0, help="Pick seed model from query2modelcard ranking by index.")
    parser.add_argument("--disable_query_rerank", action="store_true", help="Disable query dense rerank on final model ids.")
    args = parser.parse_args()

    t0 = time.time()
    payload = search_query2tab2card(
        query=args.query,
        search_type=args.search_type,
        output_json=args.output_json,
        table_top_k=args.table_top_k,
        table_resources=args.resources,
        q2m_top_k=args.q2m_top_k,
        seed_rank_index=args.seed_rank_index,
        apply_query_rerank=not bool(args.disable_query_rerank),
    )
    mids = payload.get("model_ids", []) if isinstance(payload, dict) else []
    print(f"Found {len(mids)} model ids for query: {args.query!r}")
    for i, mid in enumerate(mids[:20], 1):
        print(f"  {i}. {mid}")
    if args.output_json:
        print(f"✅ Results saved to {args.output_json}")
    print(f"Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
