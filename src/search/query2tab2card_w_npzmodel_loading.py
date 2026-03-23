"""
Query -> Tab -> Card

Pipeline:
1) query2modelcard (get seed card + top-k prefix for candidate filtering)
2) card2tab2card (tab2tab -> parquet model ids)
3) candidate_pool = tab2tab models ∩ query2modelcard[:q2m_table_candidate_k]
4) dense rerank candidate_pool by query -> take model_top_k
5) expand each top-k model with load_modelid_to_csvlist -> searched_tables + table_to_models
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Set, Tuple

from src.config import *
from src.search.card2tab2card import search_card2tab2card
from src.search.query2modelcard import dense_rerank_model_ids_by_query, search_query2modelcard
from src.utils import load_modelid_to_csvlist


def _resource_paths(resources: List[str]) -> Tuple[str, str]:
    resource_set = set(resources)
    if resource_set == {"hugging"}:
        return MODELLAKE_DB_HUGGING, EMB_NPZ_HUGGING
    if resource_set == {"hugging", "github", "arxiv"}:
        return MODELLAKE_DB, EMB_NPZ
    raise NotImplementedError(
        f"Unsupported resource combination: {resource_set}. Must be one of: {'hugging', 'github', 'arxiv'}"
    )


def _expand_models_to_parquet_tables(
    model_ids: List[str],
    *,
    table_resources: Optional[List[str]],
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    For each model id, load CSV basenames from relationship parquet (load_modelid_to_csvlist).

    Returns:
        ordered_basenames: stable unique basenames in first-seen order
        table_to_models: basename -> list of model ids (from the input set) that own that table
        model_id_to_tables: model_id -> list of basenames
    """
    model_id_to_tables: Dict[str, List[str]] = {}
    table_to_models: Dict[str, List[str]] = {}
    ordered: List[str] = []
    seen_bn: Set[str] = set()
    for mid in model_ids:
        sm = str(mid).strip()
        if not sm:
            continue
        basenames = load_modelid_to_csvlist(sm, resources=table_resources)
        norm: List[str] = []
        for b in basenames:
            bn = os.path.basename(str(b).strip())
            if not bn:
                continue
            norm.append(bn)
        model_id_to_tables[sm] = norm
        for bn in norm:
            if bn not in seen_bn:
                seen_bn.add(bn)
                ordered.append(bn)
            table_to_models.setdefault(bn, [])
            if sm not in table_to_models[bn]:
                table_to_models[bn].append(sm)
    return ordered, table_to_models, model_id_to_tables


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
    model_top_k: int = 5,
    q2m_table_candidate_k: int = 9,
) -> Dict[str, object]:
    t_pipeline_start = time.time()
    q = str(query).strip()
    if not q:
        raise ValueError("query is required")
    resources = [str(r).strip().lower() for r in (table_resources or ["hugging"]) if str(r).strip()]
    db_path, emb_npz_path = _resource_paths(resources)

    t_q2m_start = time.time()
    q2m_ids = search_query2modelcard(
        query=q,
        top_k=max(1, int(q2m_top_k)),
        output_json=None,
        retrieval_mode="dense",
        emb_npz_path=emb_npz_path,
    )
    t_q2m_elapsed = time.time() - t_q2m_start
    print(f"[q2t2c-timing] query2modelcard: {t_q2m_elapsed:.2f}s", flush=True)
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
                "query_dense_rerank": {
                    "applied": False,
                    "model_ids_before_dense_rerank": [],
                    "model_ids_after_dense_rerank": [],
                    "model_top_k": int(model_top_k),
                    "q2m_table_candidate_k": int(q2m_table_candidate_k),
                },
                "timings": {
                    "query2modelcard_s": round(t_q2m_elapsed, 4),
                    "card2tab2card_s": 0.0,
                    "query_dense_rerank_and_expand_s": 0.0,
                    "total_s": round(time.time() - t_pipeline_start, 4),
                },
            },
        }
        if output_json:
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload

    seed_idx = max(0, min(int(seed_rank_index), len(q2m_ids) - 1))
    seed_model = str(q2m_ids[seed_idx]).strip()
    t_c2t2c_start = time.time()
    card_payload = search_card2tab2card(
        model_id=seed_model,
        search_type=search_type,
        output_json="",
        db_path=db_path,
        table_top_k=table_top_k,
        table_resources=resources,
    )
    t_c2t2c_elapsed = time.time() - t_c2t2c_start
    print(f"[q2t2c-timing] card2tab2card: {t_c2t2c_elapsed:.2f}s", flush=True)
    base_ids = list(card_payload.get("model_ids", [])) if isinstance(card_payload, dict) else []
    base_ids = [str(x).strip() for x in base_ids if str(x).strip()]
    seed_s = str(seed_model).strip()
    base_no_seed = [m for m in base_ids if m != seed_s]

    cand_k = max(1, int(q2m_table_candidate_k))
    q2m_prefix = [str(x).strip() for x in q2m_ids[:cand_k] if str(x).strip()]
    q2m_prefix_set = set(q2m_prefix)
    candidate_pool = [m for m in base_no_seed if m in q2m_prefix_set]
    if not candidate_pool:
        candidate_pool = list(base_no_seed)

    t_rerank_expand_start = time.time()
    rerank_applied = False
    ranked = list(candidate_pool)
    if apply_query_rerank and candidate_pool:
        ranked = dense_rerank_model_ids_by_query(q, candidate_pool, emb_npz_path=emb_npz_path)
        rerank_applied = True

    mtk = int(model_top_k)
    if mtk > 0:
        final_ids = ranked[:mtk]
    else:
        final_ids = list(ranked)

    searched_expanded, table_to_models_expanded, model_id_to_related = _expand_models_to_parquet_tables(
        final_ids, table_resources=resources
    )
    t_rerank_expand_elapsed = time.time() - t_rerank_expand_start
    t_total_elapsed = time.time() - t_pipeline_start
    print(f"[q2t2c-timing] query_dense_rerank_and_expand: {t_rerank_expand_elapsed:.2f}s", flush=True)
    print(f"[q2t2c-timing] total: {t_total_elapsed:.2f}s", flush=True)

    card_mappings = card_payload.get("mappings", {}) if isinstance(card_payload.get("mappings"), dict) else {}
    inter_prev = card_payload.get("intermediate", {}) if isinstance(card_payload.get("intermediate"), dict) else {}
    inter_out = dict(inter_prev)
    inter_out["retrieved_table_filenames"] = list(searched_expanded)
    inter_out["table_to_models"] = table_to_models_expanded
    if not isinstance(inter_out.get("query_table_to_retrieved_tables"), dict):
        inter_out["query_table_to_retrieved_tables"] = inter_prev.get("query_table_to_retrieved_tables", {})

    payload = {
        "query": q,
        "query_seed_model_id": seed_model,
        "query2modelcard_model_ids": q2m_ids,
        "query2tab2card_model_ids": final_ids,
        "query_tables": card_payload.get("query_tables", []),
        "searched_tables": searched_expanded,
        "model_ids": final_ids,
        "mappings": {
            "card_to_related_tables": card_mappings.get("card_to_related_tables", {}),
            "query_table_to_retrieved_tables": card_mappings.get("query_table_to_retrieved_tables", {}),
            "retrieved_table_to_related_models": table_to_models_expanded,
            "model_id_to_related_tables": model_id_to_related,
            "tab2tab_retrieved_table_to_related_models": card_mappings.get(
                "retrieved_table_to_related_models", {}
            ),
        },
        "intermediate": inter_out,
        "pipeline_trace": {
            "query2modelcard": {
                "model_ids": q2m_ids,
                "seed_rank_index": seed_idx,
                "seed_model_id": seed_model,
                "q2m_table_candidate_prefix": q2m_prefix,
            },
            "card2tab2card": card_payload.get("pipeline_trace", {}),
            "query_dense_rerank": {
                "applied": rerank_applied,
                "tab2tab_candidate_model_ids": list(base_no_seed),
                "candidate_pool_after_q2m_filter": list(candidate_pool),
                "model_ids_before_dense_rerank": list(candidate_pool),
                "model_ids_after_dense_rerank": list(ranked),
                "model_ids_top_k": list(final_ids),
                "model_top_k": mtk,
                "q2m_table_candidate_k": cand_k,
            },
            "timings": {
                "query2modelcard_s": round(t_q2m_elapsed, 4),
                "card2tab2card_s": round(t_c2t2c_elapsed, 4),
                "query_dense_rerank_and_expand_s": round(t_rerank_expand_elapsed, 4),
                "total_s": round(t_total_elapsed, 4),
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
    parser.add_argument(
        "--q2m_table_candidate_k",
        type=int,
        default=9,
        help="Only tab2tab-hit models that also appear in query2modelcard top-k prefix enter rerank pool.",
    )
    parser.add_argument("--model_top_k", type=int, default=5, help="Final number of models after query rerank (0 = no cap).")
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
        model_top_k=args.model_top_k,
        q2m_table_candidate_k=args.q2m_table_candidate_k,
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
