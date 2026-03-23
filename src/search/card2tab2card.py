"""
Card -> Tab -> Card (simplified)

Pipeline:
1) Read model-related tables from relationship parquet (modelId -> csv_path)
2) tab2tab search → CSV basenames from Blend
3) load_csvs_to_modelids(basenames) → model ids (relationship parquet only; no modellake here)
"""

import os
import json
import time
import argparse
from typing import Dict, List, Optional

#from src.config import MODELLAKE_DB, CARD2TAB2CARD_OUTPUT_JSON
from src.config import *
from src.search.tab2tab import search_table2table
from src.search.query2modelcard import dense_rerank_model_ids_by_query
from src.utils import load_modelid_to_csvlist, load_csvs_to_modelids, resolve_table_path


def search_card2tab2card(   
    model_id: str,
    search_type: str = "keyword",
    output_json: str = "",
    db_path: str = MODELLAKE_DB,
    table_top_k: int = 10,
    model_top_k: int = 20,
    table_resources: Optional[List[str]] = None,
    query: Optional[str] = None,
) -> List[str]:
    """Simplified card->tab->card search using relationship parquet + tab2tab import."""
    print(f"[Card2Tab2Card] model_id={model_id} search_type={search_type} table_top_k={table_top_k} model_top_k={model_top_k} table_resources={table_resources!r}")
    # model_id -> csv_basenames (only columns for --resources, e.g. hugging-only)
    query_table_basenames = load_modelid_to_csvlist(model_id, resources=table_resources)
    # utils returns csv basenames; resolve to local csv paths for reading.
    query_tables: List[str] = []
    for base in query_table_basenames:
        resolved = resolve_table_path(base)
        if resolved and os.path.exists(resolved):
            query_tables.append(resolved)
        else:
            print(f'⚠️ No table found for {base}')

    if not query_tables:
        print(f"⚠️ No tables found for model_id={model_id}")
        if output_json:
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump({"query_model": model_id, "query_tables": [], "searched_tables": [], "model_ids": [], "intermediate": {"retrieved_table_filenames": [], "table_id_to_filename": {}, "table_to_models": {}}}, f, ensure_ascii=False, indent=2)
        return []
    else:
        print(f"query_tables example: {query_tables[0]}")
    print(f"[c2t2c-trace] query_tables ({len(query_tables)}):", flush=True)
    for qp in query_tables:
        print(f"  seed_csv: {os.path.basename(qp)}  path={qp}", flush=True)

    # table2table search (tab2tab returns CSV basenames)
    similar_basenames: List[str] = []
    for csv_path in query_tables:
        if not os.path.exists(csv_path):
            continue
        names = search_table2table(query=csv_path, search_type=search_type, k=table_top_k, db_path=db_path)
        n_list = list(names) if names else []
        sample = n_list[:15]
        more = f" ...(+{len(n_list) - 15})" if len(n_list) > 15 else ""
        print(
            f"[c2t2c-trace] tab2tab query={os.path.basename(csv_path)} search_type={search_type} k={table_top_k} "
            f"-> n_filenames={len(n_list)} sample={sample}{more}",
            flush=True,
        )
        if names:
            similar_basenames.extend(names)

    if not similar_basenames:
        print("⚠️ No similar tables returned by tab2tab.")
        # Downstream expects `--output_json` to always be written so that
        # backend/frontend don't fail with "No JSON at ...".
        if output_json:
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "query_model": model_id,
                        "query_tables": query_tables,
                        "searched_tables": [],
                        "model_ids": [],
                        "intermediate": {
                            "retrieved_table_filenames": [],
                            "table_id_to_filename": {},
                            "table_to_models": {},
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"✅ Results saved to {output_json} (empty)")
        return []

    retrieved_tables = list(dict.fromkeys(similar_basenames))
    retrieved_tables = retrieved_tables[: max(table_top_k, 1) * max(len(query_tables), 1)]
    print(
        f"[c2t2c-trace] after dedupe: unique_basenames={len(retrieved_tables)} (tab2tab → parquet only)",
        flush=True,
    )

    reverse = load_csvs_to_modelids(retrieved_tables)
    table_to_models = {
        bn: [mid for mid in reverse.get(bn, []) if mid != model_id] for bn in retrieved_tables
    }
    # Tab→card: never return the query model (defense in depth vs parquet / multi-table paths).
    similar_models = list(
        dict.fromkeys(
            mid for mids in table_to_models.values() for mid in mids if mid != model_id
        )
    )
    # Candidate models from reverse lookup (before dense rerank / top-k cap).
    candidate_model_ids = list(similar_models)

    rerank_applied = False
    if isinstance(model_top_k, int) and model_top_k > 0 and len(candidate_model_ids) > model_top_k:
        q = str(query).strip() if query is not None else ""
        if q:
            reranked = dense_rerank_model_ids_by_query(q, candidate_model_ids, emb_npz_path=EMB_NPZ)
            final_results = reranked[:model_top_k]
            rerank_applied = True
        else:
            # Fallback when no query string is supplied by caller.
            final_results = candidate_model_ids[:model_top_k]
    elif isinstance(model_top_k, int) and model_top_k == 0:
        final_results = []
    else:
        final_results = list(candidate_model_ids)

    keep_set = set(final_results)
    synced_table_to_models = {}
    for bn in retrieved_tables:
        mids = table_to_models.get(bn, [])
        kept = [m for m in mids if m in keep_set]
        if kept:
            synced_table_to_models[bn] = kept
    synced_retrieved_tables = [bn for bn in retrieved_tables if bn in synced_table_to_models]

    print("[c2t2c-trace] parquet load_csvs_to_modelids: retrieved_csv_basename -> modelIds (query model excluded per row)", flush=True)
    _rows = sorted(table_to_models.items(), key=lambda x: -len(x[1]))
    _max_tbl_lines = 60
    for bn, mids in _rows[:_max_tbl_lines]:
        prev = ", ".join(mids[:6])
        extra = f" ...+{len(mids) - 6} models" if len(mids) > 6 else ""
        print(f"  {bn} | n={len(mids)} | {prev}{extra}", flush=True)
    if len(_rows) > _max_tbl_lines:
        print(f"  ... [{len(_rows) - _max_tbl_lines} more tables omitted]", flush=True)
    print(
        f"[c2t2c-trace] unique models before model_top_k cap: {len(candidate_model_ids)} "
        f"(model_top_k={model_top_k}); after cap: {len(final_results)} | rerank_applied={rerank_applied}",
        flush=True,
    )
    print(f"[c2t2c-trace] final model_ids order: {final_results}", flush=True)
    print(
        f"[c2t2c-trace] synced searched_tables after model cap: {len(synced_retrieved_tables)} "
        f"(from raw {len(retrieved_tables)})",
        flush=True,
    )

    print(f"✅ query_tables={len(query_tables)} retrieved_tables={len(synced_retrieved_tables)} model_ids={len(final_results)}")
    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "query_model": model_id,
                    "query_tables": query_tables,
                    "searched_tables": synced_retrieved_tables,
                    "model_ids": final_results,
                    "intermediate": {
                        "retrieved_table_filenames": list(synced_retrieved_tables),
                        "table_id_to_filename": {},
                        "table_to_models": synced_table_to_models,
                    },
                    "pipeline_trace": {
                        "tab2tab": {
                            "searched_tables": list(retrieved_tables),
                            "retrieved_table_filenames": list(retrieved_tables),
                            "table_to_models": table_to_models,
                        },
                        "model_ids_before_dense_rerank": list(candidate_model_ids),
                        "model_ids_after_dense_rerank": list(final_results),
                        "after_model_cap": {
                            "searched_tables": list(synced_retrieved_tables),
                            "table_to_models": synced_table_to_models,
                        },
                        "dense_rerank_applied": rerank_applied,
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"✅ Results saved to {output_json}")
    return final_results

def main() -> None:
    parser = argparse.ArgumentParser(description="Card -> Tab -> Card search (simplified)")
    parser.add_argument("--model_id", required=True, help="Query model id")
    parser.add_argument("--search_type", choices=["single_column", "multi_column", "keyword", "unionable"], default="keyword")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--resources", nargs="+", default=["hugging"], choices=["hugging", "github", "arxiv", "llm"], help="Optional table resource filter on card2tab2card results.")
    parser.add_argument("--table_top_k", type=int, default=10, help="Top-k table ids")
    parser.add_argument("--model_top_k", type=int, default=20, help="Top-k model ids")
    parser.add_argument("--query", default="", help="Original user query for dense rerank")
    args = parser.parse_args()

    resources = [str(r).strip().lower() for r in (args.resources or []) if str(r).strip()]
    resource_set = set(resources)
    if resource_set == {'hugging'}:
        db_path = MODELLAKE_DB_HUGGING
    elif resource_set == {'hugging', 'github', 'arxiv'}:
        db_path = MODELLAKE_DB
    else:
        raise NotImplementedError(f"Unsupported resource combination: {resource_set}. Must be one of: {'hugging', 'github', 'arxiv'}")

    print(
        "[card2tab2card] artifacts: "
        f"resources={resources!r} | "
        f"RELATIONSHIP_PARQUET={os.path.abspath(RELATIONSHIP_PARQUET)} | "
        f"MODELLAKE_DB={os.path.abspath(db_path)} | "
        f"TABLE_BASE_DIRS={[os.path.abspath(d) for d in TABLE_BASE_DIRS]!r}",
        flush=True,
    )

    t0 = time.time()
    
    results = search_card2tab2card(
        model_id=args.model_id,
        search_type=args.search_type,
        table_top_k=args.table_top_k,
        output_json=args.output_json,
        db_path=db_path,
        model_top_k=args.model_top_k,
        table_resources=resources,
        query=args.query,
    )
    print(f"Found {len(results)} model ids for {args.model_id}")
    for i, mid in enumerate(results[:20], 1):
        print(f"  {i}. {mid}")
    print(f"Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
