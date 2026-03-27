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
import duckdb  # local import to keep optional dependency surface small

from typing import Dict, List, Optional

#from src.config import MODELLAKE_DB, CARD2TAB2CARD_OUTPUT_JSON
from src.config import *
from src.search.tab2tab import search_table2table
from src.search.tab2tab_aug import search_tab2tab_aug
from src.utils import load_modelid_to_csvlist, load_csvs_to_modelids


def _empty_payload(
    *,
    model_id: str,
    query_tables: List[str],
    query_table_to_retrieved_tables: Dict[str, List[str]],
    use_tab2tab_aug: bool,
) -> Dict[str, object]:
    return {
        "query_model": model_id,
        "query_tables": list(query_tables),
        "searched_tables": [],
        "model_ids": [],
        "mappings": {
            "card_to_related_tables": {model_id: list(query_tables)},
            "query_table_to_retrieved_tables": query_table_to_retrieved_tables,
            "retrieved_table_to_related_models": {},
            "model_id_to_related_tables": {},
            "tab2tab_retrieved_table_to_related_models": {},
        },
        "intermediate": {
            "retrieved_table_filenames": [],
            "table_id_to_filename": {},
            "table_to_models": {},
            "query_table_to_retrieved_tables": query_table_to_retrieved_tables,
        },
        "pipeline_trace": {"tab2tab": {"backend": "aug" if use_tab2tab_aug else "classic"}},
    }


def search_card2tab2card(
    model_id: str,  # card
    con_data: duckdb.DuckDBPyConnection,
    *,
    search_type: str = "keyword",  # search type
    table_top_k: int = 10,
    table_resources: Optional[List[str]] = None,
    use_tab2tab_aug: bool = False,
    output_json: str = "",
) -> Dict[str, object]:
    """Simplified card->tab->card search using relationship parquet + tab2tab import."""
    print(f"[Card2Tab2Card] model_id={model_id} search_type={search_type} table_top_k={table_top_k} table_resources={table_resources!r}")
    # model_id -> csv_basenames (relationship parquet)
    query_tables = load_modelid_to_csvlist(model_id, resources=table_resources)
    print(f"[c2t2c-trace] query_tables ({len(query_tables)}):", flush=True)
    if not query_tables:
        print(f"⚠️ No tables found for model_id={model_id}")
        empty_payload = _empty_payload(model_id=model_id, query_tables=[], query_table_to_retrieved_tables={}, use_tab2tab_aug=use_tab2tab_aug)
        if output_json:
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(empty_payload, f, ensure_ascii=False, indent=2)
        return empty_payload
    else:
        print(f"query_tables example: {query_tables[0]}")

    ############### table2table search → CSV basenames
    query_table_to_retrieved_tables: Dict[str, List[str]] = {}
    def _tab2tab_one(csv_basename: str) -> List[str]:
        if use_tab2tab_aug:
            names = search_tab2tab_aug(search_type=search_type, query=csv_basename, k=table_top_k, output_json=None, con_data=con_data, query_augmentation_types=["ori", "tr", "str"], candidate_augmentation_types=["ori", "tr", "str"], rerank_mode="table_level")
        else:
            names = search_table2table(search_type=search_type, query=csv_basename, k=table_top_k, output_json=None, con_data=con_data, augmentation_types=["ori"])
        return [str(n).strip() for n in (names or []) if str(n).strip()]
    retrieved_flat: List[str] = []
    for q in query_tables:
        names = _tab2tab_one(q)
        query_table_to_retrieved_tables[q] = names
        retrieved_flat.extend(names)
    if not retrieved_flat:
        print("⚠️ No similar tables returned by tab2tab.")
        empty_payload = _empty_payload(model_id=model_id, query_tables=query_tables, query_table_to_retrieved_tables=query_table_to_retrieved_tables, use_tab2tab_aug=use_tab2tab_aug)
        if output_json:
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(empty_payload, f, ensure_ascii=False, indent=2)
            print(f"✅ Results saved to {output_json} (empty)")
        return empty_payload
    retrieved_tables = list(dict.fromkeys(retrieved_flat))
    retrieved_tables = retrieved_tables[: max(table_top_k, 1) * max(len(query_tables), 1)]
    print(f"[c2t2c-trace] after dedupe: unique_basenames={len(retrieved_tables)} (tab2tab → parquet only)", flush=True)

    ############## CSV basenames -> model ids
    reverse = load_csvs_to_modelids(retrieved_tables)
    table_to_models = {bn: [mid for mid in reverse.get(bn, []) if mid != model_id] for bn in retrieved_tables}
    # Tab→card: never return the query model (defense in depth vs parquet / multi-table paths).
    similar_models = list(dict.fromkeys(mid for mids in table_to_models.values() for mid in mids if mid != model_id))
    final_results = list(similar_models)

    print("[c2t2c-trace] parquet load_csvs_to_modelids: retrieved_csv_basename -> modelIds (query model excluded per row)", flush=True)
    _rows = sorted(table_to_models.items(), key=lambda x: -len(x[1]))
    _max_tbl_lines = 60
    for bn, mids in _rows[:_max_tbl_lines]:
        prev = ", ".join(mids[:6])
        extra = f" ...+{len(mids) - 6} models" if len(mids) > 6 else ""
        print(f"  {bn} | n={len(mids)} | {prev}{extra}", flush=True)
    if len(_rows) > _max_tbl_lines:
        print(f"  ... [{len(_rows) - _max_tbl_lines} more tables omitted]", flush=True)
    print(f"[c2t2c-trace] unique models from tab2tab->parquet: {len(final_results)} (no model_top_k cap in card2tab2card)", flush=True)
    print(f"[c2t2c-trace] final model_ids order: {final_results}", flush=True)

    print(f"✅ query_tables={len(query_tables)} retrieved_tables={len(retrieved_tables)} model_ids={len(final_results)}")
    result_payload: Dict[str, object] = {
        "query_model": model_id,
        "query_tables": query_tables,
        "searched_tables": retrieved_tables,
        "model_ids": final_results,
        "mappings": {
            "card_to_related_tables": {model_id: list(query_tables)},
            "query_table_to_retrieved_tables": query_table_to_retrieved_tables,
            "retrieved_table_to_related_models": table_to_models,
            "model_id_to_related_tables": {},
            "tab2tab_retrieved_table_to_related_models": table_to_models,
        },
        "intermediate": {
            "retrieved_table_filenames": list(retrieved_tables),
            "table_id_to_filename": {},
            "table_to_models": table_to_models,
            "query_table_to_retrieved_tables": query_table_to_retrieved_tables,
        },
        "pipeline_trace": {
            "tab2tab": {"backend": "aug" if use_tab2tab_aug else "classic"},
            "model_ids_before_dense_rerank": list(final_results),
            "model_ids_after_dense_rerank": list(final_results),
            "dense_rerank_applied": False,
        },
    }
    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved to {output_json}")
    return result_payload

def main() -> None:
    parser = argparse.ArgumentParser(description="Card -> Tab -> Card search (simplified)")
    parser.add_argument("--model_id", required=True, help="Query model id")
    parser.add_argument("--search_type", choices=["single_column", "multi_column", "keyword", "unionable"], default="keyword")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--resources", nargs="+", default=["hugging"], choices=["hugging", "github", "arxiv", "llm"], help="Optional table resource filter on card2tab2card results.")
    parser.add_argument("--table_top_k", type=int, default=10, help="Top-k table ids")
    args = parser.parse_args()

    resources = [str(r).strip().lower() for r in (args.resources or []) if str(r).strip()]
    resource_set = set(resources)
    if resource_set == {'hugging'}:
        db_path = MODELLAKE_DB_HUGGING
    elif resource_set == {'hugging', 'github', 'arxiv'}:
        db_path = MODELLAKE_DB
    else:
        raise NotImplementedError(f"Unsupported resource combination: {resource_set}. Must be one of: {'hugging', 'github', 'arxiv'}")

    print("[card2tab2card] artifacts: " f"resources={resources!r} | RELATIONSHIP_PARQUET={os.path.abspath(RELATIONSHIP_PARQUET)} | MODELLAKE_DB={os.path.abspath(db_path)} | TABLE_BASE_DIRS={[os.path.abspath(d) for d in TABLE_BASE_DIRS]!r}", flush=True)

    con_data = duckdb.connect(db_path, read_only=True)

    t0 = time.time()
    
    payload = search_card2tab2card(model_id=args.model_id, search_type=args.search_type, table_top_k=args.table_top_k, output_json=args.output_json, con_data=con_data, table_resources=resources)
    results = payload.get("model_ids", []) if isinstance(payload, dict) else []
    print(f"Found {len(results)} model ids for {args.model_id}")
    for i, mid in enumerate(results[:20], 1):
        print(f"  {i}. {mid}")
    print(f"Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
