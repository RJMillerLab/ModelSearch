"""
Card -> Tab -> Card (simplified)

Pipeline:
1) Read model-related tables from relationship parquet (modelId -> csv_path)
2) Search related tables using src.search.tab2tab.search_table2table
3) Map retrieved table ids -> filenames via modellake.db
4) Reverse map filenames -> modelIds via relationship parquet
"""

import os
import json
import time
import argparse
from typing import Dict, List, Optional, Set

import duckdb

#from src.config import MODELLAKE_DB, CARD2TAB2CARD_OUTPUT_JSON
from src.config import *
from src.search.tab2tab import search_table2table
from src.utils import load_modelid_to_csvlist, load_csvs_to_modelids, resolve_table_path


def _table_ids_to_filenames(table_ids: List[int], db_path: str) -> Dict[int, str]:
    if not table_ids:
        return {}
    sql = f"""
        SELECT DISTINCT tableid, filename
        FROM modellake_index
        WHERE rowid = -1 AND tableid IN ({",".join(["?"] * len(table_ids))})
    """
    with duckdb.connect(db_path, read_only=True) as con:
        rows = con.execute(sql, table_ids).fetchall()
    return {int(tid): str(fname) for tid, fname in rows}


def search_card2tab2card(   
    model_id: str,
    search_type: str = "keyword",
    output_json: str = "",
    db_path: str = MODELLAKE_DB,
    table_top_k: int = 10,
    model_top_k: int = 20,
    table_resources: Optional[List[str]] = None,
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
                json.dump({"query_model": model_id, "query_tables": [], "searched_tables": [], "model_ids": [], "intermediate": {"retrieved_table_ids": [], "retrieved_table_filenames": [], "table_id_to_filename": {}, "table_to_models": {}}}, f, ensure_ascii=False, indent=2)
        return []
    else:
        print(f"query_tables example: {query_tables[0]}")

    # table2table search
    similar_table_ids: List[int] = []
    for csv_path in query_tables:
        if not os.path.exists(csv_path):
            continue
        ids = search_table2table(query=csv_path, search_type=search_type, k=table_top_k, db_path=db_path)
        if ids:
            similar_table_ids.extend(ids)

    if not similar_table_ids:
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
                            "retrieved_table_ids": [],
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

    # map table ids to tables name (dedupe while preserving order)
    unique_tids = list(dict.fromkeys(int(tid) for tid in similar_table_ids))
    unique_tids = unique_tids[: max(table_top_k, 1) * max(len(query_tables), 1)]
    table_id_to_filename = _table_ids_to_filenames(unique_tids, db_path)
    retrieved_tables = list(dict.fromkeys(table_id_to_filename.get(tid) for tid in unique_tids if table_id_to_filename.get(tid)))

    reverse = load_csvs_to_modelids([os.path.basename(fname) for fname in retrieved_tables])
    table_to_models = {table: [mid for mid in reverse.get(os.path.basename(table), []) if mid != model_id] for table in retrieved_tables}
    similar_models = list(dict.fromkeys(mid for mids in table_to_models.values() for mid in mids))
    # `k` in this pipeline is controlled by the frontend's per-table top-k.
    # Many different model_ids can be related to a single retrieved table,
    # so we should not cap the number of returned model_ids to `k`.
    # Keep UI-consistent cap: models capped at 50.
    final_results = similar_models[:model_top_k] if model_top_k > 0 else similar_models

    print(f"✅ query_tables={len(query_tables)} retrieved_tables={len(retrieved_tables)} model_ids={len(final_results)}")
    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "query_model": model_id,
                    "query_tables": query_tables,
                    "searched_tables": retrieved_tables,
                    "model_ids": final_results,
                    "intermediate": {
                        "retrieved_table_ids": unique_tids,
                        "retrieved_table_filenames": retrieved_tables,
                        "table_id_to_filename": {str(k_): v for k_, v in table_id_to_filename.items()},
                        "table_to_models": table_to_models,
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
    )
    print(f"Found {len(results)} model ids for {args.model_id}")
    for i, mid in enumerate(results[:20], 1):
        print(f"  {i}. {mid}")
    print(f"Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
