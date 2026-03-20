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
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

from src.config import MODELLAKE_DB, CARD2TAB2CARD_OUTPUT_JSON
from src.search.tab2tab import search_table2table
from src.utils import load_modelid_to_csvlist, load_csvs_to_modelids, resolve_table_path


def _get_csv_headers(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path, nrows=0)
    return [str(c).strip().lower() for c in df.columns if str(c).strip()]


def _build_query_from_csv(csv_path: str, search_type: str) -> Any:
    if search_type == "keyword":
        headers = _get_csv_headers(csv_path)
        return headers if headers else [os.path.basename(csv_path)]
    if search_type == "single_column":
        df = pd.read_csv(csv_path, nrows=100)
        if len(df.columns) == 0:
            return [os.path.basename(csv_path)]
        vals = df[df.columns[0]].dropna().astype(str).tolist()
        return vals if vals else [os.path.basename(csv_path)]
    if search_type in ("multi_column", "unionable"):
        return pd.read_csv(csv_path)
    raise ValueError(f"Unsupported search_type: {search_type}")


def _table_ids_to_filenames(table_ids: List[int]) -> Dict[int, str]:
    if not table_ids:
        return {}
    sql = f"""
        SELECT DISTINCT tableid, filename
        FROM modellake_index
        WHERE rowid = -1 AND tableid IN ({",".join(["?"] * len(table_ids))})
    """
    with duckdb.connect(MODELLAKE_DB, read_only=True) as con:
        rows = con.execute(sql, table_ids).fetchall()
    return {int(tid): str(fname) for tid, fname in rows}


def search_card2tab2card(   
    model_id: str,
    query: Optional[Any] = None,
    search_type: str = "keyword",
    k: int = 10,
    output_json: str = CARD2TAB2CARD_OUTPUT_JSON,
) -> List[str]:
    """Simplified card->tab->card search using relationship parquet + tab2tab import."""
    print(f"[Card2Tab2Card] model_id={model_id} search_type={search_type} k={k}")
    query_table_basenames = load_modelid_to_csvlist(model_id)
    # utils returns csv basenames; resolve to local csv paths for reading.
    query_tables: List[str] = []
    for base in query_table_basenames:
        resolved = resolve_table_path(base)
        if resolved and os.path.exists(resolved):
            query_tables.append(resolved)

    if not query_tables:
        print(f"⚠️ No tables found for model_id={model_id}")
        if output_json:
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump({"query_model": model_id, "query_tables": [], "searched_tables": [], "model_ids": [], "intermediate": {"retrieved_table_ids": [], "retrieved_table_filenames": [], "table_id_to_filename": {}, "table_to_models": {}}}, f, ensure_ascii=False, indent=2)
        return []

    similar_table_ids: List[int] = []
    if query is not None:
        ids = search_table2table(query, search_type, k)
        similar_table_ids.extend(ids if ids else [])
    else:
        for csv_path in query_tables:
            if not os.path.exists(csv_path):
                continue
            tquery = _build_query_from_csv(csv_path, search_type)
            ids = search_table2table(tquery, search_type, k)
            if ids:
                similar_table_ids.extend(ids)

    if not similar_table_ids:
        print("⚠️ No similar tables returned by tab2tab.")
        return []

    seen_tid = set()
    unique_tids: List[int] = []
    for tid in similar_table_ids:
        itid = int(tid)
        if itid not in seen_tid:
            seen_tid.add(itid)
            unique_tids.append(itid)
    unique_tids = unique_tids[: max(k, 1) * max(len(query_tables), 1)]

    table_id_to_filename = _table_ids_to_filenames(unique_tids)
    retrieved_filenames: List[str] = []
    seen_file = set()
    for tid in unique_tids:
        fname = table_id_to_filename.get(tid)
        if fname and fname not in seen_file:
            seen_file.add(fname)
            retrieved_filenames.append(fname)

    reverse = load_csvs_to_modelids(retrieved_filenames)
    similar_models: List[str] = []
    seen_mid = set()
    table_to_models: Dict[str, List[str]] = {}
    for fname in retrieved_filenames:
        base = os.path.basename(fname)
        mids = [mid for mid in reverse.get(base, []) if mid != model_id]
        table_to_models[fname] = mids
        for mid in mids:
            if mid not in seen_mid:
                seen_mid.add(mid)
                similar_models.append(mid)
    final_results = similar_models[:k] if k > 0 else similar_models

    print(f"✅ query_tables={len(query_tables)} retrieved_tables={len(retrieved_filenames)} model_ids={len(final_results)}")
    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "query_model": model_id,
                    "query_tables": query_tables,
                    "searched_tables": retrieved_filenames,
                    "model_ids": final_results,
                    "intermediate": {
                        "retrieved_table_ids": unique_tids,
                        "retrieved_table_filenames": retrieved_filenames,
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
    parser.add_argument("--query", default=None, help="Optional query (csv path or comma-separated list by search_type)")
    parser.add_argument("--search_type", choices=["single_column", "multi_column", "keyword", "unionable"], default="keyword")
    parser.add_argument("--k", type=int, default=10, help="Top-k model ids")
    parser.add_argument("--output_json", default=CARD2TAB2CARD_OUTPUT_JSON)
    args = parser.parse_args()

    t0 = time.time()
    parsed_query: Optional[Any] = None
    if args.query:
        if args.search_type in ("multi_column", "unionable"):
            parsed_query = pd.read_csv(args.query)
        else:
            if os.path.exists(args.query):
                parsed_query = _build_query_from_csv(args.query, args.search_type)
            else:
                parsed_query = [x.strip() for x in str(args.query).split(",") if x.strip()]

    results = search_card2tab2card(
        model_id=args.model_id,
        query=parsed_query,
        search_type=args.search_type,
        k=args.k,
        output_json=args.output_json,
    )
    print(f"Found {len(results)} model ids for {args.model_id}")
    for i, mid in enumerate(results[:20], 1):
        print(f"  {i}. {mid}")
    print(f"Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
