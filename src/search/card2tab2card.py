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
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

#from src.config import MODELLAKE_DB, CARD2TAB2CARD_OUTPUT_JSON
from src.config import *
from src.search.tab2tab import search_table2table
from src.utils import load_modelid_to_csvlist, load_csvs_to_modelids, resolve_table_path


def _table_resources_hugging_only(table_resources: Optional[List[str]]) -> bool:
    rs = {str(r).strip().lower() for r in (table_resources or ["hugging"]) if str(r).strip()}
    return rs == {"hugging"}


def search_card2tab2card(   
    model_id: str,
    search_type: str = "keyword",
    output_json: str = "",
    db_path: str = MODELLAKE_DB,
    table_top_k: int = 10,
    table_resources: Optional[List[str]] = None,
)-> Dict[str, object]:
    """Simplified card->tab->card search using relationship parquet + tab2tab import."""
    print(f"[Card2Tab2Card] model_id={model_id} search_type={search_type} table_top_k={table_top_k} table_resources={table_resources!r}")
    use_tab2tab_aug = bool(USE_TAB2TAB_AUG) and _table_resources_hugging_only(table_resources)
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
        empty_payload: Dict[str, object] = {
            "query_model": model_id,
            "query_tables": [],
            "searched_tables": [],
            "model_ids": [],
            "mappings": {
                "card_to_related_tables": {model_id: []},
                "query_table_to_retrieved_tables": {},
                "retrieved_table_to_related_models": {},
            },
            "intermediate": {
                "retrieved_table_filenames": [],
                "table_id_to_filename": {},
                "table_to_models": {},
                "query_table_to_retrieved_tables": {},
            },
            "pipeline_trace": {
                "tab2tab": {"backend": "aug" if use_tab2tab_aug else "classic"},
            },
        }
        if output_json:
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(empty_payload, f, ensure_ascii=False, indent=2)
        return empty_payload
    else:
        print(f"query_tables example: {query_tables[0]}")
    print(f"[c2t2c-trace] query_tables ({len(query_tables)}):", flush=True)
    for qp in query_tables:
        print(f"  seed_csv: {os.path.basename(qp)}  path={qp}", flush=True)

    if use_tab2tab_aug:
        from src.search.tab2tab_aug import search_tab2tab_aug as _search_tab2tab_aug

        print(
            "[c2t2c-trace] tab2tab_backend=aug (search_tab2tab_aug: 9 lanes + rerank; "
            f"MODELLAKE_DB_HUGGING + TRANSPOSED)",
            flush=True,
        )
    else:
        _search_tab2tab_aug = None  # type: ignore[assignment]
        print(
            f"[c2t2c-trace] tab2tab_backend=classic search_table2table db_path={os.path.abspath(db_path)}",
            flush=True,
        )

    # table2table search → CSV basenames (same list shape for aug or classic)
    similar_basenames: List[str] = []
    query_table_to_retrieved_tables: Dict[str, List[str]] = {}

    def _tab2tab_one(csv_path: str) -> Tuple[str, List[str]]:
        if use_tab2tab_aug:
            names = _search_tab2tab_aug(
                search_type=search_type,
                query=csv_path,
                k=table_top_k,
                db_original=MODELLAKE_DB_HUGGING,
                db_transposed=MODELLAKE_DB_HUGGING_TRANSPOSED,
                output_json=None,
            )
        else:
            names = search_table2table(
                query=csv_path, search_type=search_type, k=table_top_k, db_path=db_path
            )
        n_list = list(names) if names else []
        return csv_path, n_list

    paths_ok = [p for p in query_tables if os.path.exists(p)]
    parallel_tab2tab = os.environ.get("CARD2TAB2CARD_PARALLEL_TAB2TAB", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "",
    )
    rows: List[Tuple[str, List[str]]] = []
    if parallel_tab2tab and len(paths_ok) > 1:
        max_workers = min(8, len(paths_ok))
        print(
            f"[c2t2c-trace] tab2tab parallel: {len(paths_ok)} seed tables, max_workers={max_workers}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_tab2tab_one, p) for p in paths_ok]
            rows = [f.result() for f in futures]
    else:
        for p in paths_ok:
            rows.append(_tab2tab_one(p))

    for csv_path, n_list in rows:
        sample = n_list[:15]
        more = f" ...(+{len(n_list) - 15})" if len(n_list) > 15 else ""
        print(
            f"[c2t2c-trace] tab2tab query={os.path.basename(csv_path)} search_type={search_type} k={table_top_k} "
            f"-> n_filenames={len(n_list)} sample={sample}{more}",
            flush=True,
        )
        if n_list:
            similar_basenames.extend(n_list)
        query_table_to_retrieved_tables[os.path.basename(csv_path)] = list(dict.fromkeys(n_list))

    if not similar_basenames:
        print("⚠️ No similar tables returned by tab2tab.")
        empty_payload = {
            "query_model": model_id,
            "query_tables": query_tables,
            "searched_tables": [],
            "model_ids": [],
            "mappings": {
                "card_to_related_tables": {model_id: [os.path.basename(p) for p in query_tables]},
                "query_table_to_retrieved_tables": query_table_to_retrieved_tables,
                "retrieved_table_to_related_models": {},
            },
            "intermediate": {
                "retrieved_table_filenames": [],
                "table_id_to_filename": {},
                "table_to_models": {},
                "query_table_to_retrieved_tables": query_table_to_retrieved_tables,
            },
            "pipeline_trace": {
                "tab2tab": {"backend": "aug" if use_tab2tab_aug else "classic"},
            },
        }
        # Downstream expects `--output_json` to always be written so that
        # backend/frontend don't fail with "No JSON at ...".
        if output_json:
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(empty_payload, f, ensure_ascii=False, indent=2)
            print(f"✅ Results saved to {output_json} (empty)")
        return empty_payload

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
    print(
        f"[c2t2c-trace] unique models from tab2tab->parquet: {len(final_results)} (no model_top_k cap in card2tab2card)",
        flush=True,
    )
    print(f"[c2t2c-trace] final model_ids order: {final_results}", flush=True)

    print(f"✅ query_tables={len(query_tables)} retrieved_tables={len(retrieved_tables)} model_ids={len(final_results)}")
    result_payload: Dict[str, object] = {
        "query_model": model_id,
        "query_tables": query_tables,
        "searched_tables": retrieved_tables,
        "model_ids": final_results,
        "mappings": {
            "card_to_related_tables": {model_id: [os.path.basename(p) for p in query_tables]},
            "query_table_to_retrieved_tables": query_table_to_retrieved_tables,
            "retrieved_table_to_related_models": table_to_models,
        },
        "intermediate": {
            "retrieved_table_filenames": list(retrieved_tables),
            "table_id_to_filename": {},
            "table_to_models": table_to_models,
            "query_table_to_retrieved_tables": query_table_to_retrieved_tables,
        },
        "pipeline_trace": {
            "tab2tab": {
                "backend": "aug" if use_tab2tab_aug else "classic",
                "searched_tables": list(retrieved_tables),
                "retrieved_table_filenames": list(retrieved_tables),
                "table_to_models": table_to_models,
                "query_table_to_retrieved_tables": query_table_to_retrieved_tables,
            },
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

    print(
        "[card2tab2card] artifacts: "
        f"resources={resources!r} | "
        f"RELATIONSHIP_PARQUET={os.path.abspath(RELATIONSHIP_PARQUET)} | "
        f"MODELLAKE_DB={os.path.abspath(db_path)} | "
        f"TABLE_BASE_DIRS={[os.path.abspath(d) for d in TABLE_BASE_DIRS]!r}",
        flush=True,
    )

    t0 = time.time()
    
    payload = search_card2tab2card(
        model_id=args.model_id,
        search_type=args.search_type,
        table_top_k=args.table_top_k,
        output_json=args.output_json,
        db_path=db_path,
        table_resources=resources,
    )
    results = payload.get("model_ids", []) if isinstance(payload, dict) else []
    print(f"Found {len(results)} model ids for {args.model_id}")
    for i, mid in enumerate(results[:20], 1):
        print(f"  {i}. {mid}")
    print(f"Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
