"""
Query2Tab2Card preview preparation.

This module extracts pipeline relationship information (query table -> retrieved tables -> related models)
from saved search results, without running table integration.
"""

import os
from typing import Any, Dict, List, Optional, Set, Tuple

from src.utils import _get_models_to_tables_batch_sql, resolve_table_path


def _mid_from_entry(m: Any) -> Optional[str]:
    if m is None:
        return None
    if isinstance(m, dict):
        mid = m.get("model_id") or m.get("modelId")
        if mid is None:
            return None
        s = str(mid).strip()
        return s if s else None
    s = str(m).strip()
    return s if s else None


def _table_model_rows_ordered(filenames: List[str], table_to_models: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(table_to_models, dict):
        table_to_models = {}
    basename_to_key = {os.path.basename(str(k)): k for k in table_to_models.keys()}
    rows: List[Dict[str, Any]] = []
    for tp in filenames or []:
        tp_s = str(tp)
        bn = os.path.basename(tp_s)
        key = tp_s if tp_s in table_to_models else basename_to_key.get(bn)
        model_list = table_to_models.get(key, []) if key else []
        mids: List[str] = []
        for m in (model_list if isinstance(model_list, list) else []):
            mid = _mid_from_entry(m)
            if mid and mid not in mids:
                mids.append(mid)
        rows.append({"table": bn, "table_path": tp_s, "models": mids})
    return rows


def _table_resources_for_integration(search_results: Dict[str, Any]) -> Optional[List[str]]:
    from src.config import TABLE_RESOURCE_ALLOWLIST

    if "table_resources" not in search_results:
        return None
    tr = search_results.get("table_resources")
    if isinstance(tr, list) and tr:
        out = [
            str(x).strip().lower()
            for x in tr
            if str(x).strip() and str(x).strip().lower() in ("hugging", "github", "arxiv", "llm")
        ]
        if out:
            return out
    fallback = [r for r in TABLE_RESOURCE_ALLOWLIST if r in ("hugging", "github", "arxiv", "llm")]
    return fallback


def _resolve_table_paths_for_model_ids(
    model_ids: List[str], *, resources: Optional[List[str]] = None
) -> Tuple[List[str], Dict[str, List[str]]]:
    model_to_tables = _get_models_to_tables_batch_sql(model_ids, resources=resources)
    seen_paths: Set[str] = set()
    table_paths: List[str] = []
    model_to_table_paths: Dict[str, List[str]] = {mid: [] for mid in model_ids}
    for mid in model_ids:
        for basename in model_to_tables.get(mid, []):
            resolved_path = resolve_table_path(basename)
            if not resolved_path:
                continue
            if resolved_path not in seen_paths:
                seen_paths.add(resolved_path)
                table_paths.append(resolved_path)
            model_to_table_paths[mid].append(resolved_path)
    return table_paths, model_to_table_paths


def prepare_query2tab2card_preview(
    search_results: Dict[str, Any],
    search_type: str,
    tables_source: str,
    table_resources: Optional[List[str]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    parquet_resources: Optional[List[str]] = (
        table_resources
        if isinstance(table_resources, list) and table_resources
        else _table_resources_for_integration(search_results)
    )

    c2t2c = search_results.get("card2tab2card_results")
    if not isinstance(c2t2c, dict):
        return None, "search_results must contain card2tab2card_results"

    payload = c2t2c.get(search_type)
    if not isinstance(payload, dict):
        return None, f"Search type {search_type!r} not found. Available: {', '.join(c2t2c.keys())}"

    inter = payload.get("intermediate") if isinstance(payload.get("intermediate"), dict) else {}
    mappings = payload.get("mappings") if isinstance(payload.get("mappings"), dict) else {}
    mapped_t2m = mappings.get("retrieved_table_to_related_models", {})
    table_to_models = mapped_t2m if isinstance(mapped_t2m, dict) and mapped_t2m else inter.get("table_to_models", {})
    if not isinstance(table_to_models, dict):
        table_to_models = {}

    retrieved_filenames = payload.get("searched_tables", [])
    if not isinstance(retrieved_filenames, list) or not retrieved_filenames:
        retrieved_filenames = inter.get("retrieved_table_filenames", [])
    if not isinstance(retrieved_filenames, list):
        retrieved_filenames = []
    if tables_source == "intermediate" and not retrieved_filenames:
        return None, "No retrieved tables (searched_tables or intermediate.retrieved_table_filenames)"

    pipeline_trace_raw = payload.get("pipeline_trace") if isinstance(payload.get("pipeline_trace"), dict) else {}
    c2t2c_trace = (
        pipeline_trace_raw.get("card2tab2card")
        if isinstance(pipeline_trace_raw.get("card2tab2card"), dict)
        else pipeline_trace_raw
    )
    q_rerank_trace = (
        pipeline_trace_raw.get("query_dense_rerank")
        if isinstance(pipeline_trace_raw.get("query_dense_rerank"), dict)
        else {}
    )
    tab2tab_block = c2t2c_trace.get("tab2tab") if isinstance(c2t2c_trace.get("tab2tab"), dict) else {}
    after_cap = c2t2c_trace.get("after_model_cap") if isinstance(c2t2c_trace.get("after_model_cap"), dict) else {}

    if tables_source == "all_from_modelcards":
        model_ids_list = [str(x) for x in (payload.get("model_ids") or []) if x is not None]
        if not model_ids_list:
            model_ids_set: Set[str] = set()
            for model_list in table_to_models.values():
                for m in (model_list if isinstance(model_list, list) else []):
                    mid = _mid_from_entry(m)
                    if mid:
                        model_ids_set.add(mid)
            model_ids_list = list(model_ids_set)
        if not model_ids_list:
            return None, "No model IDs for all_from_modelcards (empty model_ids and table_to_models)"
        table_paths, model_to_table_paths_ts = _resolve_table_paths_for_model_ids(model_ids_list, resources=parquet_resources)
        models_with_tables_list = model_ids_list
    else:
        table_paths = list(retrieved_filenames)
        c2_ids = payload.get("model_ids") if isinstance(payload.get("model_ids"), list) else []
        if c2_ids:
            models_with_tables_list = [str(x) for x in c2_ids]
        else:
            basename_to_key = {os.path.basename(key): key for key in table_to_models}
            ids_set: Set[str] = set()
            for tp in table_paths:
                model_list = table_to_models.get(tp) or table_to_models.get(basename_to_key.get(os.path.basename(tp)))
                for m in (model_list or []):
                    mid = _mid_from_entry(m)
                    if mid:
                        ids_set.add(mid)
            models_with_tables_list = list(ids_set)
        model_to_table_paths_ts: Dict[str, List[str]] = {}
        basename_to_key = {os.path.basename(key): key for key in table_to_models}
        for tp in table_paths:
            key = tp if tp in table_to_models else basename_to_key.get(os.path.basename(tp))
            model_list = table_to_models.get(key, []) if key else []
            for m in (model_list or []):
                mid = _mid_from_entry(m)
                if mid:
                    model_to_table_paths_ts.setdefault(mid, []).append(tp)

    if not table_paths:
        return None, "No tables to integrate"

    tab2tab_trace_rows = _table_model_rows_ordered(
        tab2tab_block.get("searched_tables", []) if isinstance(tab2tab_block.get("searched_tables"), list) else [],
        tab2tab_block.get("table_to_models", {}) if isinstance(tab2tab_block.get("table_to_models"), dict) else {},
    )
    after_model_cap_trace_rows = _table_model_rows_ordered(
        after_cap.get("searched_tables", retrieved_filenames) if isinstance(after_cap.get("searched_tables"), list) else retrieved_filenames,
        after_cap.get("table_to_models", table_to_models) if isinstance(after_cap.get("table_to_models"), dict) else table_to_models,
    )

    model_ids_top_k = q_rerank_trace.get("model_ids_top_k")
    if not isinstance(model_ids_top_k, list):
        model_ids_top_k = payload.get("model_ids") if isinstance(payload.get("model_ids"), list) else []

    pipeline_trace_normalized = {
        "tab2tab": tab2tab_block if isinstance(tab2tab_block, dict) else {},
        "model_ids_before_dense_rerank": (
            q_rerank_trace.get("model_ids_before_dense_rerank")
            if isinstance(q_rerank_trace.get("model_ids_before_dense_rerank"), list)
            else c2t2c_trace.get("model_ids_before_dense_rerank", [])
        ),
        "model_ids_after_dense_rerank": (
            q_rerank_trace.get("model_ids_after_dense_rerank")
            if isinstance(q_rerank_trace.get("model_ids_after_dense_rerank"), list)
            else c2t2c_trace.get("model_ids_after_dense_rerank", [])
        ),
        "model_ids_top_k": model_ids_top_k,
        "dense_rerank_applied": bool(
            q_rerank_trace.get("applied", c2t2c_trace.get("dense_rerank_applied", False))
        ),
    }
    if after_cap:
        pipeline_trace_normalized["after_model_cap"] = after_cap

    return {
        "table_paths": table_paths,
        "model_to_table_paths_ts": model_to_table_paths_ts,
        "query_tables": list(payload.get("query_tables") or []),
        "models_with_tables_list": models_with_tables_list,
        "parquet_resources": parquet_resources,
        "pipeline_trace": pipeline_trace_normalized,
        "tab2tab_trace_rows": tab2tab_trace_rows,
        "after_model_cap_trace_rows": after_model_cap_trace_rows,
    }, None
