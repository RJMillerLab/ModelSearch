"""
Query2Tab2Card preview from ``save_full_json`` / ``get_full_map()`` — five maps only.

Callers load JSON from disk (``card2tab2card_<search_type>.json``) or build the same dict elsewhere;
they do not parse ``search_results.json`` here.
"""

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from src.utils import load_modelid_to_csvlist, resolve_table_path

class Query2Tab2CardFullMap():
    def __init__(self, path: str):
        self.path = path
        self.full_map = self.load_query2tab2card_full_map(self.path)
        self.q2c, self.card2tab, self.tab2tab, self.tab2card, self.reranked, self.query, self.seed_models, self.query_tables, self.candidate_pool, self.tab2tab_steps, self.retrieved_unique, self.model_to_all_table_paths = self.split_query2tab2card_full_map()
    def load_query2tab2card_full_map(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert frozenset({"query2card_map", "model_rerank_map", "card2tab_map", "tab2tab_map", "tab2card_map"}) == frozenset(data.keys()), f"Missing keys in {path}"
        return data
    def split_query2tab2card_full_map(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[str], str, List[str], List[str], List[str], List[Dict[str, Any]], List[str]]:
        q2c = self.full_map["query2card_map"]
        card2tab = self.full_map["card2tab_map"]
        tab2tab = self.full_map["tab2tab_map"]
        tab2card = self.full_map["tab2card_map"]
        reranked = self.full_map["model_rerank_map"]
        #query = str(q2c.keys()[0]).strip()
        query = next(iter(q2c.keys()))
        seed_models = [str(x).strip() for x in q2c[query]]
        query_tables = list(dict.fromkeys([str(t) for t in card2tab[seed_models[0]]]))
        #candidate_pool = list(set(sum(tab2card.values(), [])))
        candidate_pool = list({x for v in tab2card.values() for x in v})
        tab2tab_steps = [{
            "query_table": str(qt),
            "retrieved_tables": [str(x) for x in rts if str(x).strip()],
        } for qt, rts in tab2tab.items()]
        retrieved_unique = list(dict.fromkeys(sum(tab2tab.values(), [])))
        model_to_all_table_paths = {mid: load_modelid_to_csvlist(mid, resources=['hugging']) for mid in reranked}
        return q2c, card2tab, tab2tab, tab2card, reranked, query, seed_models, query_tables, candidate_pool, tab2tab_steps, retrieved_unique, model_to_all_table_paths

    def _tab2tab_rows_from_maps(self) -> List[Dict[str, Any]]:
        """One row per first-seen retrieved table: models from ``tab2card`` (same as UI Step 1)."""
        rows: List[Dict[str, Any]] = []
        for _qt, rts in self.tab2tab.items():
            for rt in rts:
                rt_s = str(rt)
                rows.append({"table": os.path.basename(rt_s), "table_path": rt_s, "models": [str(x).strip() for x in self.tab2card.get(rt_s, [])]})
        return rows

    def build_backend_payload(self, *, search_type: str = "", max_models: Optional[int] = None) -> Dict[str, Any]:
        model_ids = list(self.reranked)
        if max_models is not None:
            model_ids = model_ids[:max(0, int(max_models))]

        model_to_table_paths = {
            mid: [str(p) for p in self.model_to_all_table_paths[mid] if str(p).strip()]
            for mid in model_ids
        }
        model_to_table_paths = {mid: paths for mid, paths in model_to_table_paths.items() if paths}
        models_with_tables = list(model_to_table_paths.keys())
        table_paths = list(dict.fromkeys(x for v in model_to_table_paths.values() for x in v))

        tab2tab_trace_rows = self._tab2tab_rows_from_maps()
        after_model_cap_trace_rows = []
        allowed_models = set(models_with_tables)
        for row in tab2tab_trace_rows:
            models = [mid for mid in row["models"] if mid in allowed_models]
            if models:
                after_model_cap_trace_rows.append(
                    {
                        "table": row["table"],
                        "table_path": row["table_path"],
                        "models": models,
                    }
                )

        return {
            "query": self.query,
            "search_type": search_type or None,
            "query_seed_model_id": self.seed_models[0],
            "query2modelcard_model_ids": list(self.seed_models),
            "query2tab2card_model_ids": list(model_ids),
            "query_tables": list(self.query_tables),
            "searched_tables": list(self.retrieved_unique),
            "model_ids": list(model_ids),
            "models_with_tables": models_with_tables,
            "model_to_table_paths": model_to_table_paths,
            "table_paths": table_paths,
            "mappings": {
                "card_to_related_tables": dict(self.card2tab),
                "query_table_to_retrieved_tables": dict(self.tab2tab),
                "retrieved_table_to_related_models": dict(self.tab2card),
                "model_id_to_related_tables": dict(self.card2tab),
                "tab2tab_retrieved_table_to_related_models": dict(self.tab2card),
            },
            "intermediate": {
                "table_to_models": dict(self.tab2card),
                "retrieved_table_filenames": list(self.retrieved_unique),
                "query_table_to_retrieved_tables": dict(self.tab2tab),
                "table_id_to_filename": {},
            },
            "pipeline_trace": {
                "query2modelcard": {"model_ids": list(self.seed_models)},
                "query_dense_rerank": {
                    "applied": True,
                    "model_ids_top_k": list(model_ids),
                    "model_ids_before_dense_rerank": list(self.candidate_pool),
                    "model_ids_after_dense_rerank": list(model_ids),
                },
            },
            "tab2tab_trace_rows": tab2tab_trace_rows,
            "after_model_cap_trace_rows": after_model_cap_trace_rows,
            "retrieved_table_model_rows": after_model_cap_trace_rows,
            "preview_meta": {
                "source": "query2tab2card_full_map",
                "search_type": search_type or None,
                "tab2tab_trace_rows_source": "tab2tab_map+tab2card_map",
                "tables_source": "all_from_modelcards",
                "query": self.query,
                "seed_model_ids": list(self.seed_models),
            },
            "stats": {
                "models_with_tables": len(models_with_tables),
                "total_unique_tables": len(table_paths),
                "tables_source": "all_from_modelcards",
            },
        }

    def build_ui_preview(
        self,
        *,
        table_resources: Optional[List[str]] = None,
        search_type: str = "",
    ) -> Dict[str, Any]:
        payload = self.build_backend_payload(search_type=search_type)
        return {
            "query_tables": payload["query_tables"],
            "tab2tab_trace_rows": payload["tab2tab_trace_rows"],
            "after_model_cap_trace_rows": payload["after_model_cap_trace_rows"],
            "models_with_tables_list": payload["model_ids"],
            "pipeline_trace": payload["pipeline_trace"],
            "model_to_table_paths_ts": payload["model_to_table_paths"],
            "table_paths": payload["table_paths"],
            "parquet_resources": table_resources,
            "preview_meta": payload["preview_meta"],
        }
if __name__ == "__main__":
    obj = Query2Tab2CardFullMap("data_251117/jobs_251117/2026-03-28_00-19-37_e9j2/card2tab2card_keyword.json")
    print(json.dumps(obj.build_ui_preview(), indent=2))
