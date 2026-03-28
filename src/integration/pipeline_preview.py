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

    def build_ui_preview(
        self,
        *,
        table_resources: Optional[List[str]] = None,
        search_type: str = "",
    ) -> Dict[str, Any]:
        """
        Build the same preview shape the demo UI expects (``all_from_modelcards`` branch), without ``search_results.json``:

        - Query tables (from ``card2tab`` for seeds)
        - Tab2Tab rows: retrieved table -> models (``tab2card``)
        - Reranked list: ``model_rerank_map``
        - **All tables from modelcards**: ``_get_models_to_tables_batch_sql`` + ``resolve_table_path`` (parquet expansion).
        """
        tab2tab_trace_rows = self._tab2tab_rows_from_maps()
        
        table_paths = list(set(x for v in self.model_to_all_table_paths.values() for x in v))

        pipeline_trace = {
            "tab2tab": {},
            "model_ids_before_dense_rerank": self.candidate_pool,
            "model_ids_after_dense_rerank": self.reranked,
            "model_ids_top_k": self.reranked,
            "dense_rerank_applied": bool(self.candidate_pool and self.reranked),
        }

        preview_meta = {
            "source": "query2tab2card_full_map",
            "search_type": search_type or None,
            "tab2tab_trace_rows_source": "tab2tab_map+tab2card_map",
            "tables_source": "all_from_modelcards",
            "query": self.query,
            "seed_model_ids": self.seed_models,
        }

        return {
            "query_tables": self.query_tables,
            "tab2tab_trace_rows": tab2tab_trace_rows,
            "after_model_cap_trace_rows": list(tab2tab_trace_rows),
            "models_with_tables_list": self.reranked,
            "pipeline_trace": pipeline_trace,
            "model_to_table_paths_ts": self.model_to_all_table_paths,
            "table_paths": table_paths,
            "parquet_resources": table_resources,
            "preview_meta": preview_meta,
        }
if __name__ == "__main__":
    obj = Query2Tab2CardFullMap("data_251117/jobs_251117/2026-03-28_00-19-37_e9j2/card2tab2card_keyword.json")
    print(json.dumps(obj.build_ui_preview(), indent=2))
