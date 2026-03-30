"""
Query -> Tab -> Card

Pipeline:
1) query2modelcard (get seed card + top-k prefix for candidate filtering)
2) card2tab2card (tab2tab -> parquet model ids)
3) dense rerank candidate_pool by query -> take model_top_k
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Set
import duckdb

from src.config import *
from src.search.card2tab2card import Card2Tab2CardSearch
from src.search.ir_searcher import DenseSearcher
from src.utils import _paths_for_resource_set

class Query2Tab2CardSearch(Card2Tab2CardSearch):
    def __init__(self,):
        super().__init__()
        self.query2card_map: Dict[str, List[str]] = {}
        self.model_rerank_map: List[str] = []
    
    def pipeline_w_query_reranker(
        self,
        query: str,
        con_data: duckdb.DuckDBPyConnection,
        dense: DenseSearcher,
        dense_wtable: DenseSearcher,
        *,
        search_type: str = "keyword",
        table_top_k: int = 10,
        table_resources: Optional[List[str]] = None,
        use_tab2tab_aug: bool = False,
        q2m_top_k: int = 1,
        model_top_k: int = 5,
        apply_query_rerank: bool = True,
    ) -> None:
        self.query2card(dense_wtable, query, q2m_top_k)
        self.card2tab2card_pipeline(self.query2card_map[query], con_data, table_resources, search_type, table_top_k, use_tab2tab_aug)
        # Preserve tab2card discovery order so rerank/top-k is stable across runs.
        candidate_pool = list(dict.fromkeys(sum(self.tab2card_map.values(), [])))
        if apply_query_rerank and candidate_pool:
            self.query2model_reranker(dense, query, candidate_pool, model_top_k)
        else:
            ranked = self._postprocess_reranked_models(candidate_pool)
            mtk = int(model_top_k)
            if mtk > 0:
                ranked = ranked[:mtk]
            self.model_rerank_map = ranked
        
    def query2card(self, dense_wtable: DenseSearcher, query: str, q2m_top_k: int) -> None:
        results, scores = dense_wtable.search(query, q2m_top_k)
        self.query2card_map[query] = list(results)
    
    def _postprocess_reranked_models(self, ranked_models: List[str]) -> List[str]:
        """
        Enforce a simple subset cap for ambiguous tables:
        when a retrieved table maps to multiple models, keep at most one selected
        model for that table, using the global reranked order to decide the winner.
        """
        multi_model_tables: Dict[str, List[str]] = {
            str(table_path).strip(): [str(mid).strip() for mid in (model_ids or []) if str(mid).strip()]
            for table_path, model_ids in (self.tab2card_map or {}).items()
            if str(table_path).strip() and len([str(mid).strip() for mid in (model_ids or []) if str(mid).strip()]) > 1
        }
        if not multi_model_tables:
            return list(ranked_models)

        model_to_multi_tables: Dict[str, List[str]] = {}
        for table_path, model_ids in multi_model_tables.items():
            for model_id in model_ids:
                model_to_multi_tables.setdefault(model_id, []).append(table_path)

        kept: List[str] = []
        seen_models: Set[str] = set()
        claimed_tables: Set[str] = set()
        for model_id in ranked_models:
            mid = str(model_id).strip()
            if not mid or mid in seen_models:
                continue
            related_tables = model_to_multi_tables.get(mid, [])
            if related_tables and all(table_path in claimed_tables for table_path in related_tables):
                continue
            kept.append(mid)
            seen_models.add(mid)
            for table_path in related_tables:
                claimed_tables.add(table_path)
        return kept

    def query2model_reranker(self, dense: DenseSearcher, query: str, candidate_pool: List[str], model_top_k: int) -> List[str]:
        full_k = max(1, len(candidate_pool))
        results, scores = dense.search_subset(query, candidate_pool, full_k)
        postprocessed = self._postprocess_reranked_models(list(results))
        mtk = int(model_top_k)
        if mtk > 0:
            postprocessed = postprocessed[:mtk]
        self.model_rerank_map = postprocessed
        return postprocessed

    def save_full_json(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"query2card_map": self.query2card_map, "model_rerank_map": self.model_rerank_map, "card2tab_map": self.card2tab_map, "tab2tab_map": self.tab2tab_map, "tab2card_map": self.tab2card_map}, f, ensure_ascii=False, indent=2)
    def load_full_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.query2card_map = data["query2card_map"]
        self.model_rerank_map = data["model_rerank_map"]
        self.card2tab_map = data["card2tab_map"]
        self.tab2tab_map = data["tab2tab_map"]
        self.tab2card_map = data["tab2card_map"]
    def get_full_map(self) -> Dict[str, object]:
        return {
            "query2card_map": self.query2card_map,
            "card2tab_map": self.card2tab_map,
            "tab2tab_map": self.tab2tab_map,
            "tab2card_map": self.tab2card_map,
            "model_rerank_map": self.model_rerank_map,
        }

def main() -> None:
    parser = argparse.ArgumentParser(description="Query -> Tab -> Card")
    parser.add_argument("--query", required=True, help="Original user query")
    parser.add_argument("--search_type", choices=["single_column", "multi_column", "keyword", "unionable"], default="keyword")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--resources", nargs="+", default=["hugging"], choices=["hugging", "github", "arxiv"], help="Table resource filter.")
    parser.add_argument("--table_top_k", type=int, default=10, help="Top-k retrieved tables per query table.")
    parser.add_argument("--q2m_top_k", type=int, default=1, help="Top-k candidates from query2modelcard for choosing seed model.")
    parser.add_argument("--use_tab2tab_aug", action="store_true", help="Use tab2tab augmentation.")
    parser.add_argument("--model_top_k", type=int, default=5, help="Final number of models after query rerank (0 = no cap).")
    args = parser.parse_args()

    resources = [str(r).strip().lower() for r in (args.resources or ["hugging"]) if str(r).strip()]
    _, _, model_db_path = _paths_for_resource_set(resources)
    dense = DenseSearcher(emb_npz_path=EMB_NPZ)
    dense_wtable = DenseSearcher(emb_npz_path=EMB_NPZ_HUGGING)
    con_data = duckdb.connect(model_db_path, read_only=True)
    q2t2c = Query2Tab2CardSearch()
    q2t2c.pipeline_w_query_reranker(
        args.query,
        con_data,
        dense,
        dense_wtable,
        search_type=args.search_type,
        table_top_k=args.table_top_k,
        table_resources=args.resources,
        use_tab2tab_aug=args.use_tab2tab_aug,
        q2m_top_k=args.q2m_top_k,
        model_top_k=args.model_top_k,
    )
    con_data.close()
    if args.output_json:
        q2t2c.save_full_json(args.output_json)
    else:
        print(json.dumps(q2t2c.get_full_map(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
