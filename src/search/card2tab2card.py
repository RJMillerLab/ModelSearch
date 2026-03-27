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
from src.utils import load_modelid_to_csvlist, load_csvs_to_modelids, _paths_for_resource_set

class Card2Tab2CardSearch:
    def __init__(self,):
        self.card2tab_map: Dict[str, List[str]] = {}
        self.tab2tab_map: Dict[str, List[str]] = {}
        self.tab2card_map: Dict[str, List[str]] = {}
    
    def card2tab(self, modelid_list: List[str], table_resources: Optional[List[str]] = None) -> Dict[str, object]:
        card2tab_map = {}
        for model_id in modelid_list:
            query_tables = load_modelid_to_csvlist(model_id, resources=table_resources)
            card2tab_map[model_id] = query_tables
        self.card2tab_map = card2tab_map
    
    def tab2tab(self, query_tables: List[str], con_data: duckdb.DuckDBPyConnection, search_type: str = "keyword", table_top_k: int = 10, table_resources: Optional[List[str]] = None, use_tab2tab_aug: bool = False) -> List[str]:
        query_table_to_retrieved_tables = {}
        for query_table in query_tables:
            names = self.tab2tab_one(query_table, con_data, search_type, table_top_k, table_resources, use_tab2tab_aug)
            query_table_to_retrieved_tables[query_table] = names
        self.tab2tab_map = query_table_to_retrieved_tables
    
    def tab2tab_one(self, query_table: str, con_data: duckdb.DuckDBPyConnection, search_type: str = "keyword", table_top_k: int = 10, table_resources: Optional[List[str]] = None, use_tab2tab_aug: bool = False) -> List[str]:
        if use_tab2tab_aug:
            names = search_tab2tab_aug(search_type=search_type, query=query_table, k=table_top_k, output_json=None, con_data=con_data, query_augmentation_types=["ori", "tr", "str"], candidate_augmentation_types=["ori", "tr", "str"], rerank_mode="table_level")
        else:
            names = search_table2table(search_type=search_type, query=query_table, k=table_top_k, output_json=None, con_data=con_data, augmentation_types=["ori"])
        return [str(n).strip() for n in (names or []) if str(n).strip()]
    
    def tab2card(self, table_ids: List[str], modelid_list: List[str]) -> List[str]:
        tab2card_map = load_csvs_to_modelids(table_ids)
        if modelid_list is None:
            table_to_models = tab2card_map
        else: # filter out modelcard that are already in the query model cards
            table_to_models = {bn: [mid for mid in tab2card_map.get(bn, []) if mid not in modelid_list] for bn in table_ids}
        self.tab2card_map = table_to_models
    
    def save_to_json(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"card2tab_map": self.card2tab_map, "tab2tab_map": self.tab2tab_map, "tab2card_map": self.tab2card_map}, f, ensure_ascii=False, indent=2)
    
    def load_from_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.card2tab_map = data["card2tab_map"]
        self.tab2tab_map = data["tab2tab_map"]
        self.tab2card_map = data["tab2card_map"]
    
    def card2tab2card_pipeline(
        self,
        modelid_list: List[str],
        con_data: duckdb.DuckDBPyConnection,
        table_resources: Optional[List[str]] = None,
        search_type: str = "keyword",
        table_top_k: int = 10,
        use_tab2tab_aug: bool = False,
    ) -> Dict[str, object]:
        self.card2tab(modelid_list, table_resources)
        # should extend, not list, and dedup
        query_tables = list(set(sum(self.card2tab_map.values(), [])))
        self.tab2tab(query_tables, con_data, search_type, table_top_k, table_resources, use_tab2tab_aug)
        retrieved_tables = list(set(sum(self.tab2tab_map.values(), [])))
        self.tab2card(retrieved_tables, modelid_list)
    
    def get_full_map(self) -> Dict[str, object]:
        return {
            "card2tab_map": self.card2tab_map,
            "tab2tab_map": self.tab2tab_map,
            "tab2card_map": self.tab2card_map,
        }

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
    _, _, db_path = _paths_for_resource_set(resources)
    con_data = duckdb.connect(db_path, read_only=True)

    c2t2c = Card2Tab2CardSearch()
    c2t2c.card2tab2card_pipeline(modelid_list=[args.model_id], table_resources=resources, con_data=con_data, search_type=args.search_type, table_top_k=args.table_top_k, use_tab2tab_aug=False)
    if args.output_json:
        c2t2c.save_to_json(args.output_json)
    else:
        print(c2t2c.get_full_map())

if __name__ == "__main__":
    main()
