"""
Query to ModelCard Search

This module provides functions for searching model cards using a text query.
"""
import os
import json
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import threading

# Ensure repo root is on sys.path for `from src...` imports even when
# this module is launched from a different working directory.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.utils import get_device, _paths_for_resource_set
from src.config import *
from src.search.ir_searcher import DenseSearcher, SparseSearcher

class Query2ModelCardSearch:
    def __init__(self, *, query: str, top_k: int, job_id: Optional[str] = None):
        self.query = query
        self.top_k = top_k
        self.job_id = job_id
        self.results: Dict[str, Any] = {}

    # ---------- search ----------
    def search_dense(self, top_k: int, dense: DenseSearcher):
        results, scores = dense.search(self.query, top_k)
        self.results["dense"] = results
        self.results["dense_scores"] = scores
        return results, scores

    def search_sparse(self, top_k: int, sparse: SparseSearcher):
        results, scores = sparse.search(self.query, top_k)
        self.results["sparse"] = results
        self.results["sparse_scores"] = scores
        return results, scores

    def search_hybrid(self, top_k: int, sparse: SparseSearcher, dense: DenseSearcher, candidate_factor: int = 10):
        candidate_ids, _ = sparse.search(self.query, top_k * candidate_factor)
        results, scores = dense.search_subset(self.query, candidate_ids, top_k)
        self.results["hybrid"] = results
        self.results["hybrid_scores"] = scores
        return results, scores

    # ---------- IO ----------
    def save_to_json(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"query": self.query, "top_k": self.top_k, "job_id": self.job_id, "results": self.results}, f, ensure_ascii=False, indent=2)
    
    def load_from_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.query = data["query"]
        self.top_k = data["top_k"]
        self.job_id = data["job_id"]
        self.results = data["results"]

def main():
    """CLI entry point for query2modelcard search"""
    parser = argparse.ArgumentParser(description="Query to ModelCard Search")
    parser.add_argument('--query', required=True, help='Text query string')
    parser.add_argument('--top_k', type=int, default=20, help='Number of results to return (default: 20)')
    parser.add_argument('--output_json', default=None, help='Optional path to save results as JSON')
    parser.add_argument('--retrieval_mode', choices=['dense', 'sparse', 'hybrid'], required=True, help='Retrieval mode.')
    parser.add_argument('--resources', nargs='+', default=['hugging', 'github', 'arxiv'], choices=['hugging', 'github', 'arxiv'], help='Resource labels. If and only if hugging/hf is selected alone, use hugging subset indexes.')
    parser.add_argument('--candidate_factor', type=int, default=10, help='Hybrid: sparse topk multiplier.')
    
    args = parser.parse_args()
    start_time = time.time()

    resources = [str(r).strip().lower() for r in (args.resources or []) if str(r).strip()]
    emb_npz_path, sparse_index_path, _ = _paths_for_resource_set(resources)

    from src.search.ir_searcher import DenseSearcher, SparseSearcher

    dense: DenseSearcher = None
    sparse: SparseSearcher = None
    if args.retrieval_mode in {"dense", "hybrid"}:
        dense = DenseSearcher(emb_npz_path=emb_npz_path)
    if args.retrieval_mode in {"sparse", "hybrid"}:
        sparse = SparseSearcher(index_path=sparse_index_path)
    q2m = Query2ModelCardSearch(query=args.query, top_k=args.top_k)
    if args.retrieval_mode == "dense":
        q2m.search_dense(top_k=args.top_k, dense=dense)
    elif args.retrieval_mode == "sparse":
        q2m.search_sparse(top_k=args.top_k, sparse=sparse)
    elif args.retrieval_mode == "hybrid":
        q2m.search_hybrid(top_k=args.top_k, sparse=sparse, dense=dense, candidate_factor=args.candidate_factor)
    if args.output_json:
        q2m.save_to_json(args.output_json)
    
    print(f"\nTotal time: {time.time() - start_time:.2f}s (device: {get_device()})")
    print(q2m.results)

if __name__ == '__main__':
    main()

