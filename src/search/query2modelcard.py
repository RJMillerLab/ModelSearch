"""
Query to ModelCard Search

This module provides functions for searching model cards using a text query.
"""

import os
import json
import sys
import time
from typing import Any, List, Optional, Tuple
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Ensure repo root is on sys.path for `from src...` imports even when
# this module is launched from a different working directory.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.utils import get_device
from src.search.card2card import _build_faiss_index_in_memory, _get_pyserini_searcher_and_reader
#from src.config import EMB_NPZ, ENCODE_MODEL, SPARSE_INDEX
from src.config import *

def _search_sparse_query(
    query_text: str,
    *,
    top_k: int,
    sparse_index_path: str = SPARSE_INDEX,
) -> Tuple[List[str], List[float]]:
    """
    Sparse retrieval: run BM25 over Lucene index using raw query text.
    """
    searcher, _index_reader = _get_pyserini_searcher_and_reader(sparse_index_path)
    hits = searcher.search(query_text, k=top_k)
    docids = [h.docid for h in hits]
    scores = [float(h.score) for h in hits]
    return docids, scores


def search_query2modelcard(
    query: str,
    *,
    top_k: int = 20,
    output_json: Optional[str] = None,
    retrieval_mode: str = "dense",
    candidate_factor: int = 10,
    emb_npz_path: str = EMB_NPZ,
    sparse_index_path: str = SPARSE_INDEX,
) -> List[str]:
    """
    Search for model cards using a text query.

    retrieval_mode:
    - dense: FAISS (cosine similarity over embeddings)
    - sparse: Pyserini BM25 (Lucene) over raw query text
    - hybrid: sparse candidates (top_k*candidate_factor) -> dense re-ranking on candidate subset (no RRF)
    """
    if retrieval_mode not in {"dense", "sparse", "hybrid"}:
        raise ValueError(f"retrieval_mode must be one of dense|sparse|hybrid, got {retrieval_mode!r}")

    # Dense and hybrid need embeddings.
    # Backward compatible: older EMB_NPZ may have stored `ids` as dtype=object.
    data = np.load(emb_npz_path, allow_pickle=True) if retrieval_mode in {"dense", "hybrid"} else None
    ids = data["ids"].tolist() if data is not None else None
    embs = np.asarray(data["embeddings"], dtype=np.float32) if data is not None else None
    results: List[str]
    scores: List[float]
    if retrieval_mode == "sparse":
        results, scores = _search_sparse_query(query, top_k=top_k, sparse_index_path=sparse_index_path)
    else:
        # Encode query for dense and hybrid.
        model = SentenceTransformer(ENCODE_MODEL, device=get_device())
        model.eval()
        query_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
        faiss.normalize_L2(query_emb)
        if retrieval_mode == "dense":
            index, _ = _build_faiss_index_in_memory(embs)
            D, I = index.search(query_emb, top_k)
            results = [ids[i] for i in I[0]]
            scores = D[0].tolist() if len(D) > 0 else []
        else:
            # hybrid
            sparse_k = top_k * candidate_factor
            candidate_ids, _ = _search_sparse_query(query, top_k=sparse_k, sparse_index_path=sparse_index_path)

            # Filter candidates to those present in embeddings, preserve order, drop duplicates.
            seen = set()
            filtered_candidates: List[str] = []
            for cid in candidate_ids:
                if cid in ids and cid not in seen:
                    filtered_candidates.append(cid)
                    seen.add(cid)
            if not filtered_candidates:
                results, scores = [], []
            else:
                id_to_idx = {mid: i for i, mid in enumerate(ids)}
                candidate_indices = [id_to_idx[cid] for cid in filtered_candidates]
                candidate_embs = embs[candidate_indices]

                subset_index, _ = _build_faiss_index_in_memory(candidate_embs)
                subset_k = min(top_k, len(filtered_candidates))
                D, I = subset_index.search(query_emb, subset_k)
                results = [filtered_candidates[i] for i in I[0]]
                scores = D[0].tolist() if len(D) > 0 else []

    if output_json:
        result = {"query": query, "retrieval_mode": retrieval_mode, "results": results, "scores": scores}
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved to {output_json}")
    return results


def dense_rerank_model_ids_by_query(
    query: str,
    model_ids: List[Any],
    *,
    emb_npz_path: str = EMB_NPZ,
) -> List[str]:
    """
    Re-score and sort candidate model ids by cosine(query_emb, card_emb) using only rows present in emb_npz.
    Loads the full npz file but only reads embedding rows for ids in ``model_ids`` (plus one query encode).
    Ids missing from the npz are sorted last (stable).
    """
    if not model_ids:
        return []
    q = (query or "").strip()
    if not q:
        return [str(mid).strip() for mid in model_ids if str(mid).strip()]

    data = np.load(emb_npz_path, allow_pickle=True)
    ids = data["ids"].tolist()
    embs = np.asarray(data["embeddings"], dtype=np.float32)
    id_to_idx = {str(mid): i for i, mid in enumerate(ids)}

    model = SentenceTransformer(ENCODE_MODEL, device=get_device())
    model.eval()
    query_emb = model.encode([q], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    faiss.normalize_L2(query_emb)

    scored: List[Tuple[float, int, str]] = []
    for j, mid in enumerate(model_ids):
        sm = str(mid).strip()
        if not sm:
            continue
        idx = id_to_idx.get(sm)
        if idx is None:
            scored.append((float("-inf"), j, sm))
            continue
        row = embs[idx : idx + 1].copy()
        faiss.normalize_L2(row)
        sim = float(np.dot(query_emb[0], row[0]))
        scored.append((sim, j, sm))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [t[2] for t in scored]


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
    resource_set = set(resources)
    if resource_set == {'hugging'}:
        emb_npz_path = EMB_NPZ_HUGGING
        sparse_index_path = SPARSE_INDEX_HUGGING
    elif resource_set == {'hugging', 'github', 'arxiv'}:
        emb_npz_path = EMB_NPZ
        sparse_index_path = SPARSE_INDEX
    else:
        raise NotImplementedError(f"Unsupported resource combination: {resource_set}. Must be one of: {'hugging', 'github', 'arxiv'}")

    print(
        "[query2modelcard] artifacts: "
        f"resources={resources} | "
        f"embeddings_npz={os.path.abspath(emb_npz_path)} | "
        f"sparse_index={os.path.abspath(sparse_index_path)} | "
        f"encode_model={ENCODE_MODEL!r}",
        flush=True,
    )

    results = search_query2modelcard(query=args.query, top_k=args.top_k, output_json=args.output_json, retrieval_mode=args.retrieval_mode, candidate_factor=args.candidate_factor, emb_npz_path=emb_npz_path, sparse_index_path=sparse_index_path)
    
    print(f"Found {len(results)} model cards for query: '{args.query}'")
    for i, model_id in enumerate(results, 1):
        print(f"  {i}. {model_id}")
    print(f"\nTotal time: {time.time() - start_time:.2f}s (device: {get_device()})")

if __name__ == '__main__':
    main()

