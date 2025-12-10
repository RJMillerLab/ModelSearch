"""
Test script for Card2Card hybrid retrieval (sparse + dense)

This script demonstrates how to implement sparse retrieval using BM25
and combine it with existing dense retrieval (FAISS) for hybrid search.
"""

import os
import sys
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Default paths
DEFAULT_JSONL = "data/card2card_corpus.jsonl"
DEFAULT_EMB_NPZ = "data/card2card_embeddings.npz"
DEFAULT_FAISS_INDEX = "data/card2card.faiss"


def load_corpus(jsonl_path: str) -> Tuple[List[str], List[str]]:
    """
    Load corpus from JSONL file.
    
    Returns:
        (model_ids, texts): List of model IDs and corresponding texts
    """
    model_ids = []
    texts = []
    
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Corpus file not found: {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        line_num = 0
        for line in f:
            line_num += 1
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                if 'id' not in doc or 'contents' not in doc:
                    print(f"⚠️  Skipping line {line_num}: missing 'id' or 'contents' field")
                    continue
                model_ids.append(doc['id'])
                texts.append(doc['contents'])
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping line {line_num}: JSON decode error - {str(e)}")
                continue
    
    print(f"✅ Loaded {len(model_ids)} documents from {jsonl_path}")
    return model_ids, texts


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25.
    Split by whitespace and convert to lowercase.
    """
    return text.lower().split()


def build_bm25_index(texts: List[str]) -> BM25Okapi:
    """
    Build BM25 index from texts.
    
    Args:
        texts: List of document texts
    
    Returns:
        BM25Okapi index
    """
    tokenized_texts = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    print(f"✅ Built BM25 index for {len(texts)} documents")
    return bm25


def sparse_search(
    query_model_id: str,
    model_ids: List[str],
    texts: List[str],
    bm25_index: BM25Okapi,
    top_k: int = 20
) -> List[Tuple[str, float]]:
    """
    Perform sparse retrieval using BM25.
    
    Args:
        query_model_id: Model ID to search for
        model_ids: List of all model IDs
        texts: List of all texts
        bm25_index: BM25 index
        top_k: Number of results to return
    
    Returns:
        List of (model_id, score) tuples, sorted by score descending
    """
    # Find query text
    try:
        query_idx = model_ids.index(query_model_id)
    except ValueError:
        raise ValueError(f"Model ID '{query_model_id}' not found in corpus")
    
    query_text = texts[query_idx]
    tokenized_query = tokenize(query_text)
    
    # Get BM25 scores
    scores = bm25_index.get_scores(tokenized_query)
    
    # Create list of (model_id, score) pairs
    results = [(model_ids[i], float(scores[i])) for i in range(len(model_ids))]
    
    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Exclude query model itself and return top_k
    filtered_results = [(mid, score) for mid, score in results if mid != query_model_id][:top_k]
    
    return filtered_results


def dense_search(
    query_model_id: str,
    model_ids: List[str],
    emb_npz: str,
    faiss_index: str,
    top_k: int = 20
) -> List[Tuple[str, float]]:
    """
    Perform dense retrieval using FAISS.
    
    Args:
        query_model_id: Model ID to search for
        model_ids: List of all model IDs
        emb_npz: Path to embeddings NPZ file
        faiss_index: Path to FAISS index
        top_k: Number of results to return
    
    Returns:
        List of (model_id, score) tuples, sorted by score descending
    """
    # Load embeddings and IDs
    data = np.load(emb_npz)
    embs = data['embeddings']
    ids = data['ids'].tolist()
    
    # Find the index of the query model
    try:
        query_idx = ids.index(query_model_id)
    except ValueError:
        raise ValueError(f"Model ID '{query_model_id}' not found in embeddings")
    
    # Load FAISS index
    index = faiss.read_index(faiss_index)
    
    # Search
    query_emb = embs[query_idx:query_idx+1]
    D, I = index.search(query_emb, top_k + 1)
    
    # Get results (excluding self)
    results = []
    for i, score in zip(I[0], D[0]):
        if ids[i] != query_model_id:
            # Convert distance to similarity (higher is better)
            # FAISS returns L2 distance, so we use negative distance as score
            results.append((ids[i], float(-score)))
    
    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:top_k]


def reciprocal_rank_fusion(
    sparse_results: List[Tuple[str, float]],
    dense_results: List[Tuple[str, float]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Combine sparse and dense results using Reciprocal Rank Fusion (RRF).
    
    Args:
        sparse_results: List of (model_id, score) from sparse search
        dense_results: List of (model_id, score) from dense search
        k: RRF parameter (typically 60)
    
    Returns:
        List of (model_id, combined_score) tuples, sorted by score descending
    """
    # Build rank dictionaries
    sparse_ranks = {mid: rank + 1 for rank, (mid, _) in enumerate(sparse_results)}
    dense_ranks = {mid: rank + 1 for rank, (mid, _) in enumerate(dense_results)}
    
    # Get all unique model IDs
    all_ids = set(sparse_ranks.keys()) | set(dense_ranks.keys())
    
    # Calculate RRF scores
    rrf_scores = {}
    for model_id in all_ids:
        sparse_rank = sparse_ranks.get(model_id, float('inf'))
        dense_rank = dense_ranks.get(model_id, float('inf'))
        
        rrf_score = 1 / (k + sparse_rank) + 1 / (k + dense_rank)
        rrf_scores[model_id] = rrf_score
    
    # Sort by RRF score descending
    results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return results


def weighted_combination(
    sparse_results: List[Tuple[str, float]],
    dense_results: List[Tuple[str, float]],
    sparse_weight: float = 0.5,
    dense_weight: float = 0.5
) -> List[Tuple[str, float]]:
    """
    Combine sparse and dense results using weighted combination.
    
    Args:
        sparse_results: List of (model_id, score) from sparse search
        dense_results: List of (model_id, score) from dense search
        sparse_weight: Weight for sparse scores
        dense_weight: Weight for dense scores
    
    Returns:
        List of (model_id, combined_score) tuples, sorted by score descending
    """
    # Normalize scores to [0, 1] range
    def normalize_scores(results: List[Tuple[str, float]]) -> Dict[str, float]:
        if not results:
            return {}
        max_score = max(score for _, score in results)
        min_score = min(score for _, score in results)
        score_range = max_score - min_score if max_score != min_score else 1.0
        
        normalized = {}
        for mid, score in results:
            normalized[mid] = (score - min_score) / score_range
        return normalized
    
    sparse_norm = normalize_scores(sparse_results)
    dense_norm = normalize_scores(dense_results)
    
    # Get all unique model IDs
    all_ids = set(sparse_norm.keys()) | set(dense_norm.keys())
    
    # Calculate combined scores
    combined_scores = {}
    for model_id in all_ids:
        sparse_score = sparse_norm.get(model_id, 0.0)
        dense_score = dense_norm.get(model_id, 0.0)
        combined_score = sparse_weight * sparse_score + dense_weight * dense_score
        combined_scores[model_id] = combined_score
    
    # Sort by combined score descending
    results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    return results


def hybrid_search(
    query_model_id: str,
    model_ids: List[str],
    texts: List[str],
    bm25_index: BM25Okapi,
    emb_npz: str,
    faiss_index: str,
    top_k: int = 20,
    method: str = "rrf",
    sparse_weight: float = 0.5,
    dense_weight: float = 0.5
) -> List[Tuple[str, float]]:
    """
    Perform hybrid search combining sparse and dense retrieval.
    
    Args:
        query_model_id: Model ID to search for
        model_ids: List of all model IDs
        texts: List of all texts
        bm25_index: BM25 index
        emb_npz: Path to embeddings NPZ file
        faiss_index: Path to FAISS index
        top_k: Number of results to return
        method: Combination method - "rrf" or "weighted"
        sparse_weight: Weight for sparse scores (for weighted method)
        dense_weight: Weight for dense scores (for weighted method)
    
    Returns:
        List of (model_id, score) tuples, sorted by score descending
    """
    # Perform both searches
    print(f"  🔍 Performing sparse search...")
    sparse_results = sparse_search(query_model_id, model_ids, texts, bm25_index, top_k * 2)
    
    print(f"  🔍 Performing dense search...")
    dense_results = dense_search(query_model_id, model_ids, emb_npz, faiss_index, top_k * 2)
    
    # Combine results
    if method == "rrf":
        print(f"  🔀 Combining results using RRF...")
        combined_results = reciprocal_rank_fusion(sparse_results, dense_results)
    elif method == "weighted":
        print(f"  🔀 Combining results using weighted combination...")
        combined_results = weighted_combination(
            sparse_results, dense_results, sparse_weight, dense_weight
        )
    else:
        raise ValueError(f"Unknown combination method: {method}")
    
    return combined_results[:top_k]


def test_retrieval_modes(
    query_model_id: str = "Salesforce/codet5-base",
    jsonl_path: str = DEFAULT_JSONL,
    emb_npz: str = DEFAULT_EMB_NPZ,
    faiss_index: str = DEFAULT_FAISS_INDEX,
    top_k: int = 10
):
    """
    Test all retrieval modes: sparse, dense, and hybrid.
    """
    print("=" * 80)
    print("Testing Card2Card Retrieval Modes")
    print("=" * 80)
    print(f"Query Model: {query_model_id}")
    print(f"Top K: {top_k}")
    print()
    
    # Load corpus
    print("📚 Loading corpus...")
    model_ids, texts = load_corpus(jsonl_path)
    
    # Build BM25 index
    print("🔨 Building BM25 index...")
    bm25_index = build_bm25_index(texts)
    
    # Test 1: Sparse retrieval
    print("\n" + "=" * 80)
    print("Test 1: Sparse Retrieval (BM25)")
    print("=" * 80)
    sparse_results = sparse_search(query_model_id, model_ids, texts, bm25_index, top_k)
    print(f"\n✅ Found {len(sparse_results)} results:")
    for i, (mid, score) in enumerate(sparse_results[:5], 1):
        print(f"  {i}. {mid} (score: {score:.4f})")
    if len(sparse_results) > 5:
        print(f"  ... and {len(sparse_results) - 5} more")
    
    # Test 2: Dense retrieval
    print("\n" + "=" * 80)
    print("Test 2: Dense Retrieval (FAISS)")
    print("=" * 80)
    dense_results = dense_search(query_model_id, model_ids, emb_npz, faiss_index, top_k)
    print(f"\n✅ Found {len(dense_results)} results:")
    for i, (mid, score) in enumerate(dense_results[:5], 1):
        print(f"  {i}. {mid} (score: {score:.4f})")
    if len(dense_results) > 5:
        print(f"  ... and {len(dense_results) - 5} more")
    
    # Test 3: Hybrid retrieval (RRF)
    print("\n" + "=" * 80)
    print("Test 3: Hybrid Retrieval (RRF)")
    print("=" * 80)
    hybrid_rrf_results = hybrid_search(
        query_model_id, model_ids, texts, bm25_index,
        emb_npz, faiss_index, top_k, method="rrf"
    )
    print(f"\n✅ Found {len(hybrid_rrf_results)} results:")
    for i, (mid, score) in enumerate(hybrid_rrf_results[:5], 1):
        print(f"  {i}. {mid} (score: {score:.6f})")
    if len(hybrid_rrf_results) > 5:
        print(f"  ... and {len(hybrid_rrf_results) - 5} more")
    
    # Test 4: Hybrid retrieval (Weighted)
    print("\n" + "=" * 80)
    print("Test 4: Hybrid Retrieval (Weighted: 50% sparse, 50% dense)")
    print("=" * 80)
    hybrid_weighted_results = hybrid_search(
        query_model_id, model_ids, texts, bm25_index,
        emb_npz, faiss_index, top_k, method="weighted",
        sparse_weight=0.5, dense_weight=0.5
    )
    print(f"\n✅ Found {len(hybrid_weighted_results)} results:")
    for i, (mid, score) in enumerate(hybrid_weighted_results[:5], 1):
        print(f"  {i}. {mid} (score: {score:.6f})")
    if len(hybrid_weighted_results) > 5:
        print(f"  ... and {len(hybrid_weighted_results) - 5} more")
    
    # Compare results
    print("\n" + "=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    sparse_ids = {mid for mid, _ in sparse_results}
    dense_ids = {mid for mid, _ in dense_results}
    hybrid_rrf_ids = {mid for mid, _ in hybrid_rrf_results}
    hybrid_weighted_ids = {mid for mid, _ in hybrid_weighted_results}
    
    print(f"Sparse only: {len(sparse_ids)} unique models")
    print(f"Dense only: {len(dense_ids)} unique models")
    print(f"Hybrid (RRF): {len(hybrid_rrf_ids)} unique models")
    print(f"Hybrid (Weighted): {len(hybrid_weighted_ids)} unique models")
    print(f"\nOverlap (Sparse ∩ Dense): {len(sparse_ids & dense_ids)} models")
    print(f"Overlap (Sparse ∩ Hybrid RRF): {len(sparse_ids & hybrid_rrf_ids)} models")
    print(f"Overlap (Dense ∩ Hybrid RRF): {len(dense_ids & hybrid_rrf_ids)} models")
    
    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Card2Card hybrid retrieval")
    parser.add_argument('--model_id', default="Salesforce/codet5-base",
                       help='Model ID to search for')
    parser.add_argument('--jsonl', default=DEFAULT_JSONL,
                       help='Path to corpus JSONL file')
    parser.add_argument('--emb_npz', default=DEFAULT_EMB_NPZ,
                       help='Path to embeddings NPZ file')
    parser.add_argument('--faiss_index', default=DEFAULT_FAISS_INDEX,
                       help='Path to FAISS index file')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of results to return')
    
    args = parser.parse_args()
    
    try:
        test_retrieval_modes(
            query_model_id=args.model_id,
            jsonl_path=args.jsonl,
            emb_npz=args.emb_npz,
            faiss_index=args.faiss_index,
            top_k=args.top_k
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

