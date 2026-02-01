"""
ModelCard to ModelCard Search

This module provides functions for dense semantic search over model cards.
Supports sparse (BM25), dense (FAISS), and hybrid retrieval modes.
Reuses functionality from baseline1 and modelsearch modules.
"""

import os
import json
import sys
from typing import Dict, List, Optional, Tuple
import argparse
import pickle

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Add CitationLake to path for its utils
citationlake_path = os.path.join(os.path.dirname(__file__), '../../CitationLake')
if os.path.exists(citationlake_path) and citationlake_path not in sys.path:
    sys.path.insert(0, citationlake_path)

from src.baseline1.build_modelcard_jsonl import build_jsonl_from_raw, build_jsonl_from_parquet
from src.baseline1.table_retrieval_pipeline import (
    encode_corpus,
    build_faiss,
    search_neighbors
)

# Try to import CitationLake's load_combined_data
try:
    from src.utils import load_combined_data as citationlake_load_combined_data
    USE_CITATIONLAKE_UTILS = True
except ImportError:
    # Fallback to local version
    from src.utils import load_combined_data
    USE_CITATIONLAKE_UTILS = False
    citationlake_load_combined_data = None


def build_jsonl_from_citationlake_raw(raw_dir: str, field: str, output_jsonl: str) -> None:
    """
    Build JSONL corpus from CitationLake raw data using CitationLake's load_combined_data.
    Specifically uses the 'card' field.
    
    Args:
        raw_dir: Directory with raw parquet shards (should be data_citationlake/raw)
        field: Field to use (should be "card")
        output_jsonl: Output JSONL path
    """
    import json
    import os
    
    if field != "card":
        raise ValueError("build_jsonl_from_citationlake_raw only supports field='card'")
    
    if not USE_CITATIONLAKE_UTILS or citationlake_load_combined_data is None:
        raise ImportError("CitationLake utils not available. Please ensure CitationLake is accessible.")
    
    # Use CitationLake's load_combined_data to load the card field
    print(f"Loading modelcard data from {raw_dir} using CitationLake's load_combined_data...")
    df = citationlake_load_combined_data(
        data_type="modelcard",
        file_path=raw_dir,
        columns=["modelId", "card"]  # Only load needed columns
    )
    
    print(f"Loaded {len(df)} model cards")
    
    # Filter out rows with empty card
    df = df[df["card"].notna()].copy()
    df = df[df["card"].astype(str).str.strip() != ""].copy()
    
    os.makedirs(os.path.dirname(output_jsonl) if os.path.dirname(output_jsonl) else '.', exist_ok=True)
    
    written = 0
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            model_id = str(row["modelId"])
            card_text = str(row["card"]).strip()
            if not model_id or not card_text:
                continue
            doc = {"id": model_id, "contents": card_text}
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1
    
    print(f"Wrote {written} documents to {output_jsonl}")


def build_card_index(
    field: str = "card",
    raw_dir: str = "data_citationlake/raw",  # Default to CitationLake, fallback to data/raw
    parquet: Optional[str] = None,
    output_jsonl: str = "data/card2card_corpus.jsonl",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    output_npz: str = "data/card2card_embeddings.npz",
    output_index: str = "data/card2card.faiss",
    device: str = "cuda"
) -> None:
    """
    Build FAISS index for model card search.
    
    Args:
        field: Field to use ("card" or "card_readme")
        raw_dir: Directory with raw parquet shards (used when field="card")
                 Can be "data_citationlake/raw" or "data/raw"
        parquet: Path to processed parquet (used when field="card_readme")
                 Can be "data_citationlake/processed/modelcard_step1.parquet" or local path
        output_jsonl: Output JSONL corpus path
        model_name: Sentence transformer model name
        batch_size: Batch size for encoding
        output_npz: Output embeddings NPZ path
        output_index: Output FAISS index path
        device: Device to use ("cuda" or "cpu")
    """
    # Build JSONL corpus
    if field == "card_readme":
        if parquet is None:
            # Try CitationLake first, then fallback to local
            if os.path.exists("data_citationlake/processed/modelcard_step1.parquet"):
                parquet = "data_citationlake/processed/modelcard_step1.parquet"
            else:
                parquet = "data/processed/modelcard_step1.parquet"
        build_jsonl_from_parquet(parquet, field, output_jsonl)
    else:
        # For "card" field, use CitationLake's load_combined_data if available and raw_dir points to CitationLake
        if field == "card" and "data_citationlake" in raw_dir and USE_CITATIONLAKE_UTILS and citationlake_load_combined_data:
            print(f"Using CitationLake's load_combined_data to load card field from {raw_dir}")
            build_jsonl_from_citationlake_raw(raw_dir, field, output_jsonl)
        else:
            # Check if raw_dir exists, if not try alternative
            if not os.path.exists(raw_dir):
                if raw_dir == "data_citationlake/raw" and os.path.exists("data/raw"):
                    print(f"Warning: {raw_dir} not found, using data/raw instead")
                    raw_dir = "data/raw"
                elif raw_dir == "data/raw" and os.path.exists("data_citationlake/raw"):
                    print(f"Warning: {raw_dir} not found, using data_citationlake/raw instead")
                    raw_dir = "data_citationlake/raw"
            build_jsonl_from_raw(raw_dir, field, output_jsonl)
    
    # Encode corpus
    encode_corpus(output_jsonl, model_name, batch_size, output_npz, device)
    
    # Build FAISS index
    build_faiss(output_npz, output_index)
    
    print(f"✅ Card index built: {output_index}")


# Global cache for BM25 index and corpus
_bm25_cache: Optional[Tuple[object, List[str], List[str]]] = None
_bm25_cache_jsonl: Optional[str] = None


def _load_corpus_for_bm25(jsonl_path: str) -> Tuple[List[str], List[str]]:
    """
    Load corpus from JSONL file for BM25 indexing.
    
    Returns:
        (model_ids, texts): List of model IDs and corresponding texts
    """
    model_ids = []
    texts = []
    
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Corpus file not found: {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            model_ids.append(doc['id'])
            texts.append(doc['contents'])
    
    return model_ids, texts


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25. Split by whitespace and convert to lowercase."""
    return text.lower().split()


def _get_bm25_index(jsonl_path: str = "data/card2card_corpus.jsonl"):
    """
    Get or build BM25 index. Uses global cache to avoid rebuilding.
    
    Returns:
        (bm25_index, model_ids, texts): BM25 index and corpus data
    """
    global _bm25_cache, _bm25_cache_jsonl
    
    # Check if cache is valid
    if _bm25_cache is not None and _bm25_cache_jsonl == jsonl_path:
        return _bm25_cache
    
    # Import rank_bm25
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("rank-bm25 not installed. Please install it with: pip install rank-bm25")
    
    # Load corpus
    model_ids, texts = _load_corpus_for_bm25(jsonl_path)
    
    # Build BM25 index
    tokenized_texts = [_tokenize(text) for text in texts]
    bm25_index = BM25Okapi(tokenized_texts)
    
    # Cache the result
    _bm25_cache = (bm25_index, model_ids, texts)
    _bm25_cache_jsonl = jsonl_path
    
    return _bm25_cache


def _sparse_search_bm25(
    query_model_id: str,
    model_ids: List[str],
    texts: List[str],
    bm25_index,
    top_k: int = 20
) -> List[Tuple[str, float]]:
    """
    Perform sparse retrieval using BM25.
    
    Returns:
        List of (model_id, score) tuples, sorted by score descending
    """
    # Find query text
    try:
        query_idx = model_ids.index(query_model_id)
    except ValueError:
        raise ValueError(f"Model ID '{query_model_id}' not found in corpus")
    
    query_text = texts[query_idx]
    tokenized_query = _tokenize(query_text)
    
    # Get BM25 scores
    scores = bm25_index.get_scores(tokenized_query)
    
    # Create list of (model_id, score) pairs
    results = [(model_ids[i], float(scores[i])) for i in range(len(model_ids))]
    
    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Exclude query model itself and return top_k
    filtered_results = [(mid, score) for mid, score in results if mid != query_model_id][:top_k]
    
    return filtered_results


def _dense_search_faiss(
    query_model_id: str,
    model_ids: List[str],
    emb_npz: str,
    faiss_index: str,
    top_k: int = 20
) -> List[Tuple[str, float]]:
    """
    Perform dense retrieval using FAISS.
    
    Returns:
        List of (model_id, score) tuples, sorted by score descending
    """
    import numpy as np
    import faiss
    
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


def _reciprocal_rank_fusion(
    sparse_results: List[Tuple[str, float]],
    dense_results: List[Tuple[str, float]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Combine sparse and dense results using Reciprocal Rank Fusion (RRF).
    
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


def search_card2card(
    model_id: str,
    emb_npz: str = "data/card2card_embeddings.npz",
    faiss_index: str = "data/card2card.faiss",
    top_k: int = 20,
    output_json: Optional[str] = None,
    retrieval_mode: str = "dense",
    jsonl_path: str = "data/card2card_corpus.jsonl",
    hybrid_method: str = "rrf",
    sparse_weight: float = 0.5,
    dense_weight: float = 0.5
) -> List[str]:
    """
    Search for similar model cards given a model ID.
    
    Args:
        model_id: Hugging Face model ID to search for
        emb_npz: Path to embeddings NPZ file
        faiss_index: Path to FAISS index
        top_k: Number of neighbors to return
        output_json: Optional path to save results as JSON
        retrieval_mode: Retrieval mode - "sparse", "dense", or "hybrid"
        jsonl_path: Path to corpus JSONL file (required for sparse/hybrid)
        hybrid_method: Hybrid combination method - "rrf" or "weighted"
        sparse_weight: Weight for sparse scores (for weighted method)
        dense_weight: Weight for dense scores (for weighted method)
    
    Returns:
        List of similar model IDs
    """
    import numpy as np
    import faiss
    
    if retrieval_mode not in ["sparse", "dense", "hybrid"]:
        raise ValueError(f"Invalid retrieval_mode: {retrieval_mode}. Must be 'sparse', 'dense', or 'hybrid'")
    
    results = []
    
    if retrieval_mode == "dense":
        # Original dense retrieval
        data = np.load(emb_npz)
        ids = data['ids'].tolist()
        
        try:
            query_idx = ids.index(model_id)
        except ValueError:
            raise ValueError(f"Model ID '{model_id}' not found in corpus")
        
        index = faiss.read_index(faiss_index)
        embs = data['embeddings']
        query_emb = embs[query_idx:query_idx+1]
        D, I = index.search(query_emb, top_k + 1)
        
        neighbor_indices = [i for i in I[0] if i != query_idx][:top_k]
        results = [ids[i] for i in neighbor_indices]
        
    elif retrieval_mode == "sparse":
        # Sparse retrieval using BM25
        bm25_index, model_ids, texts = _get_bm25_index(jsonl_path)
        sparse_results = _sparse_search_bm25(model_id, model_ids, texts, bm25_index, top_k)
        results = [mid for mid, _ in sparse_results]
        
    elif retrieval_mode == "hybrid":
        # Hybrid retrieval: combine sparse and dense
        bm25_index, model_ids, texts = _get_bm25_index(jsonl_path)
        
        # Perform both searches
        sparse_results = _sparse_search_bm25(model_id, model_ids, texts, bm25_index, top_k * 2)
        
        # Load embeddings for dense search
        data = np.load(emb_npz)
        ids = data['ids'].tolist()
        try:
            query_idx = ids.index(model_id)
        except ValueError:
            raise ValueError(f"Model ID '{model_id}' not found in embeddings")
        
        index = faiss.read_index(faiss_index)
        embs = data['embeddings']
        query_emb = embs[query_idx:query_idx+1]
        D, I = index.search(query_emb, top_k * 2 + 1)
        
        dense_results = []
        for i, score in zip(I[0], D[0]):
            if ids[i] != model_id:
                dense_results.append((ids[i], float(-score)))
        dense_results.sort(key=lambda x: x[1], reverse=True)
        dense_results = dense_results[:top_k * 2]
        
        # Combine results
        if hybrid_method == "rrf":
            combined_results = _reciprocal_rank_fusion(sparse_results, dense_results)
        elif hybrid_method == "weighted":
            # Simple weighted combination (normalize scores first)
            def normalize_scores(score_list):
                if not score_list:
                    return {}
                max_score = max(score for _, score in score_list)
                min_score = min(score for _, score in score_list)
                score_range = max_score - min_score if max_score != min_score else 1.0
                return {mid: (score - min_score) / score_range for mid, score in score_list}
            
            sparse_norm = normalize_scores(sparse_results)
            dense_norm = normalize_scores(dense_results)
            all_ids = set(sparse_norm.keys()) | set(dense_norm.keys())
            
            combined_scores = {}
            for model_id in all_ids:
                sparse_score = sparse_norm.get(model_id, 0.0)
                dense_score = dense_norm.get(model_id, 0.0)
                combined_scores[model_id] = sparse_weight * sparse_score + dense_weight * dense_score
            
            combined_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            raise ValueError(f"Invalid hybrid_method: {hybrid_method}. Must be 'rrf' or 'weighted'")
        
        results = [mid for mid, _ in combined_results[:top_k]]
    
    # Save if requested
    if output_json:
        result = {
            "query": model_id,
            "neighbors": results,
            "retrieval_mode": retrieval_mode
        }
        if retrieval_mode == "hybrid":
            result["hybrid_method"] = hybrid_method
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved to {output_json}")
    
    return results


def search_card2card_batch(
    emb_npz: str = "data/card2card_embeddings.npz",
    faiss_index: str = "data/card2card.faiss",
    top_k: int = 20,
    output_json: str = "data/card2card_neighbors.json"
) -> Dict[str, List[str]]:
    """
    Search for similar model cards for all models in the corpus.
    
    Args:
        emb_npz: Path to embeddings NPZ file
        faiss_index: Path to FAISS index
        top_k: Number of neighbors to return per model
        output_json: Path to save results as JSON
    
    Returns:
        Dictionary mapping model_id to list of neighbor model_ids
    """
    import numpy as np
    import faiss
    from tqdm import tqdm
    
    # Load embeddings and IDs
    data = np.load(emb_npz)
    embs = data['embeddings']
    ids = data['ids'].tolist()
    
    # Load FAISS index
    index = faiss.read_index(faiss_index)
    
    # Search all
    D, I = index.search(embs, top_k + 1)
    
    # Build neighbor mapping
    neighbors = {}
    for i, neigh_indices in enumerate(tqdm(I, desc='Building neighbor mapping')):
        model_id = ids[i]
        # Exclude self
        nb = [ids[j] for j in neigh_indices if j != i][:top_k]
        neighbors[model_id] = nb
    
    # Save results
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(neighbors, f, ensure_ascii=False, indent=2)
    print(f"✅ Results saved to {output_json}")
    
    return neighbors


def main():
    """CLI entry point for card2card search"""
    parser = argparse.ArgumentParser(description="ModelCard to ModelCard Search")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Build index command
    build_parser = subparsers.add_parser('build-index', help='Build FAISS index')
    build_parser.add_argument('--field', choices=['card', 'card_readme'], default='card')
    build_parser.add_argument('--raw_dir', default='data_citationlake/raw',
                              help='Raw data directory. Can be data_citationlake/raw or data/raw')
    build_parser.add_argument('--parquet', default=None)
    build_parser.add_argument('--output_jsonl', default='data/card2card_corpus.jsonl')
    build_parser.add_argument('--model_name', default='all-MiniLM-L6-v2')
    build_parser.add_argument('--batch_size', type=int, default=256)
    build_parser.add_argument('--output_npz', default='data/card2card_embeddings.npz')
    build_parser.add_argument('--output_index', default='data/card2card.faiss')
    build_parser.add_argument('--device', default='cuda')
    
    # Search single command
    search_parser = subparsers.add_parser('search', help='Search for similar model cards')
    search_parser.add_argument('--model_id', required=True)
    search_parser.add_argument('--emb_npz', default='data/card2card_embeddings.npz')
    search_parser.add_argument('--faiss_index', default='data/card2card.faiss')
    search_parser.add_argument('--top_k', type=int, default=20)
    search_parser.add_argument('--output_json', default=None)
    search_parser.add_argument('--retrieval_mode', choices=['sparse', 'dense', 'hybrid'], default='dense',
                              help='Retrieval mode: sparse (BM25), dense (FAISS), or hybrid')
    search_parser.add_argument('--jsonl_path', default='data/card2card_corpus.jsonl',
                              help='Path to corpus JSONL file (required for sparse/hybrid)')
    search_parser.add_argument('--hybrid_method', choices=['rrf', 'weighted'], default='rrf',
                              help='Hybrid combination method: rrf or weighted')
    search_parser.add_argument('--sparse_weight', type=float, default=0.5,
                              help='Weight for sparse scores (for weighted method)')
    search_parser.add_argument('--dense_weight', type=float, default=0.5,
                              help='Weight for dense scores (for weighted method)')
    
    # Search batch command
    batch_parser = subparsers.add_parser('search-batch', help='Search for all models')
    batch_parser.add_argument('--emb_npz', default='data/card2card_embeddings.npz')
    batch_parser.add_argument('--faiss_index', default='data/card2card.faiss')
    batch_parser.add_argument('--top_k', type=int, default=20)
    batch_parser.add_argument('--output_json', default='data/card2card_neighbors.json')
    
    args = parser.parse_args()
    
    if args.command == 'build-index':
        build_card_index(
            field=args.field,
            raw_dir=args.raw_dir,
            parquet=args.parquet,
            output_jsonl=args.output_jsonl,
            model_name=args.model_name,
            batch_size=args.batch_size,
            output_npz=args.output_npz,
            output_index=args.output_index,
            device=args.device
        )
    elif args.command == 'search':
        neighbors = search_card2card(
            model_id=args.model_id,
            emb_npz=args.emb_npz,
            faiss_index=args.faiss_index,
            top_k=args.top_k,
            output_json=args.output_json,
            retrieval_mode=args.retrieval_mode,
            jsonl_path=args.jsonl_path,
            hybrid_method=args.hybrid_method,
            sparse_weight=args.sparse_weight,
            dense_weight=args.dense_weight
        )
        print(f"Found {len(neighbors)} neighbors for {args.model_id} (mode: {args.retrieval_mode})")
        for i, neighbor in enumerate(neighbors, 1):
            print(f"  {i}. {neighbor}")
    elif args.command == 'search-batch':
        neighbors = search_card2card_batch(
            emb_npz=args.emb_npz,
            faiss_index=args.faiss_index,
            top_k=args.top_k,
            output_json=args.output_json
        )
        print(f"✅ Generated neighbors for {len(neighbors)} models")
    else:
        parser.print_help()


def _test():
    """Quick test when run with no args."""
    emb_npz = "data/card2card_embeddings.npz"
    faiss_index = "data/card2card.faiss"
    if not os.path.isfile(emb_npz) or not os.path.isfile(faiss_index):
        print("Test skip: index missing (need card2card_embeddings.npz, card2card.faiss)")
        return
    print("Test card2card search (dense, top_k=5)...")
    r = search_card2card(model_id="Salesforce/codet5-base", emb_npz=emb_npz, faiss_index=faiss_index, top_k=5, retrieval_mode="dense")
    print("Neighbors:", r[:5] if r else "none")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        _test()
    else:
        main()

