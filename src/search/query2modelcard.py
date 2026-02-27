"""
Query to ModelCard Search

This module provides functions for searching model cards using a text query.
"""

import os
import json
import sys
import time
from typing import List, Optional
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def search_query2modelcard(
    query: str,
    emb_npz: str = "data/card2card_embeddings.npz",
    faiss_index: str = "data/card2card.faiss",
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 20,
    device: Optional[str] = None,
    output_json: Optional[str] = None
) -> List[str]:
    """
    Search for model cards using a text query.
    
    Args:
        query: Text query string
        emb_npz: Path to embeddings NPZ file (must match the index)
        faiss_index: Path to FAISS index
        model_name: Sentence transformer model name (must match the one used to build index)
        top_k: Number of results to return
        device: Device to use ("cuda" or "cpu")
        output_json: Optional path to save results as JSON
    
    Returns:
        List of model IDs matching the query
    """
    # Load embeddings and IDs
    data = np.load(emb_npz)
    ids = data['ids'].tolist()
    
    # Load FAISS index
    index = faiss.read_index(faiss_index)
    
    # Auto-detect device if not specified
    if device is None:
        device = _get_device()
    
    # Encode query
    model = SentenceTransformer(model_name, device=device)
    model.eval()
    query_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    query_emb = query_emb.astype('float32')
    faiss.normalize_L2(query_emb)
    
    # Search
    D, I = index.search(query_emb, top_k)
    
    # Get results
    results = [ids[i] for i in I[0]]
    
    # Save if requested
    if output_json:
        result = {
            "query": query,
            "results": results,
            "scores": D[0].tolist() if len(D) > 0 else []
        }
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved to {output_json}")
    
    return results


def main():
    """CLI entry point for query2modelcard search"""
    parser = argparse.ArgumentParser(description="Query to ModelCard Search")
    parser.add_argument('--query', required=True, help='Text query string')
    parser.add_argument('--emb_npz', default='data/card2card_embeddings.npz',
                       help='Path to embeddings NPZ file')
    parser.add_argument('--faiss_index', default='data/card2card.faiss',
                       help='Path to FAISS index')
    parser.add_argument('--model_name', default='all-MiniLM-L6-v2',
                       help='Sentence transformer model name')
    parser.add_argument('--top_k', type=int, default=20,
                       help='Number of results to return')
    parser.add_argument('--device', default=None,
                       help='Device to use (cuda or cpu). Auto-detects if not specified.')
    parser.add_argument('--output_json', default=None,
                       help='Optional path to save results as JSON')
    
    args = parser.parse_args()
    start_time = time.time()

    # Auto-detect device if not specified
    device = args.device if args.device else _get_device()
    
    results = search_query2modelcard(
        query=args.query,
        emb_npz=args.emb_npz,
        faiss_index=args.faiss_index,
        model_name=args.model_name,
        top_k=args.top_k,
        device=device,
        output_json=args.output_json
    )
    
    print(f"Found {len(results)} model cards for query: '{args.query}'")
    for i, model_id in enumerate(results, 1):
        print(f"  {i}. {model_id}")
    print(f"\nTotal time: {time.time() - start_time:.2f}s (device: {device})")

def _test():
    """Quick test when run with no args."""
    emb_npz = "data/card2card_embeddings.npz"
    faiss_index = "data/card2card.faiss"
    if not os.path.isfile(emb_npz) or not os.path.isfile(faiss_index):
        print("Test skip: index missing (need card2card_embeddings.npz, card2card.faiss)")
        return
    q = "For text-to-SQL, which models have the most complete and comparable benchmark results?"
    print("Test query2modelcard (top_k=1):", q[:60] + "...")
    r = search_query2modelcard(query=q, emb_npz=emb_npz, faiss_index=faiss_index, top_k=1, device=_get_device())
    print("Result:", r[0] if r else "none")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        _test()
    else:
        main()

