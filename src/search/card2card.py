"""
ModelCard to ModelCard Search

This module provides functions for dense semantic search over model cards.
Reuses functionality from baseline1 and modelsearch modules.
"""

import os
import json
import sys
from typing import Dict, List, Optional
import argparse

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


def search_card2card(
    model_id: str,
    emb_npz: str = "data/card2card_embeddings.npz",
    faiss_index: str = "data/card2card.faiss",
    top_k: int = 20,
    output_json: Optional[str] = None
) -> List[str]:
    """
    Search for similar model cards given a model ID.
    
    Args:
        model_id: Hugging Face model ID to search for
        emb_npz: Path to embeddings NPZ file
        faiss_index: Path to FAISS index
        top_k: Number of neighbors to return
        output_json: Optional path to save results as JSON
    
    Returns:
        List of similar model IDs
    """
    import numpy as np
    import faiss
    from tqdm import tqdm
    
    # Load embeddings and IDs
    data = np.load(emb_npz)
    embs = data['embeddings']
    ids = data['ids'].tolist()
    
    # Find the index of the query model
    try:
        query_idx = ids.index(model_id)
    except ValueError:
        raise ValueError(f"Model ID '{model_id}' not found in corpus")
    
    # Load FAISS index
    index = faiss.read_index(faiss_index)
    
    # Search
    query_emb = embs[query_idx:query_idx+1]
    D, I = index.search(query_emb, top_k + 1)
    
    # Get neighbors (excluding self)
    neighbor_indices = [i for i in I[0] if i != query_idx][:top_k]
    neighbors = [ids[i] for i in neighbor_indices]
    
    # Save if requested
    if output_json:
        result = {
            "query": model_id,
            "neighbors": neighbors
        }
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved to {output_json}")
    
    return neighbors


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
            output_json=args.output_json
        )
        print(f"Found {len(neighbors)} neighbors for {args.model_id}")
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


if __name__ == '__main__':
    main()

