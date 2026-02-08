"""
ModelCard to ModelCard Search

This module provides functions for dense semantic search over model cards.
Supports sparse (BM25), dense (FAISS), and hybrid retrieval modes.
Reuses functionality from baseline1 and modelsearch modules.
"""

import os
import json
import sys
import time
from typing import Dict, List, Optional, Tuple
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

# Try to import CitationLake's load_combined_data and get_device
try:
    from src.utils import load_combined_data as citationlake_load_combined_data, get_device
    USE_CITATIONLAKE_UTILS = True
except ImportError:
    from src.utils import load_combined_data
    USE_CITATIONLAKE_UTILS = False
    citationlake_load_combined_data = None
    def get_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"


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


# Pyserini sparse: same logic as ModelTables baseline2 (Lucene BM25, top-k retrieval).
# Train = build Lucene index once. Inference = load index + search only.
MAX_QUERY_TERMS = 1024  # Lucene BooleanQuery maxClauseCount; long query truncation


def _truncate_query_pyserini(text: str, max_terms: int = MAX_QUERY_TERMS) -> str:
    """Truncate query to at most max_terms to avoid TooManyClauses in Lucene (same as ModelTables)."""
    if not text:
        return ""
    terms = text.split()
    if len(terms) <= max_terms:
        return text
    return " ".join(terms[:max_terms])


def build_sparse_index(
    jsonl_path: str = "data/card2card_corpus.jsonl",
    corpus_dir: str = "data/card2card_sparse_corpus",
    output_index: str = "data/card2card_sparse_index",
    threads: int = 1,
) -> None:
    """
    Train: build Pyserini Lucene index from corpus JSONL (same as ModelTables baseline2).
    Writes corpus into corpus_dir and runs pyserini.index.lucene. Inference then uses output_index only.
    """
    import subprocess
    import shutil
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Corpus file not found: {jsonl_path}")
    os.makedirs(corpus_dir, exist_ok=True)
    corpus_jsonl = os.path.join(corpus_dir, "corpus.jsonl")
    print(f"Copying corpus to {corpus_jsonl} for Pyserini JsonCollection...")
    shutil.copy2(jsonl_path, corpus_jsonl)
    print("Building Lucene index (BM25, same as ModelTables baseline2)...")
    t0 = time.time()
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", os.path.abspath(corpus_dir),
        "--index", os.path.abspath(output_index),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"pyserini.index.lucene failed: {out.stderr or out.stdout}")
    print(f"✅ Sparse index saved: {output_index} (Total: {time.time() - t0:.2f}s)")


# Cache for Pyserini searcher and index reader (inference: load once per process)
_pyserini_cache: Optional[Tuple[object, object, str]] = None  # (searcher, index_reader, index_path)


def _get_pyserini_searcher_and_reader(index_path: str) -> Tuple[object, object]:
    """Inference: load Lucene searcher and index reader (cached by index_path)."""
    global _pyserini_cache
    if _pyserini_cache is not None and _pyserini_cache[2] == index_path:
        return _pyserini_cache[0], _pyserini_cache[1]
    try:
        from pyserini.search.lucene import LuceneSearcher
        from pyserini.index.lucene import LuceneIndexReader
    except ImportError:
        raise ImportError("pyserini not installed. Install with: pip install pyserini")
    t0 = time.time()
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25()
    index_reader = LuceneIndexReader(index_path)
    print(f"  [timing] load sparse index (not inference): {time.time() - t0:.2f}s")
    _pyserini_cache = (searcher, index_reader, index_path)
    return searcher, index_reader


def _get_query_text_from_index(index_reader, model_id: str) -> str:
    """Get document contents for model_id from the index (for use as query text)."""
    if hasattr(index_reader, "doc_raw"):
        raw = index_reader.doc_raw(model_id)
        if raw:
            try:
                doc = json.loads(raw)
                return doc.get("contents", raw)
            except Exception:
                return raw
    if hasattr(index_reader, "doc_contents"):
        contents = index_reader.doc_contents(model_id)
        if contents:
            return contents
    if hasattr(index_reader, "doc"):
        doc = index_reader.doc(model_id)
        if doc is not None:
            c = getattr(doc, "contents", None)
            r = getattr(doc, "raw", None)
            out = (c() if callable(c) else c) or (r() if callable(r) else r)
            if out:
                return out
    raise ValueError(f"Model ID '{model_id}' not found in sparse index (build with --storeRaw)")


def _sparse_search_pyserini(
    query_model_id: str,
    searcher,
    index_reader,
    top_k: int = 20,
) -> List[Tuple[str, float]]:
    """
    Inference: get query text from index, run BM25 search (top-k only), return (docid, score) list.
    Same logic as ModelTables search_with_pyserini.py.
    """
    t0 = time.time()
    query_text = _get_query_text_from_index(index_reader, query_model_id)
    query_text = _truncate_query_pyserini(query_text)
    print(f"  [timing] sparse: get query text + truncate: {time.time() - t0:.2f}s")
    t0 = time.time()
    hits = searcher.search(query_text, k=top_k + 1)
    results = [(h.docid, float(h.score)) for h in hits if h.docid != query_model_id][:top_k]
    print(f"  [timing] sparse: BM25 search (top-k): {time.time() - t0:.2f}s")
    return results


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
    sparse_index_path: Optional[str] = None,
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
        sparse_index_path: Path to Pyserini Lucene index (Part 1 build-sparse-index); inference only loads it
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
    inference_sec = None  # time for scoring/retrieval only (excl. loading prebuilt index)

    if retrieval_mode == "dense":
        # Dense retrieval (FAISS only)
        print("  [timing] dense retrieval (per-step):")
        t0 = time.time()
        data = np.load(emb_npz)
        ids = data['ids'].tolist()
        print(f"  [timing] load npz (not inference): {time.time() - t0:.2f}s")
        try:
            query_idx = ids.index(model_id)
        except ValueError:
            raise ValueError(f"Model ID '{model_id}' not found in corpus")
        t0 = time.time()
        index = faiss.read_index(faiss_index)
        print(f"  [timing] load FAISS (not inference): {time.time() - t0:.2f}s")
        t0 = time.time()
        embs = data['embeddings']
        query_emb = embs[query_idx:query_idx+1]
        D, I = index.search(query_emb, top_k + 1)
        inference_sec = time.time() - t0
        print(f"  [timing] FAISS search: {time.time() - t0:.2f}s")
        neighbor_indices = [i for i in I[0] if i != query_idx][:top_k]
        results = [ids[i] for i in neighbor_indices]
        
    elif retrieval_mode == "sparse":
        if not sparse_index_path or not os.path.isdir(sparse_index_path):
            raise ValueError("Sparse mode requires --sparse_index_path to a Pyserini Lucene index (run build-sparse-index first)")
        print("  [timing] sparse retrieval (per-step):")
        t0 = time.time()
        searcher, index_reader = _get_pyserini_searcher_and_reader(sparse_index_path)
        print(f"  [timing] load sparse index total (load, not inference): {time.time() - t0:.2f}s")
        t_inf = time.time()
        sparse_results = _sparse_search_pyserini(model_id, searcher, index_reader, top_k)
        inference_sec = time.time() - t_inf
        results = [mid for mid, _ in sparse_results]
        
    elif retrieval_mode == "hybrid":
        if not sparse_index_path or not os.path.isdir(sparse_index_path):
            raise ValueError("Hybrid mode requires --sparse_index_path (run build-sparse-index first)")
        print("  [timing] hybrid retrieval (per-step):")
        t0 = time.time()
        searcher, index_reader = _get_pyserini_searcher_and_reader(sparse_index_path)
        print(f"  [timing] load sparse index total (load, not inference): {time.time() - t0:.2f}s")
        t_sparse_inf = time.time()
        sparse_results = _sparse_search_pyserini(model_id, searcher, index_reader, top_k * 2)
        sparse_inference = time.time() - t_sparse_inf
        print(f"  [timing] sparse branch total: {time.time() - t0:.2f}s")
        t0 = time.time()
        data = np.load(emb_npz)
        ids = data['ids'].tolist()
        print(f"  [timing] load npz (not inference): {time.time() - t0:.2f}s")
        try:
            query_idx = ids.index(model_id)
        except ValueError:
            raise ValueError(f"Model ID '{model_id}' not found in embeddings")
        t0 = time.time()
        index = faiss.read_index(faiss_index)
        print(f"  [timing] load FAISS (not inference): {time.time() - t0:.2f}s")
        t0 = time.time()
        embs = data['embeddings']
        query_emb = embs[query_idx:query_idx+1]
        D, I = index.search(query_emb, top_k * 2 + 1)
        print(f"  [timing] hybrid: FAISS search: {time.time() - t0:.2f}s")
        faiss_search_sec = time.time() - t0
        t0 = time.time()
        dense_results = []
        for i, score in zip(I[0], D[0]):
            if ids[i] != model_id:
                dense_results.append((ids[i], float(-score)))
        dense_results.sort(key=lambda x: x[1], reverse=True)
        dense_results = dense_results[:top_k * 2]
        print(f"  [timing] hybrid: dense build results + sort: {time.time() - t0:.2f}s")
        dense_build_sec = time.time() - t0
        t0 = time.time()
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
            for mid in all_ids:
                sparse_score = sparse_norm.get(mid, 0.0)
                dense_score = dense_norm.get(mid, 0.0)
                combined_scores[mid] = sparse_weight * sparse_score + dense_weight * dense_score
            
            combined_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            raise ValueError(f"Invalid hybrid_method: {hybrid_method}. Must be 'rrf' or 'weighted'")
        print(f"  [timing] hybrid: combine (RRF/weighted): {time.time() - t0:.2f}s")
        combine_sec = time.time() - t0
        inference_sec = sparse_inference + faiss_search_sec + dense_build_sec + combine_sec
        results = [mid for mid, _ in combined_results[:top_k]]
    
    if inference_sec is not None:
        print(f"  [timing] inference (min): {inference_sec:.2f}s")

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
    build_parser.add_argument('--device', default=None,
                              help='Device (cuda or cpu). Auto-detects if not set.')

    # Build sparse index (Part 1, train): Pyserini Lucene index (same as ModelTables baseline2)
    sparse_build_parser = subparsers.add_parser('build-sparse-index', help='Train: build Pyserini Lucene BM25 index from jsonl (Part 1)')
    sparse_build_parser.add_argument('--jsonl_path', default='data/card2card_corpus.jsonl', help='Corpus JSONL')
    sparse_build_parser.add_argument('--corpus_dir', default='data/card2card_sparse_corpus', help='Dir for Pyserini JsonCollection input')
    sparse_build_parser.add_argument('--output_index', default='data/card2card_sparse_index', help='Output Lucene index directory')
    sparse_build_parser.add_argument('--threads', type=int, default=1)
    
    # Search (inference)
    search_parser = subparsers.add_parser('search', help='Search for similar model cards')
    search_parser.add_argument('--model_id', help='Single query model ID')
    search_parser.add_argument('--model_ids_file', help='File with one model_id per line: load index once, then inference per query')
    search_parser.add_argument('--emb_npz', default='data/card2card_embeddings.npz')
    search_parser.add_argument('--faiss_index', default='data/card2card.faiss')
    search_parser.add_argument('--top_k', type=int, default=20)
    search_parser.add_argument('--output_json', default=None)
    search_parser.add_argument('--retrieval_mode', choices=['sparse', 'dense', 'hybrid'], default='dense',
                              help='Retrieval mode: sparse (Pyserini BM25), dense (FAISS), or hybrid')
    search_parser.add_argument('--sparse_index_path', default='data/card2card_sparse_index',
                              help='Pyserini Lucene index dir from Part 1; inference only loads (no rebuild)')
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
    start_time = time.time()
    device = getattr(args, 'device', None) or get_device()

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
            device=device
        )
        print(f"\nTotal time: {time.time() - start_time:.2f}s (device: {device})")
    elif args.command == 'build-sparse-index':
        build_sparse_index(
            jsonl_path=args.jsonl_path,
            corpus_dir=args.corpus_dir,
            output_index=args.output_index,
            threads=getattr(args, 'threads', 1),
        )
        print(f"\nTotal time: {time.time() - start_time:.2f}s")
    elif args.command == 'search':
        model_ids_file = getattr(args, 'model_ids_file', None)
        if model_ids_file:
            with open(model_ids_file, 'r', encoding='utf-8') as f:
                model_ids_list = [line.strip() for line in f if line.strip()]
            if not model_ids_list:
                print("No model_ids in file")
            else:
                # Load once (first query), then inference (min) per query
                for i, mid in enumerate(model_ids_list):
                    if i > 0:
                        print(f"\n--- Query {i + 1} (inference only, index already loaded) ---")
                    neighbors = search_card2card(
                        model_id=mid,
                        emb_npz=args.emb_npz,
                        faiss_index=args.faiss_index,
                        top_k=args.top_k,
                        output_json=args.output_json if i == 0 else None,
                        retrieval_mode=args.retrieval_mode,
                        sparse_index_path=getattr(args, 'sparse_index_path', None),
                        hybrid_method=args.hybrid_method,
                        sparse_weight=args.sparse_weight,
                        dense_weight=args.dense_weight
                    )
                    print(f"Found {len(neighbors)} neighbors for {mid}")
                print(f"\nTotal time: {time.time() - start_time:.2f}s (device: {device})")
        else:
            if not args.model_id:
                parser.error("--model_id or --model_ids_file required")
            neighbors = search_card2card(
                model_id=args.model_id,
                emb_npz=args.emb_npz,
                faiss_index=args.faiss_index,
                top_k=args.top_k,
                output_json=args.output_json,
                retrieval_mode=args.retrieval_mode,
                sparse_index_path=getattr(args, 'sparse_index_path', None),
                hybrid_method=args.hybrid_method,
                sparse_weight=args.sparse_weight,
                dense_weight=args.dense_weight
            )
            print(f"Found {len(neighbors)} neighbors for {args.model_id} (mode: {args.retrieval_mode})")
            for i, neighbor in enumerate(neighbors, 1):
                print(f"  {i}. {neighbor}")
            print(f"\nTotal time: {time.time() - start_time:.2f}s (device: {device})")
    elif args.command == 'search-batch':
        neighbors = search_card2card_batch(
            emb_npz=args.emb_npz,
            faiss_index=args.faiss_index,
            top_k=args.top_k,
            output_json=args.output_json
        )
        print(f"✅ Generated neighbors for {len(neighbors)} models")
        print(f"\nTotal time: {time.time() - start_time:.2f}s (device: {device})")
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

