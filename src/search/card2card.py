"""
ModelCard to ModelCard Search

This module provides functions for dense semantic search over model cards.
Supports sparse (BM25), dense (FAISS), and hybrid retrieval modes.
Reuses functionality from baseline1 and modelsearch modules.
"""

import os
import re
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import faiss
import numpy as np
import duckdb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils import get_device
from src.config import EMB_NPZ, SPARSE_INDEX, CARD2CARD_NEIGHBORS_JSON, ENCODE_MODEL, MODELCARD_STEP1_PARQUET, CARD2CARD_SPARSE_CORPUS, CARD2CARD_CORPUS_JSONL

_SQL_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_sql_ident(name: str) -> str:
    if not _SQL_IDENT.match(name):
        raise ValueError(f"Invalid SQL column name: {name!r} (use letters, digits, underscore)")
    return f'"{name}"'

def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def build_card_index(batch_size: int = 256) -> None:
    """
    Stream rows from parquet via DuckDB SQL (read_parquet), encode in batches, save one .npz at the end.

    Optional corpus_jsonl: {"id": ..., "contents": ...} for build-index.
    """
    parquet_paths = [MODELCARD_STEP1_PARQUET]
    for p in parquet_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Parquet not found: {p}")

    _ensure_parent_dir(EMB_NPZ)
    id_q = _quote_sql_ident("modelId")
    text_q = _quote_sql_ident("card_readme")
    sql = f"""
        SELECT CAST({id_q} AS VARCHAR) AS id,
               CAST({text_q} AS VARCHAR) AS txt
        FROM read_parquet(?)
        WHERE {text_q} IS NOT NULL
          AND length(trim(cast({text_q} AS VARCHAR))) > 0
    """

    model = SentenceTransformer(ENCODE_MODEL, device=get_device())
    model.eval()

    all_embs: List[np.ndarray] = []
    ids: List[str] = []
    batch_ids: List[str] = []
    batch_texts: List[str] = []

    try:
        con = duckdb.connect(":memory:")
        result = con.execute(sql, [parquet_paths])
        reader = result.fetch_record_batch(4096)
        pbar = tqdm(unit="rows", desc="Encoding (SQL stream)")
        for record_batch in reader:
            col_id = record_batch.column(0)
            col_txt = record_batch.column(1)
            row_ids = col_id.to_pylist()
            row_txts = col_txt.to_pylist()
            for mid, txt in zip(row_ids, row_txts):
                if mid is None or txt is None:
                    continue
                s_id = str(mid).strip()
                s_txt = str(txt).strip()
                if not s_id or not s_txt:
                    continue
                batch_ids.append(s_id)
                batch_texts.append(s_txt)
                pbar.update(1)
                if len(batch_texts) >= batch_size:
                    try:
                        embs = model.encode(
                            batch_texts,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                            batch_size=len(batch_texts),
                        )
                        if embs is not None and getattr(embs, "size", 0) > 0:
                            all_embs.append(np.asarray(embs, dtype=np.float32))
                            ids.extend(batch_ids)
                    except Exception as e:
                        print(f"Error encoding batch (last id={batch_ids[-1]!r}): {e}")
                    finally:
                        batch_ids = []
                        batch_texts = []
        pbar.close()

        if batch_texts:
            try:
                embs = model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=len(batch_texts),
                )
                if embs is not None and getattr(embs, "size", 0) > 0:
                    all_embs.append(np.asarray(embs, dtype=np.float32))
                    ids.extend(batch_ids)
            except Exception as e:
                print(f"Error encoding final batch: {e}")
    finally:
        con.close()
    if not all_embs:
        print("No embeddings generated, skipping save.")
        return
    embs_array = np.vstack(all_embs).astype(np.float32, copy=False)
    np.savez_compressed(EMB_NPZ, embeddings=embs_array, ids=np.array(ids, dtype=object))
    print(f"Saved embeddings: {EMB_NPZ}, shape={embs_array.shape}, n_ids={len(ids)}")

def _build_faiss_index_in_memory(embs: np.ndarray) -> Tuple[faiss.Index, np.ndarray]:
    """
    Build an in-memory FAISS IndexFlatIP from embeddings.

    Returns:
        (index, normalized_embs) where normalized_embs is L2-normalized in-place copy.
    """
    embs_norm = np.ascontiguousarray(embs, dtype=np.float32).copy()
    faiss.normalize_L2(embs_norm)
    dim = embs_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs_norm)
    return index, embs_norm

def search_dense_neighbors_queries(
    *,
    query_model_ids: List[str],
    top_k: int = 20,
) -> Dict[str, List[str]]:
    """
    Dense query batch: build FAISS in-memory once, then search only provided query ids.
    """
    if not query_model_ids:
        return {}

    data = np.load(EMB_NPZ)
    embs = np.asarray(data["embeddings"], dtype=np.float32)
    ids = data["ids"].tolist()
    id_to_idx = {mid: i for i, mid in enumerate(ids)}

    query_indices: List[int] = []
    for mid in query_model_ids:
        if mid not in id_to_idx:
            raise ValueError(f"model_id not found in embeddings: {mid!r}")
        query_indices.append(id_to_idx[mid])

    print("Building FAISS index (in-memory)")
    t1 = time.time()
    index, embs_norm = _build_faiss_index_in_memory(embs)
    print(f"Time taken to build FAISS index: {time.time() - t1:.2f}s")

    query_embs = embs_norm[query_indices]
    D, I = index.search(query_embs, top_k + 1)

    neighbors: Dict[str, List[str]] = {}
    for q_pos, (q_mid, q_idx) in enumerate(zip(query_model_ids, query_indices)):
        neigh = [ids[j] for j in I[q_pos] if j != q_idx][:top_k] # exclude self
        neighbors[q_mid] = neigh
    return neighbors


def search_sparse_neighbors_queries(
    *,
    query_model_ids: List[str],
    top_k: int = 20,
) -> Dict[str, List[str]]:
    """
    Sparse query batch: load Lucene index once, then search only provided query ids.
    """
    if not query_model_ids:
        return {}
    if not os.path.isdir(SPARSE_INDEX):
        raise ValueError(
            f"Sparse mode requires --sparse_index_path to a Pyserini Lucene index directory. "
            f"Got: {SPARSE_INDEX!r}")
    searcher, index_reader = _get_pyserini_searcher_and_reader(SPARSE_INDEX)
    neighbors: Dict[str, List[str]] = {}
    for mid in tqdm(query_model_ids, desc="Sparse searching", unit="model"):
        sparse_results = _sparse_search_pyserini(mid, searcher, index_reader, top_k)
        neighbors[mid] = [x_mid for x_mid, _ in sparse_results if x_mid != mid][:top_k] # exclude self
    return neighbors

def search_hybrid_neighbors_queries(
    *,
    query_model_ids: List[str],
    top_k: int = 20,
    candidate_factor: int = 10,
) -> Dict[str, List[str]]:
    """
    Hybrid batch search (candidate re-ranking, no RRF):
      1) Sparse retrieval: top_k * candidate_factor candidates.
      2) Dense re-ranking: rebuild a FAISS subset index on those candidates,
         then score the query against the candidate subset.
      3) Return top_k by dense score over the candidate subset.
    """
    if not query_model_ids:
        return {}
    if not os.path.isdir(SPARSE_INDEX):
        raise ValueError(f"Hybrid mode requires --sparse_index_path to a Pyserini Lucene index directory. Got: {SPARSE_INDEX!r}")

    data = np.load(EMB_NPZ)
    embs = np.asarray(data["embeddings"], dtype=np.float32)
    ids = data["ids"].tolist()
    id_to_idx = {mid: i for i, mid in enumerate(ids)}

    # Normalize once (so we can do cosine similarity via inner product).
    embs_norm = np.array(embs, copy=True, dtype=np.float32)
    faiss.normalize_L2(embs_norm)

    searcher, index_reader = _get_pyserini_searcher_and_reader(SPARSE_INDEX)
    neighbors: Dict[str, List[str]] = {}
    sparse_k = top_k * candidate_factor

    for mid in tqdm(query_model_ids, desc="Hybrid searching", unit="model"):
        if mid not in id_to_idx:
            raise ValueError(f"model_id not found in embeddings: {mid!r}")

        q_idx = id_to_idx[mid]
        query_emb_norm = embs_norm[q_idx : q_idx + 1]

        # 1) Sparse candidates (already excludes self inside _sparse_search_pyserini).
        sparse_results = _sparse_search_pyserini(mid, searcher, index_reader, top_k=sparse_k)
        candidate_ids = [cand_id for cand_id, _ in sparse_results]

        # 2) Dense subset search.
        # Keep candidate order aligned with candidate_embs.
        candidate_ids_filtered: List[str] = [cid for cid in candidate_ids if cid != mid and cid in id_to_idx]
        if not candidate_ids_filtered:
            neighbors[mid] = [cand_id for cand_id, _ in sparse_results][:top_k]
            continue

        candidate_indices = [id_to_idx[cid] for cid in candidate_ids_filtered]
        candidate_embs_norm = embs_norm[candidate_indices]

        subset_k = min(sparse_k, len(candidate_ids_filtered))
        subset_index = faiss.IndexFlatIP(candidate_embs_norm.shape[1])
        subset_index.add(candidate_embs_norm)
        D, I = subset_index.search(query_emb_norm, subset_k)

        # candidate_ids_filtered does not include mid (self filtered), so self cannot appear here.
        # FAISS returns results sorted by score desc, so we can take the first top_k.
        dense_topk_ids = [candidate_ids_filtered[idx] for idx in I[0][:top_k]]
        # Safety: drop self if it somehow appears.
        dense_topk_ids = [cid for cid in dense_topk_ids if cid != mid]
        neighbors[mid] = dense_topk_ids
    return neighbors

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


def build_sparse_index(threads: int = 1) -> None:
    """
    Train: build Pyserini Lucene index from corpus JSONL (same as ModelTables baseline2).
    Writes corpus into corpus_dir and runs pyserini.index.lucene. Inference then uses output_index only.
    """
    import subprocess
    os.makedirs(CARD2CARD_SPARSE_CORPUS, exist_ok=True)
    corpus_jsonl = os.path.join(CARD2CARD_SPARSE_CORPUS, "corpus.jsonl")
    print(f"Building corpus JSONL from parquet via SQL: {corpus_jsonl}")
    parquet_paths = [MODELCARD_STEP1_PARQUET]
    for p in parquet_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Parquet not found: {p}")
    id_q = _quote_sql_ident("modelId")
    text_q = _quote_sql_ident("card_readme")
    sql = f"""
        SELECT CAST({id_q} AS VARCHAR) AS id,
               CAST({text_q} AS VARCHAR) AS txt
        FROM read_parquet(?)
        WHERE {text_q} IS NOT NULL
          AND length(trim(cast({text_q} AS VARCHAR))) > 0
    """
    with duckdb.connect(":memory:") as con:
        result = con.execute(sql, [parquet_paths])
        reader = result.fetch_record_batch(8192)
        written = 0
        with open(corpus_jsonl, "w", encoding="utf-8") as f:
            for record_batch in tqdm(reader, desc="Write corpus.jsonl (SQL stream)", unit="batch"):
                row_ids = record_batch.column(0).to_pylist()
                row_txts = record_batch.column(1).to_pylist()
                for mid, txt in zip(row_ids, row_txts):
                    if mid is None or txt is None:
                        continue
                    s_id = str(mid).strip()
                    s_txt = str(txt).strip()
                    if not s_id or not s_txt:
                        continue
                    f.write(json.dumps({"id": s_id, "contents": s_txt}, ensure_ascii=False) + "\n")
                    written += 1
    if written == 0:
        raise RuntimeError("No rows written to corpus.jsonl (check parquet path / column names / filters).")
    print(f"✅ Wrote {written} docs: {corpus_jsonl}")
    print("Building Lucene index (BM25, same as ModelTables baseline2)...")
    t0 = time.time()
    _ensure_parent_dir(SPARSE_INDEX)
    cmd = [sys.executable, "-m", "pyserini.index.lucene", "--collection", "JsonCollection", "--input", os.path.abspath(CARD2CARD_SPARSE_CORPUS), "--index", os.path.abspath(SPARSE_INDEX), "--generator", "DefaultLuceneDocumentGenerator", "--threads", str(threads), "--storePositions", "--storeDocvectors", "--storeRaw"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"pyserini.index.lucene failed: {out.stderr or out.stdout}")
    print(f"✅ Sparse index saved: {SPARSE_INDEX} (Total: {time.time() - t0:.2f}s)")


# Cache for Pyserini searcher and index reader (inference: load once per process)
_pyserini_cache: Optional[Tuple[object, object, str]] = None  # (searcher, index_reader, index_path)

def _get_pyserini_searcher_and_reader(index_path: str) -> Tuple[object, object]:
    """Inference: load Lucene searcher and index reader (cached by index_path)."""
    global _pyserini_cache
    if _pyserini_cache is not None and _pyserini_cache[2] == index_path:
        return _pyserini_cache[0], _pyserini_cache[1]
    from pyserini.search.lucene import LuceneSearcher
    from pyserini.index.lucene import LuceneIndexReader
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
            doc = json.loads(raw)
            return doc.get("contents", raw)
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


def _sparse_search_pyserini(query_model_id: str, searcher: object, index_reader: object, top_k: int = 20) -> List[Tuple[str, float]]:
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


def main():
    """CLI entry point for card2card search"""
    parser = argparse.ArgumentParser(description="ModelCard to ModelCard Search")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Build index command
    build_parser = subparsers.add_parser('build-dense-index', help='Build FAISS index')
    build_parser.add_argument('--batch_size', type=int, default=256)

    # Build sparse index (Part 1, train): Pyserini Lucene index (same as ModelTables baseline2)
    sparse_build_parser = subparsers.add_parser('build-sparse-index', help='Train: build Pyserini Lucene BM25 index from jsonl (Part 1)')
    sparse_build_parser.add_argument('--threads', type=int, default=1)
    
    # Search (inference)
    search_parser = subparsers.add_parser('search', help='Search for similar model cards')
    search_parser.add_argument('--model_ids_file', required=True, help='File with one model_id per line (one line also works).')
    search_parser.add_argument('--top_k', type=int, default=20)
    search_parser.add_argument('--retrieval_mode', choices=['sparse', 'dense', 'hybrid'], default='dense', help='Retrieval mode.')
    search_parser.add_argument('--output_json', default=None, help='Optional path to save results as JSON.')
    
    args = parser.parse_args()
    start_time = time.time()

    if args.command == 'build-dense-index':
        build_card_index(batch_size=args.batch_size)
        print(f"\nTotal time: {time.time() - start_time:.2f}s")
    elif args.command == 'build-sparse-index':
        build_sparse_index(threads=args.threads)
        print(f"\nTotal time: {time.time() - start_time:.2f}s")
    elif args.command == 'search':
        print(f"card2card search (batch queries) retrieval_mode={args.retrieval_mode}", flush=True)
        with open(args.model_ids_file, 'r', encoding='utf-8') as f:
            model_ids_list = [line.strip() for line in f if line.strip()]
        if not model_ids_list:
            raise RuntimeError(f"No model_ids found in file: {args.model_ids_file}")

        if args.retrieval_mode == 'dense':
            results = search_dense_neighbors_queries(query_model_ids=model_ids_list, top_k=args.top_k)
        elif args.retrieval_mode == 'sparse':
            results = search_sparse_neighbors_queries(query_model_ids=model_ids_list, top_k=args.top_k)
        else:
            results = search_hybrid_neighbors_queries(query_model_ids=model_ids_list, top_k=args.top_k)
        # save results to json
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nTotal time: {time.time() - start_time:.2f}s (device: {get_device()})")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

