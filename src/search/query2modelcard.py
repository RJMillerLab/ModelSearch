"""
Query to ModelCard Search

This module provides functions for searching model cards using a text query.
"""

import os
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
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

from src.utils import get_device
from src.search.card2card import _build_faiss_index_in_memory, _get_pyserini_searcher_and_reader
#from src.config import EMB_NPZ, ENCODE_MODEL, SPARSE_INDEX
from src.config import *

_RUNTIME_LOCK = threading.Lock()
_ENCODER_BY_DEVICE: Dict[str, SentenceTransformer] = {}
_DENSE_RUNTIME_BY_NPZ: Dict[str, Dict[str, Any]] = {}


def _load_status(msg: str) -> None:
    """Progress lines for slow startup / first NPZ load. Disable with BACKEND_LOAD_QUIET=1."""
    if os.environ.get("BACKEND_LOAD_QUIET", "").strip().lower() in ("1", "true", "yes"):
        return
    print(msg, flush=True)


def _get_encoder_model(device: Optional[str] = None) -> SentenceTransformer:
    """Process-level cached encoder model by runtime device."""
    dev = str(device or get_device())
    with _RUNTIME_LOCK:
        model = _ENCODER_BY_DEVICE.get(dev)
        if model is None:
            _load_status(f"[load] Loading SentenceTransformer: {ENCODE_MODEL!r} (device={dev!r}) ...")
            t0 = time.time()
            model = SentenceTransformer(ENCODE_MODEL, device=dev)
            model.eval()
            _ENCODER_BY_DEVICE[dev] = model
            _load_status(f"[load]   SentenceTransformer ready in {time.time() - t0:.2f}s")
        return model


def _get_dense_runtime(emb_npz_path: str) -> Dict[str, Any]:
    """Process-level cached dense artifacts from npz: ids, embs, id_to_idx, faiss index."""
    key = os.path.abspath(str(emb_npz_path))
    with _RUNTIME_LOCK:
        cached = _DENSE_RUNTIME_BY_NPZ.get(key)
        if cached is not None:
            return cached
        _load_status(f"[load] Loading NPZ (this can take a while): {key}")
        t_npz = time.time()
        data = np.load(key, allow_pickle=True)
        ids = [str(x) for x in data["ids"].tolist()]
        embs = np.asarray(data["embeddings"], dtype=np.float32)
        _load_status(
            f"[load]   NPZ read done: {len(ids)} ids, embeddings {embs.shape}, "
            f"elapsed {time.time() - t_npz:.2f}s"
        )
        _load_status("[load] Building in-memory FAISS index ...")
        t_faiss = time.time()
        index, _ = _build_faiss_index_in_memory(embs)
        _load_status(f"[load]   FAISS index ready in {time.time() - t_faiss:.2f}s")
        id_to_idx = {mid: i for i, mid in enumerate(ids)}
        runtime = {
            "ids": ids,
            "embs": embs,
            "index": index,
            "id_to_idx": id_to_idx,
        }
        _DENSE_RUNTIME_BY_NPZ[key] = runtime
        return runtime


def get_query2modelcard_dense_runtime(emb_npz_path: str = EMB_NPZ) -> Dict[str, Any]:
    """
    Public helper for callers (e.g. backend/query2tab2card) to reuse loaded encoder + npz/index.
    """
    runtime = dict(_get_dense_runtime(emb_npz_path))
    runtime["encoder_model"] = _get_encoder_model()
    return runtime


def _emb_npz_path_for_resource_set(resources: List[str]) -> str:
    """Same path selection as demo backend / query2tab2card._resource_paths for dense embeddings."""
    rset = {str(r).strip().lower() for r in (resources or []) if str(r).strip()}
    if rset == {"hugging"}:
        return str(EMB_NPZ_HUGGING)
    if rset == {"hugging", "github", "arxiv"}:
        return str(EMB_NPZ)
    return str(EMB_NPZ_HUGGING)


def warmup_dense_runtimes_for_backend(
    *,
    model_resources_full: Optional[List[str]] = None,
    table_resources: Optional[List[str]] = None,
    log: Optional[Any] = None,
) -> None:
    """
    Load card2card NPZ + FAISS + SentenceTransformer into **process-level** caches.

    Call once when the API server starts so the **first** search job does not pay cold I/O.
    Subsequent jobs and ``get_query2modelcard_dense_runtime`` reuse the same caches.

    Defaults match ``backend._run_pipeline_body``: full multi-source npz + table npz from
    ``TABLE_RESOURCE_ALLOWLIST`` when ``table_resources`` is omitted.
    """
    def _emit(msg: str) -> None:
        if log is not None:
            log(msg)
        else:
            print(msg, flush=True)

    mr = list(model_resources_full or ["hugging", "github", "arxiv"])
    tr = list(table_resources or TABLE_RESOURCE_ALLOWLIST)
    _emit(
        "[warmup] Loading card2card artifacts into memory (NPZ → FAISS, then shared SentenceTransformer). "
        "First server start is slow; later jobs reuse cache."
    )
    jobs = [
        ("Query2ModelCard-FULL", _emb_npz_path_for_resource_set(mr)),
        ("Query2Tab2Card", _emb_npz_path_for_resource_set(tr)),
    ]
    seen: set[str] = set()
    for label, npz_path in jobs:
        if npz_path in seen:
            _emit(f"[warmup] {label}: same file as earlier step, already cached ({npz_path})")
            continue
        seen.add(npz_path)
        if not os.path.isfile(npz_path):
            _emit(f"[warmup] {label}: skip, file missing: {npz_path}")
            continue
        t0 = time.time()
        get_query2modelcard_dense_runtime(emb_npz_path=npz_path)
        _emit(f"[warmup] {label}: dense runtime ready in {time.time() - t0:.2f}s ({npz_path})")


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


def _require_dense_runtime(dense_runtime: Dict[str, Any]) -> Tuple[List[str], np.ndarray, Any, Dict[str, int], SentenceTransformer]:
    ids = [str(x) for x in list(dense_runtime.get("ids", []))]
    embs = np.asarray(dense_runtime.get("embs"), dtype=np.float32)
    index = dense_runtime.get("index")
    id_to_idx = dense_runtime.get("id_to_idx")
    encoder_model = dense_runtime.get("encoder_model")
    if not ids or embs.size == 0 or index is None or not isinstance(id_to_idx, dict) or encoder_model is None:
        raise ValueError(
            "dense_runtime must contain non-empty ids/embs/index/id_to_idx/encoder_model. "
            "Build it once via get_query2modelcard_dense_runtime(...) and pass it in."
        )
    return ids, embs, index, id_to_idx, encoder_model


def search_query2modelcard_dense(
    query: str,
    *,
    top_k: int = 20,
    dense_runtime: Dict[str, Any],
) -> Tuple[List[str], List[float]]:
    ids, _embs, index, _id_to_idx, encoder_model = _require_dense_runtime(dense_runtime)
    query_emb = encoder_model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(query_emb)
    D, I = index.search(query_emb, top_k)
    results = [ids[i] for i in I[0]]
    scores = D[0].tolist() if len(D) > 0 else []
    return results, scores


def search_query2modelcard_sparse(
    query: str,
    *,
    top_k: int = 20,
    sparse_index_path: str = SPARSE_INDEX,
) -> Tuple[List[str], List[float]]:
    return _search_sparse_query(query, top_k=top_k, sparse_index_path=sparse_index_path)


def search_query2modelcard_hybrid(
    query: str,
    *,
    top_k: int = 20,
    candidate_factor: int = 10,
    sparse_index_path: str = SPARSE_INDEX,
    dense_runtime: Dict[str, Any],
) -> Tuple[List[str], List[float]]:
    _ids, embs, _index, id_to_idx, encoder_model = _require_dense_runtime(dense_runtime)
    sparse_k = top_k * candidate_factor
    candidate_ids, _ = _search_sparse_query(query, top_k=sparse_k, sparse_index_path=sparse_index_path)

    seen = set()
    filtered_candidates: List[str] = []
    for cid in candidate_ids:
        if cid in id_to_idx and cid not in seen:
            filtered_candidates.append(cid)
            seen.add(cid)
    if not filtered_candidates:
        return [], []

    query_emb = encoder_model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(query_emb)
    candidate_indices = [id_to_idx[cid] for cid in filtered_candidates]
    candidate_embs = embs[candidate_indices]
    subset_index, _ = _build_faiss_index_in_memory(candidate_embs)
    subset_k = min(top_k, len(filtered_candidates))
    D, I = subset_index.search(query_emb, subset_k)
    results = [filtered_candidates[i] for i in I[0]]
    scores = D[0].tolist() if len(D) > 0 else []
    return results, scores


def search_query2modelcard(
    query: str,
    *,
    top_k: int = 20,
    output_json: Optional[str] = None,
    retrieval_mode: str = "dense",
    candidate_factor: int = 10,
    emb_npz_path: str = EMB_NPZ,
    sparse_index_path: str = SPARSE_INDEX,
    dense_runtime: Dict[str, Any],
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

    results: List[str]
    scores: List[float]
    if retrieval_mode == "sparse":
        results, scores = search_query2modelcard_sparse(
            query,
            top_k=top_k,
            sparse_index_path=sparse_index_path,
        )
    elif retrieval_mode == "dense":
        results, scores = search_query2modelcard_dense(
            query,
            top_k=top_k,
            dense_runtime=dense_runtime,
        )
    else:
        results, scores = search_query2modelcard_hybrid(
            query,
            top_k=top_k,
            candidate_factor=candidate_factor,
            sparse_index_path=sparse_index_path,
            dense_runtime=dense_runtime,
        )

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
    dense_runtime: Dict[str, Any],
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

    ids, embs, _index, id_to_idx, model = _require_dense_runtime(dense_runtime)
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

    dense_runtime: Dict[str, Any] = {}
    if args.retrieval_mode in {"dense", "hybrid"}:
        dense_runtime = get_query2modelcard_dense_runtime(emb_npz_path=emb_npz_path)
    results = search_query2modelcard(
        query=args.query,
        top_k=args.top_k,
        output_json=args.output_json,
        retrieval_mode=args.retrieval_mode,
        candidate_factor=args.candidate_factor,
        emb_npz_path=emb_npz_path,
        sparse_index_path=sparse_index_path,
        dense_runtime=dense_runtime,
    )
    
    print(f"Found {len(results)} model cards for query: '{args.query}'")
    for i, model_id in enumerate(results, 1):
        print(f"  {i}. {model_id}")
    print(f"\nTotal time: {time.time() - start_time:.2f}s (device: {get_device()})")

if __name__ == '__main__':
    main()

