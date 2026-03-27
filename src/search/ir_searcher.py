import json
import os
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pyserini.index.lucene import LuceneIndexReader
from pyserini.search.lucene import LuceneSearcher

from src.config import *
from src.utils import get_device

# Lucene BooleanQuery safety (same cap as card2card_batch)
_MAX_SPARSE_QUERY_TERMS = 1024


def _truncate_sparse_query(text: str, max_terms: int = _MAX_SPARSE_QUERY_TERMS) -> str:
    if not text:
        return ""
    terms = text.split()
    if len(terms) <= max_terms:
        return text
    return " ".join(terms[:max_terms])


def _lucene_doc_text(index_reader: object, model_id: str) -> str:
    """Raw document contents for BM25 query generation (must match indexed docid)."""
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
                return str(out)
    raise ValueError(
        f"Model ID {model_id!r} not found in sparse index (build with --storeRaw on corpus)"
    )


class DenseSearcher:
    # implemeted by building faiss index in memory
    def __init__(self, emb_npz_path: str):
        """Load all resources into memory"""
        self.emb_npz_path = os.path.abspath(emb_npz_path)
        self.encoder_model = SentenceTransformer(ENCODE_MODEL, device=get_device())
        self.encoder_model.eval()
        data = np.load(self.emb_npz_path, allow_pickle=True)
        self.ids = [str(x) for x in data["ids"].tolist()]
        self.embs = np.asarray(data["embeddings"], dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.embs.shape[1])
        faiss.normalize_L2(self.embs)
        self.index.add(self.embs)
        self.id_to_idx = {mid: i for i, mid in enumerate(self.ids)}

    def search(self, query: str, top_k: int = 20) -> Tuple[List[str], List[float]]:
        q = self.encoder_model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)

        D, I = self.index.search(q, top_k)
        results = [self.ids[i] for i in I[0]]
        return results, D[0].tolist()

    def search_by_model_id(self, model_id: str, top_k: int = 20) -> Tuple[List[str], List[float]]:
        """
        Neighbor search using the **stored embedding row** for ``model_id`` as the query vector
        (same semantics as ``search_dense_neighbors_queries`` / card2card dense retrieval).
        """
        mid = str(model_id).strip()
        if mid not in self.id_to_idx:
            raise ValueError(f"model_id not found in embeddings: {mid!r}")
        idx = self.id_to_idx[mid]
        q = self.embs[idx : idx + 1].copy()
        D, I = self.index.search(q, top_k + 1)
        out_ids: List[str] = []
        out_scores: List[float] = []
        for j, sc in zip(I[0].tolist(), D[0].tolist()):
            oid = self.ids[j]
            if oid == mid:
                continue
            out_ids.append(oid)
            out_scores.append(float(sc))
            if len(out_ids) >= top_k:
                break
        return out_ids, out_scores

    def search_subset(self, query, filtered_candidates, top_k):
        # compute scores equals to faiss.search(q, X)
        # attention: equal only when the index is flat ip index, not other index
        q = self.encoder_model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        pairs = [(cid, self.id_to_idx[cid]) for cid in filtered_candidates if cid in self.id_to_idx]
        if not pairs:
            return [], []
        cids, idxs = zip(*pairs)
        X = self.embs[list(idxs)].copy()
        faiss.normalize_L2(X)
        s = (X @ q.T).ravel()
        k = min(top_k, len(s))
        top = np.argpartition(-s, k-1)[:k]
        top = top[np.argsort(-s[top])]
        return [cids[i] for i in top], s[top].tolist()
        '''# build subset index, temporarily
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        k = min(top_k, len(X))
        D, I = index.search(q, k)
        return [cids[i] for i in I[0]], D[0].tolist()''' # another way, build a tmp small faiss index

class SparseSearcher:
    # implemeted by loading Lucene searcher and index reader
    def __init__(self, index_path: str):
        self.index_path = os.path.abspath(index_path)
        self.searcher = LuceneSearcher(self.index_path)
        self.searcher.set_bm25()
        self.reader = LuceneIndexReader(self.index_path)

    def search(self, query: str, top_k: int = 20) -> Tuple[List[str], List[float]]:
        hits = self.searcher.search(query, k=top_k)
        return [h.docid for h in hits], [float(h.score) for h in hits]

    def search_by_model_id(self, model_id: str, top_k: int = 20) -> Tuple[List[str], List[float]]:
        """
        Neighbor search using this doc's **indexed text** as the BM25 query
        (same semantics as ``search_sparse_neighbors_queries`` — **not** embedding-based).
        """
        mid = str(model_id).strip()
        qtext = _lucene_doc_text(self.reader, mid)
        qtext = _truncate_sparse_query(qtext)
        hits = self.searcher.search(qtext, k=top_k + 1)
        out_ids: List[str] = []
        out_scores: List[float] = []
        for h in hits:
            if h.docid == mid:
                continue
            out_ids.append(h.docid)
            out_scores.append(float(h.score))
            if len(out_ids) >= top_k:
                break
        return out_ids, out_scores
