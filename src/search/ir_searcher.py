import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.utils import get_device
from src.config import *
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader

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
    def search(self, query: str, top_k: int = 20):
        q = self.encoder_model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)

        D, I = self.index.search(q, top_k)
        results = [self.ids[i] for i in I[0]]
        return results, D[0].tolist()
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
    def search(self, query: str, top_k: int = 20):
        hits = self.searcher.search(query, k=top_k)
        return [h.docid for h in hits], [float(h.score) for h in hits]
