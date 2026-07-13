"""Query embedding -> anchor table -> related model cards."""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import duckdb
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import ENCODE_MODEL
from src.search.card2tab2card import Card2Tab2CardSearch
from src.search.query2anc_table_embed_builder import default_table_embedding_npz
from src.utils import _paths_for_resource_set, get_device, load_csvs_to_modelids


class DenseNpzSearcher:
    """Small dense searcher for .npz embeddings, avoiding Sparse/Pyserini imports."""

    def __init__(self, emb_npz_path: str, encoder_model: SentenceTransformer | None = None):
        self.emb_npz_path = os.path.abspath(emb_npz_path)
        self.encoder_model = encoder_model or SentenceTransformer(ENCODE_MODEL, device=get_device())
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
        k = min(max(0, int(top_k)), len(self.ids))
        if k == 0:
            return [], []
        scores, indices = self.index.search(q, k)
        pairs = [(int(i), float(score)) for i, score in zip(indices[0], scores[0]) if int(i) >= 0]
        return [self.ids[i] for i, _score in pairs], [score for _i, score in pairs]

    def search_subset(self, query: str, filtered_candidates: List[str], top_k: int) -> Tuple[List[str], List[float]]:
        q = self.encoder_model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        pairs = [(cid, self.id_to_idx[cid]) for cid in filtered_candidates if cid in self.id_to_idx]
        if not pairs:
            return [], []
        cids, idxs = zip(*pairs)
        x = self.embs[list(idxs)].copy()
        faiss.normalize_L2(x)
        scores = (x @ q.T).ravel()
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [cids[i] for i in top], scores[top].tolist()


def _run_downstream(
    *,
    query: str,
    anchor_tables: List[str],
    anchor_scores: List[float],
    con_data: duckdb.DuckDBPyConnection,
    model_dense: DenseNpzSearcher | None,
    resources: List[str],
    search_type: str,
    table_top_k: int,
    model_top_k: int,
    use_tab2tab_aug: bool,
    apply_query_rerank: bool,
) -> dict[str, object]:
    search = Card2Tab2CardSearch()
    anchors = [os.path.basename(str(table).strip()) for table in anchor_tables if str(table).strip()]
    anchor_to_models = load_csvs_to_modelids(anchors)
    anchor_model_ids = list(dict.fromkeys(sum(anchor_to_models.values(), [])))

    search.card2tab_map = {"__anchor_tables__": anchors}
    search.tab2tab(anchors, con_data, search_type, table_top_k, resources, use_tab2tab_aug)
    retrieved_tables = list(dict.fromkeys(sum(search.tab2tab_map.values(), [])))
    search.tab2card(retrieved_tables, anchor_model_ids)

    candidate_pool = list(dict.fromkeys(sum(search.tab2card_map.values(), [])))
    if apply_query_rerank and candidate_pool:
        if model_dense is None:
            raise RuntimeError("query rerank requested, but model DenseSearcher is unavailable")
        ranked, _scores = model_dense.search_subset(query, candidate_pool, max(1, len(candidate_pool)))
        model_rerank_map = list(ranked)
        if model_top_k > 0:
            model_rerank_map = model_rerank_map[: int(model_top_k)]
    else:
        model_rerank_map = candidate_pool
        if model_top_k > 0:
            model_rerank_map = model_rerank_map[: int(model_top_k)]

    return {
        "query2card_map": {query: anchor_model_ids},
        "query2anchor_map": {query: anchors},
        "anchor_scores_map": {query: anchor_scores[: len(anchors)]},
        "card2tab_map": search.card2tab_map,
        "tab2tab_map": search.tab2tab_map,
        "tab2card_map": search.tab2card_map,
        "model_rerank_map": model_rerank_map,
    }


def run_one_query(
    *,
    query: str,
    table_dense: DenseNpzSearcher,
    con_data: duckdb.DuckDBPyConnection,
    model_dense: DenseNpzSearcher | None,
    resources: List[str],
    anchor_top_k: int,
    search_type: str,
    table_top_k: int,
    model_top_k: int,
    use_tab2tab_aug: bool,
    apply_query_rerank: bool,
) -> dict[str, object]:
    anchors, scores = table_dense.search(query, top_k=anchor_top_k)
    return _run_downstream(
        query=query,
        anchor_tables=list(anchors),
        anchor_scores=[float(score) for score in scores],
        con_data=con_data,
        model_dense=model_dense,
        resources=resources,
        search_type=search_type,
        table_top_k=table_top_k,
        model_top_k=model_top_k,
        use_tab2tab_aug=use_tab2tab_aug,
        apply_query_rerank=apply_query_rerank,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Query embedding -> anchor table -> related model cards.")
    parser.add_argument("--query", required=True, help="Natural-language query.")
    parser.add_argument("--resources", nargs="+", default=["hugging"], choices=["hugging", "github", "arxiv"])
    parser.add_argument(
        "--table_embeddings_npz",
        default="",
        help="Defaults to data_251117/query2anc_table_embeddings[_hugging].npz.",
    )
    parser.add_argument("--anchor_top_k", type=int, default=1, help="Top-k anchor tables from query-table embedding search.")
    parser.add_argument(
        "--search_type",
        choices=["single_column", "multi_column", "keyword", "unionable"],
        default="keyword",
        help="Downstream tab2tab search type after anchor-table selection.",
    )
    parser.add_argument("--top_k", type=int, default=0, help="Set both --table_top_k and --model_top_k.")
    parser.add_argument("--table_top_k", type=int, default=10, help="Top-k related tables per anchor table.")
    parser.add_argument("--model_top_k", type=int, default=10, help="Final number of model cards after query rerank.")
    parser.add_argument("--use_tab2tab_aug", action="store_true", help="Use tab2tab augmentation.")
    parser.add_argument("--no_query_rerank", action="store_true", help="Skip final query-to-model dense rerank.")
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    resources = [str(r).strip().lower() for r in (args.resources or ["hugging"]) if str(r).strip()]
    table_top_k = int(args.top_k) if int(args.top_k) > 0 else int(args.table_top_k)
    model_top_k = int(args.top_k) if int(args.top_k) > 0 else int(args.model_top_k)
    table_npz = args.table_embeddings_npz or default_table_embedding_npz(resources)
    if not os.path.isfile(table_npz):
        raise FileNotFoundError(
            f"Missing table embedding npz: {table_npz}. "
            "Build it with: python -m src.search.query2anc_table_embed_pre"
        )

    encoder_model = SentenceTransformer(ENCODE_MODEL, device=get_device())
    encoder_model.eval()
    table_dense = DenseNpzSearcher(emb_npz_path=table_npz, encoder_model=encoder_model)
    emb_npz_path, _sparse_index_path, model_db_path = _paths_for_resource_set(resources)

    model_dense = None
    if not args.no_query_rerank:
        model_dense = DenseNpzSearcher(emb_npz_path=emb_npz_path, encoder_model=encoder_model)

    con_data = duckdb.connect(model_db_path, read_only=True)
    try:
        result = run_one_query(
            query=args.query,
            table_dense=table_dense,
            con_data=con_data,
            model_dense=model_dense,
            resources=resources,
            anchor_top_k=args.anchor_top_k,
            search_type=args.search_type,
            table_top_k=table_top_k,
            model_top_k=model_top_k,
            use_tab2tab_aug=args.use_tab2tab_aug,
            apply_query_rerank=not args.no_query_rerank,
        )
    finally:
        con_data.close()

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
