"""Batch query embedding -> anchor table -> related model cards."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, List

import duckdb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import ENCODE_MODEL
from src.search.query2anc_table_embed_builder import default_table_embedding_npz
from src.search.query2anc_table_embed_inf import DenseNpzSearcher, run_one_query
from src.utils import _paths_for_resource_set, get_device


def _query_from_record(obj: dict[str, Any]) -> str:
    query = str(obj.get("query") or obj.get("rewritten_query") or "").strip()
    if query:
        return query
    response_text = obj.get("response_text")
    if isinstance(response_text, str) and response_text.strip():
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            return ""
        if isinstance(parsed, dict):
            return str(parsed.get("query") or "").strip()
    return ""


def load_queries(path: str) -> List[dict[str, str]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"queries file not found: {p}")

    rows: List[dict[str, str]] = []
    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                query = _query_from_record(obj)
                if not query:
                    continue
                qid = str(obj.get("id") or obj.get("custom_id") or f"q{i}").strip()
                rows.append({"id": qid, "query": query})
        return rows

    data = json.loads(p.read_text(encoding="utf-8"))
    items = data.get("queries", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"Invalid queries file format: {p}")
    for i, obj in enumerate(items):
        if isinstance(obj, str):
            query = obj.strip()
            qid = f"q{i}"
        elif isinstance(obj, dict):
            query = _query_from_record(obj)
            qid = str(obj.get("id") or obj.get("custom_id") or f"q{i}").strip()
        else:
            continue
        if query:
            rows.append({"id": qid, "query": query})
    return rows


def _safe_name(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text or "").strip())
    return s.strip("_") or "item"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch query embedding -> anchor table -> related model cards.")
    parser.add_argument("--queries_file", default="data_251117/query/query_rewrite_polished.jsonl")
    parser.add_argument("--resources", nargs="+", default=["hugging"], choices=["hugging", "github", "arxiv"])
    parser.add_argument(
        "--table_embeddings_npz",
        default="",
        help="Defaults to data_251117/query2anc_table_embeddings[_hugging].npz.",
    )
    parser.add_argument("--anchor_top_k", type=int, default=1)
    parser.add_argument(
        "--search_type",
        choices=["single_column", "multi_column", "keyword", "unionable"],
        default="keyword",
        help="Downstream tab2tab search type after anchor-table selection.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Set both table_top_k and model_top_k.")
    parser.add_argument("--table_top_k", type=int, default=0, help="Overrides --top_k for related table retrieval.")
    parser.add_argument("--model_top_k", type=int, default=0, help="Overrides --top_k for final model-card output.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--use_tab2tab_aug", action="store_true")
    parser.add_argument("--no_query_rerank", action="store_true")
    parser.add_argument(
        "--method_name",
        default="embedding_anchor",
        help="Method name used in card2tab2card_<method_name>.json for nugget eval.",
    )
    parser.add_argument("--output_json", default="tmp/card2tab2card_embedding_anchor_batch.json")
    parser.add_argument(
        "--jobs_output_dir",
        default="",
        help="Optional directory for nugget-eval compatible per-query job folders.",
    )
    parser.add_argument(
        "--jobs_json",
        default="",
        help="Optional jobs JSON path for src.batch_exp.run_batch_eval --jobs-json.",
    )
    args = parser.parse_args()

    resources = [str(r).strip().lower() for r in (args.resources or ["hugging"]) if str(r).strip()]
    method_name = _safe_name(args.method_name)
    table_top_k = int(args.table_top_k) if int(args.table_top_k) > 0 else int(args.top_k)
    model_top_k = int(args.model_top_k) if int(args.model_top_k) > 0 else int(args.top_k)
    queries = load_queries(args.queries_file)
    if args.limit > 0:
        queries = queries[: int(args.limit)]
    if not queries:
        raise RuntimeError(f"No queries loaded from {args.queries_file}")

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

    results: List[dict[str, Any]] = []
    jobs_items: List[dict[str, Any]] = []
    jobs_output_dir = Path(args.jobs_output_dir) if args.jobs_output_dir else None
    jobs_json_path = Path(args.jobs_json) if args.jobs_json else None
    if jobs_json_path and jobs_output_dir is None:
        jobs_output_dir = jobs_json_path.parent / "jobs"
    if jobs_output_dir is not None:
        jobs_output_dir.mkdir(parents=True, exist_ok=True)

    con_data = duckdb.connect(model_db_path, read_only=True)
    try:
        for item in tqdm(queries, desc="Query2AnchorTable batch", unit="query"):
            qid = item["id"]
            query = item["query"]
            try:
                result = run_one_query(
                    query=query,
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
                results.append({"id": qid, "query": query, "status": "success", "result": result})
                if jobs_output_dir is not None:
                    job_id = _safe_name(qid)
                    job_dir = jobs_output_dir / job_id
                    job_dir.mkdir(parents=True, exist_ok=True)
                    result_path = job_dir / f"card2tab2card_{method_name}.json"
                    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                    (job_dir / "job_meta.json").write_text(
                        json.dumps(
                            {
                                "job_id": job_id,
                                "query": query,
                                "source_id": qid,
                                "method": method_name,
                                "search_type": args.search_type,
                                "anchor_top_k": args.anchor_top_k,
                                "table_top_k": table_top_k,
                                "model_top_k": model_top_k,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    jobs_items.append(
                        {
                            "id": qid,
                            "job_id": job_id,
                            "query": query,
                            "search_response": {
                                "folder_path": str(job_dir),
                                "model_top_k": model_top_k,
                                "table_search_k": table_top_k,
                            },
                        }
                    )
            except Exception as exc:
                results.append({"id": qid, "query": query, "status": "error", "error": str(exc)})
    finally:
        con_data.close()

    output = {
        "queries_file": args.queries_file,
        "resources": resources,
        "anchor_top_k": args.anchor_top_k,
        "search_type": args.search_type,
        "table_top_k": table_top_k,
        "model_top_k": model_top_k,
        "results": results,
    }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))

    if jobs_json_path is not None:
        jobs_json_path.parent.mkdir(parents=True, exist_ok=True)
        jobs_json_path.write_text(json.dumps(jobs_items, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[jobs] wrote {len(jobs_items)} items: {jobs_json_path.resolve()}")


if __name__ == "__main__":
    main()
