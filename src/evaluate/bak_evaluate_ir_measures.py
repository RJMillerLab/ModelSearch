#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import ir_measures


def load_run(path: Path) -> List[Tuple[str, str, float]]:
    if path.suffix == ".run":
        run = []
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                qid, _q0, doc_id, _rank, score, _tag = line.split()
                run.append((qid, doc_id, float(score)))
        return run
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        run = []
        for qid, docs in payload.items():
            for doc_id, score in docs.items():
                run.append((str(qid), str(doc_id), float(score)))
        return run
    raise ValueError(f"Unsupported run format: {path}")


def load_subtopic_qrels(path: Path) -> List[Tuple[str, str, str, int]]:
    qrels = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            qid, subtopic, doc_id, rel = line.split()
            qrels.append((qid, subtopic, doc_id, int(rel)))
    return qrels


def collapse_qrels_for_ndcg(subtopic_qrels: Iterable[Tuple[str, str, str, int]]) -> List[ir_measures.Qrel]:
    rel_by_doc: Dict[Tuple[str, str], int] = defaultdict(int)
    for qid, _subtopic, doc_id, rel in subtopic_qrels:
        if rel > 0:
            rel_by_doc[(qid, doc_id)] += rel
    return [ir_measures.Qrel(qid, doc_id, relevance) for (qid, doc_id), relevance in rel_by_doc.items()]


def to_scored_docs(run: Iterable[Tuple[str, str, float]]) -> List[ir_measures.ScoredDoc]:
    return [ir_measures.ScoredDoc(qid, doc_id, score) for qid, doc_id, score in run]


def filter_query(items, query_id: str | None, key_idx: int = 0):
    if query_id is None:
        return list(items)
    return [item for item in items if item[key_idx] == query_id]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CoverageBench runs using ir-measures only.")
    parser.add_argument("--run", required=True, help="Path to .run or .json ranking file")
    parser.add_argument("--qrels", required=True, help="Path to subtopic qrels file")
    parser.add_argument("--query", help="Optional single query id to evaluate")
    parser.add_argument("--cutoff", type=int, default=20, help="Rank cutoff, default: 20")
    parser.add_argument("--per-query", action="store_true", help="Print per-query scores")
    parser.add_argument("--quiet-header", action="store_true", help="Suppress run/qrels/query header block")
    args = parser.parse_args()

    run_path = Path(args.run)
    qrels_path = Path(args.qrels)

    run = filter_query(load_run(run_path), args.query, key_idx=0)
    subtopic_qrels = filter_query(load_subtopic_qrels(qrels_path), args.query, key_idx=0)

    if not run:
        raise SystemExit("No run entries found for the requested input/query.")
    if not subtopic_qrels:
        raise SystemExit("No qrels found for the requested input/query.")

    scored_docs = to_scored_docs(run)
    ndcg_qrels = collapse_qrels_for_ndcg(subtopic_qrels)
    ndcg_measure = ir_measures.nDCG@args.cutoff

    ndcg_overall = ir_measures.calc_aggregate([ndcg_measure], ndcg_qrels, scored_docs)[ndcg_measure]

    if not args.quiet_header:
        print(f"run={run_path}")
        print(f"qrels={qrels_path}")
        if args.query:
            print(f"query={args.query}")
        print(f"cutoff={args.cutoff}")
        print()
    print("aggregate")
    print(f"  nDCG@{args.cutoff}: {ndcg_overall:.6f}")

    if args.per_query:
        print()
        print("per-query")
        for metric in ir_measures.iter_calc([ndcg_measure], ndcg_qrels, scored_docs):
            print(f"  {metric.query_id}: nDCG@{args.cutoff}={metric.value:.6f}")


if __name__ == "__main__":
    main()
