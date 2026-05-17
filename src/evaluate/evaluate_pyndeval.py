#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import pyndeval

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


def filter_query(items, query_id: str | None, key_idx: int = 0):
    if query_id is None:
        return list(items)
    return [item for item in items if item[key_idx] == query_id]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CoverageBench runs using pyndeval only.")
    parser.add_argument("--run", required=True, help="Path to .run or .json ranking file")
    parser.add_argument("--qrels", required=True, help="Path to subtopic qrels file")
    parser.add_argument("--query", help="Optional single query id to evaluate")
    parser.add_argument("--cutoff", type=int, default=20, help="Rank cutoff, default: 20")
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha for alpha-nDCG, default: 0.5")
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

    measures = [f"alpha-nDCG@{args.cutoff}", f"strec@{args.cutoff}"]
    by_query = pyndeval.ndeval(subtopic_qrels, run, measures=measures, alpha=args.alpha)
    alpha_key = f"alpha-nDCG@{args.cutoff}"
    strec_key = f"strec@{args.cutoff}"
    alpha_overall = mean(row[alpha_key] for row in by_query.values())
    strec_overall = mean(row[strec_key] for row in by_query.values())

    if not args.quiet_header:
        print(f"run={run_path}")
        print(f"qrels={qrels_path}")
        if args.query:
            print(f"query={args.query}")
        print(f"cutoff={args.cutoff}")
        print(f"alpha={args.alpha}")
        print()
    print("aggregate")
    print(f"  alpha-nDCG@{args.cutoff}: {alpha_overall:.6f}")
    print(f"  strec@{args.cutoff}: {strec_overall:.6f}")

    if args.per_query:
        print()
        print("per-query")
        for qid, row in sorted(by_query.items()):
            print(
                f"  {qid}: alpha-nDCG@{args.cutoff}={row[alpha_key]:.6f} "
                f"strec@{args.cutoff}={row[strec_key]:.6f}"
            )


if __name__ == "__main__":
    main()
