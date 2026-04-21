#!/usr/bin/env python3
"""Match query→header keywords against card2nugget CSV rows; write qrels + .run for pyndeval."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.config import OUTPUT_DIR
from src.evaluate.query2nugget_layer_mapping import NUGGET_SCHEMA_HEADERS

EVAL_DIR = Path(OUTPUT_DIR) / "evaluate"
DEFAULT_MAPPING = EVAL_DIR / "query_header_keyword_mapping.json"
DEFAULT_QRELS = EVAL_DIR / "real_subtopic.qrels"
DEFAULT_RUN = EVAL_DIR / "real_initial.run"
DEFAULT_DEBUG = EVAL_DIR / "query_csv_match_debug.json"


def _norm_cell(v: Any) -> str:
    if v is None:
        return ""
    text = str(v).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text)


def _norm_key(s: str) -> str:
    return (s or "").strip()


def load_query_mapping(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "queries" in data:
        inner = data["queries"]
        return list(inner) if isinstance(inner, list) else []
    if isinstance(data, list):
        return list(data)
    if isinstance(data, dict):
        return [data]
    return []


def discover_csv_paths(roots: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.csv")):
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            out.append(p)
    return out


def _row_dict(reader_fieldnames: list[str] | None, row: dict[str, Any]) -> dict[str, str]:
    """Map CSV row to NUGGET_SCHEMA_HEADERS keys (strip headers)."""
    raw = {_norm_key(k): _norm_cell(v) for k, v in row.items() if k is not None}
    out: dict[str, str] = {}
    for h in NUGGET_SCHEMA_HEADERS:
        v = raw.get(h) or raw.get(_norm_key(h))
        if v is None:
            for k, val in raw.items():
                if k.replace("_", "").lower() == h.replace("_", "").lower():
                    v = val
                    break
        out[h] = v or ""
    return out


def row_match_score(related: list[dict[str, Any]], cells: dict[str, str]) -> tuple[bool, int]:
    """
    AND across related headers: each must have nonempty cell; keywords match as
    case-insensitive substring in cell (any keyword per header). OR among keywords.
    """
    allowed = set(NUGGET_SCHEMA_HEADERS)
    items = [
        x
        for x in related
        if isinstance(x, dict) and str(x.get("header", "")).strip() in allowed
    ]
    if not items:
        return False, 0
    score = 0
    for item in items:
        h = str(item.get("header", "")).strip()
        kws = item.get("keywords")
        if isinstance(kws, str):
            kws = [kws]
        elif not isinstance(kws, list):
            kws = []
        keywords = [str(x).strip() for x in kws if str(x).strip()]
        cell = cells.get(h, "")
        if not cell:
            return False, 0
        cl = cell.lower()
        if not keywords:
            score += 1
            continue
        if not any(kw.lower() in cl for kw in keywords):
            return False, 0
        score += sum(1 for kw in keywords if kw.lower() in cl)
    return True, score


def build_qrels_and_run(
    queries: list[dict[str, Any]],
    csv_paths: list[Path],
    *,
    subtopic: str = "1",
) -> tuple[list[tuple[str, str, str, int]], list[tuple[str, str, str, int, float, str]], list[dict[str, Any]]]:
    qrels: list[tuple[str, str, str, int]] = []
    run_rows: list[tuple[str, str, str, int, float, str]] = []
    debug: list[dict[str, Any]] = []

    for qi, block in enumerate(queries):
        qid = f"q{qi:04d}"
        qtext = str(block.get("query", "")).strip()
        related = block.get("related")
        if not isinstance(related, list):
            related = []
        related = [x for x in related if isinstance(x, dict)]
        if block.get("error"):
            debug.append({"qid": qid, "query": qtext, "skipped": str(block.get("error")), "hits": []})
            continue
        if not related:
            debug.append({"qid": qid, "query": qtext, "skipped": "no_related_headers", "hits": []})
            continue

        hits: list[dict[str, Any]] = []
        csv_errors: list[str] = []
        for csv_path in csv_paths:
            table = csv_path.stem
            try:
                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames
                    for ri, row in enumerate(reader):
                        cells = _row_dict(fieldnames, row)
                        ok, sc = row_match_score(related, cells)
                        if not ok:
                            continue
                        doc_id = f"{table}#{ri}"
                        hits.append(
                            {
                                "doc_id": doc_id,
                                "table": table,
                                "row_idx": ri,
                                "score": sc,
                                "cells": {h: cells[h] for h in NUGGET_SCHEMA_HEADERS if cells.get(h)},
                            }
                        )
            except OSError as e:
                csv_errors.append(f"{csv_path}: {e}")

        hits.sort(key=lambda x: (-x["score"], x["doc_id"]))
        for h in hits:
            qrels.append((qid, subtopic, h["doc_id"], 1))

        for rank, h in enumerate(hits, start=1):
            score_f = float(max(1, h["score"]))
            run_rows.append((qid, "Q0", h["doc_id"], rank, score_f, "match"))

        entry: dict[str, Any] = {"qid": qid, "query": qtext, "hits": hits}
        if csv_errors:
            entry["csv_errors"] = csv_errors
        debug.append(entry)

    run_rows.sort(key=lambda x: (x[0], x[3]))
    return qrels, run_rows, debug


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match query_header_keyword_mapping.json to card2nugget CSVs; write qrels + TREC run.",
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING, help="query2nugget JSON output")
    parser.add_argument(
        "--csv-root",
        type=Path,
        action="append",
        default=None,
        help="Directory to scan for *.csv (repeatable). Default: evaluate/ and evaluate/batch/",
    )
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS, help="Output qrels path")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN, help="Output .run path")
    parser.add_argument("--debug-json", type=Path, default=DEFAULT_DEBUG, help="Per-query hit debug JSON")
    parser.add_argument("--subtopic", default="1", help="Subtopic id for all qrels rows (default: 1)")
    args = parser.parse_args()

    if not args.mapping.is_file():
        raise SystemExit(f"--mapping not found: {args.mapping}")

    roots = list(args.csv_root) if args.csv_root else [EVAL_DIR, EVAL_DIR / "batch"]
    csv_paths = discover_csv_paths(roots)
    if not csv_paths:
        raise SystemExit(f"No CSV files under {roots!r}")

    queries = load_query_mapping(args.mapping)
    if not queries:
        raise SystemExit("No queries in mapping JSON")

    qrels, run_rows, debug = build_qrels_and_run(queries, csv_paths, subtopic=str(args.subtopic))

    args.qrels.parent.mkdir(parents=True, exist_ok=True)
    args.run.parent.mkdir(parents=True, exist_ok=True)
    args.debug_json.parent.mkdir(parents=True, exist_ok=True)

    with open(args.qrels, "w", encoding="utf-8") as f:
        for qid, st, doc_id, rel in qrels:
            f.write(f"{qid} {st} {doc_id} {rel}\n")

    with open(args.run, "w", encoding="utf-8") as f:
        for qid, q0, doc_id, rank, score, tag in run_rows:
            f.write(f"{qid} {q0} {doc_id} {rank} {score} {tag}\n")

    with open(args.debug_json, "w", encoding="utf-8") as f:
        json.dump({"csv_files": [str(p) for p in csv_paths], "queries": debug}, f, ensure_ascii=False, indent=2)

    print(f"wrote qrels ({len(qrels)} lines) -> {args.qrels.resolve()}")
    print(f"wrote run ({len(run_rows)} lines) -> {args.run.resolve()}")
    print(f"wrote debug -> {args.debug_json.resolve()}")


if __name__ == "__main__":
    main()
