#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import duckdb
from bs4 import BeautifulSoup

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.config import CARD_CONTENT_RAW, OUTPUT_DIR
from src.utils import load_modelid_to_csvlist, read_csv_robust, resolve_table_path


SHARD_GLOB = "train-0000*-of-00006.parquet"
SINGLE_OUTPUT_JSON = os.path.join(OUTPUT_DIR, "evaluate", "single_modelcard_nuggets.json")


def _norm(v: Any) -> str:
    if v is None:
        return ""
    text = str(v).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text)


def _read_card_by_model_id(model_id: str) -> str:
    model_id = _norm(model_id)
    if not model_id:
        raise ValueError("model_id is required")

    if not Path(CARD_CONTENT_RAW).exists():
        raise FileNotFoundError(f"CARD_CONTENT_RAW does not exist: {CARD_CONTENT_RAW}")

    parquet_glob = str(Path(CARD_CONTENT_RAW) / SHARD_GLOB)
    query = """
        SELECT card
        FROM read_parquet(?, union_by_name=true)
        WHERE CAST(modelId AS VARCHAR) = ?
        LIMIT 1
    """
    with duckdb.connect(":memory:") as con:
        row = con.execute(query, [parquet_glob, model_id]).fetchone()
    if not row or row[0] is None:
        raise FileNotFoundError(f"No card found for modelId={model_id!r}")
    return str(row[0])


def _remove_markdown_tables(text: str) -> str:
    if not isinstance(text, str):
        return ""

    content_without_html = _remove_html_tables(text)
    lines = content_without_html.splitlines()
    kept: list[str] = []
    in_table = False

    def is_separator(line: str) -> bool:
        return bool(re.fullmatch(r"[\s:\-\|]*", line.strip())) and "|" in line

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if in_table:
                in_table = False
            else:
                kept.append(raw_line)
            continue

        if "|" in line:
            in_table = True
            continue

        if in_table and is_separator(line):
            continue

        if in_table:
            in_table = False

        kept.append(raw_line)

    return "\n".join(kept)


def _remove_html_tables(text: str) -> str:
    if not isinstance(text, str) or "<table" not in text.lower():
        return text if isinstance(text, str) else ""

    try:
        soup = BeautifulSoup(text, "lxml")
    except Exception:
        soup = BeautifulSoup(text, "html.parser")
    for table in soup.find_all("table"):
        table.decompose()
    return str(soup)


def _remove_citation_part(text: str) -> str:
    if not isinstance(text, str):
        return ""

    cleaned = re.sub(r"(?is)```(?:bibtex|latex)?\s*.*?```", "", text)
    lines = cleaned.splitlines()
    kept: list[str] = []
    skipping_bibtex = False

    for raw_line in lines:
        line = raw_line.strip()
        if re.match(r"(?i)^\s*(#{1,6}\s*)?(citation|cite|references|reference|bibliography)\b", line):
            break
        if re.match(r"(?i)^@(article|inproceedings|misc|techreport|phdthesis|mastersthesis|book|incollection|conference|software)\b", line):
            skipping_bibtex = True
            continue
        if skipping_bibtex:
            if not line:
                skipping_bibtex = False
            continue
        kept.append(raw_line)

    return "\n".join(kept)


def _table_to_nuggets(csv_path: str) -> list[dict[str, Any]]:
    resolved = resolve_table_path(csv_path) or (csv_path if os.path.exists(csv_path) else "")
    if not resolved:
        return []

    df = read_csv_robust(resolved)
    if df is None or df.empty:
        return []

    csv_base = os.path.basename(resolved)
    headers = [str(c) for c in df.columns.tolist()]
    if not headers:
        return []
    row_key_col = df.columns[0]
    nuggets: list[dict[str, Any]] = []
    for row_idx, (_, row) in enumerate(df.iterrows(), start=0):
        row_label = _norm(row.get(row_key_col))
        if not row_label:
            row_label = f"row_{row_idx}"
        for col in headers[1:]:
            value = row.get(col)
            text = _norm(value)
            if not text:
                continue
            nuggets.append(
                {
                    "nugget_id": f"nugget_table::{csv_base}::{row_idx:06d}::{col}",
                    "csv_basename": csv_base,
                    "row": row_label,
                    "col": col,
                    "value": text,
                }
            )
    return nuggets


def extract_nuggets_from_card(model_id: str) -> list[dict[str, Any]]:
    card = _read_card_by_model_id(model_id)
    table_csvs = load_modelid_to_csvlist(model_id, resources=["hugging"])
    related_tables = []
    for csv_path in table_csvs:
        resolved = resolve_table_path(csv_path) or (csv_path if os.path.exists(csv_path) else "")
        if resolved:
            related_tables.append(resolved)

    nugget_table: list[dict[str, Any]] = []
    for csv_path in related_tables:
        nugget_table.extend(_table_to_nuggets(csv_path))

    card_remaining = _norm(_remove_citation_part(_remove_markdown_tables(card)))

    payload = {
        "modelId": model_id,
        "card": card,
        "card_remaining": card_remaining,
        "related_tables": related_tables,
        "nugget_table": nugget_table,
        "nugget_text": [],
    }
    return [payload]


def run_single(args: argparse.Namespace) -> None:
    payloads = extract_nuggets_from_card(args.model_id)
    payload = payloads[0] if payloads else {"modelId": args.model_id, "card": "", "card_remaining": "", "related_tables": [], "nugget_table": [], "nugget_text": []}
    Path(SINGLE_OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(SINGLE_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"saved_json={SINGLE_OUTPUT_JSON}")
    print(f"nuggets={len(payload.get('nugget_table', [])) + len(payload.get('nugget_text', []))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract minimal nuggets from model cards.")
    parser.add_argument("--model-id", required=True, help="Model id to extract nuggets for.")
    args = parser.parse_args()
    run_single(args)


if __name__ == "__main__":
    main()
