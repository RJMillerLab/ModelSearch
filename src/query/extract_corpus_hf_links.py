#!/usr/bin/env python
"""
Batch extract S2ORC / corpus full text by corpusId and scan for Hugging Face links.

This script is meant to run on the remote server that has:
  1) a SQLite index with a `papers` table
  2) the corresponding NDJSON / JSONL corpus files on disk

Typical usage:

  python tmp/extract_corpus_hf_links.py \
    --ids_file tmp/new_corpusids.txt \
    --db_path /path/to/paper_index_mini.db \
    --data_directory /path/to/se_s2orc_250218 \
    --output_parquet tmp/corpus_hf_links.parquet \
    --keep_full_text

Input formats supported for `--ids_file`:
  - .txt / .list: one corpusId per line
  - .csv / .tsv: a column named corpusid or corpusId
  - .parquet: a column named corpusid or corpusId

Output columns include:
  - corpusid, title, paper_title, filename, line_index
  - full_text_len, openaccessurl
  - hf_links, hf_models, hf_datasets, hf_spaces
  - hf_link_snippets
  - full_text (optional, with --keep_full_text)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

import sys

import pandas as pd
from tqdm import tqdm

# Make the repo root importable when running this file directly from tmp/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


HF_URL_RE = re.compile(
    r"(?P<url>(?:https?://)?huggingface\.co/[^\s<>'\"(){}\[\]]+)",
    re.IGNORECASE,
)


def normalize_corpusid(value) -> Optional[str]:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def load_input_ids(ids_file: str) -> pd.DataFrame:
    path = Path(ids_file)
    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".csv"}:
        df = pd.read_csv(path)
    elif suffix in {".tsv", ".txt", ".list"}:
        if suffix == ".tsv":
            df = pd.read_csv(path, sep="\t")
        else:
            with open(path, "r", encoding="utf-8") as f:
                ids = [ln.strip() for ln in f if ln.strip()]
            df = pd.DataFrame({"corpusid": ids})
    else:
        raise ValueError(f"Unsupported ids file format: {ids_file}")

    corpus_col = None
    for candidate in ["corpusid", "corpusId", "CorpusId"]:
        if candidate in df.columns:
            corpus_col = candidate
            break
    if corpus_col is None:
        if len(df.columns) == 1:
            corpus_col = df.columns[0]
        else:
            raise ValueError(
                "Could not find corpusId column. Expected one of corpusid/corpusId "
                f"in {ids_file}"
            )

    df = df.copy()
    df["corpusid"] = df[corpus_col].map(normalize_corpusid)
    df = df[df["corpusid"].notna()].copy()
    return df


def query_papers_by_corpusids(db_path: str, corpusids: Sequence[str], batch_size: int = 1000) -> pd.DataFrame:
    corpusids = [normalize_corpusid(cid) for cid in corpusids]
    corpusids = [cid for cid in corpusids if cid]

    frames: List[pd.DataFrame] = []
    with sqlite3.connect(db_path) as conn:
        for i in range(0, len(corpusids), batch_size):
            batch = corpusids[i : i + batch_size]
            if not batch:
                continue
            placeholders = ",".join(["?"] * len(batch))
            query = (
                "SELECT corpusid, filename, line_index, title "
                "FROM papers "
                f"WHERE corpusid IN ({placeholders})"
            )
            try:
                frames.append(pd.read_sql_query(query, conn, params=list(batch)))
            except Exception:
                fallback_query = (
                    "SELECT corpusid, filename, line_index, LOWER(TRIM(title)) AS title "
                    "FROM papers "
                    f"WHERE corpusid IN ({placeholders})"
                )
                frames.append(pd.read_sql_query(fallback_query, conn, params=list(batch)))

    if not frames:
        return pd.DataFrame(columns=["corpusid", "filename", "line_index", "title"])

    db_df = pd.concat(frames, ignore_index=True)
    db_df["corpusid"] = db_df["corpusid"].map(normalize_corpusid)
    db_df["filename"] = db_df["filename"].astype(str)
    db_df["line_index"] = pd.to_numeric(db_df["line_index"], errors="coerce")
    db_df = db_df.dropna(subset=["corpusid", "filename", "line_index"]).copy()
    db_df["line_index"] = db_df["line_index"].astype(int)
    return db_df


def resolve_corpus_file(base_directory: str, filename: str) -> Optional[str]:
    if not isinstance(filename, str) or not filename.strip():
        return None
    direct = Path(base_directory) / filename
    if direct.exists():
        return str(direct)
    basename = Path(filename).name
    candidate = Path(base_directory) / basename
    if candidate.exists():
        return str(candidate)
    return None


def read_specific_lines(file_path: str, wanted_zero_based: Sequence[int]) -> Dict[int, str]:
    wanted = sorted({int(i) for i in wanted_zero_based if i is not None and int(i) >= 0})
    if not wanted:
        return {}

    wanted_set = set(wanted)
    found: Dict[int, str] = {}
    max_wanted = wanted[-1]

    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no > max_wanted:
                break
            if line_no in wanted_set:
                found[line_no] = line.rstrip("\n")
                if len(found) == len(wanted_set):
                    break
    return found


def extract_openaccessurl(paper_json: dict) -> Optional[str]:
    content = paper_json.get("content") or {}
    if not isinstance(content, dict):
        return None
    source = content.get("source") or {}
    if not isinstance(source, dict):
        return None
    oainfo = source.get("oainfo") or {}
    if not isinstance(oainfo, dict):
        return None
    value = oainfo.get("openaccessurl")
    return value if isinstance(value, str) and value.strip() else None


def extract_full_text(paper_json: dict) -> str:
    content = paper_json.get("content") or {}
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
    text = paper_json.get("text")
    if isinstance(text, str):
        return text
    return ""


def normalize_hf_url(raw_url: str) -> Optional[str]:
    if not isinstance(raw_url, str):
        return None
    text = raw_url.strip().rstrip(").,;:!?]}>\"'")
    if not text:
        return None
    if not text.lower().startswith(("http://", "https://")):
        text = "https://" + text

    try:
        parsed = urlsplit(text)
    except Exception:
        return None

    if "huggingface.co" not in parsed.netloc.lower():
        return None

    path = parsed.path.strip("/")
    if not path:
        return None

    path = path.split("?", 1)[0].split("#", 1)[0].strip("/")
    if not path:
        return None

    normalized = urlunsplit(("https", "huggingface.co", f"/{path}", "", ""))
    return normalized


def classify_hf_path(path: str) -> str:
    path = path.strip("/")
    if path.startswith("datasets/"):
        return "dataset"
    if path.startswith("spaces/"):
        return "space"
    if path.startswith("models/"):
        return "model"
    if "/" in path:
        return "model"
    return "unknown"


def extract_hf_links(text: str, context_window: int = 120) -> Tuple[List[str], List[str], List[str], List[dict]]:
    if not isinstance(text, str) or not text:
        return [], [], [], []

    links: List[str] = []
    models: List[str] = []
    datasets: List[str] = []
    spaces: List[str] = []
    snippets: List[dict] = []

    for match in HF_URL_RE.finditer(text):
        normalized = normalize_hf_url(match.group("url"))
        if not normalized:
            continue
        if normalized in links:
            continue

        parsed = urlsplit(normalized)
        path = parsed.path.strip("/")
        link_type = classify_hf_path(path)

        links.append(normalized)
        if link_type == "dataset":
            datasets.append(normalized)
        elif link_type == "space":
            spaces.append(normalized)
        elif link_type == "model":
            models.append(normalized)

        start = max(0, match.start() - context_window)
        end = min(len(text), match.end() + context_window)
        snippets.append(
            {
                "url": normalized,
                "type": link_type,
                "snippet": text[start:end].replace("\n", " ").strip(),
            }
        )

    return links, models, datasets, snippets


def load_json_line(line: str) -> Optional[dict]:
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        return None


def build_output_rows(
    input_df: pd.DataFrame,
    db_df: pd.DataFrame,
    data_directory: str,
    keep_full_text: bool = False,
    full_text_dir: Optional[str] = None,
    num_workers: int = 1,
) -> pd.DataFrame:
    merged = input_df.merge(db_df, on="corpusid", how="left", suffixes=("", "_db"))
    merged["status"] = "missing_in_db"
    merged.loc[merged["filename"].notna() & merged["line_index"].notna(), "status"] = "db_hit"

    if full_text_dir:
        Path(full_text_dir).mkdir(parents=True, exist_ok=True)

    grouped = (
        merged[merged["status"] == "db_hit"]
        .drop_duplicates(subset=["filename", "line_index"])
        .groupby("filename", sort=False)
    )

    file_cache: Dict[str, Dict[int, str]] = {}

    def read_group(filename: str, group: pd.DataFrame) -> Tuple[str, Dict[int, str]]:
        resolved = resolve_corpus_file(data_directory, filename)
        if not resolved:
            return filename, {}
        wanted = group["line_index"].astype(int).tolist()
        return filename, read_specific_lines(resolved, wanted)

    grouped_items = list(grouped)
    if num_workers <= 1 or len(grouped_items) <= 1:
        for filename, group in tqdm(grouped_items, desc="Reading corpus files", total=len(grouped_items)):
            key, value = read_group(filename, group)
            if value:
                file_cache[key] = value
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(read_group, filename, group): filename
                for filename, group in grouped_items
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Reading corpus files"):
                key, value = future.result()
                if value:
                    file_cache[key] = value

    records: List[dict] = []
    for row in tqdm(merged.itertuples(index=False), total=len(merged), desc="Parsing full text"):
        record = row._asdict()
        record.setdefault("full_text", None)
        record.setdefault("openaccessurl", None)
        record.setdefault("hf_links", [])
        record.setdefault("hf_models", [])
        record.setdefault("hf_datasets", [])
        record.setdefault("hf_spaces", [])
        record.setdefault("hf_link_snippets", [])
        record.setdefault("full_text_len", None)
        record.setdefault("full_text_path", None)

        if record["status"] != "db_hit":
            records.append(record)
            continue

        filename = record["filename"]
        line_index = int(record["line_index"])
        raw_line = file_cache.get(filename, {}).get(line_index)
        paper_json = load_json_line(raw_line) if raw_line else None
        if not paper_json:
            record["status"] = "json_parse_failed"
            records.append(record)
            continue

        full_text = extract_full_text(paper_json)
        openaccessurl = extract_openaccessurl(paper_json)
        hf_links, hf_models, hf_datasets, hf_link_snippets = extract_hf_links(full_text)

        record["openaccessurl"] = openaccessurl
        record["hf_links"] = hf_links
        record["hf_models"] = hf_models
        record["hf_datasets"] = hf_datasets
        record["hf_spaces"] = []
        record["hf_link_snippets"] = hf_link_snippets
        record["full_text_len"] = len(full_text)

        if keep_full_text:
            record["full_text"] = full_text

        if full_text_dir and full_text:
            safe_corpusid = str(record["corpusid"]).replace("/", "_")
            out_path = Path(full_text_dir) / f"{safe_corpusid}.txt"
            if not out_path.exists():
                out_path.write_text(full_text, encoding="utf-8")
            record["full_text_path"] = str(out_path)

        records.append(record)

    out_df = pd.DataFrame(records)
    if "title" in out_df.columns and "paper_title" not in out_df.columns:
        out_df["paper_title"] = out_df["title"]
    elif "paper_title" not in out_df.columns:
        out_df["paper_title"] = None

    preferred_order = [
        "corpusid",
        "title",
        "paper_title",
        "filename",
        "line_index",
        "status",
        "full_text_len",
        "openaccessurl",
        "hf_links",
        "hf_models",
        "hf_datasets",
        "hf_spaces",
        "hf_link_snippets",
        "full_text_path",
        "full_text",
    ]
    existing = [col for col in preferred_order if col in out_df.columns]
    remaining = [col for col in out_df.columns if col not in existing]
    out_df = out_df[existing + remaining]

    if "hf_link_snippets" in out_df.columns:
        out_df["hf_link_snippets"] = out_df["hf_link_snippets"].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
        )
    for col in ["hf_links", "hf_models", "hf_datasets", "hf_spaces"]:
        if col in out_df.columns:
            out_df[col] = out_df[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
            )
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract corpus full text and Hugging Face links by corpusId")
    parser.add_argument("--ids_file", required=True, help="txt/csv/tsv/parquet file with corpus ids")
    parser.add_argument("--db_path", required=True, help="SQLite index path containing table `papers`")
    parser.add_argument("--data_directory", required=True, help="Directory containing corpus NDJSON/JSONL files")
    parser.add_argument("--output_parquet", default="tmp/corpus_hf_links.parquet", help="Output parquet path")
    parser.add_argument("--batch_size", type=int, default=1000, help="SQL batch size")
    parser.add_argument("--keep_full_text", action="store_true", help="Keep full text in the output parquet")
    parser.add_argument("--full_text_dir", default=None, help="Optional directory to dump each full text as txt")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker threads to read corpus files in parallel",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick testing")
    args = parser.parse_args()

    input_df = load_input_ids(args.ids_file)
    if args.limit is not None:
        input_df = input_df.head(args.limit).copy()

    unique_ids = input_df["corpusid"].dropna().astype(str).unique().tolist()
    print(f"Loaded {len(input_df)} input rows, {len(unique_ids)} unique corpusids")

    db_df = query_papers_by_corpusids(args.db_path, unique_ids, batch_size=args.batch_size)
    print(f"DB returned {len(db_df)} rows")

    out_df = build_output_rows(
        input_df=input_df,
        db_df=db_df,
        data_directory=args.data_directory,
        keep_full_text=args.keep_full_text,
        full_text_dir=args.full_text_dir,
        num_workers=max(1, args.num_workers),
    )

    out_path = Path(args.output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(str(out_path), index=False)
    print(f"Saved {len(out_df)} rows to {out_path}")

    matched = int((out_df["status"] == "db_hit").sum()) if "status" in out_df.columns else 0
    with_links = int(
        out_df["hf_links"].fillna("[]").astype(str).ne("[]").sum()
    ) if "hf_links" in out_df.columns else 0
    print(f"Matched corpus rows: {matched}")
    print(f"Rows with HF links: {with_links}")


if __name__ == "__main__":
    main()
