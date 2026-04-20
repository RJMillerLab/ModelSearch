#!/usr/bin/env python3
"""HF model card → one LLM call → parse markdown table and save tuples to CSV."""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
load_dotenv(os.path.join(_repo_root, ".env"), override=False)

from src.config import CARD_CONTENT_RAW, OUTPUT_DIR
from src.llm.model import query_openai, setup_openai
from src.utils import load_modelid_to_csvlist, read_csv_robust, resolve_table_path

SHARD_GLOB = "train-0000*-of-00006.parquet"
EVAL_DIR = Path(OUTPUT_DIR) / "evaluate"
META_YAML = EVAL_DIR / "single_modelcard_llm_meta.yaml"
LLM_CSV = EVAL_DIR / "single_modelcard_llm.csv"
PROMPT_PATH = Path("src/evaluate/card2nugget_prompts.yaml")
PROMPT_KEY = "nugget_schema_mapping"
PROMPT_KEY_TEXT_ONLY = "nugget_schema_mapping_text_only"
TEXT_EXTRACTION_MODEL = os.getenv("MODELSEARCHDEMO_TEXT_EXTRACTION_MODEL", "gpt-4o-mini")
OUTPUT_HEADERS = [
    "Model",
    "Base_model",
    "Dataset",
    "Train_dataset",
    "Test_dataset",
    "Model_hyperparameters",
    "Model_variant_type",
    "Metric",
    "Metric_value",
    "keep_or_not",
]


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
        SELECT card FROM read_parquet(?, union_by_name=true)
        WHERE CAST(modelId AS VARCHAR) = ? LIMIT 1
    """
    with duckdb.connect(":memory:") as con:
        row = con.execute(query, [parquet_glob, model_id]).fetchone()
    if not row or row[0] is None:
        raise FileNotFoundError(f"No card found for modelId={model_id!r}")
    return str(row[0])


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


def _remove_markdown_tables(text: str) -> str:
    if not isinstance(text, str):
        return ""
    lines = _remove_html_tables(text).splitlines()
    kept: list[str] = []
    in_table = False

    def is_separator(line: str) -> bool:
        return bool(re.fullmatch(r"[\s:\-\|]*", line.strip())) and "|" in line

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            in_table = False
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


def _remove_citation_part(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"(?is)```(?:bibtex|latex)?\s*.*?```", "", text)
    lines = cleaned.splitlines()
    kept: list[str] = []
    skipping = False
    for raw_line in lines:
        line = raw_line.strip()
        if re.match(r"(?i)^\s*(#{1,6}\s*)?(citation|cite|references|reference|bibliography)\b", line):
            break
        if re.match(
            r"(?i)^@(article|inproceedings|misc|techreport|phdthesis|mastersthesis|book|incollection|conference|software)\b",
            line,
        ):
            skipping = True
            continue
        if skipping:
            if not line:
                skipping = False
            continue
        kept.append(raw_line)
    return "\n".join(kept)


def _extract_markdown_table_blocks(card: str, max_chars: int = 12000) -> str:
    if not isinstance(card, str):
        return ""
    lines = _remove_html_tables(card).splitlines()
    blocks: list[str] = []
    buf: list[str] = []
    for raw in lines:
        if "|" in raw.strip() and raw.strip():
            buf.append(raw.rstrip())
        else:
            if len(buf) >= 2:
                blocks.append("\n".join(buf))
            buf = []
    if len(buf) >= 2:
        blocks.append("\n".join(buf))
    out = "\n\n---\n\n".join(blocks)
    return out[:max_chars] if out else ""


def _linked_csv_as_tsv(paths: list[str], max_rows: int = 80) -> str:
    parts: list[str] = []
    for p in paths:
        resolved = resolve_table_path(p) or (p if os.path.exists(p) else "")
        if not resolved:
            continue
        df = read_csv_robust(resolved)
        if df is None or df.empty:
            continue
        name = os.path.basename(resolved)
        chunk = df.head(max_rows).to_csv(index=False, sep="\t")
        parts.append(f"=== {name} ===\n{chunk.rstrip()}")
    return "\n\n".join(parts) if parts else ""


def _build_table_context(card: str, related_tables: list[str]) -> str:
    md = _extract_markdown_table_blocks(card)
    csv_txt = _linked_csv_as_tsv(related_tables)
    chunks = []
    if md:
        chunks.append("MARKDOWN TABLES (from card):\n" + md)
    if csv_txt:
        chunks.append("LINKED CSV TABLES (tsv):\n" + csv_txt)
    return "\n\n".join(chunks) if chunks else "(no table text)"


def _load_prompt_template(prompt_key: str) -> str:
    if not PROMPT_PATH.exists():
        return ""
    data = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8")) or {}
    return str(data.get(prompt_key, "")).strip()


def _split_pipe_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [_norm(p) for p in s.split("|")]


def _is_markdown_separator_row(line: str) -> bool:
    cells = _split_pipe_row(line)
    if len(cells) < 2:
        return False
    return all(bool(re.fullmatch(r":?-{2,}:?", c.strip())) for c in cells if c.strip())


def _parse_markdown_table(response_text: str) -> list[dict[str, str]]:
    text = response_text or ""
    bodies: list[str] = []
    for m in re.finditer(r"```(?:markdown|md)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        bodies.append(m.group(1).strip())
    if not bodies and "|" in text:
        bodies.append(text.strip())

    rows_out: list[dict[str, str]] = []
    for body in bodies:
        lines = [ln.strip() for ln in body.splitlines() if "|" in ln and ln.strip()]
        if len(lines) < 3:
            continue
        header_cells = _split_pipe_row(lines[0])
        header_map = {h.lower().replace(" ", "_"): i for i, h in enumerate(header_cells)}
        start = 1
        if start < len(lines) and _is_markdown_separator_row(lines[start]):
            start += 1
        for ln in lines[start:]:
            if _is_markdown_separator_row(ln):
                continue
            cells = _split_pipe_row(ln)
            row: dict[str, str] = {}
            for h in OUTPUT_HEADERS:
                idx = header_map.get(h.lower(), -1)
                row[h] = cells[idx] if idx >= 0 and idx < len(cells) else ""
            if any(_norm(v) for v in row.values()):
                rows_out.append(row)
        if rows_out:
            break
    return rows_out


def _extract_global_hparams(text: str) -> str:
    src = text or ""
    parts: list[str] = []

    m_epochs = re.search(r"trained\s+for\s+(\d+)\s+epochs?", src, flags=re.IGNORECASE)
    if m_epochs:
        parts.append(f"epochs={m_epochs.group(1)}")

    m_bs = re.search(r"batch\s+sizes?\s*:\s*([^\n]+)", src, flags=re.IGNORECASE)
    if m_bs:
        parts.append(f"batch_sizes={_norm(m_bs.group(1))}")

    m_lr = re.search(r"learning\s+rates?\s*:\s*([^\n]+)", src, flags=re.IGNORECASE)
    if m_lr:
        parts.append(f"learning_rates={_norm(m_lr.group(1))}")

    return "; ".join(parts)


def _apply_hparam_fallback(rows: list[dict[str, str]], card_remaining: str) -> tuple[list[dict[str, str]], str]:
    global_hparams = _extract_global_hparams(card_remaining)
    if not global_hparams:
        return rows, ""
    out: list[dict[str, str]] = [dict(r) for r in rows]
    exists = any(_norm(r.get("Model_hyperparameters", "")) == global_hparams for r in out)
    if not exists:
        # Add one dedicated row for global training hyperparameters.
        out.append(
            {
                "Model": "",
                "Base_model": "",
                "Dataset": "",
                "Train_dataset": "",
                "Test_dataset": "",
                "Model_hyperparameters": global_hparams,
                "Model_variant_type": "",
                "Metric": "",
                "Metric_value": "",
                "keep_or_not": "keep",
            }
        )
    return out, global_hparams


def _write_llm_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(OUTPUT_HEADERS)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def read_llm_csv(path: str | Path) -> list[dict[str, str]]:
    """Load parsed tuple rows from CSV."""
    p = Path(path)
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run_single(model_id: str) -> dict[str, Any]:
    t0 = time.time()
    t_last = t0

    def _tick(stage: str) -> None:
        nonlocal t_last
        now = time.time()
        print(f"[stage] {stage} | +{now - t_last:.3f}s | total={now - t0:.3f}s")
        t_last = now

    print(f"[start] model_id={model_id}")
    card = _read_card_by_model_id(model_id)
    _tick("read_card_by_model_id")
    table_csvs = load_modelid_to_csvlist(model_id, resources=["hugging"])
    _tick("load_modelid_to_csvlist")
    related_tables: list[str] = []
    for csv_path in table_csvs:
        resolved = resolve_table_path(csv_path) or (csv_path if os.path.exists(csv_path) else "")
        if resolved:
            related_tables.append(resolved)
    _tick(f"resolve_related_tables(count={len(related_tables)})")

    card_remaining = _norm(_remove_citation_part(_remove_markdown_tables(card)))
    _tick("build_card_remaining")
    table_context = _build_table_context(card, related_tables)
    _tick("build_table_context")

    has_table_context = bool(table_context and table_context != "(no table text)")
    prompt_key = PROMPT_KEY if has_table_context else PROMPT_KEY_TEXT_ONLY
    template = _load_prompt_template(prompt_key)
    if not template:
        raise RuntimeError(f"Missing prompt key {prompt_key} in {PROMPT_PATH}")
    _tick("load_prompt_template")
    prompt = template.replace("[[TABLE_CONTEXT]]", table_context).replace("[[CARD_REMAINING]]", card_remaining)
    print(f"[prompt] selected={prompt_key}")
    _tick("compose_prompt")

    response = ""
    note = ""
    if not os.getenv("OPENAI_API_KEY"):
        note = "skip_no_api_key"
        _tick("skip_llm_no_api_key")
    else:
        setup_openai("", mode="openai")
        _tick("setup_openai")
        print("[llm] querying openai...")
        response = query_openai(
            prompt,
            mode="openai",
            model=TEXT_EXTRACTION_MODEL,
            max_tokens=4096,
            temperature=0.0,
        ) or ""
        _tick(f"query_openai(response_chars={len(response)})")

    parsed_rows = _parse_markdown_table(response)
    parsed_rows, global_hparams = _apply_hparam_fallback(parsed_rows, card_remaining)
    _write_llm_csv(LLM_CSV, parsed_rows)
    _tick(f"write_llm_csv(path={LLM_CSV})")

    meta = {
        "modelId": model_id,
        "llm_model": TEXT_EXTRACTION_MODEL,
        "prompt_key": prompt_key,
        "llm_csv": str(LLM_CSV.resolve()),
        "prompt": prompt,
        "raw_response": response,
        "parsed_rows": len(parsed_rows),
        "global_hparams_fallback": global_hparams,
        "related_tables": related_tables,
        "llm_turns": [
            {
                "turn_index": 1,
                "note": note,
            }
        ],
    }
    _tick("assemble_meta_payload")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Model card → LLM; parse markdown response table and save tuples to CSV.")
    parser.add_argument("--model-id", default=None, help="Hugging Face model id (required unless --read-csv).")
    parser.add_argument(
        "--read-csv",
        nargs="?",
        const=str(LLM_CSV),
        metavar="PATH",
        help="Print stats for parsed tuple CSV; PATH defaults to the standard single_modelcard_llm.csv.",
    )
    args = parser.parse_args()

    if args.read_csv is not None:
        p = Path(args.read_csv)
        if not p.is_file():
            print(f"missing: {p.resolve()}")
            return
        rows = read_llm_csv(p)
        if not rows:
            print(f"empty: {p.resolve()}")
            return
        headers = list(rows[0].keys())
        print(f"csv = {p.resolve()}")
        print(f"rows = {len(rows)}")
        print(f"headers = {headers}")
        return

    if not args.model_id:
        parser.error("--model-id is required unless using --read-csv")

    meta = run_single(args.model_id)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    t_write = time.time()
    with open(META_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)
    print(f"[stage] write_meta_yaml | +{time.time() - t_write:.3f}s")
    print(f"saved_csv = {LLM_CSV.resolve()}")
    print(f"saved_meta_yaml = {META_YAML.resolve()}")


if __name__ == "__main__":
    main()
