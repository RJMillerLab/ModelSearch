#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import re
import sys
from pathlib import Path
from typing import Any

import duckdb
from bs4 import BeautifulSoup
import yaml
from dotenv import load_dotenv

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
load_dotenv(os.path.join(_repo_root, ".env"), override=False)

from src.config import CARD_CONTENT_RAW, OUTPUT_DIR
from src.llm.model import query_openai, setup_openai
from src.utils import load_modelid_to_csvlist, read_csv_robust, resolve_table_path


SHARD_GLOB = "train-0000*-of-00006.parquet"
SINGLE_OUTPUT_YAML = os.path.join(OUTPUT_DIR, "evaluate", "single_modelcard_nuggets.yaml")
MODELCARD_BATCH_DIR = os.path.join(OUTPUT_DIR, "evaluate", "modelcard_nuggets_yaml")
PROMPT_PATH = Path("src/evaluate/card2nugget_prompts.yaml")
PROMPT_KEY_TEXT = "card_remaining_nugget_extraction"
PROMPT_KEY_MAP = "nugget_schema_mapping"
PROMPT_KEY_TABLE_MAP = "table_nugget_schema_mapping"
TEXT_EXTRACTION_MODEL = os.getenv("MODELSEARCHDEMO_TEXT_EXTRACTION_MODEL", "gpt-4o-mini")
ALLOWED_ENTITY_TYPES = {
    "model",
    "base_model",
    "model_variant_type",
    "model_hyperparameters",
    "dataset",
    "base_dataset",
    "infer_dataset",
    "metric",
    "score",
}
SCHEMA_DEFINITION = {
    "entity_type": "model | base_model | model_variant_type | model_hyperparameters | dataset | train_dataset | test_dataset | metric | score",
    "entity_name": "short entity name",
}
SCHEMA_TEXT = yaml.safe_dump(SCHEMA_DEFINITION, sort_keys=False, allow_unicode=True).strip()
TABLE_METRIC_NAMES = {
    "score",
    "cola",
    "sst-2",
    "mrpc",
    "sts-b",
    "qqp",
    "mnli-m",
    "mnli-mm",
    "qnli(v2)",
    "rte",
    "wnli",
    "ax",
}
NAN = "NaN"


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


def _load_prompt(prompt_key: str) -> str:
    if not PROMPT_PATH.exists():
        return ""
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    prompt = data.get(prompt_key, "")
    return str(prompt).strip()


def _replace_prompt_vars(prompt: str, **kwargs: str) -> str:
    out = prompt
    for key, value in kwargs.items():
        out = out.replace(f"[[{key}]]", value)
    return out


def _extract_yaml_payload(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        return None
    raw = raw.replace("```yaml", "").replace("```json", "").replace("```", "").strip()
    try:
        return yaml.safe_load(raw)
    except Exception:
        pass
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = raw.find(start_char)
        end = raw.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            snippet = raw[start : end + 1]
            try:
                return yaml.safe_load(snippet)
            except Exception:
                continue
    return None


def _normalize_entity_type(entity_type: str) -> str:
    text = _norm(entity_type).lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "benchmark_name": "dataset",
        "benchmark": "dataset",
        "leaderboard": "dataset",
        "dataset": "dataset",
        "train_dataset": "base_dataset",
        "training_dataset": "base_dataset",
        "base_dataset": "base_dataset",
        "test_dataset": "infer_dataset",
        "testset": "infer_dataset",
        "eval_dataset": "infer_dataset",
        "evaluation_dataset": "infer_dataset",
        "validation_dataset": "infer_dataset",
        "infer_dataset": "infer_dataset",
        "results": "score",
        "result": "score",
        "numeric_result": "score",
        "value": "score",
        "quantization": "model_variant_type",
        "quantized": "model_variant_type",
        "pretrained": "model_variant_type",
        "int8": "model_variant_type",
        "8_bit": "model_variant_type",
        "4_bit": "model_variant_type",
        "fp16": "model_variant_type",
        "bf16": "model_variant_type",
        "pruned": "model_variant_type",
        "distilled": "model_variant_type",
        "merged": "model_variant_type",
        "adapter": "model_variant_type",
        "finetuned": "model_variant_type",
        "fine_tuned": "model_variant_type",
        "fine-tuned": "model_variant_type",
        "modelvarianttype": "model_variant_type",
        "model_hparams": "model_hyperparameters",
        "hyperparameters": "model_hyperparameters",
        "architecture": "model_hyperparameters",
        "base model": "base_model",
        "baseline_model": "model",
    }
    normalized = aliases.get(text, text)
    if normalized not in ALLOWED_ENTITY_TYPES:
        return ""
    return normalized


def _normalize_entity_name(name: str) -> str:
    text = _norm(name)
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text


def _is_numeric_like(text: str) -> bool:
    value = _norm(text)
    if not value:
        return False
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        return True
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?/[+-]?\d+(?:\.\d+)?", value):
        return True
    return False


def _extract_model_from_value(text: str) -> str:
    value = _norm(text)
    if not value:
        return ""
    m = re.search(r"\(([^()]*BERT[^()]*)\)", value, flags=re.IGNORECASE)
    if m:
        return _normalize_entity_name(m.group(1))
    return ""


def _is_hparam_token(text: str) -> bool:
    value = _norm(text)
    if not value:
        return False
    return bool(re.fullmatch(r"[LHA]\s*=\s*\d+", value, flags=re.IGNORECASE))


def _compose_hparam(row: str, col: str) -> str:
    row_n = _normalize_entity_name(row)
    col_n = _normalize_entity_name(col)
    if _is_hparam_token(row_n) and _is_hparam_token(col_n):
        return f"{row_n}, {col_n}"
    if _is_hparam_token(row_n):
        return row_n
    if _is_hparam_token(col_n):
        return col_n
    return ""


def _to_standardized_schema_rows(
    model_id: str,
    nugget_table_raw: list[dict[str, Any]],
    nugget_text_normalized: list[dict[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for item in nugget_table_raw:
        row_label = _normalize_entity_name(item.get("row", ""))
        col_name = _normalize_entity_name(item.get("col", ""))
        value = _normalize_entity_name(item.get("value", ""))
        col_key = _norm(col_name).lower()
        model_hparams = _compose_hparam(row_label, col_name)

        model_name = row_label if row_label and any(ch.isalpha() for ch in row_label) and not row_label.startswith("L=") else NAN
        dataset_name = NAN
        metric_name = col_name if col_key in TABLE_METRIC_NAMES else NAN
        metric_value = value if _is_numeric_like(value) else NAN
        train_dataset = NAN
        other_features = NAN

        if metric_name == NAN and metric_value == NAN and not model_hparams:
            # Keep table context when it is not a metric/score cell.
            other_features = f"row={row_label or NAN}; col={col_name or NAN}; value={value or NAN}"

        rows.append(
            {
                "Model": model_name,
                "Base_model": NAN,
                "Dataset": dataset_name,
                "Test_dataset": NAN,
                "Metric": metric_name,
                "Metric_value": metric_value,
                "Train_dataset": train_dataset,
                "Model_hyperparameters": model_hparams or NAN,
                "Model_variant_type": NAN,
                "Other_features": other_features,
                "Source": "table",
                "csv_basename": _norm(item.get("csv_basename", "")) or NAN,
            }
        )

    for item in nugget_text_normalized:
        entity_type = _norm(item.get("entity_type", "")).lower()
        entity_name = _normalize_entity_name(item.get("entity_name", ""))
        if not entity_name:
            continue
        row = {
            "Model": NAN,
            "Base_model": NAN,
            "Dataset": NAN,
            "Test_dataset": NAN,
            "Metric": NAN,
            "Metric_value": NAN,
            "Train_dataset": NAN,
            "Model_hyperparameters": NAN,
            "Model_variant_type": NAN,
            "Other_features": NAN,
            "Source": "text",
            "csv_basename": NAN,
        }
        if entity_type == "model":
            row["Model"] = entity_name
        elif entity_type == "base_model":
            row["Base_model"] = entity_name
        elif entity_type == "dataset":
            row["Dataset"] = entity_name
        elif entity_type == "infer_dataset":
            row["Test_dataset"] = entity_name
        elif entity_type == "base_dataset":
            row["Train_dataset"] = entity_name
        elif entity_type == "metric":
            row["Metric"] = entity_name
        elif entity_type == "score":
            row["Metric_value"] = entity_name
        elif entity_type == "model_hyperparameters":
            row["Model_hyperparameters"] = entity_name
        elif entity_type == "model_variant_type":
            row["Model_variant_type"] = entity_name
        else:
            row["Other_features"] = entity_name
        rows.append(row)

    if not rows:
        rows.append(
            {
                "Model": _normalize_entity_name(model_id) or NAN,
                "Base_model": NAN,
                "Dataset": NAN,
                "Test_dataset": NAN,
                "Metric": NAN,
                "Metric_value": NAN,
                "Train_dataset": NAN,
                "Model_hyperparameters": NAN,
                "Model_variant_type": NAN,
                "Other_features": NAN,
                "Source": NAN,
                "csv_basename": NAN,
            }
        )
    return rows


def _to_simple_standardized_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    feature_keys = [
        "Model",
        "Base_model",
        "Dataset",
        "Train_dataset",
        "Test_dataset",
        "Model_hyperparameters",
        "Model_variant_type",
        "Metric",
        "Metric_value",
        "Other_features",
    ]
    simple_rows: list[dict[str, str]] = []
    for row in rows:
        clean_row: dict[str, str] = {}
        for k in feature_keys:
            value = _norm(row.get(k, ""))
            clean_row[k] = "" if not value or value == NAN else value
        simple_rows.append(clean_row)
    return simple_rows


def _extract_card_remaining_raw_nuggets(model_id: str, card_remaining: str) -> list[dict[str, Any]]:
    prompt_template = _load_prompt(PROMPT_KEY_TEXT)
    if not prompt_template or not _norm(card_remaining):
        return []
    if not os.getenv("OPENAI_API_KEY"):
        return []

    prompt = _replace_prompt_vars(
        prompt_template,
        SCHEMA_DEFINITION=SCHEMA_TEXT,
        CARD_REMAINING=card_remaining,
    )
    setup_openai("", mode="openai")
    response_text = query_openai(
        prompt,
        mode="openai",
        model=TEXT_EXTRACTION_MODEL,
        max_tokens=1800,
        temperature=0.0,
    )

    payload = _extract_yaml_payload(response_text)
    if isinstance(payload, dict):
        items = payload.get("nuggets") or payload.get("items") or payload.get("entities") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    nuggets: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        raw_mention = _normalize_entity_name(item.get("raw_mention", "") or item.get("mention", "") or item.get("text", ""))
        context = _normalize_entity_name(item.get("context", ""))
        if not raw_mention:
            continue
        nuggets.append(
            {
                "nugget_id": f"nugget_text_raw::{model_id}::{idx:03d}",
                "raw_mention": raw_mention,
                "context": context,
                "source": "text",
                "reference_doc": _norm(item.get("reference_doc", "")) or "card_remaining",
            }
        )
    return nuggets


def _map_raw_nuggets_to_schema(model_id: str, raw_nuggets: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    if source == "table":
        nuggets: list[dict[str, Any]] = []
        out_idx = 1
        for raw in raw_nuggets:
            row = _normalize_entity_name(raw.get("row", ""))
            col = _normalize_entity_name(raw.get("col", ""))
            value = _normalize_entity_name(raw.get("value", ""))
            csv_basename = _norm(raw.get("csv_basename", ""))
            model_hparams = _compose_hparam(row, col)

            # Row is usually model/variant label in result tables.
            if row and any(ch.isalpha() for ch in row) and not row.startswith("L="):
                nuggets.append(
                    {
                        "nugget_id": f"nugget_table::{model_id}::{out_idx:03d}",
                        "entity_type": "model",
                        "entity_name": row,
                        "raw_mention": row,
                        "source": "table",
                        "csv_basename": csv_basename,
                        "row": row,
                        "col": col,
                        "value": value,
                    }
                )
                out_idx += 1

            col_key = _norm(col).lower()
            if col_key in TABLE_METRIC_NAMES:
                nuggets.append(
                    {
                        "nugget_id": f"nugget_table::{model_id}::{out_idx:03d}",
                        "entity_type": "metric",
                        "entity_name": col,
                        "raw_mention": col,
                        "source": "table",
                        "csv_basename": csv_basename,
                        "row": row,
                        "col": col,
                        "value": value,
                    }
                )
                out_idx += 1

            if value and _is_numeric_like(value):
                nuggets.append(
                    {
                        "nugget_id": f"nugget_table::{model_id}::{out_idx:03d}",
                        "entity_type": "score",
                        "entity_name": value,
                        "raw_mention": value,
                        "source": "table",
                        "csv_basename": csv_basename,
                        "row": row,
                        "col": col,
                        "value": value,
                    }
                )
                out_idx += 1

            value_model = _extract_model_from_value(value)
            if value_model:
                nuggets.append(
                    {
                        "nugget_id": f"nugget_table::{model_id}::{out_idx:03d}",
                        "entity_type": "model",
                        "entity_name": value_model,
                        "raw_mention": value_model,
                        "source": "table",
                        "csv_basename": csv_basename,
                        "row": row,
                        "col": col,
                        "value": value,
                    }
                )
                out_idx += 1
            if model_hparams:
                nuggets.append(
                    {
                        "nugget_id": f"nugget_table::{model_id}::{out_idx:03d}",
                        "entity_type": "model_hyperparameters",
                        "entity_name": model_hparams,
                        "raw_mention": model_hparams,
                        "source": "table",
                        "csv_basename": csv_basename,
                        "row": row,
                        "col": col,
                        "value": value,
                    }
                )
                out_idx += 1
        return nuggets

    prompt_key = PROMPT_KEY_TABLE_MAP if source == "table" else PROMPT_KEY_MAP
    prompt_template = _load_prompt(prompt_key)
    if not prompt_template or not raw_nuggets:
        return []
    if not os.getenv("OPENAI_API_KEY"):
        return []

    if source == "table":
        prompt_nuggets = [
            {
                "row": _norm(item.get("row", "")),
                "col": _norm(item.get("col", "")),
                "value": _norm(item.get("value", "")),
            }
            for item in raw_nuggets
        ]
    else:
        prompt_nuggets = [
            {
                "raw_mention": _norm(item.get("raw_mention", "")),
                "context": _norm(item.get("context", "")),
            }
            for item in raw_nuggets
        ]

    prompt = _replace_prompt_vars(
        prompt_template,
        SCHEMA_DEFINITION=SCHEMA_TEXT,
        RAW_NUGGETS=yaml.safe_dump(prompt_nuggets, sort_keys=False, allow_unicode=True).strip(),
    )
    setup_openai("", mode="openai")
    response_text = query_openai(
        prompt,
        mode="openai",
        model=TEXT_EXTRACTION_MODEL,
        max_tokens=2200,
        temperature=0.0,
    )

    payload = _extract_yaml_payload(response_text)
    if isinstance(payload, dict):
        items = payload.get("nuggets") or payload.get("items") or payload.get("entities") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    nuggets: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        base_raw = raw_nuggets[idx - 1] if idx - 1 < len(raw_nuggets) else {}
        entity_type = _normalize_entity_type(item.get("entity_type", ""))
        row_name = _normalize_entity_name(base_raw.get("row", ""))
        value_name = _normalize_entity_name(base_raw.get("value", ""))
        raw_mention = _normalize_entity_name(item.get("raw_mention", "")) or _normalize_entity_name(base_raw.get("raw_mention", ""))
        entity_name = _normalize_entity_name(
            item.get("entity_name", "") or item.get("canonical_name", "") or raw_mention or row_name or value_name
        )
        if not entity_type or not entity_name:
            continue
        nuggets.append(
            {
                "nugget_id": f"nugget_{source}::{model_id}::{idx:03d}",
                "entity_type": entity_type,
                "entity_name": entity_name,
                "raw_mention": raw_mention,
                "source": _norm(item.get("source", "")) or _norm(base_raw.get("source", "")) or source,
                "csv_basename": _norm(item.get("csv_basename", "")) or _norm(base_raw.get("csv_basename", "")),
                "row": _norm(item.get("row", "")) or _norm(base_raw.get("row", "")),
                "col": _norm(item.get("col", "")) or _norm(base_raw.get("col", "")),
                "value": _norm(item.get("value", "")) or _norm(base_raw.get("value", "")),
            }
        )
    return nuggets


def _extract_and_map_text_nuggets(model_id: str, card_remaining: str) -> list[dict[str, Any]]:
    prompt_template = _load_prompt(PROMPT_KEY_MAP)
    if not prompt_template or not _norm(card_remaining):
        return []
    if not os.getenv("OPENAI_API_KEY"):
        return []

    prompt = _replace_prompt_vars(
        prompt_template,
        SCHEMA_DEFINITION=SCHEMA_TEXT,
        CARD_REMAINING=card_remaining,
    )
    setup_openai("", mode="openai")
    response_text = query_openai(
        prompt,
        mode="openai",
        model=TEXT_EXTRACTION_MODEL,
        max_tokens=2200,
        temperature=0.0,
    )

    payload = _extract_yaml_payload(response_text)
    row_style_items: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        row_style_items = payload.get("rows") or []
        items = payload.get("nuggets") or payload.get("items") or payload.get("entities") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    if not items and row_style_items:
        field_to_entity = {
            "Model": "model",
            "Base_model": "base_model",
            "Dataset": "dataset",
            "Train_dataset": "base_dataset",
            "Test_dataset": "infer_dataset",
            "Model_hyperparameters": "model_hyperparameters",
            "Model_variant_type": "model_variant_type",
            "Metric": "metric",
            "Metric_value": "score",
        }
        converted: list[dict[str, Any]] = []
        for row in row_style_items:
            if not isinstance(row, dict):
                continue
            for field_name, entity_type in field_to_entity.items():
                value = _normalize_entity_name(row.get(field_name, ""))
                if not value:
                    continue
                converted.append({"entity_type": entity_type, "entity_name": value, "raw_mention": value})
        items = converted

    nuggets: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        entity_type = _normalize_entity_type(item.get("entity_type", ""))
        raw_mention = _normalize_entity_name(item.get("raw_mention", "") or item.get("mention", "") or item.get("text", ""))
        context = _normalize_entity_name(item.get("context", ""))
        entity_name = _normalize_entity_name(item.get("entity_name", "") or raw_mention)
        if not entity_type or not entity_name:
            continue
        nuggets.append(
            {
                "nugget_id": f"nugget_text::{model_id}::{idx:03d}",
                "entity_type": entity_type,
                "entity_name": entity_name,
                "raw_mention": raw_mention,
                "context": context,
                "source": "text",
                "csv_basename": "",
                "row": "",
                "col": "",
                "value": "",
            }
        )
    return nuggets


def _dedup_standardized_nuggets(nuggets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for nugget in nuggets:
        key = (
            _norm(nugget.get("entity_type", "")).lower(),
            _norm(nugget.get("entity_name", "")).lower(),
            _norm(nugget.get("source", "")).lower(),
            _norm(nugget.get("csv_basename", "")).lower(),
            _norm(nugget.get("row", "")).lower(),
            _norm(nugget.get("col", "")).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(nugget)
    return deduped


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
            if text.lower() == "nan":
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

    nugget_table_raw: list[dict[str, Any]] = []
    for csv_path in related_tables:
        nugget_table_raw.extend(_table_to_nuggets(csv_path))

    card_remaining = _norm(_remove_citation_part(_remove_markdown_tables(card)))
    nugget_table_normalized = _map_raw_nuggets_to_schema(model_id, nugget_table_raw, source="table") if nugget_table_raw else []
    # Fallback to one-pass text extraction + mapping when table-based extraction yields nothing.
    nugget_text_normalized = _extract_and_map_text_nuggets(model_id, card_remaining) if not nugget_table_normalized else []
    nugget_normalized_entities = _dedup_standardized_nuggets(nugget_table_normalized + nugget_text_normalized)
    standardized_schema_rows = _to_standardized_schema_rows(model_id, nugget_table_raw, nugget_text_normalized)
    standardized_rows_simple = _to_simple_standardized_rows(standardized_schema_rows)

    payload = {
        "modelId": model_id,
        "card": card,
        "card_remaining": card_remaining,
        "related_tables": related_tables,
        "nugget_table": nugget_table_raw,
        # Main final output: normalized rows in fixed schema cells.
        "nugget_normalized": standardized_rows_simple,
    }
    return [payload]


def run_single(args: argparse.Namespace) -> None:
    payloads = extract_nuggets_from_card(args.model_id)
    payload = payloads[0] if payloads else {
        "modelId": args.model_id,
        "card": "",
        "card_remaining": "",
        "related_tables": [],
        "nugget_table": [],
        "nugget_normalized": [],
    }
    Path(SINGLE_OUTPUT_YAML).parent.mkdir(parents=True, exist_ok=True)
    with open(SINGLE_OUTPUT_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(copy.deepcopy(payload), f, sort_keys=False, allow_unicode=True)
    print(f"saved_yaml = {SINGLE_OUTPUT_YAML}")
    print(f"nuggets = {len(payload.get('nugget_normalized', []))}")


def _list_model_ids() -> list[str]:
    if not Path(CARD_CONTENT_RAW).exists():
        raise FileNotFoundError(f"CARD_CONTENT_RAW does not exist: {CARD_CONTENT_RAW}")
    parquet_glob = str(Path(CARD_CONTENT_RAW) / SHARD_GLOB)
    query = """
        SELECT DISTINCT CAST(modelId AS VARCHAR) AS modelId
        FROM read_parquet(?, union_by_name=true)
        WHERE CAST(modelId AS VARCHAR) IS NOT NULL
        ORDER BY modelId
    """
    with duckdb.connect(":memory:") as con:
        rows = con.execute(query, [parquet_glob]).fetchall()
    return [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]


def run_batch(args: argparse.Namespace) -> None:
    model_ids = _list_model_ids()
    if args.limit is not None:
        model_ids = model_ids[: args.limit]
    out_dir = Path(MODELCARD_BATCH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_id in model_ids:
        payload = extract_nuggets_from_card(model_id)[0]
        output_path = out_dir / f"{_norm(model_id).replace('/', '__')}.yaml"
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(copy.deepcopy(payload), f, sort_keys=False, allow_unicode=True)
    print(f"saved_dir = {out_dir}")
    print(f"models = {len(model_ids)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract minimal nuggets from model cards.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="Process one model card.")
    single.add_argument("--model-id", required=True, help="Model id to extract nuggets for.")

    batch = subparsers.add_parser("batch", help="Process many model cards.")
    batch.add_argument("--limit", type=int, default=None, help="Optional max number of model cards to process.")

    args = parser.parse_args()
    if args.command == "single":
        run_single(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
