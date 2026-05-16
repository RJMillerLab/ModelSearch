#!/usr/bin/env python3
"""Map a user query to nugget headers (step 1), then match CSV rows (step 2: header-nonempty).

Optional step 3 ``get_subset_rows`` (LLM) is stubbed for fine-grained narrowing; not called by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
load_dotenv(os.path.join(_repo_root, ".env"), override=False)

from src.config import OUTPUT_DIR
from src.evaluate.nugget_schema import NUGGET_SCHEMA_HEADERS
from src.llm.batch import main_batch_query
from src.llm.model import query_openai, setup_openai

QUERY_MAP_HEADERS = list(NUGGET_SCHEMA_HEADERS)

# Job-merged integrated CSV (wrap_card_query_eval) adds provenance; not used for header matching.
SOURCE_MODEL_ID_COLUMN = "source_model_id"
MODEL_ROW_COLUMNS = frozenset({"Model", "Base_model", "Model_variant_type"})
DATASET_ROW_COLUMNS = frozenset({"Dataset"})
METRIC_ROW_COLUMNS = frozenset({"Metric_name", "Metric_value"})

OUTPUT_HEADER_KEYWORD_JSON = os.path.join(OUTPUT_DIR, "evaluate", "query_header_keyword_mapping.json")
OUTPUT_QRELS = os.path.join(OUTPUT_DIR, "evaluate", "real_subtopic.qrels")
OUTPUT_RUN = os.path.join(OUTPUT_DIR, "evaluate", "real_initial.run")
OUTPUT_MATCH_DEBUG_JSON = os.path.join(OUTPUT_DIR, "evaluate", "query_csv_match_debug.json")
BATCH_DIR = Path(OUTPUT_DIR) / "evaluate" / "batch"
EVAL_DIR = Path(OUTPUT_DIR) / "evaluate"
CARD2NUGGET_DIR = Path(OUTPUT_DIR) / "card2nugget"

PROMPT_PATH = Path("src/evaluate/query2nugget_prompts.yaml")
PROMPT_KEY = "query_to_nugget_headers"
FILTER_PROMPT_KEY = "query_to_nugget_filter"
TEXT_MODEL = os.getenv("MODELSEARCHDEMO_TEXT_EXTRACTION_MODEL", "gpt-5.4-mini")


def _load_prompt_template(prompt_key: str = PROMPT_KEY) -> str:
    if not PROMPT_PATH.exists():
        return ""
    data = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8")) or {}
    return str(data.get(prompt_key, "")).strip()


def _build_filter_prompt(query: str, header_values: list[dict[str, str]], candidate_lines: list[str]) -> str:
    template = _load_prompt_template(FILTER_PROMPT_KEY)
    if not template:
        raise RuntimeError(f"Missing prompt key {FILTER_PROMPT_KEY} in {PROMPT_PATH}")
    return (
        template.replace("[[USER_QUERY]]", query.strip())
        .replace("[[HEADER_VALUES_JSON]]", json.dumps(header_values, ensure_ascii=False))
        .replace("[[CANDIDATE_ROWS]]", "\n".join(candidate_lines))
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                pass
    return None


def _extract_header_list_text(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    allowed = set(QUERY_MAP_HEADERS)
    out: list[str] = []
    seen: set[str] = set()
    for ln in lines:
        s = re.sub(r"^\s*[-*•\d\.)\s]+", "", ln).strip()
        if ":" in s:
            s = s.split(":", 1)[0].strip()
        if s in allowed and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _build_prompt_for_query(query: str) -> str:
    template = _load_prompt_template()
    if not template:
        raise RuntimeError(f"Missing prompt key {PROMPT_KEY} in {PROMPT_PATH}")
    headers_yaml = yaml.safe_dump(QUERY_MAP_HEADERS, allow_unicode=True).strip()
    return template.replace("[[HEADERS_YAML]]", headers_yaml).replace("[[USER_QUERY]]", (query or "").strip())


def _finalize_map_response(query: str, prompt: str, raw: str) -> dict[str, Any]:
    parsed = _extract_json_object(raw)
    if not parsed:
        header_list = _extract_header_list_text(raw)
        if header_list:
            related = [{"header": h, "keywords": []} for h in header_list]
            out = {"query": query, "related": related, "clusters": [{"related": related, "filters": []}], "header_list": header_list}
            out["prompt"] = prompt
            out["raw_response"] = raw
            return out
        return {
            "query": query,
            "related": [],
            "clusters": [],
            "header_list": [],
            "error": "parse_failed",
            "prompt": prompt,
            "raw_response": raw,
        }
    out = _validate_and_normalize(parsed, query)
    out["prompt"] = prompt
    out["raw_response"] = raw
    return out


def _normalize_filter_dict(f: dict[str, Any]) -> dict[str, str] | None:
    col = str(f.get("column", "")).strip()
    if col not in NUGGET_SCHEMA_HEADERS:
        return None
    needle = str(f.get("contains", "")).strip() or str(f.get("equals", "")).strip()
    if not needle:
        return None
    return {"column": col, "contains": needle}


def _normalize_related_entries(related_raw: list[Any]) -> list[dict[str, Any]]:
    allowed = set(QUERY_MAP_HEADERS)
    out: list[dict[str, Any]] = []
    for item in related_raw:
        if not isinstance(item, dict):
            continue
        h = str(item.get("header", "")).strip()
        if h not in allowed:
            continue
        kws = item.get("keywords")
        if isinstance(kws, str):
            kws = [kws]
        elif not isinstance(kws, list):
            kws = []
        keywords = [str(x).strip() for x in kws if str(x).strip()]
        entry: dict[str, Any] = {"header": h, "keywords": keywords}
        vc = str(item.get("value_contains", "")).strip()
        if vc:
            entry["value_contains"] = vc
        out.append(entry)
    return out


def _cluster_from_related_and_filters(related: list[dict[str, Any]], top_filters: list[dict[str, str]]) -> dict[str, Any]:
    filters = list(top_filters)
    related_clean: list[dict[str, Any]] = []
    for it in related:
        vc = str(it.get("value_contains", "")).strip()
        h = str(it.get("header", "")).strip()
        if vc and h in NUGGET_SCHEMA_HEADERS:
            filters.append({"column": h, "contains": vc})
        if h:
            related_clean.append({"header": h, "keywords": it.get("keywords", [])})
    return {"related": related_clean, "filters": filters}


def _clusters_from_parsed(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    clusters_raw = parsed.get("clusters")
    top_filters: list[dict[str, str]] = []
    if isinstance(parsed.get("filters"), list):
        for f in parsed["filters"]:
            if isinstance(f, dict):
                nf = _normalize_filter_dict(f)
                if nf:
                    top_filters.append(nf)

    if isinstance(clusters_raw, list) and clusters_raw:
        clusters_out: list[dict[str, Any]] = []
        for c in clusters_raw:
            if not isinstance(c, dict):
                continue
            rel = _normalize_related_entries(c.get("related", []) if isinstance(c.get("related"), list) else [])
            fils: list[dict[str, str]] = list(top_filters)
            if isinstance(c.get("filters"), list):
                for f in c["filters"]:
                    if isinstance(f, dict):
                        nf = _normalize_filter_dict(f)
                        if nf:
                            fils.append(nf)
            for it in rel:
                vc = str(it.get("value_contains", "")).strip()
                h = str(it.get("header", "")).strip()
                if vc and h in NUGGET_SCHEMA_HEADERS:
                    fils.append({"column": h, "contains": vc})
            rel_clean = [{"header": x["header"], "keywords": x.get("keywords", [])} for x in rel]
            if rel_clean or fils:
                clusters_out.append({"related": rel_clean, "filters": fils})
        if clusters_out:
            return clusters_out

    related = _normalize_related_entries(parsed.get("related", []) if isinstance(parsed.get("related"), list) else [])
    one = _cluster_from_related_and_filters(related, top_filters)
    if one["related"] or one["filters"]:
        return [one]
    return []


def _validate_and_normalize(parsed: dict[str, Any], user_query: str) -> dict[str, Any]:
    q = str(parsed.get("query", "")).strip() or user_query
    clusters = _clusters_from_parsed(parsed)
    if not clusters:
        selected = parsed.get("selected_headers", [])
        if isinstance(selected, list):
            related = [
                {"header": str(h).strip(), "keywords": []}
                for h in selected
                if str(h).strip() in QUERY_MAP_HEADERS
            ]
            if related:
                clusters = [{"related": related, "filters": []}]
    flat_related: list[dict[str, Any]] = []
    seen_h: set[str] = set()
    for c in clusters:
        for it in c.get("related", []):
            h = str(it.get("header", "")).strip()
            if h and h not in seen_h:
                seen_h.add(h)
                flat_related.append({"header": h, "keywords": list(it.get("keywords", []))})
    out: dict[str, Any] = {"query": q, "related": flat_related, "clusters": clusters, "header_list": [x["header"] for x in flat_related]}
    header_values = parsed.get("header_values", [])
    if isinstance(header_values, list):
        out["header_values"] = [
            {"header": str(x.get("header", "")).strip(), "contains": str(x.get("contains", "")).strip()}
            for x in header_values
            if isinstance(x, dict)
            and str(x.get("header", "")).strip() in QUERY_MAP_HEADERS
            and str(x.get("contains", "")).strip()
        ]
        for hv in out["header_values"]:
            h = hv["header"]
            if h not in seen_h:
                seen_h.add(h)
                out["related"].append({"header": h, "keywords": []})
                out["header_list"].append(h)
    return out


def _norm_cell(v: Any) -> str:
    if v is None:
        return ""
    text = str(v).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text)


def _norm_key(s: str) -> str:
    return (s or "").strip()


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


def count_csv_data_rows(path: Path) -> int:
    """Data rows only (exclude header). For sanity checks when building qrels/run."""
    if not path.is_file():
        return 0
    try:
        with open(path, encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            if next(r, None) is None:
                return 0
            return sum(1 for _ in r)
    except OSError:
        return 0


# Backwards-compatible name
_count_csv_data_rows = count_csv_data_rows


def _md_cell(s: str) -> str:
    """Escape pipe for markdown table cells."""
    return (s or "").replace("|", "\\|").replace("\n", " ")


def format_query2nugget_match_markdown(
    *,
    mode: Literal["structured", "llm_rerank"],
    csv_inventory: list[tuple[Path, int]],
    per_query: list[dict[str, Any]],
    qrels_lines: int,
    run_lines: int,
) -> str:
    """Markdown tables for match summary (file or stdout)."""
    mode_label = "structured (header-nonempty)" if mode == "structured" else "llm_rerank"
    lines: list[str] = [
        f"### Match mode: `{_md_cell(mode_label)}`",
        "",
        "#### Input CSV (data_rows = body rows, no header)",
        "| data_rows | path |",
        "| ---: | --- |",
    ]
    for p, n in csv_inventory:
        lines.append(f"| {n} | `{_md_cell(str(p.resolve()))}` |")
    lines.extend(["", "#### Per query", ""])
    if mode == "structured":
        lines.extend(["| qid | matched | note |", "| --- | ---: | --- |"])
        for row in per_query:
            qid = str(row.get("qid", ""))
            note = _md_cell(str(row.get("note", "")))
            m = row.get("matched")
            ms = f"{m}" if m is not None else "—"
            lines.append(f"| `{_md_cell(qid)}` | {ms} | {note} |")
    else:
        lines.extend(["| qid | hits | cand | rerank_note | note |", "| --- | ---: | ---: | --- | --- |"])
        for row in per_query:
            qid = str(row.get("qid", ""))
            note = _md_cell(str(row.get("note", "")))
            h = row.get("hits", "—")
            c = row.get("candidates", "—")
            rn = _md_cell(str(row.get("rerank_note", ""))[:48])
            lines.append(f"| `{_md_cell(qid)}` | {h} | {c} | {rn} | {note} |")
    lines.extend(
        [
            "",
            f"**TOTALS:** `qrels_lines={qrels_lines}` · `run_lines={run_lines}`",
            "",
        ]
    )
    return "\n".join(lines)


def emit_query2nugget_match_report(
    md_body: str,
    *,
    match_log_md: Path | None,
    section_heading: str,
    to_stdout: bool,
) -> None:
    """Append markdown to file and/or print (CLI without log path)."""
    if match_log_md is not None:
        match_log_md.parent.mkdir(parents=True, exist_ok=True)
        with open(match_log_md, "a", encoding="utf-8") as f:
            f.write(section_heading.rstrip() + "\n\n")
            f.write(md_body)
            if not md_body.endswith("\n"):
                f.write("\n")
            f.write("\n---\n\n")
    if to_stdout:
        print(md_body)


def _row_dict(row: dict[str, Any]) -> dict[str, str]:
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
    sm = raw.get(SOURCE_MODEL_ID_COLUMN) or raw.get("From_model") or raw.get("from_model")
    if sm:
        out[SOURCE_MODEL_ID_COLUMN] = _norm_cell(sm)
    return out


def _cells_with_provenance(cells: dict[str, str]) -> dict[str, str]:
    """Schema cells plus source_model_id when present (integrated job CSV)."""
    out = {h: cells[h] for h in NUGGET_SCHEMA_HEADERS if cells.get(h)}
    if cells.get(SOURCE_MODEL_ID_COLUMN):
        out[SOURCE_MODEL_ID_COLUMN] = cells[SOURCE_MODEL_ID_COLUMN]
    return out


def _cluster_is_header_nonempty_only(cluster: dict[str, Any]) -> bool:
    """True when cluster is header-only: no filters, no keywords, no value_contains (step-1 style)."""
    filters = cluster.get("filters", []) if isinstance(cluster.get("filters"), list) else []
    filters = [f for f in filters if isinstance(f, dict)]
    if filters:
        return False
    related = cluster.get("related", []) if isinstance(cluster.get("related"), list) else []
    related = [x for x in related if isinstance(x, dict)]
    if not related:
        return False
    for it in related:
        if str(it.get("value_contains", "")).strip():
            return False
        kws = it.get("keywords")
        if isinstance(kws, str):
            kws = [kws]
        elif not isinstance(kws, list):
            kws = []
        if any(str(x).strip() for x in kws):
            return False
    return True


def _row_matches_header_nonempty_all(cells: dict[str, str], related: list[dict[str, Any]]) -> tuple[bool, int]:
    """Step 2: row counts iff every selected header has nonempty cell(s) (per schema grouping)."""
    allowed = set(QUERY_MAP_HEADERS)
    headers: list[str] = []
    for x in related:
        if not isinstance(x, dict):
            continue
        h = str(x.get("header", "")).strip()
        if h in allowed:
            headers.append(h)
    if not headers:
        return False, 0
    for h in headers:
        if not _header_non_empty_for_row(h, cells):
            return False, 0
    return True, len(headers)


def row_match_score(related: list[dict[str, Any]], cells: dict[str, str]) -> tuple[bool, int]:
    allowed = set(QUERY_MAP_HEADERS)
    items = [x for x in related if isinstance(x, dict) and str(x.get("header", "")).strip() in allowed]
    if not items:
        return False, 0
    matched_any = False
    score = 0
    for item in items:
        h = str(item.get("header", "")).strip()
        kws = item.get("keywords")
        if isinstance(kws, str):
            kws = [kws]
        elif not isinstance(kws, list):
            kws = []
        keywords = [str(x).strip() for x in kws if str(x).strip()]
        if h in MODEL_ROW_COLUMNS:
            cell_candidates = [cells.get(k, "") for k in MODEL_ROW_COLUMNS]
        elif h in METRIC_ROW_COLUMNS:
            # Metric rows are only meaningful when the value exists.
            cell_candidates = [cells.get("Metric_value", "")]
        elif h in ("Dataset",):
            cell_candidates = [cells.get(k, "") for k in DATASET_ROW_COLUMNS]
        else:
            cell_candidates = [cells.get(h, "")]
        non_empty_cells = [c for c in cell_candidates if c]
        if not non_empty_cells:
            continue
        matched_any = True
        if not keywords:
            score += 1
            continue
        matched_keywords = sum(
            1 for kw in keywords if any(kw.lower() in cell_text.lower() for cell_text in non_empty_cells)
        )
        score += max(1, matched_keywords)
    return matched_any, score


def _haystack_for_filter_column(column: str, cells: dict[str, str]) -> str:
    if column in MODEL_ROW_COLUMNS:
        return " ".join(cells.get(k, "") for k in sorted(MODEL_ROW_COLUMNS))
    if column in METRIC_ROW_COLUMNS:
        return cells.get("Metric_value", "")
    if column in DATASET_ROW_COLUMNS:
        return " ".join(cells.get(k, "") for k in sorted(DATASET_ROW_COLUMNS))
    return cells.get(column, "")


def _filters_all_match(cells: dict[str, str], filters: list[dict[str, str]]) -> bool:
    for flt in filters:
        col = flt.get("column", "")
        needle = (flt.get("contains") or "").lower()
        if not needle:
            continue
        hay = _haystack_for_filter_column(col, cells).lower()
        if needle not in hay:
            return False
    return True


def _row_matches_cluster_structured(cells: dict[str, str], cluster: dict[str, Any]) -> tuple[bool, int]:
    filters = cluster.get("filters", []) if isinstance(cluster.get("filters"), list) else []
    filters = [f for f in filters if isinstance(f, dict)]
    related = cluster.get("related", []) if isinstance(cluster.get("related"), list) else []
    related = [x for x in related if isinstance(x, dict)]
    if filters and not _filters_all_match(cells, filters):
        return False, 0
    if not related:
        return True, 10 * len(filters) + 1
    if _cluster_is_header_nonempty_only(cluster):
        ok, sc = _row_matches_header_nonempty_all(cells, related)
        if not ok:
            return False, 0
        return True, sc + 10 * len(filters)
    ok, sc = row_match_score(related, cells)
    if not ok:
        return False, 0
    return True, sc + 10 * len(filters)


def _clusters_effective(block: dict[str, Any]) -> list[dict[str, Any]]:
    raw = block.get("clusters")
    if isinstance(raw, list) and raw:
        out: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            rel_raw = item.get("related", [])
            if not isinstance(rel_raw, list):
                rel_raw = []
            rel_n = _normalize_related_entries(rel_raw)
            fils: list[dict[str, str]] = []
            if isinstance(item.get("filters"), list):
                for f in item["filters"]:
                    if isinstance(f, dict):
                        nf = _normalize_filter_dict(f)
                        if nf:
                            fils.append(nf)
            for it in rel_n:
                vc = str(it.get("value_contains", "")).strip()
                h = str(it.get("header", "")).strip()
                if vc and h in NUGGET_SCHEMA_HEADERS:
                    fils.append({"column": h, "contains": vc})
            rel_clean = [{"header": x["header"], "keywords": x.get("keywords", [])} for x in rel_n]
            if rel_clean or fils:
                out.append({"related": rel_clean, "filters": fils})
        if out:
            return out
    rel_raw = block.get("related", [])
    if not isinstance(rel_raw, list):
        rel_raw = []
    rel_n = _normalize_related_entries(rel_raw)
    top_f: list[dict[str, str]] = []
    if isinstance(block.get("filters"), list):
        for f in block["filters"]:
            if isinstance(f, dict):
                nf = _normalize_filter_dict(f)
                if nf:
                    top_f.append(nf)
    one = _cluster_from_related_and_filters(rel_n, top_f)
    if one["related"] or one["filters"]:
        return [one]
    return []


def _header_non_empty_for_row(header: str, cells: dict[str, str]) -> bool:
    if header in MODEL_ROW_COLUMNS:
        return any(cells.get(k) for k in MODEL_ROW_COLUMNS)
    if header in METRIC_ROW_COLUMNS:
        return bool(cells.get("Metric_value", ""))
    if header in ("Dataset",):
        return any(cells.get(k) for k in DATASET_ROW_COLUMNS)
    return bool(cells.get(header, ""))


def _collect_llm_rerank_candidates(
    header_list: list[str],
    csv_paths: list[Path],
    *,
    max_rows: int = 150,
) -> tuple[list[dict[str, Any]], list[str]]:
    candidates: list[dict[str, Any]] = []
    csv_errors: list[str] = []
    headers_use = [h for h in header_list if h in QUERY_MAP_HEADERS]
    if not headers_use:
        headers_use = list(QUERY_MAP_HEADERS)
    for csv_path in csv_paths:
        table = csv_path.stem
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for ri, row in enumerate(reader):
                    if len(candidates) >= max_rows:
                        return candidates, csv_errors
                    cells = _row_dict(row)
                    if not any(_header_non_empty_for_row(h, cells) for h in headers_use):
                        continue
                    candidates.append({"table": table, "row_idx": ri, "cells": _cells_with_provenance(cells)})
        except OSError as e:
            csv_errors.append(f"{csv_path}: {e}")
    return candidates, csv_errors


def _llm_pick_rows_with_evidence(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    header_values: list[dict[str, str]],
    model: str | None,
) -> tuple[list[dict[str, Any]], str]:
    if not candidates:
        return [], ""
    if not os.getenv("OPENAI_API_KEY"):
        return [], "skip_no_api_key"
    lines = []
    for i, c in enumerate(candidates):
        lines.append(f"{i}\ttable={c['table']}\trow_idx={c['row_idx']}\tjson={json.dumps(c['cells'], ensure_ascii=False)}")
    body = _build_filter_prompt(query, header_values, lines)
    setup_openai("", mode="openai")
    raw = query_openai(body, mode="openai", model=model or TEXT_MODEL, max_tokens=1200) or ""
    parsed = _extract_json_object(raw)
    if not parsed:
        return [], "llm_parse_failed"
    picked = parsed.get("picked")
    if not isinstance(picked, list):
        return [], "llm_bad_shape"
    out: list[dict[str, Any]] = []
    for p in picked:
        if not isinstance(p, dict):
            continue
        t = str(p.get("table", "")).strip()
        try:
            ri = int(p.get("row_idx"))
        except (TypeError, ValueError):
            continue
        if not t:
            continue
        evidence: list[dict[str, Any]] = []
        raw_evidence = p.get("supporting_evidence", [])
        if isinstance(raw_evidence, list):
            for e in raw_evidence:
                if not isinstance(e, dict):
                    continue
                et = str(e.get("table", "")).strip()
                try:
                    eri = int(e.get("row_idx"))
                except (TypeError, ValueError):
                    continue
                if et:
                    evidence.append({"table": et, "row_idx": eri, "why": str(e.get("why", "")).strip()})
        out.append({"table": t, "row_idx": ri, "supporting_evidence": evidence})
    return out, ""


def _norm_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _number_with_unit(text: str) -> float | None:
    m = re.search(r"[-+]?(?:\d+\.?\d*|\d*\.?\d+)", text or "")
    if not m:
        return None
    value = float(m.group(0))
    suffix = (text or "")[m.end() : m.end() + 1].lower()
    if suffix == "b":
        value *= 1_000_000_000
    elif suffix == "m":
        value *= 1_000_000
    elif suffix == "k":
        value *= 1_000
    return value


def _is_less_than_one_b(text: str) -> bool:
    norm = re.sub(r"\s+", "", (text or "").lower())
    return any(
        token in norm
        for token in (
            "<1b",
            "<=1b",
            "lessthan1b",
            "lessthan1b",
            "under1b",
            "fewerthan1b",
            "parametercount<1b",
            "paramcount<1b",
        )
    )


def _bit_constraint_value(text: str) -> str:
    m = re.search(r"(\d+)\s*-?\s*bit", text or "", flags=re.IGNORECASE)
    return m.group(1) if m else ""


def _metric_constraints_for_answer(header_values: list[dict[str, str]]) -> list[str]:
    metric_names = [
        str(x.get("contains", "")).strip()
        for x in header_values
        if str(x.get("header", "")).strip() == "Metric_name" and str(x.get("contains", "")).strip()
    ]
    if len(metric_names) <= 1:
        return metric_names
    support_metrics = {"quantization_bits", "quantizationbits", "bitwidth", "bits", "precision"}
    answer_metrics = [m for m in metric_names if _norm_for_match(m) not in support_metrics]
    return answer_metrics or metric_names


def _answer_metrics_are_support_scalars(answer_metric_constraints: list[str]) -> bool:
    support_metrics = {"quantization_bits", "quantizationbits", "bitwidth", "bits", "precision"}
    return bool(answer_metric_constraints) and all(_norm_for_match(x) in support_metrics for x in answer_metric_constraints)


def _answer_does_not_contradict_constraints(cells: dict[str, str], header_values: list[dict[str, str]]) -> bool:
    answer_metric_constraints = _metric_constraints_for_answer(header_values)
    for cons in header_values:
        header = str(cons.get("header", "")).strip()
        needle = str(cons.get("contains", "")).strip()
        needle_norm = _norm_for_match(needle)
        if not header or not needle_norm:
            continue
        if header == "Dataset" and " " not in needle.strip():
            if needle_norm not in _norm_for_match(cells.get("Dataset", "")):
                return False
        if header == "Metric_name":
            if needle not in answer_metric_constraints:
                continue
            metric_norm = _norm_for_match(cells.get("Metric_name", ""))
            if needle_norm in {"performance", "score", "scores"}:
                if not cells.get("Metric_value"):
                    return False
            elif needle_norm not in metric_norm:
                return False
        if header == "Metric_value" and _is_less_than_one_b(needle):
            metric_norm = _norm_for_match(cells.get("Metric_name", ""))
            value = _number_with_unit(cells.get("Metric_value", ""))
            if "parameter" in metric_norm and value is not None and value >= 1_000_000_000:
                return False
        if header == "Metric_value" and _bit_constraint_value(needle) and _answer_metrics_are_support_scalars(answer_metric_constraints):
            bit_value = _bit_constraint_value(needle)
            value_norm = _norm_for_match(cells.get("Metric_value", ""))
            if value_norm not in {bit_value, f"{bit_value}bit"}:
                return False
    return True


def _has_parameter_threshold(header_values: list[dict[str, str]]) -> bool:
    has_parameter = any(
        str(x.get("header", "")).strip() == "Metric_name" and "parameter" in _norm_for_match(str(x.get("contains", "")))
        for x in header_values
    )
    has_threshold = any(
        str(x.get("header", "")).strip() == "Metric_value" and _is_less_than_one_b(str(x.get("contains", "")))
        for x in header_values
    )
    return has_parameter and has_threshold


def _relaxed_unknown_parameter_rows(candidates: list[dict[str, Any]], header_values: list[dict[str, str]]) -> list[dict[str, Any]]:
    if not _has_parameter_threshold(header_values):
        return []
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for c in candidates:
        model = str(c.get("cells", {}).get("Model", "")).strip()
        if model:
            rows_by_model.setdefault(model, []).append(c)
    out: list[dict[str, Any]] = []
    for model, rows in rows_by_model.items():
        contradicted = False
        chosen = rows[0]
        for row in rows:
            cells = row.get("cells", {})
            metric_norm = _norm_for_match(str(cells.get("Metric_name", "")))
            if "parameter" not in metric_norm:
                continue
            value = _number_with_unit(str(cells.get("Metric_value", "")))
            if value is not None and value >= 1_000_000_000:
                contradicted = True
                break
            chosen = row
        if not contradicted:
            out.append(chosen)
    return out


def get_subset_rows(
    query: str,
    query_block: dict[str, Any],
    candidate_hits: list[dict[str, Any]],
    *,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Optional LLM stage after step-2 (header-nonempty) matching.

    Use when the query needs a *subset* of rows with explicit facets — e.g. accuracy on a
    named dataset, or a specific metric value — not for broad task intents.

    Planned: prompt the model to filter ``candidate_hits`` to rows that satisfy the subset;
    return a reordered list. **Not invoked** from the default pipeline; wire from
    ``build_qrels_and_run_structured`` when ready.
    """
    _ = (query, query_block, model)
    return list(candidate_hits)


def build_qrels_and_run_structured(
    queries: list[dict[str, Any]],
    csv_paths: list[Path],
    *,
    subtopic: str = "1",
    model: str | None = None,
    match_log_md: Path | None = None,
    match_log_section: str | None = None,
    emit_match_report: bool = True,
) -> tuple[list[tuple[str, str, str, int]], list[tuple[str, str, str, int, float, str]], list[dict[str, Any]]]:
    qrels: list[tuple[str, str, str, int]] = []
    run_rows: list[tuple[str, str, str, int, float, str]] = []
    debug: list[dict[str, Any]] = []
    csv_inventory: list[tuple[Path, int]] = [(Path(p).resolve(), count_csv_data_rows(Path(p))) for p in csv_paths]
    per_query_rows: list[dict[str, Any]] = []
    for qi, block in enumerate(queries):
        qid = f"q{qi:04d}"
        qtext = str(block.get("query", "")).strip()
        if block.get("error"):
            debug.append({"qid": qid, "query": qtext, "skipped": str(block.get("error")), "hits": []})
            per_query_rows.append({"qid": qid, "matched": None, "note": f"query_map error: {block.get('error')}"})
            continue
        clusters = _clusters_effective(block)
        if not clusters:
            debug.append({"qid": qid, "query": qtext, "skipped": "no_clusters_or_filters", "hits": []})
            per_query_rows.append({"qid": qid, "matched": 0, "note": "no_clusters_or_filters"})
            continue
        hits: list[dict[str, Any]] = []
        csv_errors: list[str] = []
        for csv_path in csv_paths:
            table = csv_path.stem
            try:
                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for ri, row in enumerate(reader):
                        cells = _row_dict(row)
                        best_sc = 0
                        matched = False
                        for cluster in clusters:
                            ok, sc = _row_matches_cluster_structured(cells, cluster)
                            if ok:
                                matched = True
                                if sc > best_sc:
                                    best_sc = sc
                        if not matched:
                            continue
                        doc_id = f"{table}#{ri}"
                        hits.append({"doc_id": doc_id, "table": table, "row_idx": ri, "score": best_sc, "cells": _cells_with_provenance(cells)})
            except OSError as e:
                csv_errors.append(f"{csv_path}: {e}")
        hits.sort(key=lambda x: (-x["score"], x["doc_id"]))
        n_note = "ok" if not csv_errors else f"ok; csv_errors={len(csv_errors)}"
        per_query_rows.append({"qid": qid, "matched": len(hits), "note": n_note})
        # Optional step 3 — LLM get_subset: narrow rows when the query requires a concrete subset
        # (e.g. metric accuracy on a named dataset). Uncomment when implemented / desired.
        # hits = get_subset_rows(qtext, block, hits, model=model)
        for h in hits:
            qrels.append((qid, subtopic, h["doc_id"], 1))
        for rank, h in enumerate(hits, start=1):
            run_rows.append((qid, "Q0", h["doc_id"], rank, float(max(1, h["score"])), "match"))
        entry: dict[str, Any] = {"qid": qid, "query": qtext, "match_build": "structured", "hits": hits, "clusters": clusters}
        if csv_errors:
            entry["csv_errors"] = csv_errors
        debug.append(entry)
    run_rows.sort(key=lambda x: (x[0], x[3]))
    if emit_match_report:
        md = format_query2nugget_match_markdown(
            mode="structured",
            csv_inventory=csv_inventory,
            per_query=per_query_rows,
            qrels_lines=len(qrels),
            run_lines=len(run_rows),
        )
        emit_query2nugget_match_report(
            md,
            match_log_md=match_log_md,
            section_heading=match_log_section or "## query2nugget · structured",
            to_stdout=match_log_md is None,
        )
    return qrels, run_rows, debug


def build_qrels_and_run_llm_rerank(
    queries: list[dict[str, Any]],
    csv_paths: list[Path],
    *,
    subtopic: str = "1",
    model: str | None = None,
    match_log_md: Path | None = None,
    match_log_section: str | None = None,
    emit_match_report: bool = True,
) -> tuple[list[tuple[str, str, str, int]], list[tuple[str, str, str, int, float, str]], list[dict[str, Any]]]:
    qrels: list[tuple[str, str, str, int]] = []
    run_rows: list[tuple[str, str, str, int, float, str]] = []
    debug: list[dict[str, Any]] = []
    csv_inventory: list[tuple[Path, int]] = [(Path(p).resolve(), count_csv_data_rows(Path(p))) for p in csv_paths]
    per_query_rows: list[dict[str, Any]] = []
    for qi, block in enumerate(queries):
        qid = f"q{qi:04d}"
        qtext = str(block.get("query", "")).strip()
        if block.get("error"):
            debug.append({"qid": qid, "query": qtext, "skipped": str(block.get("error")), "hits": []})
            per_query_rows.append(
                {"qid": qid, "hits": "-", "candidates": "-", "rerank_note": "", "note": f"query_map error: {block.get('error')}"}
            )
            continue
        hl = block.get("header_list", [])
        if not isinstance(hl, list):
            hl = []
        header_list = [str(x).strip() for x in hl if str(x).strip()]
        clusters = _clusters_effective(block)
        if not header_list and not clusters:
            debug.append({"qid": qid, "query": qtext, "skipped": "no_header_list_or_clusters", "hits": []})
            per_query_rows.append({"qid": qid, "hits": 0, "candidates": 0, "rerank_note": "", "note": "no_header_list_or_clusters"})
            continue
        hl_fb: list[str] = []
        for c in clusters:
            for x in c.get("related", []):
                if isinstance(x, dict) and str(x.get("header", "")).strip():
                    hl_fb.append(str(x["header"]).strip())
        header_values_raw = block.get("header_values", [])
        header_values = header_values_raw if isinstance(header_values_raw, list) else []
        candidates, csv_errors = _collect_llm_rerank_candidates([], csv_paths)
        unfiltered_candidate_count = len(candidates)
        picked_rows, err_note = _llm_pick_rows_with_evidence(qtext, candidates, header_values=header_values, model=model)
        cand_key = {(c["table"], c["row_idx"]): c for c in candidates}
        hits: list[dict[str, Any]] = []
        valid_rank = 0
        for picked in picked_rows:
            table = str(picked.get("table", "")).strip()
            ri = picked.get("row_idx")
            key = (table, ri)
            if key not in cand_key:
                continue
            c = cand_key[key]
            if not _answer_does_not_contradict_constraints(c["cells"], header_values):
                continue
            valid_rank += 1
            doc_id = f"{table}#{ri}"
            support: list[dict[str, Any]] = []
            for e in picked.get("supporting_evidence", []):
                ekey = (str(e.get("table", "")).strip(), e.get("row_idx"))
                if ekey == key:
                    continue
                ec = cand_key.get(ekey)
                if not ec:
                    continue
                support.append(
                    {
                        "doc_id": f"{ekey[0]}#{ekey[1]}",
                        "table": ekey[0],
                        "row_idx": ekey[1],
                        "why": str(e.get("why", "")).strip(),
                        "cells": ec["cells"],
                    }
                )
            hits.append(
                {
                    "doc_id": doc_id,
                    "table": table,
                    "row_idx": ri,
                    "score": max(1, 200 - valid_rank),
                    "cells": c["cells"],
                    "supporting_evidence": support,
                }
            )
        picked_models = {str(h.get("cells", {}).get("Model", "")).strip() for h in hits}
        hit_keys = {(h["table"], h["row_idx"]) for h in hits}
        if picked_models:
            for c in candidates:
                key = (c["table"], c["row_idx"])
                if key in hit_keys:
                    continue
                if str(c.get("cells", {}).get("Model", "")).strip() not in picked_models:
                    continue
                if not _answer_does_not_contradict_constraints(c["cells"], header_values):
                    continue
                valid_rank += 1
                doc_id = f"{c['table']}#{c['row_idx']}"
                hits.append(
                    {
                        "doc_id": doc_id,
                        "table": c["table"],
                        "row_idx": c["row_idx"],
                        "score": max(1, 200 - valid_rank),
                        "cells": c["cells"],
                        "supporting_evidence": [],
                    }
                )
                hit_keys.add(key)
        if not hits:
            for c in _relaxed_unknown_parameter_rows(candidates, header_values):
                valid_rank += 1
                doc_id = f"{c['table']}#{c['row_idx']}"
                hits.append(
                    {
                        "doc_id": doc_id,
                        "table": c["table"],
                        "row_idx": c["row_idx"],
                        "score": max(1, 200 - valid_rank),
                        "cells": c["cells"],
                        "supporting_evidence": [],
                    }
                )
        for h in hits:
            qrels.append((qid, subtopic, h["doc_id"], 1))
        for rank, h in enumerate(hits, start=1):
            run_rows.append((qid, "Q0", h["doc_id"], rank, float(max(1, h["score"])), "llm_rerank"))
        entry: dict[str, Any] = {
            "qid": qid,
            "query": qtext,
            "match_build": "llm_rerank",
            "hits": hits,
            "clusters": clusters,
            "header_values": header_values,
            "llm_rerank_candidates": len(candidates),
            "llm_rerank_candidates_before_constraints": unfiltered_candidate_count,
            "llm_rerank_prefilter": "none_all_candidate_rows",
            "llm_rerank_note": err_note or "ok",
        }
        if csv_errors:
            entry["csv_errors"] = csv_errors
        debug.append(entry)
        per_query_rows.append(
            {
                "qid": qid,
                "hits": len(hits),
                "candidates": len(candidates),
                "rerank_note": err_note or "ok",
                "note": "ok" if not csv_errors else f"csv_errors={len(csv_errors)}",
            }
        )
    run_rows.sort(key=lambda x: (x[0], x[3]))
    if emit_match_report:
        md = format_query2nugget_match_markdown(
            mode="llm_rerank",
            csv_inventory=csv_inventory,
            per_query=per_query_rows,
            qrels_lines=len(qrels),
            run_lines=len(run_rows),
        )
        emit_query2nugget_match_report(
            md,
            match_log_md=match_log_md,
            section_heading=match_log_section or "## query2nugget · llm_rerank",
            to_stdout=match_log_md is None,
        )
    return qrels, run_rows, debug


def _extract_text_from_batch_line(item: dict[str, Any]) -> tuple[str, str]:
    resp = item.get("response") or {}
    body = resp.get("body") if isinstance(resp, dict) else {}
    if isinstance(body, dict):
        choices = body.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            content = message.get("content", "")
            if isinstance(content, str):
                return content, ""
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(str(c.get("text", "")))
                return "\n".join(parts), ""
    err = item.get("error")
    return "", (str(err).strip() if err else "") or "batch_output_parse_error"


def _query2nugget_skip_no_api_key_rows(clean: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for q in clean:
        try:
            prompt = _build_prompt_for_query(q)
        except RuntimeError as e:
            prompt = ""
            out.append({"query": q, "related": [], "header_list": [], "error": str(e), "prompt": prompt, "raw_response": ""})
            continue
        out.append({"query": q, "related": [], "header_list": [], "error": "skip_no_api_key", "prompt": prompt, "raw_response": ""})
    return out


def _query2nugget_batch_responses(prompts: list[str], *, model: str | None) -> tuple[dict[int, tuple[str, str]], Path, Path]:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    in_path = BATCH_DIR / f"query2nugget_batch_input_{ts}.jsonl"
    out_path = BATCH_DIR / f"query2nugget_batch_output_{ts}.jsonl"
    model_name = model or TEXT_MODEL
    with open(in_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            body: dict[str, Any] = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 800,
            }
            payload = {
                "custom_id": f"{i:06d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    main_batch_query(str(in_path), str(out_path))
    by_idx: dict[int, tuple[str, str]] = {}
    if out_path.is_file():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = str(obj.get("custom_id", "")).strip()
                if not cid.isdigit():
                    continue
                idx = int(cid)
                text, note = _extract_text_from_batch_line(obj)
                by_idx[idx] = (text, note)
    return by_idx, in_path, out_path


def _query2nugget_iter_responses(prompts: list[str], *, model: str | None) -> dict[int, tuple[str, str]]:
    by_idx: dict[int, tuple[str, str]] = {}
    setup_openai("", mode="openai")
    for i, prompt in enumerate(prompts):
        try:
            raw = query_openai(prompt, mode="openai", model=model or TEXT_MODEL, max_tokens=800) or ""
            by_idx[i] = (raw, "")
        except Exception as e:
            by_idx[i] = ("", str(e))
    return by_idx


def _assemble_query2nugget_results(clean: list[str], prompt_by_i: dict[int, str], error_by_i: dict[int, dict[str, Any]], response_by_sub: dict[int, tuple[str, str]], ok_order: list[int], *, batch_paths: tuple[Path, Path] | None) -> list[dict[str, Any]]:
    by_orig: dict[int, tuple[str, str]] = {}
    for k, orig_i in enumerate(ok_order):
        if k in response_by_sub:
            by_orig[orig_i] = response_by_sub[k]
    missing_key = "batch_missing_line" if batch_paths else "iter_missing_line"
    results: list[dict[str, Any]] = []
    for i, q in enumerate(clean):
        if i in error_by_i:
            results.append(error_by_i[i])
            continue
        prompt = prompt_by_i[i]
        if i not in by_orig:
            results.append({"query": q, "related": [], "header_list": [], "error": missing_key, "prompt": prompt, "raw_response": ""})
            continue
        raw, err_note = by_orig[i]
        if err_note:
            results.append({"query": q, "related": [], "header_list": [], "error": err_note, "prompt": prompt, "raw_response": raw})
            continue
        if not (raw or "").strip():
            results.append({"query": q, "related": [], "header_list": [], "error": "batch_empty_response", "prompt": prompt, "raw_response": raw})
            continue
        row = _finalize_map_response(q, prompt, raw)
        if batch_paths:
            row["batch_input_jsonl"] = str(batch_paths[0].resolve())
            row["batch_output_jsonl"] = str(batch_paths[1].resolve())
        results.append(row)
    return results


def map_queries(
    queries: list[str],
    *,
    model: str | None = None,
    llm_mode: Literal["batch", "iter"] = "batch",
) -> list[dict[str, Any]]:
    clean = [(q or "").strip() for q in queries if (q or "").strip()]
    if not clean:
        return []
    if not os.getenv("OPENAI_API_KEY"):
        return _query2nugget_skip_no_api_key_rows(clean)

    error_by_i: dict[int, dict[str, Any]] = {}
    prompt_by_i: dict[int, str] = {}
    ok_order: list[int] = []
    ok_prompts: list[str] = []
    for i, q in enumerate(clean):
        try:
            p = _build_prompt_for_query(q)
            prompt_by_i[i] = p
            ok_order.append(i)
            ok_prompts.append(p)
        except RuntimeError as e:
            error_by_i[i] = {"query": q, "related": [], "header_list": [], "error": str(e), "prompt": "", "raw_response": ""}

    if not ok_prompts:
        return [error_by_i[i] for i in sorted(error_by_i)]

    batch_paths: tuple[Path, Path] | None
    if llm_mode == "iter":
        print("[query2nugget] llm_mode=iter (sync chat per query)")
        response_by_sub = _query2nugget_iter_responses(ok_prompts, model=model)
        batch_paths = None
    else:
        print("[query2nugget] llm_mode=batch (OpenAI Batch API)")
        by_idx, in_path, out_path = _query2nugget_batch_responses(ok_prompts, model=model)
        response_by_sub = by_idx
        batch_paths = (in_path, out_path)

    return _assemble_query2nugget_results(clean, prompt_by_i, error_by_i, response_by_sub, ok_order, batch_paths=batch_paths)


def _save_qrels_and_run(
    *,
    mapping_results: list[dict[str, Any]],
    csv_roots: list[Path],
    qrels_path: Path,
    run_path: Path,
    debug_path: Path,
    subtopic: str,
    match_build: Literal["structured", "llm_rerank"] = "structured",
    rerank_model: str | None = None,
) -> tuple[int, int]:
    csv_paths = discover_csv_paths(csv_roots)
    if not csv_paths:
        raise RuntimeError(f"No CSV files found under: {[str(x) for x in csv_roots]}")
    builder = build_qrels_and_run_llm_rerank if match_build == "llm_rerank" else build_qrels_and_run_structured
    qrels_rows, run_rows, debug = builder(mapping_results, csv_paths, subtopic=subtopic, model=rerank_model)

    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    with open(qrels_path, "w", encoding="utf-8") as f:
        for qid, st, doc_id, rel in qrels_rows:
            f.write(f"{qid} {st} {doc_id} {rel}\n")
    with open(run_path, "w", encoding="utf-8") as f:
        for qid, q0, doc_id, rank, score, tag in run_rows:
            f.write(f"{qid} {q0} {doc_id} {rank} {score} {tag}\n")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump({"csv_paths": [str(p) for p in csv_paths], "queries": debug}, f, ensure_ascii=False, indent=2)

    return len(qrels_rows), len(run_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map a user query to nugget schema headers (LLM) with per-header keyword lists.",
    )
    parser.add_argument("--query", default=None, help="Single user search query.")
    parser.add_argument("--queries-file", default=None, help="Text file, one query per line.")
    parser.add_argument(
        "--output",
        default=OUTPUT_HEADER_KEYWORD_JSON,
        help=f"JSON output path (default: {OUTPUT_HEADER_KEYWORD_JSON})",
    )
    parser.add_argument("--model", default=None, help="OpenAI chat model (default from env or gpt-5.4-mini).")
    parser.add_argument(
        "--build-qrels-run",
        action="store_true",
        help="Also build qrels/run by matching mapping JSON against card2nugget CSV files.",
    )
    parser.add_argument(
        "--csv-root",
        action="append",
        default=None,
        help="Directory to scan for card2nugget CSV files (repeatable). Default: data_*/card2nugget/.",
    )
    parser.add_argument("--qrels-output", default=OUTPUT_QRELS, help=f"Qrels output path (default: {OUTPUT_QRELS})")
    parser.add_argument("--run-output", default=OUTPUT_RUN, help=f"Run output path (default: {OUTPUT_RUN})")
    parser.add_argument("--match-debug-output", default=OUTPUT_MATCH_DEBUG_JSON, help=f"Debug JSON for CSV matching (default: {OUTPUT_MATCH_DEBUG_JSON})")
    parser.add_argument("--subtopic", default="1", help="Subtopic id written into qrels (default: 1).")
    parser.add_argument("--llm-mode", choices=["batch", "iter"], default="batch", help="batch=OpenAI Batch API; iter=sync chat per query.")
    parser.add_argument(
        "--match-build",
        choices=["structured", "llm_rerank"],
        default="structured",
        help="How to build qrels/run: structured=header-nonempty row match; llm_rerank=LLM picks row_idx from candidate rows (uses OPENAI_API_KEY).",
    )
    args = parser.parse_args()

    queries: list[str] = []
    if args.query:
        queries.append(args.query.strip())
    if args.queries_file:
        p = Path(args.queries_file)
        if not p.is_file():
            parser.error(f"--queries-file not found: {p}")
        with open(p, encoding="utf-8") as f:
            queries.extend(line.strip() for line in f if line.strip())
    queries = [q for q in queries if q]
    if not queries:
        parser.error("Provide --query and/or --queries-file.")

    results = map_queries(queries, model=args.model, llm_mode=args.llm_mode)

    payload: dict[str, Any] = {"queries": results} if len(results) > 1 else results[0]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"saved = {out_path.resolve()}")
    for r in results:
        err = r.get("error", "")
        hl = r.get("header_list", [])
        print(f"query = {r.get('query', '')!r}  |  headers = {hl}  |  note = {err or 'ok'}")

    # Step 3 get_subset_rows (LLM) is not wired here; it stays commented inside build_qrels_and_run_structured.
    if args.build_qrels_run:
        roots = [Path(p) for p in args.csv_root] if args.csv_root else [CARD2NUGGET_DIR]
        qrels_lines, run_lines = _save_qrels_and_run(
            mapping_results=results,
            csv_roots=roots,
            qrels_path=Path(args.qrels_output),
            run_path=Path(args.run_output),
            debug_path=Path(args.match_debug_output),
            subtopic=str(args.subtopic),
            match_build=args.match_build,
            rerank_model=args.model,
        )
        print(f"saved_qrels = {Path(args.qrels_output).resolve()}  | lines = {qrels_lines}")
        print(f"saved_run = {Path(args.run_output).resolve()}  | lines = {run_lines}")
        print(f"saved_match_debug = {Path(args.match_debug_output).resolve()}")


if __name__ == "__main__":
    main()
