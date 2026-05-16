#!/usr/bin/env python3
"""CSV matching, LLM row filtering, and qrels/run output for query2nugget."""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from src.evaluate.nugget_schema import NUGGET_SCHEMA_HEADERS
from src.evaluate.query2nugget_prompting import (
    QUERY_MAP_HEADERS,
    TEXT_MODEL,
    build_filter_prompt,
    cluster_from_related_and_filters,
    extract_json_object,
    normalize_filter_dict,
    normalize_related_entries,
)
from src.llm.model import query_openai, setup_openai

# Job-merged integrated CSV (wrap_card_query_eval) adds provenance; not used for header matching.
SOURCE_MODEL_ID_COLUMN = "source_model_id"
MODEL_ROW_COLUMNS = frozenset({"Model", "Base_model", "Model_variant_type"})
DATASET_ROW_COLUMNS = frozenset({"Dataset"})
METRIC_ROW_COLUMNS = frozenset({"Metric_name", "Metric_value"})


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
            reader = csv.reader(f)
            if next(reader, None) is None:
                return 0
            return sum(1 for _ in reader)
    except OSError:
        return 0


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
            matched = row.get("matched")
            matched_text = f"{matched}" if matched is not None else "-"
            lines.append(f"| `{_md_cell(qid)}` | {matched_text} | {note} |")
    else:
        lines.extend(["| qid | hits | cand | rerank_note | note |", "| --- | ---: | ---: | --- | --- |"])
        for row in per_query:
            qid = str(row.get("qid", ""))
            note = _md_cell(str(row.get("note", "")))
            hits = row.get("hits", "-")
            candidates = row.get("candidates", "-")
            rerank_note = _md_cell(str(row.get("rerank_note", ""))[:48])
            lines.append(f"| `{_md_cell(qid)}` | {hits} | {candidates} | {rerank_note} | {note} |")
    lines.extend(["", f"**TOTALS:** `qrels_lines={qrels_lines}` · `run_lines={run_lines}`", ""])
    return "\n".join(lines)


def emit_query2nugget_match_report(
    md_body: str,
    *,
    match_log_md: Path | None,
    section_heading: str,
    to_stdout: bool,
) -> None:
    """Append markdown to file and/or print."""
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
    source_model = raw.get(SOURCE_MODEL_ID_COLUMN) or raw.get("From_model") or raw.get("from_model")
    if source_model:
        out[SOURCE_MODEL_ID_COLUMN] = _norm_cell(source_model)
    return out


def _cells_with_provenance(cells: dict[str, str]) -> dict[str, str]:
    """Schema cells plus source_model_id when present."""
    out = {h: cells[h] for h in NUGGET_SCHEMA_HEADERS if cells.get(h)}
    if cells.get(SOURCE_MODEL_ID_COLUMN):
        out[SOURCE_MODEL_ID_COLUMN] = cells[SOURCE_MODEL_ID_COLUMN]
    return out


def _cluster_is_header_nonempty_only(cluster: dict[str, Any]) -> bool:
    """True when cluster is header-only: no filters, keywords, or value constraints."""
    filters = cluster.get("filters", []) if isinstance(cluster.get("filters"), list) else []
    filters = [f for f in filters if isinstance(f, dict)]
    if filters:
        return False
    related = cluster.get("related", []) if isinstance(cluster.get("related"), list) else []
    related = [x for x in related if isinstance(x, dict)]
    if not related:
        return False
    for item in related:
        if str(item.get("value_contains", "")).strip():
            return False
        keywords = item.get("keywords")
        if isinstance(keywords, str):
            keywords = [keywords]
        elif not isinstance(keywords, list):
            keywords = []
        if any(str(x).strip() for x in keywords):
            return False
    return True


def _row_matches_header_nonempty_all(cells: dict[str, str], related: list[dict[str, Any]]) -> tuple[bool, int]:
    """Step 2: row counts iff every selected header has nonempty cell(s)."""
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
        keywords = item.get("keywords")
        if isinstance(keywords, str):
            keywords = [keywords]
        elif not isinstance(keywords, list):
            keywords = []
        keywords = [str(x).strip() for x in keywords if str(x).strip()]
        if h in MODEL_ROW_COLUMNS:
            cell_candidates = [cells.get(k, "") for k in MODEL_ROW_COLUMNS]
        elif h in METRIC_ROW_COLUMNS:
            cell_candidates = [cells.get("Metric_value", "")]
        elif h in DATASET_ROW_COLUMNS:
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
        haystack = _haystack_for_filter_column(col, cells).lower()
        if needle not in haystack:
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
        ok, score = _row_matches_header_nonempty_all(cells, related)
        if not ok:
            return False, 0
        return True, score + 10 * len(filters)
    ok, score = row_match_score(related, cells)
    if not ok:
        return False, 0
    return True, score + 10 * len(filters)


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
            related = normalize_related_entries(rel_raw)
            filters: list[dict[str, str]] = []
            if isinstance(item.get("filters"), list):
                for f in item["filters"]:
                    if isinstance(f, dict):
                        normalized = normalize_filter_dict(f)
                        if normalized:
                            filters.append(normalized)
            for it in related:
                value_contains = str(it.get("value_contains", "")).strip()
                header = str(it.get("header", "")).strip()
                if value_contains and header in NUGGET_SCHEMA_HEADERS:
                    filters.append({"column": header, "contains": value_contains})
            rel_clean = [{"header": x["header"], "keywords": x.get("keywords", [])} for x in related]
            if rel_clean or filters:
                out.append({"related": rel_clean, "filters": filters})
        if out:
            return out
    rel_raw = block.get("related", [])
    if not isinstance(rel_raw, list):
        rel_raw = []
    related = normalize_related_entries(rel_raw)
    top_filters: list[dict[str, str]] = []
    if isinstance(block.get("filters"), list):
        for f in block["filters"]:
            if isinstance(f, dict):
                normalized = normalize_filter_dict(f)
                if normalized:
                    top_filters.append(normalized)
    one = cluster_from_related_and_filters(related, top_filters)
    if one["related"] or one["filters"]:
        return [one]
    return []


def _header_non_empty_for_row(header: str, cells: dict[str, str]) -> bool:
    if header in MODEL_ROW_COLUMNS:
        return any(cells.get(k) for k in MODEL_ROW_COLUMNS)
    if header in METRIC_ROW_COLUMNS:
        return bool(cells.get("Metric_value", ""))
    if header in DATASET_ROW_COLUMNS:
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
                for row_idx, row in enumerate(reader):
                    if len(candidates) >= max_rows:
                        return candidates, csv_errors
                    cells = _row_dict(row)
                    if not any(_header_non_empty_for_row(h, cells) for h in headers_use):
                        continue
                    candidates.append({"table": table, "row_idx": row_idx, "cells": _cells_with_provenance(cells)})
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
    lines = [
        f"{i}\ttable={c['table']}\trow_idx={c['row_idx']}\tjson={json.dumps(c['cells'], ensure_ascii=False)}"
        for i, c in enumerate(candidates)
    ]
    body = build_filter_prompt(query, header_values, lines)
    setup_openai("", mode="openai")
    raw = query_openai(body, mode="openai", model=model or TEXT_MODEL, max_tokens=1200) or ""
    parsed = extract_json_object(raw)
    if not parsed:
        return [], "llm_parse_failed"
    picked = parsed.get("picked")
    if not isinstance(picked, list):
        return [], "llm_bad_shape"
    out: list[dict[str, Any]] = []
    for p in picked:
        if not isinstance(p, dict):
            continue
        table = str(p.get("table", "")).strip()
        try:
            row_idx = int(p.get("row_idx"))
        except (TypeError, ValueError):
            continue
        if not table:
            continue
        evidence: list[dict[str, Any]] = []
        raw_evidence = p.get("supporting_evidence", [])
        if isinstance(raw_evidence, list):
            for e in raw_evidence:
                if not isinstance(e, dict):
                    continue
                evidence_table = str(e.get("table", "")).strip()
                try:
                    evidence_row_idx = int(e.get("row_idx"))
                except (TypeError, ValueError):
                    continue
                if evidence_table:
                    evidence.append(
                        {"table": evidence_table, "row_idx": evidence_row_idx, "why": str(e.get("why", "")).strip()}
                    )
        out.append({"table": table, "row_idx": row_idx, "supporting_evidence": evidence})
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
    support_metrics = {"quantization_bits", "quantizationbits", "bitwidth", "bits", "precision"}
    answer_metrics = [m for m in metric_names if _norm_for_match(m) not in support_metrics]
    return answer_metrics


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
    for rows in rows_by_model.values():
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
    _ = model
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
                    for row_idx, row in enumerate(reader):
                        cells = _row_dict(row)
                        best_score = 0
                        matched = False
                        for cluster in clusters:
                            ok, score = _row_matches_cluster_structured(cells, cluster)
                            if ok:
                                matched = True
                                best_score = max(best_score, score)
                        if not matched:
                            continue
                        doc_id = f"{table}#{row_idx}"
                        hits.append(
                            {
                                "doc_id": doc_id,
                                "table": table,
                                "row_idx": row_idx,
                                "score": best_score,
                                "cells": _cells_with_provenance(cells),
                            }
                        )
            except OSError as e:
                csv_errors.append(f"{csv_path}: {e}")
        hits.sort(key=lambda x: (-x["score"], x["doc_id"]))
        per_query_rows.append({"qid": qid, "matched": len(hits), "note": "ok" if not csv_errors else f"ok; csv_errors={len(csv_errors)}"})
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
        header_list = [str(x).strip() for x in hl if str(x).strip()] if isinstance(hl, list) else []
        clusters = _clusters_effective(block)
        if not header_list and not clusters:
            debug.append({"qid": qid, "query": qtext, "skipped": "no_header_list_or_clusters", "hits": []})
            per_query_rows.append({"qid": qid, "hits": 0, "candidates": 0, "rerank_note": "", "note": "no_header_list_or_clusters"})
            continue

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
            row_idx = picked.get("row_idx")
            key = (table, row_idx)
            if key not in cand_key:
                continue
            candidate = cand_key[key]
            if not _answer_does_not_contradict_constraints(candidate["cells"], header_values):
                continue
            valid_rank += 1
            support: list[dict[str, Any]] = []
            for e in picked.get("supporting_evidence", []):
                ekey = (str(e.get("table", "")).strip(), e.get("row_idx"))
                if ekey == key:
                    continue
                evidence_candidate = cand_key.get(ekey)
                if not evidence_candidate:
                    continue
                support.append(
                    {
                        "doc_id": f"{ekey[0]}#{ekey[1]}",
                        "table": ekey[0],
                        "row_idx": ekey[1],
                        "why": str(e.get("why", "")).strip(),
                        "cells": evidence_candidate["cells"],
                    }
                )
            hits.append(
                {
                    "doc_id": f"{table}#{row_idx}",
                    "table": table,
                    "row_idx": row_idx,
                    "score": max(1, 200 - valid_rank),
                    "cells": candidate["cells"],
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
                hits.append(
                    {
                        "doc_id": f"{c['table']}#{c['row_idx']}",
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
                hits.append(
                    {
                        "doc_id": f"{c['table']}#{c['row_idx']}",
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


def save_qrels_and_run(
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
        for qid, subtopic_id, doc_id, rel in qrels_rows:
            f.write(f"{qid} {subtopic_id} {doc_id} {rel}\n")
    with open(run_path, "w", encoding="utf-8") as f:
        for qid, q0, doc_id, rank, score, tag in run_rows:
            f.write(f"{qid} {q0} {doc_id} {rank} {score} {tag}\n")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump({"csv_paths": [str(p) for p in csv_paths], "queries": debug}, f, ensure_ascii=False, indent=2)

    return len(qrels_rows), len(run_rows)
