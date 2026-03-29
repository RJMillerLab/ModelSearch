#!/usr/bin/env python3
"""
Generate markdown to inspect table integration outputs under the current job format.

What this script focuses on now:
1. Rebuild the same retrieval relationship summaries shown in the UI:
   - Query2Card summary
   - Query2Tab2Card summary
2. Print concrete tables for visual comparison:
   - query table
   - retrieved tables
   - integrated tables
3. Compute deterministic comparisons to help judge whether integration is reasonable:
   - query table vs integrated table
   - retrieved tables union schema vs integrated table
   - Query2Card integrated table vs Query2Tab2Card integrated table

The script is intentionally stdlib-first so it can still run in light environments.
If `duckdb` is available, Query2Card can also resolve model -> table mappings from the
flattened parquet index. Without it, the script will still print model IDs and all table
comparison sections.

Example:
python src/utils/check_retrieval_integration_consistency.py \
  --jobs-root jobs_251117 \
  --per-job-md \
  --preview-max-rows 0 --preview-max-cols 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import MODEL_TO_TABLES_EXPLODE_PARQUET, QUERY2MODELCARD_RETRIEVAL_MODES, TABLE_BASE_DIRS


DEFAULT_SEARCH_TYPES = ["single_column", "unionable", "keyword"]
DEFAULT_INTEGRATION_TYPE = "alite"
DEFAULT_JOB_MARKDOWN_FILENAME = "integration_review.md"


@dataclass
class TableData:
    path: str
    source_path: str
    display_name: str
    columns: List[str]
    rows: List[List[str]]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def col_count(self) -> int:
        return len(self.columns)

    @property
    def exists(self) -> bool:
        return bool(self.source_path)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_json_if_exists(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    return _load_json(path)


def _unique_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        s = str(item).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _basename(path: str) -> str:
    return os.path.basename(str(path).strip())


def _md_escape(value: Any) -> str:
    s = str(value)
    s = s.replace("\r", " ").replace("\n", " ")
    return s.replace("|", "\\|")


def _csv_cell_cleanup(value: str) -> str:
    text = str(value)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _normalize_colname(value: str) -> str:
    s = str(value or "").strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _resolve_table_path(path_or_name: str) -> str:
    raw = str(path_or_name or "").strip()
    if not raw:
        return ""
    if os.path.isfile(raw):
        return os.path.abspath(raw)
    bn = _basename(raw)
    for base in TABLE_BASE_DIRS:
        candidate = os.path.join(str(base), bn)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return ""


def _read_csv_table(path_or_name: str) -> TableData:
    resolved = _resolve_table_path(path_or_name) if not os.path.isfile(str(path_or_name)) else os.path.abspath(str(path_or_name))
    source = resolved or ""
    display_name = _basename(path_or_name) or _basename(resolved) or str(path_or_name).strip() or "(missing)"
    if not source or not os.path.isfile(source):
        return TableData(path=str(path_or_name or ""), source_path="", display_name=display_name, columns=[], rows=[])

    with open(source, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return TableData(path=str(path_or_name or ""), source_path=source, display_name=display_name, columns=[], rows=[])

    columns = [_csv_cell_cleanup(x) for x in rows[0]]
    body = [[_csv_cell_cleanup(x) for x in row] for row in rows[1:]]
    return TableData(path=str(path_or_name or ""), source_path=source, display_name=display_name, columns=columns, rows=body)


def _table_to_markdown(table: TableData, *, max_rows: Optional[int], max_cols: Optional[int], max_cell: int = 80) -> str:
    if not table.columns:
        return "_(table missing or empty)_"

    col_end = max_cols if isinstance(max_cols, int) and max_cols > 0 else len(table.columns)
    row_end = max_rows if isinstance(max_rows, int) and max_rows > 0 else len(table.rows)
    columns = table.columns[:col_end]
    lines = [
        "| " + " | ".join(_md_escape(c) for c in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in table.rows[:row_end]:
        vals: List[str] = []
        for cell in row[:col_end]:
            text = _md_escape(cell)
            if len(text) > max_cell:
                text = text[: max_cell - 3] + "..."
            vals.append(text)
        if len(vals) < len(columns):
            vals.extend("" for _ in range(len(columns) - len(vals)))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _table_to_html(table: TableData, *, max_rows: Optional[int], max_cols: Optional[int], max_cell: int = 80) -> str:
    if not table.columns:
        return "<em>(table missing or empty)</em>"

    col_end = max_cols if isinstance(max_cols, int) and max_cols > 0 else len(table.columns)
    row_end = max_rows if isinstance(max_rows, int) and max_rows > 0 else len(table.rows)
    columns = table.columns[:col_end]
    header_html = "".join(
        f"<th style=\"border:1px solid #d0d7de;padding:6px 8px;background:#f6f8fa;text-align:left;white-space:nowrap;\">{_md_escape(c)}</th>"
        for c in columns
    )
    body_rows: List[str] = []
    for row in table.rows[:row_end]:
        vals: List[str] = []
        for cell in row[:col_end]:
            text = _md_escape(cell)
            if len(text) > max_cell:
                text = text[: max_cell - 3] + "..."
            vals.append(text)
        if len(vals) < len(columns):
            vals.extend("" for _ in range(len(columns) - len(vals)))
        cells_html = "".join(
            f"<td style=\"border:1px solid #d0d7de;padding:6px 8px;vertical-align:top;white-space:nowrap;\">{v}</td>"
            for v in vals
        )
        body_rows.append(f"<tr>{cells_html}</tr>")
    return (
        "<div style=\"overflow:auto;max-width:100%;\">"
        "<table style=\"border-collapse:collapse;font-size:12px;background:#fff;\">"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _model_link_md(model_id: str) -> str:
    mid = str(model_id).strip()
    if not mid:
        return "`(missing)`"
    return f"[`{_md_escape(mid)}`](https://huggingface.co/{mid})"


def _table_label_md(path_or_name: str) -> str:
    raw = str(path_or_name).strip()
    if not raw:
        return "`(missing)`"
    resolved = _resolve_table_path(raw) if not os.path.isabs(raw) else raw
    label = _basename(raw) or raw
    if resolved and os.path.isfile(resolved):
        return f"[`{_md_escape(label)}`]({resolved})"
    return f"`{_md_escape(label)}`"


def _file_link_md(path_or_name: str, label: Optional[str] = None) -> str:
    raw = str(path_or_name).strip()
    if not raw:
        return "`(missing)`"
    resolved = raw if os.path.isabs(raw) and os.path.isfile(raw) else _resolve_table_path(raw)
    shown = label or _basename(raw) or raw
    if resolved and os.path.isfile(resolved):
        return f"[`{_md_escape(shown)}`]({resolved})"
    return f"`{_md_escape(shown)}`"


def _slug_anchor(value: str) -> str:
    s = str(value or "").strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = s.replace("_", "-").replace(" ", "-")
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _duckdb_sql_string_literal(value: Any) -> str:
    s = str(value)
    return "'" + s.replace("'", "''") + "'"


def _compare_columns(left: TableData, right: TableData) -> Dict[str, Any]:
    left_norm_to_raw: Dict[str, str] = {}
    right_norm_to_raw: Dict[str, str] = {}
    for col in left.columns:
        norm = _normalize_colname(col)
        if norm and norm not in left_norm_to_raw:
            left_norm_to_raw[norm] = col
    for col in right.columns:
        norm = _normalize_colname(col)
        if norm and norm not in right_norm_to_raw:
            right_norm_to_raw[norm] = col

    cols_left = set(left_norm_to_raw.keys())
    cols_right = set(right_norm_to_raw.keys())
    overlap_norm = sorted(cols_left & cols_right)
    only_left_norm = sorted(cols_left - cols_right)
    only_right_norm = sorted(cols_right - cols_left)
    overlap = [left_norm_to_raw[n] for n in overlap_norm]
    only_left = [left_norm_to_raw[n] for n in only_left_norm]
    only_right = [right_norm_to_raw[n] for n in only_right_norm]
    union_size = len(cols_left | cols_right)
    jaccard = (len(overlap) / union_size) if union_size else 0.0
    left_containment = (len(overlap) / len(cols_left)) if cols_left else 0.0
    right_containment = (len(overlap) / len(cols_right)) if cols_right else 0.0
    return {
        "left_rows": left.row_count,
        "left_cols": left.col_count,
        "right_rows": right.row_count,
        "right_cols": right.col_count,
        "overlap": overlap,
        "only_left": only_left,
        "only_right": only_right,
        "jaccard": jaccard,
        "left_containment": left_containment,
        "right_containment": right_containment,
    }


def _heuristic_note(stats: Dict[str, Any], *, left_name: str, right_name: str) -> str:
    jaccard = float(stats["jaccard"])
    left_containment = float(stats["left_containment"])
    if stats["left_cols"] == 0 or stats["right_cols"] == 0:
        return f"{left_name} or {right_name} is empty, so this comparison is not informative yet."
    if jaccard >= 0.6:
        return f"{left_name} and {right_name} are fairly aligned at the schema level."
    if left_containment >= 0.6:
        return f"{right_name} covers most columns from {left_name}, but it also introduces noticeable extra schema."
    if jaccard <= 0.2:
        return f"{left_name} and {right_name} look quite different in schema; integration reasonableness needs manual inspection."
    return f"{left_name} and {right_name} partially overlap, but neither clearly subsumes the other."


def _render_compare_block(title: str, stats: Dict[str, Any], *, left_name: str, right_name: str) -> List[str]:
    lines = [f"### {title}", ""]
    lines.append(f"- `{left_name}`: {stats['left_rows']} rows x {stats['left_cols']} cols")
    lines.append(f"- `{right_name}`: {stats['right_rows']} rows x {stats['right_cols']} cols")
    lines.append(f"- Column overlap: `{len(stats['overlap'])}`")
    lines.append(f"- Column Jaccard: `{stats['jaccard']:.3f}`")
    lines.append(f"- Containment `{left_name} -> {right_name}`: `{stats['left_containment']:.3f}`")
    lines.append(f"- Containment `{right_name} -> {left_name}`: `{stats['right_containment']:.3f}`")
    if stats["only_left"]:
        lines.append(f"- Only in `{left_name}`: {', '.join(f'`{_md_escape(x)}`' for x in stats['only_left'][:10])}{' ...' if len(stats['only_left']) > 10 else ''}")
    if stats["only_right"]:
        lines.append(f"- Only in `{right_name}`: {', '.join(f'`{_md_escape(x)}`' for x in stats['only_right'][:10])}{' ...' if len(stats['only_right']) > 10 else ''}")
    lines.append(f"- Read: {_heuristic_note(stats, left_name=left_name, right_name=right_name)}")
    lines.append("")
    return lines


def _iter_job_dirs(jobs_root: str) -> List[str]:
    if not os.path.isdir(jobs_root):
        return []
    return [
        os.path.join(jobs_root, name)
        for name in sorted(os.listdir(jobs_root))
        if os.path.isdir(os.path.join(jobs_root, name))
        and name not in {"job_md", "batch_runs"}
        and os.path.isfile(os.path.join(jobs_root, name, "job_meta.json"))
    ]


def _resolve_existing_file(candidates: Sequence[str]) -> str:
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return ""


def _discover_search_types(job_dir: str) -> List[str]:
    found: List[str] = []
    if not os.path.isdir(job_dir):
        return list(DEFAULT_SEARCH_TYPES)
    for name in sorted(os.listdir(job_dir)):
        m = re.match(r"^card2tab2card_(.+)\.json$", name)
        if m:
            found.append(m.group(1))
    return found or list(DEFAULT_SEARCH_TYPES)


def _query2card_seed_and_neighbors(payload: Dict[str, Any], mode: str, max_models: Optional[int]) -> Tuple[str, List[str]]:
    results = payload.get("results", {})
    dense = results.get("dense", []) if isinstance(results, dict) else []
    seed = str(dense[0]).strip() if dense else ""
    mode_items = results.get(mode, []) if isinstance(results, dict) else []
    neighbors: List[str] = []
    for item in mode_items:
        mid = str(item).strip()
        if not mid or mid == seed:
            continue
        neighbors.append(mid)
    out = _unique_keep_order(neighbors)
    if max_models is None:
        return seed, out
    return seed, out[: max(0, int(max_models))]


def _lookup_model_to_tables_duckdb(model_ids: Sequence[str], *, resources: Optional[Sequence[str]] = None) -> Tuple[Dict[str, List[str]], str]:
    mids = [str(x).strip() for x in model_ids if str(x).strip()]
    if not mids:
        return {}, "no model ids"
    if not os.path.isfile(MODEL_TO_TABLES_EXPLODE_PARQUET):
        return {}, f"missing explode parquet: {MODEL_TO_TABLES_EXPLODE_PARQUET}"
    try:
        import duckdb  # type: ignore
    except Exception as exc:
        return {}, f"duckdb unavailable: {type(exc).__name__}"

    conn = duckdb.connect(":memory:")
    try:
        values_sql = ", ".join("(" + _duckdb_sql_string_literal(mid) + ")" for mid in mids)
        where_parts = [f"e.modelId = m.modelId"]
        if resources:
            allowed = [str(x).strip() for x in resources if str(x).strip()]
            if allowed:
                res_sql = ", ".join(_duckdb_sql_string_literal(x) for x in allowed)
                where_parts.append(f"e.resource IN ({res_sql})")
        query = f"""
            WITH mids(modelId) AS (
                VALUES {values_sql}
            )
            SELECT
                m.modelId,
                e.csv_basename
            FROM mids AS m
            LEFT JOIN read_parquet('{MODEL_TO_TABLES_EXPLODE_PARQUET}') AS e
              ON {' AND '.join(where_parts)}
            ORDER BY m.modelId, e.csv_basename
        """
        rows = conn.execute(query).fetchall()
    finally:
        conn.close()

    out: Dict[str, List[str]] = {mid: [] for mid in mids}
    for model_id, csv_basename in rows:
        mid = str(model_id).strip()
        tb = str(csv_basename).strip() if csv_basename is not None else ""
        if tb and tb not in out[mid]:
            out[mid].append(tb)
    out = {mid: tables for mid, tables in out.items() if tables}
    return out, "duckdb parquet lookup"


def _build_query2card_preview(job_dir: str, *, mode: str, max_models: Optional[int]) -> Dict[str, Any]:
    path = os.path.join(job_dir, "query2modelcard.json")
    data = _load_json(path)
    seed_model_id, model_ids = _query2card_seed_and_neighbors(data, mode, max_models=max_models)
    model_to_tables, source_note = _lookup_model_to_tables_duckdb(model_ids, resources=["hugging"])
    models_with_tables = [mid for mid in model_ids if mid in model_to_tables]
    table_paths = _unique_keep_order(path for mid in models_with_tables for path in model_to_tables.get(mid, []))
    return {
        "query2modelcard_retrieval_mode": mode,
        "seed_model_id": seed_model_id,
        "model_ids": model_ids,
        "models_with_tables": models_with_tables,
        "model_to_table_paths": model_to_tables,
        "table_paths": table_paths,
        "stats": {
            "total_model_ids": len(model_ids),
            "models_with_tables": len(models_with_tables),
            "total_unique_tables": len(table_paths),
        },
        "notes": [source_note] if source_note else [],
    }


def _build_query2tab2card_preview(job_dir: str, search_type: str, *, max_models: Optional[int]) -> Dict[str, Any]:
    path = os.path.join(job_dir, f"card2tab2card_{search_type}.json")
    data = _load_json(path)
    q2c = data.get("query2card_map", {}) or {}
    card2tab = data.get("card2tab_map", {}) or {}
    tab2tab = data.get("tab2tab_map", {}) or {}
    tab2card = data.get("tab2card_map", {}) or {}
    reranked = list(data.get("model_rerank_map", []) or [])

    query = next(iter(q2c.keys()), "")
    seed_models = [str(x).strip() for x in q2c.get(query, []) if str(x).strip()]
    seed_model = seed_models[0] if seed_models else ""
    query_tables = [str(x).strip() for x in card2tab.get(seed_model, []) if str(x).strip()]

    model_ids = [str(x).strip() for x in reranked if str(x).strip()]
    if max_models is not None:
        model_ids = model_ids[: max(0, int(max_models))]
    allowed = set(model_ids)

    retrieved_rows: List[Dict[str, Any]] = []
    for _query_table, retrieved_tables in tab2tab.items():
        for table_path in retrieved_tables or []:
            tpath = str(table_path).strip()
            models = [str(x).strip() for x in tab2card.get(tpath, []) if str(x).strip() and str(x).strip() in allowed]
            if not models:
                continue
            retrieved_rows.append(
                {
                    "table": _basename(tpath),
                    "table_path": tpath,
                    "models": models,
                }
            )

    model_to_table_paths: Dict[str, List[str]] = {}
    for row in retrieved_rows:
        for mid in row["models"]:
            model_to_table_paths.setdefault(mid, []).append(row["table_path"])
    model_to_table_paths = {
        mid: _unique_keep_order(paths)
        for mid, paths in model_to_table_paths.items()
        if paths
    }

    table_paths = _unique_keep_order(row["table_path"] for row in retrieved_rows)
    models_with_tables = [mid for mid in model_ids if mid in model_to_table_paths]

    return {
        "search_type": search_type,
        "query": query,
        "seed_model_id": seed_model,
        "query_tables": query_tables,
        "model_ids": model_ids,
        "models_with_tables": models_with_tables,
        "model_to_table_paths": model_to_table_paths,
        "table_paths": table_paths,
        "retrieved_table_model_rows": retrieved_rows,
        "stats": {
            "models_with_tables": len(models_with_tables),
            "total_unique_tables": len(table_paths),
        },
    }


def _render_query2card_summary(preview: Dict[str, Any], *, heading_level: int = 2) -> List[str]:
    stats = preview.get("stats", {}) or {}
    model_ids = preview.get("model_ids", []) or []
    model_to_tables = preview.get("model_to_table_paths", {}) or {}
    heading = "#" * heading_level

    lines = [f"{heading} Query2Card Summary ({preview.get('query2modelcard_retrieval_mode', '')})", ""]
    lines.append(f"- Seed model id: `{_md_escape(preview.get('seed_model_id', ''))}`")
    lines.append(f"- UI summary header: `Model IDs (1 Query2Card Results)` ({stats.get('models_with_tables', 0)} models, {stats.get('total_unique_tables', 0)} tables)")
    for note in preview.get("notes", []) or []:
        lines.append(f"- Note: {note}")
    lines.append("")

    if not model_ids:
        lines.append("_(no Query2Card model ids found)_")
        lines.append("")
        return lines

    lines.append(f"{heading}# UI-style Lines")
    lines.append("")
    for mid in model_ids:
        tables = model_to_tables.get(mid, [])
        if tables:
            table_frag = ", ".join(_table_label_md(t) for t in tables)
        else:
            table_frag = "_(table mapping unavailable)_"
        lines.append(f"- Model id: {_model_link_md(mid)} -> Related table: {table_frag}")
    lines.append("")
    return lines


def _render_query2tab2card_summary(preview: Dict[str, Any]) -> List[str]:
    stats = preview.get("stats", {}) or {}
    lines = [f"## Query2Tab2Card Summary ({preview.get('search_type', '')})", ""]
    lines.append(f"- Seed model id: `{_md_escape(preview.get('seed_model_id', ''))}`")
    lines.append(f"- UI summary header: `Model IDs (2 Query2Tab2Card Results)` ({stats.get('models_with_tables', 0)} models, {stats.get('total_unique_tables', 0)} tables)")
    lines.append("")
    lines.append("### UI-style Lines")
    lines.append("")

    query_tables = preview.get("query_tables", []) or []
    if query_tables:
        lines.append(f"- Query table(s): {', '.join(_table_label_md(t) for t in query_tables)}")
    else:
        lines.append("- Query table(s): _(missing)_")

    rows = preview.get("retrieved_table_model_rows", []) or []
    if not rows:
        lines.append("- Retrieved table list: _(empty)_")
        lines.append("")
        return lines

    for idx, row in enumerate(rows, start=1):
        models = row.get("models", []) or []
        model_frag = ", ".join(_model_link_md(mid) for mid in models) if models else "_(none)_"
        table_name = row.get("table_path") or row.get("table") or ""
        lines.append(f"- Retrieved table {idx}: {_table_label_md(table_name)} -> related models: {model_frag}")
    lines.append("")
    return lines


def _load_integrated_table(job_dir: str, filename: str) -> TableData:
    path = os.path.join(job_dir, filename)
    if not os.path.isfile(path):
        return TableData(path=path, source_path="", display_name=filename, columns=[], rows=[])
    return _read_csv_table(path)


def _load_integrated_table_candidates(job_dir: str, filenames: Sequence[str]) -> TableData:
    path = _resolve_existing_file(os.path.join(job_dir, name) for name in filenames)
    if not path:
        fallback = filenames[0] if filenames else "(missing)"
        return TableData(path=os.path.join(job_dir, fallback), source_path="", display_name=fallback, columns=[], rows=[])
    return _read_csv_table(path)


def _load_model_integration_payload(job_dir: str, integration_type: str, retrieval_mode: str) -> Dict[str, Any]:
    filename = f"integration_model_search_{integration_type}_{retrieval_mode}.json"
    return _load_json_if_exists(os.path.join(job_dir, filename))


def _load_table_integration_payload(job_dir: str, integration_type: str, search_type: str, tables_source: str = "intermediate") -> Dict[str, Any]:
    candidates = [
        f"integration_table_search_{integration_type}_{search_type}_{tables_source}.json",
        f"integration_table_search_{integration_type}_{search_type}.json",
    ]
    path = _resolve_existing_file(os.path.join(job_dir, name) for name in candidates)
    return _load_json_if_exists(path) if path else {}


def _render_table_section(title: str, table: TableData, *, max_rows: Optional[int], max_cols: Optional[int]) -> List[str]:
    lines = [f"### {title}", ""]
    lines.append(f"- Source: {_file_link_md(table.source_path or table.path or '(missing)', label=table.display_name or 'table')}")
    lines.append(f"- Shape: `{table.row_count} x {table.col_count}`")
    lines.append("")
    lines.append(_table_to_markdown(table, max_rows=max_rows, max_cols=max_cols))
    lines.append("")
    return lines


def _render_retrieved_tables_section(title: str, tables: List[TableData], *, max_rows: Optional[int], max_cols: Optional[int]) -> List[str]:
    lines = [f"### {title}", ""]
    if not tables:
        lines.append("_(no tables)_")
        lines.append("")
        return lines
    for idx, table in enumerate(tables, start=1):
        lines.append(f"#### {idx}. `{_md_escape(table.display_name)}`")
        lines.append("")
        lines.append(f"- Source: {_file_link_md(table.source_path or table.path or '(missing)', label=table.display_name or 'table')}")
        lines.append(f"- Shape: `{table.row_count} x {table.col_count}`")
        lines.append("")
        lines.append(_table_to_markdown(table, max_rows=max_rows, max_cols=max_cols))
        lines.append("")
    return lines


def _render_horizontal_review_section(
    title: str,
    source_tables: List[TableData],
    integrated_table: TableData,
    *,
    max_rows: Optional[int],
    max_cols: Optional[int],
    source_labels: Optional[Sequence[str]] = None,
    source_label: str = "Input table",
) -> List[str]:
    lines = [f"### {title}", ""]
    cells: List[str] = []
    for idx, table in enumerate(source_tables, start=1):
        if source_labels and idx - 1 < len(source_labels) and str(source_labels[idx - 1]).strip():
            label = str(source_labels[idx - 1]).strip()
        else:
            label = f"{source_label} {idx}"
        source_html = _file_link_md(table.source_path or table.path or "(missing)", label=table.display_name or "table")
        shape = f"{table.row_count} x {table.col_count}"
        content = [
            f"<div style=\"font-weight:600;margin-bottom:4px;\">{label}</div>",
            f"<div style=\"font-size:12px;color:#57606a;margin-bottom:4px;\">Source: {source_html}</div>",
            f"<div style=\"font-size:12px;color:#57606a;margin-bottom:8px;\">Shape: <code>{shape}</code></div>",
            _table_to_html(table, max_rows=max_rows, max_cols=max_cols),
        ]
        cells.append(
            "<td style=\"vertical-align:top;padding:10px;border:1px solid #d0d7de;\">"
            + "".join(content)
            + "</td>"
        )
    if not cells:
        cells.append("<td style=\"vertical-align:top;padding:10px;border:1px solid #d0d7de;\"><em>(no input tables)</em></td>")

    integrated_source_html = _file_link_md(
        integrated_table.source_path or integrated_table.path or "(missing)",
        label=integrated_table.display_name or "integrated",
    )
    integrated_shape = f"{integrated_table.row_count} x {integrated_table.col_count}"
    colspan = len(cells)
    lines.append("<table style=\"border-collapse:collapse;width:100%;margin:8px 0 16px 0;\">")
    lines.append("<tbody>")
    lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append(
        "<tr>"
        f"<td colspan=\"{colspan}\" style=\"vertical-align:top;padding:10px;border:1px solid #d0d7de;background:#f6f8fa;\">"
        "<div style=\"font-weight:600;margin-bottom:4px;\">Integrated Result</div>"
        f"<div style=\"font-size:12px;color:#57606a;margin-bottom:4px;\">Source: {integrated_source_html}</div>"
        f"<div style=\"font-size:12px;color:#57606a;margin-bottom:8px;\">Shape: <code>{integrated_shape}</code></div>"
        f"{_table_to_html(integrated_table, max_rows=max_rows, max_cols=max_cols)}"
        "</td>"
        "</tr>"
    )
    lines.append("</tbody>")
    lines.append("</table>")
    lines.append("")
    return lines


def _union_schema_table(name: str, tables: Sequence[TableData]) -> TableData:
    cols = _unique_keep_order(col for table in tables for col in table.columns if col)
    return TableData(path=name, source_path=name, display_name=name, columns=cols, rows=[])


def build_job_markdown(
    *,
    job_dir: str,
    search_types: Optional[Sequence[str]],
    integration_type: str,
    preview_max_rows: Optional[int],
    preview_max_cols: Optional[int],
) -> str:
    job_id = os.path.basename(os.path.normpath(job_dir))
    meta_path = os.path.join(job_dir, "job_meta.json")
    meta = _load_json(meta_path) if os.path.isfile(meta_path) else {}
    active_search_types = [str(x).strip() for x in (search_types or _discover_search_types(job_dir)) if str(x).strip()]

    lines: List[str] = []
    lines.append(f"# Job `{job_id}`")
    lines.append("")
    lines.append(f"- Job dir: `{job_dir}`")
    if meta:
        lines.append(f"- Query: `{_md_escape(meta.get('query', ''))}`")
        lines.append(f"- model_top_k: `{meta.get('model_top_k', '')}`")
        lines.append(f"- table_search_k: `{meta.get('table_search_k', '')}`")
        lines.append(f"- Timestamp: `{meta.get('timestamp', '')}`")
    lines.append(f"- Search types: `{', '.join(active_search_types)}`")
    lines.append(f"- Query2Card modes: `{', '.join(QUERY2MODELCARD_RETRIEVAL_MODES)}`")
    lines.append(f"- Integration type for CSV loading only: `{integration_type}`")
    lines.append("")

    lines.append("## Contents")
    lines.append("")
    for mode in QUERY2MODELCARD_RETRIEVAL_MODES:
        lines.append(f"- [Query2Card {mode}](#query2card-summary-{_slug_anchor(mode)})")
    for search_type in active_search_types:
        anchor = _slug_anchor(search_type)
        lines.append(f"- [Query2Tab2Card {search_type}](#query2tab2card-summary-{anchor})")
        lines.append(f"- [Comparison {search_type}](#comparison-{anchor})")
    lines.append("")

    q2c_max_models = None
    if meta and meta.get("model_top_k") is not None:
        try:
            q2c_max_models = int(meta.get("model_top_k"))
        except Exception:
            q2c_max_models = None

    q2c_previews_by_mode = {
        mode: _build_query2card_preview(job_dir, mode=mode, max_models=q2c_max_models)
        for mode in QUERY2MODELCARD_RETRIEVAL_MODES
    }
    q2c_integrated_by_mode: Dict[str, TableData] = {}

    for mode in QUERY2MODELCARD_RETRIEVAL_MODES:
        preview = q2c_previews_by_mode[mode]
        lines.extend(_render_query2card_summary(preview, heading_level=3))
        integration_payload = _load_model_integration_payload(job_dir, integration_type, mode)
        input_table_paths = integration_payload.get("table_paths") or preview.get("table_paths", []) or []
        input_tables = [_read_csv_table(p) for p in input_table_paths]
        integrated_table = _load_integrated_table_candidates(
            job_dir,
            [
                f"integrated_model_search_{integration_type}_{mode}.csv",
                f"integrated_model_search_{integration_type}.csv" if mode == "dense" else "",
            ],
        )
        q2c_integrated_by_mode[mode] = integrated_table
        lines.extend(
            _render_horizontal_review_section(
                f"Query2Card Integration Review ({mode})",
                input_tables,
                integrated_table,
                max_rows=preview_max_rows,
                max_cols=preview_max_cols,
                source_label="Related table",
            )
        )

    q2c_compare_reference = q2c_integrated_by_mode.get("dense")
    if not q2c_compare_reference or not q2c_compare_reference.columns:
        q2c_compare_reference = next((tbl for tbl in q2c_integrated_by_mode.values() if tbl.columns), TableData("", "", "(missing)", [], []))

    for search_type in active_search_types:
        c2t2c_path = os.path.join(job_dir, f"card2tab2card_{search_type}.json")
        if not os.path.isfile(c2t2c_path):
            lines.append(f"## Query2Tab2Card Summary ({search_type})")
            lines.append("")
            lines.append("_(missing card2tab2card json)_")
            lines.append("")
            continue

        preview = _build_query2tab2card_preview(job_dir, search_type, max_models=None)
        lines.extend(_render_query2tab2card_summary(preview))

        query_tables = [_read_csv_table(p) for p in preview.get("query_tables", [])]
        integration_payload = _load_table_integration_payload(job_dir, integration_type, search_type, "intermediate")
        retrieved_table_paths = integration_payload.get("table_paths") or preview.get("table_paths", []) or []
        retrieved_tables = [_read_csv_table(p) for p in retrieved_table_paths]
        integration_input_paths = integration_payload.get("integration_input_table_paths") or []
        integration_input_tables: List[TableData] = []
        integration_input_labels: List[str] = []
        if integration_input_paths:
            normalized_query_paths = {str(p).strip() for p in (preview.get("query_tables", []) or []) if str(p).strip()}
            retrieved_idx = 0
            for path in integration_input_paths:
                path_s = str(path).strip()
                if not path_s:
                    continue
                integration_input_tables.append(_read_csv_table(path_s))
                if path_s in normalized_query_paths:
                    integration_input_labels.append("Query table")
                else:
                    retrieved_idx += 1
                    label_prefix = "Related table" if integration_payload.get("tables_source") == "all_from_modelcards" else "Retrieved table"
                    integration_input_labels.append(f"{label_prefix} {retrieved_idx}")
        else:
            integration_input_tables = list(query_tables) + list(retrieved_tables)
            label_prefix = "Related table" if integration_payload.get("tables_source") == "all_from_modelcards" else "Retrieved table"
            integration_input_labels = (["Query table"] if query_tables else []) + [f"{label_prefix} {idx}" for idx in range(1, len(retrieved_tables) + 1)]
        integrated_table = _load_integrated_table_candidates(
            job_dir,
            [
                f"integrated_table_search_{integration_type}_{search_type}_intermediate.csv",
                f"integrated_table_search_{integration_type}_{search_type}.csv",
            ],
        )

        if query_tables:
            lines.extend(_render_table_section(f"Query Table ({search_type})", query_tables[0], max_rows=preview_max_rows, max_cols=preview_max_cols))
        else:
            lines.extend(_render_table_section(f"Query Table ({search_type})", TableData("", "", "(missing)", [], []), max_rows=preview_max_rows, max_cols=preview_max_cols))
        lines.extend(
            _render_horizontal_review_section(
                f"Query2Tab2Card Integration Review ({search_type})",
                integration_input_tables,
                integrated_table,
                max_rows=preview_max_rows,
                max_cols=preview_max_cols,
                source_labels=integration_input_labels,
            )
        )

        lines.append(f"## Comparison ({search_type})")
        lines.append("")

        if query_tables:
            stats = _compare_columns(query_tables[0], integrated_table)
            lines.extend(_render_compare_block("Query Table vs Integrated Table", stats, left_name="query table", right_name="integrated"))

        if retrieved_tables:
            union_table = _union_schema_table(f"retrieved-union-{search_type}", retrieved_tables)
            stats = _compare_columns(union_table, integrated_table)
            lines.extend(_render_compare_block("Retrieved Tables Union Schema vs Integrated Table", stats, left_name="retrieved union", right_name="integrated"))

        if q2c_compare_reference.columns and integrated_table.columns:
            stats = _compare_columns(q2c_compare_reference, integrated_table)
            lines.extend(_render_compare_block("Query2Card Integrated vs Query2Tab2Card Integrated", stats, left_name="q2c integrated (dense if available)", right_name=f"{search_type} integrated"))

    return "\n".join(lines)


def write_job_markdown(
    *,
    job_dir: str,
    search_types: Optional[Sequence[str]],
    integration_type: str,
    preview_max_rows: Optional[int],
    preview_max_cols: Optional[int],
    output_path: Optional[str] = None,
) -> str:
    md = build_job_markdown(
        job_dir=job_dir,
        search_types=search_types,
        integration_type=integration_type,
        preview_max_rows=preview_max_rows,
        preview_max_cols=preview_max_cols,
    )
    final_output_path = output_path or os.path.join(job_dir, DEFAULT_JOB_MARKDOWN_FILENAME)
    os.makedirs(os.path.dirname(os.path.abspath(final_output_path)), exist_ok=True)
    with open(final_output_path, "w", encoding="utf-8") as f:
        f.write(md)
    return final_output_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate markdown for current job-format integration inspection.")
    parser.add_argument("--jobs-root", type=str, default="jobs_251117", help="Root directory containing job folders.")
    parser.add_argument("--job-dir", type=str, default=None, help="Inspect a single job directory.")
    parser.add_argument("--search-types", nargs="+", default=None, help="Optional override for Query2Tab2Card search types; by default auto-discover from card2tab2card_*.json.")
    parser.add_argument("--integration-type", type=str, default=DEFAULT_INTEGRATION_TYPE, help="Filename suffix for integrated CSVs, e.g. alite.")
    parser.add_argument("--preview-max-rows", type=int, default=10, help="Rows to print per table; <=0 means all.")
    parser.add_argument("--preview-max-cols", type=int, default=12, help="Columns to print per table; <=0 means all.")
    parser.add_argument("--per-job-md", action="store_true", help="Write one markdown file per job.")
    parser.add_argument("--per-job-md-dir", type=str, default=None, help="Output directory for per-job markdown; default <jobs-root>/job_md.")
    parser.add_argument("--markdown-out", type=str, default=None, help="Aggregate markdown output path when not using --per-job-md.")

    args = parser.parse_args(argv)
    search_types = [str(x).strip() for x in args.search_types if str(x).strip()] if args.search_types else None
    job_dirs = [args.job_dir] if args.job_dir else _iter_job_dirs(args.jobs_root)
    if not job_dirs:
        print(f"no job directories found under {args.jobs_root}", file=sys.stderr)
        return 2

    preview_max_rows = None if int(args.preview_max_rows) <= 0 else int(args.preview_max_rows)
    preview_max_cols = None if int(args.preview_max_cols) <= 0 else int(args.preview_max_cols)

    written_paths: List[Tuple[str, str]] = []
    for job_dir in job_dirs:
        job_id = os.path.basename(os.path.normpath(job_dir))
        if args.per_job_md:
            output_path = (
                os.path.join(args.per_job_md_dir, f"{job_id}.md")
                if args.per_job_md_dir
                else os.path.join(job_dir, DEFAULT_JOB_MARKDOWN_FILENAME)
            )
            written_path = write_job_markdown(
                job_dir=job_dir,
                search_types=search_types,
                integration_type=str(args.integration_type).strip(),
                preview_max_rows=preview_max_rows,
                preview_max_cols=preview_max_cols,
                output_path=output_path,
            )
            written_paths.append((job_id, written_path))
            print(f"[ok] generated markdown for {job_id}")
            print(f"  wrote {written_path}")
            continue

        md = build_job_markdown(
            job_dir=job_dir,
            search_types=search_types,
            integration_type=str(args.integration_type).strip(),
            preview_max_rows=preview_max_rows,
            preview_max_cols=preview_max_cols,
        )
        written_paths.append((job_id, md))
        print(f"[ok] generated markdown for {job_id}")

    if args.per_job_md:
        index_path = (
            os.path.join(args.per_job_md_dir, "README.md")
            if args.per_job_md_dir
            else os.path.join(args.jobs_root, "integration_review_index.md")
        )
        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        index_lines = [
            "# Job Markdown Index",
            "",
            f"- Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
            f"- Jobs root: `{args.jobs_root}`",
            "",
        ]
        for job_id, path in written_paths:
            rel_path = os.path.relpath(path, os.path.dirname(index_path))
            index_lines.append(f"- [{job_id}]({rel_path})")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("\n".join(index_lines))
        print(f"  wrote {index_path}")
        return 0

    out_path = args.markdown_out or os.path.join(args.jobs_root, "integration_reasonableness_report.md")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(md for _, md in written_paths))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
