#!/usr/bin/env python3
"""
Generate markdown files from all log files in logs/ directory.

**用途**：把搜索日志（card2card、tab2tab 等跑的 .log）转成 Markdown，从 log 里解析 "Results saved to xxx.json"，列出 models/tables，并可做 table integration。在 docs/build_index.md 里会用到。

Outputs:
- logs/ — input (search run logs).
- md/<log_basename>.md — one md per log.
- md/<log_basename>_materials/csv_integrated/integrated.csv — integrated table (full path in the md).

Usage (run from repo root):
  python -m src.postprocess.generate_md_from_logs --logs_dir logs --output_dir md
  python -m src.postprocess.generate_md_from_logs --log_file logs/card2tab2card_by_type.log --output_dir md
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any

import pandas as pd

from .pipeline import is_model_search_log
from .table_md_common import (
    resolve_path,
    load_classifications,
    find_csv_file,
    get_table_metadata,
    load_table_csv,
    table_to_markdown,
)


def _run_integration_and_save(
    log_name: str,
    result_data: Dict[str, Any],
    result_json_path: Optional[str],
    table_ids: List[int],
    db_path: str,
    output_dir: str,
    integration_type: str = "union",
    integration_k: int = 10,
) -> Optional[str]:
    """Run table integration when possible; save CSV to md/<log_name>_materials/csv_integrated/integrated.csv. Returns full path or None."""
    from src.integration.table_integration import integrate_tables

    out_dir = resolve_path(output_dir) / f"{log_name}_materials" / "csv_integrated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "integrated.csv"

    db_abs = str(resolve_path(db_path))

    # Card2tab2card-style: resolve filenames with same dirs as postprocess (avoids "Table not found" on server)
    intermediate = result_data.get("intermediate", {})
    retrieved = intermediate.get("retrieved_table_filenames") or []
    if retrieved:
        paths = []
        for fname in retrieved[:integration_k]:
            full = find_csv_file(fname)
            if full:
                paths.append(full)
        if paths:
            res = integrate_tables(paths, integration_type=integration_type, k=integration_k, db_path=db_abs)
        else:
            res = {}
    elif table_ids:
        # Build table paths from table_ids (tab2tab-style results)
        paths = []
        for tid in table_ids[:integration_k]:
            meta = get_table_metadata(tid, db_abs)
            if not meta:
                continue
            full = find_csv_file(meta["filename"])
            if full:
                paths.append(full)
        if not paths:
            return None
        res = integrate_tables(paths, integration_type=integration_type, k=integration_k, db_path=db_abs)
    else:
        return None

    if not res.get("success") or res.get("integrated_table") is None:
        return None
    df = res["integrated_table"]
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    out_csv = Path(out_csv).resolve()
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return str(out_csv)


# ---- extraction from log file ----

# Log basenames to skip (not search-result logs)
_SKIP_LOG_STEMS = frozenset({"backend", "frontend"})

# Fallback result JSON paths when log does not contain "Results saved to ..."
# (e.g. card2card/query2modelcard run without --output_json; multi_column may crash before save)
_LOG_DEFAULT_JSON = {
    "card2card_dense": "data/card2card_neighbors.json",
    "card2card_hybrid": "data/card2card_neighbors.json",
    "card2card_sparse": "data/card2card_neighbors.json",
    "query2modelcard": "data/query2modelcard_results.json",
    "tab2tab_by_type_multi_column": "data/tab2tab_by_type_multi_column_results.json",
}


def extract_json_path_from_log(log_path: str) -> Optional[str]:
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    patterns = [
        r"[✅✓]\s*Results saved to\s+([^\s\n]+\.json)",
        r"Results saved to\s+([^\s\n]+\.json)",
        r"saved to\s+([^\s\n]+\.json)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
        if matches:
            raw = matches[-1]
            return str(resolve_path(raw))
    # Fallback: try default path for known log names (if file exists)
    log_name = Path(log_path).stem
    default_rel = _LOG_DEFAULT_JSON.get(log_name)
    if default_rel:
        p = resolve_path(default_rel)
        if p.exists():
            return str(p)
    return None


def extract_query_from_log(log_path: str) -> Optional[str]:
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "Query Model ID:" in line or "Query Model:" in line:
                m = re.search(r"(?:Query Model ID:|Query Model:)\s*([^\s\n]+)", line)
                if m:
                    return m.group(1)
            if "--query" in line:
                m = re.search(r"--query\s+([^\s\n]+)", line)
                if m:
                    return m.group(1)
            if "Query classification:" in line or "query classification:" in line:
                m = re.search(r"(?:Query|query) classification:\s*([^\s\n]+)", line)
                if m:
                    return f"Classification: {m.group(1)}"
    return None


def extract_model_ids_from_log(log_path: str) -> List[str]:
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Numbered list: "  1. org/model-name"
    pattern = r"^\s*\d+\.\s+([a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-\.]+)"
    matches = re.findall(pattern, content, re.MULTILINE)
    return list(dict.fromkeys(matches))  # preserve order, dedup


def extract_table_ids_from_log(log_path: str) -> List[int]:
    """Extract table IDs only from the explicit 'N. Table ID: 12345' lines.
    Do not use loose number patterns so we avoid debug/candidate IDs.
    """
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Only lines that look like "  1. Table ID: 35840" (tab2tab / tab2tab_by_type output)
    pattern = r"^\s*\d+\.\s+Table ID:\s*(\d+)\s*$"
    matches = re.findall(pattern, content, re.MULTILINE)
    return [int(tid) for tid in matches]


def load_result_json(json_path: str) -> Optional[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_table_ids_from_results(data: Dict[str, Any]) -> List[int]:
    out = []
    if "results" in data:
        out.extend([int(tid) for tid in data["results"]])
    if "intermediate" in data:
        inter = data["intermediate"]
        if "retrieved_table_ids" in inter:
            out.extend([int(tid) for tid in inter["retrieved_table_ids"]])
    if isinstance(data, list):
        out.extend([int(x) for x in data if isinstance(x, (int, str)) and str(x).isdigit()])
    return list(dict.fromkeys(out))


def extract_model_ids_from_results(data: Dict[str, Any]) -> List[str]:
    if "model_ids" in data:
        return list(data["model_ids"])
    return []


def generate_markdown_from_log(
    log_path: str,
    output_dir: str = "md",
    db_path: str = "data/modellake.db",
    classification_json: Optional[str] = "data/table_classifications.json",
    max_rows: int = 50,
) -> Optional[str]:
    log_name = Path(log_path).stem
    output_path = Path(output_dir) / f"{log_name}.md"

    print(f"\nProcessing: {log_path}")

    json_path = extract_json_path_from_log(log_path)
    if not json_path:
        print(f"  No result JSON path in log")
        return None

    print(f"  Result JSON: {json_path}")

    result_data = load_result_json(json_path)
    if not result_data:
        result_data = {}

    query = extract_query_from_log(log_path) or result_data.get("query_model") or result_data.get("query") or "Unknown"

    table_ids = extract_table_ids_from_results(result_data)
    model_ids = extract_model_ids_from_results(result_data)

    if not table_ids:
        table_ids = extract_table_ids_from_log(log_path)
        if table_ids:
            print(f"  Extracted {len(table_ids)} table IDs from log")
    if not model_ids:
        model_ids = extract_model_ids_from_log(log_path)
        if model_ids:
            print(f"  Extracted {len(model_ids)} model IDs from log")

    print(f"  Query: {query} | Tables: {len(table_ids)} | Models: {len(model_ids)}")

    if not table_ids and not model_ids and query == "Unknown":
        return None

    classifications = {}
    if classification_json:
        p = resolve_path(classification_json)
        if p.exists():
            classifications = load_classifications(str(p))

    # Use shared pipeline type: model-search = models first, then tables; table-search = tables only
    is_model_search = is_model_search_log(log_name)

    json_path_str = str(Path(json_path).resolve()) if json_path else ""

    lines = [
        f"# Search Results: {log_name}\n",
        f"**Query:** `{query}`\n",
        f"**Result JSON (full path):** `{json_path_str}`\n",
    ]
    if is_model_search:
        lines.append(f"**Total models (primary):** {len(model_ids)}\n")
        lines.append(f"**Related tables (secondary):** {len(table_ids)}\n")
    else:
        lines.append(f"**Total tables:** {len(table_ids)}\n")
    lines.append("---\n")

    # Model-search: models first, then related tables.
    if model_ids:
        lines.append("\n## Model IDs (primary)\n\n")
        for i, mid in enumerate(model_ids[:20], 1):
            lines.append(f"{i}. `{mid}`\n")
        if len(model_ids) > 20:
            lines.append(f"\n*... and {len(model_ids) - 20} more*\n")
        lines.append("\n---\n")

    if table_ids:
        section_title = "\n## Related tables (secondary)\n\n" if is_model_search else "\n## Tables\n\n"
        lines.append(section_title)
        db_abs = str(resolve_path(db_path))
        for i, tableid in enumerate(table_ids[:50], 1):
            meta = get_table_metadata(tableid, db_abs)
            lines.append(f"\n### Table {i}: ID `{tableid}`\n")
            if not meta:
                lines.append("⚠️  **Table not found in database**\n")
                continue
            full_csv_path = find_csv_file(meta["filename"])
            lines.append(f"- **Filename:** `{meta['filename']}`")
            if full_csv_path:
                lines.append(f"- **Full path:** `{os.path.abspath(full_csv_path)}`")
            lines.append(f"- **Table Group:** `{meta.get('table_group', 'N/A')}`")
            lines.append(f"- **Table Type:** `{meta.get('table_type', 'N/A')}`")
            if tableid in classifications:
                lines.append(f"- **Classification:** `{classifications[tableid]}`")
            lines.append("")
            df = load_table_csv(meta["filename"], max_rows=max_rows)
            if df is not None:
                lines.append(f"#### Preview ({len(df)} rows, {len(df.columns)} columns)\n")
                lines.append(table_to_markdown(df, max_rows=max_rows))
                lines.append("\n#### Columns\n")
                for col in df.columns:
                    lines.append(f"- `{col}` ({df[col].dtype}): {df[col].notna().sum()}/{len(df)} non-null")
            else:
                lines.append("⚠️  **CSV file not found**\n")
            lines.append("---\n")
        if len(table_ids) > 50:
            lines.append(f"\n*... and {len(table_ids) - 50} more tables*\n")

    output_path = resolve_path(output_dir) / f"{log_name}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Table integration: save integrated CSV to md/<log_name>_materials/csv_integrated/ and append section to md
    integrated_csv_path = _run_integration_and_save(
        log_name=log_name,
        result_data=result_data,
        result_json_path=json_path,
        table_ids=table_ids,
        db_path=db_path,
        output_dir=output_dir,
    )
    if integrated_csv_path:
        extra = [
            "\n## Integrated table\n\n",
            f"Tables integrated (union) and saved to:\n\n",
            f"**Full path:** `{os.path.abspath(integrated_csv_path)}`\n\n",
            "---\n",
        ]
        with open(output_path, "a", encoding="utf-8") as f:
            f.write("\n".join(extra))
        print(f"  Integrated table: {integrated_csv_path}")

    print(f"  Generated: {output_path}")
    return str(output_path)


def main():
    ap = argparse.ArgumentParser(description="Generate markdown from log files")
    ap.add_argument("--logs_dir", default="logs")
    ap.add_argument("--output_dir", default="md")
    ap.add_argument("--db_path", default="data/modellake.db")
    ap.add_argument("--classification_json", default="data/table_classifications.json")
    ap.add_argument("--max_rows", type=int, default=50)
    ap.add_argument("--log_file", default=None, help="Single log file (else process all in logs_dir)")
    args = ap.parse_args()

    if args.log_file:
        log_files = [Path(args.log_file)]
    else:
        logs_dir = resolve_path(args.logs_dir)
        if not logs_dir.exists():
            print(f"Logs directory not found: {logs_dir}")
            return
        all_logs = sorted(logs_dir.glob("*.log"))
        log_files = [f for f in all_logs if f.stem not in _SKIP_LOG_STEMS]

    if not log_files:
        print("No log files found")
        return

    print(f"Found {len(log_files)} log file(s)")

    generated = []
    failed = []
    for log_file in log_files:
        out = generate_markdown_from_log(
            str(log_file),
            output_dir=args.output_dir,
            db_path=args.db_path,
            classification_json=args.classification_json if resolve_path(args.classification_json).exists() else None,
            max_rows=args.max_rows,
        )
        if out:
            generated.append(out)
        else:
            failed.append(str(log_file))

    print(f"\nGenerated: {len(generated)} | Failed: {len(failed)}")
    if failed:
        for f in failed:
            print(f"  - {f}")
    print("\nWhat this means:")
    print("  - Generated = one md file written per log (see output_dir, e.g. md/*.md)")
    print("  - Failed = log has no 'Results saved to ... .json' line (and no fallback file); skip or run that search with --output_json to get a result file")


if __name__ == "__main__":
    main()
