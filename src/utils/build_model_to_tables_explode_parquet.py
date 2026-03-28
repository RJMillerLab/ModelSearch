#!/usr/bin/env python3
"""Build flattened model<->table parquet from RELATIONSHIP_PARQUET.

Output schema:
  - modelId: str
  - csv_basename: str
  - resource: one of {"hugging","github","arxiv","llm"}
"""

from __future__ import annotations

import argparse
import os
import sys

import duckdb

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import MODEL_TO_TABLES_EXPLODE_PARQUET, RELATIONSHIP_PARQUET


def build_exploded_parquet(output_path: str, relationship_parquet: str) -> None:
    out_abs = os.path.abspath(output_path)
    rel_abs = os.path.abspath(relationship_parquet).replace("\\", "/")
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    con = duckdb.connect(":memory:")
    sql = f"""
    COPY (
        WITH exploded AS (
            SELECT modelId, regexp_extract(csv_path, '[^/]+$') AS csv_basename, 'hugging' AS resource
            FROM read_parquet('{rel_abs}'), UNNEST(coalesce(hugging_table_list_dedup, [])) AS t(csv_path)
            WHERE modelId IS NOT NULL AND csv_path IS NOT NULL
            UNION ALL
            SELECT modelId, regexp_extract(csv_path, '[^/]+$') AS csv_basename, 'github' AS resource
            FROM read_parquet('{rel_abs}'), UNNEST(coalesce(github_table_list_dedup, [])) AS t(csv_path)
            WHERE modelId IS NOT NULL AND csv_path IS NOT NULL
            UNION ALL
            SELECT modelId, regexp_extract(csv_path, '[^/]+$') AS csv_basename, 'arxiv' AS resource
            FROM read_parquet('{rel_abs}'), UNNEST(coalesce(html_table_list_mapped_dedup, [])) AS t(csv_path)
            WHERE modelId IS NOT NULL AND csv_path IS NOT NULL
            UNION ALL
            SELECT modelId, regexp_extract(csv_path, '[^/]+$') AS csv_basename, 'llm' AS resource
            FROM read_parquet('{rel_abs}'), UNNEST(coalesce(llm_table_list_mapped_dedup, [])) AS t(csv_path)
            WHERE modelId IS NOT NULL AND csv_path IS NOT NULL
        )
        SELECT DISTINCT
            CAST(modelId AS VARCHAR) AS modelId,
            CAST(csv_basename AS VARCHAR) AS csv_basename,
            CAST(resource AS VARCHAR) AS resource
        FROM exploded
        WHERE csv_basename IS NOT NULL AND csv_basename <> ''
    ) TO '{out_abs}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(sql)
    stats = con.execute(f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT modelId) AS models,
            COUNT(DISTINCT csv_basename) AS tables
        FROM read_parquet('{out_abs}')
    """).fetchone()
    con.close()
    rows, models, tables = stats
    print(f"Saved: {out_abs}")
    print(f"rows={rows}, distinct_models={models}, distinct_tables={tables}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build flattened model<->table parquet index")
    parser.add_argument(
        "--output_parquet",
        default=MODEL_TO_TABLES_EXPLODE_PARQUET,
        help=f"Output parquet path (default: {MODEL_TO_TABLES_EXPLODE_PARQUET})",
    )
    parser.add_argument(
        "--relationship_parquet",
        default=RELATIONSHIP_PARQUET,
        help=f"Input relationship parquet (default: {RELATIONSHIP_PARQUET})",
    )
    args = parser.parse_args()
    build_exploded_parquet(args.output_parquet, args.relationship_parquet)


if __name__ == "__main__":
    main()

