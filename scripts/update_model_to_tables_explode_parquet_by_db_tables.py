#!/usr/bin/env python3
"""
Update (overwrite) exploded model<->table parquet by DuckDB table list.

Behavior:
- Keep rows whose `csv_basename` exists in DuckDB index table filenames.
- If `--resources` is provided, only filter those resources; non-target resources
  remain unchanged.
- Overwrite the original parquet in place.

Prints stats:
- how many modelIds were filtered out
- how many modelIds remain
- how many csv tables remain
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from typing import List

import duckdb
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import INDEX_TABLE, MODEL_TO_TABLES_EXPLODE_PARQUET


def _norm_resources(resources: List[str]) -> List[str]:
    valid = {"hugging", "github", "arxiv", "llm"}
    out = sorted({str(r).strip().lower() for r in resources if str(r).strip()})
    bad = [r for r in out if r not in valid]
    if bad:
        raise ValueError(f"Invalid resources: {bad}. Valid: {sorted(valid)}")
    return out


def _fetch_stats(conn: duckdb.DuckDBPyConnection, parquet_path: str, resources: List[str]) -> dict:
    pq = os.path.abspath(parquet_path).replace("\\", "/")
    if resources:
        conn.register("input_resources", pd.DataFrame({"resource": resources}))
        row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT CAST(e.modelId AS VARCHAR)) AS models,
                COUNT(DISTINCT CAST(e.csv_basename AS VARCHAR)) AS csvs
            FROM read_parquet('{pq}') AS e
            INNER JOIN input_resources AS r
              ON CAST(e.resource AS VARCHAR) = r.resource
            WHERE e.csv_basename IS NOT NULL AND length(trim(CAST(e.csv_basename AS VARCHAR))) > 0
            """
        ).fetchone()
    else:
        row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT CAST(modelId AS VARCHAR)) AS models,
                COUNT(DISTINCT CAST(csv_basename AS VARCHAR)) AS csvs
            FROM read_parquet('{pq}')
            WHERE csv_basename IS NOT NULL AND length(trim(CAST(csv_basename AS VARCHAR))) > 0
            """
        ).fetchone()
    return {"rows": int(row[0] or 0), "models": int(row[1] or 0), "csvs": int(row[2] or 0)}


def update_parquet(
    *,
    parquet_path: str,
    duckdb_path: str,
    index_table: str,
    filename_col: str,
    resources: List[str],
) -> None:
    parquet_abs = os.path.abspath(parquet_path)
    db_abs = os.path.abspath(duckdb_path)

    if not os.path.isfile(parquet_abs):
        raise FileNotFoundError(f"Missing parquet: {parquet_abs}")
    if not os.path.isfile(db_abs):
        raise FileNotFoundError(f"Missing DuckDB file: {db_abs}")

    resources = _norm_resources(resources)
    tmp_out = f"{parquet_abs}.tmp.{uuid.uuid4().hex}.parquet"

    conn = duckdb.connect(":memory:")
    conn.execute(f"ATTACH '{db_abs}' AS mdb (READ_ONLY)")

    before_scope = _fetch_stats(conn, parquet_abs, resources)
    before_all = _fetch_stats(conn, parquet_abs, [])

    pq = parquet_abs.replace("\\", "/")
    tmp = tmp_out.replace("\\", "/")

    if resources:
        conn.register("input_resources", pd.DataFrame({"resource": resources}))
        sql_copy = f"""
        COPY (
            WITH filenames AS (
                SELECT DISTINCT regexp_extract(CAST({filename_col} AS VARCHAR), '[^/]+$') AS csv_basename
                FROM mdb.{index_table}
                WHERE {filename_col} IS NOT NULL
            ),
            targeted AS (
                SELECT e.*
                FROM read_parquet('{pq}') AS e
                INNER JOIN input_resources AS r
                  ON CAST(e.resource AS VARCHAR) = r.resource
                WHERE e.csv_basename IS NOT NULL AND length(trim(CAST(e.csv_basename AS VARCHAR))) > 0
            ),
            kept_targeted AS (
                SELECT t.*
                FROM targeted AS t
                INNER JOIN filenames AS f
                  ON CAST(t.csv_basename AS VARCHAR) = f.csv_basename
            ),
            untouched AS (
                SELECT e.*
                FROM read_parquet('{pq}') AS e
                LEFT JOIN input_resources AS r
                  ON CAST(e.resource AS VARCHAR) = r.resource
                WHERE r.resource IS NULL
            )
            SELECT * FROM kept_targeted
            UNION ALL
            SELECT * FROM untouched
        ) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    else:
        sql_copy = f"""
        COPY (
            WITH filenames AS (
                SELECT DISTINCT regexp_extract(CAST({filename_col} AS VARCHAR), '[^/]+$') AS csv_basename
                FROM mdb.{index_table}
                WHERE {filename_col} IS NOT NULL
            )
            SELECT e.*
            FROM read_parquet('{pq}') AS e
            INNER JOIN filenames AS f
              ON CAST(e.csv_basename AS VARCHAR) = f.csv_basename
            WHERE e.csv_basename IS NOT NULL AND length(trim(CAST(e.csv_basename AS VARCHAR))) > 0
        ) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """

    conn.execute(sql_copy)

    after_scope = _fetch_stats(conn, tmp_out, resources)
    after_all = _fetch_stats(conn, tmp_out, [])
    conn.close()

    os.replace(tmp_out, parquet_abs)

    filtered_models = max(0, before_scope["models"] - after_scope["models"])

    scope_text = ",".join(resources) if resources else "ALL"
    print(f"Updated parquet in place: {parquet_abs}")
    print(f"Scope resources: {scope_text}")
    print(
        "[scope] modelIds: "
        f"before={before_scope['models']} -> after={after_scope['models']} "
        f"(filtered_out={filtered_models})"
    )
    print(f"[scope] csv tables: before={before_scope['csvs']} -> after={after_scope['csvs']}")
    print(f"[scope] rows: before={before_scope['rows']} -> after={after_scope['rows']}")
    print(
        "[all] modelIds: "
        f"before={before_all['models']} -> after={after_all['models']}; "
        f"csv tables: before={before_all['csvs']} -> after={after_all['csvs']}; "
        f"rows: before={before_all['rows']} -> after={after_all['rows']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Overwrite exploded parquet, keeping only DB-known csv basenames.")
    parser.add_argument(
        "--parquet_path",
        default=MODEL_TO_TABLES_EXPLODE_PARQUET,
        help=f"Exploded parquet path to update in place (default: {MODEL_TO_TABLES_EXPLODE_PARQUET})",
    )
    parser.add_argument("--duckdb_path", required=True, help="DuckDB file path containing indexed table filenames.")
    parser.add_argument("--index_table", default=INDEX_TABLE, help=f"DuckDB index table name (default: {INDEX_TABLE})")
    parser.add_argument("--filename_col", default="filename", help="DuckDB filename column (default: filename)")
    parser.add_argument(
        "--resources",
        nargs="+",
        default=[],
        choices=["hugging", "github", "arxiv", "llm"],
        help="Optional resources to filter. If omitted, filter all resources.",
    )
    args = parser.parse_args()

    update_parquet(
        parquet_path=args.parquet_path,
        duckdb_path=args.duckdb_path,
        index_table=args.index_table,
        filename_col=args.filename_col,
        resources=args.resources,
    )


if __name__ == "__main__":
    main()
