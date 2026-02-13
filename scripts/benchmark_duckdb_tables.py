#!/usr/bin/env python3
"""
Benchmark: DuckDB batch query vs per-model get_tables_for_model for "get tables for modelcards".

Run: python scripts/benchmark_duckdb_tables.py

Prints speedup (e.g., "DuckDB is 12.3x faster").
"""

import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

# Try these in order
PARQUET_CANDIDATES = [
    "data_citationlake/processed/modelcard_step4_v2.parquet",
    "data_citationlake/processed/modelcard_step3_dedup.parquet",
    "data_citationlake/processed/modelcard_step3_merged_v2_251117.parquet",
]


def _get_tables_for_models_duckdb(parquet_path: str, model_ids: list) -> dict:
    """DuckDB batch query."""
    import duckdb
    import pandas as pd
    if not model_ids or not os.path.exists(parquet_path):
        return {}
    path_abs = os.path.abspath(parquet_path).replace("\\", "/")
    conn = duckdb.connect(":memory:")
    ids_sql = ",".join(repr(m) for m in model_ids)
    try:
        cols = conn.execute("DESCRIBE SELECT * FROM read_parquet(?)", [path_abs]).fetchall()
    except Exception:
        cols = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{path_abs}')").fetchall()
    list_cols = [
        c[0] for c in cols
        if c[0] != "modelId" and ("csv" in c[0].lower() or "table_list" in c[0].lower())
    ][:4]
    if not list_cols:
        conn.close()
        return {}
    select_cols = "modelId, " + ", ".join(f'"{c}"' for c in list_cols)
    try:
        full_df = conn.execute(f"""
            SELECT {select_cols} FROM read_parquet(?)
            WHERE modelId IN ({ids_sql})
        """, [path_abs]).fetchdf()
    except Exception:
        full_df = conn.execute(f"""
            SELECT {select_cols} FROM read_parquet('{path_abs}')
            WHERE modelId IN ({ids_sql})
        """).fetchdf()
    conn.close()
    model_to_tables = {m: [] for m in model_ids}
    for _, row in full_df.iterrows():
        mid = str(row["modelId"])
        for col in list_cols:
            val = row.get(col)
            if val is None or (isinstance(val, float) and __import__("pandas").isna(val)):
                continue
            items = _normalize_val_to_items(val)
            for v in items:
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    base = os.path.basename(s)
                    if base and base not in model_to_tables[mid]:
                        model_to_tables[mid].append(base)
    return model_to_tables


def _normalize_val_to_items(val):
    """Unify iteration: handle list, tuple, np.ndarray, scalar. Same logic for both paths."""
    import pandas as pd
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, (list, tuple)):
        return val
    if hasattr(val, "__iter__") and not isinstance(val, str):
        return list(val)
    return [val]


def _get_tables_per_model(parquet_path: str, model_ids: list) -> dict:
    """Legacy: load full parquet + per-model filter (simulates old approach)."""
    import pandas as pd
    if not model_ids or not os.path.exists(parquet_path):
        return {}
    df = pd.read_parquet(parquet_path)
    list_cols = [c for c in df.columns if c != "modelId" and ("csv" in c.lower() or "table_list" in c.lower())][:4]
    if not list_cols:
        return {m: [] for m in model_ids}
    model_to_tables = {m: [] for m in model_ids}
    for _, row in df[df["modelId"].isin(model_ids)].iterrows():
        mid = str(row["modelId"])
        for col in list_cols:
            val = row.get(col)
            items = _normalize_val_to_items(val)
            for v in items:
                if v is not None and str(v).strip():
                    base = os.path.basename(str(v))
                    if base and base not in model_to_tables[mid]:
                        model_to_tables[mid].append(base)
    return model_to_tables


def main():
    parquet = os.environ.get("PARQUET_PATH")
    if not parquet:
        for p in PARQUET_CANDIDATES:
            fp = os.path.join(REPO_ROOT, p)
            if os.path.exists(fp) and os.path.getsize(fp) > 100:
                parquet = fp
                break
    if not parquet or not os.path.exists(parquet):
        print("No parquet found. Try: PARQUET_PATH=/path/to/modelcard.parquet python scripts/benchmark_duckdb_tables.py")
        return 1

    # Get some model_ids from parquet
    import duckdb
    import pandas as pd
    path_abs = os.path.abspath(parquet).replace("\\", "/")
    conn = duckdb.connect(":memory:")
    try:
        sample = conn.execute("SELECT modelId FROM read_parquet(?) LIMIT 100", [path_abs]).fetchdf()
    except Exception:
        sample = conn.execute(f"SELECT modelId FROM read_parquet('{path_abs}') LIMIT 100").fetchdf()
    conn.close()
    model_ids = sample["modelId"].dropna().unique().tolist()[:10]
    if not model_ids:
        print("No modelIds in parquet")
        return 1

    print(f"Benchmark: get tables for {len(model_ids)} modelcards")
    print(f"Parquet: {parquet}\n")

    # DuckDB batch (np.ndarray/list unified via _normalize_val_to_items)
    t0 = time.perf_counter()
    duck_result = _get_tables_for_models_duckdb(parquet, model_ids)
    t_duck = time.perf_counter() - t0
    total_duck = sum(len(v) for v in duck_result.values())

    # Per-model legacy
    t0 = time.perf_counter()
    legacy_result = _get_tables_per_model(parquet, model_ids)
    t_legacy = time.perf_counter() - t0
    total_legacy = sum(len(v) for v in legacy_result.values())

    print(f"DuckDB batch:  {t_duck*1000:.2f} ms  ({total_duck} tables)")
    print(f"Per-model:     {t_legacy*1000:.2f} ms  ({total_legacy} tables)")

    # Consistency check
    duck_set = set()
    for v in duck_result.values():
        duck_set.update(v)
    legacy_set = set()
    for v in legacy_result.values():
        legacy_set.update(v)
    if duck_set == legacy_set:
        print(f"Consistency: OK (same {len(duck_set)} unique tables)")
    else:
        only_duck = duck_set - legacy_set
        only_legacy = legacy_set - duck_set
        print(f"MISMATCH: DuckDB={len(duck_set)}, Per-model={len(legacy_set)}")
        if only_duck:
            print(f"  Only in DuckDB: {list(only_duck)[:5]}")
        if only_legacy:
            print(f"  Only in Per-model: {list(only_legacy)[:5]}")

    if t_duck > 0 and t_legacy > 0:
        speedup = t_legacy / t_duck
        print(f"\nDuckDB is {speedup:.1f}x faster")
    return 0


if __name__ == "__main__":
    sys.exit(main())
