"""
Utility functions for modelsearch
"""
import os
import pandas as pd
import duckdb
import numpy as np
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from src.config import RAW_DIR, RELATIONSHIP_PARQUET, TABLE_BASE_DIRS, MODEL_TO_TABLES_EXPLODE_PARQUET, INDEX_TABLE

__all__ = ["get_device", "load_combined_data", "load_table", "resolve_table_path", "classify_resource", "classify_results", "filter_results_by_classify_results", "load_modelid_to_csvlist", "load_csvs_to_modelids", "model_id_has_resolvable_local_tables", "list_model_ids_with_tables_from_explode", "list_model_ids_with_tables_from_explode_filtered_by_modellake_db", "_load_modelid_to_csv_expand", "_get_models_to_tables_batch_sql", "_get_tables_per_model", "_get_tables_to_models_batch_sql", "_sample_model_ids", "_sample_csv_basenames", "get_tables_from_modellake_db",
    "is_model_search_log",
    "is_table_search_log",
    "get_repo_root",
    "load_classifications",
    "get_tables_metadata",
    "table_to_markdown",
    "get_model_tables_from_db",
]


def get_device() -> str:
    """Auto-detect device: cuda if available, else cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_combined_data(data_type, file_path=None, columns=None):
    """
    Load combined parquet files for modelcard or datasetcard.
    
    Args:
        data_type: "modelcard" or "datasetcard"
        file_path: Directory containing parquet shards
        columns: Optional list of columns to load
    
    Returns:
        Combined DataFrame
    """
    if file_path is None:
        file_path = RAW_DIR
    if columns is None:
        columns = []
    assert data_type in ["modelcard", "datasetcard"], "data_type must be 'modelcard' or 'datasetcard'"
    if data_type == "modelcard":
        file_names = [f"train-0000{i}-of-00004.parquet" for i in range(4)]
    elif data_type == "datasetcard":
        file_names = [f"train-0000{i}-of-00002.parquet" for i in range(2)]
    
    if columns:
        dfs = [pd.read_parquet(os.path.join(file_path, file), columns=columns) for file in file_names]
    else:
        dfs = [pd.read_parquet(os.path.join(file_path, file)) for file in file_names]
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def _sample_model_ids(limit: int = 20) -> List[str]:
    with duckdb.connect(":memory:") as con:
        rows = con.execute(
            """
            SELECT DISTINCT modelId
            FROM read_parquet(?)
            WHERE modelId IS NOT NULL
            LIMIT ?
            """,
            [RELATIONSHIP_PARQUET, limit],
        ).fetchall()
    return [str(model_id) for (model_id,) in rows if str(model_id).strip()]


def _sample_csv_basenames(limit: int = 40) -> List[str]:
    model_ids = _sample_model_ids(limit=10)
    model_to_tables = _get_models_to_tables_batch_sql(model_ids)
    csv_basenames: List[str] = []
    seen = set()
    for tables in model_to_tables.values():
        for csv_basename in tables:
            base = os.path.basename(str(csv_basename))
            if base and base not in seen:
                seen.add(base)
                csv_basenames.append(base)
            if len(csv_basenames) >= limit:
                return csv_basenames
    return csv_basenames


def _flatten_cell(value: Any) -> List[Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        out: List[Any] = []
        for v in value:
            if isinstance(v, (list, np.ndarray)):
                out.extend(_flatten_cell(v))
            else:
                out.append(v)
        return out
    if isinstance(value, np.ndarray):
        return _flatten_cell(value.tolist())
    return [value]


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

def _get_tables_per_model(model_ids: list) -> dict:
    """
    Slow path: read full relationship parquet and filter to ``model_ids``.
    Prefer batch SQL helpers (e.g. ``_get_models_to_tables_batch_sql``) for speed.
    """
    import pandas as pd
    from src.config import RELATIONSHIP_PARQUET
    df = pd.read_parquet(RELATIONSHIP_PARQUET)
    list_cols = [c for c in df.columns if c != "modelId" and ("csv" in c.lower() or "table_list" in c.lower())][:4]
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

def _relationship_table_list_column_names(
    resources: Optional[List[str]],
    *,
    available_in_parquet: set[str],
) -> List[str]:
    """
    Ordered list of relationship-parquet list columns to read.
    When resources is None/empty, use all canonical columns that exist (same union as _load_modelid_to_csv_expand).
    """
    col_by_key = {
        "hugging": "hugging_table_list_dedup",
        "github": "github_table_list_dedup",
        "arxiv": "html_table_list_mapped_dedup",
        "llm": "llm_table_list_mapped_dedup",
    }
    order = ["hugging", "github", "arxiv", "llm"]
    if not resources:
        keys = order
    else:
        norm = _normalize_allowed_resource_labels(resources)
        keys = [k for k in order if k in norm]
    out = [col_by_key[k] for k in keys if col_by_key[k] in available_in_parquet]
    return out


def _get_models_to_tables_batch_sql(model_ids: list, resources: Optional[List[str]] = None) -> dict:
    """Batch query over pre-exploded parquet: modelId -> csv_basenames."""
    mids = [str(m).strip() for m in model_ids if str(m).strip()]
    model_to_tables: Dict[str, List[str]] = {m: [] for m in mids}
    if not mids:
        return model_to_tables
    explode_path = os.path.abspath(MODEL_TO_TABLES_EXPLODE_PARQUET).replace("\\", "/")
    if not os.path.isfile(explode_path):
        raise FileNotFoundError(
            f"Missing exploded parquet: {explode_path}. "
            "Build it first with scripts/build_model_to_tables_explode_parquet.py"
        )
    resource_filter = sorted(_normalize_allowed_resource_labels(resources)) if resources else []
    conn = duckdb.connect(":memory:")
    conn.register("input_model_ids", pd.DataFrame({"modelId": mids}))
    if resource_filter:
        conn.register("input_resources", pd.DataFrame({"resource": resource_filter}))
        sql = f"""
        SELECT
            e.modelId,
            list(DISTINCT e.csv_basename ORDER BY e.csv_basename) AS csv_basenames
        FROM read_parquet('{explode_path}') AS e
        JOIN input_model_ids AS i
          ON e.modelId = i.modelId
        JOIN input_resources AS r
          ON e.resource = r.resource
        GROUP BY e.modelId
        """
    else:
        sql = f"""
        SELECT
            e.modelId,
            list(DISTINCT e.csv_basename ORDER BY e.csv_basename) AS csv_basenames
        FROM read_parquet('{explode_path}') AS e
        JOIN input_model_ids AS i
          ON e.modelId = i.modelId
        GROUP BY e.modelId
        """
    rows = conn.execute(sql).fetchall()
    conn.close()
    for mid, basenames in rows:
        key = str(mid).strip()
        model_to_tables[key] = [str(b).strip() for b in _normalize_val_to_items(basenames) if str(b).strip()]
    return model_to_tables


def list_model_ids_with_tables_from_explode(
    resources: Optional[List[str]] = None,
    explode_parquet: Optional[str] = None,
) -> List[str]:
    """
    Distinct modelIds that have ≥1 non-empty csv_basename in MODEL_TO_TABLES_EXPLODE_PARQUET.

    When ``resources`` is omitted or empty, include all sources (hugging/github/arxiv/llm).
    Otherwise restrict to normalized labels (same semantics as ``load_modelid_to_csvlist``).
    """
    explode_path = os.path.abspath(explode_parquet or MODEL_TO_TABLES_EXPLODE_PARQUET).replace("\\", "/")
    if not os.path.isfile(explode_path):
        raise FileNotFoundError(
            f"Missing exploded parquet: {explode_path}. "
            "Build it first with scripts/build_model_to_tables_explode_parquet.py"
        )
    conn = duckdb.connect(":memory:")
    resource_filter = sorted(_normalize_allowed_resource_labels(resources)) if resources else []
    if resource_filter:
        conn.register("input_resources", pd.DataFrame({"resource": resource_filter}))
        sql = f"""
        SELECT DISTINCT CAST(e.modelId AS VARCHAR) AS modelId
        FROM read_parquet('{explode_path}') AS e
        INNER JOIN input_resources AS r ON e.resource = r.resource
        WHERE e.csv_basename IS NOT NULL AND length(trim(CAST(e.csv_basename AS VARCHAR))) > 0
        ORDER BY 1
        """
    else:
        sql = f"""
        SELECT DISTINCT CAST(modelId AS VARCHAR) AS modelId
        FROM read_parquet('{explode_path}')
        WHERE csv_basename IS NOT NULL AND length(trim(CAST(csv_basename AS VARCHAR))) > 0
        ORDER BY 1
        """
    rows = conn.execute(sql).fetchall()
    conn.close()
    return [str(r[0]).strip() for r in rows if r[0] is not None and str(r[0]).strip()]


def list_model_ids_with_tables_from_explode_filtered_by_modellake_db(
    resources: Optional[List[str]] = None,
    db_path: str = "",
    index_table: str = INDEX_TABLE,
    filename_col: str = "filename",
    explode_parquet: Optional[str] = None,
) -> List[str]:
    """
    Like ``list_model_ids_with_tables_from_explode`` but only keeps rows where exploded.csv_basename
    exists in a DuckDB index table (e.g. modellake_index.filename).

    Matching is performed on basenames (strip any directory in the DB column).
    """
    if not db_path:
        raise ValueError("db_path is required")
    db_abs = os.path.abspath(db_path)
    if not os.path.isfile(db_abs):
        raise FileNotFoundError(f"Missing DuckDB file: {db_abs}")

    explode_path = os.path.abspath(explode_parquet or MODEL_TO_TABLES_EXPLODE_PARQUET).replace("\\", "/")
    if not os.path.isfile(explode_path):
        raise FileNotFoundError(
            f"Missing exploded parquet: {explode_path}. "
            "Build it first with scripts/build_model_to_tables_explode_parquet.py"
        )

    resource_filter = sorted(_normalize_allowed_resource_labels(resources)) if resources else []
    with duckdb.connect(db_abs, read_only=True) as conn:
        if resource_filter:
            conn.register("input_resources", pd.DataFrame({"resource": resource_filter}))
            sql = f"""
            WITH filenames AS (
                SELECT DISTINCT
                    regexp_extract(CAST({filename_col} AS VARCHAR), '[^/]+$') AS csv_basename
                FROM {index_table}
                WHERE {filename_col} IS NOT NULL
            )
            SELECT DISTINCT CAST(e.modelId AS VARCHAR) AS modelId
            FROM read_parquet('{explode_path}') AS e
            INNER JOIN input_resources AS r ON e.resource = r.resource
            INNER JOIN filenames AS f ON f.csv_basename = CAST(e.csv_basename AS VARCHAR)
            WHERE e.csv_basename IS NOT NULL AND length(trim(CAST(e.csv_basename AS VARCHAR))) > 0
            ORDER BY 1
            """
        else:
            sql = f"""
            WITH filenames AS (
                SELECT DISTINCT
                    regexp_extract(CAST({filename_col} AS VARCHAR), '[^/]+$') AS csv_basename
                FROM {index_table}
                WHERE {filename_col} IS NOT NULL
            )
            SELECT DISTINCT CAST(e.modelId AS VARCHAR) AS modelId
            FROM read_parquet('{explode_path}') AS e
            INNER JOIN filenames AS f ON f.csv_basename = CAST(e.csv_basename AS VARCHAR)
            WHERE e.csv_basename IS NOT NULL AND length(trim(CAST(e.csv_basename AS VARCHAR))) > 0
            ORDER BY 1
            """
        rows = conn.execute(sql).fetchall()
    return [str(r[0]).strip() for r in rows if r and r[0] is not None and str(r[0]).strip()]


def _get_tables_to_models_batch_sql(csv_basenames: List[str]) -> Dict[str, List[str]]:
    """DuckDB batch query over pre-exploded parquet: csv basename -> modelIds."""
    normalized_basenames = []
    seen = set()
    for basename in csv_basenames:
        if basename is None:
            continue
        base = os.path.basename(str(basename).strip())
        if base and base not in seen:
            seen.add(base)
            normalized_basenames.append(base)
    if not normalized_basenames:
        return {}

    explode_path = os.path.abspath(MODEL_TO_TABLES_EXPLODE_PARQUET).replace("\\", "/")
    if not os.path.isfile(explode_path):
        raise FileNotFoundError(
            f"Missing exploded parquet: {explode_path}. "
            "Build it first with scripts/build_model_to_tables_explode_parquet.py"
        )
    conn = duckdb.connect(":memory:")
    conn.register("input_basenames", pd.DataFrame({"csv_basename": normalized_basenames}))
    sql = f"""
    SELECT
        b.csv_basename,
        list(DISTINCT e.modelId ORDER BY e.modelId) FILTER (WHERE e.modelId IS NOT NULL) AS model_ids
    FROM input_basenames AS b
    LEFT JOIN read_parquet('{explode_path}') AS e
      ON e.csv_basename = b.csv_basename
    GROUP BY b.csv_basename
    """
    rows = conn.execute(sql).fetchall()
    conn.close()

    basename_to_models: Dict[str, List[str]] = {basename: [] for basename in normalized_basenames}
    for basename, model_ids in rows:
        basename_to_models[basename] = [str(model_id) for model_id in _normalize_val_to_items(model_ids) if str(model_id).strip()]
    return basename_to_models

def load_csvs_to_modelids(csv_basenames: List[str]) -> Dict[str, List[str]]:
    """
    Input: csv_basenames, Output: csv_basename -> modelIds list
    """
    return _get_tables_to_models_batch_sql(csv_basenames)

def load_modelid_to_csvlist(model_id: str, resources: Optional[List[str]] = None) -> List[str]:
    """
    Input: modelId, Output: csv_basenames list.
    If resources is set (e.g. ['hugging']), only paths from those relationship-parquet columns are returned.
    """
    return _get_models_to_tables_batch_sql([model_id], resources=resources).get(model_id, [])


@lru_cache(maxsize=16384)
def model_id_has_resolvable_local_tables(model_id: str, frozen_resources: Tuple[str, ...]) -> bool:
    """
    True if relationship parquet lists ≥1 table basename for ``frozen_resources`` and
    ``resolve_table_path`` finds an existing file under ``TABLE_BASE_DIRS``.

    Parquet alone can list basenames whose files are absent on disk; this checks local resolvability.
    """
    mid = str(model_id).strip()
    if not mid:
        return False
    res_list: Optional[List[str]] = list(frozen_resources) if frozen_resources else None
    for base in load_modelid_to_csvlist(mid, resources=res_list):
        p = resolve_table_path(base)
        if p and os.path.exists(p):
            return True
    return False


def _load_modelid_to_csv_expand(resources: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Default: load from RELATIONSHIP_PARQUET, Output: DataFrame with columns: modelId, csv_basename

    Usage:
    df_model_to_csv_expand = _load_modelid_to_csv_expand()
    return df_model_to_csv_expand.loc[df_model_to_csv_expand["modelId"] == model_id, "csv_basename"].dropna().unique().tolist()

    df_model_to_csv_expand = _load_modelid_to_csv_expand()
    return df_model_to_csv_expand.loc[df_model_to_csv_expand["csv_basename"].isin(csv_basenames), "modelId"].dropna().unique().tolist()

    Deprecated: so slow, each time generate an intermediate dataframe
    """
    parquet_path = RELATIONSHIP_PARQUET
    # When ``resources`` is omitted: union all four source list columns.
    selected_cols: List[str]
    if not resources:
        selected_cols = [
            "hugging_table_list_dedup",
            "github_table_list_dedup",
            "html_table_list_mapped_dedup",
            "llm_table_list_mapped_dedup",
        ]
    else:
        # Normalize labels to the same output values as `classify_resource()`.
        # Supported: github|hugging|arxiv|llm (unknown labels will not match any column).
        normalized = _normalize_allowed_resource_labels(resources)
        col_map = {
            "hugging": "hugging_table_list_dedup",
            "github": "github_table_list_dedup",
            "arxiv": "html_table_list_mapped_dedup",
            "llm": "llm_table_list_mapped_dedup",
        }
        selected_cols = [col_map[k] for k in ["hugging", "github", "arxiv", "llm"] if k in normalized and k in col_map]
        if not selected_cols:
            # Nothing matched -> empty DataFrame
            return pd.DataFrame({"modelId": [], "csv_basename": []})

    if len(selected_cols) == 1:
        unnest_expr = f"UNNEST(coalesce({selected_cols[0]}, [])) AS t(csv_path)"
    else:
        parts = ", ".join(f"coalesce({c}, [])" for c in selected_cols)
        unnest_expr = f"UNNEST(list_concat({parts})) AS t(csv_path)"

    sql = f"""
        SELECT DISTINCT
            modelId,
            regexp_extract(csv_path, '[^/]+$') AS csv_basename
        FROM read_parquet(?),
        {unnest_expr}
        WHERE csv_path IS NOT NULL
    """
    con = duckdb.connect()
    try:
        return con.execute(sql, [parquet_path]).df()
    finally:
        con.close()


def resolve_table_path(csv_path: str) -> Optional[str]:
    """
    Resolve CSV basename to full path. Used by integration when only filename is known."""
    base = os.path.basename(csv_path)
    dirs = TABLE_BASE_DIRS
    for base_dir in dirs:
        p = os.path.join(base_dir, base)
        if os.path.exists(p):
            return os.path.abspath(p)
    # or classify the resource and get the corresponding directory
    return None

def classify_resource(basename: str) -> str:
    # github: 32 hex + _tableN
    import re
    b = basename
    if re.fullmatch(r"[0-9a-f]{32}_table_\d+\.csv", b):
        return "github"
    # html/arxiv: 0705.2450v1_table39.csv or 1234.5678_table3.csv
    if re.fullmatch(r"\d+\.\d+(?:v\d+)?_table\d+\.csv", b):
        return "arxiv"
    # hugging: 10 hex + _tableN, or any basename matching *_tableN.csv
    if re.fullmatch(r"[0-9a-f]{10}_table\d+\.csv", b) or re.fullmatch(r".+_table\d+\.csv", b):
        return "hugging"
    raise ValueError(f"Unknown resource: {basename}")


def classify_results(name: str) -> str:
    """
    Classify a table filename (or any path-like string) into a resource origin.

    This is a convenience wrapper around `classify_resource()` so callers can
    pass raw `name` strings (paths/basenames) and still reuse the same logic.
    """
    return classify_resource(os.path.basename(str(name)))


def _normalize_allowed_resource_labels(resources: List[str] | set[str]) -> set[str]:
    """
    Normalize user-facing resource names to `classify_resource()` return values.
    """
    out: set[str] = set()
    for r in resources:
        rr = str(r).strip().lower()
        if not rr:
            continue
        if rr in {"github", "gh"}:
            out.add("github")
        elif rr in {"arxiv", "html"}:
            out.add("arxiv")
        elif rr in {"hugging", "huggingface", "hf"}:
            out.add("hugging")
        elif rr in {"unknown"}:
            out.add("unknown")
        else:
            # Keep unknown labels as-is; they will just not match.
            out.add(rr)
    return out


def filter_results_by_classify_results(
    items: List[str],
    allowed_resources: List[str] | set[str],
    *,
    keep_unknown: bool = False,
) -> List[str]:
    """
    Filter `items` by calling `classify_results(item)` for each element.

    If `classify_resource()` can't classify an item, it is treated as `unknown`
    (and kept only if `keep_unknown=True` or `allowed_resources` includes `unknown`).
    """
    allowed = _normalize_allowed_resource_labels(allowed_resources)
    wants_unknown = keep_unknown or ("unknown" in allowed)

    out: List[str] = []
    for it in items:
        try:
            label = classify_results(it)
        except ValueError:
            label = "unknown"
        if label in allowed:
            out.append(it)
        elif label == "unknown" and wants_unknown:
            out.append(it)
    return out


def load_table(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Input: csv_path, Output: pd.DataFrame
    """
    resolved = resolve_table_path(csv_path)
    if resolved and os.path.exists(resolved):
        return pd.read_csv(resolved)
    print(f"⚠️  Table not found: {csv_path}")
    return None




def get_tables_from_modellake_db(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Input: limit, Output: List[Dict[str, Any]]
    """
    from src.config import MODELLAKE_DB, INDEX_TABLE
    with duckdb.connect(MODELLAKE_DB, read_only=True) as con:
        query = f"""
        SELECT DISTINCT tableid, filename, table_group, table_type 
        FROM {INDEX_TABLE} 
        WHERE rowid = -1
        """
        if limit:
            query += f" LIMIT {limit}"
        results = con.execute(query).fetchall()
        return [
            {"tableid": row[0], "filename": row[1], "table_group": row[2], "table_type": row[3]}
            for row in results
        ]

####################
# deprecated functions
####################

def _build_basename_index() -> Dict[str, str]:
    """
    deprecated, used for building a cache of basename to path from local directories

    Usage:
    idx = _build_basename_index(dirs)
    if base in idx:
        return idx[base]
    """
    _CACHED_BASENAME_TO_PATH: Optional[Dict[str, str]] = None
    from src.config import TABLE_BASE_DIRS
    if _CACHED_BASENAME_TO_PATH is not None:
        return _CACHED_BASENAME_TO_PATH
    index: Dict[str, str] = {}
    for base in TABLE_BASE_DIRS:
        for f in os.listdir(base):
            if f.lower().endswith(".csv"):
                index[f] = os.path.join(base, f)
    _CACHED_BASENAME_TO_PATH = index
    return index

def _load_table_from_db(
    tableid: int,
    index_table: str = "modellake_index",
) -> Optional[pd.DataFrame]:
    """
    Load table content from modellake.db by tableid (no CSV).
    modellake_index: (tableid, rowid, colid, tokenized). rowid=-1 = header.
    Returns None if DB missing or schema not supported.

    Deprecated, because column type can not be preserved.
    """
    from src.config import MODELLAKE_DB
    db_path = MODELLAKE_DB
    if not db_path or not os.path.isfile(db_path):
        return None
    con = duckdb.connect(db_path, read_only=True)
    info = con.execute(f"DESCRIBE {index_table}").fetchall()
    col_names = [r[0] for r in info]
    if "tableid" not in col_names or "rowid" not in col_names or "colid" not in col_names:
        con.close()
        return None
    value_col = "tokenized" if "tokenized" in col_names else ("value" if "value" in col_names else None)
    if not value_col:
        con.close()
        return None
    headers = con.execute(
        f"SELECT colid, {value_col} FROM {index_table} WHERE tableid = ? AND rowid = -1 ORDER BY colid",
        [tableid],
    ).fetchall()
    if not headers:
        con.close()
        return None
    col_names_list = [str(h[1]) if h[1] is not None else f"col_{h[0]}" for h in headers]
    rows = con.execute(
        f"SELECT rowid, colid, {value_col} FROM {index_table} WHERE tableid = ? AND rowid >= 0 ORDER BY rowid, colid",
        [tableid],
    ).fetchall()
    con.close()
    if not rows:
        return pd.DataFrame(columns=col_names_list)
    row_data: Dict[int, Dict[int, Any]] = defaultdict(dict)
    for rowid, colid, val in rows:
        row_data[rowid][colid] = val
    row_ids = sorted(row_data.keys())
    data = [[row_data[rid].get(cid) for cid in [h[0] for h in headers]] for rid in row_ids]
    return pd.DataFrame(data, columns=col_names_list)



# Postprocess: generate markdown from logs, table comparison MD. Pipeline + CSV search in .pipeline
import json
from typing import Dict, List, Optional, Any
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

def is_model_search_log(log_name: str) -> bool:
    """True if this log is from a model-search pipeline (models first, then related tables)."""
    return any(kw in log_name for kw in ("card2card", "card2tab2card", "query2modelcard"))


def is_table_search_log(log_name: str) -> bool:
    """True if this log is from a table-search pipeline (tables only)."""
    return any(kw in log_name for kw in ("tab2tab",))


def get_repo_root() -> Path:
    """Repo root (ModelSearchDemo)."""
    return _REPO_ROOT

def load_classifications(json_path: str) -> Dict[int, str]:
    """Load table classification JSON: keys as int, values as label."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}

def _row_to_table_metadata(row: Any) -> Dict[str, Any]:
    return {
        "tableid": row[0],
        "filename": row[1],
        "table_group": row[2],
        "table_type": row[3],
    }


def get_tables_metadata(
    tableids: List[int], db_path: str, index_table: str = "modellake_index"
) -> List[Dict[str, Any]]:
    import duckdb
    """Get table metadata from modellake.db by tableids, preserving input order."""
    normalized_tableids: List[int] = []
    seen = set()
    for tableid in tableids:
        if tableid is None:
            continue
        tableid_int = int(tableid)
        if tableid_int not in seen:
            seen.add(tableid_int)
            normalized_tableids.append(tableid_int)
    if not normalized_tableids:
        return []

    with duckdb.connect(db_path, read_only=True) as con:
        placeholders = ",".join(["?" for _ in normalized_tableids])
        q = (
            f"SELECT DISTINCT tableid, filename, table_group, table_type "
            f"FROM {index_table} "
            f"WHERE tableid IN ({placeholders}) AND rowid = -1"
        )
        rows = con.execute(q, normalized_tableids).fetchall()

    metadata_by_tableid = {int(row[0]): _row_to_table_metadata(row) for row in rows}
    return [
        metadata_by_tableid[tableid]
        for tableid in normalized_tableids
        if tableid in metadata_by_tableid
    ]

def table_to_markdown(df, max_rows: int = 50) -> str:
    """Render DataFrame as a markdown table (first max_rows)."""
    if df is None or df.empty:
        return "*Empty table*"
    df_display = df.head(max_rows)
    # Prefer pipe-style markdown; if ``to_markdown`` fails, use a fenced plain-text block.
    try:
        md = df_display.to_markdown(index=False)
    except Exception:
        md = "```\n" + df_display.to_string(index=False) + "\n```"
    if len(df) > max_rows:
        md += f"\n\n*... and {len(df) - max_rows} more rows*"
    return md

def get_table_classification(tableid: int, classifications: Dict[int, str]) -> Optional[str]:
    return classifications.get(tableid)


def get_model_tables_from_db(model_id: str, db_path: str) -> List[Dict[str, Any]]:
    """Get table metadata for all tables linked to model_id."""
    from src.utils import load_modelid_to_csvlist
    import duckdb
    basenames = load_modelid_to_csvlist(model_id)
    if not basenames:
        return []
    with duckdb.connect(db_path, read_only=True) as con:
        placeholders = ",".join(["?" for _ in basenames])
        q = f"SELECT DISTINCT tableid, filename, table_group, table_type FROM modellake_index WHERE filename IN ({placeholders}) AND rowid = -1"
        rows = con.execute(q, basenames).fetchall()
        return [_row_to_table_metadata(row) for row in rows]

