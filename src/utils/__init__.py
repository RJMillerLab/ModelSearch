"""
Utility functions for modelsearch
"""
import os
import pandas as pd
import duckdb
import numpy as np
from typing import List, Dict, Any
from src.config import DATA_RAW, RELATIONSHIP_PARQUET, TABLE_BASE_DIRS

__all__ = ["get_device", "load_combined_data", "load_table", "resolve_table_path", "classify_resource", "load_modelid_to_csvlist", "load_csvs_to_modelids", "_load_modelid_to_csv_expand"]


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
        file_path = DATA_RAW
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

def load_csvs_to_modelids(csv_basenames: List[str]) -> Dict[str, List[str]]:
    """
    Input: csv_basenames, Output: modelIds list
    """
    df_model_to_csv_expand = _load_modelid_to_csv_expand()
    return df_model_to_csv_expand.loc[df_model_to_csv_expand["csv_basename"].isin(csv_basenames), "modelId"].dropna().unique().tolist()

def load_modelid_to_csvlist(model_id: str) -> List[str]:
    """
    Input: modelId, Output: csv_basenames list
    """
    df_model_to_csv_expand = _load_modelid_to_csv_expand()
    return df_model_to_csv_expand.loc[df_model_to_csv_expand["modelId"] == model_id, "csv_basename"].dropna().unique().tolist()

def _load_modelid_to_csv_expand() -> pd.DataFrame:
    """
    Default: load from RELATIONSHIP_PARQUET, Output: DataFrame with columns: modelId, csv_basename
    """
    parquet_path = RELATIONSHIP_PARQUET
    sql = """
    SELECT DISTINCT
        modelId,
        regexp_extract(csv_path, '[^/]+$') AS csv_basename
    FROM read_parquet(?),
    UNNEST(
        list_concat(
            coalesce(hugging_table_list_dedup, []),
            coalesce(github_table_list_dedup, []),
            coalesce(html_table_list_mapped_dedup, []),
            coalesce(llm_table_list_mapped_dedup, [])
        )
    ) AS t(csv_path)
    WHERE csv_path IS NOT NULL
    """
    con = duckdb.connect()
    return con.execute(sql, [parquet_path]).df()


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
    # hugging: 10 hex + _tableN OR fallback
    if re.fullmatch(r"[0-9a-f]{10}_table\d+\.csv", b) or re.fullmatch(r".+_table\d+\.csv", b):
        return "hugging"
    raise ValueError(f"Unknown resource: {basename}")

def load_table(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Input: csv_path, Output: pd.DataFrame
    """
    resolved = resolve_table_path(csv_path)
    if resolved and os.path.exists(resolved):
        return pd.read_csv(resolved)
    print(f"⚠️  Table not found: {csv_path}")
    return None
