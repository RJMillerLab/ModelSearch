"""
Utility functions for modelsearch
"""
import pandas as pd
import os

from .table_loader import load_table, resolve_table_path, load_table_from_db, TABLE_BASE_DIRS

__all__ = ["get_device", "load_combined_data", "load_table", "resolve_table_path", "load_table_from_db", "TABLE_BASE_DIRS"]


def get_device() -> str:
    """Auto-detect device: cuda if available, else cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_combined_data(data_type, file_path="data/raw", columns=[]):
    """
    Load combined parquet files for modelcard or datasetcard.
    
    Args:
        data_type: "modelcard" or "datasetcard"
        file_path: Directory containing parquet shards
        columns: Optional list of columns to load
    
    Returns:
        Combined DataFrame
    """
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

