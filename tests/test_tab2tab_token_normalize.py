"""Ensure tab2tab keyword/single_column tokens match Blend_internal df_to_index normalization."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_blend_df_to_index():
    blend_utils = (REPO_ROOT / "others" / "Blend_internal" / "src" / "utils.py").resolve()
    if not blend_utils.is_file():
        pytest.skip(f"Blend_internal utils not found: {blend_utils}")
    spec = importlib.util.spec_from_file_location("blend_df_to_index_utils", blend_utils)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.df_to_index


def test_header_normalization_matches_df_to_index():
    df_to_index = _load_blend_df_to_index()
    sys_path_insert = str(REPO_ROOT)
    import sys

    if sys_path_insert not in sys.path:
        sys.path.insert(0, sys_path_insert)
    from src.search.tab2tab import _extract_keyword_query_from_table, _normalize_header_token_for_index

    df = pd.DataFrame([[1, 2]], columns=["  WikiTQ  ", "Model"])
    df.columns.name = 0
    idx = df_to_index(df)
    header_tokens = set(idx.loc[idx["RowId"] == -1, "CellValue"].tolist())
    assert header_tokens == {"wikitq", "model"}

    assert _normalize_header_token_for_index("  WikiTQ  ") == "wikitq"
    assert _normalize_header_token_for_index("Model") == "model"


def test_cell_normalization_matches_df_to_index():
    df_to_index = _load_blend_df_to_index()
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.search.tab2tab import _normalize_cell_token_for_index

    df = pd.DataFrame([["  TaPEX  ", 1], ["GPT3.5", 2]])
    df.columns.name = 0
    idx = df_to_index(df)
    body = idx.loc[idx["RowId"] != -1, "CellValue"].tolist()
    assert "tapex" in body
    assert "gpt3.5" in body

    assert _normalize_cell_token_for_index("  TaPEX  ") == "tapex"
    assert _normalize_cell_token_for_index("GPT3.5") == "gpt3.5"


def test_extract_keyword_from_csv_headers(tmp_path):
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.search.tab2tab import _extract_keyword_query_from_table

    p = tmp_path / "t.csv"
    p.write_text("WikiTQ,Model\n1,2\n", encoding="utf-8")
    toks = _extract_keyword_query_from_table(str(p))
    assert set(toks) == {"wikitq", "model"}
