import os
import time
from typing import Dict, List

import duckdb
import pytest

from src.config import RELATIONSHIP_PARQUET
from src.utils import (
    _sample_csv_basenames,
    _sample_model_ids,
    _get_models_to_tables_batch_sql,
    _get_tables_per_model,
    _get_tables_to_models_batch_sql,
    _load_modelid_to_csv_expand,
    load_csvs_to_modelids,
    load_modelid_to_csvlist,
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(RELATIONSHIP_PARQUET),
    reason="relationship parquet is required for these integration-style tests",
)

def _normalize_mapping(mapping: Dict[str, List[str]]) -> Dict[str, set]:
    normalized: Dict[str, set] = {}
    for key, values in mapping.items():
        normalized[str(key)] = {
            str(value)
            for value in values
            if value is not None and str(value).strip()
        }
    return normalized


def _legacy_tables_to_models(csv_basenames: List[str]) -> Dict[str, List[str]]:
    normalized_basenames = [os.path.basename(str(csv)) for csv in csv_basenames if str(csv).strip()]
    expected: Dict[str, List[str]] = {basename: [] for basename in normalized_basenames}

    df = _load_modelid_to_csv_expand()
    filtered = (
        df.loc[df["csv_basename"].isin(normalized_basenames), ["csv_basename", "modelId"]]
        .dropna()
        .drop_duplicates()
    )
    for csv_basename, model_id in filtered.itertuples(index=False):
        expected[str(csv_basename)].append(str(model_id))
    return expected


def _measure_seconds(fn, repeat: int = 3):
    elapsed_values = []
    last_result = None
    for _ in range(repeat):
        start = time.perf_counter()
        last_result = fn()
        elapsed_values.append(time.perf_counter() - start)
    elapsed_values.sort()
    median_elapsed = elapsed_values[len(elapsed_values) // 2]
    return median_elapsed, last_result


def test_models_to_tables_outputs_match_ignoring_order() -> None:
    model_ids = _sample_model_ids(limit=12)
    assert model_ids, "Need sample modelIds from relationship parquet"

    batch_result = _get_models_to_tables_batch_sql(model_ids)
    legacy_result = _get_tables_per_model(model_ids)
    public_result = {model_id: load_modelid_to_csvlist(model_id) for model_id in model_ids}

    assert _normalize_mapping(batch_result) == _normalize_mapping(legacy_result)
    assert _normalize_mapping(batch_result) == _normalize_mapping(public_result)


def test_tables_to_models_outputs_match_ignoring_order() -> None:
    csv_basenames = _sample_csv_basenames(limit=25)
    assert csv_basenames, "Need sample csv basenames from relationship parquet"

    batch_result = _get_tables_to_models_batch_sql(csv_basenames)
    legacy_result = _legacy_tables_to_models(csv_basenames)
    public_result = load_csvs_to_modelids(csv_basenames)

    assert _normalize_mapping(batch_result) == _normalize_mapping(legacy_result)
    assert _normalize_mapping(batch_result) == _normalize_mapping(public_result)


def test_models_to_tables_batch_sql_speed_regression() -> None:
    model_ids = _sample_model_ids(limit=20)
    assert model_ids, "Need sample modelIds from relationship parquet"

    batch_elapsed, batch_result = _measure_seconds(
        lambda: _get_models_to_tables_batch_sql(model_ids)
    )
    legacy_elapsed, legacy_result = _measure_seconds(
        lambda: _get_tables_per_model(model_ids)
    )
    public_elapsed, public_result = _measure_seconds(
        lambda: {model_id: load_modelid_to_csvlist(model_id) for model_id in model_ids}
    )

    assert _normalize_mapping(batch_result) == _normalize_mapping(legacy_result)
    assert _normalize_mapping(batch_result) == _normalize_mapping(public_result)
    print(
        "\n[models -> tables] "
        f"batch_sql={batch_elapsed:.4f}s | "
        f"legacy_full_parquet={legacy_elapsed:.4f}s | "
        f"public_per_model={public_elapsed:.4f}s | "
        f"speedup_vs_legacy={legacy_elapsed / batch_elapsed:.2f}x | "
        f"speedup_vs_public={public_elapsed / batch_elapsed:.2f}x"
    )
    assert batch_elapsed <= legacy_elapsed * 1.5, (
        f"Expected DuckDB batch path to avoid major slowdown; "
        f"batch={batch_elapsed:.4f}s legacy={legacy_elapsed:.4f}s"
    )


def test_tables_to_models_batch_sql_speed_regression() -> None:
    csv_basenames = _sample_csv_basenames(limit=40)
    assert csv_basenames, "Need sample csv basenames from relationship parquet"

    batch_elapsed, batch_result = _measure_seconds(
        lambda: _get_tables_to_models_batch_sql(csv_basenames)
    )
    legacy_elapsed, legacy_result = _measure_seconds(
        lambda: _legacy_tables_to_models(csv_basenames)
    )
    public_elapsed, public_result = _measure_seconds(
        lambda: load_csvs_to_modelids(csv_basenames)
    )

    assert _normalize_mapping(batch_result) == _normalize_mapping(legacy_result)
    assert _normalize_mapping(batch_result) == _normalize_mapping(public_result)
    print(
        "\n[tables -> models] "
        f"batch_sql={batch_elapsed:.4f}s | "
        f"legacy_expand_df={legacy_elapsed:.4f}s | "
        f"public_wrapper={public_elapsed:.4f}s | "
        f"speedup_vs_legacy={legacy_elapsed / batch_elapsed:.2f}x | "
        f"speedup_vs_public={public_elapsed / batch_elapsed:.2f}x"
    )
    assert batch_elapsed <= legacy_elapsed * 1.5, (
        f"Expected DuckDB batch path to avoid major slowdown; "
        f"batch={batch_elapsed:.4f}s legacy={legacy_elapsed:.4f}s"
    )
