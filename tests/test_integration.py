"""
Test script for table integration functionality.

Tests integration with real search jobs under data/jobs.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import DIALITE_INTERNAL_REPO, REPO_ROOT
from src.integration.table_integration import (
    integrate_tables,
    integrate_tables_from_card2tab2card,
    integrate_tables_from_query2modelcard,
)
from src.utils import resolve_table_path

JOBS_DIR = Path(REPO_ROOT) / "data" / "jobs"
PREFERRED_SEARCH_TYPES = ("single_column", "keyword", "unionable")
ALL_INTEGRATION_MODES = ("union", "intersection", "outer_join", "alite")


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _assert_card2tab2card_schema(search_results: Dict, search_type: str) -> Dict:
    assert isinstance(search_results, dict), "search_results must be a dict"
    c2t2c = search_results.get("card2tab2card_results")
    assert isinstance(c2t2c, dict), "missing card2tab2card_results"
    payload = c2t2c.get(search_type)
    assert isinstance(payload, dict), f"missing card2tab2card_results[{search_type}]"
    assert isinstance(payload.get("intermediate"), dict), f"missing intermediate for {search_type}"
    # Strict: new format should have searched_tables; we still accept intermediate fallback for robustness
    st = payload.get("searched_tables")
    it = payload["intermediate"].get("retrieved_table_filenames")
    assert isinstance(st, list) or isinstance(it, list), "expected searched_tables or intermediate.retrieved_table_filenames"
    return payload


def _assert_query2modelcard_schema(search_results: Dict, retrieval_mode: Optional[str]) -> List:
    assert isinstance(search_results, dict), "search_results must be a dict"
    if retrieval_mode:
        all_modes = search_results.get("query2modelcard_all_modes")
        assert isinstance(all_modes, dict), "missing query2modelcard_all_modes"
        mode_data = all_modes.get(retrieval_mode)
        assert isinstance(mode_data, list), f"query2modelcard_all_modes[{retrieval_mode}] must be a list"
        return mode_data
    results = search_results.get("query2modelcard_results")
    assert isinstance(results, list), "missing query2modelcard_results list"
    return results


def _collect_resolved_table_paths(filenames: List[str], min_tables: int) -> List[str]:
    resolved_paths: List[str] = []
    seen = set()
    for filename in filenames:
        resolved = resolve_table_path(filename)
        if resolved and resolved not in seen:
            seen.add(resolved)
            resolved_paths.append(resolved)
        if len(resolved_paths) >= min_tables:
            break
    return resolved_paths


def _find_real_job_tables(min_tables: int = 2) -> Optional[Dict]:
    """Find the newest job with real search results and enough retrievable tables."""
    if not JOBS_DIR.exists():
        return None

    job_dirs = sorted((p for p in JOBS_DIR.iterdir() if p.is_dir()), reverse=True)
    for job_dir in job_dirs:
        search_results_path = job_dir / "search_results.json"
        if not search_results_path.exists():
            continue

        search_results = _load_json(search_results_path)
        card2tab2card_results = search_results.get("card2tab2card_results", {})
        if not isinstance(card2tab2card_results, dict):
            continue

        for search_type in PREFERRED_SEARCH_TYPES:
            type_results = card2tab2card_results.get(search_type, {})
            intermediate = type_results.get("intermediate", {}) if isinstance(type_results, dict) else {}
            # Prefer current schema field searched_tables if present; fallback to intermediate copy.
            retrieved_filenames = type_results.get("searched_tables") if isinstance(type_results, dict) else None
            if not isinstance(retrieved_filenames, list) or not retrieved_filenames:
                retrieved_filenames = intermediate.get("retrieved_table_filenames", [])
            if not isinstance(retrieved_filenames, list) or not retrieved_filenames:
                continue

            resolved_paths = _collect_resolved_table_paths(retrieved_filenames, min_tables=min_tables)
            if len(resolved_paths) >= min_tables:
                return {
                    "job_id": job_dir.name,
                    "job_dir": str(job_dir),
                    "search_results_path": str(search_results_path),
                    "search_type": search_type,
                    "retrieved_table_filenames": retrieved_filenames,
                    "resolved_table_paths": resolved_paths,
                }

    return None


def _has_alite_dependency() -> bool:
    alite_entry = Path(DIALITE_INTERNAL_REPO) / "alite" / "alite_fd.py"
    return alite_entry.exists()


def _print_result(result: Dict[str, Any]) -> None:
    if result["success"]:
        print(f"\n✅ Integration successful!")
        print(f"   Output shape: {result['integrated_table'].shape}")
        if len(result["integrated_table"]) > 0:
            print(f"\nFirst few rows:")
            print(result["integrated_table"].head())
        else:
            print("\n⚠️  Integration produced an empty table")
    else:
        print(f"\n❌ Integration failed: {result.get('error', 'Unknown error')}")


def _run_mode(mode: str, runner: Callable[[], Dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print(f"Running integration mode: {mode}")
    print("=" * 60)

    if mode == "alite" and not _has_alite_dependency():
        print("⚠️  Skipping ALITE: dependency not available under others/dialite.")
        return

    try:
        result = runner()
    except Exception as exc:
        print(f"\n❌ Integration raised an exception: {exc}")
        return

    _print_result(result)


def test_all_direct_integration_modes():
    """Test every integration mode with real tables from a saved job."""
    print("=" * 60)
    print("Test 1: Real Job Direct Integration by Mode")
    print("=" * 60)

    job_info = _find_real_job_tables(min_tables=3)
    if not job_info:
        print("⚠️  No saved job with at least 3 resolvable tables was found under data/jobs.")
        return

    table_paths = job_info["resolved_table_paths"][:3]
    print(f"Using job: {job_info['job_id']}")
    print(f"Search type: {job_info['search_type']}")
    print(f"Using {len(table_paths)} real tables for integration:")
    for path in table_paths:
        print(f"  - {os.path.basename(path)}")

    for mode in ALL_INTEGRATION_MODES:
        _run_mode(
            mode,
            lambda mode=mode: integrate_tables(table_paths, integration_type=mode, k=50),
        )


def test_all_search_results_integration_modes():
    """Test every integration mode from a real job search_results.json."""
    print("\n" + "=" * 60)
    print("Test 2: Real Job Search-Results Integration by Mode")
    print("=" * 60)

    job_info = _find_real_job_tables(min_tables=2)
    if not job_info:
        print("⚠️  No saved job with usable search results was found under data/jobs.")
        return

    search_results_path = job_info["search_results_path"]
    print(f"Using job: {job_info['job_id']}")
    print(f"Using search type: {job_info['search_type']}")
    print(f"Using search results: {search_results_path}")

    for mode in ALL_INTEGRATION_MODES:
        _run_mode(
            mode,
            lambda mode=mode: integrate_tables_from_card2tab2card(
                search_results_path,
                search_type=job_info["search_type"],
                integration_type=mode,
                k=10,
            ),
        )


def test_model_search_integration_from_job_json():
    """Test model-search integration (query2modelcard neighbor list) from job search_results.json."""
    print("\n" + "=" * 60)
    print("Test 3: Real Job Model-Search Integration (query2modelcard neighbors)")
    print("=" * 60)

    job_info = _find_real_job_tables(min_tables=2)
    if not job_info:
        print("⚠️  No saved job with usable search results was found under data/jobs.")
        return

    search_results_path = Path(job_info["search_results_path"])
    search_results = _load_json(search_results_path)
    rmode = str(search_results.get("query2modelcard_retrieval_mode") or "dense").strip().lower()

    # Enforce schema: backend job JSON with query2modelcard_all_modes populated for the run's retrieval mode.
    _assert_query2modelcard_schema(search_results, retrieval_mode=rmode)

    # Only need a quick sanity run for model-search integration.
    result = integrate_tables_from_query2modelcard(
        str(search_results_path),
        integration_type="union",
        k=10,
        max_models=10,
        query2modelcard_retrieval_mode=rmode,
    )
    assert result.get("success") is True, result.get("error")
    assert result.get("integrated_table") is not None
    assert isinstance(result.get("stats"), dict)


if __name__ == "__main__":
    print("Testing Table Integration Functionality\n")

    # Test 1: direct integration for every mode
    test_all_direct_integration_modes()

    # Test 2: integration from search results for every mode
    test_all_search_results_integration_modes()

    # Test 3: model-search (query2modelcard neighbors) integration from job JSON
    test_model_search_integration_from_job_json()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

