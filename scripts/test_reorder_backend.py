"""
Small helper script to run backend column-reordering tests.

Usage:
    python scripts/test_reorder_backend.py

It imports the test helper from `src/demo/backend.py` and executes it.
If no assertion is raised, it prints a success message.
"""

import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pandas as pd

from integration.table_integration import _reorder_columns_deterministic  # type: ignore


def _run_tests() -> None:
    # Basic ordering by non-null rate + all-null columns last.
    df_single = pd.DataFrame(
        {
            "A": [1, None, None],   # 1/3
            "B": [1, 2, None],      # 2/3
            "C": [1, 2, 3],         # 3/3
            "D": [None, None, None] # 0/3 (all null, should always go last)
        }
    )
    out_single = _reorder_columns_deterministic(df_single)
    assert list(out_single.columns) == ["C", "B", "A", "D"], f"single: got {list(out_single.columns)}"

    # Stable tie-breaker: if two columns have the same non-null rate,
    # preserve the original order between them.
    df_tie = pd.DataFrame(
        {
            "X": [1, None, 1, None],       # 2/4 = 0.5
            "Y": [None, 2, None, 2],       # 2/4 = 0.5 (same as X, should stay after X)
            "Z": [3, 3, 3, None],          # 3/4 = 0.75
            "W": [None, None, None, None], # 0/4 = 0.0 (all null, should be last)
        }
    )
    out_tie = _reorder_columns_deterministic(df_tie)
    assert list(out_tie.columns) == ["Z", "X", "Y", "W"], f"tie: got {list(out_tie.columns)}"


def main() -> None:
    """Run deterministic reordering tests."""
    _run_tests()
    print("OK: deterministic reorder tests passed.")


if __name__ == "__main__":
    main()

