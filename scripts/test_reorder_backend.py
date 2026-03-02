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

from demo.backend import _test_reorder_df_with_overlap  # type: ignore


def main() -> None:
    """Run the backend reordering tests."""
    _test_reorder_df_with_overlap()
    print("OK: _test_reorder_df_with_overlap passed.")


if __name__ == "__main__":
    main()

