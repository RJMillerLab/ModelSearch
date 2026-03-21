"""
Quick sanity test for card2tab2card related mapping functions:
1) model_id -> related table basenames
2) table basenames -> related model_ids

Usage:
  python -m scripts.test_related_mappings --model_id google-bert/bert-base-uncased
  python -m scripts.test_related_mappings --sample_tables 5
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

# Ensure repo root is on sys.path when run as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import VALID_MODEL_IDS_TXT
from src.utils import load_modelid_to_csvlist, load_csvs_to_modelids


def _pick_default_model_id() -> str:
    with open(VALID_MODEL_IDS_TXT, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                return s
    raise RuntimeError(f"No valid model ids found in: {VALID_MODEL_IDS_TXT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick test for related mapping functions in card2tab2card pipeline.")
    parser.add_argument("--model_id", default=None, help="Model id to test. If omitted, use first id from VALID_MODEL_IDS_TXT.")
    parser.add_argument("--sample_tables", type=int, default=5, help="How many tables to reverse-map for quick test.")
    args = parser.parse_args()

    model_id = args.model_id or _pick_default_model_id()
    sample_tables = max(1, int(args.sample_tables))

    print(f"[1] model -> tables: {model_id}")
    tables: List[str] = load_modelid_to_csvlist(model_id)
    print(f"    related tables: {len(tables)}")
    print(f"    sample tables: {tables[:sample_tables]}")
    if not tables:
        print("    no related tables found; stop.")
        return

    subset = tables[:sample_tables]
    print(f"[2] tables -> models for first {len(subset)} table(s)")
    reverse = load_csvs_to_modelids(subset)
    total_reverse_models = sum(len(v) for v in reverse.values())
    print(f"    reverse map keys: {len(reverse)}")
    print(f"    total mapped model ids (sum): {total_reverse_models}")

    # Check whether original model_id appears back in reverse mapping.
    seen_back = any(model_id in mids for mids in reverse.values())
    print(f"[3] round-trip contains original model_id: {seen_back}")

    for t in subset:
        mids = reverse.get(t, [])
        print(f"    table={t} -> model_ids(sample)={mids[:5]}")


if __name__ == "__main__":
    main()

