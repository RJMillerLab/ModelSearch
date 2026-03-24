"""
Build a txt file of model_id that have tables (non-empty csv_basename in relationship data).

Uses MODEL_TO_TABLES_EXPLODE_PARQUET (same index as load_modelid_to_csvlist / Card2Tab2Card).
Build the parquet first::

  python scripts/build_model_to_tables_explode_parquet.py

Note: The demo backend `require_seed_has_tables` is legacy API-only; table flow uses query2tab2card.
This script is still useful for offline lists / debugging.

Usage:
  python -m src.utils.build_valid_model_ids_txt --output data/valid_model_ids_with_tables_hugging.txt --resources hugging
"""

import os
import sys
import argparse

# Run from repo root so src imports work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils import list_model_ids_with_tables_from_explode


def main():
    parser = argparse.ArgumentParser(description="Extract model IDs that have tables from exploded parquet → one model_id per line.")
    parser.add_argument("--resources", nargs="+", default=["hugging", "github", "arxiv"], choices=["hugging", "github", "arxiv", "llm"], help="Resource labels (same as Card2Tab2Card --resources).")
    parser.add_argument("--output", required=True, help="Output txt path (e.g. data/valid_model_ids_with_tables.txt)")
    args = parser.parse_args()

    valid_ids = list_model_ids_with_tables_from_explode(args.resources)

    out_abs = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for mid in valid_ids:
            f.write(mid + "\n")

    print(f"Wrote {len(valid_ids)} model IDs to {args.output}")


if __name__ == "__main__":
    main()
