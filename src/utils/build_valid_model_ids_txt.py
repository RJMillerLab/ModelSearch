"""
Build a txt file of model_id that have tables (non-empty csv_basename in relationship parquet).

Note: The demo backend `require_seed_has_tables` now derives the allowlist from RELATIONSHIP_PARQUET
with the same `--resources` columns as Card2Tab2Card (cached in memory), so it stays consistent
when you run hugging-only vs multi-source. This script is still useful for offline lists / debugging.

Usage:
  python -m src.utils.build_valid_model_ids_txt --output data/valid_model_ids_with_tables_hugging.txt --resources hugging
"""

import os
import sys
import argparse

# Run from repo root so src imports work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils import _load_modelid_to_csv_expand


def main():
    parser = argparse.ArgumentParser(description="Extract model IDs that have tables from relationship parquet → one model_id per line.")
    parser.add_argument("--resources", nargs="+", default=["hugging", "github", "arxiv"], choices=["hugging", "github", "arxiv", "llm"], help="Resource labels to select from relationship parquet (e.g. hugging, github, arxiv, llm).")
    parser.add_argument("--output", required=True, help="Output txt path (e.g. data/valid_model_ids_with_tables.txt)")
    args = parser.parse_args()

    rel = _load_modelid_to_csv_expand(args.resources)
    valid_ids = sorted(rel["modelId"].dropna().astype(str).str.strip().unique())
    valid_ids = [mid for mid in valid_ids if mid]

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
