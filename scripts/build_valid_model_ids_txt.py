"""
Build a txt file of model_id that have tables (non-empty csv_basename in relationship parquet).
Used by the demo backend "Narrow down" (require_seed_has_tables): inference only loads this txt.

Usage:
  python scripts/build_valid_model_ids_txt.py --parquet data_citationlake/processed/modelcard_step3_dedup.parquet --output data/valid_model_ids_with_tables.txt
"""

import os
import sys
import argparse

# Run from repo root so src imports work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.modelsearch.compare_baselines import _read_relationships


def main():
    parser = argparse.ArgumentParser(
        description="Extract model IDs that have tables from relationship parquet → one model_id per line."
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Path to relationship parquet (e.g. data_citationlake/processed/modelcard_step3_dedup.parquet)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output txt path (e.g. data/valid_model_ids_with_tables.txt)",
    )
    args = parser.parse_args()

    parquet_path = os.path.abspath(args.parquet) if not os.path.isabs(args.parquet) else args.parquet
    if not os.path.isfile(parquet_path):
        print(f"Error: parquet not found: {parquet_path}", file=sys.stderr)
        sys.exit(1)

    rel = _read_relationships(parquet_path)
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
