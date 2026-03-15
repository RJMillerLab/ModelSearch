#!/usr/bin/env python3
"""Print paths from src.config for use by shell scripts. Run from repo root."""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import (
    MODELLAKE_DB,
    RELATIONSHIP_PARQUET,
    MODELTABLES_DATA,
    DEDUPED_HUGGING_CSVS,
    DEDUPED_GITHUB_CSVS,
    TABLES_OUTPUT,
    RAW_DIR,
    OUTPUT_DIR,
    CLASSIFICATION_JSON,
    DATA_RAW,
)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Print config paths for shell scripts")
    ap.add_argument("key", nargs="?", choices=[
        "modellake_db", "relationship_parquet", "modeltables_data",
        "deduped_hugging_csvs", "deduped_github_csvs", "tables_output", "raw_dir",
        "sample_csv", "output_dir", "classification_json",
    ], help="Which path to print (default: print all as KEY=value)")
    ap.add_argument("--report", action="store_true",
                     help="Print one path per line for detect_data_storage.sh (config paths + fixed artifact names)")
    args = ap.parse_args()
    sample_csv = os.path.join(DEDUPED_HUGGING_CSVS, "0000e35dae_table1.csv")
    paths = {
        "modellake_db": MODELLAKE_DB,
        "relationship_parquet": RELATIONSHIP_PARQUET,
        "modeltables_data": MODELTABLES_DATA,
        "deduped_hugging_csvs": DEDUPED_HUGGING_CSVS,
        "deduped_github_csvs": DEDUPED_GITHUB_CSVS,
        "tables_output": TABLES_OUTPUT,
        "raw_dir": RAW_DIR,
        "sample_csv": sample_csv,
        "output_dir": OUTPUT_DIR,
        "classification_json": CLASSIFICATION_JSON,
    }
    if args.report:
        # One path per line for storage report (config paths + fixed names under OUTPUT_DIR)
        for p in [
            os.path.join(OUTPUT_DIR, "card2card_embeddings.npz"),
            os.path.join(OUTPUT_DIR, "card2card.faiss"),
            MODELLAKE_DB,
            RELATIONSHIP_PARQUET,
            os.path.join(OUTPUT_DIR, "valid_model_ids_with_tables.txt"),
            os.path.join(OUTPUT_DIR, "card2card_sparse_index"),
            DEDUPED_HUGGING_CSVS,
            DEDUPED_GITHUB_CSVS,
            TABLES_OUTPUT,
            CLASSIFICATION_JSON,
            "config/demo_template/search_results.json",
            "fig",
            os.path.join(OUTPUT_DIR, "jobs"),
        ]:
            print(p)
        return
    if args.key:
        print(paths[args.key])
    else:
        for k, v in paths.items():
            print(f"{k}={v}")


if __name__ == "__main__":
    main()
