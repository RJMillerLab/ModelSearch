#!/usr/bin/env python3
"""
Script to get the model ID from the default CSV using duckdb to read parquet
"""
import sys
import os

default_csv = "../ModelTables/data/processed/deduped_github_csvs_v2_251117/0021c79d4e1a37579ca87328864d67a5_table_0.csv"
parquet_path = "../ModelTables/data/processed/modelcard_step3_dedup_v2_251117.parquet"
basename = os.path.basename(default_csv)

print(f"Checking CSV: {default_csv}")
print(f"File exists: {os.path.exists(default_csv)}")
print(f"Looking for basename: {basename}")
print()

# Use duckdb to read parquet (no pandas needed)
import duckdb
print("✅ Using duckdb to read parquet...")
with duckdb.connect() as con:
    print("Reading parquet file...")
    result = con.execute(
        f"SELECT modelId, github_table_list_dedup, hugging_table_list_dedup, html_table_list_mapped_dedup, llm_table_list_mapped_dedup FROM read_parquet('{parquet_path}')"
    ).fetchall()
    model_ids = set()
    for row in result:
        model_id = row[0]
        for col_idx in [1, 2, 3, 4]:
            if row[col_idx] is not None:
                table_list = row[col_idx]
                if isinstance(table_list, list):
                    for table_path in table_list:
                        if basename in str(table_path) or os.path.basename(str(table_path)) == basename:
                            model_ids.add(str(model_id))
                            break
                elif isinstance(table_list, str) and basename in table_list:
                    model_ids.add(str(model_id))
    if model_ids:
        model_ids_list = sorted(list(model_ids))
        print(f"\n✅ Found {len(model_ids_list)} model ID(s):")
        for i, mid in enumerate(model_ids_list, 1):
            print(f"   {i}. {mid}")
        print(f"\n✅ Default Model ID: {model_ids_list[0]}")
        sys.exit(0)
print("❌ No model IDs found in parquet")


# Final fallback
print("\n" + "="*60)
print("Using hardcoded fallback: Salesforce/codet5-base")
print("="*60)
print("\nDefault Model ID: Salesforce/codet5-base")
sys.exit(0)

