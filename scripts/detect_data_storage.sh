#!/usr/bin/env bash
# Detect storage usage for all ModelSearch data paths (see bak/DATA_AND_STORAGE.md).
# Paths are read from src.config via scripts/get_config_paths.py --report.
# Run from repo root, or pass root as first arg.
#
# Usage:
#   ./scripts/detect_data_storage.sh
#   ./scripts/detect_data_storage.sh [REPO_ROOT]

set -e
REPO_ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

echo "=== ModelSearch data path storage report ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "Date: $(date -Iseconds 2>/dev/null || date)"
echo ""

# Paths: same defaults as src.config (MODELTABLES_DATA, MODELLAKE_DB, RELATIONSHIP_PARQUET, TABLE_BASE_DIRS)
PATHS=(
  "data_251117/card2card_embeddings.npz"
  "data_251117/card2card_embeddings_hugging.npz"
  "data_251117/card2card_model_ids.txt"
  "data_251117/model_to_tables_explode_v2_251117.parquet"
  "data_251117/card2card.faiss"
  "data_251117/valid_model_ids_with_tables_hugging.txt"
  "data_251117/card2card_sparse_index"
  "../ModelTables/data/processed/modelcard_step3_dedup_v2_251117.parquet"
  "../ModelTables/data/processed/deduped_hugging_csvs_v2_251117"
  "../ModelTables/data/processed/deduped_github_csvs_v2_251117"
  "../ModelTables/data/processed/tables_output_v2_251117"
  "data_251117/table_classifications.json"
  "config/demo_template/search_results.json"
  "fig"
  "data_251117/jobs_251117"
)

printf "%-55s %-8s %s\n" "PATH" "EXISTS" "SIZE"
printf "%-55s %-8s %s\n" "----" "-----" "----"

grand_total_k=0
for p in "${PATHS[@]}"; do
  full="$REPO_ROOT/$p"
  if [ -e "$full" ]; then
    exists="yes"
    sz_h=$(du -sh "$full" 2>/dev/null | cut -f1)
    sz_k=$(du -sk "$full" 2>/dev/null | cut -f1)
    [ -n "$sz_k" ] && grand_total_k=$((grand_total_k + sz_k))
    printf "%-55s %-8s %s\n" "$p" "$exists" "${sz_h:-?}"
  else
    printf "%-55s %-8s %s\n" "$p" "no" "-"
  fi
done

echo ""
if [ "$grand_total_k" -gt 0 ]; then
  total_mb=$((grand_total_k / 1024))
  total_gb=$((grand_total_k / 1024 / 1024))
  if [ "$total_gb" -gt 0 ]; then
    echo "TOTAL: ~${total_gb} GB (${grand_total_k} KB)"
  else
    echo "TOTAL: ~${total_mb} MB (${grand_total_k} KB)"
  fi
else
  echo "TOTAL: no paths found (all missing or empty)"
fi

echo ""
echo "HF Spaces persistent storage tiers: Small 20GB | Medium 150GB | Large 1TB (see Space Settings)."
echo "--- End of report (paste above for review) ---"
