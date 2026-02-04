#!/usr/bin/env bash
# Run Part 2 (Inference & downstream) commands from docs/build_index.md in parallel.
# Each job logs to logs/<name>.log. Run from repo root.
#
# Usage:
#   ./scripts/run_part2_parallel.sh
#   LOG_DIR=my_logs ./scripts/run_part2_parallel.sh
#   DB_PATH=data_citationlake/modellake.db ./scripts/run_part2_parallel.sh

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG_DIR="${LOG_DIR:-logs}"
DB_PATH="${DB_PATH:-data/modellake.db}"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Part 2 parallel run (from build_index.md)"
echo "Logs: $LOG_DIR/*.log"
echo "=============================================="

# 2.1 query2modelcard
python -m src.search.query2modelcard --query "transformer model for code generation" \
  --emb_npz data/card2card_embeddings.npz --faiss_index data/card2card.faiss --top_k 20 --device cuda \
  > "$LOG_DIR/query2modelcard.log" 2>&1 &
PID_21=$!

# 2.2 card2card (dense, sparse, hybrid)
python -m src.search.card2card search --model_id google-bert/bert-base-uncased \
  --emb_npz data/card2card_embeddings.npz --faiss_index data/card2card.faiss --top_k 20 --retrieval_mode dense \
  > "$LOG_DIR/card2card_dense.log" 2>&1 &
PID_DENSE=$!

python -m src.search.card2card search --model_id google-bert/bert-base-uncased \
  --jsonl_path data/card2card_corpus.jsonl --top_k 20 --retrieval_mode sparse \
  > "$LOG_DIR/card2card_sparse.log" 2>&1 &
PID_SPARSE=$!

python -m src.search.card2card search --model_id google-bert/bert-base-uncased \
  --emb_npz data/card2card_embeddings.npz --faiss_index data/card2card.faiss \
  --jsonl_path data/card2card_corpus.jsonl --top_k 20 --retrieval_mode hybrid --hybrid_method rrf \
  > "$LOG_DIR/card2card_hybrid.log" 2>&1 &
PID_HYBRID=$!

# 2.3 card2tab2card (keyword, all, by_type)
python -m src.search.card2tab2card --model_id google-bert/bert-base-uncased --search_type keyword --k 10 \
  > "$LOG_DIR/card2tab2card_keyword.log" 2>&1 &
PID_C2T2C_KW=$!

python -m src.search.card2tab2card --model_id google-bert/bert-base-uncased --mode all \
  --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv --output_folder data \
  > "$LOG_DIR/card2tab2card_all.log" 2>&1 &
PID_C2T2C_ALL=$!

python -m src.search.card2tab2card --model_id google-bert/bert-base-uncased --mode by_type \
  --classification_json data/table_classifications.json \
  > "$LOG_DIR/card2tab2card_by_type.log" 2>&1 &
PID_C2T2C_BT=$!

# 2.4 tab2tab (keyword)
python -m src.search.tab2tab --search_type keyword --query "model_name,accuracy,task" --k 10 \
  --db_path "$DB_PATH" --output data/tab2tab_results.json \
  > "$LOG_DIR/tab2tab_keyword.log" 2>&1 &
PID_T2T=$!

# 2.5 tab2tab_by_type
python -m src.search.tab2tab_by_type \
  --query data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv \
  --classification_json data/table_classifications.json --search_type single_column --k 10 \
  --db_path "$DB_PATH" --output data/tab2tab_by_type_results.json \
  > "$LOG_DIR/tab2tab_by_type.log" 2>&1 &
PID_T2T_BT=$!

echo "Started all jobs. Waiting..."
wait $PID_21 $PID_DENSE $PID_SPARSE $PID_HYBRID $PID_C2T2C_KW $PID_C2T2C_ALL $PID_C2T2C_BT $PID_T2T $PID_T2T_BT
echo "=============================================="
echo "All Part 2 jobs finished. Check $LOG_DIR/*.log"
echo "=============================================="
