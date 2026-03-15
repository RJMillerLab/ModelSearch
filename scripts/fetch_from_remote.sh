#!/bin/bash
# Fetch ModelSearch required files from remote server

REMOTE_HOST="watgpu.cs.uwaterloo.ca"
REMOTE_USER="z6dong"
REMOTE_PATH="/u501/z6dong/Repo/ModelSearchDemo"
LOCAL_PATH="/Users/doradong/Repo/ModelSearchDemo"

echo "=========================================="
echo "Fetch ModelSearch files from remote server"
echo "=========================================="
echo ""
echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
echo "Local: ${LOCAL_PATH}"
echo ""

# Required files to fetch
FILES=(
    "data/card2card.faiss"
    "data/card2card_embeddings.npz"
    "data/card2card_corpus.jsonl"
)

# Optional files (if exist)
OPTIONAL_FILES=(
    "../ModelTables/data/modellake.db"
    "../ModelTables/data/processed/modelcard_step3_dedup_v2_251117.parquet"
    "../ModelTables/logs/parquet_schema.log"
)

echo "📥 Fetching required files..."
for file in "${FILES[@]}"; do
    echo ""
    echo "Fetching: ${file}"
    mkdir -p "${LOCAL_PATH}/$(dirname ${file})"
    scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${file}" "${LOCAL_PATH}/${file}" 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ ${file} fetched successfully"
        ls -lh "${LOCAL_PATH}/${file}" | awk '{print "   Size: " $5}'
    else
        echo "❌ ${file} fetch failed (may not exist)"
    fi
done

echo ""
echo "📥 Fetching optional files..."
for file in "${OPTIONAL_FILES[@]}"; do
    echo ""
    echo "Attempting to fetch: ${file}"
    mkdir -p "${LOCAL_PATH}/$(dirname ${file})"
    scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${file}" "${LOCAL_PATH}/${file}" 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ ${file} fetched successfully"
        ls -lh "${LOCAL_PATH}/${file}" | awk '{print "   Size: " $5}'
    else
        echo "⚠️  ${file} does not exist or fetch failed (optional file)"
    fi
done

echo ""
echo "=========================================="
echo "File fetch completed"
echo "=========================================="
echo ""
echo "Checking fetched files:"
ls -lh "${LOCAL_PATH}/data/card2card"* 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
