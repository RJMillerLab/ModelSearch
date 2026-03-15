#!/bin/bash
# Quick test script - Check environment and run basic tests

set -e

echo "=========================================="
echo "ModelSearch Quick Test Script"
echo "=========================================="
echo ""

# Check Python
echo "1. Checking Python environment..."
if ! command -v python &> /dev/null; then
    echo "❌ Python not found"
    exit 1
fi
PYTHON_VERSION=$(python --version)
echo "✅ $PYTHON_VERSION"
echo ""

# Check dependencies
echo "2. Checking critical dependencies..."
MISSING_DEPS=()
for dep in "pandas" "numpy" "sentence_transformers" "faiss" "flask"; do
    if ! python -c "import $dep" 2>/dev/null; then
        MISSING_DEPS+=("$dep")
    fi
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "❌ Missing dependencies: ${MISSING_DEPS[*]}"
    echo "   Please run: pip install -r requirements.txt"
    exit 1
fi
echo "✅ Critical dependencies installed"
echo ""

# Check data directory (path matches src.config MODELTABLES_DATA default)
echo "3. Checking data directory..."
MODELTABLES_DATA_DIR="${MODELTABLES_DATA:-../ModelTables/data}"
if [ ! -d "$MODELTABLES_DATA_DIR" ]; then
    echo "⚠️  $MODELTABLES_DATA_DIR directory does not exist"
    echo "   Some features may not be available (set MODELTABLES_DATA or DATA_ROOT if elsewhere)"
else
    echo "✅ $MODELTABLES_DATA_DIR directory exists"
fi
echo ""

# Check index files
echo "4. Checking FAISS index..."
if [ -f "data/card2card.faiss" ] && [ -f "data/card2card_embeddings.npz" ]; then
    echo "✅ FAISS index exists"
    INDEX_EXISTS=true
else
    echo "⚠️  FAISS index does not exist"
    echo "   Need to build index before running search"
    INDEX_EXISTS=false
fi
echo ""

# If index exists, run simple test
if [ "$INDEX_EXISTS" = true ]; then
    echo "5. Running simple query test..."
    echo ""
    
    # Test query
    python -m src.search.query2modelcard \
        --query "transformer model" \
        --emb_npz data/card2card_embeddings.npz \
        --faiss_index data/card2card.faiss \
        --top_k 3 \
        --device cpu \
        --output_json data/quick_test_results.json 2>&1 | head -20
    
    if [ -f "data/quick_test_results.json" ]; then
        echo ""
        echo "✅ Query test successful!"
        echo "   Results saved to: data/quick_test_results.json"
        echo ""
        echo "Top 3 results:"
        python -c "import json; data=json.load(open('data/quick_test_results.json')); [print(f\"  {i+1}. {r['model_id']}\") for i, r in enumerate(data['results'][:3])]"
    else
        echo "❌ Query test failed"
    fi
else
    echo "5. Skipping query test (index does not exist)"
    echo ""
    echo "To build index, run:"
    echo "  python -m src.search.card2card build-index \\"
    echo "    --field card \\"
    echo "    --raw_dir <raw_dir from src.config> \\"
    echo "    --output_jsonl data/card2card_corpus.jsonl \\"
    echo "    --output_npz data/card2card_embeddings.npz \\"
    echo "    --output_index data/card2card.faiss \\"
    echo "    --device cpu"
fi

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
