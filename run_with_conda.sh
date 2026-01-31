#!/bin/bash
# Start backend and frontend with conda env (pyserini_hybrid or modelsearch-demo).
# Sets KMP_DUPLICATE_LIB_OK=TRUE to avoid OpenMP duplicate lib error.

set -e
cd "$(dirname "$0")"

CONDA_BASE=$(conda info --base 2>/dev/null)
if [[ -z "$CONDA_BASE" ]]; then
    echo "conda not found"
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | grep -q "pyserini_hybrid"; then
    conda activate pyserini_hybrid
    echo "Using conda env: pyserini_hybrid"
else
    conda activate modelsearch-demo
    echo "Using conda env: modelsearch-demo (pyserini_hybrid not found)"
fi

PYTHON=$(which python)
echo "   Python: $PYTHON"
echo ""

export KMP_DUPLICATE_LIB_OK=TRUE

[[ -f ".env" ]] || echo ".env not found; some features may be unavailable"
mkdir -p logs

for port in 5002 5001; do
    if lsof -ti:$port &>/dev/null; then
        echo "Clearing port $port ..."
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
done

echo "Starting backend (port 5002)..."
nohup "$PYTHON" -m src.demo.backend > logs/backend.log 2>&1 &
BPID=$!
echo "   Backend PID: $BPID"

sleep 3
if ! kill -0 $BPID 2>/dev/null; then
    echo "Backend failed; see logs/backend.log"
    tail -30 logs/backend.log
    exit 1
fi
echo "Backend started"

echo ""
echo "Starting frontend (port 5001)..."
nohup "$PYTHON" -m src.demo.frontend > logs/frontend.log 2>&1 &
FPID=$!
echo "   Frontend PID: $FPID"

sleep 3
if ! kill -0 $FPID 2>/dev/null; then
    echo "Frontend failed; see logs/frontend.log"
    kill $BPID 2>/dev/null
    exit 1
fi
echo "Frontend started"

echo ""
echo "=========================================="
echo "Started"
echo "=========================================="
echo "Backend API:  http://localhost:5002"
echo "Frontend UI:  http://localhost:5001"
echo ""
echo "Logs: tail -f logs/backend.log or logs/frontend.log"
echo "Stop: kill $BPID $FPID"
echo ""
