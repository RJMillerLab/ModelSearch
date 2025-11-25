#!/bin/bash
# Start Backend on remote server, run Frontend locally

REMOTE_HOST="watgpu.cs.uwaterloo.ca"
REMOTE_USER="z6dong"
REMOTE_PATH="/u501/z6dong/Repo/ModelSearchDemo"
LOCAL_PORT=5002
REMOTE_PORT=5002

echo "=========================================="
echo "Start Remote Backend + Local Frontend"
echo "=========================================="
echo ""

# Check if local Frontend is running
if lsof -ti:5001 >/dev/null 2>&1; then
    echo "✅ Frontend is already running (port 5001)"
else
    echo "Starting local Frontend..."
    cd "$(dirname "$0")/.."
    python -m src.demo.frontend > logs/frontend.log 2>&1 &
    sleep 2
    if lsof -ti:5001 >/dev/null 2>&1; then
        echo "✅ Frontend started"
    else
        echo "❌ Frontend failed to start"
        exit 1
    fi
fi

echo ""
echo "Creating SSH tunnel to remote Backend..."
echo "Remote server: ${REMOTE_USER}@${REMOTE_HOST}"
echo ""

# Check if SSH tunnel already exists
if lsof -ti:${LOCAL_PORT} >/dev/null 2>&1; then
    echo "⚠️  Port ${LOCAL_PORT} is already in use, tunnel may already exist"
    echo "   If Backend is not running, please run:"
    echo "   ssh ${REMOTE_USER}@${REMOTE_HOST}"
    echo "   cd ${REMOTE_PATH}"
    echo "   python -m src.demo.backend"
else
    echo "Establishing SSH tunnel..."
    ssh -N -L ${LOCAL_PORT}:localhost:${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} &
    TUNNEL_PID=$!
    sleep 2
    
    if kill -0 $TUNNEL_PID 2>/dev/null; then
        echo "✅ SSH tunnel established (PID: $TUNNEL_PID)"
        echo ""
        echo "⚠️  Please start Backend on remote server:"
        echo "   ssh ${REMOTE_USER}@${REMOTE_HOST}"
        echo "   cd ${REMOTE_PATH}"
        echo "   python -m src.demo.backend"
        echo ""
        echo "Then access: http://localhost:5001"
        echo ""
        echo "Press Ctrl+C to stop SSH tunnel"
        wait $TUNNEL_PID
    else
        echo "❌ SSH tunnel establishment failed"
        exit 1
    fi
fi
