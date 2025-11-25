#!/bin/bash
# Simple script to start Backend on remote server

REMOTE_HOST="watgpu.cs.uwaterloo.ca"
REMOTE_USER="z6dong"
REMOTE_PATH="/u501/z6dong/Repo/ModelSearchDemo"
LOCAL_PORT=5002
REMOTE_PORT=5002

echo "=========================================="
echo "Start Remote Backend"
echo "=========================================="
echo ""

echo "Step 1: Establish SSH tunnel..."
echo "  Local port ${LOCAL_PORT} -> Remote port ${REMOTE_PORT}"
echo ""

# Start SSH tunnel (background)
ssh -N -L ${LOCAL_PORT}:localhost:${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} &
TUNNEL_PID=$!
sleep 2

if kill -0 $TUNNEL_PID 2>/dev/null; then
    echo "✅ SSH tunnel established (PID: $TUNNEL_PID)"
    echo ""
    echo "Step 2: Please run the following commands on remote server:"
    echo "----------------------------------------"
    echo "ssh ${REMOTE_USER}@${REMOTE_HOST}"
    echo "cd ${REMOTE_PATH}"
    echo "python -m src.demo.backend"
    echo "----------------------------------------"
    echo ""
    echo "Step 3: After Backend starts, access:"
    echo "  http://localhost:5001"
    echo ""
    echo "⚠️  Keep this terminal running to maintain SSH tunnel"
    echo "   Press Ctrl+C to stop tunnel"
    echo ""
    
    # Wait for user interrupt
    trap "kill $TUNNEL_PID 2>/dev/null; echo ''; echo 'SSH tunnel closed'; exit" INT TERM
    wait $TUNNEL_PID
else
    echo "❌ SSH tunnel establishment failed"
    echo "   Please check SSH connection and permissions"
    exit 1
fi
