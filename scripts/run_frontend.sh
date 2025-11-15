#!/bin/bash
#SBATCH --job-name=modelsearch_frontend
#SBATCH --output=logs/frontend_%j.out
#SBATCH --error=logs/frontend_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Load conda environment if needed
# source ~/.bashrc
# conda activate your_env_name

# Get the hostname
HOSTNAME=$(hostname)

# Get backend hostname (if backend is on different node, set this)
# Otherwise, use localhost if backend is on same node
BACKEND_HOSTNAME=${BACKEND_HOSTNAME:-localhost}

# Print connection info
echo "=========================================="
echo "ModelSearch Frontend Starting"
echo "=========================================="
echo "Hostname: $HOSTNAME"
echo "Port: 5001"
echo "Backend: $BACKEND_HOSTNAME:5000"
echo "=========================================="
echo ""
echo "To access from your local machine, create an SSH tunnel:"
echo "  ssh -L 5000:localhost:5000 -L 5001:localhost:5001 your_username@watgpu.cs.uwaterloo.ca"
if [ "$BACKEND_HOSTNAME" != "localhost" ]; then
    echo "  (If on compute nodes: ssh -L 5000:$BACKEND_HOSTNAME:5000 -L 5001:$HOSTNAME:5001 your_username@watgpu.cs.uwaterloo.ca)"
fi
echo ""
echo "Then open http://localhost:5001 in your local browser"
echo "=========================================="
echo ""

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Note: Frontend will connect to backend at http://localhost:5000 by default
# If backend is on different node, you need to modify frontend.py or use SSH tunnel
# Run frontend
python -m src.demo.frontend

