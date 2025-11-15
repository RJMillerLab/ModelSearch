#!/bin/bash
#SBATCH --job-name=modelsearch_backend
#SBATCH --output=logs/backend_%j.out
#SBATCH --error=logs/backend_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1  # Remove if no GPU needed

# Load conda environment if needed
# source ~/.bashrc
# conda activate your_env_name

# Get the hostname
HOSTNAME=$(hostname)

# Print connection info
echo "=========================================="
echo "ModelSearch Backend Starting"
echo "=========================================="
echo "Hostname: $HOSTNAME"
echo "Port: 5000"
echo "=========================================="
echo ""
echo "To access from your local machine, create an SSH tunnel:"
echo "  ssh -L 5000:localhost:5000 your_username@watgpu.cs.uwaterloo.ca"
echo "  (If on compute node: ssh -L 5000:$HOSTNAME:5000 your_username@watgpu.cs.uwaterloo.ca)"
echo ""
echo "Then access at: http://localhost:5000"
echo "=========================================="
echo ""

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Run backend (will run on port 5000 by default)
python -m src.demo.backend

