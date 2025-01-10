#!/bin/bash

###############################################################################
# SLURM Directives
###############################################################################
#SBATCH -A CPH161                # SLURM account
#SBATCH -J HydraGNN-Optuna       # Job name
#SBATCH -o job-%j.out            # Standard output (stdout)
#SBATCH -e job-%j.out            # Standard error  (stderr)
#SBATCH -t 0:30:00               # Max wall time
#SBATCH -p batch                 # Partition (queue)
#SBATCH -N 4                     # Number of nodes

###############################################################################
# Environment Setup
###############################################################################
set -x                            # Echo commands for debugging

# Disable caching to avoid some ROCm issues
export MIOPEN_DISABLE_CACHE=1

# Offline mode for certain libraries if needed
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Debug info for NCCL
export NCCL_DEBUG=INFO

# (Optional) Set HF cache directory
export HF_HOME=$PWD/hfdata

# Create a host file (optional, for distributed usage)
HOSTS=.hosts-job$SLURM_JOB_ID
HOSTFILE=hostfile.txt
srun hostname > $HOSTS
sed 's/$/ slots=8/' $HOSTS > $HOSTFILE

# Example environment variables for HPC runs
export NNODES=$SLURM_JOB_NUM_NODES
export OMP_NUM_THREADS=4

# (Optional) Sleep to ensure resources are ready
sleep 5

###############################################################################
# Launch the Optuna Script
###############################################################################
echo "Launching MD17 Optuna workflow..."

# If you have a single entry point for the entire run:
# python md17_optuna.py

# Alternatively, if you need srun for each rank or plan to run distributed:
srun -n 1 python md17_optuna.py

echo "Done."
