#!/bin/bash
# Single Perlmutter GPU node (4x A100): basic VQLS run, one MPI rank per GPU.
#
#   sbatch slurm/single_node.sh
#
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH -q regular
#SBATCH -A m5097
#SBATCH -t 1:00:00
#SBATCH --job-name=vqls_single
#SBATCH --output=vqls_single_%j.out

set -euo pipefail

# Edit these two paths if you relocate the package or the conda env.
RELEASE_DIR=/global/cfs/cdirs/m5097/Chaol/VQLS_CUDAQ/vqls_cudaq
CONDA_ENV=/global/cfs/cdirs/m5097/Chaol/VQLS_CUDAQ/.conda

CASE=tridiagonal
SIZE=10
TOL=0.01

module load conda cudatoolkit
conda activate "${CONDA_ENV}"
cd "${RELEASE_DIR}"

# Generate the LCU dataset on the fly if it is missing (CPU-only step).
python generate_lcu.py --case "${CASE}" --size "${SIZE}" --tol "${TOL}"

# 4 ranks, each sees all 4 GPUs (mqpu -> 4 virtual QPUs per rank).
srun --ntasks-per-node=4 --gpu-bind=none \
    python -u run_vqls.py --case "${CASE}" --size "${SIZE}" --tol "${TOL}" \
        --backend dense --layers 7 --maxiter 1000
