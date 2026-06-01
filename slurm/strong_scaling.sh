#!/bin/bash
# STRONG SCALING: fixed problem, growing resources -> measure speed-up.
#
# The problem (size + tol) is held constant, so the total work -- O(L^2 * nq)
# Hadamard-test circuits -- is fixed. We sweep the MPI rank count so the circuits
# are spread over more GPUs each step; runtime per cost evaluation should drop.
#
# This script runs the sweep *within a single multi-node allocation* by launching
# successive srun steps with different --ntasks-per-node. Compare the
# "avg/eval" line printed by each step.
#
#   sbatch slurm/strong_scaling.sh
#
#SBATCH -N 2
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH -q regular
#SBATCH -A m5097
#SBATCH -t 2:00:00
#SBATCH --job-name=vqls_strong
#SBATCH --output=vqls_strong_%j.out

set -euo pipefail

RELEASE_DIR=/global/cfs/cdirs/m5097/Chaol/VQLS_CUDAQ/vqls_cudaq
CONDA_ENV=/global/cfs/cdirs/m5097/Chaol/VQLS_CUDAQ/.conda

CASE=tridiagonal
SIZE=10
TOL=0.01
# A few iterations is enough to get a stable per-evaluation time.
MAXITER=5

module load conda cudatoolkit
conda activate "${CONDA_ENV}"
cd "${RELEASE_DIR}"

python generate_lcu.py --case "${CASE}" --size "${SIZE}" --tol "${TOL}"

# Total ranks across all nodes (2 nodes * tasks-per-node). With --gpu-bind=none
# each rank exposes all 4 GPUs as virtual QPUs; oversubscribing ranks per GPU
# (e.g. 16 tasks/node on 4 GPUs) raises throughput up to a saturation point.
for TPN in 4 8 16 32; do
    echo "======== strong scaling: ${SLURM_NNODES} nodes x ${TPN} ranks/node ========"
    srun --ntasks-per-node="${TPN}" --gpu-bind=none \
        python -u run_vqls.py --case "${CASE}" --size "${SIZE}" --tol "${TOL}" \
            --backend dense --maxiter "${MAXITER}"
done
