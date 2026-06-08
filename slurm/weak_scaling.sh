#!/bin/bash
# WEAK SCALING: grow the problem and the resources together -> ~constant
# work per GPU. Run this on a LOGIN NODE (it submits jobs); do NOT sbatch it:
#
#   bash slurm/weak_scaling.sh
#
# The solver's work scales as O(L^2) in the number of LCU terms L. A smaller LCU
# tolerance keeps more terms, so L (and the work) grows. We pair each tolerance
# with a node count so work-per-node stays roughly flat, then submit one job each.
# Compare the "avg/eval" times across jobs: flat == ideal weak scaling.

set -euo pipefail

RELEASE_DIR=/global/cfs/cdirs/m5097/Chaol/DVQLS/DVQLS
CONDA_ENV=/global/cfs/cdirs/m5097/Chaol/DVQLS/.conda

CASE=tridiagonal
SIZE=10
MAXITER=5

# Tolerance levels (decreasing -> more LCU terms -> more work) paired with the
# node counts that absorb that work. Tune the pairing to your problem.
TOLS=(0.1 0.05 0.02 0.01 0.005)
NODES=(1 1 2 4 8)

for i in "${!TOLS[@]}"; do
    TOL="${TOLS[$i]}"
    N="${NODES[$i]}"
    SAFE_TOL="${TOL//./p}"      # 0.01 -> 0p01 for clean filenames

    sbatch <<EOF
#!/bin/bash
#SBATCH -N ${N}
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH -q regular
#SBATCH -A m5097
#SBATCH -t 1:00:00
#SBATCH --job-name=dvqls_weak_t${SAFE_TOL}
#SBATCH --output=dvqls_weak_t${SAFE_TOL}_%j.out

set -euo pipefail
module load conda cudatoolkit
conda activate "${CONDA_ENV}"
cd "${RELEASE_DIR}"

python generate_lcu.py --case ${CASE} --size ${SIZE} --tol ${TOL}

srun --ntasks-per-node=16 --gpu-bind=none \
    python -u run_dvqls.py --case ${CASE} --size ${SIZE} --tol ${TOL} \
        --backend dense --maxiter ${MAXITER}
EOF
    echo "submitted: tol=${TOL} on ${N} node(s)"
    sleep 1
done
