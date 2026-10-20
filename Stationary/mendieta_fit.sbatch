#!/bin/bash
# ---------------------------------------------------------------------------
# scripts/fit_reference.py on one A30: the reference under a run's loss, and the network's
# representation ceiling (regression onto the rotated reference).
#   REF=runs/<r>/specref_nonlinear_fine.npz sbatch -J <name> mendieta_fit.sbatch runs/<run> [args...]
# ---------------------------------------------------------------------------
#SBATCH -p multi
#SBATCH -A ifeg
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=28G
#SBATCH -t 03:00:00
#SBATCH -J fit
#SBATCH -o logs/slurm_%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/Julia/PINN/Stationary}"
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax" MPLCONFIGDIR="$HOME/.cache/mpl"
echo "=== fit job ${SLURM_JOB_ID:-?} on $(hostname)  $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)"
run="$1"; shift
.venv/bin/python scripts/fit_reference.py "$run" --ref "${REF:?set REF}" "$@"
