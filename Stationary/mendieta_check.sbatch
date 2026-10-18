#!/bin/bash
# ---------------------------------------------------------------------------
# The resampling-stability check (scripts/fresh_sample_loss.py) for finished runs, on one A30.
#   sbatch mendieta_check.sbatch runs/<run> [runs/<run> ...]
# ---------------------------------------------------------------------------
#SBATCH -p multi
#SBATCH -A ifeg
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=28G
#SBATCH -t 01:00:00
#SBATCH -J check
#SBATCH -o logs/slurm_%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/Julia/PINN/Stationary}"
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax" MPLCONFIGDIR="$HOME/.cache/mpl"
echo "=== check job ${SLURM_JOB_ID:-?} on $(hostname)  $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)"
rc=0
for run in "$@"; do
    .venv/bin/python scripts/fresh_sample_loss.py "$run" --n 5 || rc=$?
done
exit $rc
