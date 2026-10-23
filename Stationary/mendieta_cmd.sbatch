#!/bin/bash
# ---------------------------------------------------------------------------
# Run one command (a python script and its arguments) on one A30, in the Stationary venv.
#   sbatch -J <name> mendieta_cmd.sbatch scripts/<script>.py [args...]
# ---------------------------------------------------------------------------
#SBATCH -p multi
#SBATCH -A ifeg
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=28G
#SBATCH -t 04:00:00
#SBATCH -J cmd
#SBATCH -o logs/slurm_%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/Julia/PINN/Stationary}"
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax" MPLCONFIGDIR="$HOME/.cache/mpl"
echo "=== cmd job ${SLURM_JOB_ID:-?} on $(hostname)  $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD 2>/dev/null)"
echo "+ $*"
.venv/bin/python "$@"
