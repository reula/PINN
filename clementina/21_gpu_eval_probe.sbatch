#!/bin/bash
# Probe the GPU memory/time of the reference-grid evaluation vs batch size (2 A30s).
#SBATCH -J ldc_evalp
#SBATCH -p multi
#SBATCH -A ifeg
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -o logs/mendieta_evalprobe_%j.out
#SBATCH -e logs/mendieta_evalprobe_%j.err
set -uo pipefail
REPO="${LDC_REPO:-$HOME/ldc_paper_run}"
cd "$REPO" || exit 1
mkdir -p logs
export PY="${LDC_PY:-$HOME/Julia/PINN/Stationary/.venv/bin/python}"
source Evolution_try/scripts/env.sh
PY="$(find_python)" || { echo "FATAL: no interpreter with jax" >&2; exit 1; }
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export MPLCONFIGDIR="$HOME/.cache/mpl"
echo "host $(hostname) | job ${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PY" -u clementina/ldc_eval_probe.py
