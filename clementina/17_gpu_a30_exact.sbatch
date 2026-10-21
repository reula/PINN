#!/bin/bash
# ---------------------------------------------------------------------------
# a30-preset DSGNAR on one Intel GPU with the exact residual (rows) mode: the run
# that should reproduce the A30's rho-at-target behaviour and e-7-class loss on this
# backend, instead of stalling at 4.6e-4 / rel L2 0.57-0.66.
#
#     cd ~/Julia/PINN && sbatch -p gpunode -t 01:30:00 clementina/17_gpu_a30_exact.sbatch
#
# Outputs /data/.../ldc/ldc_a30_exact/: params_flat.npy + network.json (+ DSGNAR
# checkpoint), results.npz, summary.json, figures/.
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_a30ex
#SBATCH --account=pci_143
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/oreula/Julia/PINN/clementina/a30ex_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/a30ex_%j.err

set -uo pipefail
CXXI_DIR="${CXXI_DIR:-/home/oreula/Julia/PINN/clementina}"
[ -f "$CXXI_DIR/env.sh" ] || { echo "FATAL: no $CXXI_DIR/env.sh" >&2; exit 1; }
# shellcheck source=env.sh
source "$CXXI_DIR/env.sh"
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
cxxi_pick_data
PY="$(cxxi_python)" || { echo "FATAL: no venv" >&2; exit 1; }
OUT="$CXXI_DATA/ldc/ldc_a30_exact"
mkdir -p "$OUT"
echo "host $(hostname) | job ${SLURM_JOB_ID:-?} | $(date -u '+%F %T UTC')"
cxxi_pin_gpu "$PY"
"$PY" -u "$CXXI_DIR/ldc_run.py" --mode a30 --exact-residual --float32-steps 0 \
      --outdir "$OUT" --label ldc_a30_exact
echo "ldc_run exit $?"
