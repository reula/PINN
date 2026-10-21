#!/bin/bash
# ---------------------------------------------------------------------------
# Exact-residual (rows) replay on one Intel GPU: six float64 steps (+ the four
# float32 steps) with the CountSketch replaced by the exact Gram system.
# Tests whether DSGNAR's rho settles at its 0.075 target on this backend, as it
# does on CPU/A30, instead of NaN / erratic.
#
#     cd ~/Julia/PINN && sbatch -p gpunode -t 00:30:00 clementina/16_gpu_replay_exact.sbatch
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_repex
#SBATCH --account=pci_143
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/oreula/Julia/PINN/clementina/repex_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/repex_%j.err

set -uo pipefail
CXXI_DIR="${CXXI_DIR:-/home/oreula/Julia/PINN/clementina}"
[ -f "$CXXI_DIR/env.sh" ] || { echo "FATAL: no $CXXI_DIR/env.sh" >&2; exit 1; }
# shellcheck source=env.sh
source "$CXXI_DIR/env.sh"
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
cxxi_pick_data
PY="$(cxxi_python)" || { echo "FATAL: no venv" >&2; exit 1; }
OUT="$CXXI_DATA/ldc/replay_exact"
mkdir -p "$OUT"
echo "host $(hostname) | job ${SLURM_JOB_ID:-?} | $(date -u '+%F %T UTC')"
cxxi_pin_gpu "$PY"
"$PY" -u "$CXXI_DIR/ldc_replay.py" --outdir "$OUT" \
      --float32-steps 0 --float64-steps 8 --exact-residual
echo "replay exit $?"
