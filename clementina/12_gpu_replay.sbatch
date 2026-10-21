#!/bin/bash
# ---------------------------------------------------------------------------
# Confirm that skipping the float32 pre-pass fixes the Intel DSGNAR stall:
# replay the first six float64 steps with --float32-steps 0.
#
#     cd ~/Julia/PINN && sbatch -p gpunode -t 00:30:00 clementina/12_gpu_replay.sbatch
#
# Expect rho ~0.9 and a falling loss (the A30/CPU behaviour), versus rho = NaN
# rejections when the float32 pass runs.  Output: /data/.../ldc/replay_nof32/.
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_replay
#SBATCH --account=pci_143
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/oreula/Julia/PINN/clementina/replay_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/replay_%j.err

set -uo pipefail
CXXI_DIR="${CXXI_DIR:-/home/oreula/Julia/PINN/clementina}"
[ -f "$CXXI_DIR/env.sh" ] || { echo "FATAL: no $CXXI_DIR/env.sh" >&2; exit 1; }
# shellcheck source=env.sh
source "$CXXI_DIR/env.sh"
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
cxxi_pick_data
PY="$(cxxi_python)" || { echo "FATAL: no venv" >&2; exit 1; }
OUT="$CXXI_DATA/ldc/replay_nof32"
mkdir -p "$OUT"
echo "host $(hostname) | job ${SLURM_JOB_ID:-?} | $(date -u '+%F %T UTC')"
cxxi_pin_gpu "$PY"
"$PY" -u "$CXXI_DIR/ldc_replay.py" --outdir "$OUT" --float32-steps 0 --float64-steps 6
echo "replay exit $?"
