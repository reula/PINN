#!/bin/bash
# ---------------------------------------------------------------------------
# The a30-preset DSGNAR on the Intel tile with the float32 pre-pass SKIPPED
# (--float32-steps 0), SSBroyden trimmed to one iteration: this is the run that
# should reproduce the A30's e-7-class DSGNAR result instead of stalling at 4.6e-4.
#
#     cd ~/Julia/PINN && sbatch -p gpunode -t 01:30:00 clementina/13_gpu_a30_nof32.sbatch
#
# Outputs /data/.../ldc/ldc_a30_nof32/: params_flat.npy + network.json (+ eqx
# leaves), results.npz, summary.json, figures/.  Compares against
# ldc/params/hub_a30_dsgnar_48x48x48.* (A30: loss 1.8e-07, rel L2 0.00927) and
# ldc/params/intel_a30_48x48x48_best.* (Intel with the pre-pass: rel L2 0.01139).
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_a30nof32
#SBATCH --account=pci_143
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/oreula/Julia/PINN/clementina/a30nof32_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/a30nof32_%j.err

set -uo pipefail
CXXI_DIR="${CXXI_DIR:-/home/oreula/Julia/PINN/clementina}"
[ -f "$CXXI_DIR/env.sh" ] || { echo "FATAL: no $CXXI_DIR/env.sh" >&2; exit 1; }
# shellcheck source=env.sh
source "$CXXI_DIR/env.sh"
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
cxxi_pick_data
PY="$(cxxi_python)" || { echo "FATAL: no venv" >&2; exit 1; }
OUT="$CXXI_DATA/ldc/ldc_a30_nof32"
mkdir -p "$OUT"
echo "host $(hostname) | job ${SLURM_JOB_ID:-?} | $(date -u '+%F %T UTC')"
cxxi_pin_gpu "$PY"
"$PY" -u "$CXXI_DIR/ldc_run.py" --mode a30 --float32-steps 0 --qn-maxiter 1 \
      --outdir "$OUT" --label ldc_a30_nof32
echo "ldc_run exit $?"
