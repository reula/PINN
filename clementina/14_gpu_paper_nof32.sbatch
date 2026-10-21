#!/bin/bash
# ---------------------------------------------------------------------------
# FULL paper-preset DSGNAR on one Intel tile, with the float32 pre-pass SKIPPED
# (--float32-steps 0): the paper's 2^15/2^14 points, 60-60-60-60 SIREN, sketch 4000,
# batch 4096, 200 DSGNAR steps with the paper's 1e-10 radius floor, then the two
# 1000-iteration SSBroyden runs.
#
#     cd ~/Julia/PINN && sbatch -p gpunode -t 04:00:00 clementina/14_gpu_paper_nof32.sbatch
#
# Why skip the pre-pass: on this backend the float32 probe evaluation overflows
# (probe_losses ~1e12/NaN), so with the pre-pass every float32 step is rejected and the
# radius arrives at the float64 pass already collapsed.  float64 alone behaves like the
# A30/CPU (see README 8b-quater).
#
# Outputs /data/.../ldc/ldc_paper_nof32/: params_flat.npy + network.json (+ leaves),
# results.npz, summary.json, figures/.  Reference points: A30 a30-preset DSGNAR
# (loss 1.8e-07, rel L2 0.0093), Intel a30 with pre-pass (0.0114 via SSBroyden),
# Intel paper with pre-pass (DSGNAR stalled at 4.6e-4, job 1548247).
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_papernf32
#SBATCH --account=pci_143
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/oreula/Julia/PINN/clementina/papernof32_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/papernof32_%j.err

set -uo pipefail
CXXI_DIR="${CXXI_DIR:-/home/oreula/Julia/PINN/clementina}"
[ -f "$CXXI_DIR/env.sh" ] || { echo "FATAL: no $CXXI_DIR/env.sh" >&2; exit 1; }
# shellcheck source=env.sh
source "$CXXI_DIR/env.sh"
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
cxxi_pick_data
PY="$(cxxi_python)" || { echo "FATAL: no venv" >&2; exit 1; }
OUT="$CXXI_DATA/ldc/ldc_paper_nof32"
mkdir -p "$OUT"
echo "host $(hostname) | job ${SLURM_JOB_ID:-?} | $(date -u '+%F %T UTC')"
echo "outdir $OUT"
cxxi_pin_gpu "$PY"
"$PY" -u "$CXXI_DIR/ldc_run.py" --mode paper --float32-steps 0 \
      --outdir "$OUT" --label ldc_paper_nof32
echo "ldc_run exit $?"
