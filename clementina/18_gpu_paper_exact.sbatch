#!/bin/bash
# ---------------------------------------------------------------------------
# FULL paper preset with the exact-residual fix and a float64 start:
# 2^15/2^14 points, 60-60-60-60 SIREN (11347 params), sketch 4000, batch 4096,
# probe 256, 200 DSGNAR steps (radius floor 1e-10), then the two 1000-iteration
# SSBroyden runs.  This is the run that should give the paper-configuration result
# on Clementina; the a30 analogue (1549619) reached rel L2 0.008634 in 52 min.
#
#     cd ~/Julia/PINN && sbatch -p gpunode -t 06:00:00 clementina/18_gpu_paper_exact.sbatch
#
# A periodic DSGNAR checkpoint (every 20 steps) lands in the outdir, so a time
# limit cannot lose the run.  Outputs /data/.../ldc/ldc_paper_exact/.
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_paperex
#SBATCH --account=pci_143
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/oreula/Julia/PINN/clementina/paperex_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/paperex_%j.err

set -uo pipefail
CXXI_DIR="${CXXI_DIR:-/home/oreula/Julia/PINN/clementina}"
[ -f "$CXXI_DIR/env.sh" ] || { echo "FATAL: no $CXXI_DIR/env.sh" >&2; exit 1; }
# shellcheck source=env.sh
source "$CXXI_DIR/env.sh"
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
cxxi_pick_data
PY="$(cxxi_python)" || { echo "FATAL: no venv" >&2; exit 1; }
OUT="$CXXI_DATA/ldc/ldc_paper_exact"
mkdir -p "$OUT"
echo "host $(hostname) | job ${SLURM_JOB_ID:-?} | $(date -u '+%F %T UTC')"
echo "outdir $OUT"
cxxi_pin_gpu "$PY"
"$PY" -u "$CXXI_DIR/ldc_run.py" --mode paper --exact-residual --float32-steps 0 \
      --outdir "$OUT" --label ldc_paper_exact
echo "ldc_run exit $?"
