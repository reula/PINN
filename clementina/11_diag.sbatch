#!/bin/bash
# ---------------------------------------------------------------------------
# DSGNAR numerical diagnostic on one pinned Intel GPU (read-only: no training,
# no SSBroyden).  Runs the two probes that localise the rho = NaN / radius-collapse
# problem seen in job 1549369 (Intel) but absent on the A30 and on CPU:
#
#   ldc_nan_probe.py   staged NaN census: dense Jacobian -> sketch B -> SVD ->
#                      secular solve -> predicted reductions, in float32 and float64
#   ldc_step_probe.py  ONE real DSGNAR step at the a30 initial state, dumping
#                      probe_losses / probe_rhos / lam / S / pchip_y
#
#     cd ~/Julia/PINN
#     sbatch -p gpunode -t 00:30:00 clementina/11_diag.sbatch
#
# Outputs: /data/contrib/pci_143/pinn/ldc/diag_intel/{nan_probe.json,step_probe.json,
# step_metrics.npz} and clementina/diag_<jobid>.out.
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_diag
#SBATCH --account=pci_143
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/oreula/Julia/PINN/clementina/diag_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/diag_%j.err

set -uo pipefail
CXXI_DIR="${CXXI_DIR:-/home/oreula/Julia/PINN/clementina}"
[ -f "$CXXI_DIR/env.sh" ] || { echo "FATAL: no $CXXI_DIR/env.sh" >&2; exit 1; }
# shellcheck source=env.sh
source "$CXXI_DIR/env.sh"
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
cxxi_pick_data
PY="$(cxxi_python)" || { echo "FATAL: no venv; run 01_setup_env.sh" >&2; exit 1; }
OUT="$CXXI_DATA/ldc/diag_intel"
mkdir -p "$OUT"

echo "host $(hostname) | job ${SLURM_JOB_ID:-?} | $(date -u '+%F %T UTC')"
cxxi_pin_gpu "$PY"
echo

echo "===== ldc_nan_probe.py ====="
"$PY" -u "$CXXI_DIR/ldc_nan_probe.py" --outdir "$OUT"
echo "nan_probe exit $?"
echo "===== ldc_step_probe.py --bits 32 ====="
"$PY" -u "$CXXI_DIR/ldc_step_probe.py" --outdir "$OUT" --bits 32
echo "step_probe(32) exit $?"
echo "===== ldc_step_probe.py --bits 64 ====="
"$PY" -u "$CXXI_DIR/ldc_step_probe.py" --outdir "$OUT" --bits 64
echo "step_probe(64) exit $?"
echo "===== done $(date -u '+%F %T UTC') ====="
