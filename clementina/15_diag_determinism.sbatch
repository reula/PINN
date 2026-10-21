#!/bin/bash
# ---------------------------------------------------------------------------
# Is the Intel DSGNAR rho instability caused by nondeterministic reductions?
#
# The sketch is rank-deficient (~258 of 1665 singular values above 1e-12*max), the whole
# step is jit-fused, and the plugin offers XLA's determinism switches.  This job:
#   1. replays the same six float64 steps TWICE (identical keys) -> are the metrics
#      bit-identical run to run on this device?
#   2. replays once more with XLA_FLAGS="--xla_gpu_deterministic_ops=true ...";
#      if rho becomes well-behaved (settling at the 0.07 target like CPU/A30) that flag
#      is the fix, not just a diagnostic.
#
#     cd ~/Julia/PINN && sbatch -p gpunode -t 00:30:00 clementina/15_diag_determinism.sbatch
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_determ
#SBATCH --account=pci_143
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/oreula/Julia/PINN/clementina/determ_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/determ_%j.err

set -uo pipefail
CXXI_DIR="${CXXI_DIR:-/home/oreula/Julia/PINN/clementina}"
[ -f "$CXXI_DIR/env.sh" ] || { echo "FATAL: no $CXXI_DIR/env.sh" >&2; exit 1; }
# shellcheck source=env.sh
source "$CXXI_DIR/env.sh"
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
cxxi_pick_data
PY="$(cxxi_python)" || { echo "FATAL: no venv" >&2; exit 1; }
BASE="$CXXI_DATA/ldc"
mkdir -p "$BASE"

echo "host $(hostname) | job ${SLURM_JOB_ID:-?} | $(date -u '+%F %T UTC')"
cxxi_pin_gpu "$PY"
echo

run_replay() {  # run_replay <tag>
    local tag="$1"
    echo "########## replay: $tag   XLA_FLAGS=${XLA_FLAGS:-<unset>}"
    "$PY" -u "$CXXI_DIR/ldc_replay.py" --outdir "$BASE/$tag" \
          --float32-steps 0 --float64-steps 6
    echo "exit $?"
    echo
}

run_replay replay_default_a
run_replay replay_default_b
XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0" \
    run_replay replay_deterministic
echo "===== done $(date -u '+%F %T UTC') ====="
