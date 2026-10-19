#!/bin/bash
# ---------------------------------------------------------------------------
# Step 4b: the same tiny run as 03_tiny_run.sbatch, on a CPU node.
#
# Two reasons to run it:
#   * correctness: same seed + same float64 flags as the GPU run, so the losses
#     must agree to rounding (the 0.11.1 -> 0.11.2 patch bump and the SYCL
#     backend can both move the last digits; a disagreement bigger than ~1e-8 in
#     the loss is a code problem, not a rounding problem);
#   * performance: how much the Intel GPU is actually worth here.  A 64-core
#     Xeon Max node is not a bad fallback if the experimental plugin turns out to
#     be slower or unstable for your access pattern.
#
#     cd ~/Julia/PINN/clementina
#     sbatch 05_cpu_tiny.sbatch
#     tail -n +1 tinycpu_*.out
#
# Note that cpunode can also run on the GPU nodes (they are all in cpunodes per
# the wiki), so this is cheap to get scheduled.
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_tinycpu
#SBATCH --account=pci_143
#SBATCH --partition=cpunode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --output=tinycpu_%j.out
#SBATCH --error=tinycpu_%j.err

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
cxxi_pick_data

export JAX_PLATFORMS=cpu
LABEL="${LABEL:-cxxi_tiny_cpu}"
RUNROOT="${RUNROOT:-$CXXI_DATA/$CXXI_PROJECT}"
PROJ="$CXXI_REPO/$CXXI_PROJECT"
PY="$(cxxi_python)" || { echo "FATAL: no venv with jax; run 01_setup_env.sh first" >&2; exit 1; }

cd "$PROJ" || { echo "FATAL: $PROJ not found (git clone the repo first)" >&2; exit 1; }
mkdir -p "$RUNROOT/runs" logs

echo "host      $(hostname)   $(date -u '+%F %T UTC')"
echo "job       ${SLURM_JOB_ID:-?}  partition ${SLURM_JOB_PARTITION:-?}  cpus ${SLURM_CPUS_PER_TASK:-?}"
echo "python    $PY   JAX_PLATFORMS=$JAX_PLATFORMS"
echo "outdir    $RUNROOT/runs/$LABEL"
"$PY" -c "import jax; print('devices', jax.devices(), 'backend', jax.default_backend())"
echo

if [ -f "$RUNROOT/runs/$LABEL/report.md" ]; then
    echo "runs/$LABEL/report.md exists: already done, exiting"
    exit 0
fi

time "$PY" -m wave_pinn.cli \
    --label "$LABEL" --outdir "$RUNROOT/runs/$LABEL" \
    --set T=2 --set windows=1 --set ansatz=t2 \
    --set window_ic=hard --set w_ic=100 \
    --set n_coll=512 --set sampler=random \
    --set "snapshot_times=[0,1,2]" \
    --set optimizer=dsgnar --set dsgnar_steps=10 --set dsgnar_sketch=256 \
    --set dsgnar_delta0=1.0 --set dsgnar_delta_min=1e-14 --set resample_rounds=1 \
    "$@"
rc=$?
echo
echo "=== $LABEL exit $rc   $(date -u '+%F %T UTC')"
exit $rc
