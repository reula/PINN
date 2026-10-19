#!/bin/bash
# ---------------------------------------------------------------------------
# Step 4: a small end-to-end run of the real code on one Intel GPU.
#
# It is the same wave_pinn CLI, the same DSGNAR optimiser and the same float64
# precision as the production recipe, just with T=2, 512 collocation points and
# 10 DSGNAR iterations, so it compiles the whole graph in a few minutes.
#
#     cd ~/Julia/PINN/clementina
#     sbatch 03_tiny_run.sbatch
#     tail -n +1 tiny_*.out
#     cat /data/contrib/pci_143/pinn/Evolution_try/runs/cxxi_tiny/report.md
#
# Compare its numbers with 05_cpu_tiny.sbatch (same seed, same flags, CPU
# backend): a GPU run that is much faster and agrees to ~1e-8 with the CPU one
# is a working port.
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_tiny
#SBATCH --account=pci_143
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=00:40:00
# Logs are absolute on purpose: Slurm resolves a relative --output against the
# directory you submit FROM, so a relative path scatters logs across the tree.
#SBATCH --output=/home/oreula/Julia/PINN/clementina/tiny_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/tiny_%j.err

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
cxxi_pick_data

LABEL="${LABEL:-cxxi_tiny}"
RUNROOT="${RUNROOT:-$CXXI_DATA/$CXXI_PROJECT}"
PROJ="$CXXI_REPO/$CXXI_PROJECT"
PY="$(cxxi_python)" || { echo "FATAL: no venv with jax; run 01_setup_env.sh first" >&2; exit 1; }

cd "$PROJ" || { echo "FATAL: $PROJ not found (git clone the repo first)" >&2; exit 1; }
mkdir -p "$RUNROOT/runs" logs

echo "host      $(hostname)   $(date -u '+%F %T UTC')"
echo "job       ${SLURM_JOB_ID:-?}  partition ${SLURM_JOB_PARTITION:-?}"
echo "head      $(git rev-parse --short HEAD 2>/dev/null)"
echo "python    $PY"
echo "outdir    $RUNROOT/runs/$LABEL"
xpu-smi discovery 2>&1 | head -20 || true
echo

cxxi_assert_accelerator "$PY" || { echo "FATAL: no Intel GPU visible to JAX" >&2; exit 2; }
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
