#!/bin/bash
# ---------------------------------------------------------------------------
# Step 5: production run on one Intel GPU.  The command is the one from
# Evolution_try/scripts/mendieta_run.sbatch, with Clementina's account/partition
# and the /data filesystem for outputs.
#
#     cd ~/Julia/PINN/clementina
#     sbatch 04_prod_run.sbatch <label> <window_ic> <dsgnar_sketch> [--set k=v ...]
#
# e.g. the paper's soft_all cell with the vendored 1500-column sketch:
#
#     sbatch -t 24:00:00 04_prod_run.sbatch cxxi_dsgnar_soft_all soft_all 0
#     sbatch 04_prod_run.sbatch cxxi_dsgnar_hard_s1500 hard 1500
#
# Slurm directives on the command line win, so `sbatch -t 48:00:00 ...` or
# `sbatch --gpus=2 ...` (only if you shard the code yourself -- it is single
# device) override the defaults below.
#
# A run whose runs/<label>/report.md exists is skipped, as in the Mendieta
# script, so re-submitting the same label is harmless.
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_prod
#SBATCH --account=pci_143
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
# Logs are absolute on purpose: Slurm resolves a relative --output against the
# directory you submit FROM, so a relative path scatters logs across the tree.
#SBATCH --output=/home/oreula/Julia/PINN/clementina/prod_%x_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/prod_%x_%j.err

set -uo pipefail

# Slurm copies this script to /var/spool/slurmd/job<id>/slurm_script, so
# dirname "$BASH_SOURCE" is the spool directory, NOT this one.  Locate env.sh by
# absolute path (override with CXXI_DIR=... or sbatch --export=ALL,CXXI_DIR=...),
# falling back to whatever directory the job was submitted from.
CXXI_DIR="${CXXI_DIR:-/home/oreula/Julia/PINN/clementina}"
if [ ! -f "$CXXI_DIR/env.sh" ]; then
    for d in "${SLURM_SUBMIT_DIR:-$PWD}/clementina" "${SLURM_SUBMIT_DIR:-$PWD}" \
             "$PWD/clementina" "$PWD"; do
        if [ -f "$d/env.sh" ]; then CXXI_DIR="$(cd "$d" && pwd)"; break; fi
    done
fi
if [ ! -f "$CXXI_DIR/env.sh" ]; then
    echo "FATAL: cannot find env.sh; set CXXI_DIR=/path/to/clementina" >&2
    exit 1
fi
# shellcheck source=env.sh
source "$CXXI_DIR/env.sh"
cxxi_pick_data

LABEL="${1:-}"; IC="${2:-}"; SKETCH="${3:-}"
if [ -z "$LABEL" ] || [ -z "$IC" ] || [ -z "$SKETCH" ]; then
    echo "usage: sbatch 04_prod_run.sbatch <label> <window_ic> <sketch> [--set k=v ...]" >&2
    exit 1
fi
shift 3

RUNROOT="${RUNROOT:-$CXXI_DATA/$CXXI_PROJECT}"
PROJ="$CXXI_REPO/$CXXI_PROJECT"
PY="$(cxxi_python)" || { echo "FATAL: no venv with jax; run 01_setup_env.sh first" >&2; exit 1; }

cd "$PROJ" || { echo "FATAL: $PROJ not found (git clone the repo first)" >&2; exit 1; }
mkdir -p "$RUNROOT/runs" "$RUNROOT/logs"
OUTDIR="$RUNROOT/runs/$LABEL"

echo "=== $LABEL  job ${SLURM_JOB_ID:-?} on $(hostname)  $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)"
echo "python $PY   outdir $OUTDIR"
command -v xpu-smi >/dev/null 2>&1 && xpu-smi discovery 2>&1 | head -20 || echo "xpu-smi not on this node"
echo

# A JAX that cannot see the GPU falls back to the CPU silently and the run then
# takes days: refuse to start instead.
cxxi_assert_accelerator "$PY" || { echo "FATAL: no Intel GPU visible to JAX" >&2; exit 2; }

if [ -f "$OUTDIR/report.md" ]; then
    echo "runs/$LABEL/report.md exists: done already"; exit 0
fi
rm -f "$OUTDIR/dsgnar_stop_audits.jsonl"      # a previous, interrupted attempt

time "$PY" -m wave_pinn.cli --label "$LABEL" --outdir "$OUTDIR" \
    --set T=20 --set windows=10 --set ansatz=t2 \
    --set window_ic="$IC" --set w_ic=100 \
    --set n_coll=2201 --set sampler=random --set resample_every=0 \
    --set "snapshot_times=[0,2,4,6,8,10,12,14,16,18,20]" \
    --set optimizer=dsgnar --set dsgnar_steps=300 --set dsgnar_sketch="$SKETCH" \
    --set dsgnar_delta0=1.0 --set dsgnar_delta_min=1e-14 --set resample_rounds=5 \
    --set dsgnar_audit_stops=true "$@"
rc=$?
echo "=== $LABEL exit $rc  $(date -u '+%F %T UTC')"
exit $rc
