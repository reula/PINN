#!/bin/bash
# ---------------------------------------------------------------------------
# Step 5b: the production recipe on 64 CPU cores.  Same CLI and flags as
# 04_prod_run.sbatch; use it when bench.sh (06/07) says the CPU wins at the
# production batch size.
#
#     cd ~/Julia/PINN/clementina
#     sbatch 04b_prod_run_cpu.sbatch <label> <window_ic> <dsgnar_sketch> [--set k=v ...]
#
#     sbatch 04b_prod_run_cpu.sbatch cxxi_dsgnar_soft_all soft_all 0
#     sbatch 04b_prod_run_cpu.sbatch cxxi_dsgnar_hard_s1500 hard 1500
#
# Command-line Slurm options override the directives below (-t, -c, --mem).
# A run whose runs/<label>/report.md exists is skipped, so re-submitting the
# same label is harmless.
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_prodcpu
#SBATCH --account=pci_143
#SBATCH --partition=cpunode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=08:00:00
# Logs are absolute on purpose: Slurm resolves a relative --output against the
# directory you submit FROM, so a relative path scatters logs across the tree.
#SBATCH --output=/home/oreula/Julia/PINN/clementina/prodcpu_%x_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/prodcpu_%x_%j.err

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

# Slurm copies the login shell's Lmod environment into the job (ohpc gcc 13,
# openmpi5, libfabric on LD_LIBRARY_PATH).  The pip oneAPI runtime needs none of
# it, and a gcc13 libstdc++ ahead of the system one is exactly the kind of thing
# that makes a 500 MB plugin misbehave; the venv python has its own RPATH.
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
# CPU nodes have no GPU: force the CPU backend (JAX_PLATFORMS is honoured by 0.11).
export JAX_PLATFORMS=cpu
cxxi_pick_data

LABEL="${1:-}"; IC="${2:-}"; SKETCH="${3:-}"
if [ -z "$LABEL" ] || [ -z "$IC" ] || [ -z "$SKETCH" ]; then
    echo "usage: sbatch 04b_prod_run_cpu.sbatch <label> <window_ic> <sketch> [--set k=v ...]" >&2
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
echo "python $PY   JAX_PLATFORMS=$JAX_PLATFORMS   outdir $OUTDIR"
"$PY" - <<'PYEOF'
import sys, jax
print("backend", jax.default_backend(), "| devices", jax.devices())
if jax.default_backend() != "cpu":
    print("NOTE: not on the CPU backend; this script is for cpunode", file=sys.stderr)
PYEOF

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
