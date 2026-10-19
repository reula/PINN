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
# Logs are absolute on purpose: Slurm resolves a relative --output against the
# directory you submit FROM, so a relative path scatters logs across the tree.
#SBATCH --output=/home/oreula/Julia/PINN/clementina/tinycpu_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/tinycpu_%j.err

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
