#!/bin/bash
# ---------------------------------------------------------------------------
# The LDC notebook (ldc/ldc.ipynb) as a batch job on one Intel GPU:
# DSGNAR (the paper's optimiser, vendored) vs SSBroyden (Crunch), graded against
# the reference field at Re = 100, with the notebook's figures.
#
#     cd ~/Julia/PINN
#     sbatch clementina/08_ldc.sbatch                       # a30 preset
#     sbatch --export=ALL,MODE=paper clementina/08_ldc.sbatch
#     sbatch --export=ALL,MODE=smoke clementina/08_ldc.sbatch   # pipeline check, minutes
#
# Outputs: /data/contrib/pci_143/pinn/ldc/ldc_<mode>/
#     figures/optimizers_loss_and_relL2.png, figures/fields_{u,v,p}.png,
#     results.npz, history.json, summary.json, dsgnar_flat.npy
# The job log is clementina/ldc_<jobid>.out (absolute path in the directives).
#
# Presets (memory driver: DSGNAR's dense Jacobian, batch*3*d*8 bytes):
#     smoke   2^11 points, sketch 256  -> ~minutes
#     a30     2^13 points, sketch 1665 -> the hub's A30 preset
#     paper   2^15 points, sketch 4000 -> the paper's numbers, needs headroom
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_ldc
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
#SBATCH --output=/home/oreula/Julia/PINN/clementina/ldc_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/ldc_%j.err

set -uo pipefail

# Slurm copies this script to /var/spool/slurmd/job<id>/slurm_script, so
# dirname "$BASH_SOURCE" is the spool directory, NOT this one.
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
# that makes a 500 MB plugin misbehave.
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
cxxi_pick_data

MODE="${MODE:-a30}"
LABEL="${LABEL:-ldc_${MODE}}"
OUT="$CXXI_DATA/ldc/$LABEL"
PY="$(cxxi_python)" || { echo "FATAL: no venv with jax; run 01_setup_env.sh first" >&2; exit 1; }

mkdir -p "$OUT"
echo "=== LDC  mode=$MODE  label=$LABEL"
echo "    host   $(hostname)   $(date -u '+%F %T UTC')"
echo "    job    ${SLURM_JOB_ID:-?}  partition ${SLURM_JOB_PARTITION:-?}  account ${SLURM_JOB_ACCOUNT:-?}"
echo "    python $PY"
echo "    outdir $OUT"
command -v xpu-smi >/dev/null 2>&1 && timeout 20 xpu-smi discovery 2>&1 | head -12 \
    || echo "    (xpu-smi not available)"
echo

# One GPU per job: pin Level Zero to the card Slurm allocated (the device nodes
# are not filtered per job, so JAX would otherwise enumerate every card).
cxxi_pin_gpu "$PY"
echo

"$PY" -u "$CXXI_DIR/ldc_run.py" --mode "$MODE" --outdir "$OUT" --label "$LABEL" "$@"
rc=$?
echo
echo "=== LDC exit $rc   $(date -u '+%F %T UTC')   outdir $OUT"
echo "    summary:  $OUT/summary.json"
echo "    figures:  $OUT/figures/"
exit $rc
