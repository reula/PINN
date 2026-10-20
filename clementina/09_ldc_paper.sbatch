#!/bin/bash
# ---------------------------------------------------------------------------
# The LDC notebook run through the PAPER'S OWN trainer: pinn.train64 and a
# pinn.RunConfig (this is ldc_pinn.ipynb, the reference implementation's version).
#
#     cd ~/Julia/PINN
#     sbatch clementina/09_ldc_paper.sbatch                            # MODE=paper (2^15 points)
#     sbatch --export=ALL,MODE=smoke clementina/09_ldc_paper.sbatch    # minutes, SMOKE config
#
# On the hub the SMOKE configuration reached final rel error 6.384e-02 in 11.5 s
# of compute (ldc_pinn.ipynb), which is the cheapest good result we have on this
# problem -- hence the smoke run first.
#
# Outputs: /data/contrib/pci_143/pinn/ldc/ldc_paper_<mode>/
#     summary.json (trainer rel-L2, per-channel rel L2, wall time, config),
#     results.npz, figures/fields_{u,v,p}.png
# Job log: clementina/ldcpaper_<jobid>.out (absolute path).
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_ldcp
#SBATCH --account=pci_143
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# Logs are absolute on purpose: Slurm resolves a relative --output against the
# directory you submit FROM, so a relative path scatters logs across the tree.
#SBATCH --output=/home/oreula/Julia/PINN/clementina/ldcpaper_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/ldcpaper_%j.err

set -uo pipefail

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

# Slurm copies the login shell's Lmod environment into the job; the pip oneAPI
# runtime needs none of it.
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
cxxi_pick_data

MODE="${MODE:-paper}"
LABEL="${LABEL:-ldc_paper_${MODE}}"
OUT="$CXXI_DATA/ldc/$LABEL"
PY="$(cxxi_python)" || { echo "FATAL: no venv with jax; run 01_setup_env.sh first" >&2; exit 1; }

mkdir -p "$OUT"
echo "=== LDC via paper trainer  mode=$MODE  label=$LABEL"
echo "    host   $(hostname)   $(date -u '+%F %T UTC')"
echo "    job    ${SLURM_JOB_ID:-?}  partition ${SLURM_JOB_PARTITION:-?}  account ${SLURM_JOB_ACCOUNT:-?}"
echo "    python $PY"
echo "    outdir $OUT"
echo

cxxi_pin_gpu
echo

"$PY" -u "$CXXI_DIR/ldc_paper_run.py" --mode "$MODE" --outdir "$OUT" --label "$LABEL" "$@"
rc=$?
echo
echo "=== LDC-paper exit $rc   $(date -u '+%F %T UTC')   outdir $OUT"
echo "    summary: $OUT/summary.json"
exit $rc
