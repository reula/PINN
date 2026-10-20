#!/bin/bash
# ---------------------------------------------------------------------------
# DSGNAR benchmark on one Intel GPU: FLAT (one 64 GB tile) vs COMPOSITE (one
# 128 GB device), at the production batch and a Stationary-sized batch.
# GPU-only by design: this allocation has few CPU hours.
#
#     cd ~/Julia/PINN
#     sbatch clementina/06_bench_gpu.sbatch
#
# ~6 GPU-minutes.  Rows go to /data/contrib/pci_143/pinn/bench_gpu.csv, each case
# keeps its log under .../Evolution_try/runs/cxxi_bench_<mode>_nc<points>/run.log.
# Override with sbatch --export=ALL,MODES="flat composite",NCOLLS="2201 16384",STEPS=15
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_bgpu
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
#SBATCH --output=/home/oreula/Julia/PINN/clementina/bench_gpu_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/bench_gpu_%j.err

set -uo pipefail
CXXI_DIR="${CXXI_DIR:-/home/oreula/Julia/PINN/clementina}"
[ -f "$CXXI_DIR/env.sh" ] || { echo "FATAL: no $CXXI_DIR/env.sh; set CXXI_DIR=" >&2; exit 1; }
for MODE in ${MODES:-flat composite}; do
    echo
    echo "################## tile mode: $MODE"
    bash "$CXXI_DIR/bench.sh" gpu "$MODE" || echo "mode $MODE exited $?"
done
