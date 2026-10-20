#!/bin/bash
# ---------------------------------------------------------------------------
# DSGNAR benchmark on one Intel GPU (see bench.sh for the rationale).
#
#     cd ~/Julia/PINN
#     sbatch clementina/06_bench_gpu.sbatch
#     sbatch clementina/07_bench_cpu.sbatch     # the other half, same grid
#
# Both write rows to /data/contrib/pci_143/pinn/bench_cxxi.csv and keep each
# case's full log under .../Evolution_try/runs/cxxi_bench_<tag>_nc<points>/run.log.
# Override the grid with sbatch --export=ALL,NCOLLS="2201 16384",STEPS=20,SKETCH=300
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
exec bash "$CXXI_DIR/bench.sh" gpu
