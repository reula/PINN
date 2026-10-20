#!/bin/bash
# ---------------------------------------------------------------------------
# DSGNAR benchmark on 64 CPU cores -- the other half of 06_bench_gpu.sbatch.
# Same grid, same seed, same float64 flags, so bench_cxxi.csv is comparable.
#
#     cd ~/Julia/PINN
#     sbatch clementina/07_bench_cpu.sbatch
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_bcpu
#SBATCH --account=pci_143
#SBATCH --partition=cpunode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --output=/home/oreula/Julia/PINN/clementina/bench_cpu_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/bench_cpu_%j.err

set -uo pipefail
CXXI_DIR="${CXXI_DIR:-/home/oreula/Julia/PINN/clementina}"
[ -f "$CXXI_DIR/env.sh" ] || { echo "FATAL: no $CXXI_DIR/env.sh; set CXXI_DIR=" >&2; exit 1; }
exec bash "$CXXI_DIR/bench.sh" cpu
