#!/bin/bash
# Probe the float64 DCT sizes that the Intel backend can plan (the paper-size
# exact-residual step dies creating a cuFFT batched plan).  See ldc_fft_probe.py.
#SBATCH --job-name=cxxi_fft
#SBATCH --account=pci_143
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/home/oreula/Julia/PINN/clementina/fft_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/fft_%j.err
set -uo pipefail
CXXI_DIR="${CXXI_DIR:-/home/oreula/Julia/PINN/clementina}"
source "$CXXI_DIR/env.sh"
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
cxxi_pick_data
PY="$(cxxi_python)" || { echo "FATAL: no venv" >&2; exit 1; }
echo "host $(hostname) | job ${SLURM_JOB_ID:-?}"
cxxi_pin_gpu "$PY"
"$PY" -u "$CXXI_DIR/ldc_fft_probe.py"
