#!/bin/bash
# ---------------------------------------------------------------------------
# Step 3: prove that JAX runs on an Intel GPU, before any physics.
#
#     cd ~/Julia/PINN/clementina
#     sbatch 02_smoke.sbatch
#     tail -n +1 smoke_*.out
#
# This must run inside an allocation: the login node has no GPU, so on the login
# node JAX would simply report the CPU and the test would be meaningless (that is
# why 01_setup_env.sh only checks the import).
#
# Exit codes from smoke.py: 0 all checks passed; 1 a primitive failed (the check
# name says which); 2 JAX fell back to the CPU (the plugin did not load a GPU).
# ---------------------------------------------------------------------------
#SBATCH --job-name=cxxi_smoke
#SBATCH --account=pci_143
#SBATCH --partition=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
# Logs are absolute on purpose: Slurm resolves a relative --output against the
# directory you submit FROM, so a relative path scatters logs across the tree.
#SBATCH --output=/home/oreula/Julia/PINN/clementina/smoke_%j.out
#SBATCH --error=/home/oreula/Julia/PINN/clementina/smoke_%j.err

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
cxxi_pick_data

PY="$(cxxi_python)" || { echo "FATAL: no venv with jax; run 01_setup_env.sh first" >&2; exit 1; }

echo "host      $(hostname)"
echo "date      $(date -u '+%F %T UTC')"
echo "job       ${SLURM_JOB_ID:-?}  partition ${SLURM_JOB_PARTITION:-?}  account ${SLURM_JOB_ACCOUNT:-?}"
echo "python    $PY"
echo "repo      $CXXI_REPO"
echo "data      $CXXI_DATA"
echo "ZE        ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY:-<unset>}  ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<unset>}"
echo "          (CXXI_ZE_MODE=${CXXI_ZE_MODE:-<unset>} CXXI_UNSET_ZE_MASK=${CXXI_UNSET_ZE_MASK:-0})"
echo
echo "--- xpu-smi discovery"
xpu-smi discovery 2>&1 | head -20 || echo "xpu-smi not available"
echo

"$PY" "$HERE/smoke.py" --json "$CXXI_DATA/smoke_${SLURM_JOB_ID:-local}.json" "$@"
rc=$?
echo
echo "smoke.py exit $rc"
exit $rc
