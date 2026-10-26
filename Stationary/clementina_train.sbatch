#!/bin/bash
# ---------------------------------------------------------------------------
# A Stationary run on Clementina XXI (one Intel Data Center GPU Max 1550 tile, JAX oneAPI plugin),
# on the shared quadrupolar problem (recipe_pq16k.sh), as mendieta_train.sbatch does on Mendieta.
#   sbatch -J <name> clementina_train.sbatch <name> [flags appended to the recipe...]
#   MODE=script sbatch -J <name> clementina_train.sbatch <name> scripts/<x>.py [args...]
# Environment as in clementina/README.md (env.sh: the venv, the proxy, the GPU pinning).
# ---------------------------------------------------------------------------
#SBATCH -A pci_143
#SBATCH -p gpunode
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -J st
#SBATCH -o /home/oreula/Julia/PINN/Stationary/logs/cxxi_%x_%j.out
set -eo pipefail
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
source "$HOME/Julia/PINN/clementina/env.sh"
cd "$HOME/Julia/PINN/Stationary"
PYTHON=$(cxxi_python)
cxxi_pin_gpu "$PYTHON"
cxxi_assert_accelerator "$PYTHON" || { echo "ERROR: no Intel GPU visible to JAX" >&2; exit 1; }
export JAX_ENABLE_X64=1 MPLBACKEND=Agg
NAME="$1"; shift
echo "=== $NAME  job ${SLURM_JOB_ID:-?} on $(hostname)  $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)  ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-unset}"
if [ "${MODE:-train}" = script ]; then
    "$PYTHON" -u "$@"
else
    source recipe_pq16k.sh
    "$PYTHON" -u -m stationary.train --outdir "runs/$NAME" "${S_RECIPE[@]}" "$@"
fi
