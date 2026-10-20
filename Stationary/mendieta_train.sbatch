#!/bin/bash
# ---------------------------------------------------------------------------
# One Stationary training run on one Mendieta A30, on the shared problem (recipe_pq16k.sh).
# See ../Mendieta.md.
#
#   cd ~/Julia/PINN/Stationary
#   sbatch -J <name> mendieta_train.sbatch <name> <train flags...>
#
# The flags after <name> are appended to the recipe (later flags win).  A run whose
# runs/<name>/report.json exists is skipped.  Submitted by mendieta_submit.sh.
# ---------------------------------------------------------------------------
#SBATCH -p multi
#SBATCH -A ifeg
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=28G
#SBATCH -t 14:00:00
#SBATCH -o logs/slurm_%x_%j.out
set -uo pipefail
NAME="$1"; shift
cd "${SLURM_SUBMIT_DIR:-$HOME/Julia/PINN/Stationary}"
source recipe_pq16k.sh
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
# MEMFRAC=0.95 sbatch ... : the share of the GPU's memory JAX may use (its default is 0.75, i.e.
# 17.7 GiB of an A30's 24; rot_coll256k's first allocation needed 17.76 GiB)
[ -n "${MEMFRAC:-}" ] && export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEMFRAC"
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax" MPLCONFIGDIR="$HOME/.cache/mpl"
mkdir -p logs runs "$MPLCONFIGDIR"

echo "=== $NAME  job ${SLURM_JOB_ID:-?} on $(hostname)  $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
.venv/bin/python -c "import jax; assert jax.default_backend() == 'gpu', jax.devices(); print(jax.devices())" \
    || { echo "FATAL: JAX does not see a GPU" >&2; exit 2; }
if [ -f "runs/$NAME/report.json" ]; then echo "runs/$NAME/report.json exists: done already"; exit 0; fi

.venv/bin/python -m stationary.train --outdir "runs/$NAME" "${S_RECIPE[@]}" "$@"
rc=$?
echo "=== $NAME exit $rc  $(date -u '+%F %T UTC')"
exit $rc
