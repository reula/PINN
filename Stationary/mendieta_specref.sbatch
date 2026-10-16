#!/bin/bash
# ---------------------------------------------------------------------------
# The nonlinear spectral reference (stationary/specref.py, PERTURBATION.md section 5) for one or
# more finished Stationary runs, on one Mendieta A30.  See ../Mendieta.md.
#
#   cd ~/Julia/PINN/Stationary
#   sbatch mendieta_specref.sbatch runs/<run> [runs/<run> ...]
#
# Each run directory needs config.json and params.pkl (copy them from the hub: Mendieta's /home is
# its own).  Writes runs/<run>/specref_nonlinear.{npz,json} and the comparison in the job log.  A
# run that already has specref_nonlinear.json is skipped.  SPECREF_ARGS overrides the solve;
# SPECREF_TAG names a variant (see below).
# ---------------------------------------------------------------------------
#SBATCH -p multi
#SBATCH -A ifeg
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=28G
#SBATCH -t 06:00:00
#SBATCH -J specref
#SBATCH -o logs/slurm_%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/Julia/PINN/Stationary}"
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax" MPLCONFIGDIR="$HOME/.cache/mpl"
ARGS=${SPECREF_ARGS:-"--nonlinear --ls 0,2,4,6,8 --n-radial 36 --n-coll 2500 --n-bnd 400 --iters 6"}
# SPECREF_TAG=<tag>: a variant, written as specref_nonlinear_<tag>.* (and skipped only if THAT exists)
TAG=${SPECREF_TAG:-}
[ -n "$TAG" ] && ARGS="$ARGS --tag $TAG"
DONE="specref_nonlinear${TAG:+_$TAG}.json"

echo "=== specref job ${SLURM_JOB_ID:-?} on $(hostname)  $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)"
.venv/bin/python -c "import jax; assert jax.default_backend() == 'gpu', jax.devices(); print(jax.devices())" \
    || { echo "FATAL: JAX does not see a GPU" >&2; exit 2; }
rc=0
for run in "$@"; do
    if [ -f "$run/$DONE" ]; then echo "--- $run: $DONE exists"; continue; fi
    echo "--- $run  $(date -u '+%T UTC')"
    # shellcheck disable=SC2086
    .venv/bin/python -m stationary.specref "$run" $ARGS || rc=$?
done
echo "=== done $(date -u '+%F %T UTC')  rc=$rc"
exit $rc
