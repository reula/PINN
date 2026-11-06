#!/bin/bash
# ---------------------------------------------------------------------------
# One sing_1d.py run on a Mendieta A30 (2026-11-06; the hub's jobs/hub/sing1d_batch.sh items, one job each).
#
#   cd ~/Julia/PINN/SBH && sbatch -J NAME --export=ALL,OUTDIR=NAME,SING_ARGS="..." jobs/mendieta/sing1d.sbatch
#
# Runs  sing_1d.py --out runs/OUTDIR --resume $BASE $SING_ARGS  with BASE as the hub batch's
# ("--hard_ic --no_template --N_window 32 --tfinal 8"); resubmitting the same command continues. Needs
# reference_1d/ref_single_m0.5_dr0.005_t20.npz and sing_bl_m0.5_dense.npz (git-ignored: copy them over), and
# for --stage2 the finished run's window_*.pkl under runs/. Mendieta's /home is not the hub's: results go back
# by rsync. Log: runs/slurm_NAME_JOBID.out.
# ---------------------------------------------------------------------------
#SBATCH -p multi
#SBATCH -A ifeg
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=28G
#SBATCH -t 2-00:00:00
#SBATCH -o runs/slurm_%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_sing1d
PY=../Stationary/.venv/bin/python
BASE=${BASE:---hard_ic --no_template --N_window 32 --tfinal 8}
echo "=== $(hostname) $(nvidia-smi -L | tr '\n' ' ') $OUTDIR  HEAD $(git rev-parse --short HEAD)  $(date -u '+%F %T UTC')"
echo "ARGS: $BASE $SING_ARGS"
"$PY" -c "import jax; d = jax.devices(); print('jax', jax.__version__, d); raise SystemExit(d[0].platform != 'gpu')" \
    || { echo "ERROR: JAX does not see a GPU" >&2; exit 1; }
"$PY" -u sing_1d.py --out "runs/$OUTDIR" --resume $BASE $SING_ARGS
echo "=== done $(date -u '+%F %T UTC')"
