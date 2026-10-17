#!/bin/bash
# EEccz4_BH_puncture.py on Mendieta (CCAD-UNC, nodes with 2 x A30). See SBH/README.md.
#
# Submit from SBH/ (the script and Crunch/ are there); outputs go to runs/$OUTDIR:
#   sbatch jobs/mendieta/bh.sbatch                                         # PRESET=single, 1 GPU
#   sbatch --export=ALL,PRESET=bbh,OUTDIR=bbh_a jobs/mendieta/bh.sbatch
#   sbatch --gres=gpu:2 -c 16 --mem=56G --export=ALL,NGPU=2,OUTDIR=x jobs/mendieta/bh.sbatch
#   sbatch -p short -t 01:00:00 --export=ALL,PRESET=smoke,OUTDIR=smoke jobs/mendieta/bh.sbatch
#   sbatch -p short -t 01:00:00 --export=ALL,PRESET=bench jobs/mendieta/bh.sbatch   # loss timings
# Resume (same OUTDIR; `multi` allows 2 days, the BBH run needs longer):
#   sbatch --export=ALL,OUTDIR=x,EXTRA_ARGS="--resume_checkpoint runs/x/pinn_checkpoint.pkl" jobs/mendieta/bh.sbatch
#
# PRESET  single : one puncture, m = 0.5, t <= 10 (40 windows) -- the validation baseline
#         bbh    : the script defaults (equal-mass bound orbit, t <= 50, 200 windows)
#         smoke  : a few minutes, tiny network
#         bench  : tools/bench_loss.py + tools/check_taylor.py, no training
# PROD_ARGS default: --fast_hessian --loss_plot_every 10 --blowup_factor 100 (the MN5
# production flags). Add --deriv_mode taylor through EXTRA_ARGS (or PROD_ARGS).
#SBATCH -p multi
#SBATCH -A ifeg
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=28G
#SBATCH -t 2-00:00:00
#SBATCH -J bh
#SBATCH -o runs/slurm_%x_%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f EEccz4_BH_puncture.py ] || { echo "ERROR: submit from SBH/ (no EEccz4_BH_puncture.py here)" >&2; exit 1; }

PYTHON=${PYTHON:-../Stationary/.venv/bin/python}
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_bh
mkdir -p "$JAX_COMPILATION_CACHE_DIR" runs

NGPU=${NGPU:-1}
PRESET=${PRESET:-single}
PROD_ARGS=${PROD_ARGS---fast_hessian --loss_plot_every 10 --blowup_factor 100}

echo "$(hostname) | $(nvidia-smi -L | tr '\n' ' ')"
# Refuse a silent CPU fallback (it would take days).
$PYTHON -c "import sys, jax, flax, optax, tqdm, matplotlib; d = jax.devices(); print('python', sys.version.split()[0], '| jax', jax.__version__, '| flax', flax.__version__, '| optax', optax.__version__, '|', d); sys.exit(0 if d[0].platform == 'gpu' and len(d) >= $NGPU else 1)" \
  || { echo "ERROR: JAX does not see $NGPU GPU(s)" >&2; exit 1; }

if [ "$PRESET" = bench ]; then
  for n in 8000 32000; do $PYTHON -u tools/bench_loss.py --n $n --reps 5; done
  $PYTHON -u tools/check_taylor.py --n 32000 --reps 5
  exit 0
fi

case "$PRESET" in
  single) PRESET_ARGS="--n_stars 1 --BL_masses 0.5 --BL_momenta 0 --BL_momenta_y 0 --tfinal 10 --N_window 40 --snapshot_save_every 12000" ;;
  bbh)    PRESET_ARGS="--snapshot_save_every 12000" ;;
  smoke)  PRESET_ARGS="--Nint 3000 --NI 600 --Nb 1200 --HIDDEN 20 --N_LAYERS 4 --tfinal 0.5 --N_window 2 --EPOCHS_QN 200 --N_change 50 --snapshot_save_every 100 --loss_save_every 50" ;;
  *) echo "ERROR: unknown PRESET=$PRESET (single|bbh|smoke|bench)" >&2; exit 1 ;;
esac

OUTDIR=runs/${OUTDIR:-${PRESET}_${SLURM_JOB_ID:-x}}
case "$EXTRA_ARGS" in
  *--resume_checkpoint*) ;;
  *) if [ -e "$OUTDIR/pinn_checkpoint.pkl" ]; then
       echo "ERROR: $OUTDIR already holds a run; to continue it pass" >&2
       echo "       EXTRA_ARGS=\"--resume_checkpoint $OUTDIR/pinn_checkpoint.pkl\"" >&2
       exit 1
     fi ;;
esac
mkdir -p "$OUTDIR"
LOG="runs/slurm_${SLURM_JOB_NAME:-x}_${SLURM_JOB_ID:-x}.out"
[ -f "$LOG" ] && mv -n "$LOG" "$OUTDIR/"

# Record exactly what ran next to the outputs.
STAMP=$(date +%Y%m%d_%H%M%S)
cp EEccz4_BH_puncture.py "$OUTDIR/EEccz4_BH_puncture.py.$STAMP"
{ echo "job ${SLURM_JOB_ID:-?} $STAMP git $(git rev-parse --short HEAD 2>/dev/null)"
  md5sum EEccz4_BH_puncture.py Crunch/Optimizers/minimize_backtracking.py
  echo "PRESET=$PRESET NGPU=$NGPU PROD_ARGS=$PROD_ARGS EXTRA_ARGS=$EXTRA_ARGS"; } >> "$OUTDIR/code_md5.txt"

echo "start $(date)"
time $PYTHON -u EEccz4_BH_puncture.py --outdir "$OUTDIR" --n_gpus $NGPU $PROD_ARGS $PRESET_ARGS ${EXTRA_ARGS}
echo "end $(date)"
