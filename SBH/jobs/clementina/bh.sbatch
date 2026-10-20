#!/bin/bash
# EEccz4_BH_puncture.py on Clementina XXI (Intel Data Center GPU Max 1550, JAX oneAPI plugin).
# Same presets and variables as jobs/mendieta/bh.sbatch; see SBH/README.md and clementina/README.md.
#
# Submit from SBH/ on the login node (after `source ~/Julia/PINN/clementina/env.sh && git pull`):
#   sbatch -t 00:40:00 --export=ALL,PRESET=bench jobs/clementina/bh.sbatch
#   sbatch --export=ALL,PRESET=single,OUTDIR=x,EXTRA_ARGS="--stop_after_window 8" jobs/clementina/bh.sbatch
# Outputs: $CXXI_DATA/SBH/runs/$OUTDIR (= /data/contrib/pci_143/pinn/SBH/runs/...); the log first
# goes to SBH/jobs/clementina/logs/ and is moved next to the outputs.
#
# Intel specifics (from clementina/env.sh): Level Zero is pinned to the allocated card
# (cxxi_pin_gpu), the job refuses a CPU fallback (cxxi_assert_accelerator), and no
# `module load oneapi` (the venv ships its own runtime). CXXI_ZE_MODE=COMPOSITE gives one
# 128 GB device per card instead of two 64 GB tiles.
#SBATCH -A pci_143
#SBATCH -p gpunode
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -J bh
#SBATCH -o /home/oreula/Julia/PINN/SBH/jobs/clementina/logs/%x_%j.out

set -eo pipefail
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
SBH_DIR=${SBH_DIR:-$HOME/Julia/PINN/SBH}
cd "$SBH_DIR"
source "$HOME/Julia/PINN/clementina/env.sh"
cxxi_pick_data
PYTHON=$(cxxi_python)
cxxi_pin_gpu "$PYTHON"
cxxi_assert_accelerator "$PYTHON" || { echo "ERROR: no Intel GPU visible to JAX" >&2; exit 1; }
export JAX_ENABLE_X64=1 MPLBACKEND=Agg
RUNS="$CXXI_DATA/SBH/runs"
mkdir -p "$RUNS"

NGPU=${NGPU:-1}
PRESET=${PRESET:-single}
PROD_ARGS=${PROD_ARGS---fast_hessian --loss_plot_every 10 --blowup_factor 100 --deriv_mode taylor}
echo "$(hostname) | ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-unset} ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY:-FLAT} | python $PYTHON"

if [ "$PRESET" = probe ]; then          # which ops fail on the oneAPI backend (tools/oneapi_probe.py)
  $PYTHON -u tools/oneapi_probe.py
  exit 0
fi
if [ "$PRESET" = bench ]; then
  for n in 8000 32000; do $PYTHON -u tools/bench_loss.py --n $n --reps 5; done
  $PYTHON -u tools/check_taylor.py --n 32000 --reps 5
  exit 0
fi
if [ "$PRESET" = eval ]; then
  case "${EVAL_PHYSICS:-single}" in
    single) PHYS="--n_stars 1 --BL_masses 0.5 --BL_momenta 0 --BL_momenta_y 0 --tfinal 10 --N_window 40" ;;
    bbh)    PHYS="" ;;
  esac
  $PYTHON -u tools/eval_heldout.py --runs $EVAL_RUNS ${EVAL_WINDOWS:+--windows $EVAL_WINDOWS} \
      --out "$RUNS/heldout_${SLURM_JOB_ID:-x}.csv" -- $PHYS
  exit 0
fi

case "$PRESET" in
  single) PRESET_ARGS="--n_stars 1 --BL_masses 0.5 --BL_momenta 0 --BL_momenta_y 0 --tfinal 10 --N_window 40 --snapshot_save_every 12000" ;;
  bbh)    PRESET_ARGS="--snapshot_save_every 12000" ;;
  smoke)  PRESET_ARGS="--Nint 3000 --NI 600 --Nb 1200 --HIDDEN 20 --N_LAYERS 4 --tfinal 0.5 --N_window 2 --EPOCHS_QN 200 --N_change 50 --snapshot_save_every 100 --loss_save_every 50" ;;
  *) echo "ERROR: unknown PRESET=$PRESET (single|bbh|smoke|bench|eval)" >&2; exit 1 ;;
esac

OUTDIR=$RUNS/${OUTDIR:-${PRESET}_${SLURM_JOB_ID:-x}}
case "$EXTRA_ARGS" in
  *--resume_checkpoint*) ;;
  *) if [ -e "$OUTDIR/pinn_checkpoint.pkl" ]; then
       echo "ERROR: $OUTDIR already holds a run; to continue it pass" >&2
       echo "       EXTRA_ARGS=\"--resume_checkpoint $OUTDIR/pinn_checkpoint.pkl\"" >&2
       exit 1
     fi ;;
esac
mkdir -p "$OUTDIR"
LOG="$SBH_DIR/jobs/clementina/logs/${SLURM_JOB_NAME:-x}_${SLURM_JOB_ID:-x}.out"
[ -f "$LOG" ] && cp "$LOG" "$OUTDIR/" 2>/dev/null || true

STAMP=$(date +%Y%m%d_%H%M%S)
cp EEccz4_BH_puncture.py "$OUTDIR/EEccz4_BH_puncture.py.$STAMP"
{ echo "job ${SLURM_JOB_ID:-?} $STAMP git $(git rev-parse --short HEAD 2>/dev/null) host $(hostname)"
  md5sum EEccz4_BH_puncture.py Crunch/Optimizers/minimize_backtracking.py dsgnar.py
  echo "PRESET=$PRESET NGPU=$NGPU PROD_ARGS=$PROD_ARGS EXTRA_ARGS=$EXTRA_ARGS"; } >> "$OUTDIR/code_md5.txt"

echo "start $(date)"
time $PYTHON -u EEccz4_BH_puncture.py --outdir "$OUTDIR" --n_gpus $NGPU $PROD_ARGS $PRESET_ARGS ${EXTRA_ARGS}
echo "end $(date)"
cp "$LOG" "$OUTDIR/" 2>/dev/null || true
