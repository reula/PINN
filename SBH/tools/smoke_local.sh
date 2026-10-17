#!/bin/bash
# CPU smoke test on the laptop (a few minutes on an M2): 2 windows, tiny network.
#   bash tools/smoke_local.sh [outdir] [extra flags...]
# Uses ../Stationary/.venv (JAX 0.11.1; needs tqdm, installed 2026-10-17).
set -eo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-runs/smoke_local}; shift || true
mkdir -p "$OUT"
export JAX_ENABLE_X64=1 MPLBACKEND=Agg
time ${PYTHON:-../Stationary/.venv/bin/python} -u EEccz4_BH_puncture.py --outdir "$OUT" \
    --Nint 3000 --NI 600 --Nb 1200 --HIDDEN 20 --N_LAYERS 4 --tfinal 0.5 --N_window 2 \
    --EPOCHS_QN 200 --N_change 50 --snapshot_save_every 100 --loss_save_every 50 "$@"
