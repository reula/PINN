#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Several sing_1d.py runs on the hub GPU, NPAR at a time (2026-11-03).
#
#   NPAR=3 bash jobs/hub/sing1d_batch.sh "NAME:ARGS" "NAME:ARGS" ...
#
# Each item runs as  SCRIPT=sing_1d.py OUTDIR=NAME SING_ARGS="$BASE ARGS" jobs/hub/sing.sh --run  with its own
# log, logs/hub_NAME.log; the next item starts when one of the NPAR running ones ends. BASE defaults to
# "--hard_ic --no_template --N_window 32 --tfinal 8". Memory: a 1D run is small in points but the SSBroyden
# inverse Hessian is a dense n^2 matrix (8730 parameters: 0.58 GiB per copy, several copies), so each run gets
# XLA_PYTHON_CLIENT_MEM_FRACTION = 0.8 / NPAR (seven at 0.12 each ran out of memory, 2026-11-03; at 0.9 / NPAR one
# of five failed loading compiled code, which needs memory outside the pools, 2026-11-04). Never two batches at once.
# Resumable item by item (sing.sh passes --resume): relaunching the same list continues where it stopped.
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
mkdir -p logs
NPAR=${NPAR:-3}
BASE=${BASE:---hard_ic --no_template --N_window 32 --tfinal 8}
FRAC=$(awk -v n="$NPAR" 'BEGIN { printf "%.3f", 0.8 / n }')
echo "=== sing1d batch $(date -u '+%F %T UTC'), NPAR=$NPAR, MEM_FRACTION=$FRAC each, $# runs"
for v in "$@"; do
    name="${v%%:*}"
    args="${v#*:}"
    while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do sleep 30; done
    echo "--- start $name ($args)  $(date -u '+%T')"
    XLA_PYTHON_CLIENT_MEM_FRACTION=$FRAC SCRIPT=sing_1d.py OUTDIR="$name" SING_ARGS="$BASE $args" \
        bash jobs/hub/sing.sh --run >> "logs/hub_$name.log" 2>&1 &
    sleep 30
done
wait
echo "=== sing1d batch done $(date -u '+%F %T UTC')"
