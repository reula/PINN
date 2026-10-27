#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# stage2.py on the CCAD JupyterHub GPU (the Mendieta queue was too slow, 2026-10-27).
# Same command as jobs/mendieta/bh.sbatch PRESET=stage2.  The frozen run (default runs/long_con10)
# must already be in SBH/runs/ (runs/ is not in git: copied over serafin, same filesystem).
#
#   OUTDIR=stage2_con10_adm_g10 STAGE2_ARGS="..." bash jobs/hub/stage2.sh
#
# launches itself detached and returns; it first waits for the Stationary hub batches to end
# (WAIT_FOR, below).  Resumable: relaunching the same command continues from
# runs/$OUTDIR/stage2_params.pkl (--resume).  Log: logs/hub_$OUTDIR.log.
# Python: the Stationary venv (read-only use; tqdm, missing there, goes to SBH/.pydeps).
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
mkdir -p logs
OUTDIR=${OUTDIR:-stage2}
if [ "${1:-}" != "--run" ]; then
    setsid nohup bash "$HERE/jobs/hub/stage2.sh" --run >> "logs/hub_$OUTDIR.log" 2>&1 < /dev/null &
    echo "launched (pid $!); follow with:  tail -f $HERE/logs/hub_$OUTDIR.log"
    exit 0
fi
# Share nothing: wait until no process matches WAIT_FOR (default: the Stationary hub batches,
# Paper/hub_*.sh, which belong to another session).  WAIT_FOR= (empty) starts at once.
WAIT_FOR=${WAIT_FOR-Paper/hub_}
if [ -n "$WAIT_FOR" ] && pgrep -f "$WAIT_FOR" > /dev/null; then
    echo "--- waiting for '$WAIT_FOR' to finish  $(date -u '+%F %T UTC')"
    while pgrep -f "$WAIT_FOR" > /dev/null; do sleep 60; done
fi
echo "=== hub stage2 $OUTDIR $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)"
echo "STAGE2_RUN=${STAGE2_RUN:-runs/long_con10}  STAGE2_ARGS=${STAGE2_ARGS:-}"

PY="$HERE/../Stationary/.venv/bin/python"
export JAX_ENABLE_X64=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
# JAX caps itself at 75 % of the GPU (17.7 of the A30's 24 GB): --w_adm 1 ran out at 17.7 GiB in the
# second block (2026-10-27).  The GPU is ours once WAIT_FOR is clear, so allow 95 %.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
export PYTHONPATH="$HERE/.pydeps${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c "import tqdm" 2>/dev/null || "$PY" -m pip install -q --target "$HERE/.pydeps" tqdm
"$PY" -c "import jax; d = jax.devices(); print('jax', jax.__version__, d); raise SystemExit(d[0].platform != 'gpu')" \
    || { echo "ERROR: JAX does not see a GPU" >&2; exit 1; }
[ -f "${STAGE2_RUN:-runs/long_con10}/pinn_checkpoint.pkl" ] || { echo "ERROR: no ${STAGE2_RUN:-runs/long_con10}" >&2; exit 1; }

"$PY" -u stage2.py "${STAGE2_RUN:-runs/long_con10}" --out "runs/$OUTDIR" --resume ${STAGE2_ARGS:-} \
    -- --n_stars 1 --BL_masses 0.5 --BL_momenta 0 --BL_momenta_y 0 --tfinal 10 --N_window 40
echo "=== done $(date -u '+%F %T UTC')"
