#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# sing_pinn.py (template + correction ansatz) on the CCAD JupyterHub GPU (2026-10-28).
#
#   OUTDIR=sing_bl SING_ARGS="--template bl" bash jobs/hub/sing.sh
#   OUTDIR=sing_tr SING_ARGS="--template trumpet" bash jobs/hub/sing.sh
#
# launches itself detached and returns; waits for the Stationary hub batches first (WAIT_FOR, as in
# stage2.sh).  The template tables (reference_1d/*.npz, git-ignored) are generated on the CPU if they
# are missing (about 20 and 40 min).  Resumable: relaunching continues after the last finished window.
# Log: logs/hub_$OUTDIR.log.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
mkdir -p logs
OUTDIR=${OUTDIR:-sing}
if [ "${1:-}" != "--run" ]; then
    setsid nohup bash "$HERE/jobs/hub/sing.sh" --run >> "logs/hub_$OUTDIR.log" 2>&1 < /dev/null &
    echo "launched (pid $!); follow with:  tail -f $HERE/logs/hub_$OUTDIR.log"
    exit 0
fi
WAIT_FOR=${WAIT_FOR-Paper/hub_}
if [ -n "$WAIT_FOR" ] && pgrep -f "$WAIT_FOR" > /dev/null; then
    echo "--- waiting for '$WAIT_FOR' to finish  $(date -u '+%F %T UTC')"
    while pgrep -f "$WAIT_FOR" > /dev/null; do sleep 60; done
fi
echo "=== hub sing $OUTDIR $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)"
echo "SING_ARGS=${SING_ARGS:-}"

PY="$HERE/../Stationary/.venv/bin/python"
export JAX_ENABLE_X64=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
export PYTHONPATH="$HERE/.pydeps${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c "import tqdm" 2>/dev/null || "$PY" -m pip install -q --target "$HERE/.pydeps" tqdm
"$PY" -c "import jax; d = jax.devices(); print('jax', jax.__version__, d); raise SystemExit(d[0].platform != 'gpu')" \
    || { echo "ERROR: JAX does not see a GPU" >&2; exit 1; }

cd reference_1d
[ -f sing_bl_m0.5_dense.npz ] || JAX_PLATFORMS=cpu "$PY" -u sph_reference.py --dr 0.005 --tfinal 10 \
    --snap_dt 0.02 --r_save 1.0 --out sing_bl_m0.5_dense.npz
[ -f ref_single_m0.5_dr0.005_t20.npz ] || JAX_PLATFORMS=cpu "$PY" -u sph_reference.py --dr 0.005 --tfinal 20 \
    --snap_dt 0.25 --r_save 30 --out ref_single_m0.5_dr0.005_t20.npz
cd ..

"$PY" -u sing_pinn.py --out "runs/$OUTDIR" --resume ${SING_ARGS:-}
echo "=== done $(date -u '+%F %T UTC')"
