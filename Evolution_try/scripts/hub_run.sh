#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Launch a wave_pinn run, detached, with every path and variable set explicitly.
#
# Why this exists: copy-pasting a chain of shell commands into a JupyterHub is
# fragile.  A fresh terminal starts in $HOME, so a relative `logs/` does not
# resolve, and any `$PY` / `$COMMON` set in an earlier paste is simply gone --
# which turns the launch into `env ... -m wave_pinn.cli` and fails on the
# redirection.  This script sets everything itself, so there is nothing to carry
# over between commands.
#
# Usage (from anywhere, on the hub):
#   bash ~/serafin/Julia/PINN/Evolution_try/scripts/hub_run.sh
#   bash ~/serafin/Julia/PINN/Evolution_try/scripts/hub_run.sh T20win_ssbroyden ssbroyden
#   PY=/path/to/python bash .../hub_run.sh mylabel dsgnar
#
# Detached with setsid+nohup: a JupyterHub single-user server is NOT a job
# scheduler, so a process started from a JupyterLab terminal dies with that
# terminal, and a notebook cell dies with the server.
#
# Everything this project produces stays inside Evolution_try/: runs/ and logs/.
#
# Env overrides: PY (interpreter), T, WINDOWS, NCOLL, SNAP (time slices to report)
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # Evolution_try/
cd "$HERE"
source "$HERE/scripts/env.sh"

LABEL="${1:-T20win_dsgnar}"
OPT="${2:-dsgnar}"
T="${T:-20}"
WINDOWS="${WINDOWS:-10}"
NCOLL="${NCOLL:-2201}"
SNAP="${SNAP:-[0,2,4,6,8,10,12,14,16,18,20]}"

PY="$(find_python)" || {
    echo "FATAL: no interpreter with jax+numpy+scipy+optax. Set PY=/path/to/python," >&2
    echo "       or create this project's own venv:  see requirements.txt" >&2
    exit 1
}

mkdir -p logs .mplcache
echo "checkout : $HERE"
echo "python   : $PY"
"$PY" -c 'import jax; print("jax      :", jax.__version__, jax.devices())' || true

# A CPU-only device on a GPU hub means the CUDA jax is not in that interpreter.
"$PY" - <<'EOF' || true
import jax, sys
if not any(d.platform == "gpu" for d in jax.devices()):
    print("WARNING: no GPU visible -- JAX falls back to CPU silently. "
          "Expected on a Mac; on the hub check PY and the spawner's GPU request.")
EOF

if [ "${1:-}" = "--check" ]; then
    echo "checkout : $HERE"
    echo "python   : $PY"
    "$PY" - <<'EOF'
import sys
import jax, numpy, scipy, optax, matplotlib
print("jax      :", jax.__version__, jax.devices())
print("numpy    :", numpy.__version__, "| scipy", scipy.__version__, "| optax", optax.__version__)
if not any(d.platform == "gpu" for d in jax.devices()):
    print("WARNING  : no GPU visible. Expected on a Mac; on the hub check PY, and note")
    print("           that the JupyterHub spawner must request a GPU -- a CPU-only")
    print("           environment turns a 30-minute run into a 4-hour one silently.")
from scipy.optimize._trustregion_exact import IterativeSubproblem   # optim/trustregion.py
from scipy.linalg.lapack import dpotrf
from jax.scipy.fft import dct                                       # optim/dsgnar.py
import jax.ops
assert hasattr(jax.ops, "segment_sum")
assert hasattr(jax.lax, "map")
from wave_pinn.optim.ssbroyden import load_minimize
m, where = load_minimize()
print("crunch   :", "OK from " + str(where) if m is not None else "UNAVAILABLE")
print("private APIs (scipy trust-exact, jax.scipy.fft.dct, jax.ops.segment_sum): OK")
EOF
    echo
    echo "--- test suite (the gate; ~2-3 min) ---"
    "$PY" -m unittest discover -s "$HERE/tests" 2>&1 | tail -4
    exit 0
fi

case "$OPT" in
    dsgnar)      OPTFLAGS="--set optimizer=dsgnar --set dsgnar_steps=200 \
                --set dsgnar_sketch=733 --set dsgnar_delta0=1.0 --set dsgnar_delta_min=1e-15" ;;
    ssbroyden)   OPTFLAGS="--set optimizer=ssbroyden --set qn_steps=8000 --set qn_block=250 \
                --set qn_gtol=1e-16" ;;
    trustregion) OPTFLAGS="--set optimizer=trustregion --set tr_maxiter=200 --set tr_chunk=128" ;;
    *) echo "FATAL: unknown optimizer '$OPT' (dsgnar | ssbroyden | trustregion)" >&2; exit 1 ;;
esac

LOG="$HERE/logs/$LABEL.log"
OUT="$HERE/runs/$LABEL"
LAUNCH="nohup"
command -v setsid >/dev/null 2>&1 && LAUNCH="setsid nohup"

# shellcheck disable=SC2086
$LAUNCH env MPLBACKEND=Agg MPLCONFIGDIR="$HERE/.mplcache" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$PY" -m wave_pinn.cli \
        --label "$LABEL" --outdir "$OUT" \
        --set T="$T" --set windows="$WINDOWS" --set ansatz=t2sat \
        --set n_coll="$NCOLL" --set sampler=random --set resample_every=250 \
        --set "snapshot_times=$SNAP" \
        $OPTFLAGS \
    > "$LOG" 2>&1 < /dev/null &
echo $! > "$HERE/logs/$LABEL.pid"

sleep 20
echo "--- $LOG (first 20 s) ---"
tail -12 "$LOG" || true
echo
echo "monitor:  tail -f $LOG"
echo "          grep -E '^\\[win |^\\[chain' $LOG"
echo "alive?:   kill -0 \$(cat $HERE/logs/$LABEL.pid) && echo running || echo stopped"
