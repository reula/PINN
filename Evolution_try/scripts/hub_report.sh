#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Read-only diagnostics: where am I, which interpreter will the scripts pick,
# what device does it see, and what is running.  Changes nothing, starts nothing.
#
#   bash <checkout>/Evolution_try/scripts/hub_report.sh
#
# Written for pasting back to someone who cannot reach the machine: it prints the
# checkout and its git state, the interpreter chosen by scripts/env.sh and its jax
# version and devices, the GPU, any running wave_pinn jobs with their log tails,
# and the run directories.
# ---------------------------------------------------------------------------
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # Evolution_try/
cd "$HERE"

echo "===== where ====="
hostname
pwd -P
echo "HOME=$HOME"

echo "===== checkout ====="
git -C "$HERE" log --oneline -1 2>&1 | head -1
git -C "$HERE" rev-parse --abbrev-ref HEAD 2>/dev/null
echo "-- ahead/behind origin:"
git -C "$HERE" status -sb 2>/dev/null | head -1
echo "-- dirty files:"
git -C "$HERE" status --short 2>/dev/null | head -15

echo "===== interpreter (what the scripts will pick) ====="
# shellcheck source=./env.sh
source "$HERE/scripts/env.sh"
PY="$(find_python)" && echo "PY=$PY" || echo "PY: NONE FOUND -- set PY=/path/to/python"
if [ -n "${PY:-}" ]; then
    "$PY" -c 'import sys, jax, numpy, scipy, optax, matplotlib
print("python   :", sys.version.split()[0])
print("jax      :", jax.__version__, jax.devices())
print("numpy    :", numpy.__version__, "| scipy", scipy.__version__, "| optax", optax.__version__)
gpu = [d for d in jax.devices() if d.platform == "gpu"]
print("GPU      :", "yes" if gpu else "NO -- JAX will use CPU silently")' 2>&1 | tail -6
fi

echo "===== gpu ====="
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
           --format=csv,noheader 2>/dev/null || echo "no nvidia-smi (expected on a Mac)"

echo "===== private APIs ====="
if [ -n "${PY:-}" ]; then
    "$PY" - <<'EOF' 2>&1 | tail -4
try:
    from scipy.optimize._trustregion_exact import IterativeSubproblem   # optim/trustregion.py
    from scipy.linalg.lapack import dpotrf
    from jax.scipy.fft import dct                                       # optim/dsgnar.py
    import jax.ops
    assert hasattr(jax.ops, "segment_sum") and hasattr(jax.lax, "map")
    print("scipy trust-exact, jax.scipy.fft.dct, jax.ops.segment_sum: OK")
except Exception as exc:
    print("MISSING API:", type(exc).__name__, exc)
from wave_pinn.optim.ssbroyden import load_minimize
m, where = load_minimize()
print("capability SSBroyden:", "OK from " + str(where) if m is not None else "UNAVAILABLE")
EOF
fi

echo "===== running jobs ====="
found=0
for f in "$HERE"/logs/*.pid; do
    [ -e "$f" ] || continue
    found=1
    printf '%-34s ' "$(basename "$f")"
    kill -0 "$(cat "$f")" 2>/dev/null && echo "RUNNING (pid $(cat "$f"))" || echo "stopped"
done
[ "$found" = 0 ] && echo "(no .pid files)"
pgrep -af "wave_pinn" 2>/dev/null | grep -v pgrep | head -5 || true

echo "===== logs (newest first) ====="
ls -lt "$HERE/logs" 2>/dev/null | head -8
for f in "$HERE"/logs/*.log; do
    [ -e "$f" ] || continue
    echo "--- $(basename "$f")"
    tail -4 "$f"
done

echo "===== runs (newest first) ====="
ls -lt "$HERE/runs" 2>/dev/null | head -14
