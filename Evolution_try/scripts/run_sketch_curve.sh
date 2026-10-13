#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The sketch-size curve: T = 2, everything fixed except s, to convergence.
#
#   bash scripts/run_sketch_curve.sh                 # s = 733 1100 1500
#   SKETCHES="1100 1500" bash scripts/run_sketch_curve.sh
#
# Why: the sketch auditor measures the ANGLE between the lifted DSGNAR step and the
# exact Gauss-Newton step, and it improves monotonically past the paper's range --
# 0.42 at s = d/3, 0.62 at d/2, 0.75 at 0.68 d.  Angle is not accuracy, and this
# turns one into the other.  Measured on CPU so far:
#
#     s =  733   loss 2.243e-14   rel L2 6.756e-06
#     s = 1100   loss 2.918e-20   rel L2 4.116e-09     1640x, for 2.35x the wall
#
# The s = 1500 point is the one that decides where to stop, and on a GPU all three
# are cheap enough to do in one go.
#
# Sequential on purpose: two of these at once on this hub is a measured OOM, not a
# slowdown.  Resumable: a point whose runs/<label>/report.md exists is skipped.
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
source "$HERE/scripts/env.sh"

SKETCHES="${SKETCHES:-733 1100 1500}"
NCOLL="${NCOLL:-2201}"
STEPS="${STEPS:-400}"

PY="$(find_python)" || { echo "FATAL: no interpreter with jax; set PY=" >&2; exit 1; }
"$PY" -c 'import jax; print("python   :", jax.__version__, jax.devices())' || true

"$PY" - <<'EOF' || { echo "FATAL: float64 is not enabled; refusing." >&2; exit 1; }
from wave_pinn.train import configure_jax
from wave_pinn.config import Config
import jax.numpy as jnp
cfg = Config(); configure_jax(cfg)
assert jnp.zeros(1).dtype == jnp.float64, "x64 did NOT take effect"
print("F64      : ON")
EOF

mkdir -p logs
printf '\n=== sketch curve %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" | tee -a logs/sketch_curve.log
printf '%-8s %-8s %14s %14s %10s\n' "s" "s/d" "final loss" "rel L2" "wall" | tee -a logs/sketch_curve.log

for S in $SKETCHES; do
    lab="t2_s${S}"
    log="$HERE/logs/$lab.log"
    if [ -f "$HERE/runs/$lab/report.md" ]; then
        echo "$lab -- already done, skipping" | tee -a logs/sketch_curve.log
        continue
    fi
    t0=$(date +%s)
    echo "--- s = $S  ($(( S * 100 / NCOLL ))% of d)  start $(date '+%H:%M:%S')"
    "$PY" -m wave_pinn.cli --label "$lab" --outdir "$HERE/runs/$lab" \
        --set T=2 --set windows=1 --set optimizer=dsgnar \
        --set dsgnar_steps="$STEPS" --set dsgnar_sketch="$S" \
        --set dsgnar_hashes=4 --set dsgnar_calibrate=false --set dsgnar_backtrack_steps=-1 \
        --set n_coll="$NCOLL" --set sampler=random \
        --set resample_every=0 --set resample_rounds=1 --set log_every=25 \
        >"$log" 2>&1
    rc=$?
    took=$(( $(date +%s) - t0 ))
    if [ $rc -ne 0 ]; then
        echo "s = $S FAILED (exit $rc); see $log" | tee -a logs/sketch_curve.log
        tail -5 "$log"
        continue
    fi
    loss=$(grep -oE 'loss [0-9.e+-]+ -> [0-9.e+-]+' "$log" | head -1 | awk '{print $4}')
    iters=$(grep -oE 'in [0-9]+ iterations' "$log" | head -1 | awk '{print $1}')
    rel=$(grep -oE 'space-time relative L2 = [0-9.e+-]+' "$log" | head -1 | awk '{print $NF}')
    printf '%-8s %-8s %14s %14s %8s s  (%s iterations)\n' \
        "$S" "$(( S * 100 / NCOLL ))%" "$loss" "$rel" "$took" "$iters" \
        | tee -a logs/sketch_curve.log
done

echo | tee -a logs/sketch_curve.log
echo "Then pick the size and run the focused batch:" | tee -a logs/sketch_curve.log
echo "  DSGNAR_SKETCH=1100 FOCUSED=1 bash scripts/batch_runs.sh" | tee -a logs/sketch_curve.log
