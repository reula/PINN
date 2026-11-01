#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Two attempts to get below the stall of the hard-end networks of the S2 scan, for one amplitude
# (ms_s2_025 or ms_s2_030; their SSBroyden stages stall at 2.5e-8 and ~1e-6 after w5):
#   dsg    DSGNAR on top of the stall: from ms_s2_<amp>_w5, 5 cycles of <= 60 iterations, sketch 2000
#          (0.88 d of the 2285 parameters), stage-2 target 0.075 throughout, a new sample every 100
#          iterations -- the settings that took m5_dsg_cyc from 9.4e-9 to 3.2e-9 (DSGNAR.md 9.2 and
#          round 5).  DSGNAR stalled near 1e-4 from a rotated start (rounds 14-16), so the odds are
#          modest; the 16384-point recipe, as in those runs.
#   b40    a bigger network: axisym_hybrid width 40 depth 6 from scratch, Adam 5000 + 3 SSBroyden/
#          Wolfe stages of 10000, 32768 points, redraws every 250 (the settings of hub_hard2.sh).
#          The solution is more complicated near the horizon, a wider network should help.
#   bash Paper/hub_hard3.sh --run <025|030>      Log: logs/hub_hard3_<amp>.log (when nohup'ed)
# About 3 h on one A30 per amplitude.  Resumable.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
mkdir -p logs
AMP=${2:?usage: hub_hard3.sh --run <025|030>}
case $AMP in 025) S2=-0.25;; 030) S2=-0.30;; *) echo "amplitude 025 or 030"; exit 2;; esac
PY="$HERE/.venv/bin/python"
export JAX_ENABLE_X64=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
source recipe_pq16k.sh
FINAL=(--scale-exps 3,4,3,4 --outer-shell-fracs 1.0 --no-pin-h-robin --no-pin-lam)
COM=(--com-pin 1,1.5,3,10,30,100)
DSG=(--steps 0 --lbfgs-steps 30000 --qn-method dsgnar --dsgnar-paper --dsgnar-reweight 0
     --dsgnar-chunk 16 --log-every 10 --resample-every 100 --dsgnar-steps 300 --dsgnar-sketch 2000
     --dsgnar-stage2-ratio 0.075 --dsgnar-cycles 5 --dsgnar-cycle-steps 60)
DENSE=(--n-coll 32768 --n-bnd 2048 --n-bnd-outer 2048 --resample-every 250)
SSB=(--steps 0 --qn-method ssbroyden --qn-linesearch wolfe --lbfgs-steps 10000 --plateau-min-iters -1)
tag=ms_s2_$AMP
echo "=== hub_hard3 $AMP $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)"

run() {   # run <name> <flags...>
    local name=$1; shift
    if [ -f "runs/$name/report.json" ]; then echo "--- $name: done, skipped"; return; fi
    local res=(); [ -f "runs/$name/ckpt.pkl" ] && res=(--resume auto) && echo "--- $name: resuming"
    echo "--- $name  $(date -u '+%T UTC')"
    "$PY" -u -m stationary.train --outdir "runs/$name" "${S_RECIPE[@]}" "${FINAL[@]}" "${COM[@]}" \
        --lam-bc-S2 "$S2" "${res[@]}" "$@"
}
# 1. DSGNAR on top of the stall
run ${tag}_dsg --init-from runs/${tag}_w5/params.pkl "${DSG[@]}"
"$PY" -u scripts/fresh_sample_loss.py runs/${tag}_dsg --n 5
# 2. the wider network
run ${tag}_b40_adam --width 40 --depth 6 --steps 5000 --plateau-min-iters -1 --lbfgs-steps 0 "${DENSE[@]}"
prev=${tag}_b40_adam
for i in 1 2 3; do
    run ${tag}_b40_w$i --width 40 --depth 6 --init-from runs/$prev/params.pkl "${SSB[@]}" "${DENSE[@]}"
    prev=${tag}_b40_w$i
done
"$PY" -u scripts/fresh_sample_loss.py runs/$prev --n 5
echo "=== hub_hard3 $AMP finished $(date -u '+%F %T UTC')"
