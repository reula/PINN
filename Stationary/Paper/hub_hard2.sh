#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The networks of the hard end of the S2 scan, continued (ms_s2_025_w2: loss 3.2e-8, fresh/training
# 1.5, still falling; ms_s2_030_w2: loss 1.6e-6, fresh/training 265 = overfitting).  From each w2:
# three more SSBroyden/Wolfe stages of 10000 with twice the points (n_coll 32768, n_bnd 2048 each)
# and a new sample every 250 iterations (the scan used 16384 / 1024 / 500), then the fresh-sample
# check.  Grading against the spectral references is done afterwards (hub_hardref.sh).
#   bash Paper/hub_hard2.sh --run          (sbatch/nohup it; resumable)  Log: logs/hub_hard2.log
# About 3 h on one A30 (about 17 min per stage at twice the points).
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
mkdir -p logs
PY="$HERE/.venv/bin/python"
export JAX_ENABLE_X64=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
source recipe_pq16k.sh
FINAL=(--scale-exps 3,4,3,4 --outer-shell-fracs 1.0 --no-pin-h-robin --no-pin-lam)
COM=(--com-pin 1,1.5,3,10,30,100)
DENSE=(--n-coll 32768 --n-bnd 2048 --n-bnd-outer 2048 --resample-every 250)
SSB=(--steps 0 --qn-method ssbroyden --qn-linesearch wolfe --lbfgs-steps 10000 --plateau-min-iters -1)
echo "=== hub_hard2 $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)"
for v in "025:-0.25" "030:-0.30"; do
    tag=ms_s2_${v%%:*}; S2=${v#*:}; prev=${tag}_w2
    for i in 3 4 5; do
        name=${tag}_w$i
        if [ -f "runs/$name/report.json" ]; then echo "--- $name: done, skipped"; prev=$name; continue; fi
        res=(); [ -f "runs/$name/ckpt.pkl" ] && res=(--resume auto) && echo "--- $name: resuming"
        echo "--- $name from $prev  $(date -u '+%T UTC')"
        "$PY" -u -m stationary.train --outdir "runs/$name" "${S_RECIPE[@]}" "${SSB[@]}" "${DENSE[@]}" \
            --init-from "runs/$prev/params.pkl" "${res[@]}" "${FINAL[@]}" "${COM[@]}" --lam-bc-S2 "$S2"
        prev=$name
    done
    "$PY" -u scripts/fresh_sample_loss.py "runs/$prev" --n 5
done
echo "=== hub_hard2 finished $(date -u '+%F %T UTC')"
