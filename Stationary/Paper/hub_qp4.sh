#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The off-axis quadrupole run with a wider 3-D network: qp_rot3 (hub_qp3.sh, width 20, 2347
# parameters, 16384 points) plateaued at 1.17e-6, rms dh 1.2e-3, the same as qp_rot2 with ladder
# and pins, so the suspect is capacity.  Same formulation as qp_rot3 (no ladder, no pins,
# centre of mass 48 x 32, rho^4 weights), width 40, and N collocation points (default 32768,
# boundary N/16).  If it runs out of GPU memory:  N=24576 bash Paper/hub_qp4.sh
#   qp_rot4_n${N}_adam (Adam 5000), qp_rot4_n${N}_w1 .. w4 (SSBroyden, strong Wolfe, 10000 each), then the check.
#
#   bash Paper/hub_qp4.sh         launches itself detached and returns (N=... to change the points)
# Resumable like hub_qp.sh.  Log: logs/hub_qp4.log.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
mkdir -p logs
if [ "${1:-}" != "--run" ]; then
    N=${N:-32768} setsid nohup bash "$HERE/Paper/hub_qp4.sh" --run >> logs/hub_qp4.log 2>&1 < /dev/null &
    echo "launched (pid $!); follow with:  tail -f $HERE/logs/hub_qp4.log"
    exit 0
fi
N=${N:-32768}
echo "=== hub_qp4 $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)  N $N"

PY="$HERE/.venv/bin/python"
export JAX_ENABLE_X64=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
source recipe_pq16k.sh
F=(--n-coll $N --n-bnd $((N / 16)) --n-bnd-outer $((N / 16))
   --arch hybrid --width 40 --depth 6 --lam-bc-axis 1,0,1 --scale-exps 3,4,3,4 --pin-lam-potential
   --com-pin 1,1.5,3,10,30,100 --outer-shell-fracs 1.0 --no-pin-h-robin --no-pin-lam)
SSB=(--steps 0 --qn-method ssbroyden --qn-linesearch wolfe --lbfgs-steps 10000 --resample-every 500
     --plateau-min-iters -1)
REF=runs/ib16k_ssbroyden/specref_nonlinear_fine.npz

if [ ! -f runs/qp_rot4_n${N}_adam/report.json ]; then
    echo "--- qp_rot4_n${N}_adam (Adam)  $(date -u '+%T UTC')"
    "$PY" -u -m stationary.train --outdir runs/qp_rot4_n${N}_adam "${S_RECIPE[@]}" --steps 5000 \
        --plateau-min-iters -1 --lbfgs-steps 0 "${F[@]}"
fi
prev=qp_rot4_n${N}_adam
for i in 1 2 3 4; do
    name=qp_rot4_n${N}_w$i
    if [ ! -f "runs/$name/report.json" ]; then
        res=(); [ -f "runs/$name/ckpt.pkl" ] && res=(--resume auto)
        echo "--- $name from $prev  $(date -u '+%T UTC')"
        "$PY" -u -m stationary.train --outdir "runs/$name" "${S_RECIPE[@]}" "${SSB[@]}" \
            --init-from "runs/$prev/params.pkl" "${res[@]}" "${F[@]}"
    fi
    prev=$name
done
[ -f runs/qp_rot4_n${N}_w4/fresh_samples.json ] || "$PY" -u scripts/fresh_sample_loss.py runs/qp_rot4_n${N}_w4 --n 5
"$PY" -u scripts/grade_rotated.py runs/qp_rot4_n${N}_w4 --ref "$REF"
echo "=== hub_qp4 finished $(date -u '+%F %T UTC')"
curl -s -H "Title: hub_qp4 finished" -d "qp_rot4_n$N (no ladder, no pins, COM) done on the hub" \
     https://ntfy.sh/pinn-reula-f0798d1ec301 > /dev/null || true
