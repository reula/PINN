#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The paper's quadrupole runs, SSBroyden with the strong-Wolfe line search, on the JupyterHub A30.
# Starting points: the Adam stages qp_com_adam, qp_nocom_adam, qp_rot_adam (run on Mendieta, round P
# of mendieta_submit.sh; copied into runs/).  Stages of 10000 iterations, resampling every 500:
#   qp_com_w1, qp_com_w2          with the centre-of-mass condition
#   qp_nocom_w1, qp_nocom_w2      the same without it
#   qp_rot_w1 .. qp_rot_w3        the datum about (1,0,1)/sqrt 2, 3-D metric-only network
#   ab_{noladder,nopins,none}_*   qp_com from scratch without the Robin ladder / the pins / both
# After each chain: the loss on 5 fresh samples and the grade against the (rotated) fine reference.
#
#   bash Paper/hub_qp.sh           launches itself detached (setsid nohup) and returns
#   bash Paper/hub_qp.sh --run     the work itself (what the detached copy runs)
# Resumable: a finished stage (report.json) is skipped, an interrupted one resumes from its
# ckpt.pkl, so after a container restart the same command continues.  Log: logs/hub_qp.log.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # <repo>/Stationary
cd "$HERE"
mkdir -p logs
if [ "${1:-}" != "--run" ]; then
    setsid nohup bash "$HERE/Paper/hub_qp.sh" --run >> logs/hub_qp.log 2>&1 < /dev/null &
    echo "launched (pid $!); follow with:  tail -f $HERE/logs/hub_qp.log"
    exit 0
fi

PY="$HERE/.venv/bin/python"
export JAX_ENABLE_X64=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
source recipe_pq16k.sh
RHO4=(--scale-exps 3,4,3,4)
PP=(--pin-lam-potential)
COM=(--com-pin 1,1.5,3,10,30,100)
ROT=(--arch hybrid --width 20 --depth 6 --lam-bc-axis 1,0,1)
SSB=(--steps 0 --qn-method ssbroyden --qn-linesearch wolfe --lbfgs-steps 10000 --resample-every 500
     --plateau-min-iters -1)
REF=runs/ib16k_ssbroyden/specref_nonlinear_fine.npz
echo "=== hub_qp $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)  $("$PY" -c 'import jax; print(jax.devices())')"

stage() {   # stage <name> <init run> <flags...>
    local name=$1 init=$2; shift 2
    if [ -f "runs/$name/report.json" ]; then echo "--- $name: done, skipped"; return; fi
    local res=()
    [ -f "runs/$name/ckpt.pkl" ] && res=(--resume auto) && echo "--- $name: resuming from its checkpoint"
    echo "--- $name from $init  $(date -u '+%T UTC')"
    "$PY" -u -m stationary.train --outdir "runs/$name" "${S_RECIPE[@]}" "${SSB[@]}" \
        --init-from "runs/$init/params.pkl" "${res[@]}" "$@"
}
adam() {    # adam <name> <flags...>: the 5000-step Adam warm-up from scratch
    local name=$1; shift
    if [ -f "runs/$name/report.json" ]; then echo "--- $name: done, skipped"; return; fi
    echo "--- $name (Adam)  $(date -u '+%T UTC')"
    "$PY" -u -m stationary.train --outdir "runs/$name" "${S_RECIPE[@]}" --steps 5000 --plateau-min-iters -1 \
        --lbfgs-steps 0 "$@"
}
check() {   # check <run>
    [ -f "runs/$1/fresh_samples.json" ] || "$PY" -u scripts/fresh_sample_loss.py "runs/$1" --n 5
    "$PY" -u scripts/grade_rotated.py "runs/$1" --ref "$REF"
}

stage qp_com_w1   qp_com_adam   "${RHO4[@]}" "${PP[@]}" "${COM[@]}"
stage qp_com_w2   qp_com_w1     "${RHO4[@]}" "${PP[@]}" "${COM[@]}"
check qp_com_w2
stage qp_nocom_w1 qp_nocom_adam "${RHO4[@]}" "${PP[@]}"
stage qp_nocom_w2 qp_nocom_w1   "${RHO4[@]}" "${PP[@]}"
check qp_nocom_w2
stage qp_rot_w1   qp_rot_adam   "${ROT[@]}" "${RHO4[@]}" "${PP[@]}" "${COM[@]}"
stage qp_rot_w2   qp_rot_w1     "${ROT[@]}" "${RHO4[@]}" "${PP[@]}" "${COM[@]}"
stage qp_rot_w3   qp_rot_w2     "${ROT[@]}" "${RHO4[@]}" "${PP[@]}" "${COM[@]}"
check qp_rot_w3
# Ablations (2026-10-27): with the centre of mass fixed, are the Robin ladder and the averaged pins
# still needed?  qp_com from scratch without the ladder (Robin at rho_out only), without the pins,
# and without both; graded against the same reference.
BASE=("${RHO4[@]}" "${PP[@]}" "${COM[@]}")
for v in "ab_noladder:--outer-shell-fracs 1.0" "ab_nopins:--no-pin-h-robin --no-pin-lam" \
         "ab_none:--outer-shell-fracs 1.0 --no-pin-h-robin --no-pin-lam"; do
    tag=${v%%:*}; read -r -a X <<< "${v#*:}"
    adam  ${tag}_adam               "${BASE[@]}" "${X[@]}"
    stage ${tag}_w1 ${tag}_adam     "${BASE[@]}" "${X[@]}"
    stage ${tag}_w2 ${tag}_w1       "${BASE[@]}" "${X[@]}"
    check ${tag}_w2
done
echo "=== hub_qp finished $(date -u '+%F %T UTC')"
curl -s -H "Title: hub_qp finished" -d "Stationary paper runs (Wolfe) done on the hub: qp_com, qp_nocom, qp_rot" \
     https://ntfy.sh/pinn-reula-f0798d1ec301 > /dev/null || true
