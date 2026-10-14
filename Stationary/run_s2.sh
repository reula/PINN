#!/usr/bin/env bash
# The next production run, detached: closing the terminal is safe.
#
#   bash run_s2.sh                 # launch
#   tail -f logs/run_S2.driver.log # what it launched with
#   tail -f logs/pq_r100_phihyb025_S2.log   # progress
#   kill $(cat runs/pq_r100_phihyb025_S2/run.pid)
#
# What it imposes (all defaults of the launcher, listed here so the run is self-documenting):
#   * the outer Robin condition on FOUR spheres, rho_out*(1.0, 0.9, 0.8, 0.7) -- now genuinely in
#     the objective, not only in the diagnostics (that was the S1 bug);
#   * the SCALAR condition on the Geroch-Hansen potential P = (1-lam^2)/(4 lam) with the dipolar
#     power dropped from its kernel, so P's dipole must vanish and lambda's physical l=0 rho^-2
#     term (2M^2) is untouched;
#   * the metric condition unchanged (base 2, order 3): its rho^-2 content is physical too, and
#     its dipole is a gauge mode -- a centre-of-mass condition, later;
#   * correct Ricci source, metric and lambda pins on, w_inner = w_outer = 100, reweighting off.
#
# Overrides: TAG=... STEPS=... LBFGS=... NCOLL=... NBND=... SHELLS=1.0,0.9,0.8 FORCE=1
#            EXTRA="--fourier 8"
#
# If the card is full: SHELLS=1.0,0.9 (three shells) halves the outer cost, NCOLL=16384
# halves the collocation cost.  S2 died with RESOURCE_EXHAUSTED on a 16 MiB allocation.
set -u
cd "$(dirname "$0")"
TAG="${TAG:-_S2}"
STEPS="${STEPS:-500}"
LBFGS="${LBFGS:-20000}"
NCOLL="${NCOLL:-32768}"
NBND="${NBND:-2048}"
EXTRA="${EXTRA:-}"
SHELLS="${SHELLS:-1.0,0.9,0.8,0.7}"      # the outer-condition ladder, as fractions of rho_out
RUN="runs/pq_r100_phihyb025${TAG}"
DRIVER="logs/run${TAG}.driver.log"
mkdir -p logs

if [ "${_DETACHED:-0}" != "1" ]; then
    _DETACHED=1 nohup setsid bash "$0" "$@" > "$DRIVER" 2>&1 &
    sleep 2
    echo "launched detached (safe to close this terminal)"
    echo "  driver : $DRIVER"
    echo "  run    : $RUN"
    echo "  watch  : tail -f logs/pq_r100_phihyb025${TAG}.log"
    echo "  report : cat $RUN/report.txt   (written when it ends)"
    echo "  stop   : kill \$(cat $RUN/run.pid)"
    exit 0
fi

echo "=== ${TAG} $(date -u '+%Y-%m-%d %H:%M:%S UTC')  HEAD $(git rev-parse --short HEAD 2>/dev/null)"
if pgrep -f "stationary.train" > /dev/null && [ "${FORCE:-0}" != "1" ]; then
    echo "!! a training run is already going; stop it or set FORCE=1 (two runs share the A30's" \
         "fp64 units and both crawl)"; exit 1
fi
if [ -f "$RUN/train.exit" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "!! $RUN already finished (exit $(cat "$RUN/train.exit")); use TAG=... for a new one" \
         "or FORCE=1 to overwrite"; exit 1
fi
RHO_OUT=100 TAG="$TAG" bash run_phi_hybrid.sh --only G --steps "$STEPS" --n-coll "$NCOLL" \
    --n-bnd "$NBND" --n-bnd-outer "$NBND" --radial log --lbfgs-steps "$LBFGS" \
    --outer-shell-fracs "$SHELLS" $EXTRA \
    || { echo "!! launch failed"; exit 1; }
echo "=== launched $(date -u '+%H:%M:%S UTC'); the wrapper writes train.exit and then the report"
