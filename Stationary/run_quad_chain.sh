#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The two quarter-quadrupole diagnostics, back to back in one chain:
#
#   1. runs/production_quad_quarter_pin     the far-field VALUE pins (--pin-far)
#   2. runs/production_quad_quarter_lamrad  the rho-scaled RADIAL DERIVATIVE of the
#                                           lambda equation (--w-lam-eq-radial 1)
#
# Same problem in both (S2 = -1/12, order-3 Robin conditions, 500 Adam + 8000 SSBroyden,
# cold start, everything else as in runs/production_quad_quarter), so the two mechanisms are
# tested separately against the same failure: that run converged onto the "lambda = const"
# branch (lambda(rho_out) = 0.31 against 0.9885193) with its outer Robin residual at 7e-06.
# See HUB.md 6d (the pins) and 6e (the radial term).
#
# Both runs are launched through run_hub.sh, which detaches the training and chains the
# post-processing (diagnostics, profile, report) after it; this script only waits for one to
# finish before starting the next.  Expect ~2 h of training each plus ~15 min of
# post-processing, so ~4.5 h for the pair.
#
# Environment: PY (interpreter), TRAIN_THREADS/POST_THREADS, JAX_CACHE, as in run_hub.sh.
# Command line: anything here is appended to BOTH runs, so
#     ./run_quad_chain.sh --vtk                      # also write the VisIt files
#     ./run_quad_chain.sh --lbfgs-steps 16000        # a longer cap
#     TAG=_trial ./run_quad_chain.sh --steps 3 --lbfgs-steps 1 --n-coll 128 --n-bnd 32 \
#                                    --width 8 --depth 2 --no-figures    # plumbing trial
#   (TAG suffixes both run directories; FORCE=1 redoes a run that already has params.pkl.)
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [ -z "${PY:-}" ]; then
    if [ -x "$HERE/.venv/bin/python" ]; then
        PY="$HERE/.venv/bin/python"
    else
        PY="python"
    fi
fi
if ! "$PY" -c "import stationary" 2>/dev/null; then
    echo "error: '$PY' cannot import the stationary package." >&2
    echo "       Use the checkout's environment:  PY=$HERE/.venv/bin/python $0 ..." >&2
    exit 1
fi

TAG="${TAG:-}"
FORCE="${FORCE:-0}"

# The shared physics: the production shell SCALED by 1/100 (rho in [0.01, 1], R0/100, inner
# areal radius 0.01), k = 1 derived, quarter quadrupole (3 S2 = -(1-eps) = -0.25), harmonic
# gauge from h, order-3 Robin conditions with lambda_inf = 1, weights and ramp as in the
# quarter run that failed.
BASE=(
    --arch axisym_hybrid
    --steps 500 --lbfgs-steps 8000
    --R0 0.005773502691896258 --rho-in 0.01 --inner-radius 0.01 --rho-out 1.0
    --ref-solution --ref-asymptotic 1.0
    --outer-bc robin --robin-exps 2,3,1 --robin-orders h=3,lam=3 --no-robin-G --lam-inf 1.0
    --decay-feature --radial log --pde-ramp-steps 500
    --w-inner 100 --w-outer 100 --reweight-every 1500
    --n-coll 16384 --n-bnd 1024
    --lam-bc-S2 -0.08333333333333333
)

show_verdict() {                # show_verdict NAME
    local out="runs/$1"
    if [ -f "$out/report.txt" ]; then
        grep -E "final loss|lambda\(rho_out\)|outer BC residuals|far-field VALUES|radial eq term|inner sphere areal|family read off|max \|lambda - lambda_fam" \
             "$out/report.txt" | sed 's/^/   /' || true
        "$PY" - "$out" <<'PYEOF' 2>/dev/null || true
import json, sys
try:
    r = json.load(open(sys.argv[1] + "/report.json"))
except Exception:
    raise SystemExit(0)
keys = ("pin_lam", "pin_h_tan", "pin_h_rr", "pde_lam_eq_radial", "qn_stopped_at")
print("   report.json: " + " ".join(f"{k}={r[k]:.3e}" if isinstance(r.get(k), float)
                                    else f"{k}={r.get(k)}" for k in keys if k in r))
PYEOF
    else
        echo "   (no report.txt -- check logs/$1.log)" >&2
    fi
}

run_one() {                     # run_one NAME -- <extra flags...>
    local name="$1"; shift 2
    local out="runs/$name"
    if [ "$FORCE" -eq 0 ] && [ -f "$out/params.pkl" ]; then
        echo "== $name: already done ($out/params.pkl) -- FORCE=1 to redo"
        show_verdict "$name"
        return 0
    fi
    # NOTE the invocation: run_hub.sh is a SHELL script, so it is run by bash with PY in its
    # environment.  Writing "$PY" run_hub.sh would hand the script to Python (SyntaxError).
    local -a cmd=(env "PY=$PY" JAX_ENABLE_X64=1 bash "$HERE/run_hub.sh" "${BASE[@]}" "$@" --outdir "$out")
    echo
    echo "===================================================================="
    echo "== $name"
    printf '  %q' "${cmd[@]}"; echo
    echo "===================================================================="
    "${cmd[@]}" || { echo "!! $name: run_hub.sh failed to launch" >&2; return 1; }
    local pid
    pid="$(cat "$out/run.pid" 2>/dev/null || true)"
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        echo "!! $name: not running -- see logs/$name.log" >&2
        tail -20 "logs/$name.log" 2>/dev/null
        return 1
    fi
    echo "-- $name running as pid $pid; waiting (log: logs/$name.log)"
    while kill -0 "$pid" 2>/dev/null; do sleep 20; done
    echo "-- $name finished"
    show_verdict "$name"
}

echo "chain: pinned value pins, then the radial-derivative equation term"
echo "       (run under nohup/tmux if the terminal may close: the runs themselves are"
echo "        detached, but this waiting loop is not)"

run_one "production_quad_quarter_pin$TAG" -- --pin-far --w-pin 100 "$@" \
    || echo "!! the pinned run did not start; continuing with the radial one anyway" >&2

run_one "production_quad_quarter_lamrad$TAG" -- --w-lam-eq-radial 1 "$@" \
    || echo "!! the radial run did not start" >&2

echo
echo "== both runs are done; comparing them against the earlier quarter run"
want=()
for n in production_quad_quarter production_quad_quarter_pin production_quad_quarter_lamrad; do
    [ -f "runs/$n${TAG}/params.pkl" ] && want+=("runs/$n${TAG}")
done
if [ "${#want[@]}" -ge 1 ]; then
    env PY="$PY" JAX_ENABLE_X64=1 "$PY" -m stationary.compare "${want[@]}" || true
else
    echo "   nothing to compare (no params.pkl found)"
fi
echo
echo "== verdict to look for: lambda(rho_out) -> 0.9885193 and k -> 1 in a run whose"
echo "   final loss is ~1e-10, with the plateau rule having stopped it before 8000"
echo "   iterations.  A run that lands at 0.31 with a tiny outer Robin residual and a"
echo "   PDE-dominated loss has gone to the same wrong branch as before."
