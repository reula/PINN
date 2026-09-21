#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The control ladder, run in sequence (ONE GPU: never in parallel), followed by a
# side-by-side comparison of everything it produced.
#
#   PY=$PWD/.venv/bin/python ./run_ladder.sh            # steps 1-3  (~30 min)
#   PY=$PWD/.venv/bin/python ./run_ladder.sh 3 4        # chosen steps
#   PY=$PWD/.venv/bin/python ./run_ladder.sh all        # 1-5 (~2-4 h, unattended)
#   PY=$PWD/.venv/bin/python ./run_ladder.sh --dry-run  # print the commands, launch nothing
#   PY=$PWD/.venv/bin/python ./run_ladder.sh --compare  # only re-print the comparison table
#   PY=$PWD/.venv/bin/python ./run_ladder.sh --clean          # list what a clean would delete
#   PY=$PWD/.venv/bin/python ./run_ladder.sh --clean --force  # delete it (this ladder only)
#   PY=$PWD/.venv/bin/python ./run_ladder.sh sweep      # measure w_outer per order (~10 min)
#
# Recommended order after a clean slate:
#   sweep -> read the table -> rerun the controls with the measured weights:
#       W_OUTER_ORD2=<from the sweep> W_OUTER_ORD4=<from the sweep> ./run_ladder.sh 1 2 3
#   then the physics:  ./run_ladder.sh 5 6 7
#
#   step 1  order 1, float32, 64x4 f8      runs/control_ord1b      ~7 min
#   step 2  order 2, float32, same size    runs/control_ord2       ~7 min
#   step 3  order 4, x64,     same size    runs/control_ord4_x64   ~15 min
#   step 4  order 1, float32, BIG net      runs/control_ord1_big   ~20-40 min
#   step 5  the dipole, order 1, small net runs/dipole_small       ~7 min
#   step 6  the dipole, order 1, BIG net   runs/dipole_big         ~20-40 min
#   step 7  the dipole at order 2 (same weight as control_ord2)    runs/dipole_ord2_small
#           -- this is the direct test of the claim that order 1 is enough for the dipole:
#           if the l = 1 amplitude and lambda(rho_out) agree with step 5 to ~1e-5, it is.
#
# Steps 1-3 were run on 20 September.  They at first looked like "order 1 wins by two
# orders of magnitude", but that was an artefact of the OUTER WEIGHT: the order-n Robin
# residual at initialisation is ~(gain)^n larger (8.8e-4 for n = 1, 2.1e-1 for n = 2), so
# w_outer = 100 means something completely different at each order and the stiff outer term
# takes over the (globally clipped) gradient.  A fixed-budget sweep with the weight matched
# gives order 2 *better* than order 1:
#
#   order 1, w_outer 100   loss 3.4e-04   lam_eq 1.8e-02
#   order 2, w_outer 100   loss 1.2e-03   lam_eq 3.1e-02   <- the ladder runs
#   order 2, w_outer  10   loss 2.3e-04   lam_eq 1.2e-02
#   order 2, w_outer   1   loss 4.5e-04   lam_eq 7.8e-03
#
# So step 4 re-runs order 2 at full budget with a matched weight.  Order 1 is still the
# right choice for the physics for now: the dipole's l = 1 tail at rho_out is only
# ~S1 (rho_in/rho_out)^2 = 1e-5, so an order-1 condition biases it by ~1e-5, thirty times
# below the accuracy the control reaches (3e-4).  Higher order will matter again if we push
# the solution below ~1e-4, because order 1 is inconsistent with the exact solution at the
# 6.6e-5 level.  See HUB.md section 6.
#
# Each step is launched with run_hub.sh, which detaches it, logs it, checkpoints it and
# runs the figures + report when it ends; this script only waits for the run to finish
# before starting the next one.  A step whose run directory already has params.pkl is
# skipped (use --force to redo it).
#
# Environment: PY (interpreter, as in run_hub.sh), plus the usual run_hub.sh overrides.
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="${PY:-python}"
FORCE=0
DRY=0
CLEAN=0
COMPARE_ONLY=0
WANT=()

# The shared physics: the M2 control geometry, k = 1 (derived), lambda -> 1, order-1..4
# Robin conditions chosen per step, weights and ramp as in HUB.md section 7.
BASE=(--arch axisym_hybrid --steps 20000 --ref-solution --ref-asymptotic 1.0
      --R0 0.5773502691896258 --rho-in 1.0 --inner-radius 1.0 --rho-out 100
      --outer-bc robin --no-robin-G --lam-inf 1.0 --decay-feature --radial log
      --pde-ramp-steps 500 --w-inner 100 --w-outer 100 --reweight-every 1500)
SMALL=(--n-coll 4096  --n-bnd 256  --width 64  --depth 4 --fourier 8)
BIG=(--n-coll 16384 --n-bnd 1024 --width 256 --depth 6 --fourier 16)
NICE=(--lbfgs-steps 3000)

LADDER_RUNS=(control_ord1b control_ord2 control_ord4_x64 control_ord1_big \
             dipole_small dipole_big dipole_ord2_small)
SWEEP_RUNS=(sweep_ord2_w100 sweep_ord2_w10 sweep_ord2_w1 \
            sweep_ord4_w100 sweep_ord4_w10 sweep_ord4_w1)

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --force)   FORCE=1; shift ;;
        --compare) COMPARE_ONLY=1; shift ;;
        --clean)   CLEAN=1; shift ;;
        sweep)     WANT=(0); shift ;;
        all)       WANT=(1 2 3 4 5 6 7); shift ;;
        [1-7])     WANT+=("$1"); shift ;;
        *) echo "unknown argument '$1' (want sweep, 1..7, all, --dry-run, --force, --compare, --clean)" >&2
           exit 1 ;;
    esac
done
# --compare means "print the table, launch nothing"
if [ "$COMPARE_ONLY" = "1" ]; then
    WANT=()
elif [ ${#WANT[@]} -eq 0 ]; then
    WANT=(1 2 3)
fi

# ------------------------------------------------------------------ one step
run_one() {                     # run_one NAME X64(0|1) -- <extra flags...>
    local name="$1" x64="$2"; shift 3
    local out="runs/$name"
    if [ "$FORCE" -eq 0 ] && [ -f "$out/params.pkl" ]; then
        echo "== $name: already done ($out/params.pkl) -- --force to redo"
        return 0
    fi
    # NOTE the invocation: run_hub.sh is a SHELL script, so it is run by bash with PY in
    # its environment (env VAR=value bash script ...).  Writing "$PY" run_hub.sh would
    # hand the script to Python and produce a SyntaxError about a `case` line.
    local -a cmd=(env "PY=$PY")
    [ "$x64" = "1" ] && cmd+=(JAX_ENABLE_X64=1)
    # LADDER_EXTRA overrides anything, for a quick trial of the machinery:
    #   LADDER_EXTRA="--steps 20 --lbfgs-steps 0 --n-coll 64 --n-bnd 32" ./run_ladder.sh 1
    local -a tail_args=("$@")
    # (never expand an empty array: bash 3.2 with set -u calls that unbound)
    if [ -n "${LADDER_EXTRA:-}" ]; then
        local -a extra=()
        read -r -a extra <<< "$LADDER_EXTRA"
        tail_args+=("${extra[@]}")
    fi
    cmd+=(bash "$HERE/run_hub.sh" "${BASE[@]}" "${tail_args[@]}" --outdir "$out")
    echo
    echo "===================================================================="
    echo "== step $name"
    printf '  %q' "${cmd[@]}"; echo
    echo "===================================================================="
    if [ "$DRY" = "1" ]; then return 0; fi
    "${cmd[@]}" || { echo "!! $name: run_hub.sh failed to launch" >&2; return 1; }
    local pid
    pid="$(cat "$out/run.pid" 2>/dev/null || true)"
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        echo "!! $name: not running -- see logs/$(basename "$out").log" >&2
        tail -20 "logs/$(basename "$out").log" 2>/dev/null
        return 1
    fi
    echo "-- $name running as pid $pid; waiting (log: logs/$name.log)"
    while kill -0 "$pid" 2>/dev/null; do sleep 20; done
    echo "-- $name finished"
    # the run wrote its report itself; show the lines that decide the next step
    if [ -f "$out/report.txt" ]; then
        grep -E "final loss|lambda\(rho_out\)|outer BC residuals|inner BC residuals|inner sphere areal|family read off|max \|lambda - lambda_fam" \
             "$out/report.txt" | sed 's/^/   /' || true
    else
        echo "   (no report.txt -- check logs/$name.log)" >&2
    fi
}

# ------------------------------------------------------------------ clean
# Deletes ONLY what this ladder produces (the run directories above and their logs), and
# lists them first: `--clean` on its own just prints the plan, `--clean --force` deletes.
if [ "$CLEAN" = "1" ]; then
    echo "this ladder owns these; nothing else is touched:"
    for n in "${LADDER_RUNS[@]}" "${SWEEP_RUNS[@]}"; do
        [ -e "runs/$n" ] && echo "  runs/$n"
        [ -e "logs/$n.log" ] && echo "  logs/$n.log"
    done
    [ -n "${LADDER_EXTRA:-}" ] && echo "  (LADDER_EXTRA is set: $LADDER_EXTRA)"
    if [ "$FORCE" != "1" ]; then
        echo
        echo "--clean alone does not delete anything.  Re-run with --clean --force."
        exit 0
    fi
    for n in "${LADDER_RUNS[@]}" "${SWEEP_RUNS[@]}"; do
        rm -rf "runs/$n" "logs/$n.log" "logs/$n.resume.log"
    done
    echo "deleted.  runs/ now holds:"
    ls -1 runs/ | sed 's/^/  /'
    exit 0
fi

# ------------------------------------------------------------------ sweep
# The order-n Robin residual at initialisation is (gain)^n larger, so one fixed w_outer is
# not a fair comparison across orders; this measures the weight instead of guessing it.
# Short runs (1/4 of the steps, 1/4 of the collocation) for two orders and three weights,
# then a table of the WEIGHT-INDEPENDENT metrics -- the loss alone shrinks just by lowering
# the weight, so judge on the PDE residuals and the outer BC residuals.
SWEEP=(--steps 5000 --lbfgs-steps 500 --n-coll 1024 --n-bnd 256)
run_sweep() {
    local n="$1" x64="$2"; shift 2
    local out="runs/$n"
    if [ -f "$out/params.pkl" ] && [ "$FORCE" != "1" ]; then
        echo "== $n: already done (--force to redo)"; return 0
    fi
    local -a cmd=(env "PY=$PY")
    [ "$x64" = "1" ] && cmd+=(JAX_ENABLE_X64=1)
    local -a tail_args=("$@")
    cmd+=(bash "$HERE/run_hub.sh" "${BASE[@]}" "${SWEEP[@]}" "${tail_args[@]}" --outdir "$out")
    echo; echo "== sweep $n"
    printf '  %q' "${cmd[@]}"; echo
    if [ "$DRY" = "1" ]; then return 0; fi
    "${cmd[@]}" >/dev/null 2>&1 || { echo "!! $n failed to launch" >&2; return 1; }
    local pid; pid="$(cat "$out/run.pid" 2>/dev/null || true)"
    while [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; do sleep 10; done
    echo "-- $n done"
}

if [ "${WANT[0]:-}" = "0" ]; then
    for w in 100 10 1; do run_sweep "sweep_ord2_w$w" 0 --robin-orders h=2,lam=2 --w-outer "$w"; done
    for w in 100 10 1; do run_sweep "sweep_ord4_w$w" 1 --robin-orders h=4,lam=4 --w-outer "$w"; done
    if [ "$DRY" = "1" ]; then echo; echo "(--dry-run: nothing launched)"; exit 0; fi
    echo
    echo "=== weight-independent metrics per weight ==="
    MPLBACKEND=Agg MPLCONFIGDIR="${MPLCONFIGDIR:-$HERE/.mplcache}" \
        "$PY" -m stationary.compare runs/sweep_ord2_w100 runs/sweep_ord2_w10 runs/sweep_ord2_w1 \
            runs/sweep_ord4_w100 runs/sweep_ord4_w10 runs/sweep_ord4_w1 \
            2>&1 | tee logs/sweep_compare.txt
    echo
    echo "read the rows 'outer: h', 'outer: lambda', 'lam_eq', 'ricci' and 'difference at rho_out'."
    echo "then re-run the full-budget controls with the chosen weights, e.g.:"
    echo "  W_OUTER_ORD2=10 W_OUTER_ORD4=1 PY=... ./run_ladder.sh 1 2 3"
    exit 0
fi

# (bash 3.2 refuses to expand an EMPTY array under set -u, so guard the loop)
if [ ${#WANT[@]} -gt 0 ]; then
for s in "${WANT[@]}"; do
    case "$s" in
        1) run_one control_ord1b   0 -- "${SMALL[@]}" "${NICE[@]}" --robin-orders h=1,lam=1 ;;
        2) run_one control_ord2    0 -- "${SMALL[@]}" "${NICE[@]}" --robin-orders h=2,lam=2 \
                    --w-outer "${W_OUTER_ORD2:-10}" ;;
        3) run_one control_ord4_x64 1 -- "${SMALL[@]}" "${NICE[@]}" --robin-orders h=4,lam=4 \
                    --w-outer "${W_OUTER_ORD4:-1}" ;;
        4) run_one control_ord1_big 0 -- "${BIG[@]}"   "${NICE[@]}" --robin-orders h=1,lam=1 ;;
        5) run_one dipole_small     0 -- "${SMALL[@]}" "${NICE[@]}" --robin-orders h=1,lam=1 \
                    --lam-bc-S1 0.1 ;;
        6) run_one dipole_big       0 -- "${BIG[@]}"   "${NICE[@]}" --robin-orders h=1,lam=1 \
                    --lam-bc-S1 0.1 ;;
        7) run_one dipole_ord2_small 0 -- "${SMALL[@]}" "${NICE[@]}" --robin-orders h=2,lam=2 \
                    --w-outer "${W_OUTER_ORD2:-10}" --lam-bc-S1 0.1 ;;
    esac || true
done
fi

[ "$DRY" = "1" ] && { echo; echo "(--dry-run: nothing launched)"; exit 0; }

# ------------------------------------------------------------------ comparison
RUNS=()
for n in control_ord1b control_ord2 control_ord4_x64 control_ord1_big \
         dipole_small dipole_big dipole_ord2_small; do
    [ -f "runs/$n/config.json" ] && RUNS+=("runs/$n")
done
[ ${#RUNS[@]} -eq 0 ] && { echo "no runs to compare yet" >&2; exit 0; }

echo
echo "=== comparing ${RUNS[*]} ==="
mkdir -p logs
MPLBACKEND=Agg MPLCONFIGDIR="${MPLCONFIGDIR:-$HERE/.mplcache}" \
    "$PY" -m stationary.compare "${RUNS[@]}" --json logs/compare.json \
    2>&1 | tee logs/ladder_compare.txt

echo
echo "the table above is also in logs/ladder_compare.txt (raw numbers: logs/compare.json)"
