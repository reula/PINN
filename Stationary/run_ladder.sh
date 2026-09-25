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
#   ./run_ladder.sh sweep        # prints the weight-independent metrics per weight
#   W_OUTER_ORD2=10 W_OUTER_ORD4=1 ./run_ladder.sh 1 2 3     # put NUMBERS here
#   ./run_ladder.sh 4 5 6 7      # capacity, then the two dipoles and the order-2 check
#
# W_OUTER_ORD2/W_OUTER_ORD4 are optional NUMBERS taken from the sweep; the defaults are
# 10 (order 2) and 1 (order 4).  PY is optional too: without it the checkout's .venv is
# used, which on this hub is the CUDA environment.
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
# The Laplacian recipe (tests/test_laplace_robin.py, Laplace_Robin.md), as steps 8-10:
#   ./run_ladder.sh recipe        # 8 9 10
#   ./run_ladder.sh recipe         # 8 9 10 11 12, the approved set, in order:
#   step 8   recipe_control        control S1 = 0, order 1   (the lambda(100) = 0.9885 check)
#   step 9   recipe_dipole         dipole  S1 = 0.1, order 1 (the physics baseline)
#   step 10  recipe_ramp_ord{2,3}  the control continued at order 2 (w 10) then 3 (w 1),
#                                  each warm-started from the previous phase; A is phase 1
#   step 11  recipe_dipole_ord2    dipole at order 2, w_outer 10, cold-started
#   step 12  recipe_dipole_ord3    dipole at order 3, w_outer 1,  cold-started
#   (13 recipe_bignet and 14 recipe_ord3_alone are optional extras, not in `recipe`)
#
#   ./run_ladder.sh 15            # production_run_quad: the quadrupole inner data
#       lambda = lambda_0 - lambda_0 (1-eps) (z^2-(x^2+y^2)/2)/r^2,  eps = 0.1
#       i.e. --lam-bc-S2 -0.3 at lambda_0 = 1/3.  One run, cold-started, Robin order 3,
#       with --vtk: it writes runs/production_quad/vtk/solution.vtk for VisIt when it is
#       post-processed (graded Cartesian grid in PHYSICAL coordinates, 1..100).
#       QN_CAP=8000 ./run_ladder.sh 15     # if 20000 quasi-Newton iterations is too long
# The ingredients come from that file: a dense quasi-Newton wants a SMALL network (its
# inverse Hessian is n_params^2: 2250 params -> 40 MB, the 14533-parameter production net ->
# 1.57 GB), points saturate at 3x, the shell may be rescaled by 1/100 (a relabelling --
# verified numerically here, identical to float32 round-off), and the order ramp is the
# cheapest route to a small field error.
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
# PY is an environment variable (as in run_hub.sh).  If it is not given, use the
# checkout's own venv -- on this hub that is the CUDA environment -- rather than whatever
# `python` happens to be (a `(base)` conda python has no jax and no stationary).
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
FORCE=0
DRY=0
CLEAN=0
COMPARE_ONLY=0
WANT=()

# The shared physics: the M2 control geometry, k = 1 (derived), lambda -> 1, order-1..4
# Robin conditions chosen per step, weights and ramp as in HUB.md section 7.
# The production shell, SCALED by 1/100 (rho in [0.01, 1], R0/100, areal radius 0.01): the
# same problem as [1, 100] -- verified: identical loss groups at identical parameters, and
# two 400-step runs agreeing to 7e-8 at step 1 -- but every length is O(1), which is what
# keeps the residual and activation magnitudes in a sane range.  Never mix normalisations
# across the two shells (--scale-ref* on one side only breaks the equivalence by 100^p).
BASE=(--arch axisym_hybrid --steps 20000 --ref-solution --ref-asymptotic 1.0
      --R0 0.005773502691896258 --rho-in 0.01 --inner-radius 0.01 --rho-out 1.0
      --outer-bc robin --no-robin-G --lam-inf 1.0 --decay-feature --radial log
      --pde-ramp-steps 500 --w-inner 100 --w-outer 100 --reweight-every 1500)
SMALL=(--n-coll 4096  --n-bnd 256  --width 64  --depth 4 --fourier 8)
BIG=(--n-coll 16384 --n-bnd 1024 --width 256 --depth 6 --fourier 16)
NICE=(--lbfgs-steps 3000)

# The Laplacian recipe's base: the SAME problem with rho rescaled by 1/100 (shell [0.01, 1],
# R0 -> R0/100), which is a relabelling -- verified numerically: with the default residual
# normalisation (local rho, i.e. no --scale-ref*) the two shells agree to float32 round-off
# (7e-8 at step 1) and stay within 4% after 400 steps.  Do NOT mix normalisations across the
# two shells: --scale-ref-rho-in on one side only breaks the equivalence by 100^p.
# The network defaults are now the Laplacian's 20x6 with no Fourier features, so the recipe
# steps only add the point count and the quasi-Newton budget.
QNET=()
# 3x the round numbers, which is where that file measures the field error to saturate.
# more than 2000 evaluation points: 16384 collocation + 1024 on each sphere
QPT=(--n-coll 16384 --n-bnd 1024)
# Short Adam warm-up (the quasi-Newton phase needs a single fixed batch and a bracketed line
# search) and then SSBroyden to convergence; Crunch ships with this repo (Jax/ is tracked),
# so a git pull puts it on the hub.
# Adam warm-up capped at 2000 steps and the quasi-Newton phase at 6000 iterations; both stop
# early on the plateau rule (total loss AND outer Robin, 1e-4 relative over 3 blocks of 100,
# the package defaults), so the caps are only a safety net.
# NO ADAM PHASE by default: start cold in the quasi-Newton phase.  Measured: a cold
# --steps 0 run goes 4.75 -> 3.2e-02 in 100 iterations, and the Adam warm-up was what made
# initial_scale necessary (with H = I the first step is -grad, which the Wolfe search cannot
# bracket AFTER Adam has enlarged it).  If a phase ever reports `0 iterations, status 3
# (zoom failed)`, put a short warm-up back with --steps 500.  Note that the adaptive PDE
# reweighting, the pde_ramp_steps ramp and the resampling all live inside the Adam loop and
# therefore do not run: the PDE groups keep their configured (dimensional) weights.
QADAM=(--steps 0 --lbfgs-steps 6000)
# WARM STARTS RUN NO ADAM (--steps 0, straight into SSBroyden).  Restarting Adam at
# lr = 1e-3 on a solution the quasi-Newton phase has already converged is how a warm-started
# phase diverges: Adam is not a descent method and that step is huge next to the local
# curvature.  The quasi-Newton phase cannot diverge -- Crunch's line search returns the state
# UNCHANGED when it fails (handle_ls_failure) -- so removing the Adam warm-up removes the
# only way a phase can blow up.  Measured on a cold start: 4.75 -> 3.2e-02 in 100 iterations
# with no Adam at all.
QADAM_RAMP=(--steps 0 --lbfgs-steps 6000)

LADDER_RUNS=(control_ord1b control_ord2 control_ord4_x64 control_ord1_big \
             dipole_small dipole_big dipole_ord2_small \
             recipe_control recipe_dipole recipe_ramp_ord2 recipe_ramp_ord3 \
             recipe_dipole_ord2 recipe_dipole_ord3 \
             recipe_bignet recipe_ord3_alone \
             production_quad)
SWEEP_RUNS=(sweep_ord2_w100 sweep_ord2_w10 sweep_ord2_w1 \
            sweep_ord4_w100 sweep_ord4_w10 sweep_ord4_w1)

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --force)   FORCE=1; shift ;;
        --compare) COMPARE_ONLY=1; shift ;;
        --clean)   CLEAN=1; shift ;;
        sweep)     WANT=(0); shift ;;
        recipe)    WANT=(8 9 10 11 12); shift ;;
        [89]|10|11|12|13|14|15) WANT+=("$1"); shift ;;
        all)       WANT=(1 2 3 4 5 6 7); shift ;;

        [1-7])     WANT+=("$1"); shift ;;
        *) echo "unknown argument '$1' (want sweep, recipe, 1..10, all, --dry-run, --force, --compare, --clean)" >&2
           exit 1 ;;
    esac
done
# --compare means "print the table, launch nothing"
if [ "$COMPARE_ONLY" = "1" ]; then
    WANT=()
elif [ ${#WANT[@]} -eq 0 ]; then
    WANT=(1 2 3)
fi

# ------------------------------------------------------------------ weights
# W_OUTER_ORD2 / W_OUTER_ORD4 are the Robin weights for the higher-order controls, chosen
# from `./run_ladder.sh sweep`.  They are NUMBERS; anything else (a leftover "<from the
# sweep>", say) is refused here rather than confusing bash later.
for v in W_OUTER_ORD2 W_OUTER_ORD4; do
    eval "val=\${$v:-}"
    if [ -n "$val" ] && ! printf '%s' "$val" | grep -Eq '^[0-9]+([.][0-9]*)?$'; then
        echo "error: $v='$val' is not a number." >&2
        echo "       Run ./run_ladder.sh sweep first, then e.g." >&2
        echo "         W_OUTER_ORD2=10 W_OUTER_ORD4=1 ./run_ladder.sh 1 2 3" >&2
        echo "       or simply omit them: the defaults are 10 (order 2) and 1 (order 4)." >&2
        exit 1
    fi
done

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
        # A: the control, S1 = 0, order 1
        8) run_one recipe_control 1 -- "${QPT[@]}" "${QADAM[@]}" --robin-orders h=1,lam=1 ;;
        # B: the dipole, S1 = 0.1, order 1
        9) run_one recipe_dipole  1 -- "${QPT[@]}" "${QADAM[@]}" --robin-orders h=1,lam=1 \
                    --lam-bc-S1 0.1 ;;
        # the dipole at the two higher Robin orders, cold-started, each at its own weight
        # (10 measured best for order 2, 1 for order 3 -- the weight is not transferable
        # across orders, so the three dipole runs are compared at each order's best)
        11) run_one recipe_dipole_ord2 1 -- "${QPT[@]}" "${QADAM[@]}" \
                    --robin-orders h=2,lam=2 --w-outer 10 --lam-bc-S1 0.1 ;;
        12) run_one recipe_dipole_ord3 1 -- "${QPT[@]}" "${QADAM[@]}" \
                    --robin-orders h=3,lam=3 --w-outer 1 --lam-bc-S1 0.1 ;;
        13) run_one recipe_bignet 1 -- "${QPT[@]}" "${QADAM[@]}" \
                    --width 64 --depth 4 --fourier 8 --qn-max-h-gb 3 \
                    --robin-orders h=1,lam=1 ;;
        14) run_one recipe_ord3_alone 1 -- "${QPT[@]}" "${QADAM[@]}" --robin-orders h=3,lam=3 \
                    --w-outer 1 ;;
        # production_run_quad: lambda = lambda_0 - lambda_0 (1-eps) (z^2-(x^2+y^2)/2)/r^2
        # with eps = 0.1, i.e. S2 = -lambda_0 (1-eps) = -0.3 at lambda_0 = 1/3.
        # ONE run, cold-started, Robin order 3 (the accurate condition: order 1 is floored at
        # 6.4e-05 by its own inconsistency, order 3 reproduces the exact solution to 1.6e-08).
        # The cap is a parameter because order 3 costs about 0.85 s per iteration at 16384
        # points on the hub: QN_CAP=20000 (the default) allows up to ~4.7 h, so lower it if
        # the run has to finish sooner -- the plateau rule stops it earlier when it can.
        15) run_one production_quad 1 -- "${QPT[@]}" "${QADAM[@]}" \
                    --lbfgs-steps "${QN_CAP:-20000}" \
                    --robin-orders h=3,lam=3 --w-outer 1 --lam-bc-S2 -0.3 --vtk ;;
        10) run_one recipe_ramp_ord2 1 -- "${QPT[@]}" "${QADAM_RAMP[@]}" \
                    --robin-orders h=2,lam=2 --w-outer 10 \
                    --init-from "$HERE/runs/recipe_control/params.pkl"
            run_one recipe_ramp_ord3 1 -- "${QPT[@]}" "${QADAM_RAMP[@]}" \
                    --robin-orders h=3,lam=3 --w-outer 1 \
                    --init-from "$HERE/runs/recipe_ramp_ord2/params.pkl" ;;
    esac || true
done
fi

[ "$DRY" = "1" ] && { echo; echo "(--dry-run: nothing launched)"; exit 0; }

# ------------------------------------------------------------------ comparison
RUNS=()
for n in control_ord1b control_ord2 control_ord4_x64 control_ord1_big \
         dipole_small dipole_big dipole_ord2_small \
         recipe_control recipe_dipole recipe_ramp_ord3 \
         recipe_dipole_ord2 recipe_dipole_ord3 production_quad; do
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
