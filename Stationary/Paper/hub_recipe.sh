#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The paper's runs redone, or added, in ONE recipe (Paper/NOTES.md, "Evidence review", 2026-11-06).
#
# THE RECIPE (the final formulation of the quadrupole runs, ab_none):
#   inner sphere: lambda and the induced metric only (h_rr, h_rA free; current code, after 0773b6a);
#   outer sphere: Robin of order 3 on h and on the Geroch-Hansen potential P, none on Gamma, at
#                 rho_out only (no ladder), no averaged pins;
#   centre of mass: d = (its target) on six spheres spread over the shell, w_com = 100; the target
#                 is 0 for problem data, and the exact solution's own flux for exact data (Weyl):
#                 losses.com_target;
#   loss: rho^4 weights (--scale-exps 3,4,3,4) in a chart with rho_in = 1, w_inner = w_outer = 100;
#   network: metric only, depth 6; axisymmetric width 20 at 16384 + 1024 + 1024 points,
#            3-D width 40 at 32768 + 2048 + 2048;
#   training: Adam 5000, then SSBroyden (strong Wolfe) in stages of 10000, points redrawn every 500
#            (2 stages axisymmetric, 4 in 3-D); every run checked on 5 fresh samples.
# What a phase tests is the ONLY thing it changes.
#
# Phases, in priority order (each run skipped when its report.json exists, resumed from ckpt.pkl):
#   A  Weyl rods, axisymmetric, in the recipe (they were trained 2026-09 with the full exact h
#      imposed on the inner sphere, 4096 points, Armijo): wr_prod, wr_unequal, wr_close4, wr_close35.
#      Chart rho_in = 1 (rods scaled by 1/rho_in_phys); shells [7.5,75], [4,40], [3.5,35] physical.
#   B  off-axis quadrupole, width or points?  qp_rot3 = width 20 at 16384, qp_rot4 = width 40 at
#      32768 changed both.  rp_w40_n16384 and rp_w20_n32768 separate them.
#   C  the spectral reference at S2 = -1/12: l <= 14, 48 radial (tag l14) on ab_none_w2's problem;
#      A - B of the two references (scripts/compare_refs.py); ab_none_w2, fn_nocom_w2, fn_tan_w2 graded
#      against both (grade_rotated_final / _l14), moments against l14.
#   D  seeds 1 and 2 of ab_none (with COM), fn_nocom (without), fn_tan (with, inner data in H^1).
#   E  the hard end, width 40 at 65536 points redrawn every 250 (the width-40 runs at 32768 overfit:
#      fresh/train 11 at -1/4, 170 at -0.30): hs_025_n65k, hs_030_n65k.
#   F  rotated Weyl (rods of wr_close35 tilted 45 deg) in the 3-D recipe: wr_rot45.
#
#   bash Paper/hub_recipe.sh [PHASES]          detached (setsid nohup); PHASES default ABCDEF
#   bash Paper/hub_recipe.sh --run [PHASES]    the work itself;  log logs/hub_recipe.log
#   SMOKE=1 bash Paper/hub_recipe.sh --run A   tiny points/iterations into runs/_smoke/ (pipeline check)
# Estimated on one A30: A 1.7 h, B 2.5 h, C 2 h, D 2.5 h, E 3 h, F 2.2 h.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # <repo>/Stationary
cd "$HERE"
mkdir -p logs
if [ "${1:-}" != "--run" ]; then
    setsid nohup bash "$HERE/Paper/hub_recipe.sh" --run "${1:-ABCDEF}" >> logs/hub_recipe.log 2>&1 < /dev/null &
    echo "launched (pid $!); follow with:  tail -F $HERE/logs/hub_recipe.log"
    exit 0
fi
PHASES=${2:-ABCDEF}

PY="$HERE/.venv/bin/python"
export JAX_ENABLE_X64=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache-$USER

# ----------------------------------------------------------------- the recipe
NUM=(--depth 6 --ckpt-every 500 --resample-every 500 --decay-feature --log-resources --no-figures
     --outer-bc robin --robin-orders h=3,lam=3 --no-robin-G
     --eq-weights compat=0,ricci=1,gauge=1,lam_eq=1 --scale-exps 3,4,3,4
     --outer-shell-fracs 1.0 --no-pin-h-robin --no-pin-lam)
AX=(--arch axisym_hybrid --width 20 --n-coll 16384 --n-bnd 1024 --n-bnd-outer 1024)
TD=(--arch hybrid --width 40 --n-coll 32768 --n-bnd 2048 --n-bnd-outer 2048)
QP=(--R0 0.5773502691896258 --rho-in 1.0 --rho-out 100.0 --inner-radius 1.0 --vtk-physical-inner 1.0
    --lam0 0.33333333333333337 --lam-inf 1.0 --ref-solution --ref-asymptotic 1.0
    --com-pin 1,1.5,3,10,30,100)
S2_Q=-0.08333333333333333
ADAM=(--steps 5000 --plateau-min-iters -1 --lbfgs-steps 0)
SSB=(--steps 0 --qn-method ssbroyden --qn-linesearch wolfe --lbfgs-steps 10000 --plateau-min-iters -1)
NFRESH=5
SPEC_L14=(--nonlinear --ls 0,2,4,6,8,10,12,14 --n-radial 48 --n-coll 4000 --n-bnd 600 --iters 14 --tag l14)
R=runs
if [ "${SMOKE:-0}" = "1" ]; then          # the same pipeline, tiny: checks flags and tools only
    ADAM=(--steps 30 --plateau-min-iters -1 --lbfgs-steps 0)
    SSB=(--steps 0 --qn-method ssbroyden --qn-linesearch wolfe --lbfgs-steps 20 --qn-block 10 --plateau-min-iters -1)
    AX=(--arch axisym_hybrid --width 20 --n-coll 256 --n-bnd 64 --n-bnd-outer 64)
    TD=(--arch hybrid --width 40 --n-coll 256 --n-bnd 64 --n-bnd-outer 64)
    SPEC_L14=(--nonlinear --ls 0,2 --n-radial 6 --n-coll 300 --n-bnd 60 --iters 1 --tag l14smoke)
    NFRESH=1; R=runs/_smoke
fi
REF_FINAL=runs/ab_none_w2/specref_nonlinear_fine.npz      # l <= 10, 36 radial, final formulation
REF_L14=runs/ab_none_w2/specref_nonlinear_l14.npz
[ "${SMOKE:-0}" = "1" ] && REF_L14=runs/ab_none_w2/specref_nonlinear_l14smoke.npz
REF_025=runs/ms_s2_025_w2/specref_nonlinear_damp.npz       # l <= 14, 48 radial
REF_030=runs/ms_s2_030_w2/specref_nonlinear_big_l20.npz    # l <= 20, 48 radial
echo "=== hub_recipe $PHASES $(date -u '+%F %T UTC')  HEAD $(git rev-parse --short HEAD)  smoke=${SMOKE:-0}"

train() {   # train <name> <flags...>
    local name=$1; shift
    if [ -f "$R/$name/report.json" ]; then echo "--- $name: done, skipped"; return; fi
    local res=(); [ -f "$R/$name/ckpt.pkl" ] && res=(--resume auto) && echo "--- $name: resuming"
    echo "--- $name  $(date -u '+%T UTC')"
    "$PY" -u -m stationary.train --outdir "$R/$name" "$@" ${res[@]+"${res[@]}"}
}
chain() {   # chain <tag> <n stages> <flags...>: Adam, then n SSBroyden stages, then the fresh check
    local tag=$1 n=$2; shift 2
    train ${tag}_adam "$@" "${ADAM[@]}"
    local prev=${tag}_adam
    for i in $(seq 1 "$n"); do
        train ${tag}_w$i "$@" "${SSB[@]}" --init-from "$R/$prev/params.pkl"; prev=${tag}_w$i
    done
    [ -f "$R/$prev/fresh_samples.json" ] || "$PY" -u scripts/fresh_sample_loss.py "$R/$prev" --n $NFRESH
}
grade() {   # grade <run> <ref> <tag>
    "$PY" -u scripts/grade_rotated.py "$1" --ref "$2" --tag "$3" || echo "!!! grade $1 $3 failed"
}
moments() { # moments <run> [<ref> <tag>]
    if [ $# -ge 3 ]; then "$PY" -u scripts/gh_moments_runs.py "$1" --ref "$2" --tag "$3" || echo "!!! moments $1 failed"
    else "$PY" -u scripts/gh_moments_runs.py "$1" || echo "!!! moments $1 failed"; fi
}
note() { curl -s -H "Title: hub_recipe $1" -d "$2" https://ntfy.sh/pinn-reula-f0798d1ec301 > /dev/null || true; }
weyl() {    # weyl <rho_in physical> <rods: half-length a, b, half-gap in physical units>: chart rho_in = 1
    "$PY" - "$@" <<'EOF'
import sys
r, a, b, g = map(float, sys.argv[1:])
print(f"--weyl --weyl-half-length {a/r!r} --weyl-half-length-b {b/r!r} --weyl-half-gap {g/r!r} "
      f"--rho-in 1.0 --rho-out 10.0 --vtk-physical-inner {r!r} --com-pin 1,1.5,3,6,10")
EOF
}

# ----------------------------------------------------------------- A: Weyl, axisymmetric
if [[ $PHASES == *A* ]]; then
    for v in "wr_prod:7.5 1 1 0.5" "wr_unequal:7.5 1 0.1 0.5" "wr_close4:4 1 0.1 0.5" "wr_close35:3.5 1 0.1 0.5"; do
        tag=${v%%:*}; read -r -a W <<< "$(weyl ${v#*:})"
        chain $tag 2 "${NUM[@]}" "${AX[@]}" "${W[@]}"
        moments $R/${tag}_w2
    done
    note A "Weyl rods in the recipe done (wr_prod, wr_unequal, wr_close4, wr_close35)"
fi
# ----------------------------------------------------------------- B: off-axis, width against points
if [[ $PHASES == *B* ]]; then
    ROT=(--lam-bc-axis 1,0,1 --lam-bc-S2 $S2_Q)
    chain rp_w40_n16384 4 "${NUM[@]}" "${QP[@]}" "${ROT[@]}" "${TD[@]}" --n-coll 16384 --n-bnd 1024 --n-bnd-outer 1024
    chain rp_w20_n32768 4 "${NUM[@]}" "${QP[@]}" "${ROT[@]}" "${TD[@]}" --width 20
    for r in rp_w40_n16384_w4 rp_w20_n32768_w4; do grade $R/$r $REF_FINAL final; moments $R/$r $REF_FINAL final; done
    note B "off-axis width/points pair done (rp_w40_n16384, rp_w20_n32768)"
fi
# ----------------------------------------------------------------- C: how good is the yardstick at -1/12?
if [[ $PHASES == *C* ]]; then
    [ -f "$REF_L14" ] || { echo "--- specref l14 on ab_none_w2  $(date -u '+%T UTC')"
                           "$PY" -u -m stationary.specref runs/ab_none_w2 "${SPEC_L14[@]}"; }
    "$PY" -u scripts/compare_refs.py runs/ab_none_w2 --a $REF_FINAL --b $REF_L14 --out fine_vs_l14 || true
    for r in ab_none_w2 fn_nocom_w2 fn_tan_w2; do
        grade runs/$r $REF_FINAL final; grade runs/$r $REF_L14 l14; moments runs/$r $REF_L14 l14
    done
    note C "-1/12 reference l14 and regrades done"
fi
# ----------------------------------------------------------------- D: seeds
if [[ $PHASES == *D* ]]; then
    for s in 1 2; do
        chain sd${s}_ab_none  2 "${NUM[@]}" "${QP[@]}" "${AX[@]}" --lam-bc-S2 $S2_Q --seed $s
        chain sd${s}_fn_nocom 2 "${NUM[@]}" "${QP[@]}" "${AX[@]}" --lam-bc-S2 $S2_Q --seed $s --com-pin ""
        chain sd${s}_fn_tan   2 "${NUM[@]}" "${QP[@]}" "${AX[@]}" --lam-bc-S2 $S2_Q --seed $s --inner-tangential
        for r in sd${s}_ab_none_w2 sd${s}_fn_nocom_w2 sd${s}_fn_tan_w2; do
            grade $R/$r $REF_FINAL final; moments $R/$r $REF_FINAL final
        done
    done
    note D "seeds 1, 2 of ab_none, fn_nocom, fn_tan done"
fi
# ----------------------------------------------------------------- E: the hard end, denser points
if [[ $PHASES == *E* ]]; then
    DENSE=(--width 40 --n-coll 65536 --n-bnd 4096 --n-bnd-outer 4096 --resample-every 250)
    [ "${SMOKE:-0}" = "1" ] && DENSE=(--width 40 --resample-every 250)
    chain hs_025_n65k 3 "${NUM[@]}" "${QP[@]}" "${AX[@]}" --lam-bc-S2 -0.25 "${DENSE[@]}"
    chain hs_030_n65k 3 "${NUM[@]}" "${QP[@]}" "${AX[@]}" --lam-bc-S2 -0.30 "${DENSE[@]}"
    grade $R/hs_025_n65k_w3 $REF_025 damp; moments $R/hs_025_n65k_w3 $REF_025 damp
    grade $R/hs_030_n65k_w3 $REF_030 big_l20; moments $R/hs_030_n65k_w3 $REF_030 big_l20
    note E "hard end at 65536 points done (hs_025_n65k, hs_030_n65k)"
fi
# ----------------------------------------------------------------- F: rotated Weyl, 3-D
if [[ $PHASES == *F* ]]; then
    read -r -a W <<< "$(weyl 3.5 1 0.1 0.5)"
    chain wr_rot45 4 "${NUM[@]}" "${TD[@]}" "${W[@]}" --weyl-rotate-deg 45
    moments $R/wr_rot45_w4
    note F "rotated Weyl in the 3-D recipe done (wr_rot45)"
fi
# ----------------------------------------------------------------- G, H: complete phase B's 2 x 2 (Mendieta)
# width 20 / 40 x 16384 / 32768 points.  qp_rot3 (20, 16384) and qp_rot4_n32768 (40, 32768) have the
# recipe's config exactly (config diff: only the later default fields), but were graded against an
# older reference: G regrades them against REF_FINAL.  H retrains (40, 32768) in the recipe, a
# reproduction check of qp_rot4.  Mendieta: sbatch ... --wrap "bash Paper/hub_recipe.sh --run G".
if [[ $PHASES == *G* ]]; then
    for r in qp_rot3_w4 qp_rot4_n32768_w4; do grade $R/$r $REF_FINAL final; moments $R/$r $REF_FINAL final; done
    note G "qp_rot3, qp_rot4_n32768 regraded against the final reference"
fi
if [[ $PHASES == *H* ]]; then
    ROT=(--lam-bc-axis 1,0,1 --lam-bc-S2 $S2_Q)
    chain rp_w40_n32768 4 "${NUM[@]}" "${QP[@]}" "${ROT[@]}" "${TD[@]}"
    grade $R/rp_w40_n32768_w4 $REF_FINAL final; moments $R/rp_w40_n32768_w4 $REF_FINAL final
    note H "rp_w40_n32768 (width 40, 32768 points, recipe) done"
fi
# ----------------------------------------------------------------- I, J: H^1 inner data off-axis, bigger network
# The author's note: at width 40 (rot_tan) the H^1 inner term slowed training and cost more than it
# gained.  Does a bigger network change that?  Width 60 at 65536 points (G/H: a wider network needs
# more points), without (I) and with (J) --inner-tangential; nothing else differs.  One job each.
# Width 60 has 19027 parameters: the dense SSBroyden inverse Hessian is 2.70 GB, over the default
# 2 GB cap (--qn-max-H-gb), under which train.py silently falls back to L-BFGS -- the first I/J runs
# (2026-11-08, moved to runs/_invalid/) did, and stalled at loss 2e-2.  The cap is raised here.
W60=(--width 60 --n-coll 65536 --n-bnd 4096 --n-bnd-outer 4096 --qn-max-H-gb 4)
if [[ $PHASES == *I* ]]; then
    ROT=(--lam-bc-axis 1,0,1 --lam-bc-S2 $S2_Q)
    chain rp_w60_n65k 4 "${NUM[@]}" "${QP[@]}" "${ROT[@]}" "${TD[@]}" "${W60[@]}"
    grade $R/rp_w60_n65k_w4 $REF_FINAL final; moments $R/rp_w60_n65k_w4 $REF_FINAL final
    note I "rp_w60_n65k done"
fi
if [[ $PHASES == *J* ]]; then
    ROT=(--lam-bc-axis 1,0,1 --lam-bc-S2 $S2_Q)
    chain rp_w60_n65k_tan 4 "${NUM[@]}" "${QP[@]}" "${ROT[@]}" "${TD[@]}" "${W60[@]}" --inner-tangential
    grade $R/rp_w60_n65k_tan_w4 $REF_FINAL final; moments $R/rp_w60_n65k_tan_w4 $REF_FINAL final
    note J "rp_w60_n65k_tan done"
fi
# ----------------------------------------------------------------- K: Schwarzschild tests with Wolfe
# NOT the paper's Schwarzschild runs: with order 3 and p = (2,1,2) they leave errors ~2e-4 (2026-11-09).
# The paper's are order 2, Paper/commands_schwarzschild.sh (sx_*, on Mendieta).
# Paper/commands_schwarzschild.sh (Armijo, order-2 Robin on h, Gamma and P for all three) redone with
# the strong-Wolfe line search used everywhere else, and the outer conditions of the paper: order 3 on
# h and P; the first-order network, which has Gamma as an output, also gets the order-3 condition on
# Gamma.  Spherical ansatz, 20000 Adam + 2000 SSBroyden as before.
SCHW=(--R0 1.0 --rho-in 2.0 --rho-out 100.0 --outer-bc robin --ref-solution --ref-asymptotic 1.0
      --lam0 0.3819660112501051 --lam-inf 1.0 --width 64 --depth 4 --fourier 8 --decay-feature
      --n-coll 4096 --n-bnd 256 --steps 20000 --pde-ramp-steps 1000 --reweight-every 2000
      --qn-method ssbroyden --qn-linesearch wolfe --lbfgs-steps 2000 --plateau-min-iters -1
      --no-pin-h-robin --resample-every 500 --ckpt-every 500 --no-figures)
if [[ $PHASES == *K* ]]; then
    train sw_first_order "${SCHW[@]}" --arch sym --robin-source --robin-orders h=3,G=3,lam=3
    train sw_metric_manuf "${SCHW[@]}" --arch sym_hybrid --robin-source --robin-orders h=3,lam=3 --no-robin-G
    train sw_metric_real "${SCHW[@]}" --arch sym_hybrid --robin-orders h=3,lam=3 --no-robin-G
    note K "Schwarzschild tests with Wolfe done"
fi
echo "=== hub_recipe $PHASES finished $(date -u '+%F %T UTC')"
note done "phases $PHASES finished"
