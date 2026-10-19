#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Submit the Stationary optimiser runs to Mendieta, one job (one A30) each, all at once; a run
# that needs another run's parameters waits for it (--dependency=afterok).  On the login node:
#
#   bash ~/Julia/PINN/Stationary/mendieta_submit.sh          # DRY=1 to only print
#
# Every arm starts from runs/ib16k_adam/params_adam.pkl (the hub's warm-up, copied here), so the
# arms are comparable with each other and with the hub's ib16k_* runs (DSGNAR.md section 9.2).
# Round 2 (2026-10-16): DSGNAR without the stage-2 collapse, a larger sketch, a DSGNAR -> SSBroyden
# hand-over, and SSBroyden continued past its 20000-iteration cap.
# ---------------------------------------------------------------------------
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p logs
DRY="${DRY:-0}"
INIT="runs/ib16k_adam/params_adam.pkl"
[ -f "$INIT" ] || { echo "FATAL: $INIT missing (copy it from the hub)" >&2; exit 1; }

PAPER="--qn-method dsgnar --dsgnar-paper --dsgnar-reweight 0 --dsgnar-chunk 16 --log-every 10 --resample-every 100"
DSG="--steps 0 --init-from $INIT --lbfgs-steps 30000 --dsgnar-steps 300 $PAPER"

declare -A JID

submit() {      # submit <name> <dependency name or -> <flags...>
    local name="$1" dep="$2"; shift 2
    if [ -f "runs/$name/report.json" ]; then echo "  done     $name"; return; fi
    local q; q=$(squeue -h -u "$USER" -n "$name" -o %i | head -1)
    if [ -n "$q" ]; then echo "  queued   $name (job $q)"; JID[$name]=$q; return; fi
    local depflag=()
    if [ "$dep" != "-" ]; then
        if [ -n "${JID[$dep]:-}" ]; then depflag=(--dependency="afterok:${JID[$dep]}")
        elif [ ! -f "runs/$dep/params.pkl" ]; then echo "  SKIP     $name: needs $dep"; return; fi
    fi
    if [ "$DRY" = "1" ]; then
        echo "  DRY      sbatch -J $name ${depflag[*]:-} mendieta_train.sbatch $name $*"; JID[$name]=DRY
    else
        JID[$name]=$(sbatch --parsable -J "$name" --mem="${MEM:-48G}" "${depflag[@]}" mendieta_train.sbatch "$name" "$@")
        echo "  submit   $name -> job ${JID[$name]}${depflag:+ (after ${dep})}"
    fi
}

echo "checkout $HERE  HEAD $(git rev-parse --short HEAD)"
# shellcheck disable=SC2086
{
submit m2_dsg_s1100_t075    - $DSG --dsgnar-sketch 1100 --dsgnar-stage2-ratio 0.075   # never switch
submit m2_dsg_s1100_t15     - $DSG --dsgnar-sketch 1100 --dsgnar-stage2-ratio 0.15    # a reachable stage 2
submit m2_dsg_s2000_t075    - $DSG --dsgnar-sketch 2000 --dsgnar-stage2-ratio 0.075   # s = 0.88 d
# hand-over: DSGNAR's first stage (100 iterations), then SSBroyden from where it stopped
submit m2_dsg_stage1        - $DSG --dsgnar-sketch 1100 --dsgnar-stage2-ratio 0.075 --dsgnar-steps 100
# m2_dsg_then_ssb: ran (OOM-killed at 18250 of 20000, loss 2.04e-8; DSGNAR.md 9.2), not resubmitted
# submit m2_dsg_then_ssb    m2_dsg_stage1 --steps 0 --init-from runs/m2_dsg_stage1/params.pkl \
#                                         --qn-method ssbroyden --lbfgs-steps 20000
# SSBroyden continued from the hub's 20000-iteration result
submit m2_ssb_continued     - --steps 0 --init-from runs/ib16k_ssbroyden/params.pkl \
                              --qn-method ssbroyden --lbfgs-steps 40000
# round 3 (2026-10-16): the residuals weighted by rho^4 instead of rho^2 (scale_exps +2), the
# weighting under which the spectral reference's far field became determined (PERTURBATION.md 5.2b)
RHO4="--scale-exps 3,4,3,4"
# SSBroyden's host RSS grows ~3.3 MB per iteration (m2_dsg_then_ssb was killed at 18250 iterations
# and 60.9 GB; the nodes have 64 GB): two jobs of 10000, the second from the first's parameters
submit m3_ssb_rho4_a        - --steps 0 --init-from $INIT --qn-method ssbroyden --lbfgs-steps 10000 $RHO4
submit m3_ssb_rho4          m3_ssb_rho4_a --steps 0 --init-from runs/m3_ssb_rho4_a/params.pkl \
                              --qn-method ssbroyden --lbfgs-steps 10000 $RHO4
submit m3_dsg_s2000_rho4    - $DSG --dsgnar-sketch 2000 --dsgnar-stage2-ratio 0.075 $RHO4
# round 4 (2026-10-17), all from m3_dsg_s2000_rho4 (1.4e-6, still falling at its 300-iteration
# budget; SSBroyden from the rho^2 Adam start stalled at iteration 0, loss 1.2e4):
#   a. DSGNAR continued, same problem;  b. SSBroyden from there;
#   c. DSGNAR continued with the Robin condition on rho_out ONLY (no inner shells) and HYBRID
#      collocation (half log-uniform, half uniform in volume) -- every run so far sampled log only.
M3="runs/m3_dsg_s2000_rho4/params.pkl"
DSG4="$DSG --dsgnar-sketch 2000 --dsgnar-stage2-ratio 0.075 $RHO4 --init-from $M3"
submit m4_dsg_rho4_cont     m3_dsg_s2000_rho4 $DSG4
submit m4_ssb_rho4          m3_dsg_s2000_rho4 --steps 0 --init-from $M3 --qn-method ssbroyden \
                              --lbfgs-steps 10000 $RHO4
submit m4_dsg_rho4_hyb_1sh  m3_dsg_s2000_rho4 $DSG4 --outer-shell-fracs 1.0 --radial hybrid \
                              --radial-log-frac 0.5
#   d. DSGNAR continued with the dimensionless exponents and a far weight 1 + (rho/10)^4 on the
#      loss instead of rho^4 from rho_in: the inner equations keep their weight (Config.far_weight)
# m4_dsg_far10: cancelled before it ran (it had the lam pin; see round 5)
# submit m4_dsg_far10       m3_dsg_s2000_rho4 $DSG --dsgnar-sketch 2000 --dsgnar-stage2-ratio 0.075 \
#                             --init-from $M3 --far-weight 10,4
# round 5 (2026-10-17): the lam pin <(rho d_rho + 1)(lam - 1)> is violated by the exact solution
# (4.3e-7 at rho_out = 100, the whole rho^4 loss floor); --pin-lam-potential puts it on P.  All from
# m4_ssb_rho4 except the wider network, which needs its own warm-up.
M4="runs/m4_ssb_rho4/params.pkl"
PP="--pin-lam-potential"
submit m5_dsg_cyc           m4_ssb_rho4 $DSG --dsgnar-sketch 2000 --dsgnar-stage2-ratio 0.075 $RHO4 $PP \
                              --init-from $M4 --dsgnar-cycles 5 --dsgnar-cycle-steps 60
submit m5_ssb_norsmp        m4_ssb_rho4 --steps 0 --init-from $M4 --qn-method ssbroyden \
                              --lbfgs-steps 10000 --resample-every 0 $RHO4 $PP
submit m5_dsg_far10         m4_ssb_rho4 $DSG --dsgnar-sketch 2000 --dsgnar-stage2-ratio 0.075 $PP \
                              --init-from $M4 --far-weight 10,4
submit m5_w40               - --width 40 --depth 6 --steps 5000 --plateau-min-iters -1 --qn-method ssbroyden \
                              --lbfgs-steps 8000 $RHO4 $PP
# round 6 (2026-10-17): the wide network was still falling at its 8000-iteration cap (2.35e-7);
# 2 x 10000 more SSBroyden iterations (the RSS limit), no resampling
W40="--width 40 --depth 6 --steps 0 --qn-method ssbroyden --lbfgs-steps 10000 --resample-every 0 $RHO4 $PP"
submit m6_w40_a             m5_w40 $W40 --init-from runs/m5_w40/params.pkl
submit m6_w40               m6_w40_a $W40 --init-from runs/m6_w40_a/params.pkl
# round 7 (2026-10-18): the centre-of-mass gauge d = 0 (COM_GAUGE.md 4b) on six spheres, from the
# best run, whose d is -0.025 on every sphere (asymptotic chart translated by a ~ -0.04)
M5="runs/m5_ssb_norsmp/params.pkl"
COM="--com-pin 1,1.5,3,10,30,100"
submit m7_ssb_com           m5_ssb_norsmp --steps 0 --init-from $M5 --qn-method ssbroyden \
                              --lbfgs-steps 10000 --resample-every 0 $RHO4 $PP $COM
submit m7_dsg_com           m5_ssb_norsmp $DSG --dsgnar-sketch 2000 --dsgnar-stage2-ratio 0.075 $RHO4 $PP \
                              $COM --init-from $M5 --dsgnar-cycles 5 --dsgnar-cycle-steps 60
# round 8 (2026-10-18): (1) the COM-pinned run continued, and SSBroyden from it; (2) data WITHOUT the
# reflection symmetry, S1 = 0.05 (the reference for it is solved afterwards, all l = 0..10); (3) the
# harmonic (DeTurck) Ricci from the same start as m7_dsg_com
M7="runs/m7_dsg_com/params.pkl"
DSGC="$DSG --dsgnar-sketch 2000 --dsgnar-stage2-ratio 0.075 $RHO4 $PP $COM --dsgnar-cycle-steps 60"
submit m8_dsg_com_cont      m7_dsg_com $DSGC --init-from $M7 --dsgnar-cycles 5
submit m8_ssb_com           m7_dsg_com --steps 0 --init-from $M7 --qn-method ssbroyden \
                              --lbfgs-steps 10000 --resample-every 0 $RHO4 $PP $COM
submit m8_s1_dsg            m7_dsg_com $DSGC --init-from $M7 --dsgnar-cycles 8 --lam-bc-S1 0.05
submit m8_harm              m5_ssb_norsmp $DSGC --init-from $M5 --dsgnar-cycles 5 --ricci-form harmonic
# round 9 (2026-10-18): SSBroyden with the strong-Wolfe line search (--qn-linesearch wolfe) on the two
# starts where the Armijo version stalled at iteration 0, with the resampling those runs had (every
# 500), and the COM one also without resampling
WOLFE="--qn-method ssbroyden --qn-linesearch wolfe --lbfgs-steps 10000"
# m9_ssbw_rho4: ran (moved 1.2e4 -> 1.1e-2, then OOM at 60.6 GB after 911 iterations); not resubmitted
# submit m9_ssbw_rho4       - --steps 0 --init-from $INIT $WOLFE $RHO4
submit m9_ssbw_com          m5_ssb_norsmp --steps 0 --init-from $M5 $WOLFE $RHO4 $PP $COM
submit m9_ssbw_com_nors     m5_ssb_norsmp --steps 0 --init-from $M5 $WOLFE --resample-every 0 $RHO4 $PP $COM
# round 11 (2026-10-18): OFF-AXIS data and the 3-D network.  S2 = -1/12 about e = (1,0,1)/sqrt 2: the
# axisymmetric problem rotated by 45 deg, so the exact solution is the rotated axisymmetric one and the
# fine reference, rotated, grades it (scripts/grade_rotated.py).  The general metric-only network
# (hybrid, width 20 x depth 6: 2347 parameters), the recommended workflow, the COM gauge now on all
# three components.
ROT="--arch hybrid --width 20 --depth 6 --lam-bc-axis 1,0,1"
submit rot_adam             - --steps 5000 --plateau-min-iters -1 --lbfgs-steps 0 $ROT $RHO4 $PP $COM
submit rot_dsg              rot_adam $DSG --dsgnar-sketch 2000 --dsgnar-stage2-ratio 0.075 $ROT $RHO4 $PP $COM \
                              --init-from runs/rot_adam/params.pkl --dsgnar-cycles 5 --dsgnar-cycle-steps 60
submit rot_ssb              rot_dsg --steps 0 --init-from runs/rot_dsg/params.pkl --qn-method ssbroyden \
                              --qn-linesearch wolfe --lbfgs-steps 10000 --resample-every 500 \
                              --plateau-patience 8 --qn-reuse-objective $ROT $RHO4 $PP $COM
if [ "$DRY" != "1" ] && [ -n "${JID[rot_ssb]:-}" ] && ! squeue -h -u "$USER" -n check_rot -o %i | grep -q .; then
    echo "  submit   check_rot -> job $(GRADE_REF=runs/ib16k_ssbroyden/specref_nonlinear_fine.npz \
          sbatch --parsable -J check_rot --dependency=afterany:${JID[rot_ssb]} \
          mendieta_check.sbatch runs/rot_dsg runs/rot_ssb) (after rot_ssb)"
fi
# round 14 (2026-10-19): more DSGNAR.  rot_dsg2 (priority -- the physics run): the off-axis 3-D run
# continued from rot_ssb (2.36e-2, error the size of the quadrupole) with 10 cycles and the sketch at
# 0.9 d (2100 of 2347; rot_dsg's 2000 collapsed its trust region after cycle 2).  wf_dsg2: the
# workflow's DSGNAR continued (10 more cycles) from wf_dsg (8.6e-5), where SSBroyden took no first step.
submit rot_dsg2             rot_ssb $DSG --dsgnar-sketch 2100 --dsgnar-stage2-ratio 0.075 $ROT $RHO4 $PP $COM \
                              --init-from runs/rot_ssb/params.pkl --dsgnar-cycles 10 --dsgnar-cycle-steps 60
submit wf_dsg2              wf_dsg $DSG --dsgnar-sketch 2000 --dsgnar-stage2-ratio 0.075 $RHO4 $PP $COM \
                              --init-from runs/wf_dsg/params.pkl --dsgnar-cycles 10 --dsgnar-cycle-steps 60
if [ "$DRY" != "1" ] && [ -n "${JID[rot_dsg2]:-}" ] && ! squeue -h -u "$USER" -n check_rot2 -o %i | grep -q .; then
    echo "  submit   check_rot2 -> job $(GRADE_REF=runs/ib16k_ssbroyden/specref_nonlinear_fine.npz \
          sbatch --parsable -J check_rot2 --dependency=afterany:${JID[rot_dsg2]} \
          mendieta_check.sbatch runs/rot_dsg2) (after rot_dsg2)"
fi
# round 15 (2026-10-19): SSBroyden on the MERGED Crunch (now the default, --qn-crunch merged) from
# wf_dsg, where the legacy one took no step (rounds 10, 13); on the Mac it moves with both line
# searches.  Recipe step 4 as written, the Armijo default; compare with wf_dsg2 (more DSGNAR, same start)
submit wf_ssbm              wf_dsg --steps 0 --init-from runs/wf_dsg/params.pkl --qn-method ssbroyden \
                              --lbfgs-steps 10000 --resample-every 500 --plateau-patience 8 $RHO4 $PP $COM
# round 16 (2026-10-19): the off-axis 3-D run is stuck -- rot_dsg2 accepted 0 of 150 DSGNAR steps from
# rot_ssb (2.36e-2, Ricci-dominated, h error 6x the perturbation in rho 1-3).  Nothing has ever
# converged with the 3-D hybrid network (the Weyl runs were axisymmetric), so the CONTROL: the same
# network, budget and objective on ON-AXIS data (ctl_hyb_*; graded unrotated).  And SSBroyden on the
# merged Crunch from rot_ssb (rot_ssbm), where the legacy one stalled.
HYB="--arch hybrid --width 20 --depth 6"
submit ctl_hyb_adam         - --steps 5000 --plateau-min-iters -1 --lbfgs-steps 0 $HYB $RHO4 $PP $COM
submit ctl_hyb_dsg          ctl_hyb_adam $DSG --dsgnar-sketch 2100 --dsgnar-stage2-ratio 0.075 $HYB $RHO4 $PP $COM \
                              --init-from runs/ctl_hyb_adam/params.pkl --dsgnar-cycles 5 --dsgnar-cycle-steps 60
if [ "$DRY" != "1" ] && [ -n "${JID[ctl_hyb_dsg]:-}" ] && ! squeue -h -u "$USER" -n check_ctl -o %i | grep -q .; then
    echo "  submit   check_ctl -> job $(GRADE_REF=runs/ib16k_ssbroyden/specref_nonlinear_fine.npz \
          sbatch --parsable -J check_ctl --dependency=afterany:${JID[ctl_hyb_dsg]} \
          mendieta_check.sbatch runs/ctl_hyb_dsg) (after ctl_hyb_dsg)"
fi
submit rot_ssbm             rot_ssb --steps 0 --init-from runs/rot_ssb/params.pkl --qn-method ssbroyden \
                              --lbfgs-steps 10000 --resample-every 500 --plateau-patience 8 $ROT $RHO4 $PP $COM
if [ "$DRY" != "1" ] && [ -n "${JID[rot_ssbm]:-}" ] && ! squeue -h -u "$USER" -n check_rotm -o %i | grep -q .; then
    echo "  submit   check_rotm -> job $(GRADE_REF=runs/ib16k_ssbroyden/specref_nonlinear_fine.npz \
          sbatch --parsable -J check_rotm --dependency=afterany:${JID[rot_ssbm]} \
          mendieta_check.sbatch runs/rot_ssbm) (after rot_ssbm)"
fi
# round 12 (2026-10-18): SSBroyden's per-block memory growth.  The run that was killed at 60.6 GB
# (m9_ssbw_rho4), again, with --qn-reuse-objective (the same objective object across blocks and
# Crunch's minimize_bfgs jitted once: 5x faster on a CPU test, numerically the same up to evaluation
# order).  The log's per-block "resources: rss=" line is the measurement.
submit m12_ssbw_rho4_reuse  - --steps 0 --init-from $INIT $WOLFE $RHO4 --qn-reuse-objective
# round 10 (2026-10-18): the recommended workflow end to end, FROM SCRATCH (README "Recommended
# workflow"), with resampling throughout so the result is checked to be sample-stable:
#   wf_adam  Adam 5000 under the final objective (rho^4, pin on P, COM gauge)
#   wf_dsg   DSGNAR s = 2000, 5 restart cycles of <= 60 (each cycle a fresh sample)
#   wf_ssb   SSBroyden (strong Wolfe), resampling every 500, plateau over 8 blocks (a resample moves
#            the loss by ~ the 1% the plateau rule asks for, so 3 blocks stopped runs early)
#   then mendieta_check.sbatch: the loss on 5 fresh samples (scripts/fresh_sample_loss.py)
# 2026-10-19: wf_ssb and rot_ssb were cancelled before they started (jobs 1180615/1180619 and their
# checks) and resubmitted with --qn-reuse-objective: without it every Wolfe SSBroyden run so far grew
# 3-8 GB per block and the 10000-iteration plan would pass the 48 GB job limit (REPORT_OPTIMISERS.md)
BEST="$RHO4 $PP $COM"
submit wf_adam              - --steps 5000 --plateau-min-iters -1 --lbfgs-steps 0 $BEST
submit wf_dsg               wf_adam $DSG --dsgnar-sketch 2000 --dsgnar-stage2-ratio 0.075 $BEST \
                              --init-from runs/wf_adam/params.pkl --dsgnar-cycles 5 --dsgnar-cycle-steps 60
submit wf_ssb               wf_dsg --steps 0 --init-from runs/wf_dsg/params.pkl --qn-method ssbroyden \
                              --qn-linesearch wolfe --lbfgs-steps 10000 --resample-every 500 \
                              --plateau-patience 8 --qn-reuse-objective $BEST
if [ "$DRY" != "1" ] && [ -n "${JID[wf_ssb]:-}" ] && ! squeue -h -u "$USER" -n check -o %i | grep -q .; then
    echo "  submit   check -> job $(sbatch --parsable --dependency=afterany:${JID[wf_ssb]} \
          mendieta_check.sbatch runs/wf_dsg runs/wf_ssb) (after wf_ssb)"
fi
# round 13 (2026-10-19): wf_ssb (Wolfe) stalled at its start, 8 blocks of one failed line search
# (status 5 = maxiter in both Wolfe tries) -- as m9_ssbw_com did; the one SSBroyden run that moved
# from a COM start, m8_ssb_com, used Armijo and no resampling.  Both arms Armijo, with and without
# the resampling of step 4; --qn-reuse-objective (with the cache clearing on a new objective)
submit wf_ssba              wf_dsg --steps 0 --init-from runs/wf_dsg/params.pkl --qn-method ssbroyden \
                              --lbfgs-steps 10000 --resample-every 500 --plateau-patience 8 \
                              --qn-reuse-objective $BEST
submit wf_ssba_nors         wf_dsg --steps 0 --init-from runs/wf_dsg/params.pkl --qn-method ssbroyden \
                              --lbfgs-steps 10000 --resample-every 0 --qn-reuse-objective $BEST
}
squeue -u "$USER" -o "%.10i %.24j %.8T %.10M %R" 2>/dev/null
