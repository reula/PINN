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
        JID[$name]=$(sbatch --parsable -J "$name" "${depflag[@]}" mendieta_train.sbatch "$name" "$@")
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
submit m2_dsg_then_ssb      m2_dsg_stage1 --steps 0 --init-from runs/m2_dsg_stage1/params.pkl \
                                          --qn-method ssbroyden --lbfgs-steps 20000
# SSBroyden continued from the hub's 20000-iteration result
submit m2_ssb_continued     - --steps 0 --init-from runs/ib16k_ssbroyden/params.pkl \
                              --qn-method ssbroyden --lbfgs-steps 40000
# round 3 (2026-10-16): the residuals weighted by rho^4 instead of rho^2 (scale_exps +2), the
# weighting under which the spectral reference's far field became determined (PERTURBATION.md 5.2b)
RHO4="--scale-exps 3,4,3,4"
submit m3_ssb_rho4          - --steps 0 --init-from $INIT --qn-method ssbroyden --lbfgs-steps 20000 $RHO4
submit m3_dsg_s2000_rho4    - $DSG --dsgnar-sketch 2000 --dsgnar-stage2-ratio 0.075 $RHO4
}
squeue -u "$USER" -o "%.10i %.24j %.8T %.10M %R" 2>/dev/null
