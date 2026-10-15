#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Submit the Evolution_try DSGNAR experiments to Mendieta, one SLURM job (one A30) per run, all at
# once.  Run on the Mendieta login node, from anywhere:
#
#   bash ~/Julia/PINN/Evolution_try/scripts/mendieta_submit.sh          # submit what is not done
#   DRY=1 bash .../mendieta_submit.sh                                    # print, submit nothing
#   squeue -u $USER ;  tail -f ~/Julia/PINN/Evolution_try/logs/slurm_<label>_<jobid>.out
#
# The runs (README 6.11, 6.13): the paper's rules one at a time on the soft_all cell, all four on
# the soft_all and hard cells, then the two stall audits.  Baselines are the earlier batch's
# runs/dsgnar_soft_all_nc2201 (7.8e-06) and runs/dsgnar_hard_nc2201 (7.7e-03), on the hub.
# A run whose runs/<label>/report.md exists is skipped (also re-checked inside the job).
# ---------------------------------------------------------------------------
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
mkdir -p logs
DRY="${DRY:-0}"

WARM="--set window_warm_start=0.01"
OVER="--set dsgnar_reject_overshoot=true"
TARG="--set dsgnar_stage1_ratio=0.075 --set dsgnar_delta_min=1e-8"
RW="--set dsgnar_reweight_alpha=0.05"
ALL="$WARM $OVER $TARG $RW"

# label | window_ic | sketch | extra flags
RUNS=(
  "pp_soft_all_warm|soft_all|0|$WARM"
  "pp_soft_all_overshoot|soft_all|0|$OVER"
  "pp_soft_all_targets|soft_all|0|$TARG"
  "pp_soft_all_reweight|soft_all|0|$RW"
  "pp_soft_all_all|soft_all|0|$ALL"
  "pp_hard_all|hard|0|$ALL"
  "dsgnar_soft_all_nc2201_audit|soft_all|0|"
  "dsgnar_hard_nc2201_s1500_audit|hard|1500|"
)

echo "checkout $HERE  HEAD $(git rev-parse --short HEAD)"
for row in "${RUNS[@]}"; do
    IFS='|' read -r label ic sk extra <<<"$row"
    if [ -f "runs/$label/report.md" ]; then
        echo "  done     $label"; continue
    fi
    if squeue -h -u "$USER" -n "$label" -o %i | grep -q .; then
        echo "  queued   $label (job $(squeue -h -u "$USER" -n "$label" -o %i | head -1))"; continue
    fi
    # shellcheck disable=SC2086
    if [ "$DRY" = "1" ]; then
        echo "  DRY      sbatch -J $label scripts/mendieta_run.sbatch $label $ic $sk $extra"
    else
        jid=$(sbatch --parsable -J "$label" scripts/mendieta_run.sbatch "$label" "$ic" "$sk" $extra)
        echo "  submit   $label  -> job $jid"
    fi
done
squeue -u "$USER" -o "%.10i %.32j %.8T %.10M %R" 2>/dev/null
