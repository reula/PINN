#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One detached batch, two DSGNAR experiments, strictly one run at a time:
#
#   S. Stationary: can DSGNAR take these losses down?  Two changes are tested together.  The
#      sketch: Evolution_try measured s = d/3 -> d/2 as worth ~1000x, and Stationary ran at
#      s = 128 = d/18 because the M x s column matrix did not fit -- the fused JVP + CountSketch
#      removes it.  And the paper's own rules (DSGNAR.md section 9, `--dsgnar-paper`): per-
#      condition reweighting, overshoot rejection, no line scan, no calibration, targets
#      0.075/0.5, delta_min 1e-8.  From ONE Adam warm-up:
#        S0  Adam warm-up                       (writes params_adam.pkl; no quasi-Newton)
#        S1  SSBroyden, to its plateau          the baseline
#        S2  DSGNAR s = 1100, paper, no reweighting
#        S3  DSGNAR s = 1100, --dsgnar-paper    the paper, as written (with reweighting)
#        S4  DSGNAR s = 512,  paper, no reweighting   (the sketch size)
#        R   the nonlinear spectral reference (stationary/linref.py) and every arm against it
#      ib16k_*: the inner condition is the INDUCED metric only (2026-10-15).  Every earlier
#      Stationary run -- sk_*, sk16k_*, the pq_* production runs -- also imposed h_rA = 0 on the
#      inner sphere, which over-determines the quadrupole problem (losses.inner_bc_raw); none of
#      them is comparable with these.
#
#   E. Evolution_try, the open stall (its README 6.11).  Every DSGNAR phase that stops on
#      "radius below threshold" is audited against the EXACT Jacobian on the batch it stopped
#      on (--set dsgnar_audit_stops=true, wave_pinn/optim/stall_audit.py).  The first attempt
#      (dsgnar_hard_nc2201_audit, 2026-10-14, killed in window 10 with 110 records) found the
#      HARD hand-over's stops sketch-limited: s = 733 below rank(J), which grew 824 -> 1500 with
#      the window, and the stop loss tracked it (corr 0.84).  Two runs test that:
#        E1  soft_all, n_coll 2201, s = d/3   -- the hand-over whose error does NOT grow:
#                                                 are its stops sketch-limited too?
#        E2  hard,     n_coll 2201, s = 1500  -- s above rank(J): the rising floor should go
#      Before them, the paper's rules one at a time (Evolution README 6.13), on the soft_all
#      cell (baseline: runs/dsgnar_soft_all_nc2201, 7.8e-06) and all four on the hard cell
#      (baseline: runs/dsgnar_hard_nc2201, 7.7e-03), audits on throughout:
#        P*  warm start | overshoot rejection | targets 0.075 + delta_min 1e-8 | reweighting | all
#
#   bash run_batch_sketch.sh            # launch, detached; prints what is left to do
#   DRY=1 bash run_batch_sketch.sh      # print the plan, launch nothing
#   tail -f logs/batch_sketch.driver.log
#   cat logs/batch_sketch.summary.txt   # written at the end
#
# Resumable, which is what makes it a batch on a JupyterHub: closing the browser can get the
# server culled, and that kills every process in its container, setsid included (HUB.md, top).
# Nothing survives that; what this script guarantees is that the SAME command continues where
# it stopped -- a run whose report exists is skipped (Stationary: runs/<label>/report.json,
# Evolution: runs/<label>/report.md) and an interrupted run is restarted from its beginning.
# No run here is longer than ~1.5 h, so a death costs at most that one run.  After any
# interruption, relaunch with the same command.
#
# The card: the A30 is shared, and two jobs on it is a measured OOM.  The batch WAITS
# (polling every POLL seconds) until no other training process holds it.
#
# Env: ADAM_STEPS (5000)  QN_STEPS (20000, a cap: SSBroyden stops on its plateau)
#      DSG_STEPS (300)  DSG_RESAMPLE (100, DSGNAR's redraw period)  ONLY (S default | E | empty = both)  POLL (300)  GPU_FREE_MB (1500)
#      RUN_HOURS (6, per-run backstop)  FORCE_START (1 = skip the wait)
# ---------------------------------------------------------------------------
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # .../Stationary
EVO="$(cd "$HERE/../Evolution_try" && pwd)"
cd "$HERE"
mkdir -p logs "$EVO/logs"
DRIVER="$HERE/logs/batch_sketch.driver.log"
SUMMARY="$HERE/logs/batch_sketch.summary.txt"

ADAM_STEPS="${ADAM_STEPS:-5000}"
QN_STEPS="${QN_STEPS:-20000}"
DSG_STEPS="${DSG_STEPS:-300}"
# The recipe redraws every 500 iterations -- right for SSBroyden's tens of thousands, never
# reached by a 300-iteration DSGNAR phase, which would then fit one frozen sample.  Each redraw
# costs one recompile (~1 min); 100 gives three samples per run.
DSG_RESAMPLE="${DSG_RESAMPLE:-100}"
# Stationary only by default: the Evolution_try runs moved to Mendieta (SLURM, one GPU each, all at
# once: Evolution_try/scripts/mendieta_submit.sh).  ONLY= (empty) still runs both parts here.
ONLY="${ONLY-S}"
POLL="${POLL:-300}"
GPU_FREE_MB="${GPU_FREE_MB:-1500}"
RUN_HOURS="${RUN_HOURS:-6}"
DRY="${DRY:-0}"

SPY="${PY:-$HERE/.venv/bin/python}"
# shellcheck source=/dev/null
source "$EVO/scripts/env.sh"
EPY="$(find_python)" || { echo "FATAL: no interpreter with jax for Evolution_try" >&2; exit 1; }

export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/jaxcache-$USER}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HERE/.mplcache}"
# launched from a notebook, the batch inherits MPLBACKEND=module://matplotlib_inline..., which a
# plain Python rejects at `import matplotlib` (what killed every linref step on 2026-10-15)
export MPLBACKEND=Agg

E1_LABEL="dsgnar_soft_all_nc2201_audit"
E2_LABEL="dsgnar_hard_nc2201_s1500_audit"
# sk16k_*: the problem size of the production runs S2/S4, which fit the hub's 13.5 GiB share.
# The first attempt (sk_*) used run_dsgnar.sh's 27768 / 4548, sized before the four outer
# shells existed: SSBroyden's compiled loss+grad asked for 22.9 GiB, DSGNAR's probe losses
# for 13.4 GiB at every s -- the objective, not the sketch, did not fit.
SP="ib16k"
# the DSGNAR arms: "<name>|<flags>"
ARMS=("${SP}_paperNW_s1100|--dsgnar-sketch 1100 --dsgnar-paper --dsgnar-reweight 0"
      "${SP}_paper_s1100|--dsgnar-sketch 1100 --dsgnar-paper"
      "${SP}_paperNW_s512|--dsgnar-sketch 512 --dsgnar-paper --dsgnar-reweight 0")
S_ADAM="runs/${SP}_adam"
done_s() { [ -f "$HERE/runs/$1/report.json" ] && echo "   [done]"; }
done_e() { [ -f "$EVO/runs/$1/report.md" ] && echo "   [done]"; }

# the problem, shared with mendieta_train.sbatch
source "$HERE/recipe_pq16k.sh"

plan() {
    echo "=== batch_sketch  $(date -u '+%Y-%m-%d %H:%M:%S UTC')  HEAD $(git -C "$HERE" rev-parse --short HEAD 2>/dev/null)"
    echo "    stationary python : $SPY"
    echo "    evolution  python : $EPY"
    if [ "$ONLY" != "E" ]; then
        echo "    S0  $S_ADAM               Adam $ADAM_STEPS steps (no plateau stop)$(done_s ${SP}_adam)"
        echo "    S1  runs/${SP}_ssbroyden         SSBroyden, cap $QN_STEPS iterations$(done_s ${SP}_ssbroyden)"
        for arm in "${ARMS[@]}"; do
            printf '    S*  %-30s DSGNAR %s it., %s%s\n' "runs/${arm%%|*}" "$DSG_STEPS" \
                "${arm#*|}" "$(done_s "${arm%%|*}")"
        done
    fi
    if [ "$ONLY" != "S" ]; then
        for lab in pp_soft_all_warm pp_soft_all_overshoot pp_soft_all_targets \
                   pp_soft_all_reweight pp_soft_all_all pp_hard_all; do
            printf '    P*  Evolution_try/runs/%-28s paper ablation%s\n' "$lab" "$(done_e "$lab")"
        done
        echo "    E1  Evolution_try/runs/$E1_LABEL     soft_all, s = d/3, audits$(done_e "$E1_LABEL")"
        echo "    E2  Evolution_try/runs/$E2_LABEL   hard, s = 1500, audits$(done_e "$E2_LABEL")"
    fi
}

if [ "$DRY" = "1" ]; then plan; exit 0; fi

if [ "${_DETACHED:-0}" != "1" ]; then
    if pgrep -f "_DETACHED_BATCH_SKETCH" > /dev/null; then
        echo "a batch_sketch driver is already running: this one will WAIT for it to finish,"
        echo "then do whatever is left (finished runs are skipped)"
    fi
    _DETACHED=1 nohup setsid bash "$0" _DETACHED_BATCH_SKETCH "$@" >> "$DRIVER" 2>&1 &
    sleep 1
    echo "launched detached; relaunch the same command after any interruption"
    echo "  driver  : $DRIVER"
    echo "  summary : $SUMMARY   (at the end)"
    plan
    exit 0
fi

# queue behind an earlier driver: two drivers would race for the same runs
while pgrep -f "_DETACHED_BATCH_SKETCH" | grep -vx "$$" > /dev/null; do
    [ "${_queued:-0}" = 0 ] && echo "    $(date -u '+%H:%M:%S') another batch_sketch driver is running; queued behind it"
    _queued=1
    sleep "$POLL"
done
plan

# ---- wait for the card ------------------------------------------------------------
card_busy() {
    if pgrep -f "stationary.train|wave_pinn.cli" > /dev/null; then return 0; fi
    if command -v nvidia-smi > /dev/null 2>&1; then
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ -n "${used:-}" ] && [ "$used" -gt "$GPU_FREE_MB" ] && return 0
    fi
    return 1
}
wait_card() {
    [ "${FORCE_START:-0}" = "1" ] && return
    local waited=0
    while card_busy; do
        [ $waited -eq 0 ] && echo "    $(date -u '+%H:%M:%S') the card is busy (another training run, or memory in use); waiting"
        sleep "$POLL"; waited=$((waited + POLL))
    done
    [ $waited -gt 0 ] && echo "    $(date -u '+%H:%M:%S') card free after $((waited / 60)) min"
}

run() {          # run <label> <log> <cmd...>  -- foreground, with a time backstop
    local label="$1" log="$2"; shift 2
    wait_card
    echo "--- $label  start $(date -u '+%H:%M:%S UTC')   log: $log"
    local t0; t0=$(date +%s)
    timeout "$((RUN_HOURS * 3600))" "$@" > "$log" 2>&1
    local rc=$?
    echo "--- $label  exit $rc after $(( ($(date +%s) - t0) / 60 )) min"
    return $rc
}

# ---- S: the Stationary sketch-size comparison --------------------------------------
s_run() {        # s_run <name> <flags...>
    local name="$1"; shift
    # wait FIRST, then decide: a run left going by a stopped driver finishes during the wait, and
    # must then be skipped, not started again on top of its own results
    wait_card
    if [ -f "runs/$name/report.json" ]; then
        echo "--- $name done already, skipping"; return 0
    fi
    run "$name" "logs/$name.log" "$SPY" -m stationary.train --outdir "runs/$name" \
        "${S_RECIPE[@]}" "$@"
}

if [ "$ONLY" != "E" ]; then
    s_run "${SP}_adam" --steps "$ADAM_STEPS" --lbfgs-steps 0 --plateau-min-iters -1
    if [ ! -f "$S_ADAM/params_adam.pkl" ]; then
        echo "!! no $S_ADAM/params_adam.pkl: the warm-up failed, so the Stationary arms cannot start"
    else
        s_run "${SP}_ssbroyden" --steps 0 --init-from "$S_ADAM/params_adam.pkl" \
            --lbfgs-steps "$QN_STEPS" --qn-method ssbroyden
        for arm in "${ARMS[@]}"; do
            # shellcheck disable=SC2086
            s_run "${arm%%|*}" --steps 0 --init-from "$S_ADAM/params_adam.pkl" \
                --lbfgs-steps 30000 --qn-method dsgnar --dsgnar-steps "$DSG_STEPS" \
                --dsgnar-chunk 16 --log-every 10 --resample-every "$DSG_RESAMPLE" ${arm#*|}
        done
        # R: the nonlinear spectral reference, solved once per run directory (same BVP, same
        # inner condition), and each run against it -- a reference that SOLVES the quadrupole
        # problem, unlike --ref-solution (spherical).  Writes <run>/linref_nonlinear.{npz,json}.
        for name in "${SP}_ssbroyden" "${ARMS[@]%%|*}"; do
            [ -f "runs/$name/report.json" ] || continue
            [ -f "runs/$name/linref_nonlinear.json" ] && { echo "--- linref $name done already"; continue; }
            run "linref $name" "logs/${name}.linref.log" "$SPY" -m stationary.linref "runs/$name" \
                --nonlinear --ls 0,2,4,6,8 --n-radial 36 --n-coll 2500 --n-bnd 400 --iters 6
        done
    fi
fi

# ---- E: the Evolution_try stall audits ---------------------------------------------
e_run() {        # e_run <label> <window_ic> <sketch> [extra --set flags...]
    local lab="$1" ic="$2" sk="$3"; shift 3
    wait_card
    if [ -f "$EVO/runs/$lab/report.md" ]; then
        echo "--- $lab done already, skipping"; return 0
    fi
    # an interrupted attempt left its audit records behind; start the file afresh
    rm -f "$EVO/runs/$lab/dsgnar_stop_audits.jsonl"
    ( cd "$EVO" && run "$lab" "$EVO/logs/$lab.log" \
        "$EPY" -m wave_pinn.cli --label "$lab" --outdir "$EVO/runs/$lab" \
        --set T=20 --set windows=10 --set ansatz=t2 \
        --set window_ic="$ic" --set w_ic=100 \
        --set n_coll=2201 --set sampler=random --set resample_every=0 \
        --set "snapshot_times=[0,2,4,6,8,10,12,14,16,18,20]" \
        --set optimizer=dsgnar --set dsgnar_steps=300 --set dsgnar_sketch="$sk" \
        --set dsgnar_delta0=1.0 --set dsgnar_delta_min=1e-14 --set resample_rounds=5 \
        --set dsgnar_audit_stops=true "$@" )
}

PAPER_WARM=(--set window_warm_start=0.01)
PAPER_OVER=(--set dsgnar_reject_overshoot=true)
PAPER_TARG=(--set dsgnar_stage1_ratio=0.075 --set dsgnar_delta_min=1e-8)
PAPER_RW=(--set dsgnar_reweight_alpha=0.05)
PAPER_ALL=("${PAPER_WARM[@]}" "${PAPER_OVER[@]}" "${PAPER_TARG[@]}" "${PAPER_RW[@]}")

if [ "$ONLY" != "S" ]; then
    e_run pp_soft_all_warm      soft_all 0 "${PAPER_WARM[@]}"
    e_run pp_soft_all_overshoot soft_all 0 "${PAPER_OVER[@]}"
    e_run pp_soft_all_targets   soft_all 0 "${PAPER_TARG[@]}"
    e_run pp_soft_all_reweight  soft_all 0 "${PAPER_RW[@]}"
    e_run pp_soft_all_all       soft_all 0 "${PAPER_ALL[@]}"
    e_run pp_hard_all           hard     0 "${PAPER_ALL[@]}"
    e_run "$E1_LABEL" soft_all 0
    e_run "$E2_LABEL" hard 1500
fi

# ---- summary ---------------------------------------------------------------------
{
echo "=== batch_sketch summary  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
"$SPY" - "$HERE" "$SP" <<'PYEOF'
import json, os, sys
root, sp = sys.argv[1], sys.argv[2]
print("\nStationary (same Adam start; loss at the original weights; 'vs reference' = against the")
print("nonlinear spectral reference of linref.py -- NOT the report's rms_d*, which measure the")
print("distance to the SPHERICAL solution)")
print(f"{'run':<22} {'final loss':>12} {'wall (s)':>9}  stopped")
for name in [f"{sp}_adam", f"{sp}_ssbroyden"] + sorted(
        d for d in os.listdir(os.path.join(root, "runs"))
        if d.startswith(sp + "_") and d not in (f"{sp}_adam", f"{sp}_ssbroyden")):
    p = os.path.join(root, "runs", name, "report.json")
    if not os.path.exists(p):
        print(f"{name:<22} (no report)"); continue
    r = json.load(open(p))
    stop = r.get("qn_stopped_at", "")
    lr = os.path.join(root, "runs", name, "linref_nonlinear.json")
    if os.path.exists(lr):
        c = json.load(open(lr))["compare"]
        ref = f"vs reference: rms dh {c['rms_dh']:.2e}, rms dlam {c['rms_dlam']:.2e}"
    else:
        ref = "(no reference comparison)"
    print(f"{name:<22} {r.get('final_loss', float('nan')):>12.4e} {r.get('wall_seconds', 0):>9.0f}  "
          f"{stop}  {ref}")
PYEOF
"$EPY" - "$EVO/runs" <<'PYEOF'
import json, os, sys
root = sys.argv[1]
print("\nEvolution_try paper ablation (space-time rel L2 of the T = 20 chain; window losses)")
for lab in ["dsgnar_soft_all_nc2201", "pp_soft_all_warm", "pp_soft_all_overshoot",
            "pp_soft_all_targets", "pp_soft_all_reweight", "pp_soft_all_all",
            "dsgnar_hard_nc2201", "pp_hard_all"]:
    p = os.path.join(root, lab, "windows.json")
    if not os.path.exists(p):
        print(f"  {lab:<26} (no windows.json)"); continue
    w = json.load(open(p))
    rel = w.get("metrics", {}).get("rel_l2_space_time", float("nan"))
    losses = " ".join(f"{r['final_loss']:.0e}" for r in w.get("windows", []))
    print(f"  {lab:<26} {rel:.3e}   {losses}")
PYEOF
for lab in dsgnar_hard_nc2201_audit "$E1_LABEL" "$E2_LABEL"; do
"$EPY" - "$EVO/runs/$lab" <<'PYEOF'
import json, os, sys
import numpy as np
d = sys.argv[1]
p = os.path.join(d, "dsgnar_stop_audits.jsonl")
print(f"\nEvolution_try stall audits ({p})")
if not os.path.exists(p):
    print("  (none)"); sys.exit(0)
recs = [json.loads(l) for l in open(p)]
print(f"{'win':>3} {'att':>3} {'n':>2} {'loss min..max':>21} {'rank1e-10':>9} {'s':>5} "
      f"{'exact LM best':>13} {'cos sk':>7}")
groups = {}
for r in recs:
    groups.setdefault((r.get("window"), r.get("attempt")), []).append(r)
for (w, a), rs in sorted(groups.items()):
    L = [r["loss"] for r in rs]
    print(f"{w:>3} {a:>3} {len(rs):>2} {min(L):>10.2e}..{max(L):<10.2e} "
          f"{int(np.median([r['spectrum']['rank']['1e-10'] for r in rs])):>9} {rs[0]['s']:>5} "
          f"{np.median([r['best_exact_ratio'] for r in rs]):>13.3f} "
          f"{np.median([r['sketched'][0]['cos_median'] for r in rs]):>+7.3f}")
L = np.log10([r["loss"] for r in recs]); K = [r["spectrum"]["rank"]["1e-10"] for r in recs]
if len(recs) > 2:
    print(f"corr(log10 loss, rank) = {np.corrcoef(L, K)[0, 1]:+.2f} over {len(recs)} stops")
print("'exact LM best' = the lowest loss an exact Levenberg-Marquardt step reaches from the stop,")
print("as a fraction of the stop's loss; 'cos sk' = the sketched step's alignment with it.")
PYEOF
[ -f "$EVO/runs/$lab/report.md" ] && { echo; sed -n '1,40p' "$EVO/runs/$lab/report.md"; }
done
} > "$SUMMARY" 2>&1
echo "=== done $(date -u '+%H:%M:%S UTC'); summary in $SUMMARY"
