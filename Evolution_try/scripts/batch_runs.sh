#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The batch: every optimiser against every hand-over, at two collocation sizes,
# strictly one run at a time.
#
#   bash scripts/batch_runs.sh --dry-run     # print the plan and the estimates
#   bash scripts/batch_runs.sh               # run it (resumable, see below)
#   WITH_BROYDEN=1 bash scripts/batch_runs.sh
#
# Why sequential: this hub shares one A30 through HAMI, and two of these runs at
# once is a measured OOM (`Device 0 OOM 19334492688 / 19327352832`) rather than a
# slowdown.  Nothing here runs in parallel on purpose.
#
# Resumable: a run whose `runs/<label>/report.md` exists is skipped, so the batch
# can be stopped and restarted (or resumed after a disconnection) without redoing
# hours of work.  Delete a directory to force that one to run again -- or let
# `overwrite` archive it, which is the default inside the package.
#
# Everything runs in float64, and that is checked before the first run rather than
# assumed: `configure_jax` is what turns x64 on, and without it every dtype
# silently becomes float32 and every loss floor rises by four orders of magnitude.
#
# Env overrides: PY, WITH_BROYDEN, DSGNAR_STEPS, DSGNAR_ROUNDS, QN_STEPS, NCOLLS
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
source "$HERE/scripts/env.sh"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1
WITH_BROYDEN="${WITH_BROYDEN:-0}"
DSGNAR_STEPS="${DSGNAR_STEPS:-300}"
DSGNAR_ROUNDS="${DSGNAR_ROUNDS:-5}"
QN_STEPS="${QN_STEPS:-1500}"
QN_ROUNDS="${QN_ROUNDS:-2}"
NCOLLS="${NCOLLS:-2201 8192}"
SNAPSHOT="[0,2,4,6,8,10,12,14,16,18,20]"

PY="$(find_python)" || { echo "FATAL: no interpreter with jax; set PY=" >&2; exit 1; }

echo "checkout : $HERE"
echo "python   : $PY"
"$PY" -c 'import jax; print("jax      :", jax.__version__, jax.devices())' || true

# ---- gate 1: float64 is actually on ---------------------------------------
"$PY" - <<'EOF' || { echo "FATAL: float64 is not enabled; refusing to launch." >&2; exit 1; }
from wave_pinn.train import configure_jax
from wave_pinn.config import Config
import jax.numpy as jnp
cfg = Config()
assert cfg.precision == "float64", f"precision defaults to {cfg.precision!r}"
configure_jax(cfg)                      # this is what sets jax_enable_x64
d = jnp.zeros(1).dtype
assert d == jnp.float64, f"x64 did NOT take effect: jnp.zeros(1).dtype == {d}"
from wave_pinn.model import init_params
import jax
p = init_params(cfg, jax.random.PRNGKey(0))
flat, _ = jax.flatten_util.ravel_pytree(p)
assert flat.dtype == jnp.float64, f"parameters are {flat.dtype}"
print(f"F64      : ON   (jax_enable_x64, parameters {flat.dtype}, eps {jnp.finfo(jnp.float64).eps:.3e})")
EOF

# ---- gate 2: jaxopt, only if the Broyden column is wanted -----------------
HAVE_JAXOPT=0
"$PY" -c 'import jaxopt' 2>/dev/null && HAVE_JAXOPT=1
if [ "$WITH_BROYDEN" = "1" ] && [ "$HAVE_JAXOPT" = "0" ]; then
    echo
    echo "WARNING: WITH_BROYDEN=1 but jaxopt is not installed in $PY."
    echo "         The Broyden column will be skipped.  To add it:"
    echo "           $PY -m pip install jaxopt==0.8.5"
    echo "         (it is a root finder that, measured, diverges on this problem;"
    echo "          see the note in wave_pinn/optim/jaxopt_broyden.py)"
    echo
fi

# ---- the matrix -----------------------------------------------------------
# optimizer  hand-over  n_coll
MATRIX=()
for nc in $NCOLLS; do
    for ic in hard soft soft_all; do
        MATRIX+=("dsgnar $ic $nc")
        MATRIX+=("ssbroyden $ic $nc")
    done
done
[ "$WITH_BROYDEN" = "1" ] && [ "$HAVE_JAXOPT" = "1" ] && for nc in $NCOLLS; do
    for ic in hard soft soft_all; do
        MATRIX+=("jaxopt_broyden $ic $nc")
    done
done

flags_for() {
    local opt="$1"
    case "$opt" in
        dsgnar)    echo "--set optimizer=dsgnar --set dsgnar_steps=$DSGNAR_STEPS \
--set dsgnar_sketch=0 --set dsgnar_delta0=1.0 --set dsgnar_delta_min=1e-14 \
--set resample_rounds=$DSGNAR_ROUNDS" ;;
        ssbroyden) # Two rounds, not five: each SSBroyden round runs to its own plateau,
                   # thousands of iterations, so five per window is not a batch.  But not
                   # ONE either -- with a single round it never redraws, and SSBroyden is the
                   # method measured to fit a frozen sample rather than estimate the residual
                   # (train 9.6e-12 against an independent sample's 3.0e-05 on T=20, a factor
                   # of 3e6).  Two rounds buys the "loss on next sample" column, which is what
                   # says how much of its answer was sample-fitting, at the same total budget.
                   echo "--set optimizer=ssbroyden --set qn_steps=$QN_STEPS --set qn_block=250 \
--set qn_gtol=1e-12 --set resample_rounds=$QN_ROUNDS" ;;
        jaxopt_broyden) echo "--set optimizer=jaxopt_broyden --set broyden_steps=60 \
--set resample_rounds=1" ;;
    esac
}

label_for() { echo "$1_$2_nc$3"; }

# ---- plan -----------------------------------------------------------------
echo
echo "=== plan: ${#MATRIX[@]} runs, sequential ==="
printf '%-6s %-34s %-9s %s\n' "#" "label" "n_coll" "optimiser / hand-over"
i=0
for row in "${MATRIX[@]}"; do
    read -r opt ic nc <<<"$row"
    i=$((i + 1))
    lab="$(label_for "$opt" "$ic" "$nc")"
    state="to run"
    [ -f "$HERE/runs/$lab/report.md" ] && state="done, will skip"
    printf '%-6s %-34s %-9s %-24s %s\n' "$i" "$lab" "$nc" "$opt/$ic" "$state"
done

# Rough, and now scaled for ROUNDS.  The two finished T=20 windowed runs took
# 1043 s and 1384 s for dsgnar at n_coll=2201, but those were ONE round per window
# (~185 iterations, stopping on the radius criterion).  Five rounds per window is
# roughly 3x the iterations, so ~4400 s.  Only the first run can settle it: it
# reports its own wall time, and everything after can be read against that.
PER_RUN_2201_DSGNAR=4400
echo
echo "rough wall time.  UNRELIABLE until run 1 reports: it is scaled from the two"
echo "finished one-round runs, and the rounds above change the iteration count.  The"
echo "JVP cost scales with n_coll; SSBroyden needs far more iterations than DSGNAR.":
tot=0
for row in "${MATRIX[@]}"; do
    read -r opt ic nc <<<"$row"
    [ -f "$HERE/runs/$(label_for "$opt" "$ic" "$nc")/report.md" ] && continue
    base=$PER_RUN_2201_DSGNAR
    # SSBroyden: n*log-ish per step on the dense inverse Hessian plus a line search,
    # against DSGNAR's sketched JVP -- measured on CPU as 0.44 s/iteration until
    # convergence, but it needs thousands of iterations where DSGNAR needs hundreds.
    if [ "$opt" = "ssbroyden" ]; then base=$((base * 2 / 3)); fi
    if [ "$opt" = "jaxopt_broyden" ]; then base=300; fi
    est=$(( base * nc / 2201 ))
    tot=$((tot + est))
done
printf '  remaining: about %d h %02d min\n' $((tot / 3600)) $(((tot % 3600) / 60))

if [ "$DRY" = "1" ]; then
    echo
    echo "--dry-run: nothing launched."
    exit 0
fi

# ---- run ------------------------------------------------------------------
mkdir -p logs
: > logs/batch.log
started=$(date +%s)
i=0
for row in "${MATRIX[@]}"; do
    read -r opt ic nc <<<"$row"
    i=$((i + 1))
    lab="$(label_for "$opt" "$ic" "$nc")"
    log="$HERE/logs/$lab.log"
    if [ -f "$HERE/runs/$lab/report.md" ]; then
        echo "[$i/${#MATRIX[@]}] $lab -- already done, skipping"
        continue
    fi
    t0=$(date +%s)
    echo "[$i/${#MATRIX[@]}] $lab -- start $(date '+%H:%M:%S')  ($opt / $ic / n_coll=$nc)"
    # shellcheck disable=SC2046
    "$PY" -m wave_pinn.cli \
        --label "$lab" --outdir "$HERE/runs/$lab" \
        --set T=20 --set windows=10 --set ansatz=t2 \
        --set window_ic="$ic" --set w_ic=100 \
        --set n_coll="$nc" --set sampler=random --set resample_every=0 \
        --set "snapshot_times=$SNAPSHOT" \
        $(flags_for "$opt") \
        >"$log" 2>&1
    rc=$?
    took=$(( $(date +%s) - t0 ))
    elapsed=$(( $(date +%s) - started ))
    if [ $rc -eq 0 ]; then
        echo "[$i/${#MATRIX[@]}] $lab done in $((took / 60)) min  (batch so far: $((elapsed / 3600)) h $(((elapsed % 3600) / 60)) min)"
    else
        echo "[$i/${#MATRIX[@]}] $lab FAILED (exit $rc) after $((took / 60)) min -- log: $log"
    fi
    tail -3 "$log"
    echo "$lab exit=$rc seconds=$took" >> logs/batch.log
done

echo
echo
echo "=== batch finished in $(( ($(date +%s) - started) / 60 )) min ==="
grep -c "exit=0" logs/batch.log | xargs -I{} echo "{} runs succeeded"
grep "exit=[^0]" logs/batch.log && echo "^ failures above; their logs are in logs/" || true
echo "now:  $PY scripts/report_batch.py"
