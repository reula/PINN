#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Keep the CCAD JupyterHub server alive while a detached batch runs on it (run on the Mac mini).
#
# The hub culls a server after about one hour without user activity, and a running GPU process does
# NOT count as activity.  hub_recipe.sh died that way on 2026-11-06: the last terminal command was at
# 16:31 UTC, and the container was gone at about 17:32 UTC, partway through its first SSBroyden stage.
# This loop types a no-op into a fresh hub terminal (SBH/tools/hub_run.py) every INTERVAL seconds,
# which is user activity.  It stops by itself when LOG (on serafin, the same filesystem as the hub)
# contains DONE, and it stops if the server has gone (it never starts one).
#
#   (nohup bash Stationary/hub_keepalive.sh [LOG] [DONE] > ~/hub_keepalive.log 2>&1 < /dev/null &)   # macOS: no setsid
#   defaults: LOG = Julia/PINN/Stationary/logs/hub_recipe.log, DONE = 'hub_recipe .* finished'
#   RELAUNCH='<hub command>': when the server has gone, start a new one and send this command (e.g.
#   "cd ~/serafin/Julia/PINN/Stationary && bash Paper/hub_recipe.sh F"; the batch skips finished runs and
#   resumes the one in progress from ckpt.pkl), at most MAX_RELAUNCH times (default 3).  The hub also
#   stops a server ~24 h after it started (HUB.md 19).  Without RELAUNCH it notifies and stops.
# ---------------------------------------------------------------------------
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"      # <repo>
LOG=${1:-Julia/PINN/Stationary/logs/hub_recipe.log}
DONE=${2:-hub_recipe .* finished}
INTERVAL=${INTERVAL:-1200}
PY="$ROOT/Stationary/.venv/bin/python"                         # has websocket-client
RELAUNCH=${RELAUNCH:-}
MAX_RELAUNCH=${MAX_RELAUNCH:-3}
nrel=0
ntfy() { curl -s -H "Title: $1" -d "$2" https://ntfy.sh/pinn-reula-f0798d1ec301 > /dev/null || true; }
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -i "$HOME/.ssh/id_ccad" reula@serafin.ccad.unc.edu.ar)
while true; do
    if "${SSH[@]}" "grep -qE '$DONE' $LOG" 2>/dev/null; then
        echo "$(date -u '+%F %T') batch finished; keepalive stops"; exit 0
    fi
    if ! "$PY" "$ROOT/SBH/tools/hub_run.py" status 2>&1 | grep -q 'ready=True'; then
        if [ -n "$RELAUNCH" ] && [ "$nrel" -lt "$MAX_RELAUNCH" ]; then
            nrel=$((nrel + 1))
            echo "$(date -u '+%F %T') server gone; starting it and relaunching ($nrel of $MAX_RELAUNCH)"
            if "$PY" "$ROOT/SBH/tools/hub_run.py" run "$RELAUNCH" --start --wait 20 --close; then
                ntfy "hub relaunched" "server had stopped; relaunched ($nrel/$MAX_RELAUNCH): $RELAUNCH"
            else
                ntfy "hub relaunch failed" "could not restart the server for $LOG"; exit 1
            fi
            sleep "$INTERVAL"; continue
        fi
        echo "$(date -u '+%F %T') server not running; keepalive stops (relaunch the batch: it resumes)"
        ntfy "hub server gone" "the hub server stopped while $LOG was running"
        exit 1
    fi
    "$PY" "$ROOT/SBH/tools/hub_run.py" run "true" --wait 3 --close > /dev/null 2>&1 \
        && echo "$(date -u '+%F %T') ping" || echo "$(date -u '+%F %T') ping failed"
    sleep "$INTERVAL"
done
