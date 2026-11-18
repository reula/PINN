#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Keep trying to start the CCAD JupyterHub server and launch one command there (run on the Mac mini). (2026-11-18)
#
#   (nohup bash jobs/mini/hub_retry.sh CMD_FILE MENDIETA_JOB HUB_LOG [MAX_TRIES] > ~/hub_retry_sbh.log 2>&1 < /dev/null &)
#
# Why: on 2026-11-18 Mendieta's queue was ~7 h long and the hub could not start a server either (its GPU node had
# no free card memory: "CardInsufficientMemory"). Every try asks for the server (SBH/tools/hub_run.py --start, which
# waits 10 min) and, if it comes up, types the one-line command of CMD_FILE (a detached, resumable batch) and then
# becomes the keepalive of that batch (Stationary/hub_keepalive.sh with RELAUNCH = the same command; HUB_LOG is the
# batch log on serafin, its last line "sing1d batch done"). Between tries it sleeps 20 min. It stops without
# launching when the Mendieta job MENDIETA_JOB is no longer PENDING (the same run started there) or after MAX_TRIES
# (default 12, about 6 h). It never cancels anything.
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"          # SBH
PY="$HERE/../Stationary/.venv/bin/python"
CMD="$(cat "$1")"; MJOB=$2; HUB_LOG=$3; MAX=${4:-12}
for n in $(seq 1 "$MAX"); do
    st=$(ssh -o BatchMode=yes -o ConnectTimeout=20 mendieta.ccad.unc.edu.ar "squeue -h -j $MJOB -o %T" 2>/dev/null | tail -1)
    if [ "$st" != "PENDING" ]; then
        echo "$(date -u '+%F %T') Mendieta job $MJOB is '${st:-gone}', not PENDING: no hub launch; stop"; exit 0
    fi
    echo "$(date -u '+%F %T') try $n of $MAX: asking the hub for a server"
    if "$PY" "$HERE/tools/hub_run.py" run "$CMD" --start --wait 45 --close > "$HOME/hub_retry_sbh.last" 2>&1; then
        echo "$(date -u '+%F %T') LAUNCHED on the hub:"; tail -6 "$HOME/hub_retry_sbh.last"
        echo "$(date -u '+%F %T') now the keepalive of that batch"
        RELAUNCH="$CMD" exec bash "$HERE/../Stationary/hub_keepalive.sh" "$HUB_LOG" 'sing1d batch done'
    fi
    echo "$(date -u '+%F %T') no server: $(tail -1 "$HOME/hub_retry_sbh.last" | cut -c1-200)"
    sleep 1200
done
echo "$(date -u '+%F %T') gave up after $MAX tries"
