#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Send run news to the user's phone (ntfy) from the Mac mini; detached, survives the Claude session. (2026-11-19)
#
#   (nohup bash jobs/mini/ntfy_runs.sh JOBID:NAME [JOBID:NAME ...] > ~/ntfy_runs.log 2>&1 < /dev/null &)
#
# Every 5 min it looks at each Mendieta job: when it ends it sends its state and last loss line (or the error);
# while it runs it sends one line every 8 windows (held loss, s per window). Results only, never credentials.
# Stops when every job has ended.
# ---------------------------------------------------------------------------
set -uo pipefail
ntfy() { curl -s -H "Title: $1" -d "$2" https://ntfy.sh/pinn-reula-f0798d1ec301 > /dev/null || true; }
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=ERROR mendieta.ccad.unc.edu.ar)
DONE=(); LASTW=()          # indexed by the numeric job id (macOS bash 3.2 has no associative arrays)
while true; do
    left=0
    for v in "$@"; do
        j=${v%%:*}; n=${v#*:}
        [ -n "${DONE[$j]:-}" ] && continue
        left=1
        st=$("${SSH[@]}" "squeue -h -j $j -o %T 2>/dev/null" | tail -1)
        f="~/Julia/PINN/SBH/runs/slurm_${n}_${j}.out"
        if [ -z "$st" ]; then
            msg=$("${SSH[@]}" "grep -h 'RESOURCE\|Error' $f | tail -1 | cut -c1-200; echo \"windows: \$(ls ~/Julia/PINN/SBH/runs/$n 2>/dev/null | grep -c window_)\"; grep -h '  it ' $f | tail -1 | cut -c1-160")
            ntfy "$n ended" "$msg"; DONE[$j]=1
        elif [ "$st" = RUNNING ]; then
            w=$("${SSH[@]}" "ls ~/Julia/PINN/SBH/runs/$n 2>/dev/null | grep -c window_")
            if [ "${w:-0}" -gt 0 ] && [ $((w % 8)) -eq 0 ] && [ "${LASTW[$j]:-}" != "$w" ]; then
                msg=$("${SSH[@]}" "grep -h '  it ' $f | tail -1 | cut -c1-160")
                ntfy "$n: $w windows" "$msg"; LASTW[$j]=$w
            fi
        fi
    done
    [ $left -eq 0 ] && exit 0
    sleep 300
done
