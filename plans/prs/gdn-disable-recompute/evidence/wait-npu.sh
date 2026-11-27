#!/usr/bin/env bash
# Print the ids of N idle Ascend NPUs, comma-separated, for ASCEND_RT_VISIBLE_DEVICES.
# Idle = no process listed on the card and AICore utilisation <= MAX_UTIL.
# HBM usage is not a criterion: an idle 910B3 already reports about 3.4 GB.
# Usage: wait-npu.sh [N=1]      env: MAX_UTIL=5  SLEEP_SEC=600  ONCE=1 (fail instead of waiting)
set -euo pipefail
need="${1:-1}"
max_util="${MAX_UTIL:-5}"
sleep_sec="${SLEEP_SEC:-600}"

idle_ids() {
    npu-smi info | awk -F'|' -v max="$max_util" '
        /Process id/ { in_procs = 1; next }
        !in_procs && $2 ~ /^ *[0-9]+ +[0-9][0-9A-Za-z]+ *$/ { split($2, a, " "); id = a[1]; want_chip = 1; next }
        !in_procs && want_chip { split($4, u, " "); util[id] = u[1]; hbm[id] = u[5]; want_chip = 0; next }
        in_procs && $3 ~ /^ *[0-9]+ *$/ { split($2, a, " "); busy[a[1]] = 1 }
        END { for (i in util) if (!(i in busy) && util[i] + 0 <= max) print hbm[i], i }
    ' | sort -n | awk '{ print $2 }'
}

while true; do
    mapfile -t ids < <(idle_ids)
    if (( ${#ids[@]} >= need )); then
        (IFS=,; echo "${ids[*]:0:need}")
        exit 0
    fi
    echo "$(date '+%F %T') only ${#ids[@]} idle NPU(s), need $need" >&2
    [[ -n "${ONCE:-}" ]] && exit 1
    sleep "$sleep_sec"
done
