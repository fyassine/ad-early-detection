#!/usr/bin/env bash
# =============================================================================
# monitor_postproc_live.sh
#
# Real-time dashboard for Fritz continuous postprocessing jobs:
#   - Tracks Elapsed time from log start banner
#   - Shows OASIS3 / ADNI / Total progress & remaining counts
#   - Calculates live ETA based on throughput
#   - Displays active worker processes (PID, runtime, CPU, subject/session)
#
# Usage:
#   bash monitor_postproc_live.sh             # Run once
#   bash monitor_postproc_live.sh --watch 5   # Live update every 5s
#   watch -n 5 bash monitor_postproc_live.sh  # Using standard watch
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/DATA/PREPROCESSING/logs/postprocess_local"

# Auto-select latest postprocess log unless specified
LOG_FILE="${1:-}"
if [[ -z "$LOG_FILE" || "$LOG_FILE" == "--watch" || "$LOG_FILE" == "-w" ]]; then
    LOG_FILE="$(ls -t "${LOG_DIR}"/postprocess_local_*.log 2>/dev/null | head -1 || true)"
fi

WATCH_INTERVAL=0
if [[ "${1:-}" == "--watch" || "${1:-}" == "-w" ]]; then
    WATCH_INTERVAL="${2:-5}"
elif [[ "${2:-}" == "--watch" || "${2:-}" == "-w" ]]; then
    WATCH_INTERVAL="${3:-5}"
fi

render_dashboard() {
    if [[ ! -f "$LOG_FILE" ]]; then
        echo "No postprocessing log found under ${LOG_DIR}."
        return
    fi

    # Read start timestamp from first line [YYYY-MM-DD HH:MM:SS]
    local first_line start_str start_ts now_ts elapsed
    first_line="$(head -n 1 "$LOG_FILE" 2>/dev/null || true)"
    start_str="$(echo "$first_line" | grep -oE "^\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\]" | tr -d "[]" || true)"
    
    if [[ -n "$start_str" ]]; then
        start_ts="$(date -d "$start_str" +%s 2>/dev/null || date +%s)"
    else
        start_ts="$(date +%s)"
    fi
    
    now_ts="$(date +%s)"
    elapsed=$(( now_ts - start_ts ))
    (( elapsed < 0 )) && elapsed=0

    local oasis_total=142
    local adni_total=272
    local total_all=$(( oasis_total + adni_total ))

    local oasis_done adni_done done_n eta
    oasis_done=$(grep -c "\[sub-OAS.*done —" "$LOG_FILE" 2>/dev/null || echo 0)
    adni_done=$(grep -c "\[sub-ADNI.*done —" "$LOG_FILE" 2>/dev/null || echo 0)
    done_n=$(( oasis_done + adni_done ))

    eta=0
    if [ "$done_n" -gt 0 ] && [ "$total_all" -gt "$done_n" ]; then
        eta=$(( elapsed * (total_all - done_n) / done_n ))
    fi

    local elapsed_fmt eta_fmt
    elapsed_fmt="$(date -u -d @"$elapsed" +%H:%M:%S 2>/dev/null || echo "00:00:00")"
    eta_fmt="$(date -u -d @"$eta" +%H:%M:%S 2>/dev/null || echo "00:00:00")"

    printf "=== Postprocessing Progress ===\n"
    printf "Log     : %s\n" "${LOG_FILE##*/}"
    printf "Elapsed : %s\n" "$elapsed_fmt"
    printf "OASIS3  : %3d / %d (%2d%%)\n" "$oasis_done" "$oasis_total" "$(( oasis_done * 100 / oasis_total ))"
    printf "ADNI    : %3d / %d (%2d%%)\n" "$adni_done" "$adni_total" "$(( adni_done * 100 / adni_total ))"
    printf "Total   : %3d / %d (%2d%%)\n" "$done_n" "$total_all" "$(( done_n * 100 / total_all ))"
    printf "ETA     : %s\n\n" "$eta_fmt"

    printf -- "--- Active Workers ---\n"
    ps -wwo pid,etime,%cpu,args -C python 2>/dev/null | grep -E "final_reorient\.py|qc_motion_gate\.py" | grep -v grep | awk '
      BEGIN { printf "%-9s %-9s %-7s %-18s %s\n", "PID", "ELAPSED", "CPU", "SCRIPT", "SUBJECT / SESSION" }
      {
        pid=$1; etime=$2; cpu=$3; script=""; subj="";
        for (i=4; i<=NF; i++) {
          if ($i ~ /\.py$/) script=$i;
          if ($i ~ /sub-/) subj=$i;
        }
        sub(/.*\//, "", script);
        sub(/.*\//, "", subj);
        sub(/_task-.*/, "", subj);
        sub(/\.nii\.gz.*/, "", subj);
        printf "%-9s %-9s %-7s %-18s %s\n", pid, etime, cpu"%", script, subj;
      }' || true
}

if [[ "$WATCH_INTERVAL" -gt 0 ]]; then
    trap 'echo ""; exit 0' INT TERM
    while true; do
        clear
        render_dashboard
        sleep "$WATCH_INTERVAL"
    done
else
    render_dashboard
fi
