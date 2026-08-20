#!/usr/bin/env bash
# =============================================================================
# monitor_flatten_progress.sh
#
# Read-only oversight for the two Fritz-side flatten jobs:
#   - postprocess_local.sh --flatten-only  (denoised -> __fmri_wholebrain_sch200_flat__)
#   - flatten_fmriprep.sh                  (raw preproc -> __fmriprep_wholebrain_flat__)
# plus the upstream CORE -> Fritz copy they both depend on:
#   - pull_derivatives_from_core.sh (rsyncs derivatives/{fmriprep,postprocessed}
#     for OASIS3/ADNI from CORE onto Fritz, in that fixed 4-leg order). A
#     flatten target sitting short is often just this pull not having reached
#     that subject yet, not a flatten-job problem — reported first, above the
#     4 flatten targets, so that's obvious at a glance instead of requiring a
#     separate manual check.
#
# Reports, per cohort x stage (4 targets: ADNI/OASIS3 x postprocessed/fmriprep):
#   subjects done / target, percent, elapsed since the underlying job started
#   (read from that job's own log banner, not from when this monitor started —
#   so the "time passed" figure is correct even if the monitor is restarted),
#   average and recent throughput, and an ETA. Also flags whether the
#   underlying job process is still alive, so a stalled/crashed job (process
#   gone, target not reached) is visibly different from one still running.
#
# NEVER writes to, moves, or deletes anything under DATA/<COHORT>/** — it only
# counts sub-* directories and reads log files. Safe to run alongside the
# jobs it watches, safe to Ctrl+C and restart anytime.
#
# Ticks every --interval seconds, appending one snapshot to its own log (tee'd
# to stdout, same convention as the jobs it watches) so `tail -f` shows a
# running history. Exits on its own once every target has reached 100% and no
# matching job process remains — no need to remember to kill it. --once
# prints a single snapshot and exits immediately (no loop, no log-tailing use
# case — for a quick manual check).
#
# Usage:
#   bash monitor_flatten_progress.sh [--interval SECONDS] [--once]
#        [--postproc-log PATH] [--fmriprep-log PATH] [--pull-log PATH]
#
# By default, --postproc-log / --fmriprep-log / --pull-log auto-select the
# most recently modified log under logs/postprocess_local/, logs/flatten_fmriprep/,
# and logs/pull_derivatives/ respectively — override if you want to point at
# an older run's log specifically.
# =============================================================================

set -euo pipefail

# ─── Colors ─────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    C_CYAN=$'\033[36m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_BOLD=$'\033[1m'; C_RESET=$'\033[0m'
else
    C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_BOLD=""; C_RESET=""
fi
ts() { date '+%Y-%m-%d %H:%M:%S'; }
info()    { echo "${C_CYAN}[$(ts)] $*${C_RESET}"; }
success() { echo "${C_GREEN}[$(ts)] $*${C_RESET}"; }
warn()    { echo "${C_YELLOW}[$(ts)] WARN: $*${C_RESET}"; }
error()   { echo "${C_RED}[$(ts)] ERROR: $*${C_RESET}" >&2; }
die()     { error "$*"; exit 1; }

# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

LOG_DIR="${REPO_ROOT}/DATA/PREPROCESSING/logs/monitor_flatten_progress"
POSTPROC_LOG_DIR="${REPO_ROOT}/DATA/PREPROCESSING/logs/postprocess_local"
FMRIPREP_LOG_DIR="${REPO_ROOT}/DATA/PREPROCESSING/logs/flatten_fmriprep"
PULL_LOG_DIR="${REPO_ROOT}/DATA/PREPROCESSING/logs/pull_derivatives"

# ─── Parse arguments ────────────────────────────────────────────────────────
INTERVAL=120
ONCE=false
POSTPROC_LOG=""
FMRIPREP_LOG=""
PULL_LOG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval)     INTERVAL="$2"; shift 2 ;;
        --once)         ONCE=true; shift ;;
        --postproc-log) POSTPROC_LOG="$2"; shift 2 ;;
        --fmriprep-log) FMRIPREP_LOG="$2"; shift 2 ;;
        --pull-log)     PULL_LOG="$2"; shift 2 ;;
        -h|--help)      sed -n '2,38p' "$0"; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

[[ "$INTERVAL" =~ ^[0-9]+$ && "$INTERVAL" -ge 5 ]] || die "--interval must be an integer >= 5 (seconds)."

latest_log() {
    local dir="$1"
    [[ -d "$dir" ]] || { echo ""; return; }
    ls -t "$dir"/*.log 2>/dev/null | head -1
}
[[ -n "$POSTPROC_LOG" ]] || POSTPROC_LOG="$(latest_log "$POSTPROC_LOG_DIR")"
[[ -n "$FMRIPREP_LOG" ]] || FMRIPREP_LOG="$(latest_log "$FMRIPREP_LOG_DIR")"
[[ -n "$PULL_LOG" ]] || PULL_LOG="$(latest_log "$PULL_LOG_DIR")"

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/monitor_flatten_progress_$(date +%Y%m%d_%H%M%S).log"
if ! $ONCE; then
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

# ─── The 4 cohort x stage targets, single source of truth for both the report
# loop and the exit condition (was previously duplicated — a real bug: the
# exit condition didn't know a target's job process had already exited, so it
# looped forever once a target settled short of its full source count due to
# legitimate per-subject exclusions, e.g. missing confounds/no denoise output).
TARGET_LABEL=(
    "ADNI postprocessed -> flat"
    "OASIS3 postprocessed -> flat"
    "ADNI fmriprep -> flat"
    "OASIS3 fmriprep -> flat"
)
TARGET_SRC=(
    "${REPO_ROOT}/DATA/ADNI/derivatives/postprocessed"
    "${REPO_ROOT}/DATA/OASIS3/derivatives/postprocessed"
    "${REPO_ROOT}/DATA/ADNI/derivatives/fmriprep"
    "${REPO_ROOT}/DATA/OASIS3/derivatives/fmriprep"
)
TARGET_FLAT=(
    "${REPO_ROOT}/DATA/ADNI/__fmri_wholebrain_sch200_flat__/fmri"
    "${REPO_ROOT}/DATA/OASIS3/__fmri_wholebrain_sch200_flat__/fmri"
    "${REPO_ROOT}/DATA/ADNI/__fmriprep_wholebrain_flat__/fmri"
    "${REPO_ROOT}/DATA/OASIS3/__fmriprep_wholebrain_flat__/fmri"
)
TARGET_LOG=("$POSTPROC_LOG" "$POSTPROC_LOG" "$FMRIPREP_LOG" "$FMRIPREP_LOG")
TARGET_PATTERN=(
    "postprocess_local.sh --dataset adni"
    "postprocess_local.sh --dataset oasis3"
    "flatten_fmriprep.sh"
    "flatten_fmriprep.sh"
)

# ─── Helpers ────────────────────────────────────────────────────────────────

# Count immediate sub-* directories under $1 (0 if the dir doesn't exist yet).
count_subjects() {
    local dir="$1"
    [[ -d "$dir" ]] || { echo 0; return; }
    find "$dir" -mindepth 1 -maxdepth 1 -type d -name 'sub-*' 2>/dev/null | wc -l
}

# Count sub-* directories under $1 that contain at least one file (recursively)
# — i.e. actually hold flattened output, not just a directory stub created
# for a subject whose denoise step passed motion QC but produced nothing to
# land (postprocess_local.sh / flatten_fmriprep.sh create the dir either way,
# so a bare `count_subjects` on the flat product overstates real completion).
# Prints "<nonempty> <empty>" on one line — a bash function invoked via a
# `$(...)` command substitution runs in a subshell, so a global-variable
# side channel for the second value would silently not propagate to the
# caller; printing both and having the caller `read` them avoids that trap.
count_nonempty_subjects() {
    local dir="$1"
    [[ -d "$dir" ]] || { echo "0 0"; return; }
    local d nonempty=0 empty=0
    for d in "$dir"/sub-*/; do
        [[ -d "$d" ]] || continue
        if [[ -n "$(find "$d" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
            nonempty=$(( nonempty + 1 ))
        else
            empty=$(( empty + 1 ))
        fi
    done
    echo "$nonempty $empty"
}

# Epoch seconds of a job log's first line (the "[TIMESTAMP] ====" start banner).
# Empty string if the log is missing/unparseable.
job_start_epoch() {
    local logfile="$1"
    [[ -n "$logfile" && -f "$logfile" ]] || { echo ""; return; }
    local line ts_str
    line="$(head -1 "$logfile" 2>/dev/null || true)"
    ts_str="$(sed -n 's/^\[\([0-9-]\{10\} [0-9:]\{8\}\)\].*/\1/p' <<< "$line")"
    [[ -n "$ts_str" ]] || { echo ""; return; }
    date -d "$ts_str" +%s 2>/dev/null || echo ""
}

human_duration() {
    local s="$1"
    (( s < 0 )) && s=0
    local h=$(( s / 3600 )) m=$(( (s % 3600) / 60 )) sec=$(( s % 60 ))
    if (( h > 0 )); then echo "${h}h${m}m"
    elif (( m > 0 )); then echo "${m}m${sec}s"
    else echo "${sec}s"
    fi
}

pid_alive() {
    pgrep -f "$1" > /dev/null 2>&1
}

# ─── CORE -> Fritz pull status ──────────────────────────────────────────────
# pull_derivatives_from_core.sh rsyncs 4 legs in this fixed order (see its own
# section headers, e.g. "Pulling OASIS3/fmriprep <- CORE"). A flatten target
# reported short below is frequently just its leg here not having reached
# that subject yet, not a problem with the flatten job itself.
PULL_PROC_PATTERN="pull_derivatives_from_core.sh"
PULL_LEG_LABEL=("OASIS3/fmriprep" "OASIS3/postprocessed" "ADNI/fmriprep" "ADNI/postprocessed")
PULL_LEG_DEST=(
    "${REPO_ROOT}/DATA/OASIS3/derivatives/fmriprep"
    "${REPO_ROOT}/DATA/OASIS3/derivatives/postprocessed"
    "${REPO_ROOT}/DATA/ADNI/derivatives/fmriprep"
    "${REPO_ROOT}/DATA/ADNI/derivatives/postprocessed"
)

report_pull_status() {
    local alive="unknown"
    if pid_alive "$PULL_PROC_PATTERN"; then alive="RUNNING"; else alive="NOT RUNNING"; fi

    if [[ -z "$PULL_LOG" ]]; then
        warn "CORE -> Fritz pull: no log found under ${PULL_LOG_DIR} — cannot tell whether derivatives are fully landed."
        return
    fi

    info "${C_BOLD}CORE -> Fritz pull${C_RESET} (${PULL_LOG##*/}): process ${alive}"
    local i n=${#PULL_LEG_LABEL[@]} n_complete=0
    for (( i = 0; i < n; i++ )); do
        local leg="${PULL_LEG_LABEL[$i]}" dest="${PULL_LEG_DEST[$i]}"
        local complete_line remote_line
        complete_line="$(grep -F "${leg}: pull complete." "$PULL_LOG" 2>/dev/null | tail -1)"
        if [[ -n "$complete_line" ]]; then
            local leg_ts
            leg_ts="$(sed -n 's/^\[\([0-9-]\{10\} [0-9:]\{8\}\)\].*/\1/p' <<< "$complete_line")"
            info "    ${leg}: done (${leg_ts})"
            n_complete=$(( n_complete + 1 ))
            continue
        fi
        remote_line="$(grep -F "${leg}: remote size" "$PULL_LOG" 2>/dev/null | tail -1)"
        if [[ -n "$remote_line" ]]; then
            local remote_size landed
            remote_size="$(sed -n 's/.*remote size = \([^,]*\),.*/\1/p' <<< "$remote_line")"
            landed="$(du -sh "$dest" 2>/dev/null | cut -f1)"
            if [[ "$alive" == "RUNNING" ]]; then
                info "    ${leg}: in progress — ~${landed:-?} landed of ${remote_size:-?} remote"
            else
                warn "    ${leg}: incomplete (~${landed:-?} of ${remote_size:-?} remote) and process NOT running — check ${PULL_LOG} for errors before rerunning."
            fi
        else
            info "    ${leg}: not started yet"
        fi
    done
    if [[ "$alive" == "NOT RUNNING" && "$n_complete" -lt "$n" ]]; then
        warn "    pull process is NOT running but only ${n_complete}/${n} legs completed — flatten targets below may be sourced from partial data. Rerun pull_derivatives_from_core.sh."
    fi
}

# Previous-tick state, keyed by target index, for the "recent rate" trend.
declare -A PREV_DONE=()
declare -A PREV_EPOCH=()

# Set by report_target on each call: "1" if this target needs no further
# watching (100% reached, OR its job process has already exited — a one-shot
# job that exited without hitting the full source count has permanently
# settled short, e.g. subjects excluded for missing confounds/no denoise
# output; it will not make further progress until someone reruns it).
TARGET_SETTLED=0

# ─── One target's report ────────────────────────────────────────────────────
# args: index label cohort source_dir flat_dir job_log proc_pattern
report_target() {
    local idx="$1" label="$2" source_dir="$3" flat_dir="$4" job_log="$5" proc_pattern="$6"
    TARGET_SETTLED=0

    local target raw_dir_n done_n empty_n no_dir_n remaining pct now
    target="$(count_subjects "$source_dir")"
    raw_dir_n="$(count_subjects "$flat_dir")"
    read -r done_n empty_n < <(count_nonempty_subjects "$flat_dir")
    no_dir_n=$(( target - raw_dir_n ))
    now="$(date +%s)"

    if (( target == 0 )); then
        warn "${label}: no source subjects found at ${source_dir} (nothing to track yet)."
        TARGET_SETTLED=1
        return
    fi
    remaining=$(( target - done_n ))
    pct=$(( done_n * 100 / target ))

    if (( remaining <= 0 )); then
        success "${label}: ${done_n}/${target} (100%) — COMPLETE."
        TARGET_SETTLED=1
        return
    fi

    local alive="unknown"
    if [[ -n "$proc_pattern" ]]; then
        if pid_alive "$proc_pattern"; then alive="RUNNING"; else alive="NOT RUNNING"; fi
    fi
    if [[ "$alive" == "NOT RUNNING" ]]; then
        TARGET_SETTLED=1
    fi

    local start_epoch elapsed_str avg_rate avg_eta_str recent_str
    start_epoch="$(job_start_epoch "$job_log")"
    if [[ -n "$start_epoch" ]]; then
        local elapsed=$(( now - start_epoch ))
        elapsed_str="$(human_duration "$elapsed")"
        if (( elapsed > 0 && done_n > 0 )); then
            avg_rate=$(( elapsed / done_n ))   # seconds/subject, integer
            local eta_s=$(( avg_rate * remaining ))
            avg_eta_str="~$(human_duration "$eta_s") (avg ${avg_rate}s/subject since job start)"
        else
            avg_eta_str="not enough data yet"
        fi
    else
        elapsed_str="unknown (no job log found)"
        avg_eta_str="unknown"
    fi

    recent_str="(first tick — no trend yet)"
    if [[ -n "${PREV_DONE[$idx]:-}" ]]; then
        local d_done=$(( done_n - PREV_DONE[$idx] ))
        local d_time=$(( now - PREV_EPOCH[$idx] ))
        if (( d_done > 0 && d_time > 0 )); then
            local recent_rate=$(( d_time / d_done ))
            local recent_eta_s=$(( recent_rate * remaining ))
            recent_str="~$(human_duration "$recent_eta_s") (recent ${recent_rate}s/subject, last $(human_duration "$d_time"))"
        else
            recent_str="no progress since last tick ($(human_duration "$d_time") ago)$([[ "$alive" == "NOT RUNNING" ]] && echo " — process not found, check for errors" || echo "")"
        fi
    fi
    PREV_DONE[$idx]=$done_n
    PREV_EPOCH[$idx]=$now

    info "${C_BOLD}${label}${C_RESET}: ${done_n}/${target} (${pct}%)   remaining ${remaining} (${no_dir_n} no dir, ${empty_n} empty dir)   process: ${alive}"
    if (( TARGET_SETTLED )); then
        warn "    job process has exited with ${remaining} subject(s) never reaching the flat product — settled short, not stalled. Check the job log for WARN/ERROR lines (missing confounds, no denoise output, etc.) before rerunning."
    else
        info "    elapsed: ${elapsed_str}   ETA (avg): ${avg_eta_str}"
        info "    ETA (recent trend): ${recent_str}"
    fi
}

# ─── Main loop ──────────────────────────────────────────────────────────────
# Runs all 4 targets and returns whether every one has settled (see
# TARGET_SETTLED above) — the exit condition, not just "100% reached".
run_once_all_settled() {
    local i n_targets=${#TARGET_LABEL[@]}
    local all_settled=1
    info "════════════════════════════════════════════════════════════════"
    report_pull_status
    info "────────────────────────────────────────────────────────────────"
    for (( i = 0; i < n_targets; i++ )); do
        report_target "$i" "${TARGET_LABEL[$i]}" "${TARGET_SRC[$i]}" "${TARGET_FLAT[$i]}" \
            "${TARGET_LOG[$i]}" "${TARGET_PATTERN[$i]}"
        (( TARGET_SETTLED )) || all_settled=0
    done
    info "════════════════════════════════════════════════════════════════"
    return $(( all_settled ? 0 : 1 ))
}

trap 'info "monitor stopped (Ctrl+C) — the jobs it was watching keep running untouched. Rerun anytime to resume watching."; exit 0' INT

if $ONCE; then
    run_once_all_settled || true
    exit 0
fi

info "Starting oversight loop (interval ${INTERVAL}s). Log: ${LOG_FILE}"
info "postprocess_local log : ${POSTPROC_LOG:-<none found>}"
info "flatten_fmriprep log  : ${FMRIPREP_LOG:-<none found>}"
info "pull_derivatives log  : ${PULL_LOG:-<none found>}"

while true; do
    if run_once_all_settled; then
        success "All 4 targets settled (100% reached, or job process exited) — nothing left to watch. Exiting."
        exit 0
    fi
    sleep "$INTERVAL"
done
