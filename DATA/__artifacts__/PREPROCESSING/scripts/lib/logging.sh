# Shared logging helpers for the preprocessing pipeline.
# Source this from a Slurm job or interactive driver:
#     source "${REPO_ROOT}/scripts/lib/logging.sh"
#
# Provides:
#   progress_filter            stdin->stdout filter that strips tqdm progress-bar spam
#   make_run_logdir <stage> <subject>   echoes (and creates) a per-stage/subject/run log dir
#   write_run_summary <dir> [k=v ...]   writes summary.txt + echoes a one-line banner
#
# Design notes:
# - Tools (dcm2niix, MRIQC, fMRIPrep, nilearn) write tqdm progress with carriage returns, so a
#   single physical line accumulates thousands of frames. progress_filter expands \r to newlines
#   and drops only the bar lines (a leading "<digits>%|"), keeping real warnings/errors. The raw,
#   unfiltered stream is still captured by Slurm's own #SBATCH --output as a safety net.

# Resolve repo root if the caller didn't set it (default to the known install path).
: "${REPO_ROOT:=/dss/dsshome1/0A/di54lup/PREPROCESSING}"
LOG_ROOT="${REPO_ROOT}/logs"

progress_filter() {
    # Expand carriage returns, then drop tqdm progress-bar lines like " 45%|████ | 12/180 [..it/s]".
    stdbuf -oL tr '\r' '\n' | grep --line-buffered -vaE '^[[:space:]]*[0-9]+%\|'
}

make_run_logdir() {
    # Usage: make_run_logdir <stage_dirname> <subject_label>
    # Echoes the created directory path: logs/<stage>/sub-<subject>/<timestamp>[_job<jobid>]
    local stage="$1" subject="$2"
    local ts; ts="$(date +%Y%m%d-%H%M%S)"
    local suffix=""
    [ -n "${SLURM_JOB_ID:-}" ] && suffix="_job${SLURM_JOB_ID}"
    local dir="${LOG_ROOT}/${stage}/sub-${subject}/${ts}${suffix}"
    mkdir -p "${dir}"
    echo "${dir}"
}

write_run_summary() {
    # Usage: write_run_summary <run_dir> [extra "key: value" lines ...]
    local dir="$1"; shift
    local summary="${dir}/summary.txt"
    {
        echo "stage_dir   : ${dir#${LOG_ROOT}/}"
        echo "job_id      : ${SLURM_JOB_ID:-<interactive>}"
        echo "job_name    : ${SLURM_JOB_NAME:-<interactive>}"
        echo "node        : ${SLURMD_NODENAME:-$(hostname)}"
        echo "start_time  : $(date '+%Y-%m-%d %H:%M:%S %Z')"
        local line
        for line in "$@"; do echo "${line}"; done
    } > "${summary}"
    echo ">> logs: ${dir}"
    cat "${summary}"
}
