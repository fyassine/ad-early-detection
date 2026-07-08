#!/bin/bash
#================================================================
# download_smri_bids.sh
#================================================================
#
# Resumable wrapper around download_oasis_scans_bids.sh for
# downloading OASIS-3 structural T1w MRI scans in BIDS format,
# for the same MR sessions already downloaded by download_bold_bids.sh.
#
# T1w and BOLD scans acquired in the same visit share one MR session
# (experiment_id, e.g. OAS30001_MR_d0129), so this script reuses the
# bold sessions CSV and writes into the same subject/session tree,
# under anat/ instead of func/.
#
# Features:
#   - Skips sessions already successfully downloaded (non-empty anat/ folder)
#   - Logs all output to a timestamped log file
#   - Can be safely interrupted (Ctrl+C) and re-run — resumes from where it left off
#   - Authenticates once per run via a session cookie (no password re-prompting)
#
# Usage:
#   bash download_smri_bids.sh
#
# Credentials are read from NITRC_USERNAME / NITRC_PASSWORD in the repo-root
# .env file (or already-exported env vars) for non-interactive tmux/background use.
# ----------------------------------------------------------------

set -euo pipefail

# ----------------------------------------------------------------
# Configuration — edit if paths change
# ----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script lives at DATA/OASIS3/src/oasis-scripts-master/download_scans/ — go up 5 levels to reach the project root
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

SESSIONS_CSV="${REPO_ROOT}/DATA/OASIS3/sessions/oasis3_bold_sessions.csv"
OUTPUT_DIR="${REPO_ROOT}/DATA/OASIS3/__bold_and_smri__"
LOG_DIR="${REPO_ROOT}/logs/oasis-download"

if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

if [ -z "${NITRC_USERNAME:-}" ] || [ -z "${NITRC_PASSWORD:-}" ]; then
    echo "ERROR: NITRC_USERNAME and NITRC_PASSWORD must be set in ${REPO_ROOT}/.env (or exported in the environment)." >&2
    exit 1
fi

USERNAME="$NITRC_USERNAME"
SCAN_TYPE="T1w"

# ----------------------------------------------------------------
# Setup
# ----------------------------------------------------------------
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/download_smri_${TIMESTAMP}.log"
PROGRESS_FILE="${LOG_DIR}/progress_smri_${TIMESTAMP}.txt"

echo "========================================" | tee -a "$LOG_FILE"
echo "OASIS-3 sMRI (T1w) BIDS Download" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "Sessions CSV: $SESSIONS_CSV" | tee -a "$LOG_FILE"
echo "Output dir:   $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file:     $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

PASSWORD="$NITRC_PASSWORD"

# ----------------------------------------------------------------
# Authenticate once and get cookie jar
# ----------------------------------------------------------------
COOKIE_JAR=".cookies-$(date +%Y%m%d%s).txt"

echo "Authenticating with NITRC-IR..." | tee -a "$LOG_FILE"
if ! curl -f -k -s -u "${USERNAME}:${PASSWORD}" \
    --cookie-jar "$COOKIE_JAR" \
    "https://www.nitrc.org/ir/data/JSESSION" > /dev/null; then
    echo "ERROR: Authentication failed. Check username/password." | tee -a "$LOG_FILE"
    rm -f "$COOKIE_JAR"
    exit 1
fi
echo "Authentication successful." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ----------------------------------------------------------------
# Cleanup on exit
# ----------------------------------------------------------------
cleanup() {
    echo "" | tee -a "$LOG_FILE"
    echo "Cleaning up session cookie..." | tee -a "$LOG_FILE"
    curl -i -k --cookie "$COOKIE_JAR" -X DELETE \
        "https://www.nitrc.org/ir/data/JSESSION" >> "$LOG_FILE" 2>&1 || true
    rm -f "$COOKIE_JAR"
    echo "Session ended: $(date)" | tee -a "$LOG_FILE"
}
trap cleanup EXIT

# ----------------------------------------------------------------
# Download helper functions (mirrors oasis-scripts internals)
# ----------------------------------------------------------------
download_file() {
    local OUTPUT=${1}
    local URL=${2}
    curl -H 'Expect:' --keepalive-time 2 -k --cookie "$COOKIE_JAR" -o "$OUTPUT" "$URL"
}

get_url() {
    local URL=${1}
    curl -H 'Expect:' --keepalive-time 2 -k --cookie "$COOKIE_JAR" "$URL"
}

move_to_bids() {
    local DIRNAME=$1
    local EXPERIMENT_ID=$2

    for SCAN_FOLDER_PATH in "$DIRNAME/$EXPERIMENT_ID/scans"/*/; do
        [ -d "$SCAN_FOLDER_PATH" ] || continue
        SCAN_FOLDERNAME=$(basename "$SCAN_FOLDER_PATH")

        for SCAN_FILE_PATH in "$DIRNAME/$EXPERIMENT_ID/scans/$SCAN_FOLDERNAME/resources/"*/files/*; do
            [ -f "$SCAN_FILE_PATH" ] || continue

            SCAN_TYPE=$(echo "$SCAN_FOLDERNAME" | cut -d- -f2)
            SCAN_FILENAME=$(basename "$SCAN_FILE_PATH")

            # Fix _sess- -> _ses- for BIDS compliance
            if [[ "$SCAN_FILENAME" == *"_sess-"* ]]; then
                NEW_NAME="${SCAN_FILENAME//_sess-/_ses-}"
                mv "$(dirname "$SCAN_FILE_PATH")/$SCAN_FILENAME" \
                   "$(dirname "$SCAN_FILE_PATH")/$NEW_NAME"
                SCAN_FILENAME="$NEW_NAME"
                SCAN_FILE_PATH="$(dirname "$SCAN_FILE_PATH")/$NEW_NAME"
            fi

            scan_subject=$(echo "$SCAN_FILENAME" | cut -d_ -f1 | cut -d- -f2)
            scan_session_raw=$(echo "$SCAN_FILENAME" | cut -d_ -f2 | cut -d- -f2)
            # Normalize days to 4-digit zero-padded
            scan_session="d$(printf "%04d" "$((10#${scan_session_raw:1}))")"

            subject_folder="sub-${scan_subject}"
            session_folder="ses-${scan_session}"
            dest_base="$DIRNAME/$subject_folder/$session_folder"

            case "$SCAN_TYPE" in
                bold|asl) dest_type="func" ;;
                T1w|T2w|FLAIR|T2star|angio) dest_type="anat" ;;
                fieldmap) dest_type="fmap" ;;
                dwi|dti) dest_type="dwi" ;;
                swi|minIP|GRE) dest_type="swi" ;;
                *) dest_type="other" ;;
            esac

            dest_path="$dest_base/$dest_type"
            mkdir -p "$dest_path"
            mv "$SCAN_FILE_PATH" "$dest_path/"
            chmod -R u=rwX,g=rwX "$dest_path"
        done
    done
}

# ----------------------------------------------------------------
# Count total sessions
# ----------------------------------------------------------------
TOTAL=$(tail -n +2 "$SESSIONS_CSV" | wc -l)
DONE=0
SKIPPED=0
FAILED=0

echo "Total sessions to process: $TOTAL" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ----------------------------------------------------------------
# Main download loop
# ----------------------------------------------------------------
while IFS=, read -r EXPERIMENT_ID; do
    # Strip carriage returns (Windows line endings)
    EXPERIMENT_ID="${EXPERIMENT_ID//$'\r'/}"
    [ -z "$EXPERIMENT_ID" ] && continue

    SUBJECT_ID=$(echo "$EXPERIMENT_ID" | cut -d_ -f1)

    # Skip already-downloaded sessions:
    # Check if sub-OASXXXXX/ses-dXXXX/anat/ exists and is non-empty
    DAYS_RAW=$(echo "$EXPERIMENT_ID" | grep -oP 'd\d+$' || echo "d0000")
    DAYS_NUM=$(printf "%04d" "$((10#${DAYS_RAW:1}))")
    BIDS_SUBJ="sub-${SUBJECT_ID/OAS3/OAS3}"
    BIDS_SESS="ses-d${DAYS_NUM}"
    EXPECTED_ANAT="${OUTPUT_DIR}/${BIDS_SUBJ}/${BIDS_SESS}/anat"

    if [ -d "$EXPECTED_ANAT" ] && [ -n "$(ls -A "$EXPECTED_ANAT" 2>/dev/null)" ]; then
        echo "[SKIP] ${EXPERIMENT_ID} — already downloaded" | tee -a "$LOG_FILE"
        SKIPPED=$((SKIPPED + 1))
        DONE=$((DONE + 1))
        continue
    fi

    # Determine project ID
    PROJECT_ID=OASIS3
    if [[ "$EXPERIMENT_ID" == "OAS4"* ]]; then PROJECT_ID=OASIS4; fi
    if [[ "$EXPERIMENT_ID" == "OAS3"*"_AV1451"* ]]; then PROJECT_ID=OASIS3_AV1451; fi

    DOWNLOAD_URL="https://www.nitrc.org/ir/data/archive/projects/${PROJECT_ID}/subjects/${SUBJECT_ID}/experiments/${EXPERIMENT_ID}/scans/${SCAN_TYPE}/files?format=tar.gz"

    echo "[$(date +%H:%M:%S)] [${DONE}/${TOTAL}] Downloading ${EXPERIMENT_ID}..." | tee -a "$LOG_FILE"
    echo "  URL: $DOWNLOAD_URL" >> "$LOG_FILE"

    download_file "$OUTPUT_DIR/$EXPERIMENT_ID.tar.gz" "$DOWNLOAD_URL" >> "$LOG_FILE" 2>&1

    if tar tf "$OUTPUT_DIR/$EXPERIMENT_ID.tar.gz" &> /dev/null; then
        echo "  [OK] tar.gz valid, extracting..." | tee -a "$LOG_FILE"
        tar -xzC "$OUTPUT_DIR" -f "$OUTPUT_DIR/$EXPERIMENT_ID.tar.gz" >> "$LOG_FILE" 2>&1
        move_to_bids "$OUTPUT_DIR" "$EXPERIMENT_ID"
        rm -rf "$OUTPUT_DIR/$EXPERIMENT_ID"
        echo "  [OK] Done: ${EXPERIMENT_ID}" | tee -a "$LOG_FILE"
        echo "$EXPERIMENT_ID" >> "$PROGRESS_FILE"
    else
        echo "  [WARN] No valid T1w scan found for ${EXPERIMENT_ID}" | tee -a "$LOG_FILE"
        FAILED=$((FAILED + 1))
    fi

    # Always clean up the tar.gz
    rm -f "$OUTPUT_DIR/$EXPERIMENT_ID.tar.gz"

    DONE=$((DONE + 1))

    # Progress summary every 50 sessions
    if (( DONE % 50 == 0 )); then
        echo "" | tee -a "$LOG_FILE"
        echo "=== Progress: ${DONE}/${TOTAL} | Skipped: ${SKIPPED} | Failed: ${FAILED} ===" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    fi

done < <(tail -n +2 "$SESSIONS_CSV")

# ----------------------------------------------------------------
# Final summary
# ----------------------------------------------------------------
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Download complete: $(date)" | tee -a "$LOG_FILE"
echo "Total sessions:  $TOTAL" | tee -a "$LOG_FILE"
echo "Downloaded:      $((DONE - SKIPPED - FAILED))" | tee -a "$LOG_FILE"
echo "Skipped (already done): $SKIPPED" | tee -a "$LOG_FILE"
echo "Failed (no T1w scan):   $FAILED" | tee -a "$LOG_FILE"
echo "Output dir: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file:   $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
