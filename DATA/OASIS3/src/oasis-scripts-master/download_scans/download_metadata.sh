#!/bin/bash
#================================================================
# download_metadata.sh
#================================================================
#
# Resumable wrapper around XNAT API for downloading OASIS-3
# clinical/demographics/diagnosis metadata spreadsheets.
#
# Features:
#   - Logs all output to a timestamped log file
#   - Authenticates once per run via a session cookie
#   - Automatically extracts and flattens CSV spreadsheets into __metadata__
#
# Usage:
#   bash download_metadata.sh
#
# Credentials can be supplied via NITRC_PASSWORD env var if set, otherwise prompts.
# ----------------------------------------------------------------

set -euo pipefail

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

OUTPUT_DIR="${REPO_ROOT}/DATA/OASIS3/__metadata__"
LOG_DIR="${REPO_ROOT}/logs/oasis-download"

USERNAME="stephanwudocing"
EXPERIMENT_ID="CENTRAL04_E04187"

# ----------------------------------------------------------------
# Setup
# ----------------------------------------------------------------
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/download_metadata_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "OASIS-3 Clinical Metadata Download" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "Output dir:   $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file:     $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ----------------------------------------------------------------
# Read password securely
# ----------------------------------------------------------------
if [ -n "${NITRC_PASSWORD:-}" ]; then
    PASSWORD="$NITRC_PASSWORD"
else
    read -s -p "Enter NITRC-IR password for ${USERNAME}: " PASSWORD
    echo ""
fi

# ----------------------------------------------------------------
# Helper: URL-encode a string
# ----------------------------------------------------------------
escape_chars_for_URL() {
    local input=${1}
    echo "${input}" | sed -e 's/%/%25/g;' \
        -e 's/ /%20/g; s/</%3C/g; s/>/%3E/g; s/#/%23/g; s/+/%2B/g; s/{/%7B/g; s/}/%7D/g; s/|/%7C/g; s/\\/%5C/g; s/\^/%5E/g; s/~/%7E/g; s/\[/%5B/g; s/\]/%5D/g; s/`/%60/g; s/;/%3B/g; s/\//%2F/g; s/?/%3F/g; s/:/%3A/g; s/@/%40/g; s/=/%3D/g; s/\&/%26/g; s/\$/%24/g'
}

# ----------------------------------------------------------------
# Authenticate once and get cookie jar
# ----------------------------------------------------------------
COOKIE_JAR=".cookies-$(date +%Y%m%d%s).txt"

ENC_USER=$(escape_chars_for_URL "${USERNAME}")
ENC_PASS=$(escape_chars_for_URL "${PASSWORD}")

echo "Authenticating with NITRC-IR..." | tee -a "$LOG_FILE"
if ! curl -f -k -s -u "${ENC_USER}:${ENC_PASS}" \
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
# Download clinical data ZIP
# ----------------------------------------------------------------
DOWNLOAD_URL="https://www.nitrc.org/ir/data/experiments/${EXPERIMENT_ID}/scans/ALL/files?format=zip"
ZIP_FILE="${OUTPUT_DIR}/metadata_temp_${TIMESTAMP}.zip"
EXTRACT_DIR="${OUTPUT_DIR}/metadata_temp_${TIMESTAMP}"

echo "Downloading metadata from XNAT..." | tee -a "$LOG_FILE"
echo "URL: $DOWNLOAD_URL" >> "$LOG_FILE"

if curl -H 'Expect:' --keepalive-time 2 -k --cookie "$COOKIE_JAR" -o "$ZIP_FILE" "$DOWNLOAD_URL"; then
    echo "Download completed. Extracting..." | tee -a "$LOG_FILE"
    mkdir -p "$EXTRACT_DIR"
    unzip -q "$ZIP_FILE" -d "$EXTRACT_DIR" >> "$LOG_FILE" 2>&1
    
    echo "Moving CSV files to output directory..." | tee -a "$LOG_FILE"
    # Find all CSV files and move them to OUTPUT_DIR
    find "$EXTRACT_DIR" -type f -name "*.csv" -exec mv -f {} "$OUTPUT_DIR/" \;
    
    echo "Cleaning up temporary files..." | tee -a "$LOG_FILE"
    rm -f "$ZIP_FILE"
    rm -rf "$EXTRACT_DIR"
    
    echo "SUCCESS: Metadata CSV files downloaded and placed in $OUTPUT_DIR" | tee -a "$LOG_FILE"
else
    echo "ERROR: Failed to download metadata." | tee -a "$LOG_FILE"
    rm -f "$ZIP_FILE"
    rm -rf "$EXTRACT_DIR"
    exit 1
fi
