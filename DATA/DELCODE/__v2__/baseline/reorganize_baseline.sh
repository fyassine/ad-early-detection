#!/usr/bin/env bash
#
# Reorganize fMRI baseline scans into flat sub-*/ directories.
# Moves all .nii.gz files from cohort subdirectories into baseline/sub-XXXX/
# (no ses-01/ nesting). Uses mv (instant metadata rename on same NFS volume).
#
# Usage:
#   ./reorganize_baseline.sh --dry-run   # Preview operations
#   ./reorganize_baseline.sh             # Execute moves
#
set -euo pipefail

BASELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
DRY_RUN=false
VERIFY_ONLY=false

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [--dry-run|--verify]" >&2
    exit 1
fi

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
elif [[ "${1:-}" == "--verify" ]]; then
    VERIFY_ONLY=true
elif [[ $# -eq 1 ]]; then
    echo "Usage: $0 [--dry-run|--verify]" >&2
    exit 1
fi

if $DRY_RUN; then
    echo "[DRY RUN] No files will be moved."
    echo ""
fi

declare -a MOVE_SOURCES=()
declare -a MOVE_DESTINATIONS=()
declare -a MOVE_COHORTS=()

declare -A PLANNED_DESTINATIONS=()
declare -A CREATED_TARGETS=()
declare -A COHORT_MOVE_COUNTS=()
declare -A COHORT_SUBJECTS=()

MOVED=0
ERRORS=0
WARNINGS=0

warn() {
    echo "WARNING: $1" >&2
    ((WARNINGS += 1))
}

error() {
    echo "ERROR: $1" >&2
    ((ERRORS += 1))
}

mark_subject() {
    local cohort="$1" sub_id="$2"
    COHORT_SUBJECTS["$cohort::$sub_id"]=1
}

queue_move() {
    local cohort="$1" src="$2" dst="$3"

    if [[ ! -f "$src" ]]; then
        error "missing source file: $src"
        return
    fi

    if [[ -e "$dst" ]]; then
        error "destination already exists: $dst"
        return
    fi

    if [[ -n "${PLANNED_DESTINATIONS[$dst]:-}" ]]; then
        error "destination collision: $dst already planned for ${PLANNED_DESTINATIONS[$dst]}"
        return
    fi

    PLANNED_DESTINATIONS["$dst"]="$src"
    MOVE_SOURCES+=("$src")
    MOVE_DESTINATIONS+=("$dst")
    MOVE_COHORTS+=("$cohort")
    COHORT_MOVE_COUNTS["$cohort"]=$(( ${COHORT_MOVE_COUNTS[$cohort]:-0} + 1 ))
}

print_mkdir_once() {
    local target_dir="$1"
    if [[ -z "${CREATED_TARGETS[$target_dir]:-}" ]]; then
        echo "  mkdir -p $target_dir"
        CREATED_TARGETS["$target_dir"]=1
    fi
}

cleanup_known_temp_files() {
    local cohort_path="$1"

    while IFS= read -r -d '' temp_file; do
        echo "  Removing temp file $temp_file"
        rm -f "$temp_file"
    done < <(find "$cohort_path" -type f -name '*.nii.gz.PVte0I' -print0)
}

execute_moves() {
    local index src dst target_dir total_moves completed left

    total_moves=${#MOVE_SOURCES[@]}
    for index in "${!MOVE_SOURCES[@]}"; do
        src="${MOVE_SOURCES[$index]}"
        dst="${MOVE_DESTINATIONS[$index]}"
        target_dir="$(dirname "$dst")"

        print_progress "$MOVED" "$total_moves"
        mkdir -p "$target_dir"
        mv "$src" "$dst"
        ((MOVED += 1))
    done

    print_progress "$MOVED" "$total_moves"
    echo ""
}

print_progress() {
    local completed="$1" total="$2"
    local left bar_width filled empty percent bar

    if [[ "$total" -eq 0 ]]; then
        return
    fi

    left=$(( total - completed ))
    bar_width=40
    filled=$(( completed * bar_width / total ))
    empty=$(( bar_width - filled ))
    percent=$(( completed * 100 / total ))
    bar=$(printf '%*s' "$filled" '' | tr ' ' '#')
    bar+=$(printf '%*s' "$empty" '' | tr ' ' '-')

    printf '\r[%s] %4d%%  %d/%d done, %d left' "$bar" "$percent" "$completed" "$total" "$left"
}

verify_post_run() {
    local cohort cohort_path leftover_nii leftover_other leftover_temp
    local target_subjects target_files nested_dirs issues

    issues=0
    target_subjects=$(find "$BASELINE_DIR" -maxdepth 1 -type d -name 'sub-*' | wc -l)
    target_files=$(find "$BASELINE_DIR" -mindepth 2 -maxdepth 2 -type f -path "$BASELINE_DIR/sub-*/*.nii.gz" | wc -l)
    nested_dirs=$(find "$BASELINE_DIR" -mindepth 2 -maxdepth 2 -type d -path "$BASELINE_DIR/sub-*/*" | wc -l)

    echo "=== Verification ==="
    echo "Root-level subject directories: $target_subjects"
    echo "NIfTI files under root-level subject directories: $target_files"

    if [[ "$nested_dirs" -gt 0 ]]; then
        echo "  WARNING: found $nested_dirs nested directory/directories under root-level subject folders" >&2
        ((issues += 1))
    else
        echo "Nested directories under root-level subject folders: 0"
    fi

    echo "Cohort leftovers:"
    for cohort in "${STRUCTURED_COHORTS[@]}" "Delcode_Converter_graph_data"; do
        cohort_path="$BASELINE_DIR/$cohort"
        if [[ ! -d "$cohort_path" ]]; then
            echo "  $cohort: removed"
            continue
        fi

        leftover_nii=$(find "$cohort_path" -type f -name '*.nii.gz' | wc -l)
        leftover_temp=$(find "$cohort_path" -type f -name '*.nii.gz.PVte0I' | wc -l)
        leftover_other=$(find "$cohort_path" -type f ! -name '*.nii.gz' ! -name '*.nii.gz.PVte0I' | wc -l)

        echo "  $cohort: $leftover_nii NIfTI, $leftover_temp temp, $leftover_other other file(s) remain"

        if [[ "$leftover_nii" -gt 0 || "$leftover_temp" -gt 0 || "$leftover_other" -gt 0 ]]; then
            ((issues += 1))
        fi
    done

    if [[ "$issues" -gt 0 ]]; then
        echo "Verification failed with $issues issue(s)." >&2
        return 1
    fi

    echo "Verification passed."
}

if $VERIFY_ONLY; then
    verify_post_run
    exit $?
fi

# ─── Cohorts with sub-*/ses-01/*.nii.gz structure ─────────────────────────────
STRUCTURED_COHORTS=(
    "Delcode_AD_graph_data"
    "Delcode_MCI_exclude_converter_graph_data"
    "Delcode_healthy_graph_data_demographics"
)

for cohort in "${STRUCTURED_COHORTS[@]}"; do
    COHORT_DIR="$BASELINE_DIR/$cohort/pre-raw"
    if [[ ! -d "$COHORT_DIR" ]]; then
        echo "SKIP: $COHORT_DIR not found"
        continue
    fi

    echo "=== Processing $cohort ==="
    for sub_dir in "$COHORT_DIR"/sub-*/; do
        [[ -d "$sub_dir" ]] || continue
        sub_id=$(basename "$sub_dir")
        target_dir="$BASELINE_DIR/$sub_id"
        mark_subject "$cohort" "$sub_id"

        while IFS= read -r -d '' unexpected_dir; do
            warn "$cohort has unexpected session directory: $unexpected_dir"
        done < <(find "$sub_dir" -mindepth 1 -maxdepth 1 -type d ! -name 'ses-01' -print0)

        if [[ ! -d "$sub_dir/ses-01" ]]; then
            warn "$cohort subject missing ses-01 directory: $sub_dir"
            continue
        fi

        if $DRY_RUN; then
            print_mkdir_once "$target_dir"
        fi

        found_files=0
        while IFS= read -r -d '' nii_file; do
            found_files=1
            filename=$(basename "$nii_file")
            dst="$target_dir/$filename"
            queue_move "$cohort" "$nii_file" "$dst"
            if $DRY_RUN && [[ $ERRORS -eq 0 ]]; then
                echo "  mv $nii_file -> $dst"
            fi
        done < <(find "$sub_dir/ses-01" -type f -name "*.nii.gz" -print0)

        if [[ $found_files -eq 0 ]]; then
            warn "$cohort subject has no .nii.gz files under ses-01: $sub_dir"
        fi
    done
    echo ""
done

# ─── Converter cohort (flat files) ────────────────────────────────────────────
CONVERTER_DIR="$BASELINE_DIR/Delcode_Converter_graph_data/pre-raw"
if [[ -d "$CONVERTER_DIR" ]]; then
    echo "=== Processing Delcode_Converter_graph_data ==="
    while IFS= read -r -d '' nii_file; do
        filename=$(basename "$nii_file")
        if [[ ! "$filename" =~ ^(sub-[^_]+)_ses- ]]; then
            warn "cannot extract subject ID from converter filename: $filename"
            continue
        fi

        sub_id="${BASH_REMATCH[1]}"
        target_dir="$BASELINE_DIR/$sub_id"
        mark_subject "Delcode_Converter_graph_data" "$sub_id"

        if $DRY_RUN; then
            print_mkdir_once "$target_dir"
        fi

        dst="$target_dir/$filename"
        queue_move "Delcode_Converter_graph_data" "$nii_file" "$dst"
        if $DRY_RUN && [[ $ERRORS -eq 0 ]]; then
            echo "  mv $nii_file -> $dst"
        fi
    done < <(find "$CONVERTER_DIR" -maxdepth 1 -type f -name '*.nii.gz' -print0)
    echo ""
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
echo "================================================"
echo "Planned moves by cohort:"
for cohort in "${STRUCTURED_COHORTS[@]}" "Delcode_Converter_graph_data"; do
    echo "  $cohort: ${COHORT_MOVE_COUNTS[$cohort]:-0} file(s)"
done
echo "Warnings: $WARNINGS"

if [[ $ERRORS -gt 0 ]]; then
    echo "Errors: $ERRORS"
    echo "Aborting before any moves." >&2
    exit 1
fi

if $DRY_RUN; then
    echo "[DRY RUN] Would move ${#MOVE_SOURCES[@]} files. Errors: $ERRORS"
    echo ""
    echo "Run without --dry-run to execute:"
    echo "  ./reorganize_baseline.sh"
else
    execute_moves
    echo "Moved $MOVED files. Errors: $ERRORS"
    echo ""
    # Clean up empty cohort directories
    echo "=== Cleaning up empty directories ==="
    for cohort in "${STRUCTURED_COHORTS[@]}" "Delcode_Converter_graph_data"; do
        cohort_path="$BASELINE_DIR/$cohort"
        if [[ -d "$cohort_path" ]]; then
            cleanup_known_temp_files "$cohort_path"
            remaining=$(find "$cohort_path" -type f 2>/dev/null | wc -l)
            if [[ "$remaining" -eq 0 ]]; then
                echo "  Removing empty $cohort/"
                rm -rf "$cohort_path"
            else
                echo "  WARNING: $cohort/ still has $remaining file(s), not removing"
            fi
        fi
    done

    echo ""
    verify_post_run
fi

echo ""
echo "Done. Total sub-* dirs in baseline: $(find "$BASELINE_DIR" -maxdepth 1 -type d -name 'sub-*' | wc -l)"
