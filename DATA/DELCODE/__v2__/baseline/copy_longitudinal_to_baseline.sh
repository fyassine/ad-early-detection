#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINE_DIR="$SCRIPT_DIR"
LONGITUDINAL_DIR="$(cd "$SCRIPT_DIR/../longitudinal" && pwd)"

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

declare -a COPY_SOURCES=()
declare -a COPY_DESTINATIONS=()
declare -a COPY_VISITS=()
declare -a COPY_TYPES=()

declare -A PLANNED_DESTINATIONS=()
declare -A CREATED_TARGETS=()
declare -A VISIT_COPY_COUNTS=()
declare -A VISIT_OVERWRITE_COUNTS=()
declare -A VISIT_SUBJECTS=()
declare -A TYPE_COPY_COUNTS=()

COPIED=0
ERRORS=0
WARNINGS=0
OVERWRITES=0
NEW_SUBJECT_DIRS=0

warn() {
    echo "WARNING: $1" >&2
    ((WARNINGS += 1))
}

error() {
    echo "ERROR: $1" >&2
    ((ERRORS += 1))
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

print_mkdir_once() {
    local target_dir="$1"
    if [[ -z "${CREATED_TARGETS[$target_dir]:-}" ]]; then
        echo "  mkdir -p $target_dir"
        CREATED_TARGETS["$target_dir"]=1
    fi
}

should_copy_file() {
    local filename="$1"

    case "$filename" in
        *_desc-sliced_bold.nii.gz|*_desc-smoothed_bold.nii.gz)
            return 1
            ;;
        *_bold_reoriented.nii.gz)
            return 0
            ;;
        *_bold.nii.gz)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

file_kind() {
    local filename="$1"
    if [[ "$filename" == *_bold_reoriented.nii.gz ]]; then
        echo "bold_reoriented"
    else
        echo "bold"
    fi
}

extract_visit() {
    local visit_dir_name="$1"
    if [[ "$visit_dir_name" =~ ^Postprocessed_(M[0-9]+)$ ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi

    return 1
}

rename_for_visit() {
    local filename="$1" visit="$2"
    local renamed

    renamed="${filename/_ses-01_/_ses-01_${visit}_}"
    if [[ "$renamed" == "$filename" ]]; then
        return 1
    fi

    echo "$renamed"
}

queue_copy() {
    local src="$1" dst="$2" visit="$3" file_type="$4"

    if [[ ! -f "$src" ]]; then
        error "missing source file: $src"
        return
    fi

    if [[ -n "${PLANNED_DESTINATIONS[$dst]:-}" ]]; then
        error "destination collision: $dst already planned for ${PLANNED_DESTINATIONS[$dst]}"
        return
    fi

    if [[ -e "$dst" ]]; then
        ((OVERWRITES += 1))
        ((VISIT_OVERWRITE_COUNTS["$visit"] += 1))
    fi

    PLANNED_DESTINATIONS["$dst"]="$src"
    COPY_SOURCES+=("$src")
    COPY_DESTINATIONS+=("$dst")
    COPY_VISITS+=("$visit")
    COPY_TYPES+=("$file_type")
    ((VISIT_COPY_COUNTS["$visit"] += 1))
    ((TYPE_COPY_COUNTS["$file_type"] += 1))
}

build_copy_plan() {
    local visit_dir visit_dir_name visit sub_dir sub_id target_dir filename dst_name dst file_type
    local found_visit_dirs

    found_visit_dirs=0

    for visit_dir in "$LONGITUDINAL_DIR"/Postprocessed_*; do
        [[ -d "$visit_dir" ]] || continue
        found_visit_dirs=1
        visit_dir_name=$(basename "$visit_dir")

        if ! visit=$(extract_visit "$visit_dir_name"); then
            warn "skipping unexpected visit directory: $visit_dir"
            continue
        fi

        for sub_dir in "$visit_dir"/sub-*; do
            [[ -d "$sub_dir" ]] || continue
            sub_id=$(basename "$sub_dir")
            target_dir="$BASELINE_DIR/$sub_id"

            if [[ ! -d "$target_dir" && -z "${CREATED_TARGETS[$target_dir]:-}" ]]; then
                CREATED_TARGETS["$target_dir"]=planned
                ((NEW_SUBJECT_DIRS += 1))
            fi

            VISIT_SUBJECTS["$visit::$sub_id"]=1

            while IFS= read -r -d '' source_file; do
                filename=$(basename "$source_file")
                if ! should_copy_file "$filename"; then
                    continue
                fi

                if ! dst_name=$(rename_for_visit "$filename" "$visit"); then
                    warn "cannot inject visit label into filename: $source_file"
                    continue
                fi

                file_type=$(file_kind "$filename")
                dst="$target_dir/$dst_name"
                queue_copy "$source_file" "$dst" "$visit" "$file_type"
            done < <(find "$sub_dir" -maxdepth 1 -type f -name '*.nii.gz' -print0)
        done
    done

    if [[ "$found_visit_dirs" -eq 0 ]]; then
        error "no longitudinal visit directories found under $LONGITUDINAL_DIR"
    fi
}

print_plan_summary() {
    local visit

    echo "================================================"
    echo "Planned copies by visit:"
    for visit in M0 M12 M24 M36 M48 M60; do
        echo "  $visit: ${VISIT_COPY_COUNTS[$visit]:-0} file(s), ${VISIT_OVERWRITE_COUNTS[$visit]:-0} overwrite(s)"
    done
    echo "Planned copies by file type:"
    echo "  bold: ${TYPE_COPY_COUNTS[bold]:-0} file(s)"
    echo "  bold_reoriented: ${TYPE_COPY_COUNTS[bold_reoriented]:-0} file(s)"
    echo "New subject directories to create: $NEW_SUBJECT_DIRS"
    echo "Total overwrites: $OVERWRITES"
    echo "Warnings: $WARNINGS"
}

print_overwrite_preview() {
    local index src dst
    local printed

    printed=0
    if [[ "$OVERWRITES" -eq 0 ]]; then
        return
    fi

    echo "Overwrite preview (first 20):"
    for index in "${!COPY_SOURCES[@]}"; do
        src="${COPY_SOURCES[$index]}"
        dst="${COPY_DESTINATIONS[$index]}"
        if [[ -e "$dst" ]]; then
            echo "  cp -f $src -> $dst"
            ((printed += 1))
            if [[ "$printed" -ge 20 ]]; then
                break
            fi
        fi
    done
}

execute_copies() {
    local total_copies index src dst target_dir

    total_copies=${#COPY_SOURCES[@]}
    for index in "${!COPY_SOURCES[@]}"; do
        src="${COPY_SOURCES[$index]}"
        dst="${COPY_DESTINATIONS[$index]}"
        target_dir=$(dirname "$dst")

        print_progress "$COPIED" "$total_copies"
        mkdir -p "$target_dir"
        cp -f "$src" "$dst"
        ((COPIED += 1))
    done

    print_progress "$COPIED" "$total_copies"
    echo ""
}

verify_copy_state() {
    local index visit file_type src dst issues missing unexpected
    local target_subjects

    issues=0
    missing=0
    unexpected=0

    echo "=== Verification ==="
    target_subjects=$(find "$BASELINE_DIR" -maxdepth 1 -type d -name 'sub-*' | wc -l)
    echo "Root-level subject directories in baseline: $target_subjects"

    for index in "${!COPY_SOURCES[@]}"; do
        dst="${COPY_DESTINATIONS[$index]}"
        if [[ ! -f "$dst" ]]; then
            echo "  MISSING: $dst" >&2
            ((missing += 1))
        fi
    done

    for visit in M0 M12 M24 M36 M48 M60; do
        echo "  $visit expected target files: ${VISIT_COPY_COUNTS[$visit]:-0}"
    done

    unexpected=$(find "$BASELINE_DIR" -maxdepth 2 -type f \( -name '*_desc-sliced_bold.nii.gz' -o -name '*_desc-smoothed_bold.nii.gz' \) | wc -l)
    echo "Excluded sliced/smoothed files under baseline subject folders: $unexpected"

    if [[ "$missing" -gt 0 ]]; then
        ((issues += 1))
        echo "Verification failed: $missing expected target file(s) missing." >&2
    fi

    echo "Longitudinal source tree remains at: $LONGITUDINAL_DIR"

    if [[ "$issues" -gt 0 ]]; then
        return 1
    fi

    echo "Verification passed."
}

build_copy_plan

print_plan_summary
print_overwrite_preview

if [[ "$ERRORS" -gt 0 ]]; then
    echo "Errors: $ERRORS"
    echo "Aborting before any copy." >&2
    exit 1
fi

if $VERIFY_ONLY; then
    verify_copy_state
    exit $?
fi

if $DRY_RUN; then
    echo "[DRY RUN] Would copy ${#COPY_SOURCES[@]} file(s). Errors: $ERRORS"
    echo ""
    echo "Run without --dry-run to execute:"
    echo "  ./copy_longitudinal_to_baseline.sh"
    exit 0
fi

execute_copies
echo "Copied $COPIED file(s). Errors: $ERRORS"
echo ""
verify_copy_state