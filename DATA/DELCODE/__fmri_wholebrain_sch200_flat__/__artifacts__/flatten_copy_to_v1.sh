#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
V0_ROOT="$(cd "$SCRIPT_DIR/../__v0__/fmri" && pwd)"
V2_LONG_ROOT="$(cd "$SCRIPT_DIR/../__v2__/longitudinal" && pwd)"
DEST_ROOT="$(cd "$SCRIPT_DIR/fmri" && pwd)"
ARTIFACTS_DIR="$SCRIPT_DIR/_artifacts"

MODE="dry-run"
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [--dry-run|--run|--verify]" >&2
    exit 1
fi

if [[ "${1:-}" == "--run" ]]; then
    MODE="run"
elif [[ "${1:-}" == "--verify" ]]; then
    MODE="verify"
elif [[ "${1:-}" == "--dry-run" || $# -eq 0 ]]; then
    MODE="dry-run"
else
    echo "Usage: $0 [--dry-run|--run|--verify]" >&2
    exit 1
fi

mkdir -p "$ARTIFACTS_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="$ARTIFACTS_DIR/delcode_v1_copy_manifest_${TIMESTAMP}.tsv"
SUMMARY="$ARTIFACTS_DIR/delcode_v1_copy_summary_${TIMESTAMP}.txt"

# Keep-first deterministic order for __v0__ categories.
CATEGORIES=(
    "AD_postprocessed_v2"
    "Converter_postprocessed_v2"
    "Healthy_postprocessed_v2"
    "MCI_SCD_postprocessed_v2"
)

# Longitudinal scope excludes M0 by design.
VISITS=("M12" "M24" "M36" "M48" "M60")

copyable_file() {
    local filename="$1"
    [[ "$filename" == *_bold_reoriented.nii.gz ]]
}

inject_visit() {
    local filename="$1"
    local visit="$2"
    local renamed

    renamed="${filename/_ses-01_/_ses-01_${visit}_}"
    if [[ "$renamed" == "$filename" ]]; then
        return 1
    fi

    printf '%s' "$renamed"
}

append_manifest() {
    local phase="$1" visit="$2" category="$3" subject="$4" src="$5" dst="$6" action="$7" note="$8"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$phase" "$visit" "$category" "$subject" "$src" "$dst" "$action" "$note" >> "$MANIFEST"
}

print_summary() {
    local copied="$1" skipped_dup="$2" skipped_existing="$3" errors="$4" planned="$5"
    {
        echo "Mode: $MODE"
        echo "Source v0: $V0_ROOT"
        echo "Source v2 longitudinal: $V2_LONG_ROOT"
        echo "Destination: $DEST_ROOT"
        echo "Planned copies: $planned"
        echo "Copied files: $copied"
        echo "Skipped (v0 duplicate subjects keep-first): $skipped_dup"
        echo "Skipped (destination already exists): $skipped_existing"
        echo "Errors: $errors"
        echo "Manifest: $MANIFEST"
    } | tee "$SUMMARY"
}

verify_destination() {
    local invalid_names
    local nested_ses_dirs

    echo "=== Verification ==="
    nested_ses_dirs=$(find "$DEST_ROOT" -type d -name 'ses-*' | wc -l)
    echo "Nested ses-* directories under destination: $nested_ses_dirs"

    invalid_names=$(find "$DEST_ROOT" -type f -name '*.nii.gz' | grep -Ev '_ses-01_M(0|12|24|36|48|60)_' | wc -l || true)
    echo "Files missing visit token pattern (_ses-01_M*): $invalid_names"

    if [[ "$nested_ses_dirs" -ne 0 || "$invalid_names" -ne 0 ]]; then
        echo "Verification failed." >&2
        return 1
    fi

    echo "Verification passed."
}

if [[ "$MODE" == "verify" ]]; then
    verify_destination
    exit 0
fi

if [[ ! -d "$V0_ROOT" || ! -d "$V2_LONG_ROOT" || ! -d "$DEST_ROOT" ]]; then
    echo "Required paths missing." >&2
    echo "V0_ROOT=$V0_ROOT" >&2
    echo "V2_LONG_ROOT=$V2_LONG_ROOT" >&2
    echo "DEST_ROOT=$DEST_ROOT" >&2
    exit 1
fi

printf '%s\n' "phase\tvisit\tcategory\tsubject\tsrc\tdst\taction\tnote" > "$MANIFEST"

declare -A SEEN_V0_SUBJECTS=()
COPIED=0
SKIPPED_DUP=0
SKIPPED_EXISTING=0
ERRORS=0
PLANNED=0

# Phase 1: __v0__ baseline -> M0
for category in "${CATEGORIES[@]}"; do
    category_dir="$V0_ROOT/$category"
    if [[ ! -d "$category_dir" ]]; then
        append_manifest "v0" "M0" "$category" "" "" "" "skip" "missing category dir"
        continue
    fi

    while IFS= read -r -d '' subject_dir; do
        subject="$(basename "$subject_dir")"

        if [[ -n "${SEEN_V0_SUBJECTS[$subject]:-}" ]]; then
            ((SKIPPED_DUP += 1))
            append_manifest "v0" "M0" "$category" "$subject" "$subject_dir" "" "skip" "duplicate subject keep-first"
            continue
        fi

        ses_dir="$subject_dir/ses-01"
        if [[ ! -d "$ses_dir" ]]; then
            ((ERRORS += 1))
            append_manifest "v0" "M0" "$category" "$subject" "$subject_dir" "" "error" "missing ses-01"
            continue
        fi

        mapfile -t files < <(find "$ses_dir" -maxdepth 1 -type f -name '*_bold_reoriented.nii.gz' | sort)
        if [[ "${#files[@]}" -eq 0 ]]; then
            ((ERRORS += 1))
            append_manifest "v0" "M0" "$category" "$subject" "$ses_dir" "" "error" "no *_bold_reoriented.nii.gz"
            continue
        fi

        src="${files[0]}"
        base_name="$(basename "$src")"
        if ! renamed="$(inject_visit "$base_name" "M0")"; then
            ((ERRORS += 1))
            append_manifest "v0" "M0" "$category" "$subject" "$src" "" "error" "cannot inject visit token"
            continue
        fi

        dst_dir="$DEST_ROOT/$subject"
        dst="$dst_dir/$renamed"
        ((PLANNED += 1))

        if [[ -e "$dst" ]]; then
            ((SKIPPED_EXISTING += 1))
            append_manifest "v0" "M0" "$category" "$subject" "$src" "$dst" "skip" "destination exists"
            SEEN_V0_SUBJECTS["$subject"]="$category"
            continue
        fi

        if [[ "$MODE" == "run" ]]; then
            mkdir -p "$dst_dir"
            cp -f "$src" "$dst"
            ((COPIED += 1))
            append_manifest "v0" "M0" "$category" "$subject" "$src" "$dst" "copy" "ok"
        else
            append_manifest "v0" "M0" "$category" "$subject" "$src" "$dst" "plan" "dry-run"
        fi

        SEEN_V0_SUBJECTS["$subject"]="$category"
    done < <(find "$category_dir" -mindepth 1 -maxdepth 1 -type d -name 'sub-*' -print0 | sort -z)
done

# Phase 2: __v2__/longitudinal M12+ (exclude M0)
for visit in "${VISITS[@]}"; do
    visit_dir="$V2_LONG_ROOT/Postprocessed_${visit}"
    if [[ ! -d "$visit_dir" ]]; then
        append_manifest "v2" "$visit" "Postprocessed_${visit}" "" "" "" "skip" "missing visit dir"
        continue
    fi

    while IFS= read -r -d '' subject_dir; do
        subject="$(basename "$subject_dir")"

        mapfile -t files < <(find "$subject_dir" -maxdepth 1 -type f -name '*_bold_reoriented.nii.gz' | sort)
        if [[ "${#files[@]}" -eq 0 ]]; then
            ((ERRORS += 1))
            append_manifest "v2" "$visit" "Postprocessed_${visit}" "$subject" "$subject_dir" "" "error" "no *_bold_reoriented.nii.gz"
            continue
        fi

        src="${files[0]}"
        base_name="$(basename "$src")"
        if ! renamed="$(inject_visit "$base_name" "$visit")"; then
            ((ERRORS += 1))
            append_manifest "v2" "$visit" "Postprocessed_${visit}" "$subject" "$src" "" "error" "cannot inject visit token"
            continue
        fi

        dst_dir="$DEST_ROOT/$subject"
        dst="$dst_dir/$renamed"
        ((PLANNED += 1))

        if [[ -e "$dst" ]]; then
            ((SKIPPED_EXISTING += 1))
            append_manifest "v2" "$visit" "Postprocessed_${visit}" "$subject" "$src" "$dst" "skip" "destination exists"
            continue
        fi

        if [[ "$MODE" == "run" ]]; then
            mkdir -p "$dst_dir"
            cp -f "$src" "$dst"
            ((COPIED += 1))
            append_manifest "v2" "$visit" "Postprocessed_${visit}" "$subject" "$src" "$dst" "copy" "ok"
        else
            append_manifest "v2" "$visit" "Postprocessed_${visit}" "$subject" "$src" "$dst" "plan" "dry-run"
        fi
    done < <(find "$visit_dir" -mindepth 1 -maxdepth 1 -type d -name 'sub-*' -print0 | sort -z)
done

print_summary "$COPIED" "$SKIPPED_DUP" "$SKIPPED_EXISTING" "$ERRORS" "$PLANNED"

if [[ "$MODE" == "run" ]]; then
    verify_destination
fi
