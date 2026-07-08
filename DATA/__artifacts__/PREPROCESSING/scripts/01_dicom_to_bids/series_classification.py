"""Maps dcm2niix output (NIfTI + BIDS JSON sidecar pairs) to BIDS categories.

Rules below were derived from inspecting the actual SeriesDescription/ImageType fields in
SAMPLE/03a0a6663-M0_T1_01 (dzne/delcode protocol). Series not matched by any rule are treated
as "unmapped" and are not converted into the BIDS tree by default (logged for manual review)
rather than guessed at, since DELCODE add-on sequences (FLASH, IR-EPI, T2, a second 3iso
fieldmap with no geometry match to anything else in this dataset) are out of scope for the
resting-state connectivity pipeline.
"""
from dataclasses import dataclass


@dataclass
class Classification:
    category: str          # "anat_T1w" | "anat_FLAIR" | "func_bold" | "fmap_magnitude" |
                            # "fmap_phasediff" | "dwi" | "skip" | "unmapped"
    rec_label: str | None = None   # e.g. "ND" for the non-distortion-corrected variant


def classify(series_description: str, image_type: list[str]) -> Classification:
    desc = series_description.lower()
    is_nd = "_nd" in desc or desc.endswith("nd")

    # Scanner-derived secondary series: never independent acquisitions, always discard.
    if "mocoseries" in desc:
        return Classification("skip")
    if "localizer" in desc:
        return Classification("skip")
    if any(tag in desc for tag in ["_adc", "_tracew", "_fa", "_colfa"]):
        return Classification("skip")

    if "mprage" in desc:
        return Classification("anat_T1w", rec_label="ND" if is_nd else None)

    if "flair" in desc:
        return Classification("anat_FLAIR", rec_label="ND" if is_nd else None)

    if "restingstate" in desc.replace(" ", ""):
        return Classification("func_bold")

    if "fieldmap" in desc:
        # Siemens dual-echo GRE fieldmap: magnitude images carry ImageType 'M', the
        # phase-difference map carries 'P'. EchoNumbers distinguishes magnitude echo 1/2
        # when both are bundled in one series (as in this dataset: 94 files = 47 slices x 2 echoes).
        if "P" in image_type:
            return Classification("fmap_phasediff")
        return Classification("fmap_magnitude")

    if "dti_v2_2iso" in desc and not any(t in desc for t in ["_adc", "_tracew", "_fa", "_colfa"]):
        return Classification("dwi")

    # FLASH, IR-EPI, T2, and any fieldmap whose geometry doesn't match a func/dwi run in this
    # dataset (e.g. the 3iso pair when only a 3.5iso BOLD run exists) fall through here.
    return Classification("unmapped")
