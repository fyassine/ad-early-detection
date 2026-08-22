"""Regression tests for DATA/PREPROCESSING/src/fritz/final_reorient.py.

reorient_to_radiological() previously negated the x direction-cosine without
compensating the translation term, anchoring the flip at world x=0 instead of
the image's own bounding box and shifting the whole volume by
(nx-1)*voxel_size_x (192mm on a typical 97-voxel, 2mm-iso MNI image) — enough
to move ADNI/OASIS-3 BOLD data entirely outside the Schaefer atlas's field of
view and silently block FC extraction. These tests pin the two properties
that regression broke: the physical bounding box must be preserved, and the
voxel data array must be untouched (header-only edit).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import nibabel as nib

from DATA.PREPROCESSING.src.fritz.final_reorient import reorient_to_radiological


def _corners_world(affine: npt.NDArray[np.float64], shape: tuple[int, int, int]) -> npt.NDArray[np.float64]:
    idx = np.array(np.meshgrid(*[[0, s - 1] for s in shape])).T.reshape(-1, 3)
    homogeneous = np.hstack([idx, np.ones((idx.shape[0], 1))])
    return (affine @ homogeneous.T).T[:, :3]


def _mni_like_affine() -> npt.NDArray[np.float64]:
    return np.array(
        [
            [-2.0, 0.0, 0.0, 96.0],
            [0.0, 2.0, 0.0, -132.0],
            [0.0, 0.0, 2.0, -78.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def test_bounding_box_preserved() -> None:
    shape = (97, 115, 97)
    img = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), _mni_like_affine())
    canonical = nib.as_closest_canonical(img)

    reoriented = reorient_to_radiological(img)

    before = _corners_world(canonical.affine, shape)
    after = _corners_world(reoriented.affine, shape)
    np.testing.assert_allclose(after.min(axis=0), before.min(axis=0), atol=1e-6)
    np.testing.assert_allclose(after.max(axis=0), before.max(axis=0), atol=1e-6)


def test_x_translation_not_shifted_by_full_extent() -> None:
    # The regression this pins: negating the direction cosine without compensating
    # the translation shifts the centroid by exactly (nx-1)*voxel_size_x — 192mm here.
    shape = (97, 115, 97)
    img = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), _mni_like_affine())
    canonical = nib.as_closest_canonical(img)

    reoriented = reorient_to_radiological(img)

    before_centroid = _corners_world(canonical.affine, shape).mean(axis=0)
    after_centroid = _corners_world(reoriented.affine, shape).mean(axis=0)
    np.testing.assert_allclose(after_centroid, before_centroid, atol=1e-6)


def test_voxel_data_untouched() -> None:
    shape = (10, 11, 12)
    rng = np.random.default_rng(0)
    data = rng.random(shape, dtype=np.float64).astype(np.float32)
    img = nib.Nifti1Image(data, _mni_like_affine())
    canonical = nib.as_closest_canonical(img)

    reoriented = reorient_to_radiological(img)

    np.testing.assert_array_equal(np.asarray(reoriented.dataobj), np.asarray(canonical.dataobj))


def test_x_direction_cosine_negated() -> None:
    img = nib.Nifti1Image(np.zeros((10, 11, 12), dtype=np.float32), _mni_like_affine())
    canonical = nib.as_closest_canonical(img)

    reoriented = reorient_to_radiological(img)

    np.testing.assert_allclose(reoriented.affine[:3, 0], -canonical.affine[:3, 0])
    np.testing.assert_allclose(reoriented.affine[:3, 1:3], canonical.affine[:3, 1:3])
