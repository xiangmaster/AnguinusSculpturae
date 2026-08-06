"""Shared NIfTI reading helper for the MAMA-SYNTH dataset builders.

Copied verbatim from ``src/preprocessing/extract_tcga_tiff_pairs.py`` in the companion
``mama-synth`` repository, which is the only function the blosc2 builders needed from
that module. It lives here so this package has no dependency on the rest of that file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np


def load_nifti_float32(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load a NIfTI image as float32 and return array plus 3D voxel spacing."""
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    return data, spacing
