"""NIfTI reading shared by the MAMA-SYNTH dataset builders.

Loads a volume as float32 together with its 3D voxel spacing, which is what the builders
need to pick the through-plane axis and record in-plane spacing.
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
