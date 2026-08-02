"""Post-processing helpers backed by the C++ numerical implementation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import _traditional_dic as _backend


def save_least_squares_strain_csv(path, points, displacement, *, radius=None, elements=None, min_samples=6, green_lagrange=True):
    points = np.asarray(points, dtype=np.float64)
    displacement = np.asarray(displacement, dtype=np.float64)
    if elements is None:
        if radius is None:
            raise ValueError("radius is required for subset least-squares strain")
        strains = _backend.postprocess.compute_least_squares_strain_2d(points, displacement, float(radius), int(min_samples), bool(green_lagrange))
    else:
        strains = _backend.postprocess.compute_mesh_least_squares_strain_2d(points, displacement, np.asarray(elements, dtype=np.int32), int(min_samples), bool(green_lagrange))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "du_dx", "du_dy", "dv_dx", "dv_dy", "exx", "eyy", "exy", "sample_count", "valid"])
        for point, strain in zip(points, strains):
            writer.writerow([point[0], point[1], strain.du_dx, strain.du_dy, strain.dv_dx, strain.dv_dy, strain.exx, strain.eyy, strain.exy, strain.sample_count, int(strain.valid)])
