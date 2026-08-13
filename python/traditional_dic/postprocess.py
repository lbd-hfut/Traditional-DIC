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


def save_surface_strain_csv(path, faces, points_ref, points_def, valid_faces=None, min_face_area=0.0):
    """Write triangular 3D strain for an already reconstructed surface."""
    faces = np.asarray(faces, dtype=np.int32)
    points_ref = np.asarray(points_ref, dtype=np.float64)
    points_def = np.asarray(points_def, dtype=np.float64)
    if valid_faces is None:
        valid_faces = np.ones(len(faces), dtype=bool)
    if min_face_area > 0.0:
        areas = 0.5 * np.linalg.norm(
            np.cross(points_ref[faces[:, 1]] - points_ref[faces[:, 0]], points_ref[faces[:, 2]] - points_ref[faces[:, 0]]), axis=1
        )
        valid_faces &= areas >= float(min_face_area)
    strains = _backend.postprocess.compute_surface_strain(
        faces, points_ref, points_def,
        np.asarray(valid_faces, dtype=bool),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["face_id", "n1", "n2", "n3", "area", "Epc1", "Epc2", "Eeq", "EShearMax", "epc1", "epc2", "eeq", "valid"])
        for index, (face, strain) in enumerate(zip(faces, strains), start=1):
            writer.writerow([index, *(int(node) + 1 for node in face), strain.area, strain.Epc1, strain.Epc2, strain.Eeq, strain.EShearMax, strain.epc1, strain.epc2, strain.eeq, int(strain.valid)])
