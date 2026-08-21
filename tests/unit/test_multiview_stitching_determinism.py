from __future__ import annotations

import numpy as np
import pytest

from traditional_dic._traditional_dic import clean_surface_outliers_cpp


def _fixed_surface_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a deterministic dense surface exercising the FLANN cleaner."""
    nx, ny = 24, 24
    points = np.asarray(
        [
            (float(x), float(y), 0.02 * np.sin(0.15 * x) * np.cos(0.12 * y))
            for y in range(ny)
            for x in range(nx)
        ],
        dtype=np.float64,
    )
    faces: list[tuple[int, int, int]] = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            i = y * nx + x
            faces.extend(((i, i + 1, i + nx), (i + 1, i + nx + 1, i + nx)))
    faces_array = np.asarray(faces, dtype=np.int64)
    displacement = np.column_stack(
        (
            0.01 * points[:, 0],
            -0.005 * points[:, 1],
            0.002 * np.sin(0.2 * points[:, 0] + 0.1 * points[:, 1]),
        )
    )
    return (
        np.ascontiguousarray(points),
        np.ascontiguousarray(points + displacement),
        np.ascontiguousarray(faces_array),
        np.ones(len(points), dtype=np.uint8),
    )


@pytest.mark.unit
def test_surface_cleaning_is_structurally_repeatable() -> None:
    reference, deformed, faces, initial_valid = _fixed_surface_fixture()
    outputs = [
        clean_surface_outliers_cpp(
            reference,
            deformed,
            faces,
            initial_valid,
            8,
            6.0,
            6.0,
            4.0,
        )
        for _ in range(5)
    ]

    first_points = np.asarray(outputs[0]["valid_points"], dtype=np.uint8)
    first_faces = np.asarray(outputs[0]["valid_faces"], dtype=np.uint8)
    for output in outputs[1:]:
        assert np.array_equal(first_points, np.asarray(output["valid_points"], dtype=np.uint8))
        assert np.array_equal(first_faces, np.asarray(output["valid_faces"], dtype=np.uint8))
        assert int(output["removed_points"]) == int(outputs[0]["removed_points"])
        assert int(output["removed_faces"]) == int(outputs[0]["removed_faces"])
