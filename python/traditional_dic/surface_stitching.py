"""MultiDIC-style pair-surface overlap removal and boundary stitching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SurfaceMesh:
    reference: np.ndarray
    deformed: np.ndarray
    faces: np.ndarray
    quality: np.ndarray
    pair_index: int
    pair_name: str


@dataclass
class StitchResult:
    reference: np.ndarray
    deformed: np.ndarray
    faces: np.ndarray
    point_pair_indices: np.ndarray
    face_pair_indices: np.ndarray
    face_quality: np.ndarray
    overlap_removed_faces: int
    zipper_faces: int
    hole_faces: int


@dataclass
class SurfaceCleanResult:
    result: StitchResult
    valid_points: np.ndarray
    valid_faces: np.ndarray
    removed_points: int
    removed_faces: int


def _mean_edge_length(vertices: np.ndarray, faces: np.ndarray) -> float:
    if len(faces) == 0:
        return 0.0
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    finite = lengths[np.isfinite(lengths) & (lengths > 0)]
    return float(np.mean(finite)) if len(finite) else 0.0


def _boundary_edges(faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    undirected = np.sort(edges, axis=1)
    unique, counts = np.unique(undirected, axis=0, return_counts=True)
    return unique[counts == 1]


def _boundary_loops(faces: np.ndarray) -> list[np.ndarray]:
    edges = _boundary_edges(faces)
    if len(edges) == 0:
        return []
    adjacency: dict[int, list[int]] = {}
    for a, b in edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    unused = {tuple(sorted((int(a), int(b)))) for a, b in edges}
    loops: list[np.ndarray] = []
    while unused:
        a, b = next(iter(unused))
        loop = [a, b]
        unused.discard((a, b))
        previous, current = a, b
        while True:
            candidates = [v for v in adjacency.get(current, []) if v != previous]
            next_vertex = next((v for v in candidates if tuple(sorted((current, v))) in unused), None)
            if next_vertex is None:
                break
            unused.discard(tuple(sorted((current, next_vertex))))
            loop.append(next_vertex)
            previous, current = current, next_vertex
            if current == loop[0]:
                break
        if len(loop) >= 3:
            loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def _remove_overlap_faces(
    vertices1: np.ndarray,
    faces1: np.ndarray,
    quality1: np.ndarray,
    vertices2: np.ndarray,
    faces2: np.ndarray,
    quality2: np.ndarray,
    min_gap_factor: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Port the MultiDIC boundary-removal policy using boundary proximity.

    MultiDIC ray-traces boundary vertices onto the other surface. The pair
    surfaces here are sampled dense point fields, so nearest boundary samples
    are the equivalent stable test and avoid an additional mesh-ray library.
    """
    from scipy.spatial import cKDTree

    keep1 = np.ones(len(faces1), dtype=bool)
    keep2 = np.ones(len(faces2), dtype=bool)
    mean_edge = np.mean([_mean_edge_length(vertices1, faces1), _mean_edge_length(vertices2, faces2)])
    min_gap = max(float(min_gap_factor) * mean_edge, 1.0e-9)
    removed = 0
    # MultiDIC removes one selected boundary region at a time.  Batch the
    # same quality-ranked decision here because rebuilding boundary loops for
    # every individual face is prohibitively expensive for DIC-sized meshes.
    for _ in range(5):
        active1 = faces1[keep1]
        active2 = faces2[keep2]
        loops1 = _boundary_loops(active1)
        loops2 = _boundary_loops(active2)
        boundary1 = np.unique(np.concatenate(loops1)) if loops1 else np.zeros(0, dtype=np.int64)
        boundary2 = np.unique(np.concatenate(loops2)) if loops2 else np.zeros(0, dtype=np.int64)
        if len(boundary1) == 0 or len(boundary2) == 0:
            break
        tree2 = cKDTree(vertices2[boundary2])
        tree1 = cKDTree(vertices1[boundary1])
        dist1, _ = tree2.query(vertices1[boundary1], k=1)
        dist2, _ = tree1.query(vertices2[boundary2], k=1)
        close_threshold = max(2.0 * min_gap, 0.75 * mean_edge)
        close1 = boundary1[dist1 < close_threshold]
        close2 = boundary2[dist2 < close_threshold]
        if len(close1) == 0 and len(close2) == 0:
            break
        score1 = np.full(len(vertices1), -np.inf)
        score2 = np.full(len(vertices2), -np.inf)
        for face, q in zip(faces1[keep1], quality1[keep1]):
            score1[face] = np.maximum(score1[face], float(q))
        for face, q in zip(faces2[keep2], quality2[keep2]):
            score2[face] = np.maximum(score2[face], float(q))
        worst1 = float(np.max(score1[close1])) if len(close1) else -np.inf
        worst2 = float(np.max(score2[close2])) if len(close2) else -np.inf
        if worst1 >= worst2 and len(close1):
            candidates = close1[np.argsort(score1[close1])[::-1]]
            faces = faces1
            keep = keep1
        elif len(close2):
            candidates = close2[np.argsort(score2[close2])[::-1]]
            faces = faces2
            keep = keep2
        else:
            break
        batch_size = min(256, max(8, len(candidates) // 100))
        selected = candidates[:batch_size]
        remove = np.any(faces[:, :, None] == selected[None, None, :], axis=(1, 2)) & keep
        if not np.any(remove):
            break
        keep[remove] = False
        removed += int(np.count_nonzero(remove))
    return keep1, keep2, removed


def _zip_boundaries(
    vertices1: np.ndarray,
    faces1: np.ndarray,
    vertices2: np.ndarray,
    faces2: np.ndarray,
    min_gap_factor: float,
) -> np.ndarray:
    """Create Delaunay-style bridge faces between the closest boundary loops."""
    from scipy.spatial import Delaunay, cKDTree

    loops1 = _boundary_loops(faces1)
    loops2 = _boundary_loops(faces2)
    if not loops1 or not loops2:
        return np.zeros((0, 3), dtype=np.int64)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for loop1 in loops1:
        for candidate2 in loops2:
            d = float(np.mean(cKDTree(vertices2[candidate2]).query(vertices1[loop1], k=1)[0]))
            if best is None or d < best[0]:
                best = (d, loop1, candidate2)
    if best is None:
        return np.zeros((0, 3), dtype=np.int64)
    distance, loop1, loop2 = best
    mean_edge = np.mean([_mean_edge_length(vertices1, faces1), _mean_edge_length(vertices2, faces2)])
    if distance > 3.0 * max(mean_edge, float(min_gap_factor) * mean_edge, 1.0e-9):
        return np.zeros((0, 3), dtype=np.int64)
    points = np.vstack((vertices1[loop1], vertices2[loop2]))
    if len(points) < 3:
        return np.zeros((0, 3), dtype=np.int64)
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh[:2].T
    try:
        simplices = Delaunay(projected).simplices
    except Exception:
        return np.zeros((0, 3), dtype=np.int64)
    split = len(loop1)
    faces: list[list[int]] = []
    max_edge = 4.0 * max(mean_edge, 1.0e-9)
    for simplex in simplices:
        labels = simplex < split
        if labels.all() or (~labels).all():
            continue
        tri = simplex.astype(np.int64)
        tri_points = points[tri]
        edges = np.linalg.norm(tri_points - np.roll(tri_points, 1, axis=0), axis=1)
        if np.all(edges <= max_edge):
            tri = np.where(tri < split, tri, tri + len(vertices1))
            faces.append(tri.tolist())
    return np.asarray(faces, dtype=np.int64).reshape((-1, 3))


def stitch_surfaces(meshes: list[SurfaceMesh], min_gap_factor: float = 0.2) -> StitchResult:
    if not meshes:
        raise ValueError("At least one surface is required")
    current = meshes[0]
    point_pair = np.full(len(current.reference), current.pair_index, dtype=np.int64)
    face_pair = np.full(len(current.faces), current.pair_index, dtype=np.int64)
    face_quality = current.quality[current.faces].mean(axis=1)
    overlap_removed = 0
    zipper_faces = 0
    for incoming in meshes[1:]:
        incoming_face_quality = incoming.quality[incoming.faces].mean(axis=1)
        keep_current, keep_incoming, removed = _remove_overlap_faces(
            current.reference,
            current.faces,
            face_quality,
            incoming.reference,
            incoming.faces,
            incoming_face_quality,
            min_gap_factor,
        )
        overlap_removed += removed
        current.faces = current.faces[keep_current]
        face_pair = face_pair[keep_current]
        face_quality = face_quality[keep_current]
        incoming_faces = incoming.faces[keep_incoming]
        bridge = _zip_boundaries(current.reference, current.faces, incoming.reference, incoming_faces, min_gap_factor)
        offset = len(current.reference)
        current.faces = np.vstack((current.faces, incoming_faces + offset, bridge))
        face_pair = np.concatenate((face_pair, np.full(len(incoming_faces), incoming.pair_index), np.full(len(bridge), -1)))
        face_quality = np.concatenate((face_quality, incoming_face_quality[keep_incoming], np.full(len(bridge), np.nan)))
        zipper_faces += len(bridge)
        current.reference = np.vstack((current.reference, incoming.reference))
        current.deformed = np.vstack((current.deformed, incoming.deformed))
        current.quality = np.concatenate((current.quality, incoming.quality))
        point_pair = np.concatenate((point_pair, np.full(len(incoming.reference), incoming.pair_index)))
    return StitchResult(
        reference=current.reference,
        deformed=current.deformed,
        faces=current.faces,
        point_pair_indices=point_pair,
        face_pair_indices=face_pair,
        face_quality=face_quality,
        overlap_removed_faces=overlap_removed,
        zipper_faces=zipper_faces,
        hole_faces=0,
    )


def _clean_stitched_surface_python(
    result: StitchResult,
    *,
    neighbor_count: int = 8,
    distance_sigma: float = 6.0,
    displacement_sigma: float = 6.0,
    face_edge_scale: float = 4.0,
) -> SurfaceCleanResult:
    """Remove invalid, geometrically isolated, and displacement-outlier points."""
    from scipy.spatial import cKDTree

    reference = np.asarray(result.reference, dtype=np.float64)
    deformed = np.asarray(result.deformed, dtype=np.float64)
    faces = np.asarray(result.faces, dtype=np.int64)
    finite_points = np.all(np.isfinite(reference), axis=1) & np.all(np.isfinite(deformed), axis=1)
    valid_points = finite_points.copy()
    point_indices = np.flatnonzero(finite_points)
    if len(point_indices) > neighbor_count:
        tree = cKDTree(reference[point_indices])
        distances, _ = tree.query(reference[point_indices], k=neighbor_count + 1)
        local_scale = distances[:, -1]
        center = float(np.median(local_scale))
        mad = float(np.median(np.abs(local_scale - center)))
        threshold = center + float(distance_sigma) * max(1.4826 * mad, center * 0.25, 1.0e-9)
        valid_points[point_indices[local_scale > threshold]] = False

        # A point can have normal spatial density while still carrying an
        # implausible displacement. Compare it with the robust local median
        # displacement, independently of the source valid flag.
        displacement = deformed - reference
        neighbor_indices = tree.query(reference[point_indices], k=neighbor_count + 1)[1][:, 1:]
        neighbor_displacement = displacement[point_indices[neighbor_indices]]
        local_median_displacement = np.median(neighbor_displacement, axis=1)
        displacement_residual = np.linalg.norm(
            displacement[point_indices] - local_median_displacement,
            axis=1,
        )
        residual_center = float(np.median(displacement_residual))
        residual_mad = float(np.median(np.abs(displacement_residual - residual_center)))
        residual_threshold = residual_center + float(displacement_sigma) * max(
            1.4826 * residual_mad,
            residual_center * 0.5,
            1.0e-9,
        )
        valid_points[point_indices[displacement_residual > residual_threshold]] = False

    face_in_bounds = (faces >= 0).all(axis=1) & (faces < len(reference)).all(axis=1)
    face_finite = np.zeros(len(faces), dtype=bool)
    face_finite[face_in_bounds] = np.all(np.isfinite(reference[faces[face_in_bounds]]), axis=(1, 2))
    face_vertices_valid = np.zeros(len(faces), dtype=bool)
    face_vertices_valid[face_in_bounds] = np.all(valid_points[faces[face_in_bounds]], axis=1)
    candidate_faces = face_in_bounds & face_finite & face_vertices_valid
    if np.any(candidate_faces):
        candidate = faces[candidate_faces]
        edges = np.vstack((candidate[:, [0, 1]], candidate[:, [1, 2]], candidate[:, [2, 0]]))
        lengths = np.linalg.norm(reference[edges[:, 0]] - reference[edges[:, 1]], axis=1)
        finite_lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
        typical_edge = float(np.median(finite_lengths)) if len(finite_lengths) else 0.0
        edge_ok = np.max(lengths.reshape(3, -1), axis=0) <= float(face_edge_scale) * max(typical_edge, 1.0e-9)
        face_indices = np.flatnonzero(candidate_faces)
        candidate_faces[face_indices[~edge_ok]] = False

    # Remove vertices that no longer belong to any valid face from rendering.
    used_points = np.zeros(len(reference), dtype=bool)
    if np.any(candidate_faces):
        used_points[np.unique(faces[candidate_faces])] = True
    valid_points &= used_points
    valid_faces = candidate_faces.copy()
    valid_face_indices = np.flatnonzero(candidate_faces)
    if len(valid_face_indices):
        valid_faces[valid_face_indices] = np.all(valid_points[faces[valid_face_indices]], axis=1)
    cleaned = StitchResult(
        reference=reference,
        deformed=deformed,
        faces=faces[valid_faces],
        point_pair_indices=result.point_pair_indices,
        face_pair_indices=result.face_pair_indices[valid_faces],
        face_quality=result.face_quality[valid_faces],
        overlap_removed_faces=result.overlap_removed_faces,
        zipper_faces=result.zipper_faces,
        hole_faces=result.hole_faces,
    )
    return SurfaceCleanResult(
        result=cleaned,
        valid_points=valid_points,
        valid_faces=valid_faces,
        removed_points=int(np.count_nonzero(~valid_points)),
        removed_faces=int(np.count_nonzero(~valid_faces)),
    )


def clean_stitched_surface(
    result: StitchResult,
    *,
    neighbor_count: int = 8,
    distance_sigma: float = 6.0,
    displacement_sigma: float = 6.0,
    face_edge_scale: float = 4.0,
) -> SurfaceCleanResult:
    """Clean a stitched field with the C++ backend, falling back to SciPy."""
    try:
        from ._traditional_dic import clean_surface_outliers_cpp
    except ImportError:
        return _clean_stitched_surface_python(
            result,
            neighbor_count=neighbor_count,
            distance_sigma=distance_sigma,
            displacement_sigma=displacement_sigma,
            face_edge_scale=face_edge_scale,
        )

    reference = np.asarray(result.reference, dtype=np.float64)
    deformed = np.asarray(result.deformed, dtype=np.float64)
    faces = np.asarray(result.faces, dtype=np.int64)
    initial_valid = np.all(np.isfinite(reference), axis=1) & np.all(np.isfinite(deformed), axis=1)
    output = clean_surface_outliers_cpp(
        np.ascontiguousarray(reference),
        np.ascontiguousarray(deformed),
        np.ascontiguousarray(faces),
        np.ascontiguousarray(initial_valid.astype(np.uint8)),
        int(neighbor_count),
        float(distance_sigma),
        float(displacement_sigma),
        float(face_edge_scale),
    )
    valid_points = np.asarray(output["valid_points"], dtype=np.uint8).astype(bool)
    valid_faces = np.asarray(output["valid_faces"], dtype=np.uint8).astype(bool)
    cleaned = StitchResult(
        reference=reference,
        deformed=deformed,
        faces=faces[valid_faces],
        point_pair_indices=result.point_pair_indices,
        face_pair_indices=result.face_pair_indices[valid_faces],
        face_quality=result.face_quality[valid_faces],
        overlap_removed_faces=result.overlap_removed_faces,
        zipper_faces=result.zipper_faces,
        hole_faces=result.hole_faces,
    )
    return SurfaceCleanResult(
        result=cleaned,
        valid_points=valid_points,
        valid_faces=valid_faces,
        removed_points=int(output["removed_points"]),
        removed_faces=int(output["removed_faces"]),
    )


def write_stitch_visualizations(result: StitchResult, output_dir, *, max_faces: int | None = 70000) -> None:
    """Write MultiDIC-style triangular surface and face-measure views."""
    from .visualization import plot_stitched_surface_fields

    plot_stitched_surface_fields(result, output_dir, max_faces=max_faces)
