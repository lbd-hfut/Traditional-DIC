"""Visualization helpers for DIC fields and calibration products."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    from . import _traditional_dic as _backend

    _has_backend = True
except ImportError:
    _has_backend = False


def visualization_dir_for_result(case_root: str | Path, result_path: str | Path, *, result_root: str = "result") -> Path:
    """Map a case result path to the mirrored case/visualization path."""
    case_root = Path(case_root).resolve()
    result_path = Path(result_path).resolve()
    base = (case_root / result_root).resolve()
    try:
        rel = result_path.relative_to(base)
    except ValueError:
        rel = result_path.name if result_path.is_absolute() else result_path
    return case_root / "visualization" / rel


def _valid_percentile_limits(values: np.ndarray, *, clip_percentile: tuple[float, float] = (1.0, 99.0)) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lower, upper = np.percentile(finite, clip_percentile)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = float(np.min(finite))
        upper = float(np.max(finite))
    if upper <= lower:
        upper = lower + 1.0e-12
    return float(lower), float(upper)


def _field_component_values(field: Mapping[str, np.ndarray] | np.ndarray, component: str) -> tuple[np.ndarray, np.ndarray]:
    component = str(component).lower()
    if isinstance(field, Mapping):
        valid = np.asarray(field.get("valid", np.ones_like(field["x"], dtype=bool)), dtype=bool)
        xy = np.column_stack((np.asarray(field["x"], dtype=np.float64), np.asarray(field["y"], dtype=np.float64)))
        if component in {"u", "x", "ux"}:
            values = np.asarray(field["u"], dtype=np.float64)
        elif component in {"v", "y", "uy"}:
            values = np.asarray(field["v"], dtype=np.float64)
        elif component in {"mag", "magnitude", "umag"}:
            values = np.hypot(np.asarray(field["u"], dtype=np.float64), np.asarray(field["v"], dtype=np.float64))
        else:
            values = np.asarray(field[component], dtype=np.float64)
        return xy[valid], values[valid]
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("field array must have columns x, y, value or x, y, u, v")
    xy = arr[:, :2]
    if arr.shape[1] >= 4 and component in {"u", "x", "ux"}:
        values = arr[:, 2]
    elif arr.shape[1] >= 4 and component in {"v", "y", "uy"}:
        values = arr[:, 3]
    elif arr.shape[1] >= 4 and component in {"mag", "magnitude", "umag"}:
        values = np.hypot(arr[:, 2], arr[:, 3])
    else:
        values = arr[:, 2]
    finite = np.isfinite(xy).all(axis=1) & np.isfinite(values)
    return xy[finite], values[finite]


def _mesh_edge_segments(
    nodes: np.ndarray,
    elements: np.ndarray,
    element_type: str,
) -> list[np.ndarray]:
    """Return unique boundary segments for 2D T3, Q4, or Q8 elements."""
    edge_ids = {
        "T3": ((0, 1), (1, 2), (2, 0)),
        "Q4": ((0, 1), (1, 2), (2, 3), (3, 0)),
        "Q8": ((0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 3), (3, 7), (7, 0)),
    }
    try:
        local_edges = edge_ids[str(element_type).upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported 2D mesh element type: {element_type}") from exc

    points = np.asarray(nodes, dtype=np.float64).reshape((-1, 2))
    connectivity = np.asarray(elements, dtype=np.int64)
    segments: list[np.ndarray] = []
    seen: set[tuple[int, int]] = set()
    for element in connectivity:
        for a, b in local_edges:
            if a >= len(element) or b >= len(element):
                continue
            node_a, node_b = int(element[a]), int(element[b])
            if node_a < 0 or node_b < 0 or node_a >= len(points) or node_b >= len(points):
                continue
            key = tuple(sorted((node_a, node_b)))
            if key not in seen:
                seen.add(key)
                segments.append(points[[node_a, node_b]])
    return segments


def _mesh_shape_functions(element_type: str, xi: float, eta: float) -> np.ndarray:
    element_type = str(element_type).upper()
    if element_type == "T3":
        return np.array((1.0 - xi - eta, xi, eta), dtype=np.float64)
    if element_type == "Q4":
        return 0.25 * np.array(
            (
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ),
            dtype=np.float64,
        )
    if element_type == "Q8":
        return np.array(
            (
                -0.25 * (1.0 - xi) * (1.0 - eta) * (1.0 + xi + eta),
                -0.25 * (1.0 + xi) * (1.0 - eta) * (1.0 - xi + eta),
                -0.25 * (1.0 + xi) * (1.0 + eta) * (1.0 - xi - eta),
                -0.25 * (1.0 - xi) * (1.0 + eta) * (1.0 + xi - eta),
                0.5 * (1.0 - xi * xi) * (1.0 - eta),
                0.5 * (1.0 + xi) * (1.0 - eta * eta),
                0.5 * (1.0 - xi * xi) * (1.0 + eta),
                0.5 * (1.0 - xi) * (1.0 - eta * eta),
            ),
            dtype=np.float64,
        )
    raise ValueError(f"unsupported 2D mesh element type: {element_type}")


def densify_2d_mesh_displacement_field(
    nodes: np.ndarray,
    elements: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    element_type: str,
    *,
    valid: np.ndarray | None = None,
    samples_per_axis: int = 17,
) -> dict[str, np.ndarray]:
    """Interpolate nodal mesh displacements to dense points with shape functions."""
    element_type = str(element_type).upper()
    points = np.asarray(nodes, dtype=np.float64).reshape((-1, 2))
    connectivity = np.asarray(elements, dtype=np.int64)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if len(u) != len(points) or len(v) != len(points):
        raise ValueError("nodal displacement arrays must match the mesh node count")
    valid_nodes = np.isfinite(u) & np.isfinite(v)
    if valid is not None:
        valid_nodes &= np.asarray(valid, dtype=bool).reshape(-1)

    if element_type == "T3":
        divisions = max(1, samples_per_axis - 3)
        parameters = [(i / divisions, j / divisions) for i in range(divisions + 1) for j in range(divisions + 1 - i)]
    elif element_type in {"Q4", "Q8"}:
        coordinates = np.linspace(-1.0, 1.0, max(2, samples_per_axis))
        parameters = [(xi, eta) for xi in coordinates for eta in coordinates]
    else:
        raise ValueError(f"unsupported 2D mesh element type: {element_type}")

    dense_xy: list[np.ndarray] = []
    dense_uv: list[np.ndarray] = []
    for element in connectivity:
        if np.any(element < 0) or np.any(element >= len(points)) or not np.all(valid_nodes[element]):
            continue
        element_xy = points[element]
        element_uv = np.column_stack((u[element], v[element]))
        for xi, eta in parameters:
            shape = _mesh_shape_functions(element_type, xi, eta)
            if len(shape) != len(element):
                continue
            dense_xy.append(shape @ element_xy)
            dense_uv.append(shape @ element_uv)

    if not dense_xy:
        empty = np.empty(0, dtype=np.float64)
        return {"x": empty, "y": empty, "u": empty, "v": empty, "valid": np.empty(0, dtype=bool)}
    xy = np.asarray(dense_xy, dtype=np.float64)
    uv = np.asarray(dense_uv, dtype=np.float64)
    return {"x": xy[:, 0], "y": xy[:, 1], "u": uv[:, 0], "v": uv[:, 1], "valid": np.ones(len(xy), dtype=bool)}


def plot_2d_field_overlay(
    reference_image: str | Path | np.ndarray,
    field: Mapping[str, np.ndarray] | np.ndarray,
    out_path: str | Path,
    *,
    component: str = "mag",
    title: str | None = None,
    label: str | None = None,
    cmap: str = "jet",
    alpha: float = 0.8,
    point_size: float = 7.0,
    clip_percentile: tuple[float, float] = (1.0, 99.0),
    mesh_nodes: np.ndarray | None = None,
    mesh_elements: np.ndarray | None = None,
    mesh_element_type: str | None = None,
) -> Path:
    """Overlay a 2D scalar field and optional mesh edges on the reference image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from PIL import Image

    if isinstance(reference_image, np.ndarray):
        image = np.asarray(reference_image)
    else:
        image = np.asarray(Image.open(reference_image).convert("L"))
    xy, values = _field_component_values(field, component)
    vmin, vmax = _valid_percentile_limits(values, clip_percentile=clip_percentile)
    if str(component).lower() in {"mag", "magnitude", "umag"}:
        vmin = min(0.0, vmin)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6.0, image.shape[1] / 180.0), max(4.0, image.shape[0] / 180.0)), dpi=160)
    ax.imshow(image, cmap="gray")
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=point_size, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, alpha=alpha)
    if mesh_nodes is not None or mesh_elements is not None or mesh_element_type is not None:
        if mesh_nodes is None or mesh_elements is None or mesh_element_type is None:
            raise ValueError("mesh_nodes, mesh_elements, and mesh_element_type must be provided together")
        segments = _mesh_edge_segments(mesh_nodes, mesh_elements, mesh_element_type)
        ax.add_collection(LineCollection(segments, colors="gray", linewidths=1.0))
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_aspect("equal")
    ax.set_title(title or f"{component} field")
    ax.axis("off")
    fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label=label or str(component))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_3d_scatter_field(
    points: np.ndarray,
    values: np.ndarray,
    out_path: str | Path,
    *,
    title: str = "3D field",
    label: str = "value",
    cmap: str = "jet",
    max_points: int | None = 120000,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(values)
    points = points[valid]
    values = values[valid]
    if max_points is not None and len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, int(max_points), dtype=np.int64)
        points = points[idx]
        values = values[idx]
    vmin, vmax = _valid_percentile_limits(values)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=values, s=2.0, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
    _set_equal_3d_limits(ax, points)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.1, label=label)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_3d_surface_field(
    points: np.ndarray,
    faces: np.ndarray,
    point_values: np.ndarray,
    out_path: str | Path,
    *,
    title: str = "3D surface field",
    label: str = "value",
    cmap: str = "jet",
    max_faces: int | None = 70000,
) -> Path:
    """Render a triangular surface field; C++ prepares face values when available."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    faces = np.asarray(faces, dtype=np.int64).reshape((-1, 3))
    point_values = np.asarray(point_values, dtype=np.float64).reshape(-1)
    prepared = _prepare_surface_field(points, faces, point_values)
    render_faces = np.asarray(prepared["faces"], dtype=np.int64)
    face_values = np.asarray(prepared["face_values"], dtype=np.float64)
    valid_faces = np.asarray(prepared["valid_faces"], dtype=np.uint8).astype(bool)
    render_faces = render_faces[valid_faces]
    face_values = face_values[valid_faces]
    if max_faces is not None and len(render_faces) > int(max_faces):
        idx = np.linspace(0, len(render_faces) - 1, int(max_faces), dtype=np.int64)
        render_faces = render_faces[idx]
        face_values = face_values[idx]

    vmin, vmax = _valid_percentile_limits(face_values)
    normalizer = Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = plt.cm.ScalarMappable(norm=normalizer, cmap=cmap)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    collection = Poly3DCollection(points[render_faces], facecolors=mapper.to_rgba(face_values), edgecolors="none", linewidths=0.0)
    ax.add_collection3d(collection)
    _set_equal_3d_limits(ax, points[np.unique(render_faces)] if len(render_faces) else points)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.colorbar(mapper, ax=ax, shrink=0.72, pad=0.1, label=label)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _prepare_surface_field(points: np.ndarray, faces: np.ndarray, point_values: np.ndarray) -> dict[str, np.ndarray]:
    if _has_backend and hasattr(_backend, "visualization"):
        return {key: np.asarray(value) for key, value in _backend.visualization.prepare_surface_field(points, faces, point_values).items()}
    valid = (
        (faces >= 0).all(axis=1)
        & (faces < len(points)).all(axis=1)
        & np.isfinite(points[faces]).all(axis=(1, 2))
        & np.isfinite(point_values[faces]).all(axis=1)
    )
    return {
        "faces": faces,
        "face_centers": np.where(valid[:, None], points[faces].mean(axis=1), np.nan),
        "face_values": np.where(valid, point_values[faces].mean(axis=1), np.nan),
        "valid_faces": valid.astype(np.uint8),
    }


def _set_equal_3d_limits(ax, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size == 0:
        finite = np.zeros((1, 3), dtype=np.float64)
    lo = finite.min(axis=0)
    hi = finite.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = max(float(np.max(hi - lo)) * 0.55, 1.0e-12)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def plot_stitched_surface_fields(result, output_dir: str | Path, *, max_faces: int | None = 70000) -> None:
    """Write MultiDIC-style stitched surface plots."""
    output_dir = Path(output_dir)
    reference = np.asarray(result.reference, dtype=np.float64)
    deformed = np.asarray(result.deformed, dtype=np.float64)
    faces = np.asarray(result.faces, dtype=np.int64)
    displacement = deformed - reference
    magnitude = np.linalg.norm(displacement, axis=1)
    plot_3d_surface_field(reference, faces, reference[:, 2], output_dir / "stitched_reference_scene.png", title="Stitched reference surface", label="Z", cmap="viridis", max_faces=max_faces)
    plot_3d_surface_field(deformed, faces, deformed[:, 2], output_dir / "stitched_deformed_scene.png", title="Stitched deformed surface", label="Z", cmap="viridis", max_faces=max_faces)
    plot_3d_surface_field(reference, faces, magnitude, output_dir / "stitched_displacement_umag.png", title="Stitched displacement magnitude", label="Umag", cmap="jet", max_faces=max_faces)
    for idx, name in enumerate(("ux", "uy", "uz")):
        plot_3d_surface_field(reference, faces, displacement[:, idx], output_dir / f"stitched_displacement_{name}.png", title=f"Stitched displacement {name.upper()}", label=name.upper(), cmap="jet", max_faces=max_faces)
