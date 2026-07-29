"""2D Mesh-DIC Python API."""

from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    from . import _traditional_dic as _backend
    from .config import load_config, normalize_mesh_config
    _has_backend = True
except ImportError:
    _has_backend = False


def _load_config(config):
    if config is None:
        return None
    if isinstance(config, (str, Path)):
        return normalize_mesh_config(load_config(config))
    if isinstance(config, dict):
        return normalize_mesh_config(config)
    raise TypeError(f"config must be str, Path, dict, or None, got {type(config)}")


def mesh(
    reference: np.ndarray,
    deformed: np.ndarray,
    nodes: np.ndarray,
    elements: np.ndarray,
    element_type: str = "Q4",
    config: Optional[Union[str, Path, dict]] = None,
    solver: str = "icgn",
    criterion: str = "ssd",
    initialization: str = "integer_search",
    bspline_degree: int = 5,
    max_iterations: int = 30,
    convergence_threshold: float = 1e-3,
    search_radius: int = 20,
    regularization_alpha: float = 0.0,
    one_based: bool = False,
):
    """Run 2D Mesh-DIC from numpy images and mesh arrays."""
    if not _has_backend:
        raise ImportError(
            "C++ backend _traditional_dic not found. "
            "Build with -DTRADITIONAL_DIC_BUILD_PYTHON=ON."
        )
    if not isinstance(reference, np.ndarray) or reference.ndim != 2:
        raise ValueError("reference must be a 2D numpy array")
    if not isinstance(deformed, np.ndarray) or deformed.ndim != 2:
        raise ValueError("deformed must be a 2D numpy array")
    if reference.shape != deformed.shape:
        raise ValueError("reference and deformed must have the same shape")
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes must have shape (n_nodes, 2)")
    if elements.ndim != 2:
        raise ValueError("elements must have shape (n_elements, nodes_per_element)")

    config_dict = _load_config(config)
    if config_dict is None:
        config_dict = {
            "mesh": {
                "solver": solver,
                "criterion": criterion,
                "max_iterations": max_iterations,
                "convergence_threshold": convergence_threshold,
                "search_radius": search_radius,
                "regularization_alpha": regularization_alpha,
            },
            "interpolation": {
                "degree": bspline_degree,
            },
            "initialization": {
                "method": initialization,
                "integer_search": {
                    "search_radius": search_radius,
                },
            },
        }

    return _backend.mesh.compute(
        reference,
        deformed,
        nodes,
        elements,
        element_type,
        config_dict,
        one_based,
    )


def generate_mesh_from_roi(
    roi: np.ndarray,
    element_type: str = "Q4",
    target_element_size: float = 20.0,
    min_element_size: float = 0.0,
    max_element_size: float = 0.0,
    min_element_quality: float = 0.1,
    config: Optional[dict] = None,
):
    """Generate one annulus mesh dictionary from an ROI mask."""
    if not _has_backend:
        raise ImportError("C++ backend _traditional_dic not found.")
    generation_config = dict(config or {})
    generation_config.setdefault("target_element_size", target_element_size)
    generation_config.setdefault("min_element_size", min_element_size)
    generation_config.setdefault("max_element_size", max_element_size)
    generation_config.setdefault("min_element_quality", min_element_quality)
    return _backend.mesh.generate_mesh_from_roi(
        np.asarray(roi, dtype=np.uint8),
        element_type,
        generation_config,
    )


def generate_annulus_meshes_from_mask(
    roi: np.ndarray,
    target_element_size: float = 35.0,
    min_element_size: float = 18.0,
    max_element_size: float = 55.0,
    min_element_quality: float = 0.1,
    config: Optional[dict] = None,
):
    """Generate T3/Q4/Q8 annulus meshes from an ROI mask through the C++ backend."""
    if not _has_backend:
        raise ImportError("C++ backend _traditional_dic not found.")
    generation_config = dict(config or {})
    generation_config.setdefault("target_element_size", target_element_size)
    generation_config.setdefault("min_element_size", min_element_size)
    generation_config.setdefault("max_element_size", max_element_size)
    generation_config.setdefault("min_element_quality", min_element_quality)
    return _backend.mesh.generate_annulus_meshes_from_mask(
        np.asarray(roi, dtype=np.uint8),
        generation_config,
    )
