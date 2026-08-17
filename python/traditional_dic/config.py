"""Configuration helpers for Traditional-DIC Python workflows."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file as a dictionary."""
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def normalize_subset_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return a subset config dictionary accepted by the pybind subset API."""
    return deepcopy(config or {})


def normalize_mesh_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Translate YAML mesh config into the dictionary expected by pybind mesh."""
    src = deepcopy(config or {})
    out: dict[str, Any] = {}

    mesh = dict(src.get("mesh", {}) or {})
    optimization = dict(src.get("optimization", {}) or {})
    initialization = dict(src.get("initialization", {}) or {})
    interpolation = dict(src.get("interpolation", {}) or {})

    method = str(optimization.get("method", "fedic_element_icgn")).strip().lower()
    if method not in {"fedic_element_icgn", "fedic_element_fgn"}:
        raise ValueError(
            "Mesh optimization.method must be 'fedic_element_icgn' or 'fedic_element_fgn'."
        )
    objective = str(optimization.get("objective", "ssd")).strip().lower()
    if objective not in {"ssd", "znssd"}:
        raise ValueError("Mesh optimization.objective must be 'ssd' or 'znssd'.")

    out["mesh"] = {
        "max_iterations": optimization.get("max_iterations", mesh.get("max_iterations", 30)),
        "convergence_threshold": optimization.get(
            "convergence_threshold", mesh.get("convergence_threshold", 1.0e-3)
        ),
        "regularization_alpha": optimization.get("regularization_alpha", mesh.get("regularization_alpha", 0.0)),
        "mirror_image_padding": optimization.get("mirror_image_padding", mesh.get("mirror_image_padding", False)),
        "optimization_method": method,
        "photometric_objective": objective,
        "search_radius": initialization.get("fedic_fft", {}).get("search_radius", mesh.get("search_radius", 30)),
    }
    out["interpolation"] = interpolation
    out["initialization"] = {
        "method": initialization.get("method", "fedic_fft"),
        "boundary_interpolation_init": initialization.get("boundary_interpolation_init", True),
        "boundary_direct_prior_seed": initialization.get("boundary_direct_prior_seed", False),
        "fedic_fft": initialization.get("fedic_fft", {}),
        "pyramid": initialization.get("pyramid", {}),
        "sift_prior": initialization.get("sift_prior", {}),
        "quality_control": initialization.get("quality_control", {}),
    }
    return out


def _method_tag(value: Any, aliases: dict[str, str]) -> str:
    """Map a config value to a compact, filesystem-safe tag token."""
    key = str(value).strip().lower()
    return aliases.get(key, key)


def subset_method_tag(config: dict[str, Any] | None) -> str:
    """Build a compact tag from the subset algorithm settings."""
    cfg = config or {}
    criterion = _method_tag((cfg.get("correlation", {}) or {}).get("criterion", "znssd"), {})
    solver = _method_tag(
        (cfg.get("optimization", {}) or {}).get("method", "icgn"),
        {"fgn": "fgn", "forward_gauss_newton": "fgn"},
    )
    order = _method_tag(
        (cfg.get("shape_function", {}) or {}).get("order", "1"),
        {
            "first_order": "1st", "first": "1st", "1": "1st",
            "second_order": "2nd", "second": "2nd", "2": "2nd",
        },
    )
    return f"{criterion}_{solver}_{order}"


def mesh_method_tag(config: dict[str, Any] | None) -> str:
    """Build a compact tag from the mesh objective and solver method."""
    cfg = config or {}
    optimization = dict(cfg.get("optimization", {}) or {})
    objective = _method_tag(optimization.get("objective", "ssd"), {})
    solver = _method_tag(
        optimization.get("method", "fedic_element_icgn"),
        {"fedic_element_icgn": "icgn", "fedic_element_fgn": "fgn", "icgn": "icgn", "fgn": "fgn"},
    )
    return "_".join([objective, solver])


def mesh_generation_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Extract mesh-generation settings from a loaded config dictionary."""
    return deepcopy((config or {}).get("mesh_generation", {}) or {})
