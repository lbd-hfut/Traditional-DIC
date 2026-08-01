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

    out["mesh"] = {
        "max_iterations": optimization.get("max_iterations", mesh.get("max_iterations", 30)),
        "convergence_threshold": optimization.get(
            "convergence_threshold", mesh.get("convergence_threshold", 1.0e-3)
        ),
        "regularization_alpha": optimization.get("regularization_alpha", mesh.get("regularization_alpha", 0.0)),
        "mirror_image_padding": optimization.get("mirror_image_padding", mesh.get("mirror_image_padding", False)),
        "optimization_method": method,
        "search_radius": initialization.get("fedic_fft", {}).get("search_radius", mesh.get("search_radius", 30)),
    }
    out["interpolation"] = interpolation
    out["initialization"] = {
        "method": initialization.get("method", "fedic_fft"),
        "fedic_fft": initialization.get("fedic_fft", {}),
        "quality_control": initialization.get("quality_control", {}),
    }
    return out


def mesh_generation_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Extract mesh-generation settings from a loaded config dictionary."""
    return deepcopy((config or {}).get("mesh_generation", {}) or {})
