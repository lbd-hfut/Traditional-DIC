"""Canonical Traditional-DIC workflow capability contract.

Transport adapters (CLI and MCP) import this module rather than maintaining
independent solver/capability tables.  The F4 manifest projection is kept
separate only to preserve its established serialized shape.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


WORKFLOW_NAMES: dict[str, str] = {
    "subset-2d": "subset_2d",
    "mesh-2d": "mesh_2d",
    "stereo-3d": "stereo_3d",
    "multiview-3d": "multiview_3d",
}
WORKFLOW_DISPLAY = {value: key for key, value in WORKFLOW_NAMES.items()}
WORKFLOW_KINDS = tuple(WORKFLOW_NAMES.values())

CAPABILITY_CONTRACT: dict[str, Any] = {
    "subset_2d": {"supported": True, "solver": "subset"},
    "mesh_2d": {"supported": True, "solver": "mesh"},
    "stereo_3d": {"supported": True, "solver": "subset"},
    "multiview_3d": {"supported": True, "solver": "subset"},
}


def capability_contract() -> dict[str, Any]:
    """Return a detached machine-readable capability mapping."""
    return deepcopy(CAPABILITY_CONTRACT)


def f4_capability_contract() -> dict[str, Any]:
    """Return the established F4 manifest projection without changing shape."""
    return {
        "subset_2d": True,
        "mesh_2d": True,
        "stereo_3d_solver": CAPABILITY_CONTRACT["stereo_3d"]["solver"],
        "multiview_3d_solver": CAPABILITY_CONTRACT["multiview_3d"]["solver"],
    }


__all__ = [
    "CAPABILITY_CONTRACT",
    "WORKFLOW_DISPLAY",
    "WORKFLOW_KINDS",
    "WORKFLOW_NAMES",
    "capability_contract",
    "f4_capability_contract",
]
