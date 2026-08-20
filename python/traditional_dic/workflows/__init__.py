"""Stable workflow-control facade for Traditional-DIC.

The facade accepts only the normalized F1/F2 contracts and delegates all
scientific computation to the existing Python/native APIs.
"""

from .common import WorkflowContext, WorkflowRunResult
from .subset_2d import run_subset_2d
from .mesh_2d import run_mesh_2d
from .stereo_3d import run_stereo_3d
from .multiview_3d import run_multiview_3d

__all__ = [
    "WorkflowContext",
    "WorkflowRunResult",
    "run_subset_2d",
    "run_mesh_2d",
    "run_stereo_3d",
    "run_multiview_3d",
]
