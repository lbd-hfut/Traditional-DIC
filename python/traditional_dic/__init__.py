"""Traditional-DIC Python API skeleton."""
from .subset import subset
from .mesh import mesh, generate_mesh_from_roi, generate_annulus_meshes_from_mask
from . import calibration
from .stereo import (
    reconstruct_from_field_files,
    reconstruct_from_fields,
    stereo,
)
from .multiview import (
    CameraPairSelectionOptions,
    PairMaskGenerationResult,
    Pairwise3DOptions,
    Pairwise3DRunResult,
    MultiviewScaleRecoveryOptions,
    MultiviewScaleRecoveryResult,
    PairwiseSurfaceStitchOptions,
    PairwiseSurfaceStitchRunResult,
    PairwiseDICOptions,
    PairwiseDICRunResult,
    MultiviewMaskOptions,
    compute_pairwise_2d_dic,
    compute_pairwise_3d_dic,
    generate_pair_masks_from_calibration,
    generate_masks_from_calibration,
    multiview,
    recover_multiview_calibration_scale,
    save_pair_selection_report,
    select_camera_pairs,
    stitch_pairwise_3d_surfaces,
)
from .config import load_config, normalize_mesh_config, normalize_subset_config
from . import visualization

__all__ = [
    "subset",
    "mesh",
    "generate_mesh_from_roi",
    "generate_annulus_meshes_from_mask",
    "calibration",
    "stereo",
    "reconstruct_from_fields",
    "reconstruct_from_field_files",
    "multiview",
    "generate_masks_from_calibration",
    "MultiviewMaskOptions",
    "CameraPairSelectionOptions",
    "PairMaskGenerationResult",
    "Pairwise3DOptions",
    "Pairwise3DRunResult",
    "MultiviewScaleRecoveryOptions",
    "MultiviewScaleRecoveryResult",
    "PairwiseSurfaceStitchOptions",
    "PairwiseSurfaceStitchRunResult",
    "PairwiseDICOptions",
    "PairwiseDICRunResult",
    "generate_pair_masks_from_calibration",
    "compute_pairwise_2d_dic",
    "compute_pairwise_3d_dic",
    "recover_multiview_calibration_scale",
    "stitch_pairwise_3d_surfaces",
    "select_camera_pairs",
    "save_pair_selection_report",
    "load_config",
    "normalize_mesh_config",
    "normalize_subset_config",
    "visualization",
]
