# Traditional-DIC

Traditional-DIC is a C++17 and Python implementation of conventional digital image correlation (DIC). It provides YAML-driven 2D Subset-DIC, 2D finite-element Mesh-DIC, stereo 3D-DIC, and multi-view 3D-DIC workflows. Numerical products are written to a case `result/` tree and figures to `visualization/`.

## Scope

- **Subset-DIC**: circular and ROI-truncated subsets, mirror padding, integer-search/SIFT priors, seed selection, ncorr-style reliability propagation, first- and second-order shape functions, SSD/ZNSSD, ICGN, and Forward Gauss-Newton.
- **Mesh-DIC**: T3/Q4/Q8 elements, automatic ROI meshing or file meshes, FE-DIC FFT initialization, sparse global solve, regularization, and nodal/dense fields.
- **Shared modules**: floating-point images, binary masks, weighted SSD/ZNSSD/ZNCC, B-spline interpolation (degree 1/3/5), gradients, and ROI-aware initialization.
- **3D workflows**: chessboard calibration, stereo triangulation, multi-view COLMAP-like calibration, metric-scale recovery, pair selection, pairwise reconstruction, and surface stitching.

```text
Image / ROI -> preprocessing and interpolation -> initialization
            -> Subset-DIC or Mesh-DIC -> camera geometry
            -> stereo/multi-view reconstruction -> 3D displacement
```

Subset-DIC and Mesh-DIC share image, mask, interpolation, correlation, and initialization modules, while retaining separate sampling, propagation/topology, and solver logic. Reconstruction modules consume 2D fields and do not own 2D optimizer logic.

## Layout

```text
include/dic/       Public C++ interfaces
src/               C++ implementation
bindings/, python/ pybind11 binding and Python workflow layer
examples/          YAML-driven applications
config/            Algorithm and case-path YAML files
case/              Input cases and generated products
docs/              Supplementary design notes
```

## Requirements

Required: CMake 3.16+, a C++17 compiler, Eigen3, yaml-cpp, Python 3, NumPy, Pillow, and PyYAML. Optional dependencies are OpenCV (image I/O, SIFT, calibration), Ceres, pybind11, SciPy, and Matplotlib.

## Build on Windows/MSVC

The examples call the Python binding. Activate the intended Conda environment before configuring, then use the Visual Studio x64 developer environment:

```powershell
conda activate <environment>
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake -S . -B build -DTRADITIONAL_DIC_BUILD_PYTHON=ON'
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build build --config Release'
python -c "import traditional_dic; print(\"traditional_dic import succeeded\")"
```

Use the same `build/` directory for later builds. If CMake cannot find packages, activate the Conda environment containing them or set `CMAKE_PREFIX_PATH`/`OpenCV_DIR` before configuring. Python, OpenCV, and the compiler must use the same architecture.

## Configuration

Algorithm YAML files contain no case I/O paths:

| File | Contents |
| --- | --- |
| `config/subset_2d.yaml` | subset, shape, objective, optimizer, initialization, propagation |
| `config/mesh_2d.yaml` | elements, meshing, global solve, regularization, interpolation |
| `config/calibration.yaml` | calibration board, detection, calibration, scale recovery |
| `config/stereo_3d.yaml` | stereo workflow and reconstruction options |
| `config/multiview_3d.yaml` | self-calibration, pair ROI, pairwise solve, scale, stitching |

All case-specific paths are in [`config/case_paths.yaml`](config/case_paths.yaml). Except for `case_root`, paths are relative to the case root.

```yaml
mono_2d:
  case_root: case/mono_DIC/ring
  images_dir: .
stereo_3d:
  case_root: case/stereo_DIC/plate_center_load
  left_images_dir: cam1
  right_images_dir: cam2
multiview_3d:
  case_root: case/multi_DIC/CylinderDIC
  images: { root: images }
  roi: { mode: auto } # auto | last_image
```

**Image convention**: mono files are sorted, with first=reference, last=ROI, and every intermediate file=deformed. Stereo left/right directories use first=reference and last=deformed. Multi-view `auto` uses first=reference and last=deformed, then generates pair ROIs. Multi-view `last_image` uses first=reference, penultimate=deformed, last=camera ROI; the manual ROI is used for temporal and stereo-disparity DIC whenever that camera is master.

All workflows support `result_root: result` and `visualization_root: visualization` in their path section. CSV/JSON/mesh/reconstruction data go below `result/`; figures go below `visualization/`.

## Run

All commands accept `--paths-config`; use it to select a new case without editing algorithm YAML.

```powershell
python examples\subset_2d.py --paths-config config\case_paths.yaml --config config\subset_2d.yaml
python examples\mesh_2d.py --paths-config config\case_paths.yaml --config config\mesh_2d.yaml --element all
python examples\stereo_3d.py --paths-config config\case_paths.yaml --stereo-config config\stereo_3d.yaml --calibration-config config\calibration.yaml --solver subset --compute-fields
python examples\multiview_3d.py --paths-config config\case_paths.yaml --config config\multiview_3d.yaml --solver subset
```

Subset results are written as `result/subset/<frame>/displacements.csv` and `stats.json`. Mesh runs write generation, nodal, and dense products per frame and element. Stereo uses three left-reference fields: `L0 -> R0` reference disparity, `L0 -> L1` temporal displacement, and `L0 -> R1` deformed disparity, then triangulates `X0`, `X1`, and `U3D = X1 - X0`. Multi-view runs self-calibration, scale recovery, pair selection, ROI generation/loading, pairwise 2D DIC, pairwise 3D reconstruction, and stitching. Use `--solver subset|mesh|both`; `--resume` reuses complete pairwise 2D fields.

## Numerical Conventions

SSD and ZNSSD are lower-is-better; ZNCC is higher-is-better and is used for initialization/quality scoring rather than the current Gauss-Newton objective. Zero sample weight excludes an ROI sample. ZNSSD/ZNCC require nonzero variance in both vectors. Images normally use unit-intensity grayscale. Inverse-compositional solvers reuse reference gradients/Hessian terms; forward solvers evaluate warped deformed gradients. Mirror image padding stabilizes edge interpolation while padded ROI remains invalid.

Integer search uses circular ROI-aware support and may use an image pyramid. Subset seed selection applies texture and quality tests before optional subpixel refinement. `reliability_propagation.spacing` is the ncorr gap count: full-resolution stride is `spacing + 1`, not image downsampling. 2D fields provide coordinates, `u`, `v`, quality, and `valid`; 3D fields add `X0/Y0/Z0`, `X1/Y1/Z1`, `Ux/Uy/Uz/Umag`, reprojection error, and validity.

## Documentation Status

This README consolidates the operational architecture, preprocessing, correlation, initialization, stereo, multi-view, configuration, build, and output conventions required to use the project. It is intended to remain the primary documentation entry point when supplementary design notes are removed.

## Acknowledgements

Traditional-DIC draws on ideas, numerical conventions, validation references, and workflow structure from these open-source projects. Please cite their original work when your research relies on them:

- [justinblaber/ncorr_2D_matlab](https://github.com/justinblaber/ncorr_2D_matlab)
- [YangMechanicsGroupUTAustin/2D_FE_Global_DIC](https://github.com/YangMechanicsGroupUTAustin/2D_FE_Global_DIC)
- [SolavLab/DuoDIC](https://github.com/SolavLab/DuoDIC)
- [MultiDIC/MultiDIC](https://github.com/MultiDIC/MultiDIC)
- [colmap/colmap](https://github.com/colmap/colmap)

## Citation

If Traditional-DIC contributes to a publication, cite the version or commit used:

```bibtex
@software{leebda_traditional_dic_2026,
  author = {LeeBDa},
  title = {Traditional-DIC: Traditional Digital Image Correlation Workflows},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/lbd-hfut/Traditional-DIC},
  note = {Version <used-version>}
}
```

Replace `<used-version>` with the release tag or commit hash. Use a release DOI when one becomes available.

## Development and License

Keep Subset-DIC and Mesh-DIC solver changes isolated unless a shared numerical module must change. Do not place case I/O paths in algorithm YAML. Build directories, case `result/`, case `visualization/`, Python caches, and compiled extensions are generated products and should not be committed. Traditional-DIC is released under the [MIT License](LICENSE).
