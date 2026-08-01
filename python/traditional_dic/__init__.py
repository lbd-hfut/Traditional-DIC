"""Traditional-DIC Python API skeleton."""

import os as _os
import sys as _sys

# ---------------------------------------------------------------------------
# On Windows, Python ≥ 3.8 uses a secure DLL search that ignores %PATH%.
# The compiled extension (_traditional_dic.pyd) depends on OpenCV and MinGW
# runtime DLLs.  We add the directories that contain those DLLs before the
# first import attempt so the loader can resolve them.
# ---------------------------------------------------------------------------
if _sys.platform == "win32":
    # The directory containing _traditional_dic.pyd itself (this package dir)
    _pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
    _os.add_dll_directory(_pkg_dir)

    # MinGW runtime (libgcc_s_seh-1, libstdc++-6, libwinpthread-1)
    _mingw_bin = _os.path.join(
        _os.environ.get("MSYS2_ROOT", "D:/Microsoft Visual Studio"), "mingw64", "bin"
    )
    if _os.path.isdir(_mingw_bin):
        _os.add_dll_directory(_mingw_bin)

    # OpenCV DLLs – try the install layout under the repo root first
    # pkg_dir = .../python/traditional_dic → repo_root = ... (2 levels up)
    _repo_root = _os.path.dirname(_os.path.dirname(_pkg_dir))
    _opencv_bin = _os.path.join(_repo_root, "opencv-build", "install", "x64", "mingw", "bin")
    if _os.path.isdir(_opencv_bin):
        _os.add_dll_directory(_opencv_bin)

    # Anaconda / Miniconda library directory (avif.dll and friends).
    # Try CONDA_PREFIX first, then scan %PATH% for Library\bin siblings.
    _conda_lib = None
    for _candidate in [
        _os.environ.get("CONDA_PREFIX", ""),
        *(_os.environ.get("PATH", "").split(";")),
    ]:
        _candidate = _candidate.strip()
        if not _candidate:
            continue
        _d = _os.path.normpath(_os.path.join(_candidate, "..", "Library", "bin"))
        if _os.path.isdir(_d):
            _conda_lib = _d
            break
    if _conda_lib:
        _os.add_dll_directory(_conda_lib)

from .subset import subset
from .mesh import mesh
from .stereo import stereo
from .multiview import multiview

__all__ = ["subset", "mesh", "stereo", "multiview"]
