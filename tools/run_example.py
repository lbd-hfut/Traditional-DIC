"""Launch a DIC example with the built runtime DLL directories registered.

On Windows, Python 3.8+ no longer searches PATH for dependent DLLs, so
``import traditional_dic`` fails unless the directories holding the built
runtime (the MSYS2 OpenCV ``bin/`` and the collected ``build/_opencv/runtime``
dependencies) are registered via ``os.add_dll_directory``. This helper does
that registration and then runs an example script through ``runpy`` so its
``argparse`` sees the original arguments unchanged.

Usage (from the repository root):

    python tools/run_example.py examples/subset_2d.py [--paths-config ...]

Any script can be launched, not only those under ``examples/``. The repo root
is resolved from this file's location, so the command works from any working
directory. Requires the project to be built first (``cmake --build build``).

This is a standalone helper; no production code depends on it.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
RUNTIME_DIR = REPO_ROOT / "build" / "_opencv" / "runtime"
OPENCV_BIN_DIR = REPO_ROOT / "build" / "_opencv" / "ucrt64" / "bin"


def setup_dll_directories() -> list[Path]:
    """Register the DLL directories required to import ``traditional_dic``.

    Returns the directories that were actually registered (empty on
    non-Windows, where DLL registration is a no-op).
    """
    if os.name != "nt":
        return []
    registered: list[Path] = []
    for directory in (RUNTIME_DIR, OPENCV_BIN_DIR):
        if directory.is_dir():
            os.add_dll_directory(str(directory))
            registered.append(directory)
    if not registered:
        raise RuntimeError(
            f"runtime DLL directories not found under {REPO_ROOT / 'build' / '_opencv'}: "
            "build the project first (cmake --build build), or set a custom layout "
            "and call os.add_dll_directory yourself."
        )
    return registered


def _warn_about_pythonhome() -> None:
    """Print a hint when a global PYTHONHOME could break the interpreter."""
    home = os.environ.get("PYTHONHOME")
    if not home:
        return
    try:
        mismatch = Path(home).resolve() != Path(sys.prefix).resolve()
    except OSError:
        mismatch = True
    if mismatch:
        print(
            f"warning: PYTHONHOME={home!r} does not match sys.prefix={sys.prefix!r}; "
            "this can break imports (unset it when using a conda environment).",
            file=sys.stderr,
        )


def _print_usage() -> None:
    print(__doc__)
    example_dir = REPO_ROOT / "examples"
    if example_dir.is_dir():
        scripts = sorted(p.name for p in example_dir.glob("*.py"))
        if scripts:
            print("Available examples:")
            for name in scripts:
                print(f"    python tools/run_example.py examples/{name}")


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        _print_usage()
        return 0

    example = Path(argv[0])
    if not example.is_absolute():
        example = REPO_ROOT / example
    example = example.resolve()
    if not example.is_file():
        print(f"error: example script not found: {argv[0]}", file=sys.stderr)
        _print_usage()
        return 2

    _warn_about_pythonhome()
    setup_dll_directories()

    sys.argv = [str(example), *argv[1:]]
    runpy.run_path(str(example), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
