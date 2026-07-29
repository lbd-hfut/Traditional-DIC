"""Result I/O helpers."""

try:
    from . import _traditional_dic as _backend
    _has_backend = True
except ImportError:
    _has_backend = False


def _require_backend():
    if not _has_backend:
        raise ImportError(
            "C++ backend _traditional_dic not found. "
            "Build with -DTRADITIONAL_DIC_BUILD_PYTHON=ON."
        )


def save_displacement_csv(result, path):
    _require_backend()
    return _backend.io.save_displacement_csv(result, str(path))


def load_displacement_csv(path):
    _require_backend()
    return _backend.io.load_displacement_csv(str(path))
