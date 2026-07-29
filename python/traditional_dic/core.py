"""Core image and mask helpers."""

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


def load_image(path):
    _require_backend()
    return _backend.core.load_image(str(path))


def load_mask(path):
    _require_backend()
    return _backend.core.load_mask(str(path))


def normalize_image(image, method="none"):
    _require_backend()
    return _backend.core.normalize_image(image, method)
