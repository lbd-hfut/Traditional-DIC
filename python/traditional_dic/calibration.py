try:
    from ._traditional_dic import calibration as _calibration
except ImportError as exc:  # pragma: no cover - import-time environment guard
    _calibration = None
    _import_error = exc


def _require_backend():
    if _calibration is None:
        raise ImportError("traditional_dic C++ calibration backend is not available") from _import_error
    return _calibration


def __getattr__(name):
    backend = _require_backend()
    return getattr(backend, name)


def detect_calibration_board(image_path, board, options=None):
    backend = _require_backend()
    if options is None:
        return backend.detect_calibration_board(image_path, board)
    return backend.detect_calibration_board(image_path, board, options)


def calibrate_mono_zhang(image_paths, board, options=None):
    backend = _require_backend()
    if options is None:
        return backend.calibrate_mono_zhang(image_paths, board)
    return backend.calibrate_mono_zhang(image_paths, board, options)


def calibrate_stereo_zhang(left_image_paths, right_image_paths, board, options=None):
    backend = _require_backend()
    if options is None:
        return backend.calibrate_stereo_zhang(left_image_paths, right_image_paths, board)
    return backend.calibrate_stereo_zhang(left_image_paths, right_image_paths, board, options)


def calibrate_multiview_colmap_like(image_paths, options=None):
    backend = _require_backend()
    if options is None:
        return backend.calibrate_multiview_colmap_like(image_paths)
    return backend.calibrate_multiview_colmap_like(image_paths, options)


def load_calibration(path):
    """Load calibration data from a file.

    File IO will be wired once the project settles on YAML/OpenCV/JSON exchange
    formats for camera parameters.
    """
    raise NotImplementedError(f"Calibration file loading is not implemented yet: {path}")
