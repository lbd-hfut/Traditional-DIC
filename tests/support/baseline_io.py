from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SUPPORTED_SCHEMA_VERSION = "1.0"


class BaselineFormatError(ValueError):
    pass


def _validate_schema(document: dict[str, Any], path: Path) -> None:
    version = document.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise BaselineFormatError(
            f"unsupported or missing schema_version in {path}: {version!r}; "
            f"expected {SUPPORTED_SCHEMA_VERSION!r}"
        )


def load_json_baseline(
    path: Path, *, required_keys: Iterable[str] = ()
) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineFormatError(f"cannot load JSON baseline {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise BaselineFormatError(f"JSON baseline must be an object: {path}")
    _validate_schema(document, path)
    missing = sorted(set(required_keys) - document.keys())
    if missing:
        raise BaselineFormatError(f"missing required keys in {path}: {missing}")
    return document


def load_npz_baseline(
    path: Path, *, required_arrays: Iterable[str] = ()
) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise BaselineFormatError(f"cannot load NPZ baseline {path}: {exc}") from exc
    if "schema_version" not in arrays or arrays["schema_version"].shape != ():
        raise BaselineFormatError(f"missing scalar schema_version in {path}")
    version = str(arrays["schema_version"].item())
    if version != SUPPORTED_SCHEMA_VERSION:
        raise BaselineFormatError(
            f"unsupported schema_version in {path}: {version!r}; "
            f"expected {SUPPORTED_SCHEMA_VERSION!r}"
        )
    missing = sorted(set(required_arrays) - arrays.keys())
    if missing:
        raise BaselineFormatError(f"missing required arrays in {path}: {missing}")
    return arrays


def load_provenance(path: Path) -> dict[str, Any]:
    return load_json_baseline(
        path,
        required_keys=(
            "baseline_commit",
            "repository",
            "source_case",
            "source_artifacts",
            "config_sha256",
            "environment",
            "extraction_method",
        ),
    )
