"""Locations shared by source-tree and installed Traditional-DIC runtimes."""
from __future__ import annotations

from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parent
_SOURCE_ROOT = _PACKAGE_DIR.parents[1]


def runtime_root() -> Path:
    """Return the source checkout root when present, otherwise the package root."""
    return _SOURCE_ROOT if (_SOURCE_ROOT / "config").is_dir() else _PACKAGE_DIR


def default_config_dir() -> Path:
    """Return the one installed-or-source location of bundled default YAML files."""
    source_config = _SOURCE_ROOT / "config"
    return source_config if source_config.is_dir() else _PACKAGE_DIR / "resources" / "config"


def default_config_path(value: str | Path) -> Path:
    """Resolve a canonical ``config/*.yaml`` reference to the bundled file."""
    return default_config_dir() / Path(value).name


def default_paths_config() -> Path:
    return default_config_path("case_paths.yaml")


def resolve_config_reference(value: str | Path, root: str | Path) -> Path:
    """Resolve an explicit path first, then a canonical bundled config reference."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    rooted = Path(root).resolve() / path
    if rooted.exists():
        return rooted
    bundled = default_config_path(path)
    return bundled if bundled.exists() else rooted
