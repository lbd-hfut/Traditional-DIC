from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_hashes(root: Path, paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256_file(path) for path in paths}


def verify_relative_hashes(root: Path, expected: Mapping[str, str]) -> None:
    for relative_path, expected_hash in expected.items():
        actual_hash = sha256_file(root / relative_path)
        if actual_hash != expected_hash:
            raise AssertionError(
                f"SHA256 mismatch for {relative_path}: "
                f"expected={expected_hash}, actual={actual_hash}"
            )
