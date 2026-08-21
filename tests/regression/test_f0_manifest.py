from pathlib import Path

import pytest

from tests.support.baseline_io import load_json_baseline
from tests.support.provenance import verify_relative_hashes


pytestmark = pytest.mark.regression


def test_f0_manifest_covers_and_verifies_all_baseline_files(f0_data_root: Path) -> None:
    manifest = load_json_baseline(
        f0_data_root / "manifest.json",
        required_keys=("baseline_commit", "baseline_scope", "immutable_at_pytest_runtime", "datasets", "files"),
    )
    assert manifest["baseline_commit"] == "806832419b0ab3ac40050d8c05c3bd0bed5098f6"
    assert manifest["immutable_at_pytest_runtime"] is True
    expected_files = {
        str(path.relative_to(f0_data_root))
        for path in f0_data_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert set(manifest["files"]) == expected_files
    verify_relative_hashes(
        f0_data_root,
        {relative: metadata["sha256"] for relative, metadata in manifest["files"].items()},
    )
