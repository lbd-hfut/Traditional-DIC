from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.support.baseline_io import load_json_baseline, load_npz_baseline, load_provenance
from tests.support.provenance import verify_relative_hashes
from tests.support.regression_compare import assert_exact, assert_numeric


pytestmark = pytest.mark.regression
NUMERIC = {"atol": 1e-12, "rtol": 1e-10, "max_abs": 1e-10, "max_rmse": 1e-12}
DISPLACEMENT_HEADER = ["x", "y", "u", "v", "du_dx", "du_dy", "dv_dx", "dv_dy", "correlation", "valid"]
STRAIN_HEADER = ["x", "y", "du_dx", "du_dy", "dv_dx", "dv_dy", "exx", "eyy", "exy", "sample_count", "valid"]


def read_csv(path: Path) -> dict[str, np.ndarray]:
    table = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8"))
    return {name: np.asarray(table[name]) for name in table.dtype.names or ()}


def test_subset_ring_portable_baseline_integrity(f0_data_root: Path) -> None:
    root = f0_data_root / "subset_ring"
    arrays = load_npz_baseline(
        root / "baseline.npz",
        required_arrays=tuple(f"displacement__{name}" for name in DISPLACEMENT_HEADER)
        + tuple(f"strain__{name}" for name in STRAIN_HEADER),
    )
    summary = load_json_baseline(root / "summary.json", required_keys=("headers", "row_counts", "statistics", "source_artifacts"))
    provenance = load_provenance(root / "provenance.json")

    assert summary["headers"]["displacements"] == DISPLACEMENT_HEADER
    assert summary["headers"]["strain"] == STRAIN_HEADER
    assert summary["row_counts"] == {"displacements": 102400, "strain": 48928}
    assert len(arrays["displacement__x"]) == 102400
    assert len(arrays["strain__x"]) == 48928
    assert_exact(arrays["displacement__valid"], arrays["displacement__valid"].astype(bool), field="subset.valid_domain")
    assert int(np.count_nonzero(arrays["displacement__valid"])) == 48928
    assert summary["statistics"]["valid_points"] == 48928
    assert [item["role"] for item in provenance["input_order"]] == ["reference", "deformed", "roi"]
    verify_relative_hashes(f0_data_root.parents[2], provenance["config_sha256"])
    verify_relative_hashes(f0_data_root.parents[2], provenance["input_sha256"])


@pytest.mark.requires_local_history
def test_subset_ring_local_history_matches_golden(repository_root: Path, f0_data_root: Path) -> None:
    source = repository_root / "case/mono_DIC/ring/result/subset/002_znssd_icgn_1st"
    if not source.is_dir():
        pytest.skip("ignored local Subset historical artifacts are unavailable")
    expected = load_npz_baseline(f0_data_root / "subset_ring/baseline.npz")
    actual_groups = {
        "displacement": read_csv(source / "displacements.csv"),
        "strain": read_csv(source / "strain.csv"),
    }
    exact_fields = {"x", "y", "valid", "sample_count"}
    for group, actual in actual_groups.items():
        for name, values in actual.items():
            golden = expected[f"{group}__{name}"]
            if name in exact_fields:
                assert_exact(golden, values, field=f"subset.{group}.{name}")
            else:
                assert_numeric(golden, values, field=f"subset.{group}.{name}", **NUMERIC)
    summary = load_json_baseline(f0_data_root / "subset_ring/summary.json")
    actual_stats = json.loads((source / "stats.json").read_text(encoding="utf-8"))
    for name, value in summary["statistics"].items():
        if isinstance(value, int):
            assert value == actual_stats[name]
        else:
            assert_numeric(value, actual_stats[name], field=f"subset.stats.{name}", **NUMERIC)
