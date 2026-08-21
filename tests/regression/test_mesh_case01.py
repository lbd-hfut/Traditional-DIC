from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.support.baseline_io import load_json_baseline, load_npz_baseline, load_provenance
from tests.support.provenance import verify_relative_hashes
from tests.support.regression_compare import assert_exact, assert_numeric


pytestmark = pytest.mark.regression
FAMILIES = ("T3", "Q4", "Q8")
EXPECTED_COUNTS = {"T3": (119, 194, 53544), "Q4": (115, 92, 57500), "Q8": (321, 92, 57500)}
HEADERS = {
    "final_U": ["node", "x", "y", "u", "v", "mag"],
    "dense_U": ["id", "x", "y", "u", "v", "mag", "valid"],
    "dense_strain": ["x", "y", "du_dx", "du_dy", "dv_dx", "dv_dy", "exx", "eyy", "exy", "sample_count", "valid"],
}
NUMERIC = {"atol": 1e-12, "rtol": 1e-10, "max_abs": 1e-10, "max_rmse": 1e-12}


def read_csv(path: Path) -> dict[str, np.ndarray]:
    table = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8"))
    return {name: np.asarray(table[name]) for name in table.dtype.names or ()}


def read_table(path: Path, *, dtype: type) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path, delimiter=",", comments="#", dtype=dtype))


@pytest.mark.parametrize("family", FAMILIES)
def test_mesh_portable_baseline_integrity(f0_data_root: Path, family: str) -> None:
    root = f0_data_root / "mesh_case01" / family
    required = ("nodes", "elements") + tuple(
        f"{group}__{name}" for group, header in HEADERS.items() for name in header
    )
    arrays = load_npz_baseline(root / "baseline.npz", required_arrays=required)
    summary = load_json_baseline(root / "summary.json", required_keys=("element_family", "topology", "headers", "row_counts", "statistics", "source_artifacts"))
    nodes, elements, dense = EXPECTED_COUNTS[family]
    assert summary["element_family"] == family
    assert summary["headers"] == HEADERS
    assert summary["topology"]["node_count"] == nodes
    assert summary["topology"]["element_count"] == elements
    assert summary["row_counts"] == {"final_U": nodes, "dense_U": dense, "dense_strain": dense}
    assert arrays["nodes"].shape[0] == nodes
    assert arrays["elements"].shape[0] == elements
    assert arrays["elements"].shape[1] == {"T3": 4, "Q4": 5, "Q8": 9}[family]
    assert_exact(arrays["nodes"][:, 0], np.arange(1, nodes + 1), field=f"mesh.{family}.node_ids")
    assert_exact(arrays["elements"][:, 0], np.arange(1, elements + 1), field=f"mesh.{family}.element_ids")
    assert_exact(arrays["final_U__node"], arrays["nodes"][:, 0], field=f"mesh.{family}.nodal_mapping")
    assert int(np.min(arrays["elements"][:, 1:])) >= 1
    assert int(np.max(arrays["elements"][:, 1:])) <= nodes


def test_mesh_provenance_and_input_contract(repository_root: Path, f0_data_root: Path) -> None:
    provenance = load_provenance(f0_data_root / "mesh_case01/provenance.json")
    assert len(provenance["input_order"]) == 11
    assert provenance["input_order"][0]["role"] == "reference"
    assert provenance["input_order"][-1]["role"] == "roi"
    assert all(item["role"] == "deformed" for item in provenance["input_order"][1:-1])
    verify_relative_hashes(repository_root, provenance["config_sha256"])
    verify_relative_hashes(repository_root, provenance["input_sha256"])


@pytest.mark.requires_local_history
@pytest.mark.parametrize("family", FAMILIES)
def test_mesh_local_history_matches_golden(repository_root: Path, f0_data_root: Path, family: str) -> None:
    source = repository_root / "case/mono_DIC/01/result/mesh/test1_001_ssd_fgn" / family
    if not source.is_dir():
        pytest.skip(f"ignored local Mesh {family} historical artifacts are unavailable")
    expected = load_npz_baseline(f0_data_root / "mesh_case01" / family / "baseline.npz")
    assert_exact(expected["nodes"], read_table(source / f"nodes_{family}.txt", dtype=np.float64), field=f"mesh.{family}.nodes")
    assert_exact(expected["elements"], read_table(source / f"elements_{family}.txt", dtype=np.int64), field=f"mesh.{family}.elements")

    exact_fields = {"node", "id", "x", "y", "valid", "sample_count"}
    for group, filename in (("final_U", "final_U.csv"), ("dense_U", "dense_U.csv"), ("dense_strain", "dense_strain.csv")):
        actual = read_csv(source / filename)
        assert list(actual) == HEADERS[group]
        for name, values in actual.items():
            golden = expected[f"{group}__{name}"]
            if name in exact_fields:
                assert_exact(golden, values, field=f"mesh.{family}.{group}.{name}")
            else:
                assert_numeric(golden, values, field=f"mesh.{family}.{group}.{name}", **NUMERIC)

    expected_summary = load_json_baseline(f0_data_root / "mesh_case01" / family / "summary.json")["statistics"]
    actual_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    for name in ("nodes", "elements", "dense_samples", "solver", "criterion", "initialization", "nodes_file", "elements_file", "nodal_displacement_file", "dense_displacement_file"):
        assert expected_summary[name] == actual_summary[name]
    for name in ("mag_mean", "mag_max"):
        assert_numeric(expected_summary[name], actual_summary[name], field=f"mesh.{family}.summary.{name}", **NUMERIC)
