import json

import numpy as np
import pytest

from tests.support.baseline_io import (
    BaselineFormatError,
    load_json_baseline,
    load_npz_baseline,
    load_provenance,
)


pytestmark = pytest.mark.unit


def test_npz_load_and_required_arrays(tmp_path) -> None:
    path = tmp_path / "baseline.npz"
    np.savez_compressed(path, schema_version=np.array("1.0"), u=np.array([1.0]))
    loaded = load_npz_baseline(path, required_arrays=("u",))
    np.testing.assert_array_equal(loaded["u"], [1.0])


def test_npz_missing_required_array_fails_closed(tmp_path) -> None:
    path = tmp_path / "baseline.npz"
    np.savez_compressed(path, schema_version=np.array("1.0"))
    with pytest.raises(BaselineFormatError, match="missing required arrays"):
        load_npz_baseline(path, required_arrays=("u",))


def test_npz_unsupported_schema_fails_closed(tmp_path) -> None:
    path = tmp_path / "baseline.npz"
    np.savez_compressed(path, schema_version=np.array("99"))
    with pytest.raises(BaselineFormatError, match="unsupported schema_version"):
        load_npz_baseline(path)


def test_malformed_npz_fails_closed(tmp_path) -> None:
    path = tmp_path / "baseline.npz"
    path.write_bytes(b"not an archive")
    with pytest.raises(BaselineFormatError, match="cannot load NPZ"):
        load_npz_baseline(path)


def test_json_load_and_required_keys(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"schema_version": "1.0", "rows": 2}))
    assert load_json_baseline(path, required_keys=("rows",))["rows"] == 2


def test_json_missing_key_fails_closed(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"schema_version": "1.0"}))
    with pytest.raises(BaselineFormatError, match="missing required keys"):
        load_json_baseline(path, required_keys=("rows",))


def test_json_unsupported_schema_fails_closed(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"schema_version": "2.0"}))
    with pytest.raises(BaselineFormatError, match="unsupported or missing schema_version"):
        load_json_baseline(path)


def test_malformed_json_fails_closed(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text("{")
    with pytest.raises(BaselineFormatError, match="cannot load JSON"):
        load_json_baseline(path)


def test_provenance_requires_contract(tmp_path) -> None:
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps({"schema_version": "1.0"}))
    with pytest.raises(BaselineFormatError, match="missing required keys"):
        load_provenance(path)
