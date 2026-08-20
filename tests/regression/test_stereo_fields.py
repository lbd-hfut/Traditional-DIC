from pathlib import Path

import numpy as np
import pytest

from tests.support.baseline_io import load_json_baseline, load_npz_baseline
from tests.support.regression_compare import assert_exact


pytestmark = pytest.mark.regression
FIELD_CASES = (
    ("reference_disparity", "cam1/00_L.bmp", "cam2/00_R.bmp", "L0", "R0"),
    ("left_temporal", "cam1/00_L.bmp", "cam1/04_L.bmp", "L0", "Llast"),
    ("deformed_disparity", "cam1/00_L.bmp", "cam2/04_R.bmp", "L0", "Rlast"),
)


@pytest.mark.parametrize("name,source_a,source_b,reference_role,deformed_role", FIELD_CASES)
def test_stereo_subset_field_semantics_and_payload(
    repository_root: Path,
    name: str,
    source_a: str,
    source_b: str,
    reference_role: str,
    deformed_role: str,
) -> None:
    root = repository_root / "tests/data/f0b/stereo_plate/fields"
    metadata = load_json_baseline(
        root / f"{name}.json",
        required_keys=("field_name", "solver", "source_image_a", "source_image_b", "reference_semantic", "deformed_semantic", "roi_source", "subset_config", "subset_config_sha256", "header", "row_count", "valid_count", "quality", "tolerance_policy"),
    )
    arrays = load_npz_baseline(root / f"{name}.npz", required_arrays=("id", "x", "y", "u", "v", "correlation", "valid"))
    prefix = "case/stereo_DIC/plate_center_load/"
    assert metadata["field_name"] == name
    assert metadata["solver"] == "subset"
    assert metadata["source_image_a"] == prefix + source_a
    assert metadata["source_image_b"] == prefix + source_b
    assert metadata["reference_semantic"] == reference_role
    assert metadata["deformed_semantic"] == deformed_role
    assert metadata["roi_source"] == prefix + "ROI.bmp"
    assert metadata["header"] == ["id", "x", "y", "u", "v", "correlation", "valid"]
    assert metadata["row_count"] == 97200
    assert metadata["valid_count"] == 7654
    assert metadata["tolerance_policy"]["name"] == "INITIAL_F0B_FIELD_TOLERANCE"
    assert_exact(arrays["id"], np.arange(1, 97201), field=f"{name}.id")
    assert int(np.count_nonzero(arrays["valid"])) == 7654
    assert set(np.unique(arrays["valid"])) <= {0, 1}
    assert np.isfinite(arrays["u"]).all()
    assert np.isfinite(arrays["v"]).all()
    assert np.isfinite(arrays["correlation"]).all()


def test_stereo_field_structural_domains_match(repository_root: Path) -> None:
    root = repository_root / "tests/data/f0b/stereo_plate/fields"
    fields = {name: load_npz_baseline(root / f"{name}.npz") for name, *_ in FIELD_CASES}
    reference = fields["reference_disparity"]
    for name, arrays in fields.items():
        assert_exact(reference["id"], arrays["id"], field=f"{name}.id_domain")
        assert_exact(reference["x"], arrays["x"], field=f"{name}.x_domain")
        assert_exact(reference["y"], arrays["y"], field=f"{name}.y_domain")
        assert_exact(reference["valid"], arrays["valid"], field=f"{name}.valid_mask")
