import numpy as np
import pytest

from tests.support.stereo_baseline import numeric_difference, within_policy


pytestmark = pytest.mark.unit


def test_numeric_difference_reports_required_metrics() -> None:
    metrics = numeric_difference([1.0, 2.0], [1.0, 2.2])
    assert metrics["shape"] == [2]
    assert metrics["max_abs_diff"] == pytest.approx(0.2)
    assert metrics["mean_abs_diff"] == pytest.approx(0.1)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.02))


def test_numeric_difference_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        numeric_difference([1.0], [[1.0]])


def test_numeric_difference_rejects_special_value_mismatch() -> None:
    with pytest.raises(ValueError, match="NaN location mismatch"):
        numeric_difference([np.nan], [1.0])
    with pytest.raises(ValueError, match="Inf location/sign mismatch"):
        numeric_difference([np.inf], [-np.inf])


def test_within_policy_checks_max_and_rmse() -> None:
    metrics = {"max_abs_diff": 0.2, "rmse": 0.1}
    assert within_policy(metrics, atol=0.2, max_rmse=0.1)
    assert not within_policy(metrics, atol=0.19, max_rmse=0.1)
    assert not within_policy(metrics, atol=0.2, max_rmse=0.09)
