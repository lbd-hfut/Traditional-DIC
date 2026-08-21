import numpy as np
import pytest

from tests.support.regression_compare import assert_exact, assert_numeric


pytestmark = pytest.mark.unit


def test_exact_pass() -> None:
    assert_exact([1, 2, 3], [1, 2, 3], field="ids")


def test_exact_fail_has_diagnostics() -> None:
    with pytest.raises(AssertionError, match=r"field='ids'.*mismatch_count=1.*worst_index=\(1,\)"):
        assert_exact([1, 2, 3], [1, 9, 3], field="ids")


def test_exact_shape_mismatch() -> None:
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_exact([1, 2], [[1, 2]], field="shape")


def test_numeric_pass_and_metrics() -> None:
    result = assert_numeric(
        [1.0, 2.0], [1.0, 2.000001], field="u", atol=2e-6, rtol=0.0
    )
    assert result.max_abs_diff == pytest.approx(1e-6)
    assert result.rmse == pytest.approx(1e-6 / np.sqrt(2.0))


def test_numeric_fail_has_required_diagnostics() -> None:
    with pytest.raises(AssertionError) as raised:
        assert_numeric([1.0, 2.0], [1.0, 2.1], field="v", atol=1e-6, rtol=0.0)
    message = str(raised.value)
    for fragment in (
        "field='v'",
        "shape=(2,)",
        "atol=",
        "rtol=",
        "max_abs_diff=",
        "mean_abs_diff=",
        "rmse=",
        "failing_count=1",
        "worst_index=(1,)",
        "expected=",
        "actual=",
    ):
        assert fragment in message


def test_numeric_shape_mismatch() -> None:
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_numeric([1.0], [[1.0]], field="shape", atol=0.0, rtol=0.0)


def test_nan_equality() -> None:
    assert_numeric([np.nan, 1.0], [np.nan, 1.0], field="nan", atol=0.0, rtol=0.0)


def test_nan_mismatch() -> None:
    with pytest.raises(AssertionError, match="failing_count=1"):
        assert_numeric([np.nan], [1.0], field="nan", atol=1.0, rtol=1.0)


def test_inf_sign_mismatch() -> None:
    with pytest.raises(AssertionError, match="failing_count=1"):
        assert_numeric([np.inf], [-np.inf], field="inf", atol=0.0, rtol=0.0)


def test_mask_mismatch() -> None:
    with pytest.raises(AssertionError, match="field='valid'"):
        assert_exact([True, False], [True, True], field="valid")


def test_max_abs_threshold() -> None:
    with pytest.raises(AssertionError, match="max_abs_diff="):
        assert_numeric(
            [0.0, 0.0],
            [0.01, -0.01],
            field="u",
            atol=0.02,
            rtol=0.0,
            max_abs=0.005,
        )


def test_rmse_threshold() -> None:
    with pytest.raises(AssertionError, match="rmse="):
        assert_numeric(
            [0.0, 0.0],
            [0.01, 0.01],
            field="u",
            atol=0.02,
            rtol=0.0,
            max_rmse=0.005,
        )
