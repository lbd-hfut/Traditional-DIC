from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NumericDiagnostics:
    field: str
    shape: tuple[int, ...]
    atol: float
    rtol: float
    max_abs_diff: float
    mean_abs_diff: float
    rmse: float
    max_relative_error: float
    failing_count: int
    worst_index: tuple[int, ...] | None
    expected_value: Any
    actual_value: Any

    def format(self) -> str:
        return (
            f"numeric comparison failed: field={self.field!r}, shape={self.shape}, "
            f"atol={self.atol:.17g}, rtol={self.rtol:.17g}, "
            f"max_abs_diff={self.max_abs_diff:.17g}, "
            f"mean_abs_diff={self.mean_abs_diff:.17g}, rmse={self.rmse:.17g}, "
            f"max_relative_error={self.max_relative_error:.17g}, "
            f"failing_count={self.failing_count}, worst_index={self.worst_index}, "
            f"expected={self.expected_value!r}, actual={self.actual_value!r}"
        )


def assert_exact(expected: Any, actual: Any, *, field: str) -> None:
    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)
    if expected_array.shape != actual_array.shape:
        raise AssertionError(
            f"exact comparison shape mismatch: field={field!r}, "
            f"expected_shape={expected_array.shape}, actual_shape={actual_array.shape}"
        )

    equal = np.equal(expected_array, actual_array)
    if np.issubdtype(expected_array.dtype, np.inexact) and np.issubdtype(
        actual_array.dtype, np.inexact
    ):
        equal = equal | (np.isnan(expected_array) & np.isnan(actual_array))
    if bool(np.all(equal)):
        return

    index = tuple(int(i) for i in np.argwhere(~equal)[0])
    raise AssertionError(
        f"exact comparison failed: field={field!r}, shape={expected_array.shape}, "
        f"mismatch_count={int(np.count_nonzero(~equal))}, worst_index={index}, "
        f"expected={expected_array[index]!r}, actual={actual_array[index]!r}"
    )


def assert_numeric(
    expected: Any,
    actual: Any,
    *,
    field: str,
    atol: float,
    rtol: float,
    max_abs: float | None = None,
    max_rmse: float | None = None,
) -> NumericDiagnostics:
    expected_array = np.asarray(expected, dtype=np.float64)
    actual_array = np.asarray(actual, dtype=np.float64)
    if expected_array.shape != actual_array.shape:
        raise AssertionError(
            f"numeric comparison shape mismatch: field={field!r}, "
            f"expected_shape={expected_array.shape}, actual_shape={actual_array.shape}"
        )

    expected_nan = np.isnan(expected_array)
    actual_nan = np.isnan(actual_array)
    expected_posinf = np.isposinf(expected_array)
    actual_posinf = np.isposinf(actual_array)
    expected_neginf = np.isneginf(expected_array)
    actual_neginf = np.isneginf(actual_array)
    special_mismatch = (
        (expected_nan != actual_nan)
        | (expected_posinf != actual_posinf)
        | (expected_neginf != actual_neginf)
    )

    finite = np.isfinite(expected_array) & np.isfinite(actual_array)
    abs_diff = np.zeros(expected_array.shape, dtype=np.float64)
    abs_diff[finite] = np.abs(actual_array[finite] - expected_array[finite])
    abs_diff[special_mismatch] = np.inf
    tolerance = np.full(expected_array.shape, atol, dtype=np.float64)
    tolerance[finite] += rtol * np.abs(expected_array[finite])
    failing = special_mismatch | (finite & (abs_diff > tolerance))

    finite_diffs = abs_diff[finite]
    if finite_diffs.size:
        max_abs_diff = float(np.max(finite_diffs))
        mean_abs_diff = float(np.mean(finite_diffs))
        rmse = float(np.sqrt(np.mean(np.square(finite_diffs))))
        denominator = np.abs(expected_array[finite])
        relative = np.divide(
            finite_diffs,
            denominator,
            out=np.zeros_like(finite_diffs),
            where=denominator > 0,
        )
        max_relative_error = float(np.max(relative))
    else:
        max_abs_diff = mean_abs_diff = rmse = max_relative_error = 0.0

    threshold_failed = (
        (max_abs is not None and max_abs_diff > max_abs)
        or (max_rmse is not None and rmse > max_rmse)
    )
    failing_count = int(np.count_nonzero(failing))
    if failing_count:
        worst_flat = int(np.argmax(abs_diff))
        worst_index = tuple(int(i) for i in np.unravel_index(worst_flat, abs_diff.shape))
    elif threshold_failed and abs_diff.size:
        worst_flat = int(np.argmax(abs_diff))
        worst_index = tuple(int(i) for i in np.unravel_index(worst_flat, abs_diff.shape))
    else:
        worst_index = None

    expected_value = expected_array[worst_index] if worst_index is not None else None
    actual_value = actual_array[worst_index] if worst_index is not None else None
    diagnostics = NumericDiagnostics(
        field=field,
        shape=expected_array.shape,
        atol=atol,
        rtol=rtol,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        rmse=rmse,
        max_relative_error=max_relative_error,
        failing_count=failing_count,
        worst_index=worst_index,
        expected_value=expected_value,
        actual_value=actual_value,
    )
    if failing_count or threshold_failed:
        raise AssertionError(diagnostics.format())
    return diagnostics
