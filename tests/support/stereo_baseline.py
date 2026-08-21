from __future__ import annotations

from typing import Any

import numpy as np


def numeric_difference(expected: Any, actual: Any) -> dict[str, Any]:
    left = np.asarray(expected, dtype=np.float64)
    right = np.asarray(actual, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        raise ValueError("NaN location mismatch")
    if not np.array_equal(np.isposinf(left), np.isposinf(right)) or not np.array_equal(
        np.isneginf(left), np.isneginf(right)
    ):
        raise ValueError("Inf location/sign mismatch")
    finite = np.isfinite(left) & np.isfinite(right)
    difference = np.abs(right[finite] - left[finite])
    if difference.size == 0:
        return {"shape": list(left.shape), "max_abs_diff": 0.0, "mean_abs_diff": 0.0, "rmse": 0.0}
    return {
        "shape": list(left.shape),
        "max_abs_diff": float(np.max(difference)),
        "mean_abs_diff": float(np.mean(difference)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
    }


def within_policy(metrics: dict[str, Any], *, atol: float, max_rmse: float) -> bool:
    return metrics["max_abs_diff"] <= atol and metrics["rmse"] <= max_rmse
