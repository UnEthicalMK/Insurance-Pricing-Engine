"""Phase 9: metric correctness tests."""
from __future__ import annotations
import numpy as np
from src.evaluation.metrics import actuarial_gini, decile_lift, calibration

def test_gini_perfect_prediction_is_one() -> None:
    y = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    assert actuarial_gini(y, y) > 0.99

def test_gini_perfect_inverse_is_negative_one() -> None:
    """Check if perfectly wrong ranking yields negative Gini."""
    y = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    y_inverse = y[::-1] # Reverse the predictions
    assert actuarial_gini(y, y_inverse) < -0.99

def test_gini_all_zeros_handled_gracefully() -> None:
    """Ensure no division-by-zero errors when no claims exist."""
    y = np.zeros(100)
    p = np.random.default_rng(42).uniform(0, 1, 100)
    assert actuarial_gini(y, p) == 0.0

def test_gini_random_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    y = rng.gamma(2.0, 100.0, 5000)
    pred = rng.permutation(y)
    assert abs(actuarial_gini(y, pred)) < 0.1

def test_lift_increases_with_risk() -> None:
    y = np.arange(100.0)
    d = decile_lift(y, y)
    assert d["avg_actual_loss_per_decile"].is_monotonic_increasing

def test_no_nan_outputs() -> None:
    rng = np.random.default_rng(1)
    y = rng.gamma(2.0, 50.0, 1000)
    p = rng.gamma(2.0, 50.0, 1000)
    assert not np.isnan(actuarial_gini(y, p))
    assert not decile_lift(y, p).isna().any().any()

def test_shape_consistency() -> None:
    rng = np.random.default_rng(2)
    y = rng.gamma(2.0, 50.0, 1000)
    p = rng.gamma(2.0, 50.0, 1000)
    assert len(decile_lift(y, p)) == 10
    assert len(calibration(y, p)) == 10