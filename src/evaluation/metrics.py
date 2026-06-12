"""Phase 5: actuarial Gini, lift/decile, and calibration metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src import config

def actuarial_gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized actuarial Gini evaluated via fast dot-product formulation."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    
    # Sort true values by predicted values
    order = np.argsort(y_pred)
    sorted_true = y_true[order]
    
    # Fast Gini using linear weights (avoids O(N) cumsum memory allocation)
    weights = np.arange(1, n + 1)
    
    def _gini_core(sorted_array: np.ndarray) -> float:
        total = np.sum(sorted_array)
        if total == 0:
            return 0.0
        # The exact discrete Gini calculation
        return (np.sum((2 * weights - n - 1) * sorted_array)) / (n * total)

    gini_model = _gini_core(sorted_true)
    
    # Calculate perfect Gini by sorting true values by themselves
    order_perfect = np.argsort(y_true)
    gini_perfect = _gini_core(y_true[order_perfect])
    
    return float(gini_model / gini_perfect) if gini_perfect != 0 else 0.0

def decile_lift(y_true: np.ndarray, y_pred: np.ndarray,
                n_deciles: int = config.N_DECILES) -> pd.DataFrame:
    """Average actual loss per prediction decile (ascending risk)."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["decile"] = pd.qcut(df["y_pred"].rank(method="first"),
                           n_deciles, labels=False)
    out = df.groupby("decile")["y_true"].mean().reset_index()
    out.columns = ["decile", "avg_actual_loss_per_decile"]
    return out

def calibration(y_true: np.ndarray, y_pred: np.ndarray,
                n_buckets: int = config.N_CALIB_BUCKETS) -> pd.DataFrame:
    """Predicted vs actual mean loss per quantile bucket."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["bucket"] = pd.qcut(df["y_pred"].rank(method="first"),
                           n_buckets, labels=False)
    out = df.groupby("bucket").agg(
        predicted=("y_pred", "mean"), actual=("y_true", "mean")).reset_index()
    return out

def calibration_error(calib: pd.DataFrame) -> float:
    """Volume-weighted percentage error gap between predicted and actual."""
    # Prevents division by zero for completely safe buckets
    safe_actual = np.where(calib["actual"] == 0, 1e-6, calib["actual"])
    percentage_errors = np.abs(calib["predicted"] - calib["actual"]) / safe_actual
    return float(np.mean(percentage_errors) * 100) # Returned as a percentage

def evaluate_model(name: str, y_true: np.ndarray,
                   y_pred: np.ndarray) -> dict:
    """Compute all metrics for one model into a serializable dict."""
    calib = calibration(y_true, y_pred)
    return {
        "model": name,
        "gini": actuarial_gini(y_true, y_pred),
        "calibration_error_pct": calibration_error(calib),
        "decile_lift": decile_lift(y_true, y_pred).to_dict(orient="records"),
        "calibration": calib.to_dict(orient="records"),
    }