"""Phase 7: LightGBM feature importance (no SHAP)."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMRegressor
from src import config

def plot_importance(model: LGBMRegressor) -> pd.DataFrame:
    """Plot and return LightGBM feature importances based on Gain."""
    
    # Extract 'gain' rather than the default 'split'
    imp = pd.DataFrame({
        "feature": model.booster_.feature_name(),
        "importance": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=True)

    # Normalize gain to percentages for easier business interpretation
    imp["importance"] = imp["importance"] / imp["importance"].sum() * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["feature"], imp["importance"], color="#4C72B0", edgecolor="black")
    
    ax.set_title("LightGBM Feature Importance (Gain)", fontweight="bold")
    ax.set_xlabel("Relative Contribution to Premium Pricing (%)")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(config.FIGURES / "feature_importance.png", dpi=300)
    plt.close(fig)
    
    return imp