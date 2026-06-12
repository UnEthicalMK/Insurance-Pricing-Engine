"""Phase 6: evaluation visualizations."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src import config
from src.evaluation.metrics import calibration

def gini_curve(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    """Plot the cumulative-gains (Lorenz) curve."""
    order = np.argsort(y_pred)
    cum = np.cumsum(np.asarray(y_true)[order]) / np.sum(y_true)
    x = np.linspace(0, 1, len(cum))
    
    fig, ax = plt.subplots(figsize=(7, 6))
    # Model curve
    ax.plot(x, cum, label=f"{name} Model", color="#4C72B0", linewidth=2)
    # Line of equality (No Skill)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random Assignment")
    
    ax.set_title(f"Actuarial Gini (Lorenz Curve) - {name}", fontweight="bold")
    ax.set_xlabel("Cumulative Percentage of Policies (Sorted by Predicted Risk)")
    ax.set_ylabel("Cumulative Percentage of Actual Losses")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(config.FIGURES / f"gini_{name}.png", dpi=300)
    plt.close(fig)

def lift_chart(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    """Plot decile lift chart comparing actual vs predicted risk."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    # Force 1-10 decile labels for cleaner x-axis
    df["decile"] = pd.qcut(df["y_pred"].rank(method="first"), 10, labels=False) + 1
    
    # Calculate both actual and predicted means per decile
    agg = df.groupby("decile").mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    # Actual losses as bars
    ax.bar(agg["decile"], agg["y_true"], color="#8DA0CB", alpha=0.7, edgecolor="black", label="Actual Average Loss")
    # Predicted losses as an overlay line
    ax.plot(agg["decile"], agg["y_pred"], color="#FC8D62", marker="o", linewidth=2, markersize=8, label="Predicted Average Premium")

    ax.set_title(f"Decile Lift Chart - {name}", fontweight="bold")
    ax.set_xlabel("Predicted Risk Decile (1 = Safest, 10 = Riskiest)")
    ax.set_ylabel("Pure Premium (Currency)")
    ax.set_xticks(range(1, 11))
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(config.FIGURES / f"lift_{name}.png", dpi=300)
    plt.close(fig)

def calibration_curve(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    """Plot predicted vs actual calibration curve."""
    c = calibration(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    
    ax.plot(c["predicted"], c["actual"], "o-", color="#66C2A5", linewidth=2, markersize=6)
    
    # Dynamic limits to ensure perfect square scaling
    lim_max = max(c["predicted"].max(), c["actual"].max())
    lim_max += lim_max * 0.05 # Add 5% padding
    
    ax.plot([0, lim_max], [0, lim_max], "--", color="grey", label="Perfect Calibration")
    
    ax.set_xlabel("Predicted Expected Loss")
    ax.set_ylabel("Actual Expected Loss")
    ax.set_title(f"Calibration Curve - {name}", fontweight="bold")
    ax.set_xlim([0, lim_max])
    ax.set_ylim([0, lim_max])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(config.FIGURES / f"calibration_{name}.png", dpi=300)
    plt.close(fig)

def generate_all(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    """Generate every evaluation plot for one model."""
    config.ensure_dirs()
    gini_curve(y_true, y_pred, name)
    lift_chart(y_true, y_pred, name)
    calibration_curve(y_true, y_pred, name)