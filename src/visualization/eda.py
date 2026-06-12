"""Phase 2: exploratory data analysis plots."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from src import config

def run_eda(df: pd.DataFrame) -> None:
    """Generate distribution plots and a correlation heatmap."""
    config.ensure_dirs()

    # 1. Frequency (Zero-Inflated)
    fig, ax = plt.subplots(figsize=(7, 4))
    df["claim_count"].value_counts().sort_index().plot(kind="bar", color="#4C72B0", edgecolor="black", ax=ax)
    ax.set_title("Claim Count Distribution (Zero Inflation)", fontweight="bold")
    ax.set_ylabel("Number of Policies")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "eda_claim_count.png", dpi=300)
    plt.close(fig)

    # 2. Severity (Heavy Tail - Log Scaled)
    fig, ax = plt.subplots(figsize=(7, 4))
    positive_costs = df.loc[df["claim_cost"] > 0, "claim_cost"]
    # Log-scale bins to properly visualize the Gamma tail
    bins = np.logspace(np.log10(positive_costs.min()), np.log10(positive_costs.max()), 60)
    ax.hist(positive_costs, bins=bins, color="#DD8452", edgecolor="black")
    ax.set_xscale("log")
    ax.set_title("Claim Cost Distribution (Log Scale / Heavy Tail)", fontweight="bold")
    ax.set_xlabel("Cost (Log Scale)")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "eda_claim_cost.png", dpi=300)
    plt.close(fig)

    # 3. Pure Premium (Heavy Tail - Log Scaled)
    fig, ax = plt.subplots(figsize=(7, 4))
    positive_premium = df.loc[df["pure_premium"] > 0, "pure_premium"]
    bins_prem = np.logspace(np.log10(positive_premium.min()), np.log10(positive_premium.max()), 60)
    ax.hist(positive_premium, bins=bins_prem, color="#55A868", edgecolor="black")
    ax.set_xscale("log")
    ax.set_title("Pure Premium Distribution (Log Scale)", fontweight="bold")
    ax.set_xlabel("Premium (Log Scale)")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "eda_pure_premium.png", dpi=300)
    plt.close(fig)

    # 4. Correlation Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.drop(columns=config.CATEGORICAL + ["claim_count", "claim_cost", "pure_premium"]).corr()
    # Mask the upper triangle for a cleaner quant aesthetic
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                vmin=-1, vmax=1, center=0, square=True, linewidths=.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "eda_correlation.png", dpi=300)
    plt.close(fig)
    
    print(f"EDA figures saved -> {config.FIGURES}")

def main() -> None:
    run_eda(pd.read_csv(config.RAW_CSV))

if __name__ == "__main__":
    main()