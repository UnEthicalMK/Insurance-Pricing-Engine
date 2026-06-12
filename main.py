"""Phase 8: end-to-end pipeline orchestration."""
from __future__ import annotations
import json
import pandas as pd
from src import config
from src.evaluation.feature_importance import plot_importance
from src.evaluation.metrics import evaluate_model
from src.models.pure_premium_glm import train_and_predict
from src.models.tweedie_model import train_tweedie
from src.simulation.create_dataset import generate_dataset
from src.visualization.eda import run_eda
from src.visualization import plots


def write_report(glm_eval: dict, tweedie_eval: dict) -> None:
    """Write the final markdown report (Phase 10)."""
    table = (
        "| Model | Actuarial Gini | Calibration Error (WMAPE) |\n"
        "| :--- | :--- | :--- |\n"
        f"| **GLM** | {glm_eval['gini']:.4f} | {glm_eval['calibration_error_pct']:.2f}% |\n"
        f"| **Tweedie** | {tweedie_eval['gini']:.4f} | {tweedie_eval['calibration_error_pct']:.2f}% |\n"
    )
    md = f"""# Insurance Pricing Prototype - Final Report

## Problem statement
Estimate the pure premium (expected loss) per policy and compare an actuarial
GLM approach (Poisson frequency x Gamma severity) against a LightGBM model
trained with Tweedie loss.

## Data generation process
{config.N_POLICIES:,} synthetic policies. Frequency driven by a non-linear (U-shaped) age risk,
mileage, vehicle age and region effects via a Poisson process; severity driven by a
Gamma process scaled by vehicle value. Target: pure_premium = claim_count * claim_cost.

## Model comparison (Out-of-Sample)
{table}

## Evaluation results
Metrics computed via actuarial Gini (fast dot-product formulation), decile lift, and volume-weighted quantile calibration evaluated strictly on a 20% unseen holdout set.
See `reports/figures/` for Gini, lift, calibration curves, and feature importance.

## Business interpretation
A higher Gini means better risk discrimination (ranking riskier policies higher).
Lower calibration error means predicted premiums match observed losses more closely,
which is essential for adequate, fair, and solvent pricing.
"""
    config.FINAL_REPORT.write_text(md)


def main() -> None:
    """Run all phases in order."""
    config.ensure_dirs()

    # 1. Generate dataset
    df = generate_dataset()
    df.to_csv(config.RAW_CSV, index=False)

    # 2. EDA
    run_eda(df)

    # 3. GLM models
    glm = train_and_predict(df)
    glm.to_csv(config.GLM_PRED_CSV, index=False)

    # 4. Tweedie model
    model, tweedie = train_tweedie(df)
    tweedie.to_csv(config.TWEEDIE_PRED_CSV, index=False)

    # 5-6. Evaluate + plots
    glm_eval = evaluate_model("GLM", glm["actual_pure_premium"], glm["glm_pure_premium"])
    tweedie_eval = evaluate_model(
        "Tweedie", tweedie["actual_pure_premium"], tweedie["tweedie_pure_premium"])
    plots.generate_all(glm["actual_pure_premium"], glm["glm_pure_premium"], "GLM")
    plots.generate_all(
        tweedie["actual_pure_premium"], tweedie["tweedie_pure_premium"], "Tweedie")

    # 7. Feature importance
    plot_importance(model)

    # 8. Save metrics + report
    config.METRICS_JSON.write_text(json.dumps(
        {"GLM": glm_eval, "Tweedie": tweedie_eval}, indent=2))
    write_report(glm_eval, tweedie_eval)
    print("Pipeline complete. See reports/ and data/processed/.")


if __name__ == "__main__":
    main()
