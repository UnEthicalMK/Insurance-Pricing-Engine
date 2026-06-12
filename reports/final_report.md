# Insurance Pricing Prototype - Final Report

## Problem statement
Estimate the pure premium (expected loss) per policy and compare an actuarial
GLM approach (Poisson frequency x Gamma severity) against a LightGBM model
trained with Tweedie loss.

## Data generation process
100,000 synthetic policies. Frequency driven by a non-linear (U-shaped) age risk,
mileage, vehicle age and region effects via a Poisson process; severity driven by a
Gamma process scaled by vehicle value. Target: pure_premium = claim_count * claim_cost.

## Model comparison (Out-of-Sample)
| Model | Actuarial Gini | Calibration Error (WMAPE) |
| :--- | :--- | :--- |
| **GLM** | 0.2249 | 3.50% |
| **Tweedie** | 0.2499 | 10.62% |


## Evaluation results
Metrics computed via actuarial Gini (fast dot-product formulation), decile lift, and volume-weighted quantile calibration evaluated strictly on a 20% unseen holdout set.
See `reports/figures/` for Gini, lift, calibration curves, and feature importance.

## Business interpretation
A higher Gini means better risk discrimination (ranking riskier policies higher).
Lower calibration error means predicted premiums match observed losses more closely,
which is essential for adequate, fair, and solvent pricing.
