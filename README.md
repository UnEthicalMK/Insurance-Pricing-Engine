# Quantitative Insurance Pricing Engine
### Actuarial GLM vs. Gradient-Boosted Tweedie — A Frequency-Severity Risk Pricing Benchmark

> A reproducible quantitative research prototype that pits a classical **Poisson-Gamma GLM** against a **LightGBM model trained under Tweedie loss** on 100,000 synthetic motor-insurance policies.
> The core thesis: *Can machine learning outperform traditional actuarial statistics at both ranking risk and pricing it?*

---

## 1. Executive Summary

This project simulates a portfolio of 100,000 policies with a known, deliberately non-linear risk structure (a U-shaped driver-age curve, mileage/vehicle-age loadings, and regional effects) alongside a zero-inflated, heavy-tailed loss distribution. Two pricing engines are trained on the identical dataset and benchmarked strictly out-of-sample on the metrics prioritized by actuarial and quantitative pricing desks: **risk discrimination** (Actuarial Gini) and **pricing accuracy** (Volume-Weighted Calibration Error).

The results capture a fundamental trade-off in financial modeling—**predictive ranking power vs. absolute pricing stability:**

| Model | Actuarial Gini (Higher = Better Ranking) | Calibration Error WMAPE (Lower = Better Pricing) |
| :--- | :--- | :--- |
| **Traditional GLM** (Poisson-Gamma) | 0.2249 | **3.50%** |
| **LightGBM** (Tweedie Loss) | **0.2499** | 10.62% |

**Verdict:** LightGBM is the superior **underwriter** (accurately identifying high-risk profiles), while the GLM is the superior **accountant** (collecting the correct aggregate premium to maintain solvency). Neither model is optimally deployed in isolation.

---

## 2. The Mathematical Problem

Pure premium (the expected loss cost per policy) is the product of two distinct random processes:

$$\mathbb{E}[\text{Pure Premium}] = \mathbb{E}[\text{Claim Frequency}] \times \mathbb{E}[\text{Claim Severity}]$$

Modeling this financial data presents three significant quantitative hurdles:
* **Zero-Inflation:** The vast majority of policyholders never file a claim, creating a massive point mass at exactly zero.
* **Heavy-Tails:** The minority who do claim generate losses that follow a highly skewed, long-tailed distribution.
* **Non-Linearity:** Underlying risk drivers (such as age) do not scale linearly or monotonically.

The **Tweedie distribution** (a compound Poisson-Gamma distribution parameterised by a variance power $1 < p < 2$) serves as the theoretical bridge for this structure, allowing the Gradient Booster to model the pure premium directly in a single step.

---

## 3. Methodology

### Synthetic Data Engine
A controlled stochastic simulator (`src/simulation/create_dataset.py`) generates 100,000 policies with a hidden ground-truth risk function, establishing an absolute mathematical target to evaluate model quality.
* **Features:** `driver_age` (Beta), `vehicle_value` (Lognormal), `vehicle_age`, `annual_mileage`, `region`, and derived `driving_experience`.
* **Frequency:** Modeled via $N \sim \text{Poisson}(\lambda)$, where $\lambda$ encodes a non-linear U-shaped age risk alongside mileage and regional scalar loadings.
* **Severity:** Modeled via $X \sim \text{Gamma}(\alpha, \beta)$, dynamically scaled by `vehicle_value` and forced to exactly zero when no claim occurs.

### Actuarial Baseline (Two-Step GLM)
* **Poisson GLM** applied to claim counts via a logarithmic link function to predict expected frequency.
* **Gamma GLM** applied exclusively to positive-claim rows to predict expected severity.
* **Combined Output:** $\text{Expected Premium} = \text{Predicted Frequency} \times \text{Predicted Severity}$.

### Machine Learning Challenger
* **LightGBM Regressor** utilizing the `tweedie` objective function ($p=1.5$).
* Implements native categorical splitting and stochastic gradient boosting parameters to prevent in-sample memorization of the simulated risk surface.

---

## 4. Results & Interpretation

**Risk Discrimination (LightGBM Wins)**
The tree architecture naturally partitions the U-shaped age curve, successfully isolating the high-risk profiles of both young and old drivers without requiring manual splines or polynomial basis expansions—a structural blind spot for standard linear GLMs. This architectural advantage lifts the out-of-sample Actuarial Gini from 0.2249 to 0.2499, representing a materially stronger ability to segment and rank the riskiest policies.

**Pricing Accuracy (GLM Wins)**
The Tweedie-boosted model is inherently sensitive to extreme long-tail claims, causing it to misprice absolute currency targets at the edges of the risk pool (resulting in a 10.62% Weighted Mean Absolute Percentage Error). Conversely, the rigid statistical constraints of the Poisson-Gamma structure inherently anchor predictions to population means, delivering a vastly superior calibration error of 3.50%. Deploying LightGBM directly to production would risk systematically overcharging safe drivers (causing churn) and undercharging risky ones (causing capital loss).

**Sanity Checks**
* **Exploratory Data Analysis (EDA):** Log-scaled histograms confirm the successful generation of a zero-inflated, heavy-tailed target variable.
* **Feature Importance:** Gain-based importance metrics confirm the LightGBM model successfully reverse-engineered the simulator's logic, surfacing `driver_age` and `vehicle_value` as the dominant quantitative risk factors.

---

## 5. Strategic Conclusion

> The enterprise pattern for quantitative risk pricing is a **hybrid architecture**. Quantitative developers should leverage Stochastic Gradient Boosting to discover complex, non-linear feature interactions, and subsequently feed those interactions as structural inputs into a rigidly calibrated Generalized Linear Model—or apply a post-hoc calibration layer (Isotonic Regression / Platt Scaling) directly onto the ML outputs to close the pricing gap.

---
