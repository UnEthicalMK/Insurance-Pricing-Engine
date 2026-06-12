"""Central configuration. No magic numbers elsewhere."""
from __future__ import annotations
from pathlib import Path

RANDOM_SEED: int = 42
N_POLICIES: int = 100_000

ROOT = Path(__file__).resolve().parents[1]
DATA_SIM = ROOT / "data" / "simulated"
DATA_PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

RAW_CSV = DATA_SIM / "insurance_data.csv"
GLM_PRED_CSV = DATA_PROC / "glm_predictions.csv"
TWEEDIE_PRED_CSV = DATA_PROC / "tweedie_predictions.csv"
TWEEDIE_MODEL = DATA_PROC / "tweedie_model.joblib"
METRICS_JSON = REPORTS / "metrics_summary.json"
FINAL_REPORT = REPORTS / "final_report.md"

FEATURES = [
    "driver_age", "vehicle_value", "vehicle_age",
    "annual_mileage", "driving_experience", "region",
]
CATEGORICAL = ["region"]
REGIONS = ["north", "south", "east", "west", "central"]

TEST_SIZE: float = 0.2
N_DECILES: int = 10
N_CALIB_BUCKETS: int = 10

TWEEDIE_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}


def ensure_dirs() -> None:
    """Create all output directories if missing."""
    for d in (DATA_SIM, DATA_PROC, REPORTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
