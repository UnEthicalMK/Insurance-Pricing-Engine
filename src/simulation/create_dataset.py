"""Phase 1: synthetic data engine (Poisson frequency + Gamma severity)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src import config


def _generate_features(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Generate raw policy features with realistic distributions."""
    # Skewed driver age: beta scaled to 18-80
    driver_age = (18 + rng.beta(2.0, 3.0, n) * 62).round().astype(int)
    vehicle_value = np.clip(rng.lognormal(mean=9.8, sigma=0.5, size=n), 2_000, 200_000)
    vehicle_age = rng.integers(1, 21, n)
    annual_mileage = rng.integers(5_000, 50_001, n)
    region = rng.choice(config.REGIONS, size=n)
    driving_experience = np.clip(driver_age - 18, 0, None)
    return pd.DataFrame({
        "driver_age": driver_age,
        "vehicle_value": vehicle_value.round(2),
        "vehicle_age": vehicle_age,
        "annual_mileage": annual_mileage,
        "region": region,
        "driving_experience": driving_experience,
    })


def _frequency_lambda(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Hidden Poisson risk function -> expected claim frequency lambda."""
    age = df["driver_age"].to_numpy()
    # U-shaped age risk: young and old drivers riskier
    age_risk = ((age - 45.0) / 25.0) ** 2
    mileage_risk = (df["annual_mileage"].to_numpy() - 5_000) / 45_000
    vehicle_age_risk = df["vehicle_age"].to_numpy() / 20.0
    region_effect = pd.Series(
        {"north": 0.0, "south": 0.1, "east": -0.1, "west": 0.2, "central": 0.05}
    ).reindex(df["region"]).to_numpy()

    linear = (
        -2.3
        + 0.8 * age_risk
        + 0.6 * mileage_risk
        + 0.5 * vehicle_age_risk
        + region_effect
    )
    return np.exp(linear)


def _severity(df: pd.DataFrame, claim_count: np.ndarray,
              rng: np.random.Generator) -> np.ndarray:
    """Gamma severity per policy; depends on vehicle_value. Zero if no claim."""
    has_claim = claim_count > 0
    mean_cost = 500 + 0.02 * df["vehicle_value"].to_numpy()
    shape = 2.0
    scale = mean_cost / shape
    cost = np.zeros(len(df))
    cost[has_claim] = rng.gamma(shape, scale[has_claim])
    return cost.round(2)


def generate_dataset(n: int = config.N_POLICIES,
                     seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Generate the full synthetic insurance dataset."""
    rng = np.random.default_rng(seed)
    df = _generate_features(rng, n)
    lam = _frequency_lambda(df, rng)
    df["claim_count"] = rng.poisson(lam)
    df["claim_cost"] = _severity(df, df["claim_count"].to_numpy(), rng)
    df["pure_premium"] = (df["claim_count"] * df["claim_cost"]).round(2)
    return df


def main() -> None:
    """Generate and persist the dataset."""
    config.ensure_dirs()
    df = generate_dataset()
    df.to_csv(config.RAW_CSV, index=False)
    print(f"Saved {len(df):,} rows -> {config.RAW_CSV}")
    print(f"Zero-claim share: {(df['claim_count'] == 0).mean():.1%}")


if __name__ == "__main__":
    main()
