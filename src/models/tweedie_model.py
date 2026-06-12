"""Phase 4: LightGBM Tweedie model for pure premium."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from src import config


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals as pandas category dtype for LightGBM."""
    out = df[config.FEATURES].copy()
    for col in config.CATEGORICAL:
        out[col] = out[col].astype("category")
    return out


def train_tweedie(df: pd.DataFrame) -> tuple[LGBMRegressor, pd.DataFrame]:
    """Train LightGBM with Tweedie loss and return model + prediction frame."""
    X = _prepare(df)
    y = df["pure_premium"]
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    model = LGBMRegressor(**config.TWEEDIE_PARAMS)
    model.fit(X_train, y_train, categorical_feature=config.CATEGORICAL)

    preds = pd.DataFrame({
        "tweedie_pure_premium": model.predict(X_test),
        "actual_pure_premium": y_test.to_numpy(),
    }, index=idx_test)
    return model, preds


def main() -> None:
    config.ensure_dirs()
    df = pd.read_csv(config.RAW_CSV)
    model, preds = train_tweedie(df)
    joblib.dump(model, config.TWEEDIE_MODEL)
    preds.to_csv(config.TWEEDIE_PRED_CSV, index=False)
    print(f"Tweedie model -> {config.TWEEDIE_MODEL}")
    print(f"Tweedie predictions -> {config.TWEEDIE_PRED_CSV}")


if __name__ == "__main__":
    main()
