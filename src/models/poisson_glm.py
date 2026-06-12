"""Phase 3: Poisson GLM for claim frequency."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import config


def build_pipeline() -> Pipeline:
    """Build preprocessing + Poisson regressor pipeline."""
    numeric = [f for f in config.FEATURES if f not in config.CATEGORICAL]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("glm", PoissonRegressor(alpha=1e-4, max_iter=300))])


def train_frequency(df: pd.DataFrame) -> Pipeline:
    """Fit the Poisson frequency model on claim_count."""
    model = build_pipeline()
    model.fit(df[config.FEATURES], df["claim_count"])
    return model


def predict_frequency(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Predict expected claim frequency."""
    return model.predict(df[config.FEATURES])
