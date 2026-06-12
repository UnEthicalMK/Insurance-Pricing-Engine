"""Phase 3: Gamma GLM for claim severity (claim_count > 0 only)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import GammaRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import config


def build_pipeline() -> Pipeline:
    """Build preprocessing + Gamma regressor pipeline."""
    numeric = [f for f in config.FEATURES if f not in config.CATEGORICAL]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("glm", GammaRegressor(alpha=1e-4, max_iter=300))])


def train_severity(df: pd.DataFrame) -> Pipeline:
    """Fit the Gamma severity model on positive-claim rows."""
    positive = df[df["claim_count"] > 0]
    model = build_pipeline()
    model.fit(positive[config.FEATURES], positive["claim_cost"])
    return model


def predict_severity(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Predict expected claim severity."""
    return model.predict(df[config.FEATURES])
