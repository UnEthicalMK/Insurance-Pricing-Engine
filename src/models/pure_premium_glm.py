"""Phase 3: combine frequency x severity into a pure-premium prediction."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src import config
from src.models.gamma_glm import predict_severity, train_severity
from src.models.poisson_glm import predict_frequency, train_frequency


def train_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Train both GLMs on the training set and predict on the holdout set."""
    
    # 1. Split the data exactly the same way as the Tweedie model
    df_train, df_test = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    
    # 2. Train ONLY on the training subset
    freq_model = train_frequency(df_train)
    sev_model = train_severity(df_train)
    
    # 3. Predict ONLY on the unseen holdout subset
    freq = predict_frequency(freq_model, df_test)
    sev = predict_severity(sev_model, df_test)
    
    out = pd.DataFrame({
        "freq_pred": freq,
        "sev_pred": sev,
        "glm_pure_premium": freq * sev,
        "actual_pure_premium": df_test["pure_premium"].to_numpy(),
    }, index=df_test.index) # Preserve the original test indices
    
    return out


def main() -> None:
    config.ensure_dirs()
    df = pd.read_csv(config.RAW_CSV)
    preds = train_and_predict(df)
    preds.to_csv(config.GLM_PRED_CSV, index=False)
    print(f"GLM OUT-OF-SAMPLE predictions saved -> {config.GLM_PRED_CSV}")


if __name__ == "__main__":
    main()