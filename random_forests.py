# random_forests.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop accidental index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Remove impossible measurements
    df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]

    # Feature engineering: volume
    df["volume"] = df["x"] * df["y"] * df["z"]

    # IQR capping for noisy numerics (don't cap carat)
    for col in ["volume", "depth", "table"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        df[col] = df[col].clip(low, high)

    return df


def build_pipeline() -> Pipeline:
    # Columns
    CATEG = ["cut", "color", "clarity"]
    NUM = ["carat", "depth", "table", "x", "y", "z", "volume"]

    # Gemology (worst -> best)
    cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    color_order = ["J", "I", "H", "G", "F", "E", "D"]
    clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

    encoder = OrdinalEncoder(
        categories=[cut_order, color_order, clarity_order],
        dtype=float
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("ord", encoder, CATEG),
            ("num", "passthrough", NUM),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    rf = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("prep", preprocess),
        ("model", rf),
    ])

    return pipe, CATEG + NUM


def train(csv_path: str = "diamonds.csv", out_path: str = "diamond_price_pipeline.joblib"):
    # 1) Load & clean
    df = load_and_clean(csv_path)

    TARGET = "price"
    pipe, features = build_pipeline()

    X = df[features].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, shuffle=True
    )

    # 2) Fit
    pipe.fit(X_train, y_train)

    # 3) Evaluate
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Random Forest — MAE: {mae:,.2f} | MSE: {mse:,.2f} | R²: {r2:.4f}")

    # 4) Save single artifact (pipeline + metadata)
    bundle = {
        "pipe": pipe,
        "features": features,
        "target": TARGET,
        "model": "RandomForestRegressor",
        "version": 1,
    }
    joblib.dump(bundle, out_path)
    print(f"Saved pipeline to: {Path(out_path).resolve()}")


if __name__ == "__main__":
    train()
