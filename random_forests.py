# random_forests.py
from pathlib import Path
import pandas as pd  
import joblib        

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------
# 1) Data loading & cleaning
# ----------------------------
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
        df[col] = df[col].clip(low, high)  # Fix: complete the capping logic

    return df


# ----------------------------
# 2) Pipeline builder
# ----------------------------
def build_pipeline() -> tuple[Pipeline, list[str]]:
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

    # Base RF (will be overridden if tuning=True)
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

    features = CATEG + NUM  # Fix: return the features list
    return pipe, features


# ----------------------------
# 3) Optional compact GridSearch
# ----------------------------
def tune_random_forest(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Small, fast grid like your notebook version:
    1 × 2 × 1 × 2 × 1 × 2 = 8 combos × 3 folds = 24 fits
    """
    param_grid = {
        "model__n_estimators": [100],
        "model__max_depth": [None, 20],
        "model__min_samples_split": [2],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt"],
        "model__bootstrap": [True, False]
    }

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="r2",
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    gs.fit(X_train, y_train)

    print("Best RF params:", gs.best_params_)
    print("Best RF CV R²:", gs.best_score_)

    # Optionally refit with a larger forest for accuracy using the best params:
    best_pipe = gs.best_estimator_
    best_rf: RandomForestRegressor = best_pipe.named_steps["model"]
    # bump trees for final model if you want a stronger fit
    best_rf.set_params(n_estimators=300, n_jobs=-1, random_state=42)
    best_pipe.fit(X_train, y_train)

    return best_pipe


# ----------------------------
# 4) Train entrypoint
# ----------------------------
def train(
    csv_path: str = "diamonds.csv",
    out_path: str = "diamond_price_pipeline.joblib",
    tuning: bool = True  # turn on to use the compact GridSearchCV
):
    # Load & clean
    df = load_and_clean(csv_path)

    TARGET = "price"
    pipe, features = build_pipeline()

    X = df[features].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, shuffle=True
    )

    # Fit (with or without tuning)
    if tuning:
        pipe = tune_random_forest(pipe, X_train, y_train)  # Fix: complete the if block
    else:
        pipe.fit(X_train, y_train)  # Fix: complete the else block

    # Evaluate
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Random Forest — MAE: {mae:,.2f} | MSE: {mse:,.2f} | R²: {r2:.4f}")

    # Save bundle
    bundle = {
        "pipe": pipe,
        "features": features,
        "target": TARGET,
        "model": "RandomForestRegressor",
        "version": 2,
        "tuning": tuning
    }
    joblib.dump(bundle, out_path)
    print(f"Saved pipeline to: {Path(out_path).resolve()}")


if __name__ == "__main__":
    train()