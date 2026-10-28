"""Entrena y serializa el pipeline de predicción de calificación final."""

from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "StudentsPerformance.csv"
MODEL_PATH = Path(__file__).resolve().parent / "student-score-v1.joblib"
TARGET_COLUMN = "Final_Exam_Score"

N_ESTIMATORS = 50
MAX_DEPTH = 14
MIN_SAMPLES_LEAF = 10
MAX_FEATURES = 0.8
JOBLIB_COMPRESSION = 3


def build_pipeline(X_train: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    """Crea el pipeline con las mismas reglas del modelo original."""
    categorical_cols = [
        column
        for column in X_train.columns
        if X_train[column].nunique() < 10 and X_train[column].dtype == "object"
    ]
    numerical_cols = [
        column
        for column in X_train.columns
        if X_train[column].dtype in ("int64", "float64")
    ]
    feature_columns = categorical_cols + numerical_cols

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="constant"), numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    max_depth=MAX_DEPTH,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    max_features=MAX_FEATURES,
                    random_state=0,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline, feature_columns


def train() -> None:
    student_data = pd.read_csv(DATA_PATH)
    y = student_data[TARGET_COLUMN]
    X = student_data.drop(columns=[TARGET_COLUMN])
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    pipeline, feature_columns = build_pipeline(X_train_full)
    pipeline.fit(X_train_full[feature_columns], y_train)

    predictions = pipeline.predict(X_valid_full[feature_columns])
    mae = mean_absolute_error(y_valid, predictions)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    dump(pipeline, MODEL_PATH, compress=JOBLIB_COMPRESSION)
    print(f"MAE: {mae:.4f}")


if __name__ == "__main__":
    train()
