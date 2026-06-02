import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from data_manager import read_all, FEATURE_COLS, TARGET_COL

MODEL_PATH = Path(__file__).parent.parent / "models" / "rf_model.joblib"


def train(n_estimators: int = 100, random_state: int = 42) -> dict:
    df = read_all()
    X, y = df[FEATURE_COLS], df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    y_pred = model.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "n_samples": len(df),
        "model_path": str(MODEL_PATH),
    }


def load_model() -> RandomForestRegressor:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Aucun modèle trouvé à {MODEL_PATH}. Lancez l'entraînement d'abord."
        )
    return joblib.load(MODEL_PATH)


def predict(features: dict) -> float:
    model = load_model()
    X = pd.DataFrame([features])[FEATURE_COLS]
    return float(model.predict(X)[0])
