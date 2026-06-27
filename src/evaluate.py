import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error

BASE_PATH = Path(__file__).resolve().parent.parent

COLUMN_NAMES = (
    ["unit_number", "time", "op_settings1", "op_settings2", "op_settings3"]
    + [f"sensor{i}" for i in range(1, 22)]
)

USELESS_FEATURES = [
    "sensor1", "sensor5", "sensor10",
    "sensor16", "sensor18", "sensor19",
    "op_settings3",
]

RUL_CAP = 125
WINDOW_SIZE = 20


def load_test_data() -> pd.DataFrame:
    """Load and parse the raw FD001 test file."""
    path = BASE_PATH / "data/raw/files/test_FD001.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES, engine="python")
    print(f"[evaluate] Loaded test data: {df.shape}")
    return df


def drop_useless_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop low-variance / non-informative sensor columns."""
    df = df.drop(columns=USELESS_FEATURES, errors="ignore")
    print(f"[evaluate] After feature removal: {df.shape}")
    return df


def create_rolling_features(df: pd.DataFrame, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    """Add per-engine expanding mean and rolling std for each sensor."""
    sensor_cols = [c for c in df.columns if c.startswith("sensor")]
    print(f"[evaluate] Creating rolling features for {len(sensor_cols)} sensors...")
    for sensor in sensor_cols:
        df[f"{sensor}_roll_mean"] = (
            df.groupby("unit_number")[sensor]
            .transform(lambda x: x.expanding(min_periods=1).mean())
        )
        df[f"{sensor}_roll_std"] = (
            df.groupby("unit_number")[sensor]
            .transform(
                lambda x: x.rolling(window=window_size, min_periods=2)
                .std().bfill().fillna(0)
            )
        )
    print(f"[evaluate] Feature matrix shape: {df.shape}")
    return df


def get_train_features() -> list:
    """Return the ordered list of feature columns used during training."""
    train_data = pd.read_csv(BASE_PATH / "data/processed/processed_train_FD001_final.csv")
    features = [c for c in train_data.columns if c not in ["unit_number", "time", "RUL"]]
    print(f"[evaluate] Training features: {len(features)}")
    return features


def align_features(df: pd.DataFrame, train_features: list) -> np.ndarray:
    """Align test features to training schema, then scale."""
    X_test = df.drop(columns=["unit_number", "time"])
    X_test = X_test.reindex(columns=train_features, fill_value=0)
    print(f"[evaluate] Aligned test shape: {X_test.shape}")

    scaler = joblib.load(BASE_PATH / "models/scaler_final.pkl")
    X_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    print("[evaluate] Scaling complete.")
    return X_scaled


def get_final_cycle_samples(df_raw: pd.DataFrame, X_scaled: pd.DataFrame) -> np.ndarray:
    """Select only the last cycle of each engine unit."""
    final_rows = df_raw.groupby("unit_number").tail(1).index
    X_final = X_scaled.loc[final_rows].values
    print(f"[evaluate] Final engine samples: {X_final.shape}")
    return X_final


def load_true_rul() -> np.ndarray:
    """Load and cap the ground-truth RUL values."""
    true_rul = pd.read_csv(
        BASE_PATH / "data/raw/files/RUL_FD001.txt",
        sep=r"\s+", header=None, usecols=[0]
    ).iloc[:, 0].values
    true_rul = np.clip(true_rul, 0, RUL_CAP)
    print(f"[evaluate] True RUL samples: {len(true_rul)}")
    return true_rul


def evaluate(predictions: np.ndarray, true_rul: np.ndarray) -> dict:
    """Compute RMSE, NASA scoring function, and error statistics."""
    errors = predictions - true_rul
    rmse = np.sqrt(mean_squared_error(true_rul, predictions))

    nasa_penalty = np.where(
        errors < 0,
        np.exp(-errors / 13) - 1,
        np.exp(errors / 10) - 1,
    )
    nasa_score = float(np.sum(nasa_penalty))

    return {
        "rmse": rmse,
        "nasa_score": nasa_score,
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "early_predictions": int(np.sum(errors < 0)),
        "late_predictions": int(np.sum(errors > 0)),
    }


def print_report(metrics: dict, predictions: np.ndarray, true_rul: np.ndarray) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("VANGUARD — RUL EVALUATION REPORT (FD001)")
    print("=" * 60)
    print(f"  RMSE             : {metrics['rmse']:.2f} cycles")
    print(f"  NASA Score       : {metrics['nasa_score']:.2f}")
    print(f"  Mean Error       : {metrics['mean_error']:.2f}")
    print(f"  Std Error        : {metrics['std_error']:.2f}")
    print(f"  Early Predictions: {metrics['early_predictions']}")
    print(f"  Late  Predictions: {metrics['late_predictions']}")

    print("\n  Sample Predictions (first 10 engines):")
    errors = predictions - true_rul
    for i in range(min(10, len(predictions))):
        print(
            f"    Engine {i+1:3d} | "
            f"True={true_rul[i]:6.1f} | "
            f"Pred={predictions[i]:6.1f} | "
            f"Error={errors[i]:+6.1f}"
        )

    print("\n" + "=" * 60)
    rmse = metrics["rmse"]
    if rmse < 15:
        print("  ✅  TARGET ACHIEVED  (RMSE < 15 cycles)")
    elif rmse < 20:
        print("  🟡  GOOD PERFORMANCE (RMSE < 20 cycles)")
    elif rmse < 25:
        print("  🟠  DECENT           (RMSE < 25 cycles)")
    else:
        print("  ❌  NEEDS IMPROVEMENT")
    print("=" * 60)


def main() -> dict:
    """Full evaluation pipeline. Returns metrics dict."""
    df = load_test_data()
    df = drop_useless_features(df)
    df = create_rolling_features(df)

    train_features = get_train_features()
    X_scaled = align_features(df, train_features)
    X_final = get_final_cycle_samples(df, X_scaled)
    true_rul = load_true_rul()

    model = joblib.load(BASE_PATH / "models/rul_regressor.joblib")
    print("[evaluate] Model loaded.")

    predictions = np.clip(model.predict(X_final), 0, RUL_CAP)
    metrics = evaluate(predictions, true_rul)
    print_report(metrics, predictions, true_rul)
    return metrics


if __name__ == "__main__":
    main()