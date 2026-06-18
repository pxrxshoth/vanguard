"""
test1_ultra_simple.py - Ultra Simple FD001 Evaluation
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error

# ============================================================
# CONFIG
# ============================================================

base_path = Path("C:/Users/acer/Desktop/Vangaurd")

print("=" * 60)
print("ULTRA SIMPLE TEST")
print("=" * 60)

# ============================================================
# LOAD TEST DATA
# ============================================================

column_names = (
    ["unit_number", "time",
     "op_settings1", "op_settings2", "op_settings3"]
    + [f"sensor{i}" for i in range(1, 22)]
)

test_df = pd.read_csv(
    base_path / "data/raw/files/test_FD001.txt",
    sep=r"\s+",
    header=None,
    names=column_names,
    engine="python"
)

print(f"Loaded test data: {test_df.shape}")

# ============================================================
# REMOVE USELESS FEATURES
# ============================================================

useless_features = [
    "sensor1",
    "sensor5",
    "sensor10",
    "sensor16",
    "sensor18",
    "sensor19",
    "op_settings3"
]

test_df = test_df.drop(
    columns=useless_features,
    errors="ignore"
)

print(f"After feature removal: {test_df.shape}")

# ============================================================
# CREATE ROLLING FEATURES
# ============================================================

sensor_cols = [
    col for col in test_df.columns
    if col.startswith("sensor")
]

window_size = 20

print("\nCreating rolling features...")

for sensor in sensor_cols:

    # Expanding mean
    test_df[f"{sensor}_roll_mean"] = (
        test_df.groupby("unit_number")[sensor]
        .transform(
            lambda x: x.expanding(
                min_periods=1
            ).mean()
        )
    )

    # Rolling std
    test_df[f"{sensor}_roll_std"] = (
        test_df.groupby("unit_number")[sensor]
        .transform(
            lambda x:
            x.rolling(
                window=window_size,
                min_periods=2
            )
            .std()
            .bfill()
            .fillna(0)
        )
    )

print(f"Feature matrix shape: {test_df.shape}")

# ============================================================
# LOAD TRAINING FEATURE LIST
# ============================================================

train_data = pd.read_csv(
    base_path /
    "data/processed/processed_train_FD001_final.csv"
)

train_features = [
    col
    for col in train_data.columns
    if col not in [
        "unit_number",
        "time",
        "RUL"
    ]
]

print(f"Training features: {len(train_features)}")

# ============================================================
# ALIGN TEST FEATURES
# ============================================================

X_test = test_df.drop(
    columns=["unit_number", "time"]
)

X_test = X_test.reindex(
    columns=train_features,
    fill_value=0
)

print(f"Aligned test shape: {X_test.shape}")

# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load(
    base_path / "models/scaler_final.pkl"
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

print("Scaling complete")

# ============================================================
# GET FINAL CYCLE OF EACH ENGINE
# ============================================================

final_rows = (
    test_df.groupby("unit_number")
    .tail(1)
    .index
)

X_final = X_test_scaled.loc[final_rows].values

print(f"Final engine samples: {X_final.shape}")

# ============================================================
# LOAD TRUE RUL
# ============================================================

true_rul = pd.read_csv(
    base_path / "data/raw/files/RUL_FD001.txt",
    sep=r"\s+",
    header=None,
    usecols=[0]
)

true_rul = true_rul.iloc[:, 0].values

# Match training cap
true_rul = np.clip(
    true_rul,
    0,
    125
)

print(f"True RUL samples: {len(true_rul)}")

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    base_path / "models/rul_regressor.joblib"
)

print("Model loaded")

# ============================================================
# PREDICT
# ============================================================

predictions = model.predict(X_final)

predictions = np.clip(
    predictions,
    0,
    125
)

# ============================================================
# METRICS
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        true_rul,
        predictions
    )
)

errors = predictions - true_rul

nasa_penalty = np.where(
    errors < 0,
    np.exp(-errors / 13) - 1,
    np.exp(errors / 10) - 1
)

nasa_score = np.sum(nasa_penalty)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"RMSE        : {rmse:.2f} cycles")
print(f"NASA Score  : {nasa_score:.2f}")
print(f"Mean Error  : {np.mean(errors):.2f}")
print(f"Std Error   : {np.std(errors):.2f}")

print(
    f"Early Predictions : "
    f"{np.sum(errors < 0)}"
)

print(
    f"Late Predictions  : "
    f"{np.sum(errors > 0)}"
)

print("\nSample Predictions")

for i in range(min(10, len(predictions))):
    print(
        f"Engine {i+1:3d} | "
        f"True={true_rul[i]:6.1f} | "
        f"Pred={predictions[i]:6.1f} | "
        f"Error={errors[i]:6.1f}"
    )

print("\n" + "=" * 60)

if rmse < 15:
    print(" TARGET ACHIEVED (RMSE < 15)")
elif rmse < 20:
    print(" GOOD PERFORMANCE (RMSE < 20)")
elif rmse < 25:
    print(" DECENT PERFORMANCE (RMSE < 25)")
else:
    print(" NEEDS IMPROVEMENT")

print("=" * 60)