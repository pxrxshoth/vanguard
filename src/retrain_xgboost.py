"""
retrain_final_model.py - Train model on the final data
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error

base_path = Path("C:/Users/acer/Desktop/Vangaurd")

print("="*60)
print("RETRAINING MODEL ON FINAL DATA")
print("="*60)

# Load the final data
print("\n1. Loading final training data...")
data = pd.read_csv(base_path / "data/processed/processed_train_FD001_final.csv")
print(f"   Data shape: {data.shape}")

# Prepare features
exclude_cols = ['unit_number', 'time', 'RUL']
X = data.drop(columns=exclude_cols)
y = data['RUL']

print(f"   Features: {X.shape[1]}")
print(f"   Target range: {y.min()} to {y.max()}")

# Split by engine
print("\n2. Splitting by engine...")
train_mask = data['unit_number'] <= 80
val_mask = data['unit_number'] > 80

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]

print(f"   Training: {len(X_train)} samples (engines 1-80)")
print(f"   Validation: {len(X_val)} samples (engines 81-100)")

# Train model
print("\n3. Training XGBoost...")
model = XGBRegressor(
    n_estimators=250,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1
)

# Sample weights for critical region
weights = np.where(y_train < 30, 5.0, 1.0)
weights = np.where(y_train < 15, 10.0, weights)

model.fit(X_train, y_train, sample_weight=weights, verbose=False)

# Evaluate
print("\n4. Evaluating on validation set...")
pred_val = model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))

# Check critical region
val_critical = y_val < 30
if val_critical.sum() > 0:
    critical_rmse = np.sqrt(mean_squared_error(y_val[val_critical], pred_val[val_critical]))
    print(f"   Overall RMSE: {rmse_val:.4f}")
    print(f"   Critical RMSE (RUL<30): {critical_rmse:.4f}")
else:
    print(f"   Validation RMSE: {rmse_val:.4f}")

# Retrain on full data
print("\n5. Retraining on full dataset...")
full_weights = np.where(y < 30, 5.0, 1.0)
full_weights = np.where(y < 15, 10.0, full_weights)

model.fit(X, y, sample_weight=full_weights, verbose=False)

# Save model
model_path = base_path / "models/rul_regressor.joblib"
joblib.dump(model, model_path)
print(f"\n[SUCCESS] Model saved to: {model_path}")

print("\n" + "="*60)
if rmse_val < 15:
    print(f"[SUCCESS] Validation RMSE = {rmse_val:.4f} (<15)")
else:
    print(f"[INFO] Validation RMSE = {rmse_val:.4f} (target: <15)")
print("="*60)