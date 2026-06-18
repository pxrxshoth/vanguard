import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path

base_path = Path("C:/Users/acer/Desktop/Vangaurd")

train_df = pd.read_csv(base_path / "data/processed/processed_train_FD001.csv")
exclude_features = ['unit_number', 'time', 'RUL', 'raw_RUL', 'anomaly_score']
train_features = [col for col in train_df.columns if col not in exclude_features]

columns = ['unit_number', 'time', 'op_setting1', 'op_setting2', 'op_setting3'] + [f'sensor{i}' for i in range(1, 22)]
raw_train_df = pd.read_csv(base_path / 'data/raw/files/train_FD001.txt', sep=r"\s+", header=None, names=columns)

constant_sensors = ['sensor1', 'sensor5', 'sensor6', 'sensor10', 'sensor16', 'sensor18', 'sensor19']
raw_train_df = raw_train_df.drop(columns=constant_sensors, errors='ignore')

sensor_cols = [col for col in raw_train_df.columns if 'sensor' in col]
window = 20

for sensor in sensor_cols:
    roll_mean = raw_train_df.groupby('unit_number')[sensor].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    ).values
    raw_train_df[f'{sensor}_mean'] = roll_mean
    
    roll_std = raw_train_df.groupby('unit_number')[sensor].transform(
        lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
    ).values
    raw_train_df[f'{sensor}_std'] = roll_std

raw_train_df = raw_train_df.rename(columns={
    'op_setting1': 'op_settings1',
    'op_setting2': 'op_settings2',
    'op_setting3': 'op_settings3'
})

X_train_raw = raw_train_df.drop(columns=['unit_number', 'time'], errors='ignore')
X_train_raw = X_train_raw.reindex(columns=train_features, fill_value=0)

scaler = MinMaxScaler()
scaler.fit(X_train_raw)

scaler_path = base_path / "models/scaler_no_rul.pkl"
joblib.dump(scaler, scaler_path)
joblib.dump(scaler, base_path / "models/scaler.pkl")

