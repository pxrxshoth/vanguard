import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib

base_path = Path("C:/Users/acer/Desktop/Vangaurd")

columns = ['unit_number', 'time', 'op_settings1', 'op_settings2', 'op_settings3'] + [f'sensor{i}' for i in range(1, 22)]
raw_data = pd.read_csv(base_path / 'data/raw/files/train_FD001.txt', sep=r"\s+", header=None, names=columns)



useless = ['sensor1', 'sensor5', 'sensor10', 'sensor16', 'sensor18', 'sensor19', 'op_settings3']
raw_data = raw_data.drop(columns=useless, errors='ignore')

sensor_cols = [col for col in raw_data.columns if 'sensor' in col]

window = 20

for sensor in sensor_cols:
    roll_mean = raw_data.groupby('unit_number')[sensor].transform(
        lambda x: x.expanding(min_periods=1).mean()
    )
    raw_data[f'{sensor}_roll_mean'] = roll_mean
    
   
    roll_std = raw_data.groupby('unit_number')[sensor].transform(
        lambda x: x.rolling(window=window, min_periods=2).std().fillna(method='bfill').fillna(0)
    )
    raw_data[f'{sensor}_roll_std'] = roll_std


max_time = raw_data.groupby('unit_number')['time'].transform('max')
raw_rul = max_time - raw_data['time']
raw_data['RUL'] = raw_rul.clip(upper=125)

exclude_cols = ['unit_number', 'time', 'RUL']
feature_cols = [col for col in raw_data.columns if col not in exclude_cols]
X = raw_data[feature_cols]
y = raw_data['RUL']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

processed_df = X_scaled.copy()
processed_df['unit_number'] = raw_data['unit_number'].values
processed_df['time'] = raw_data['time'].values
processed_df['RUL'] = y.values


processed_path = base_path / "data/processed/processed_train_FD001_final.csv"
processed_df.to_csv(processed_path, index=False)

scaler_path = base_path / "models/scaler_final.pkl"
joblib.dump(scaler, scaler_path)

