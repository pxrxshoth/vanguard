import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path
import time

base_path = Path("C:/Users/acer/Desktop/Vangaurd")

data = pd.read_csv(base_path / "data/processed/processed_train_FD001_final.csv")
exclude_cols = ['unit_number', 'time', 'RUL']
X_df = data.drop(columns=exclude_cols)
feature_cols = X_df.columns.tolist()

SEQ_LEN = 10
BATCH_SIZE = 128
EPOCHS = 20
HIDDEN_DIM = 64
LR = 0.001


def create_sequences(df, seq_length):
    sequences = []
    arr = df.values
    for i in range(len(arr) - seq_length):
        sequences.append(arr[i:i+seq_length])
    return np.array(sequences)

X_seq = create_sequences(X_df.head(6000), SEQ_LEN)
X_tensor = torch.tensor(X_seq, dtype=torch.float32)

dataset = TensorDataset(X_tensor, X_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded


input_dim = len(feature_cols)
model = LSTMAutoencoder(input_dim, HIDDEN_DIM)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

start_time = time.time()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch+1) % 5 == 0:
        print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(dataloader):.6f}")


model_save_path = base_path / "models/anomaly_detector_ae.pth"
torch.save(model.state_dict(), model_save_path)

model.eval()
with torch.no_grad():
    sample_out = model(X_tensor[:1000].to(device))
    mse = torch.mean(torch.pow(X_tensor[:1000].to(device) - sample_out, 2), dim=2)
    threshold = np.percentile(mse.cpu().numpy(), 95)

iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X_df.head(10000)) 

if_save_path = base_path / "models/anomaly_detector_if.joblib"
joblib.dump(iso_forest, if_save_path)


