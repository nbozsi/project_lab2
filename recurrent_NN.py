# --- Imports ---
import polars as pl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# --- Step 1: Load and preprocess data with Polars ---
# Example: df has 10 features and 1 continuous target


df = pl.read_parquet("data/joined_df.parquet")
# df = df.drop_nans().drop_nulls()

# Convert to numpy arrays
features = df.select(pl.exclude(["Időpont", "System Direction (kWh)"])).to_numpy()
target = df["System Direction (kWh)"].to_numpy()

scaler_X = StandardScaler()
features = scaler_X.fit_transform(features)

scaler_y = StandardScaler()
target = scaler_y.fit_transform(target.reshape(-1, 1)).flatten()


# --- Step 2: Create a sliding window dataset ---
class TimeSeriesDataset(Dataset):
    def __init__(self, features, target, seq_len=30):
        self.X, self.y = [], []
        for i in range(len(features) - seq_len):
            self.X.append(features[i : i + seq_len])
            self.y.append(target[i + seq_len])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


seq_len = 30
dataset = TimeSeriesDataset(features, target, seq_len)

# Split into train/test
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# --- Step 3: Define the GRU model ---
class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # take output from the last time step
        return out


input_size = features.shape[1]
hidden_size = 64
num_layers = 2
model = GRURegressor(input_size, hidden_size, num_layers)

# --- Step 4: Training setup ---
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Step 5: Training loop ---
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)

        if torch.isnan(loss):
            print("⚠️ NaN detected — skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

# --- Step 6: Evaluation ---
model.eval()
preds, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        output = model(X_batch).cpu().numpy()
        preds.append(output)
        actuals.append(y_batch.numpy())

preds = np.concatenate(preds)
actuals = np.concatenate(actuals)
mse = np.mean((preds - actuals) ** 2)
print(f"Test MSE: {mse:.4f}")
