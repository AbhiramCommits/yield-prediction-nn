import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("data/yield_data.csv")

X = df.drop(columns=["yield_percent"]).values.astype(np.float32)
y = df["yield_percent"].values.astype(np.float32).reshape(-1, 1)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_val_t = torch.tensor(X_val)
y_val_t = torch.tensor(y_val)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)
test_ds = TensorDataset(X_test_t, y_test_t)

batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)


class YieldNet(nn.Module):
    def __init__(self, input_dim=6, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = YieldNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
loss_history = []

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * Xb.size(0)
    train_loss /= len(train_ds)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            loss = criterion(pred, yb)
            val_loss += loss.item() * Xb.size(0)
    val_loss /= len(val_ds)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{epochs}  train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")

    loss_history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    })

model.eval()
all_preds = []
with torch.no_grad():
    for Xb, _ in test_loader:
        Xb = Xb.to(device)
        all_preds.append(model(Xb).cpu().numpy())
y_pred = np.concatenate(all_preds).flatten()
y_test_np = y_test.flatten()

rmse = np.sqrt(mean_squared_error(y_test_np, y_pred))
r2 = r2_score(y_test_np, y_pred)

print(f"\nTest RMSE: {rmse:.4f}")
print(f"Test R²:   {r2:.4f}")

torch.save(model.state_dict(), "models/yield_net.pt")
print("Model saved to models/yield_net.pt")

loss_df = pd.DataFrame(loss_history)
loss_df.to_csv("results/loss_history.csv", index=False)
print("Loss history saved to results/loss_history.csv")
