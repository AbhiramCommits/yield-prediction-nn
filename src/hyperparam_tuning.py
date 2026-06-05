import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from itertools import product


class YieldNet(nn.Module):
    def __init__(self, input_dim=6, hidden_sizes=(128, 64, 32), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            preds.append(model(Xb).cpu().numpy())
            actuals.append(yb.numpy())
    y_pred = np.concatenate(preds).flatten()
    y_true = np.concatenate(actuals).flatten()
    return np.sqrt(mean_squared_error(y_true, y_pred))


def train_one_config(hidden_sizes, lr, dropout, X_train, y_train, X_val, y_val,
                     device, epochs=50, batch_size=64):
    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(
        torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = YieldNet(hidden_sizes=hidden_sizes, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * Xb.size(0)
        epoch_train_loss /= len(train_ds)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                loss = criterion(pred, yb)
                epoch_val_loss += loss.item() * Xb.size(0)
        epoch_val_loss /= len(val_ds)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

    val_rmse = evaluate(model, val_loader, device)
    return model, val_rmse, train_losses, val_losses


def main():
    df = pd.read_csv("data/yield_data.csv")
    X = df.drop(columns=["yield_percent"]).values.astype(np.float32)
    y = df["yield_percent"].values.astype(np.float32).reshape(-1, 1)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rates = [0.001, 0.0005]
    dropout_rates = [0.1, 0.2, 0.3]
    hidden_sizes_list = [[128, 64, 32], [256, 128, 64]]

    best_val_rmse = float("inf")
    best_config = None
    best_model = None
    best_train_losses = None
    best_val_losses = None

    results = []

    for lr, dropout, hidden_sizes in product(
        learning_rates, dropout_rates, hidden_sizes_list
    ):
        config_str = f"lr={lr}, dropout={dropout}, hidden={hidden_sizes}"
        print(f"Training: {config_str}")

        model, val_rmse, train_losses, val_losses = train_one_config(
            hidden_sizes=hidden_sizes,
            lr=lr,
            dropout=dropout,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            device=device,
            epochs=50,
        )

        results.append({
            "lr": lr,
            "dropout": dropout,
            "hidden_sizes": str(hidden_sizes),
            "val_rmse": val_rmse,
        })
        print(f"  Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_config = (hidden_sizes, lr, dropout)
            best_model = model
            best_train_losses = train_losses
            best_val_losses = val_losses

    results_df = pd.DataFrame(results).sort_values("val_rmse")
    results_df.to_csv("results/grid_search_results.csv", index=False)

    print(f"\nBest config: hidden={best_config[0]}, lr={best_config[1]}, "
          f"dropout={best_config[2]}")
    print(f"Best val RMSE: {best_val_rmse:.4f}")

    torch.save(best_model.state_dict(), "models/yield_net_best.pt")

    loss_df = pd.DataFrame({
        "epoch": range(1, 51),
        "train_loss": best_train_losses,
        "val_loss": best_val_losses,
    })
    loss_df.to_csv("results/best_config_loss_history.csv", index=False)

    print("\nSaved best model to models/yield_net_best.pt")
    print("Saved grid search results to results/grid_search_results.csv")
    print("Saved loss history to results/best_config_loss_history.csv")


if __name__ == "__main__":
    main()
