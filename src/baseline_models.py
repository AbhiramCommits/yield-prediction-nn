import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("data/yield_data.csv")

X = df.drop(columns=["yield_percent"])
y = df["yield_percent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append(
        {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2}
    )

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
results_df.to_csv("results/baseline_comparison.csv", index=False)

print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["Model"]
y_pred_best = predictions[best_name]

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_best, alpha=0.3, s=8, color="steelblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Yield (%)")
plt.ylabel("Predicted Yield (%)")
plt.title(f"{best_name} — Predicted vs Actual")
r2 = r2_score(y_test, y_pred_best)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
plt.text(
    0.05, 0.95,
    f"R² = {r2:.4f}\nRMSE = {rmse:.2f}",
    transform=plt.gca().transAxes, fontsize=11, verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)
plt.tight_layout()
plt.savefig("results/baseline_predictions.png", dpi=150)
plt.close()
print(f"\nSaved predicted-vs-actual plot to results/baseline_predictions.png")
