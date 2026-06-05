import numpy as np
import pandas as pd
import os

np.random.seed(42)

n = 8000

temperature_C = np.random.uniform(200, 400, n)
pressure_mTorr = np.random.uniform(10, 500, n)
gas_flow_sccm = np.random.uniform(50, 500, n)
rf_power_W = np.random.uniform(100, 1000, n)
deposition_time_s = np.random.uniform(60, 600, n)
chamber_humidity = np.random.uniform(20, 80, n)

yield_percent = (
    85.0
    - 0.05 * (temperature_C - 300)
    - 0.03 * (pressure_mTorr - 255)
    + 0.02 * (gas_flow_sccm - 275)
    + 0.01 * (rf_power_W - 550)
    - 0.04 * (deposition_time_s - 330)
    - 0.06 * (chamber_humidity - 50)
    - 0.0003 * (temperature_C - 300) ** 2
    - 0.0001 * (pressure_mTorr - 255) ** 2
    + 0.0002 * (deposition_time_s - 330) * (temperature_C - 300) / 1000.0
    + np.random.normal(0, 3.5, n)
)

yield_percent = np.clip(yield_percent, 60, 99)

df = pd.DataFrame({
    "temperature_C": temperature_C,
    "pressure_mTorr": pressure_mTorr,
    "gas_flow_sccm": gas_flow_sccm,
    "rf_power_W": rf_power_W,
    "deposition_time_s": deposition_time_s,
    "chamber_humidity": chamber_humidity,
    "yield_percent": yield_percent,
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/yield_data.csv", index=False)

print(f"Shape: {df.shape}")
print("\nDescriptive statistics:")
print(df.describe())
