import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42
N = 50000
rng = np.random.default_rng(RANDOM_STATE)

# Synthetic transaction data for education/demo purposes.
amount = np.round(np.exp(rng.normal(np.log(900), 1.0, N)), 2).clip(20, 100000)
hour = rng.integers(0, 24, N)
transactions_last_1h = rng.poisson(1.5, N).clip(0, 15)
failed_attempts = rng.poisson(0.25, N).clip(0, 6)
account_age_days = rng.integers(1, 2500, N)
avg_amount_30d = np.round(np.exp(rng.normal(np.log(800), 0.65, N)), 2).clip(50, 20000)
new_device = rng.binomial(1, 0.08, N)
new_beneficiary = rng.binomial(1, 0.12, N)
location_change = rng.binomial(1, 0.06, N)
is_international_ip = rng.binomial(1, 0.015, N)
device_change_24h = rng.binomial(1, 0.05, N)
velocity_10m = rng.poisson(0.5, N).clip(0, 10)

amount_ratio = amount / (avg_amount_30d + 1)
night = ((hour <= 5) | (hour >= 23)).astype(int)

# Generate a low base-rate fraud label using a nonlinear risk score.
risk = (
    -5.8
    + 0.75 * night
    + 0.85 * new_device
    + 1.00 * new_beneficiary
    + 1.10 * location_change
    + 1.35 * is_international_ip
    + 0.85 * device_change_24h
    + 0.25 * velocity_10m
    + 0.20 * transactions_last_1h
    + 0.45 * failed_attempts
    + 0.55 * np.log1p(amount_ratio)
    + 0.20 * (amount > 15000)
    - 0.00015 * account_age_days
)
prob = 1 / (1 + np.exp(-risk))
fraud = rng.binomial(1, prob)

df = pd.DataFrame({
    "transaction_id": [f"UPI{100000+i}" for i in range(N)],
    "amount": amount,
    "hour": hour,
    "transactions_last_1h": transactions_last_1h,
    "failed_attempts": failed_attempts,
    "account_age_days": account_age_days,
    "avg_amount_30d": avg_amount_30d,
    "new_device": new_device,
    "new_beneficiary": new_beneficiary,
    "location_change": location_change,
    "is_international_ip": is_international_ip,
    "device_change_24h": device_change_24h,
    "velocity_10m": velocity_10m,
    "is_fraud": fraud,
})

df["amount_ratio"] = df["amount"] / (df["avg_amount_30d"] + 1)
df["night_transaction"] = ((df["hour"] <= 5) | (df["hour"] >= 23)).astype(int)

Path("data").mkdir(exist_ok=True)
df.to_csv("data/upi_transactions.csv", index=False)

print("Dataset created:", df.shape)
print(df["is_fraud"].value_counts())
print("Fraud rate:", round(df["is_fraud"].mean() * 100, 2), "%")
