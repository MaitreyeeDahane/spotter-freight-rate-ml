from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = Path("train-test.csv")

TRAIN_END = "2025-08-31"
VALIDATION_START = "2025-09-01"


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])


# ---------------------------------------------------------
# Temporal split
# ---------------------------------------------------------

train = df[df["date"] <= TRAIN_END].copy()
valid = df[df["date"] >= VALIDATION_START].copy()

print("=" * 60)
print("TEMPORAL BASELINE")
print("=" * 60)

print(f"Full development data : {len(df):,}")
print(f"Training rows          : {len(train):,}")
print(f"Validation rows        : {len(valid):,}")

print(
    f"Training period       : "
    f"{train['date'].min().date()} → {train['date'].max().date()}"
)

print(
    f"Validation period     : "
    f"{valid['date'].min().date()} → {valid['date'].max().date()}"
)


# ---------------------------------------------------------
# Target
# ---------------------------------------------------------

y_train = train["posted_rate"].to_numpy(dtype=float)
y_valid = valid["posted_rate"].to_numpy(dtype=float)


# ---------------------------------------------------------
# BASELINE A
# Median prediction
# ---------------------------------------------------------

median_prediction = np.median(y_train)

pred_median = np.full(
    shape=len(valid),
    fill_value=median_prediction,
    dtype=float,
)

print("\n" + "-" * 60)
print("BASELINE A — MEDIAN")
print("-" * 60)

print(f"Training median rate : ${median_prediction:,.2f}")
print(f"MAE                  : ${mae(y_valid, pred_median):,.2f}")
print(f"RMSE                 : ${rmse(y_valid, pred_median):,.2f}")
print(f"R²                   : {r2(y_valid, pred_median):.4f}")


# ---------------------------------------------------------
# BASELINE B
# Distance-only linear regression
# ---------------------------------------------------------

x_train = train["distance"].to_numpy(dtype=float)
x_valid = valid["distance"].to_numpy(dtype=float)

# Fit y = intercept + coefficient * distance
x_mean = np.mean(x_train)
y_mean = np.mean(y_train)

slope = np.sum(
    (x_train - x_mean) * (y_train - y_mean)
) / np.sum(
    (x_train - x_mean) ** 2
)

intercept = y_mean - slope * x_mean

pred_distance = intercept + slope * x_valid


print("\n" + "-" * 60)
print("BASELINE B — DISTANCE-ONLY LINEAR REGRESSION")
print("-" * 60)

print(f"Intercept            : ${intercept:,.2f}")
print(f"Distance coefficient : ${slope:,.4f} per mile")
print(f"MAE                  : ${mae(y_valid, pred_distance):,.2f}")
print(f"RMSE                 : ${rmse(y_valid, pred_distance):,.2f}")
print(f"R²                   : {r2(y_valid, pred_distance):.4f}")


# ---------------------------------------------------------
# Improvement over median
# ---------------------------------------------------------

median_mae = mae(y_valid, pred_median)
distance_mae = mae(y_valid, pred_distance)

median_rmse = rmse(y_valid, pred_median)
distance_rmse = rmse(y_valid, pred_distance)

print("\n" + "=" * 60)
print("BASELINE COMPARISON")
print("=" * 60)

print(
    f"MAE improvement from distance model: "
    f"{(median_mae - distance_mae) / median_mae * 100:.2f}%"
)

print(
    f"RMSE improvement from distance model: "
    f"{(median_rmse - distance_rmse) / median_rmse * 100:.2f}%"
)