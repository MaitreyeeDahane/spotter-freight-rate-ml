import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "train-test.csv"

TRAIN_END = "2025-08-31"
VALIDATION_START = "2025-09-01"
VALIDATION_END = "2025-10-31"


# ============================================================
# METRICS
# ============================================================

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])


# ============================================================
# TEMPORAL SPLIT
# ============================================================

train = df[
    df["date"] <= TRAIN_END
].copy()

validation = df[
    (df["date"] >= VALIDATION_START)
    & (df["date"] <= VALIDATION_END)
].copy()


# ============================================================
# TARGET
# ============================================================

y_train = train["posted_rate"].to_numpy(dtype=float)
y_validation = validation["posted_rate"].to_numpy(dtype=float)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# ------------------------------------------------------------
# Weight
# ------------------------------------------------------------

weight_median = train["weight"].median()

train_weight = train["weight"].fillna(
    weight_median
)

validation_weight = validation["weight"].fillna(
    weight_median
)


# ------------------------------------------------------------
# Date features
# ------------------------------------------------------------

train_month = train["date"].dt.month
validation_month = validation["date"].dt.month

train_day_of_week = train["date"].dt.dayofweek
validation_day_of_week = validation["date"].dt.dayofweek


# ------------------------------------------------------------
# Equipment one-hot encoding
# ------------------------------------------------------------

equipment_categories = sorted(
    train["equipment"].dropna().unique()
)


def encode_equipment(series, categories):

    encoded = []

    for category in categories:
        encoded.append(
            (series == category).astype(float).to_numpy()
        )

    return np.column_stack(encoded)


train_equipment = encode_equipment(
    train["equipment"],
    equipment_categories
)

validation_equipment = encode_equipment(
    validation["equipment"],
    equipment_categories
)


# ============================================================
# BUILD FEATURE MATRICES
# ============================================================

X_train = np.column_stack([
    train["distance"].to_numpy(dtype=float),
    train_weight.to_numpy(dtype=float),
    train_month.to_numpy(dtype=float),
    train_day_of_week.to_numpy(dtype=float),
    train_equipment
])

X_validation = np.column_stack([
    validation["distance"].to_numpy(dtype=float),
    validation_weight.to_numpy(dtype=float),
    validation_month.to_numpy(dtype=float),
    validation_day_of_week.to_numpy(dtype=float),
    validation_equipment
])


feature_names = [
    "distance",
    "weight",
    "month",
    "day_of_week"
] + [
    f"equipment_{category}"
    for category in equipment_categories
]


# ============================================================
# MODEL
# ============================================================

model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)


print("=" * 70)
print("EXPERIMENT 04 — HISTOGRAM GRADIENT BOOSTING")
print("=" * 70)

print()
print("TRAINING MODEL...")

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

predictions = model.predict(
    X_validation
)


# ============================================================
# EVALUATION
# ============================================================

validation_mae = mae(
    y_validation,
    predictions
)

validation_rmse = rmse(
    y_validation,
    predictions
)

validation_r2 = r2(
    y_validation,
    predictions
)


# Compare with Experiment 3
experiment_03_mae = 123.36
experiment_03_rmse = 634.29

mae_improvement = (
    (experiment_03_mae - validation_mae)
    / experiment_03_mae
    * 100
)

rmse_improvement = (
    (experiment_03_rmse - validation_rmse)
    / experiment_03_rmse
    * 100
)


# ============================================================
# RESULTS
# ============================================================

print()
print("TEMPORAL SPLIT")
print("-" * 70)
print(f"Training rows       : {len(train):,}")
print(f"Validation rows     : {len(validation):,}")
print(
    f"Training period     : "
    f"{train['date'].min().date()} → "
    f"{train['date'].max().date()}"
)
print(
    f"Validation period   : "
    f"{validation['date'].min().date()} → "
    f"{validation['date'].max().date()}"
)

print()
print("FEATURES")
print("-" * 70)

for feature in feature_names:
    print(f"  {feature}")

print()
print("MODEL")
print("-" * 70)
print("HistGradientBoostingRegressor")
print("max_iter           : 300")
print("learning_rate      : 0.05")
print("max_leaf_nodes     : 31")
print("l2_regularization  : 1.0")
print("random_state       : 42")

print()
print("RESULTS")
print("-" * 70)
print(f"MAE                 : ${validation_mae:,.2f}")
print(f"RMSE                : ${validation_rmse:,.2f}")
print(f"R²                  : {validation_r2:.4f}")

print()
print("VS EXPERIMENT 03")
print("-" * 70)
print(f"MAE improvement     : {mae_improvement:.2f}%")
print(f"RMSE improvement    : {rmse_improvement:.2f}%")

print()
print("=" * 70)