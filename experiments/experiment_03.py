import numpy as np
import pandas as pd


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
# OLS
# ============================================================

def fit_ols(X, y):
    X_with_intercept = np.column_stack([
        np.ones(len(X)),
        X
    ])

    return np.linalg.pinv(X_with_intercept) @ y


def predict_ols(X, coefficients):
    X_with_intercept = np.column_stack([
        np.ones(len(X)),
        X
    ])

    return X_with_intercept @ coefficients


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


y_train = train["posted_rate"].to_numpy(dtype=float)
y_validation = validation["posted_rate"].to_numpy(dtype=float)


# ============================================================
# DISTANCE
# ============================================================

train_distance = train["distance"].to_numpy(dtype=float)
validation_distance = validation["distance"].to_numpy(dtype=float)

distance_mean = train_distance.mean()
distance_std = train_distance.std()

train_distance_scaled = (
    train_distance - distance_mean
) / distance_std

validation_distance_scaled = (
    validation_distance - distance_mean
) / distance_std


# ============================================================
# WEIGHT
# ============================================================

weight_median = train["weight"].median()

train_weight = (
    train["weight"]
    .fillna(weight_median)
    .to_numpy(dtype=float)
)

validation_weight = (
    validation["weight"]
    .fillna(weight_median)
    .to_numpy(dtype=float)
)

weight_mean = train_weight.mean()
weight_std = train_weight.std()

train_weight_scaled = (
    train_weight - weight_mean
) / weight_std

validation_weight_scaled = (
    validation_weight - weight_mean
) / weight_std


# ============================================================
# EQUIPMENT ENCODING
# ============================================================

equipment_categories = sorted(
    train["equipment"].dropna().unique()
)


def encode_equipment(series, categories):
    columns = []

    for category in categories:
        columns.append(
            (series == category).astype(float).to_numpy()
        )

    return np.column_stack(columns)


train_equipment = encode_equipment(
    train["equipment"],
    equipment_categories
)

validation_equipment = encode_equipment(
    validation["equipment"],
    equipment_categories
)


# ============================================================
# MODEL 1 — EXPERIMENT 2 MODEL
#
# distance + distance² + weight + equipment
# ============================================================

X_train_base = np.column_stack([
    train_distance_scaled,
    train_distance_scaled ** 2,
    train_weight_scaled,
    train_equipment
])

X_validation_base = np.column_stack([
    validation_distance_scaled,
    validation_distance_scaled ** 2,
    validation_weight_scaled,
    validation_equipment
])

coef_base = fit_ols(
    X_train_base,
    y_train
)

pred_base = predict_ols(
    X_validation_base,
    coef_base
)


# ============================================================
# MODEL 2 — INTERACTION MODEL
#
# distance
# distance²
# weight
# equipment
# distance × equipment
# weight × equipment
# ============================================================

distance_equipment_interactions_train = (
    train_distance_scaled[:, None]
    * train_equipment
)

distance_equipment_interactions_validation = (
    validation_distance_scaled[:, None]
    * validation_equipment
)

weight_equipment_interactions_train = (
    train_weight_scaled[:, None]
    * train_equipment
)

weight_equipment_interactions_validation = (
    validation_weight_scaled[:, None]
    * validation_equipment
)


X_train_interaction = np.column_stack([
    train_distance_scaled,
    train_distance_scaled ** 2,
    train_weight_scaled,
    train_equipment,
    distance_equipment_interactions_train,
    weight_equipment_interactions_train
])

X_validation_interaction = np.column_stack([
    validation_distance_scaled,
    validation_distance_scaled ** 2,
    validation_weight_scaled,
    validation_equipment,
    distance_equipment_interactions_validation,
    weight_equipment_interactions_validation
])


coef_interaction = fit_ols(
    X_train_interaction,
    y_train
)

pred_interaction = predict_ols(
    X_validation_interaction,
    coef_interaction
)


# ============================================================
# EVALUATION
# ============================================================

base_mae = mae(y_validation, pred_base)
base_rmse = rmse(y_validation, pred_base)
base_r2 = r2(y_validation, pred_base)

interaction_mae = mae(
    y_validation,
    pred_interaction
)

interaction_rmse = rmse(
    y_validation,
    pred_interaction
)

interaction_r2 = r2(
    y_validation,
    pred_interaction
)


mae_improvement = (
    (base_mae - interaction_mae)
    / base_mae
    * 100
)

rmse_improvement = (
    (base_rmse - interaction_rmse)
    / base_rmse
    * 100
)


# ============================================================
# RESULTS
# ============================================================

print("=" * 70)
print("EXPERIMENT 03 — FEATURE INTERACTIONS")
print("=" * 70)

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
print("Base model:")
print("  distance")
print("  distance²")
print("  weight")
print("  equipment")

print()
print("Interaction model:")
print("  distance")
print("  distance²")
print("  weight")
print("  equipment")
print("  distance × equipment")
print("  weight × equipment")

print()
print("MODEL 1 — EXPERIMENT 02 BASE")
print("-" * 70)
print(f"MAE                 : ${base_mae:,.2f}")
print(f"RMSE                : ${base_rmse:,.2f}")
print(f"R²                  : {base_r2:.4f}")

print()
print("MODEL 2 — WITH INTERACTIONS")
print("-" * 70)
print(f"MAE                 : ${interaction_mae:,.2f}")
print(f"RMSE                : ${interaction_rmse:,.2f}")
print(f"R²                  : {interaction_r2:.4f}")

print()
print("IMPROVEMENT FROM INTERACTIONS")
print("-" * 70)
print(f"MAE improvement     : {mae_improvement:.2f}%")
print(f"RMSE improvement    : {rmse_improvement:.2f}%")

print()
print("=" * 70)