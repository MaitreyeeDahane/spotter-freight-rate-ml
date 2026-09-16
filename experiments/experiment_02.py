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
    """
    Ordinary least squares using the Moore-Penrose
    pseudoinverse.
    """
    X_with_intercept = np.column_stack([
        np.ones(len(X)),
        X
    ])

    coefficients = np.linalg.pinv(X_with_intercept) @ y

    return coefficients


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


# ============================================================
# TARGET
# ============================================================

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

train_weight = train["weight"].fillna(weight_median).to_numpy(dtype=float)
validation_weight = validation["weight"].fillna(weight_median).to_numpy(dtype=float)

# Scale weight using training statistics.
weight_mean = train_weight.mean()
weight_std = train_weight.std()

train_weight_scaled = (
    train_weight - weight_mean
) / weight_std

validation_weight_scaled = (
    validation_weight - weight_mean
) / weight_std


# ============================================================
# EQUIPMENT ONE-HOT ENCODING
# ============================================================

equipment_categories = sorted(
    train["equipment"].dropna().unique()
)

print("=" * 70)
print("EQUIPMENT CATEGORIES")
print("=" * 70)
print(equipment_categories)

for category in equipment_categories:
    print(f"  {category}")

print()


def encode_equipment(series, categories):
    encoded_columns = []

    for category in categories:
        encoded_columns.append(
            (series == category).astype(float).to_numpy()
        )

    return np.column_stack(encoded_columns)


train_equipment = encode_equipment(
    train["equipment"],
    equipment_categories
)

validation_equipment = encode_equipment(
    validation["equipment"],
    equipment_categories
)


# ============================================================
# MODEL 1 — LINEAR DISTANCE
# ============================================================

X_train_linear = train_distance_scaled.reshape(-1, 1)
X_validation_linear = validation_distance_scaled.reshape(-1, 1)

coef_linear = fit_ols(
    X_train_linear,
    y_train
)

pred_linear = predict_ols(
    X_validation_linear,
    coef_linear
)


# ============================================================
# MODEL 2 — QUADRATIC DISTANCE
# ============================================================

X_train_quadratic = np.column_stack([
    train_distance_scaled,
    train_distance_scaled ** 2
])

X_validation_quadratic = np.column_stack([
    validation_distance_scaled,
    validation_distance_scaled ** 2
])

coef_quadratic = fit_ols(
    X_train_quadratic,
    y_train
)

pred_quadratic = predict_ols(
    X_validation_quadratic,
    coef_quadratic
)


# ============================================================
# MODEL 3 — DISTANCE + WEIGHT + EQUIPMENT
# ============================================================

X_train_load = np.column_stack([
    train_distance_scaled,
    train_distance_scaled ** 2,
    train_weight_scaled,
    train_equipment
])

X_validation_load = np.column_stack([
    validation_distance_scaled,
    validation_distance_scaled ** 2,
    validation_weight_scaled,
    validation_equipment
])

coef_load = fit_ols(
    X_train_load,
    y_train
)

pred_load = predict_ols(
    X_validation_load,
    coef_load
)


# ============================================================
# EVALUATION
# ============================================================

results = []

models = [
    ("Linear Distance", pred_linear),
    ("Quadratic Distance", pred_quadratic),
    ("Distance + Weight + Equipment", pred_load),
]

for model_name, predictions in models:

    results.append({
        "Model": model_name,
        "MAE": mae(y_validation, predictions),
        "RMSE": rmse(y_validation, predictions),
        "R2": r2(y_validation, predictions)
    })


results_df = pd.DataFrame(results)


# ============================================================
# IMPROVEMENT CALCULATIONS
# ============================================================

baseline_mae = results_df.loc[
    results_df["Model"] == "Linear Distance",
    "MAE"
].iloc[0]

baseline_rmse = results_df.loc[
    results_df["Model"] == "Linear Distance",
    "RMSE"
].iloc[0]

load_mae = results_df.loc[
    results_df["Model"] == "Distance + Weight + Equipment",
    "MAE"
].iloc[0]

load_rmse = results_df.loc[
    results_df["Model"] == "Distance + Weight + Equipment",
    "RMSE"
].iloc[0]

mae_improvement = (
    (baseline_mae - load_mae)
    / baseline_mae
    * 100
)

rmse_improvement = (
    (baseline_rmse - load_rmse)
    / baseline_rmse
    * 100
)


# ============================================================
# RESULTS
# ============================================================

print("=" * 70)
print("EXPERIMENT 02 — LOAD CHARACTERISTICS")
print("=" * 70)

print()
print("TEMPORAL SPLIT")
print("-" * 70)
print(f"Training rows       : {len(train):,}")
print(f"Validation rows     : {len(validation):,}")
print(
    f"Training period     : "
    f"{train['date'].min().date()} → {train['date'].max().date()}"
)
print(
    f"Validation period   : "
    f"{validation['date'].min().date()} → "
    f"{validation['date'].max().date()}"
)

print()
print("PREPROCESSING")
print("-" * 70)
print(f"Training weight median used for imputation : {weight_median:,.2f}")
print(f"Equipment categories                      : {equipment_categories}")

print()
print("MODEL COMPARISON")
print("-" * 70)

for _, row in results_df.iterrows():

    print(f"\n{row['Model']}")
    print(f"  MAE  : ${row['MAE']:,.2f}")
    print(f"  RMSE : ${row['RMSE']:,.2f}")
    print(f"  R²   : {row['R2']:.4f}")

print()
print("IMPROVEMENT VS LINEAR DISTANCE")
print("-" * 70)
print(f"MAE improvement     : {mae_improvement:.2f}%")
print(f"RMSE improvement    : {rmse_improvement:.2f}%")

print()
print("=" * 70)