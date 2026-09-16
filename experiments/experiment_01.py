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
# LINEAR REGRESSION USING NUMPY
# ============================================================

def fit_linear_model(X, y):
    """
    Fits ordinary least squares regression using
    the Moore-Penrose pseudoinverse.
    """
    X_with_intercept = np.column_stack([
        np.ones(len(X)),
        X
    ])

    coefficients = np.linalg.pinv(X_with_intercept) @ y

    return coefficients


def predict_linear_model(X, coefficients):
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

target = "posted_rate"


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
# PREPARE DATA
# ============================================================

train_distance = train["distance"].to_numpy(dtype=float)
validation_distance = validation["distance"].to_numpy(dtype=float)

y_train = train[target].to_numpy(dtype=float)
y_validation = validation[target].to_numpy(dtype=float)


# ============================================================
# IMPORTANT:
# Scale using TRAINING statistics only.
# ============================================================

distance_mean = train_distance.mean()
distance_std = train_distance.std()

train_x = (
    train_distance - distance_mean
) / distance_std

validation_x = (
    validation_distance - distance_mean
) / distance_std


# ============================================================
# MODEL 1 — LINEAR DISTANCE
# ============================================================

X_train_linear = train_x.reshape(-1, 1)
X_validation_linear = validation_x.reshape(-1, 1)

linear_coefficients = fit_linear_model(
    X_train_linear,
    y_train
)

linear_predictions = predict_linear_model(
    X_validation_linear,
    linear_coefficients
)


# ============================================================
# MODEL 2 — QUADRATIC DISTANCE
# ============================================================

X_train_quadratic = np.column_stack([
    train_x,
    train_x ** 2
])

X_validation_quadratic = np.column_stack([
    validation_x,
    validation_x ** 2
])

quadratic_coefficients = fit_linear_model(
    X_train_quadratic,
    y_train
)

quadratic_predictions = predict_linear_model(
    X_validation_quadratic,
    quadratic_coefficients
)


# ============================================================
# EVALUATION
# ============================================================

linear_mae = mae(y_validation, linear_predictions)
linear_rmse = rmse(y_validation, linear_predictions)
linear_r2 = r2(y_validation, linear_predictions)

quadratic_mae = mae(y_validation, quadratic_predictions)
quadratic_rmse = rmse(y_validation, quadratic_predictions)
quadratic_r2 = r2(y_validation, quadratic_predictions)


mae_change = (
    (linear_mae - quadratic_mae)
    / linear_mae
    * 100
)

rmse_change = (
    (linear_rmse - quadratic_rmse)
    / linear_rmse
    * 100
)


# ============================================================
# RESULTS
# ============================================================

print("=" * 65)
print("EXPERIMENT 01 — NONLINEAR DISTANCE")
print("=" * 65)

print()
print("TEMPORAL SPLIT")
print("-" * 65)
print(f"Training rows       : {len(train):,}")
print(f"Validation rows     : {len(validation):,}")
print(f"Training period     : {train['date'].min().date()} → {train['date'].max().date()}")
print(f"Validation period   : {validation['date'].min().date()} → {validation['date'].max().date()}")

print()
print("MODEL 1 — LINEAR DISTANCE")
print("-" * 65)
print(f"MAE                 : ${linear_mae:,.2f}")
print(f"RMSE                : ${linear_rmse:,.2f}")
print(f"R²                  : {linear_r2:.4f}")

print()
print("MODEL 2 — QUADRATIC DISTANCE")
print("-" * 65)
print(f"MAE                 : ${quadratic_mae:,.2f}")
print(f"RMSE                : ${quadratic_rmse:,.2f}")
print(f"R²                  : {quadratic_r2:.4f}")

print()
print("IMPROVEMENT FROM QUADRATIC MODEL")
print("-" * 65)
print(f"MAE improvement     : {mae_change:.2f}%")
print(f"RMSE improvement    : {rmse_change:.2f}%")

print()
print("QUADRATIC COEFFICIENTS")
print("-" * 65)
print(f"Intercept           : {quadratic_coefficients[0]:,.4f}")
print(f"Linear term         : {quadratic_coefficients[1]:,.4f}")
print(f"Quadratic term      : {quadratic_coefficients[2]:,.4f}")

print()
print("=" * 65)