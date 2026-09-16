import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "train-test.csv"

TRAIN_END = "2025-08-31"
VAL_START = "2025-09-01"
VAL_END = "2025-10-31"


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

train = df[df["date"] <= TRAIN_END].copy()

val = df[
    (df["date"] >= VAL_START) &
    (df["date"] <= VAL_END)
].copy()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(data, train_reference, include_calendar=False):

    data = data.copy()

    # ----------------------------
    # Weight
    # ----------------------------

    weight_median = train_reference["weight"].median()

    data["weight_imputed"] = data["weight"].fillna(
        weight_median
    )

    # ----------------------------
    # Distance
    # ----------------------------

    data["distance_sq"] = data["distance"] ** 2

    # ----------------------------
    # Equipment
    # ----------------------------

    data["reefer"] = (
        data["equipment"] == "Reefer"
    ).astype(float)

    data["flatbed"] = (
        data["equipment"] == "Flatbed"
    ).astype(float)

    # ----------------------------
    # Interactions
    # ----------------------------

    data["distance_reefer"] = (
        data["distance"] * data["reefer"]
    )

    data["distance_flatbed"] = (
        data["distance"] * data["flatbed"]
    )

    data["weight_reefer"] = (
        data["weight_imputed"] * data["reefer"]
    )

    data["weight_flatbed"] = (
        data["weight_imputed"] * data["flatbed"]
    )

    # ----------------------------
    # Geography
    # ----------------------------

    geo_cols = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon"
    ]

    for col in geo_cols:

        mean = train_reference[col].mean()
        std = train_reference[col].std()

        if pd.isna(std) or std == 0:
            std = 1.0

        data[f"{col}_scaled"] = (
            data[col].fillna(mean) - mean
        ) / std

    features = [
        "distance",
        "distance_sq",
        "weight_imputed",

        "reefer",
        "flatbed",

        "distance_reefer",
        "distance_flatbed",

        "weight_reefer",
        "weight_flatbed",

        "pickup_lat_scaled",
        "pickup_lon_scaled",
        "delivery_lat_scaled",
        "delivery_lon_scaled"
    ]

    # ----------------------------
    # Optional calendar features
    # ----------------------------

    if include_calendar:

        data["month"] = data["date"].dt.month
        data["day_of_week"] = data["date"].dt.dayofweek

        features += [
            "month",
            "day_of_week"
        ]

    return data[features]


# ============================================================
# TARGET
# ============================================================

y_train = train["posted_rate"].values
y_val = val["posted_rate"].values


# ============================================================
# MODEL A — CURRENT ENGINEERED BASE
# ============================================================

X_train_a = build_features(
    train,
    train,
    include_calendar=False
)

X_val_a = build_features(
    val,
    train,
    include_calendar=False
)

model_a = LinearRegression()

model_a.fit(
    X_train_a,
    y_train
)

pred_a = model_a.predict(
    X_val_a
)


# ============================================================
# MODEL B — + CALENDAR
# ============================================================

X_train_b = build_features(
    train,
    train,
    include_calendar=True
)

X_val_b = build_features(
    val,
    train,
    include_calendar=True
)

model_b = LinearRegression()

model_b.fit(
    X_train_b,
    y_train
)

pred_b = model_b.predict(
    X_val_b
)


# ============================================================
# METRICS
# ============================================================

def evaluate(y_true, prediction):

    mae = mean_absolute_error(
        y_true,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            prediction
        )
    )

    r2 = r2_score(
        y_true,
        prediction
    )

    return mae, rmse, r2


a_mae, a_rmse, a_r2 = evaluate(
    y_val,
    pred_a
)

b_mae, b_rmse, b_r2 = evaluate(
    y_val,
    pred_b
)


# ============================================================
# TRAINING METRICS
# ============================================================

train_pred_a = model_a.predict(
    X_train_a
)

train_pred_b = model_b.predict(
    X_train_b
)

a_train_mae, a_train_rmse, a_train_r2 = evaluate(
    y_train,
    train_pred_a
)

b_train_mae, b_train_rmse, b_train_r2 = evaluate(
    y_train,
    train_pred_b
)


# ============================================================
# GENERALIZATION GAPS
# ============================================================

a_mae_gap = (
    (a_mae - a_train_mae)
    / a_train_mae
    * 100
)

a_rmse_gap = (
    (a_rmse - a_train_rmse)
    / a_train_rmse
    * 100
)

b_mae_gap = (
    (b_mae - b_train_mae)
    / b_train_mae
    * 100
)

b_rmse_gap = (
    (b_rmse - b_train_rmse)
    / b_train_rmse
    * 100
)


# ============================================================
# RESULTS
# ============================================================

print("=" * 75)
print("STEP 18 — CALENDAR FEATURE EXPERIMENT")
print("=" * 75)

print()
print("TEMPORAL SPLIT")
print("-" * 75)
print("Training   : Jan 1 – Aug 31, 2025")
print("Validation : Sep 1 – Oct 31, 2025")

print()
print("MODEL A — CURRENT ENGINEERED FEATURES")
print("-" * 75)

print("TRAIN")
print(f"MAE  : ${a_train_mae:,.2f}")
print(f"RMSE : ${a_train_rmse:,.2f}")
print(f"R²   : {a_train_r2:.4f}")

print()
print("VALIDATION")
print(f"MAE  : ${a_mae:,.2f}")
print(f"RMSE : ${a_rmse:,.2f}")
print(f"R²   : {a_r2:.4f}")

print()
print("GENERALIZATION GAP")
print(f"MAE  : {a_mae_gap:+.2f}%")
print(f"RMSE : {a_rmse_gap:+.2f}%")

print()
print("MODEL B — + MONTH + DAY OF WEEK")
print("-" * 75)

print("TRAIN")
print(f"MAE  : ${b_train_mae:,.2f}")
print(f"RMSE : ${b_train_rmse:,.2f}")
print(f"R²   : {b_train_r2:.4f}")

print()
print("VALIDATION")
print(f"MAE  : ${b_mae:,.2f}")
print(f"RMSE : ${b_rmse:,.2f}")
print(f"R²   : {b_r2:.4f}")

print()
print("GENERALIZATION GAP")
print(f"MAE  : {b_mae_gap:+.2f}%")
print(f"RMSE : {b_rmse_gap:+.2f}%")

print()
print("CHANGE FROM MODEL A → MODEL B")
print("-" * 75)

mae_change = (
    (a_mae - b_mae)
    / a_mae
    * 100
)

rmse_change = (
    (a_rmse - b_rmse)
    / a_rmse
    * 100
)

print(f"MAE improvement  : {mae_change:+.2f}%")
print(f"RMSE improvement : {rmse_change:+.2f}%")

print()
print("=" * 75)