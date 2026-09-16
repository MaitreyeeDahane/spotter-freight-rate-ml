import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "train-test.csv"

TRAIN_END = "2025-06-30"
VAL_START = "2025-07-01"
VAL_END = "2025-08-31"


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

def build_features(data, train_reference):

    data = data.copy()

    # Weight
    weight_median = train_reference["weight"].median()

    data["weight_imputed"] = data["weight"].fillna(
        weight_median
    )

    # Distance
    data["distance_sq"] = data["distance"] ** 2

    # Equipment
    data["reefer"] = (
        data["equipment"] == "Reefer"
    ).astype(float)

    data["flatbed"] = (
        data["equipment"] == "Flatbed"
    ).astype(float)

    # Interactions
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

    # Geography
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

    return data[features]


# ============================================================
# FEATURES
# ============================================================

X_train_base = build_features(train, train)
X_val_base = build_features(val, train)

y_train = train["posted_rate"].values
y_val = val["posted_rate"].values


# ============================================================
# MODEL A — CURRENT BEST
# ============================================================

model_a = LinearRegression()

model_a.fit(
    X_train_base,
    y_train
)

pred_a_train = model_a.predict(X_train_base)
pred_a_val = model_a.predict(X_val_base)


# ============================================================
# ROUTE FEATURES
# ============================================================

train["route"] = (
    train["pickup"] + " → " + train["delivery"]
)

val["route"] = (
    val["pickup"] + " → " + val["delivery"]
)


# Historical route statistics

route_stats = (
    train.groupby("route")["posted_rate"]
    .agg(
        route_median="median",
        route_mean="mean",
        route_count="count"
    )
)


global_median = train["posted_rate"].median()
global_mean = train["posted_rate"].mean()


# ------------------------------------------------------------
# Training leave-one-out route features
# ------------------------------------------------------------

route_sum = train.groupby("route")["posted_rate"].transform("sum")
route_count = train.groupby("route")["posted_rate"].transform("count")

train_route_count = route_count - 1

train_route_mean = np.where(
    train_route_count > 0,
    (route_sum - train["posted_rate"]) / train_route_count,
    global_mean
)

route_median_map = (
    train.groupby("route")["posted_rate"]
    .median()
)

train_route_median = train["route"].map(
    route_median_map
)

train_route_median = np.where(
    train_route_count >= 2,
    train_route_median,
    global_median
)


# ------------------------------------------------------------
# Validation historical route features
# ------------------------------------------------------------

val_route_stats = val["route"].map(
    route_stats["route_mean"]
)

val_route_median = val["route"].map(
    route_stats["route_median"]
)

val_route_count = val["route"].map(
    route_stats["route_count"]
)


val_route_mean = val_route_stats.fillna(
    global_mean
)

val_route_median = val_route_median.fillna(
    global_median
)

val_route_count = val_route_count.fillna(
    0
)


# ============================================================
# ADD ROUTE FEATURES
# ============================================================

X_train_route = X_train_base.copy()
X_val_route = X_val_base.copy()

X_train_route["route_count"] = train_route_count.values
X_train_route["route_mean"] = train_route_mean
X_train_route["route_median"] = train_route_median

X_val_route["route_count"] = val_route_count.values
X_val_route["route_mean"] = val_route_mean.values
X_val_route["route_median"] = val_route_median.values


# ============================================================
# MODEL B — ROUTE AWARE
# ============================================================

model_b = LinearRegression()

model_b.fit(
    X_train_route,
    y_train
)

pred_b_train = model_b.predict(
    X_train_route
)

pred_b_val = model_b.predict(
    X_val_route
)


# ============================================================
# METRICS
# ============================================================

def evaluate(y_true, pred):

    mae = mean_absolute_error(
        y_true,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            pred
        )
    )

    r2 = r2_score(
        y_true,
        pred
    )

    return mae, rmse, r2


# Model A
a_train_mae, a_train_rmse, a_train_r2 = evaluate(
    y_train,
    pred_a_train
)

a_val_mae, a_val_rmse, a_val_r2 = evaluate(
    y_val,
    pred_a_val
)


# Model B
b_train_mae, b_train_rmse, b_train_r2 = evaluate(
    y_train,
    pred_b_train
)

b_val_mae, b_val_rmse, b_val_r2 = evaluate(
    y_val,
    pred_b_val
)


# ============================================================
# GENERALIZATION GAPS
# ============================================================

a_mae_gap = (
    (a_val_mae - a_train_mae)
    / a_train_mae
    * 100
)

a_rmse_gap = (
    (a_val_rmse - a_train_rmse)
    / a_train_rmse
    * 100
)

b_mae_gap = (
    (b_val_mae - b_train_mae)
    / b_train_mae
    * 100
)

b_rmse_gap = (
    (b_val_rmse - b_train_rmse)
    / b_train_rmse
    * 100
)


# ============================================================
# EXTREME RATE CHECK
# ============================================================

threshold = train["posted_rate"].quantile(0.99)

extreme_mask = (
    y_val >= threshold
)

normal_mask = ~extreme_mask


a_normal_mae = mean_absolute_error(
    y_val[normal_mask],
    pred_a_val[normal_mask]
)

a_extreme_mae = mean_absolute_error(
    y_val[extreme_mask],
    pred_a_val[extreme_mask]
)

b_normal_mae = mean_absolute_error(
    y_val[normal_mask],
    pred_b_val[normal_mask]
)

b_extreme_mae = mean_absolute_error(
    y_val[extreme_mask],
    pred_b_val[extreme_mask]
)


# ============================================================
# RESULTS
# ============================================================

print("=" * 75)
print("STEP 16 — TEMPORAL ROBUSTNESS CHECK")
print("=" * 75)

print()
print("TEMPORAL SPLIT")
print("-" * 75)
print("Training   : Jan 1 – Jun 30, 2025")
print("Validation : Jul 1 – Aug 31, 2025")

print()
print("DATA")
print("-" * 75)
print(f"Training rows   : {len(train):,}")
print(f"Validation rows : {len(val):,}")

print()
print("MODEL A — ENGINEERED LINEAR")
print("-" * 75)
print("TRAIN")
print(f"MAE  : ${a_train_mae:,.2f}")
print(f"RMSE : ${a_train_rmse:,.2f}")
print(f"R²   : {a_train_r2:.4f}")

print()
print("VALIDATION")
print(f"MAE  : ${a_val_mae:,.2f}")
print(f"RMSE : ${a_val_rmse:,.2f}")
print(f"R²   : {a_val_r2:.4f}")

print()
print("GENERALIZATION GAP")
print(f"MAE  : {a_mae_gap:+.2f}%")
print(f"RMSE : {a_rmse_gap:+.2f}%")

print()
print("MODEL B — + ROUTE STATISTICS")
print("-" * 75)
print("TRAIN")
print(f"MAE  : ${b_train_mae:,.2f}")
print(f"RMSE : ${b_train_rmse:,.2f}")
print(f"R²   : {b_train_r2:.4f}")

print()
print("VALIDATION")
print(f"MAE  : ${b_val_mae:,.2f}")
print(f"RMSE : ${b_val_rmse:,.2f}")
print(f"R²   : {b_val_r2:.4f}")

print()
print("GENERALIZATION GAP")
print(f"MAE  : {b_mae_gap:+.2f}%")
print(f"RMSE : {b_rmse_gap:+.2f}%")

print()
print("EXTREME-RATE CHECK")
print("-" * 75)
print(f"Training 99th percentile : ${threshold:,.2f}")
print(f"Validation extreme rows  : {extreme_mask.sum():,}")

print()
print("MODEL A")
print(f"Normal MAE  : ${a_normal_mae:,.2f}")
print(f"Extreme MAE : ${a_extreme_mae:,.2f}")

print()
print("MODEL B")
print(f"Normal MAE  : ${b_normal_mae:,.2f}")
print(f"Extreme MAE : ${b_extreme_mae:,.2f}")

print()
print("=" * 75)