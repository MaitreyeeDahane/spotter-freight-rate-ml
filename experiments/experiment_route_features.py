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
# CREATE ROUTE
# ============================================================

train["route"] = train["pickup"] + " → " + train["delivery"]
val["route"] = val["pickup"] + " → " + val["delivery"]


# ============================================================
# ROUTE STATISTICS
#
# IMPORTANT:
# These statistics are calculated ONLY from the
# historical training period.
# ============================================================

route_stats = (
    train.groupby("route")["posted_rate"]
    .agg(
        route_median="median",
        route_mean="mean",
        route_count="count"
    )
    .reset_index()
)


# Merge historical route information

val = val.merge(
    route_stats,
    on="route",
    how="left"
)


# ============================================================
# GLOBAL FALLBACKS
# ============================================================

global_median = train["posted_rate"].median()
global_mean = train["posted_rate"].mean()


# Whether route existed historically

val["route_seen"] = val["route_count"].notna().astype(float)


# For unseen routes, use global statistics

val["route_median"] = val["route_median"].fillna(global_median)
val["route_mean"] = val["route_mean"].fillna(global_mean)
val["route_count"] = val["route_count"].fillna(0)


# ============================================================
# STANDARD MODEL FEATURES
# Same structure as current best model
# ============================================================

def build_base_features(data, train_reference):

    data = data.copy()

    # Weight imputation using training median
    weight_median = train_reference["weight"].median()

    data["weight_imputed"] = data["weight"].fillna(
        weight_median
    )

    # Nonlinear distance
    data["distance_sq"] = data["distance"] ** 2

    # Equipment encoding
    data["reefer"] = (
        data["equipment"] == "Reefer"
    ).astype(float)

    data["flatbed"] = (
        data["equipment"] == "Flatbed"
    ).astype(float)

    # Equipment interactions
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

    return data


# ============================================================
# BUILD BASE FEATURES
# ============================================================

train_features = build_base_features(train, train)
val_features = build_base_features(val, train)


base_columns = [
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


# ============================================================
# EXPERIMENT A
# CURRENT BEST MODEL
# ============================================================

X_train_base = train_features[base_columns]
X_val_base = val_features[base_columns]

y_train = train["posted_rate"].values
y_val = val["posted_rate"].values

base_model = LinearRegression()

base_model.fit(
    X_train_base,
    y_train
)

base_pred = base_model.predict(
    X_val_base
)


# ============================================================
# EXPERIMENT B
# ADD ROUTE FEATURES
# ============================================================

# Route statistics for TRAINING rows themselves must be
# generated without using the row's own target.
#
# To avoid leakage, use leave-one-out route statistics.

train_route_sum = train.groupby("route")["posted_rate"].transform("sum")
train_route_count = train.groupby("route")["posted_rate"].transform("count")

train_features["route_count"] = train_route_count - 1

train_features["route_mean"] = np.where(
    train_features["route_count"] > 0,
    (train_route_sum - train["posted_rate"]) /
    train_features["route_count"],
    global_mean
)

# For the median, use the route median as a historical
# descriptor. For singleton routes, fall back globally.
#
# To remain conservative, only use route median when
# at least 2 other observations exist.

route_median_map = train.groupby("route")["posted_rate"].median()

train_features["route_median"] = train["route"].map(
    route_median_map
)

train_features["route_median"] = np.where(
    train_features["route_count"] >= 2,
    train_features["route_median"],
    global_median
)

# Validation route statistics are strictly historical
val_features["route_count"] = val["route_count"]
val_features["route_mean"] = val["route_mean"]
val_features["route_median"] = val["route_median"]

route_columns = base_columns + [
    "route_count",
    "route_mean",
    "route_median"
]


X_train_route = train_features[route_columns]
X_val_route = val_features[route_columns]


route_model = LinearRegression()

route_model.fit(
    X_train_route,
    y_train
)

route_pred = route_model.predict(
    X_val_route
)


# ============================================================
# METRIC FUNCTION
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


base_mae, base_rmse, base_r2 = evaluate(
    y_val,
    base_pred
)

route_mae, route_rmse, route_r2 = evaluate(
    y_val,
    route_pred
)


# ============================================================
# RESULTS
# ============================================================

print("=" * 75)
print("STEP 13 — ROUTE-AWARE FEATURES")
print("=" * 75)

print()
print("TEMPORAL SPLIT")
print("-" * 75)
print("Training   : Jan 1 – Aug 31, 2025")
print("Validation : Sep 1 – Oct 31, 2025")

print()
print("ROUTE COVERAGE")
print("-" * 75)

seen_routes = (val["route_count"] > 0).sum()

print(f"Training unique routes   : {train['route'].nunique():,}")
print(f"Validation unique routes : {val['route'].nunique():,}")
print(f"Validation seen routes   : {seen_routes:,}")
print(
    f"Validation unseen routes : "
    f"{len(val) - seen_routes:,}"
)

print()
print("MODEL A — CURRENT BEST")
print("-" * 75)
print(f"MAE  : ${base_mae:,.2f}")
print(f"RMSE : ${base_rmse:,.2f}")
print(f"R²   : {base_r2:.4f}")

print()
print("MODEL B — + ROUTE STATISTICS")
print("-" * 75)
print(f"MAE  : ${route_mae:,.2f}")
print(f"RMSE : ${route_rmse:,.2f}")
print(f"R²   : {route_r2:.4f}")

print()
print("CHANGE")
print("-" * 75)

mae_improvement = (
    (base_mae - route_mae)
    / base_mae
    * 100
)

rmse_improvement = (
    (base_rmse - route_rmse)
    / base_rmse
    * 100
)

print(f"MAE improvement  : {mae_improvement:+.2f}%")
print(f"RMSE improvement : {rmse_improvement:+.2f}%")

print()
print("=" * 75)