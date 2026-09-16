import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
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

def build_features(data, train_reference):

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

    # Useful nonlinear distance representation
    data["distance_log"] = np.log1p(
        data["distance"]
    )

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
    # Equipment interactions
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

    # ----------------------------
    # Calendar
    # ----------------------------

    data["month"] = data["date"].dt.month
    data["day_of_week"] = data["date"].dt.dayofweek

    # Cyclic calendar representation
    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    data["dow_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["dow_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    # ----------------------------
    # Final feature set
    # ----------------------------

    features = [
        "distance",
        "distance_sq",
        "distance_log",
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
        "delivery_lon_scaled",

        "month",
        "day_of_week",

        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
    ]

    return data[features]


# ============================================================
# FEATURES
# ============================================================

X_train = build_features(train, train)
X_val = build_features(val, train)

y_train = train["posted_rate"].values
y_val = val["posted_rate"].values


# ============================================================
# MODEL
#
# Conservative complexity:
# - limited tree depth
# - limited leaves
# - learning rate kept low
# - early stopping enabled
# ============================================================

model = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.035,
    max_leaf_nodes=15,
    max_depth=5,
    min_samples_leaf=30,
    l2_regularization=2.0,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    random_state=42
)


# ============================================================
# TRAIN
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)


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


train_mae, train_rmse, train_r2 = evaluate(
    y_train,
    train_pred
)

val_mae, val_rmse, val_r2 = evaluate(
    y_val,
    val_pred
)


# ============================================================
# GENERALIZATION GAP
# ============================================================

mae_gap = val_mae - train_mae
rmse_gap = val_rmse - train_rmse

mae_gap_pct = (
    (val_mae - train_mae)
    / train_mae
    * 100
)

rmse_gap_pct = (
    (val_rmse - train_rmse)
    / train_rmse
    * 100
)


# ============================================================
# EXTREME-RATE ERROR
# ============================================================

threshold = y_train.copy()

extreme_threshold = np.percentile(
    y_train,
    99
)

val_extreme_mask = (
    y_val >= extreme_threshold
)

normal_mask = ~val_extreme_mask

extreme_mae = mean_absolute_error(
    y_val[val_extreme_mask],
    val_pred[val_extreme_mask]
)

normal_mae = mean_absolute_error(
    y_val[normal_mask],
    val_pred[normal_mask]
)


# ============================================================
# RESULTS
# ============================================================

print("=" * 75)
print("STEP 14 — RICH NONLINEAR TREE MODEL")
print("=" * 75)

print()
print("TEMPORAL SPLIT")
print("-" * 75)
print("Training   : Jan 1 – Aug 31, 2025")
print("Validation : Sep 1 – Oct 31, 2025")

print()
print("FEATURES")
print("-" * 75)
print("Distance + distance² + log(distance)")
print("Weight")
print("Equipment")
print("Distance × equipment")
print("Weight × equipment")
print("Geographic coordinates")
print("Calendar + cyclic calendar features")

print()
print("MODEL")
print("-" * 75)
print(model)

print()
print(f"Actual boosting iterations : {model.n_iter_}")

print()
print("TRAINING PERFORMANCE")
print("-" * 75)
print(f"MAE  : ${train_mae:,.2f}")
print(f"RMSE : ${train_rmse:,.2f}")
print(f"R²   : {train_r2:.4f}")

print()
print("OUT-OF-TIME VALIDATION PERFORMANCE")
print("-" * 75)
print(f"MAE  : ${val_mae:,.2f}")
print(f"RMSE : ${val_rmse:,.2f}")
print(f"R²   : {val_r2:.4f}")

print()
print("GENERALIZATION GAP")
print("-" * 75)
print(f"MAE gap  : ${mae_gap:,.2f} ({mae_gap_pct:+.2f}%)")
print(f"RMSE gap : ${rmse_gap:,.2f} ({rmse_gap_pct:+.2f}%)")

print()
print("EXTREME-RATE ANALYSIS")
print("-" * 75)
print(f"Extreme threshold : ${extreme_threshold:,.2f}")
print(
    f"Validation extreme rows : "
    f"{val_extreme_mask.sum():,}"
)
print(f"Normal-rate MAE : ${normal_mae:,.2f}")
print(f"Extreme-rate MAE: ${extreme_mae:,.2f}")

print()
print("=" * 75)