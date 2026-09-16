import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score
)


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
# EXTREME THRESHOLD
#
# Calculated ONLY from training period.
# ============================================================

extreme_threshold = train["posted_rate"].quantile(0.99)

train["is_extreme"] = (
    train["posted_rate"] >= extreme_threshold
).astype(int)

val["is_extreme_actual"] = (
    val["posted_rate"] >= extreme_threshold
).astype(int)


# ============================================================
# FEATURE ENGINEERING
#
# Only features available from the load itself.
# No target-derived features.
# ============================================================

def build_features(data, train_reference):

    data = data.copy()

    weight_median = train_reference["weight"].median()

    data["weight_imputed"] = data["weight"].fillna(
        weight_median
    )

    # Distance
    data["distance_sq"] = data["distance"] ** 2
    data["distance_log"] = np.log1p(data["distance"])

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
        "delivery_lon_scaled"
    ]

    return data[features]


# ============================================================
# BUILD FEATURES
# ============================================================

X_train = build_features(train, train)
X_val = build_features(val, train)

y_train = train["posted_rate"].values
y_val = val["posted_rate"].values

y_train_extreme = train["is_extreme"].values


# ============================================================
# STAGE 1 — EXTREME CLASSIFIER
# ============================================================

classifier = HistGradientBoostingClassifier(
    learning_rate=0.04,
    max_iter=250,
    max_leaf_nodes=15,
    max_depth=4,
    min_samples_leaf=40,
    l2_regularization=3.0,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=25,
    random_state=42
)

classifier.fit(
    X_train,
    y_train_extreme
)

train_extreme_probability = classifier.predict_proba(
    X_train
)[:, 1]

val_extreme_probability = classifier.predict_proba(
    X_val
)[:, 1]


# ============================================================
# STAGE 2A — NORMAL REGRESSION
#
# Train on normal observations only.
# ============================================================

normal_train_mask = (
    train["is_extreme"] == 0
)

normal_model = LinearRegression()

normal_model.fit(
    X_train[normal_train_mask],
    y_train[normal_train_mask]
)


# ============================================================
# STAGE 2B — EXTREME REGRESSION
#
# Small extreme sample, therefore keep this model simple.
# ============================================================

extreme_train_mask = (
    train["is_extreme"] == 1
)

extreme_model = LinearRegression()

extreme_model.fit(
    X_train[extreme_train_mask],
    y_train[extreme_train_mask]
)


# ============================================================
# GENERATE BOTH REGRESSION PREDICTIONS
# ============================================================

normal_prediction = normal_model.predict(
    X_val
)

extreme_prediction = extreme_model.predict(
    X_val
)


# ============================================================
# SOFT BLENDING
#
# Probability determines how much the prediction moves
# toward the extreme-rate model.
#
# This avoids a hard 0/1 classifier boundary.
# ============================================================

final_prediction = (
    (1 - val_extreme_probability) * normal_prediction
    + val_extreme_probability * extreme_prediction
)

final_prediction = np.maximum(
    final_prediction,
    0
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


train_normal_prediction = normal_model.predict(
    X_train
)

train_extreme_prediction = extreme_model.predict(
    X_train
)

train_blended_prediction = (
    (1 - train_extreme_probability) *
    train_normal_prediction
    +
    train_extreme_probability *
    train_extreme_prediction
)

train_blended_prediction = np.maximum(
    train_blended_prediction,
    0
)


train_mae, train_rmse, train_r2 = evaluate(
    y_train,
    train_blended_prediction
)

val_mae, val_rmse, val_r2 = evaluate(
    y_val,
    final_prediction
)


# ============================================================
# CLASSIFICATION QUALITY
# ============================================================

roc_auc = roc_auc_score(
    val["is_extreme_actual"],
    val_extreme_probability
)


# ============================================================
# EXTREME / NORMAL ERROR
# ============================================================

val_extreme_mask = (
    val["is_extreme_actual"] == 1
)

val_normal_mask = ~val_extreme_mask

normal_mae = mean_absolute_error(
    y_val[val_normal_mask],
    final_prediction[val_normal_mask]
)

extreme_mae = mean_absolute_error(
    y_val[val_extreme_mask],
    final_prediction[val_extreme_mask]
)


# ============================================================
# GENERALIZATION GAP
# ============================================================

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
# RESULTS
# ============================================================

print("=" * 75)
print("STEP 15 — TWO-STAGE EXTREME-RATE MODEL")
print("=" * 75)

print()
print("TEMPORAL SPLIT")
print("-" * 75)
print("Training   : Jan 1 – Aug 31, 2025")
print("Validation : Sep 1 – Oct 31, 2025")

print()
print("EXTREME DEFINITION")
print("-" * 75)
print(f"Training 99th percentile : ${extreme_threshold:,.2f}")
print(
    f"Training extreme rows   : "
    f"{train['is_extreme'].sum():,}"
)
print(
    f"Validation extreme rows : "
    f"{val['is_extreme_actual'].sum():,}"
)

print()
print("CLASSIFIER")
print("-" * 75)
print(f"Validation ROC-AUC : {roc_auc:.4f}")

print()
print("TRAINING PERFORMANCE")
print("-" * 75)
print(f"MAE  : ${train_mae:,.2f}")
print(f"RMSE : ${train_rmse:,.2f}")
print(f"R²   : {train_r2:.4f}")

print()
print("OUT-OF-TIME VALIDATION")
print("-" * 75)
print(f"MAE  : ${val_mae:,.2f}")
print(f"RMSE : ${val_rmse:,.2f}")
print(f"R²   : {val_r2:.4f}")

print()
print("GENERALIZATION GAP")
print("-" * 75)
print(f"MAE gap  : {mae_gap_pct:+.2f}%")
print(f"RMSE gap : {rmse_gap_pct:+.2f}%")

print()
print("ERROR BY REGIME")
print("-" * 75)
print(f"Normal-rate MAE  : ${normal_mae:,.2f}")
print(f"Extreme-rate MAE : ${extreme_mae:,.2f}")

print()
print("=" * 75)