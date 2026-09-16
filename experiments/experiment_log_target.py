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
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

train = df[
    df["date"] <= TRAIN_END
].copy()

val = df[
    (df["date"] >= VAL_START) &
    (df["date"] <= VAL_END)
].copy()


# ============================================================
# FEATURE ENGINEERING
# Same feature family as current best model
# ============================================================

def build_features(data, train_reference):
    data = data.copy()

    # ----------------------------
    # Basic numeric features
    # ----------------------------
    data["weight_imputed"] = data["weight"].fillna(
        train_reference["weight"].median()
    )

    data["distance_sq"] = data["distance"] ** 2

    # ----------------------------
    # Equipment encoding
    # Drop Dry Van as reference category
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
    # Geographic features
    # Standardize using TRAINING statistics
    # ----------------------------
    geo_cols = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
    ]

    for col in geo_cols:
        train_mean = train_reference[col].mean()
        train_std = train_reference[col].std()

        if train_std == 0 or pd.isna(train_std):
            train_std = 1.0

        data[f"{col}_scaled"] = (
            data[col].fillna(train_mean) - train_mean
        ) / train_std

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
        "delivery_lon_scaled",
    ]

    return data[features]


# ============================================================
# BUILD FEATURES
# ============================================================

X_train = build_features(train, train)
X_val = build_features(val, train)

y_train = train["posted_rate"].values
y_val = val["posted_rate"].values


# ============================================================
# LOG-TARGET MODEL
# ============================================================

y_train_log = np.log1p(y_train)

model = LinearRegression()
model.fit(X_train, y_train_log)

pred_log = model.predict(X_val)

# Convert back to dollars
pred = np.expm1(pred_log)

# Safety
pred = np.maximum(pred, 0)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(y_val, pred)
rmse = np.sqrt(mean_squared_error(y_val, pred))
r2 = r2_score(y_val, pred)

print("=" * 70)
print("STEP 12 — LOG-TARGET REGRESSION")
print("=" * 70)

print(f"Training rows : {len(train):,}")
print(f"Validation rows : {len(val):,}")

print()
print("TEMPORAL SPLIT")
print("-" * 70)
print("Training   : Jan 1 – Aug 31, 2025")
print("Validation : Sep 1 – Oct 31, 2025")

print()
print("FEATURES")
print("-" * 70)
print("Distance + distance²")
print("Weight + equipment")
print("Distance × equipment")
print("Weight × equipment")
print("Geographic coordinates")

print()
print("TARGET TRANSFORMATION")
print("-" * 70)
print("Training target: log1p(posted_rate)")
print("Predictions: expm1() inverse transformation")

print()
print("RESULTS")
print("-" * 70)
print(f"MAE  : ${mae:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"R²   : {r2:.4f}")

print()
print("=" * 70)