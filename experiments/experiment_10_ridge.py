import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
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

train = df[df["date"] <= TRAIN_END].copy()

val = df[
    (df["date"] >= VAL_START) &
    (df["date"] <= VAL_END)
].copy()


# ============================================================
# FEATURE ENGINEERING
# Same feature set as current best model
# ============================================================

def build_features(data, train_reference):

    data = data.copy()

    # ----------------------------
    # Weight imputation
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

    return data[features]


# ============================================================
# FEATURES / TARGET
# ============================================================

X_train = build_features(train, train)
X_val = build_features(val, train)

y_train = train["posted_rate"].values
y_val = val["posted_rate"].values


# ============================================================
# MODELS
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Ridge alpha=1": Ridge(
        alpha=1.0
    ),

    "Ridge alpha=10": Ridge(
        alpha=10.0
    ),

    "Ridge alpha=100": Ridge(
        alpha=100.0
    )
}


# ============================================================
# EVALUATION
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


print("=" * 80)
print("STEP 19 — RIDGE REGULARIZATION ROBUSTNESS TEST")
print("=" * 80)

print()
print("TEMPORAL SPLIT")
print("-" * 80)
print("Training   : Jan 1 – Aug 31, 2025")
print("Validation : Sep 1 – Oct 31, 2025")

results = []


# ============================================================
# TRAIN / EVALUATE
# ============================================================

for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    train_pred = model.predict(
        X_train
    )

    val_pred = model.predict(
        X_val
    )

    train_mae, train_rmse, train_r2 = evaluate(
        y_train,
        train_pred
    )

    val_mae, val_rmse, val_r2 = evaluate(
        y_val,
        val_pred
    )

    mae_gap = (
        (val_mae - train_mae)
        / train_mae
        * 100
    )

    rmse_gap = (
        (val_rmse - train_rmse)
        / train_rmse
        * 100
    )

    results.append({
        "model": name,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "val_r2": val_r2,
        "mae_gap": mae_gap,
        "rmse_gap": rmse_gap
    })


# ============================================================
# DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(results)

for _, row in results_df.iterrows():

    print()
    print(row["model"])
    print("-" * 80)

    print(
        f"Train MAE  : ${row['train_mae']:,.2f}"
    )

    print(
        f"Val MAE    : ${row['val_mae']:,.2f}"
    )

    print(
        f"Train RMSE : ${row['train_rmse']:,.2f}"
    )

    print(
        f"Val RMSE   : ${row['val_rmse']:,.2f}"
    )

    print(
        f"Val R²     : {row['val_r2']:.4f}"
    )

    print(
        f"MAE gap    : {row['mae_gap']:+.2f}%"
    )

    print(
        f"RMSE gap   : {row['rmse_gap']:+.2f}%"
    )


# ============================================================
# COMPARISON AGAINST LINEAR REGRESSION
# ============================================================

linear_mae = results_df.loc[
    results_df["model"] == "Linear Regression",
    "val_mae"
].iloc[0]

linear_rmse = results_df.loc[
    results_df["model"] == "Linear Regression",
    "val_rmse"
].iloc[0]


print()
print("=" * 80)
print("CHANGE RELATIVE TO LINEAR REGRESSION")
print("=" * 80)

for _, row in results_df.iterrows():

    if row["model"] == "Linear Regression":
        continue

    mae_change = (
        (linear_mae - row["val_mae"])
        / linear_mae
        * 100
    )

    rmse_change = (
        (linear_rmse - row["val_rmse"])
        / linear_rmse
        * 100
    )

    print()
    print(row["model"])
    print(
        f"Validation MAE change  : {mae_change:+.2f}%"
    )
    print(
        f"Validation RMSE change : {rmse_change:+.2f}%"
    )

print()
print("=" * 80)