import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"

TRAIN_FILE = os.path.join(DATA_DIR, "train-test.csv")
VALIDATION_FILE = os.path.join(DATA_DIR, "validation.csv")
DECEMBER_FILE = os.path.join(DATA_DIR, "december-chart-inputs.csv")
TEMPLATE_FILE = os.path.join(
    DATA_DIR,
    "validation-predictions-template.csv"
)

VALIDATION_OUTPUT = "validation_predictions.csv"
DECEMBER_OUTPUT = "december_predictions.csv"

# Final model intentionally uses only features that are
# available or deterministically derivable at inference time.


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
december_df = pd.read_csv(DECEMBER_FILE)


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_dates(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


train_df = prepare_dates(train_df)
validation_df = prepare_dates(validation_df)
december_df = prepare_dates(december_df)


# ============================================================
# CITY → COORDINATE MAPPING
# ============================================================

# December inputs do not contain coordinate columns.
# Coordinates are deterministic attributes of the cities in
# the development data, so we learn the mapping from labeled
# development data only.

def build_city_coordinate_maps(train):

    pickup_map = (
        train[
            ["pickup", "pickup_lat", "pickup_lon"]
        ]
        .dropna()
        .drop_duplicates("pickup")
        .set_index("pickup")
        .to_dict("index")
    )

    delivery_map = (
        train[
            ["delivery", "delivery_lat", "delivery_lon"]
        ]
        .dropna()
        .drop_duplicates("delivery")
        .set_index("delivery")
        .to_dict("index")
    )

    return pickup_map, delivery_map


pickup_coord_map, delivery_coord_map = (
    build_city_coordinate_maps(train_df)
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(
    data,
    train_reference,
    pickup_map,
    delivery_map
):

    data = data.copy()

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    # Use supplied coordinates when available.
    # Otherwise derive them from the city mapping.
    # Final fallback = training mean.
    #
    # This makes the same pipeline compatible with:
    #   - train_test.csv
    #   - validation.csv
    #   - december_chart_inputs.csv

    coordinate_pairs = [
        ("pickup", "pickup_lat", "pickup_lat"),
        ("pickup", "pickup_lon", "pickup_lon"),
        ("delivery", "delivery_lat", "delivery_lat"),
        ("delivery", "delivery_lon", "delivery_lon"),
    ]

    for city_col, coord_col, output_col in coordinate_pairs:

        if coord_col in data.columns:

            data[output_col] = data[coord_col]

        else:

            data[output_col] = np.nan

            mapping = (
                pickup_map
                if city_col == "pickup"
                else delivery_map
            )

            data[output_col] = data[city_col].map(
                lambda city: (
                    mapping.get(city, {}).get(coord_col, np.nan)
                )
            )

    # --------------------------------------------------------
    # Training-derived imputation values
    # --------------------------------------------------------

    weight_median = train_reference["weight"].median()

    # --------------------------------------------------------
    # Weight
    # --------------------------------------------------------

    data["weight_imputed"] = data["weight"].fillna(
        weight_median
    )

    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    data["distance_sq"] = (
        data["distance"] ** 2
    )

    # --------------------------------------------------------
    # Equipment
    # --------------------------------------------------------

    data["reefer"] = (
        data["equipment"] == "Reefer"
    ).astype(float)

    data["flatbed"] = (
        data["equipment"] == "Flatbed"
    ).astype(float)

    # --------------------------------------------------------
    # Equipment interactions
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Geographic scaling
    # --------------------------------------------------------

    geo_columns = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon"
    ]

    for column in geo_columns:

        train_mean = train_reference[column].mean()
        train_std = train_reference[column].std()

        if pd.isna(train_std) or train_std == 0:
            train_std = 1.0

        data[f"{column}_scaled"] = (
            data[column]
            .fillna(train_mean)
            .sub(train_mean)
            .div(train_std)
        )

    # --------------------------------------------------------
    # Final feature set
    # --------------------------------------------------------

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

    X = data[features].copy()

    # Safety check
    if X.isna().any().any():
        missing_columns = X.columns[
            X.isna().any()
        ].tolist()

        raise ValueError(
            "Missing values remain in model features: "
            + str(missing_columns)
        )

    return X


# ============================================================
# BUILD FEATURES
# ============================================================

X_train = build_features(
    train_df,
    train_df,
    pickup_coord_map,
    delivery_coord_map
)

X_validation = build_features(
    validation_df,
    train_df,
    pickup_coord_map,
    delivery_coord_map
)

X_december = build_features(
    december_df,
    train_df,
    pickup_coord_map,
    delivery_coord_map
)

y_train = train_df["posted_rate"]


# ============================================================
# FINAL MODEL
# ============================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

validation_predictions = model.predict(
    X_validation
)

validation_predictions = np.asarray(
    validation_predictions,
    dtype=float
)

if not np.isfinite(validation_predictions).all():
    raise ValueError(
        "Validation predictions contain non-finite values."
    )

if (validation_predictions <= 0).any():
    raise ValueError(
        "Validation predictions contain non-positive values."
    )


# ============================================================
# CREATE REQUIRED VALIDATION FILE
# ============================================================

template = pd.read_csv(TEMPLATE_FILE)

required_template_columns = [
    "load_id",
    "predicted_rate"
]

if list(template.columns) != required_template_columns:
    raise ValueError(
        "Unexpected validation template columns: "
        f"{template.columns.tolist()}"
    )

if len(template) != len(validation_df):
    raise ValueError(
        "Validation template row count does not match "
        "validation.csv."
    )

validation_output = template.copy()

validation_output["predicted_rate"] = (
    validation_predictions
)

validation_output.to_csv(
    VALIDATION_OUTPUT,
    index=False
)


# ============================================================
# DECEMBER PREDICTIONS
# ============================================================

december_predictions = model.predict(
    X_december
)

december_predictions = np.asarray(
    december_predictions,
    dtype=float
)

if not np.isfinite(december_predictions).all():
    raise ValueError(
        "December predictions contain non-finite values."
    )

if (december_predictions <= 0).any():
    raise ValueError(
        "December predictions contain non-positive values."
    )


# ============================================================
# CREATE DECEMBER PREDICTION FILE
# ============================================================

december_output = december_df[
    [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date"
    ]
].copy()

december_output["date"] = (
    december_output["date"]
    .dt.strftime("%Y-%m-%d")
)

december_output["predicted_rate"] = (
    december_predictions
)

december_output.to_csv(
    DECEMBER_OUTPUT,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 75)
print("FINAL MODEL TRAINING COMPLETE")
print("=" * 75)

print()
print("Training rows        :", len(train_df))
print("Validation rows      :", len(validation_df))
print("December rows        :", len(december_df))

print()
print("Model                : Linear Regression")

print()
print("Features")
print("-" * 75)

for feature in X_train.columns:
    print(" -", feature)

print()
print("Outputs")
print("-" * 75)

print(
    "Validation predictions:",
    VALIDATION_OUTPUT
)

print(
    "December predictions  :",
    DECEMBER_OUTPUT
)

print()
print("Prediction statistics")
print("-" * 75)

print(
    f"Validation min : ${validation_predictions.min():,.2f}"
)

print(
    f"Validation max : ${validation_predictions.max():,.2f}"
)

print(
    f"Validation mean: ${validation_predictions.mean():,.2f}"
)

print(
    f"December min   : ${december_predictions.min():,.2f}"
)

print(
    f"December max   : ${december_predictions.max():,.2f}"
)

print(
    f"December mean  : ${december_predictions.mean():,.2f}"
)

print()
print("=" * 75)