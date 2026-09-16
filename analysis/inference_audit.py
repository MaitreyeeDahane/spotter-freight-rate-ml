import pandas as pd
import numpy as np


# ============================================================
# LOAD
# ============================================================

train = pd.read_csv("train-test.csv")
validation = pd.read_csv("validation.csv")
december = pd.read_csv("december-chart-inputs.csv")


print("=" * 75)
print("STEP 17 — INFERENCE COMPATIBILITY AUDIT")
print("=" * 75)


# ============================================================
# COLUMN COMPARISON
# ============================================================

print()
print("COLUMN AVAILABILITY")
print("-" * 75)

train_cols = set(train.columns)
validation_cols = set(validation.columns)
december_cols = set(december.columns)

print("TRAINING ONLY")
print(sorted(train_cols - validation_cols))

print()
print("VALIDATION ONLY")
print(sorted(validation_cols - train_cols))

print()
print("DECEMBER ONLY")
print(sorted(december_cols - train_cols))


# ============================================================
# COMMON PREDICTOR COLUMNS
# ============================================================

predictor_cols = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal"
]

print()
print("PREDICTOR AVAILABILITY")
print("-" * 75)

for col in predictor_cols:

    print(
        f"{col:18s} | "
        f"train={'YES' if col in train_cols else 'NO':3s} | "
        f"validation={'YES' if col in validation_cols else 'NO':3s} | "
        f"december={'YES' if col in december_cols else 'NO':3s}"
    )


# ============================================================
# CITY → COORDINATE CONSISTENCY
# ============================================================

print()
print("CITY → COORDINATE CONSISTENCY")
print("-" * 75)


def check_city_coordinate_consistency(
    data,
    city_col,
    lat_col,
    lon_col
):

    result = (
        data.groupby(city_col)[[lat_col, lon_col]]
        .nunique()
    )

    inconsistent = result[
        (result[lat_col] > 1) |
        (result[lon_col] > 1)
    ]

    return inconsistent


pickup_inconsistent = check_city_coordinate_consistency(
    train,
    "pickup",
    "pickup_lat",
    "pickup_lon"
)

delivery_inconsistent = check_city_coordinate_consistency(
    train,
    "delivery",
    "delivery_lat",
    "delivery_lon"
)


print(
    f"Pickup cities with inconsistent coordinates : "
    f"{len(pickup_inconsistent)}"
)

print(
    f"Delivery cities with inconsistent coordinates: "
    f"{len(delivery_inconsistent)}"
)


# ============================================================
# CITY COVERAGE
# ============================================================

train_pickup_cities = set(
    train["pickup"].dropna().unique()
)

train_delivery_cities = set(
    train["delivery"].dropna().unique()
)

december_pickup_cities = set(
    december["pickup"].dropna().unique()
)

december_delivery_cities = set(
    december["delivery"].dropna().unique()
)


print()
print("DECEMBER CITY COVERAGE")
print("-" * 75)

print(
    "Unseen December pickup cities:",
    sorted(
        december_pickup_cities -
        train_pickup_cities
    )
)

print(
    "Unseen December delivery cities:",
    sorted(
        december_delivery_cities -
        train_delivery_cities
    )
)


# ============================================================
# BUILD CITY → COORDINATE MAP
# ============================================================

pickup_coordinates = (
    train[
        ["pickup", "pickup_lat", "pickup_lon"]
    ]
    .dropna()
    .drop_duplicates()
)

delivery_coordinates = (
    train[
        ["delivery", "delivery_lat", "delivery_lon"]
    ]
    .dropna()
    .drop_duplicates()
)


pickup_coordinate_map = (
    pickup_coordinates
    .set_index("pickup")
    .to_dict("index")
)

delivery_coordinate_map = (
    delivery_coordinates
    .set_index("delivery")
    .to_dict("index")
)


# ============================================================
# TEST DECEMBER COORDINATE DERIVATION
# ============================================================

december_test = december.copy()

december_test["derived_pickup_lat"] = (
    december_test["pickup"]
    .map(
        {
            city: values["pickup_lat"]
            for city, values
            in pickup_coordinate_map.items()
        }
    )
)

december_test["derived_pickup_lon"] = (
    december_test["pickup"]
    .map(
        {
            city: values["pickup_lon"]
            for city, values
            in pickup_coordinate_map.items()
        }
    )
)

december_test["derived_delivery_lat"] = (
    december_test["delivery"]
    .map(
        {
            city: values["delivery_lat"]
            for city, values
            in delivery_coordinate_map.items()
        }
    )
)

december_test["derived_delivery_lon"] = (
    december_test["delivery"]
    .map(
        {
            city: values["delivery_lon"]
            for city, values
            in delivery_coordinate_map.items()
        }
    )
)


# ============================================================
# DERIVATION RESULTS
# ============================================================

derived_cols = [
    "derived_pickup_lat",
    "derived_pickup_lon",
    "derived_delivery_lat",
    "derived_delivery_lon"
]

print()
print("DECEMBER COORDINATE DERIVATION")
print("-" * 75)

for col in derived_cols:

    missing = december_test[col].isna().sum()

    print(
        f"{col:25s}: "
        f"{len(december_test) - missing}/{len(december_test)} "
        f"rows successfully derived"
    )


# ============================================================
# DECEMBER INPUT CHARACTERISTICS
# ============================================================

print()
print("DECEMBER INPUTS")
print("-" * 75)

print(
    "Rows:",
    len(december)
)

print(
    "Pickup:",
    december["pickup"].unique()
)

print(
    "Delivery:",
    december["delivery"].unique()
)

print(
    "Equipment:",
    december["equipment"].unique()
)

print(
    "Distance:",
    december["distance"].unique()
)

print(
    "Weight:",
    december["weight"].unique()
)

print(
    "Date range:",
    december["date"].min(),
    "→",
    december["date"].max()
)


# ============================================================
# MISSINGNESS
# ============================================================

print()
print("DECEMBER MISSING VALUES")
print("-" * 75)

print(
    december.isna().sum()
)


# ============================================================
# MARKET / QUOTE AVAILABILITY
# ============================================================

print()
print("UNAVAILABLE FEATURES")
print("-" * 75)

for col in ["market_index", "quote_signal"]:

    print(
        f"{col}: "
        f"available in December = "
        f"{col in december.columns}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("AUDIT COMPLETE")
print("=" * 75)