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
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])


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
# TARGET
# ============================================================

y_train = train["posted_rate"].to_numpy(dtype=float)
y_validation = validation["posted_rate"].to_numpy(dtype=float)


# ============================================================
# DISTANCE
# ============================================================

train_distance = train["distance"].to_numpy(dtype=float)
validation_distance = validation["distance"].to_numpy(dtype=float)

distance_mean = train_distance.mean()
distance_std = train_distance.std()

train_distance_scaled = (
    train_distance - distance_mean
) / distance_std

validation_distance_scaled = (
    validation_distance - distance_mean
) / distance_std


# ============================================================
# WEIGHT
# ============================================================

weight_median = train["weight"].median()

train_weight = (
    train["weight"]
    .fillna(weight_median)
    .to_numpy(dtype=float)
)

validation_weight = (
    validation["weight"]
    .fillna(weight_median)
    .to_numpy(dtype=float)
)

weight_mean = train_weight.mean()
weight_std = train_weight.std()

train_weight_scaled = (
    train_weight - weight_mean
) / weight_std

validation_weight_scaled = (
    validation_weight - weight_mean
) / weight_std


# ============================================================
# EQUIPMENT
# ============================================================

equipment_categories = sorted(
    train["equipment"].dropna().unique()
)


def encode_equipment(series, categories):

    columns = []

    for category in categories:
        columns.append(
            (series == category).astype(float).to_numpy()
        )

    return np.column_stack(columns)


train_equipment = encode_equipment(
    train["equipment"],
    equipment_categories
)

validation_equipment = encode_equipment(
    validation["equipment"],
    equipment_categories
)


# ============================================================
# GEOGRAPHIC FEATURES
# ============================================================

geo_columns = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon"
]

train_geo = train[geo_columns].to_numpy(dtype=float)
validation_geo = validation[geo_columns].to_numpy(dtype=float)

geo_mean = train_geo.mean(axis=0)
geo_std = train_geo.std(axis=0)

train_geo_scaled = (
    train_geo - geo_mean
) / geo_std

validation_geo_scaled = (
    validation_geo - geo_mean
) / geo_std


# ============================================================
# BUILD CURRENT BEST MODEL FEATURES
# ============================================================

distance_equipment_train = (
    train_distance_scaled[:, None]
    * train_equipment
)

distance_equipment_validation = (
    validation_distance_scaled[:, None]
    * validation_equipment
)

weight_equipment_train = (
    train_weight_scaled[:, None]
    * train_equipment
)

weight_equipment_validation = (
    validation_weight_scaled[:, None]
    * validation_equipment
)


X_train = np.column_stack([
    train_distance_scaled,
    train_distance_scaled ** 2,
    train_weight_scaled,
    train_equipment,
    distance_equipment_train,
    weight_equipment_train,
    train_geo_scaled
])

X_validation = np.column_stack([
    validation_distance_scaled,
    validation_distance_scaled ** 2,
    validation_weight_scaled,
    validation_equipment,
    distance_equipment_validation,
    weight_equipment_validation,
    validation_geo_scaled
])


# ============================================================
# FIT OLS
# ============================================================

X_train_intercept = np.column_stack([
    np.ones(len(X_train)),
    X_train
])

coefficients = (
    np.linalg.pinv(X_train_intercept)
    @ y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

X_validation_intercept = np.column_stack([
    np.ones(len(X_validation)),
    X_validation
])

predictions = (
    X_validation_intercept
    @ coefficients
)


# ============================================================
# ERROR DATAFRAME
# ============================================================

analysis = validation[
    [
        "load_id",
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "posted_rate"
    ]
].copy()

analysis["predicted_rate"] = predictions

analysis["error"] = (
    analysis["posted_rate"]
    - analysis["predicted_rate"]
)

analysis["absolute_error"] = (
    analysis["error"].abs()
)

analysis["squared_error"] = (
    analysis["error"] ** 2
)


# ============================================================
# OVERALL METRICS
# ============================================================

mae = analysis["absolute_error"].mean()

rmse = np.sqrt(
    analysis["squared_error"].mean()
)

print("=" * 75)
print("ERROR ANALYSIS — CURRENT BEST MODEL")
print("=" * 75)

print()
print("OVERALL")
print("-" * 75)
print(f"Validation rows     : {len(analysis):,}")
print(f"MAE                 : ${mae:,.2f}")
print(f"RMSE                : ${rmse:,.2f}")

# ============================================================
# ERROR DISTRIBUTION
# ============================================================

percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]

print()
print("ABSOLUTE ERROR DISTRIBUTION")
print("-" * 75)

for p in percentiles:
    value = np.percentile(
        analysis["absolute_error"],
        p
    )

    print(
        f"{p:>5.1f}th percentile    : ${value:,.2f}"
    )


# ============================================================
# TOP 20 ERRORS
# ============================================================

print()
print("TOP 20 ABSOLUTE ERRORS")
print("-" * 75)

top_errors = analysis.sort_values(
    "absolute_error",
    ascending=False
).head(20)

display_columns = [
    "load_id",
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "posted_rate",
    "predicted_rate",
    "error",
    "absolute_error"
]

print(
    top_errors[display_columns].to_string(
        index=False
    )
)


# ============================================================
# ERROR CONTRIBUTION OF LARGEST ERRORS
# ============================================================

total_squared_error = (
    analysis["squared_error"].sum()
)

print()
print("CONTRIBUTION TO TOTAL SQUARED ERROR")
print("-" * 75)

for fraction in [0.1, 0.5, 1.0, 2.0, 5.0]:

    n = max(
        1,
        int(len(analysis) * fraction / 100)
    )

    largest = analysis.nlargest(
        n,
        "squared_error"
    )

    contribution = (
        largest["squared_error"].sum()
        / total_squared_error
        * 100
    )

    print(
        f"Top {fraction:g}% of rows "
        f"({n:,} rows) → "
        f"{contribution:.2f}% of total squared error"
    )


# ============================================================
# ERROR BY EQUIPMENT
# ============================================================

print()
print("ERROR BY EQUIPMENT")
print("-" * 75)

equipment_summary = (
    analysis
    .groupby("equipment")
    .agg(
        rows=("load_id", "count"),
        mae=("absolute_error", "mean"),
        rmse=(
            "squared_error",
            lambda x: np.sqrt(x.mean())
        ),
        median_abs_error=(
            "absolute_error",
            "median"
        )
    )
    .sort_values("mae")
)

print(
    equipment_summary.to_string(
        float_format=lambda x: f"{x:,.2f}"
    )
)


# ============================================================
# DISTANCE BUCKETS
# ============================================================

analysis["distance_bucket"] = pd.cut(
    analysis["distance"],
    bins=[
        -np.inf,
        250,
        500,
        750,
        1000,
        1500,
        2000,
        3000,
        np.inf
    ]
)

print()
print("ERROR BY DISTANCE")
print("-" * 75)

distance_summary = (
    analysis
    .groupby("distance_bucket", observed=True)
    .agg(
        rows=("load_id", "count"),
        mae=("absolute_error", "mean"),
        rmse=(
            "squared_error",
            lambda x: np.sqrt(x.mean())
        ),
        median_rate=(
            "posted_rate",
            "median"
        )
    )
)

print(
    distance_summary.to_string(
        float_format=lambda x: f"{x:,.2f}"
    )
)


# ============================================================
# TARGET-RATE BUCKETS
# ============================================================

analysis["target_bucket"] = pd.cut(
    analysis["posted_rate"],
    bins=[
        -np.inf,
        1000,
        2000,
        3000,
        4000,
        5000,
        7500,
        10000,
        np.inf
    ]
)

print()
print("ERROR BY ACTUAL POSTED RATE")
print("-" * 75)

target_summary = (
    analysis
    .groupby("target_bucket", observed=True)
    .agg(
        rows=("load_id", "count"),
        mae=("absolute_error", "mean"),
        rmse=(
            "squared_error",
            lambda x: np.sqrt(x.mean())
        ),
        median_error=(
            "error",
            "median"
        )
    )
)

print(
    target_summary.to_string(
        float_format=lambda x: f"{x:,.2f}"
    )
)


# ============================================================
# EXTREME TARGET OBSERVATIONS
# ============================================================

threshold = analysis["posted_rate"].quantile(0.99)

extreme_target = analysis[
    analysis["posted_rate"] >= threshold
]

print()
print("EXTREME TARGET ANALYSIS")
print("-" * 75)
print(
    f"99th percentile target threshold : "
    f"${threshold:,.2f}"
)

print(
    f"Rows above threshold             : "
    f"{len(extreme_target):,}"
)

if len(extreme_target) > 0:

    print(
        f"MAE on extreme targets          : "
        f"${extreme_target['absolute_error'].mean():,.2f}"
    )

    print(
        f"RMSE on extreme targets         : "
        f"${np.sqrt(extreme_target['squared_error'].mean()):,.2f}"
    )

    print(
        f"Median actual rate              : "
        f"${extreme_target['posted_rate'].median():,.2f}"
    )

    print(
        f"Median predicted rate           : "
        f"${extreme_target['predicted_rate'].median():,.2f}"
    )


print()
print("=" * 75)