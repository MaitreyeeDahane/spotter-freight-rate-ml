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
# DEFINE EXTREME THRESHOLD
#
# IMPORTANT:
# Threshold is calculated ONLY from training data.
# ============================================================

extreme_threshold = train["posted_rate"].quantile(0.99)

train["is_extreme"] = (
    train["posted_rate"] >= extreme_threshold
)

validation["is_extreme_using_train_threshold"] = (
    validation["posted_rate"] >= extreme_threshold
)


# ============================================================
# HELPER
# ============================================================

def print_numeric_summary(data, label):

    print()
    print(label)
    print("-" * 75)

    columns = [
        "posted_rate",
        "distance",
        "weight",
        "market_index",
        "quote_signal"
    ]

    summary = data[columns].agg([
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max"
    ]).T

    print(
        summary.to_string(
            float_format=lambda x: f"{x:,.2f}"
        )
    )


# ============================================================
# BASIC COUNTS
# ============================================================

print("=" * 75)
print("EXTREME-RATE INVESTIGATION")
print("=" * 75)

print()
print("DATASETS")
print("-" * 75)
print(f"Training rows       : {len(train):,}")
print(f"Validation rows     : {len(validation):,}")

print()
print("EXTREME THRESHOLD")
print("-" * 75)
print(
    f"Training 99th percentile threshold : "
    f"${extreme_threshold:,.2f}"
)

train_extreme = train[
    train["is_extreme"]
].copy()

train_normal = train[
    ~train["is_extreme"]
].copy()

validation_extreme = validation[
    validation["is_extreme_using_train_threshold"]
].copy()

validation_normal = validation[
    ~validation["is_extreme_using_train_threshold"]
].copy()

print(
    f"Training extreme rows              : "
    f"{len(train_extreme):,} "
    f"({len(train_extreme) / len(train) * 100:.2f}%)"
)

print(
    f"Validation extreme rows            : "
    f"{len(validation_extreme):,} "
    f"({len(validation_extreme) / len(validation) * 100:.2f}%)"
)


# ============================================================
# NUMERIC COMPARISON — TRAINING
# ============================================================

print()
print("=" * 75)
print("TRAINING — NORMAL VS EXTREME")
print("=" * 75)

print_numeric_summary(
    train_normal,
    "NORMAL TRAINING ROWS"
)

print_numeric_summary(
    train_extreme,
    "EXTREME TRAINING ROWS"
)


# ============================================================
# NUMERIC COMPARISON — VALIDATION
# ============================================================

print()
print("=" * 75)
print("VALIDATION — NORMAL VS EXTREME")
print("=" * 75)

print_numeric_summary(
    validation_normal,
    "NORMAL VALIDATION ROWS"
)

print_numeric_summary(
    validation_extreme,
    "EXTREME VALIDATION ROWS"
)


# ============================================================
# EQUIPMENT DISTRIBUTION
# ============================================================

print()
print("=" * 75)
print("EQUIPMENT DISTRIBUTION")
print("=" * 75)

print()
print("TRAINING — NORMAL")
print("-" * 75)

print(
    train_normal["equipment"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .to_string()
)

print()
print("TRAINING — EXTREME")
print("-" * 75)

print(
    train_extreme["equipment"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .to_string()
)

print()
print("VALIDATION — NORMAL")
print("-" * 75)

print(
    validation_normal["equipment"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .to_string()
)

print()
print("VALIDATION — EXTREME")
print("-" * 75)

print(
    validation_extreme["equipment"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .to_string()
)


# ============================================================
# MONTH DISTRIBUTION
# ============================================================

for name, data in [
    ("TRAINING NORMAL", train_normal),
    ("TRAINING EXTREME", train_extreme),
    ("VALIDATION NORMAL", validation_normal),
    ("VALIDATION EXTREME", validation_extreme)
]:

    print()
    print(name)
    print("-" * 75)

    monthly = (
        data["date"]
        .dt.month
        .value_counts()
        .sort_index()
    )

    print(monthly.to_string())


# ============================================================
# EXTREME RATE BY MONTH
# ============================================================

print()
print("=" * 75)
print("EXTREME-RATE FREQUENCY BY MONTH")
print("=" * 75)

train_month_stats = (
    train
    .groupby(train["date"].dt.month)
    .agg(
        rows=("posted_rate", "size"),
        extreme_rows=("is_extreme", "sum"),
        median_rate=("posted_rate", "median"),
        mean_rate=("posted_rate", "mean")
    )
)

train_month_stats["extreme_pct"] = (
    train_month_stats["extreme_rows"]
    / train_month_stats["rows"]
    * 100
)

print(
    train_month_stats.to_string(
        float_format=lambda x: f"{x:,.2f}"
    )
)


# ============================================================
# ROUTE ANALYSIS
# ============================================================

train["route"] = (
    train["pickup"]
    + " → "
    + train["delivery"]
)

validation["route"] = (
    validation["pickup"]
    + " → "
    + validation["delivery"]
)

train_extreme["route"] = (
    train_extreme["pickup"]
    + " → "
    + train_extreme["delivery"]
)

validation_extreme["route"] = (
    validation_extreme["pickup"]
    + " → "
    + validation_extreme["delivery"]
)


print()
print("=" * 75)
print("ROUTE ANALYSIS")
print("=" * 75)

print()
print(
    f"Unique training routes           : "
    f"{train['route'].nunique():,}"
)

print(
    f"Unique validation routes         : "
    f"{validation['route'].nunique():,}"
)

print(
    f"Unique extreme training routes   : "
    f"{train_extreme.assign(route=train_extreme['pickup'] + ' → ' + train_extreme['delivery'])['route'].nunique():,}"
)

print(
    f"Unique extreme validation routes : "
    f"{validation_extreme.assign(route=validation_extreme['pickup'] + ' → ' + validation_extreme['delivery'])['route'].nunique():,}"
)


# ============================================================
# MOST FREQUENT EXTREME ROUTES — TRAINING
# ============================================================

print()
print("TOP EXTREME TRAINING ROUTES")
print("-" * 75)

route_extreme_counts = (
    train_extreme
    .groupby("route")
    .agg(
        extreme_rows=("load_id", "count"),
        median_rate=("posted_rate", "median"),
        max_rate=("posted_rate", "max")
    )
    .sort_values(
        ["extreme_rows", "max_rate"],
        ascending=False
    )
    .head(20)
)

print(
    route_extreme_counts.to_string(
        float_format=lambda x: f"{x:,.2f}"
    )
)


# ============================================================
# ROUTES WITH EXTREME RATE BEHAVIOR
#
# Only examine routes with at least 2 training observations.
# ============================================================

route_stats = (
    train
    .groupby("route")
    .agg(
        rows=("posted_rate", "size"),
        extreme_rows=("is_extreme", "sum"),
        median_rate=("posted_rate", "median"),
        max_rate=("posted_rate", "max"),
        mean_rate=("posted_rate", "mean")
    )
)

route_stats["extreme_pct"] = (
    route_stats["extreme_rows"]
    / route_stats["rows"]
    * 100
)

interesting_routes = (
    route_stats[
        route_stats["rows"] >= 2
    ]
    .sort_values(
        ["extreme_rows", "max_rate"],
        ascending=False
    )
    .head(30)
)

print()
print("ROUTES WITH MULTIPLE TRAINING OBSERVATIONS")
print("-" * 75)

print(
    interesting_routes.to_string(
        float_format=lambda x: f"{x:,.2f}"
    )
)


# ============================================================
# EXTREME VALIDATION ROUTES
# ============================================================

print()
print("EXTREME VALIDATION ROUTES")
print("-" * 75)

validation_extreme_routes = (
    validation_extreme
    .groupby("route")
    .agg(
        extreme_rows=("load_id", "count"),
        median_rate=("posted_rate", "median"),
        max_rate=("posted_rate", "max")
    )
    .sort_values(
        ["extreme_rows", "max_rate"],
        ascending=False
    )
    .head(20)
)

print(
    validation_extreme_routes.to_string(
        float_format=lambda x: f"{x:,.2f}"
    )
)


# ============================================================
# MARKET / QUOTE SIGNAL COMPARISON
# ============================================================

print()
print("=" * 75)
print("MARKET / QUOTE SIGNAL")
print("=" * 75)

market_summary = pd.DataFrame({
    "group": [
        "Training Normal",
        "Training Extreme",
        "Validation Normal",
        "Validation Extreme"
    ],
    "market_mean": [
        train_normal["market_index"].mean(),
        train_extreme["market_index"].mean(),
        validation_normal["market_index"].mean(),
        validation_extreme["market_index"].mean()
    ],
    "market_median": [
        train_normal["market_index"].median(),
        train_extreme["market_index"].median(),
        validation_normal["market_index"].median(),
        validation_extreme["market_index"].median()
    ],
    "quote_mean": [
        train_normal["quote_signal"].mean(),
        train_extreme["quote_signal"].mean(),
        validation_normal["quote_signal"].mean(),
        validation_extreme["quote_signal"].mean()
    ],
    "quote_median": [
        train_normal["quote_signal"].median(),
        train_extreme["quote_signal"].median(),
        validation_normal["quote_signal"].median(),
        validation_extreme["quote_signal"].median()
    ]
})

print(
    market_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:,.4f}"
    )
)


# ============================================================
# EXTREME OBSERVATIONS
# ============================================================

print()
print("=" * 75)
print("TOP 25 EXTREME TRAINING OBSERVATIONS")
print("=" * 75)

columns = [
    "load_id",
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
    "posted_rate"
]

print(
    train_extreme
    .sort_values("posted_rate", ascending=False)
    [columns]
    .head(25)
    .to_string(index=False)
)


print()
print("=" * 75)
print("TOP 25 EXTREME VALIDATION OBSERVATIONS")
print("=" * 75)

print(
    validation_extreme
    .sort_values("posted_rate", ascending=False)
    [columns]
    .head(25)
    .to_string(index=False)
)


print()
print("=" * 75)
print("END OF EXTREME-RATE INVESTIGATION")
print("=" * 75)