

# Freight Rate Prediction

A leakage-aware machine learning solution for predicting freight load rates from shipment, equipment, distance, weight, and geographic information.

The solution focuses on **temporal validation, careful feature engineering, reproducibility, and inference compatibility** rather than unnecessary model complexity.

---

## Problem

The objective is to predict the `posted_rate` for freight loads using the labeled development dataset provided with the assessment.

The final pipeline:

- Trains on the provided labeled development data
- Predicts rates for every load in the validation dataset
- Generates predictions for the fixed December 2025 scenario
- Produces the required prediction files and December prediction chart

---

## Approach

### 1. Data Understanding

The development data contains shipment-level information including:

- Pickup and delivery locations
- Pickup and delivery coordinates
- Distance
- Equipment type
- Weight
- Date
- Market and quote-related signals
- `posted_rate` as the prediction target

`load_id` is treated only as an identifier and is excluded from the model.

---

### 2. Temporal Validation

The data follows a chronological structure, with the development data covering January–October 2025 and the supplied validation data covering November–December 2025.

Because the final prediction setting is forward-looking, a random split was not used as the primary validation strategy.

The main model-selection split was:

```text
Training   → January–August 2025
Validation → September–October 2025
````

A second chronological split was also used as a robustness check:

```text
Training   → January–June 2025
Validation → July–August 2025
```

This was used to check whether model behavior remained consistent across different time periods.

---

## Feature Engineering

The final model uses a compact set of engineered features.

### Distance

Distance is the strongest observed predictor of freight rate.

Both the original distance and a quadratic distance term are used:

```text
distance
distance²
```

This allows the model to capture the observed nonlinear relationship between distance and rate.

### Weight

Missing weight values are imputed using the median calculated from the training data.

### Equipment

Equipment type is represented using indicator variables for:

* Reefer
* Flatbed

Dry Van is used as the reference category.

### Equipment Interactions

Equipment-specific effects are captured through:

```text
distance × Reefer
distance × Flatbed
weight × Reefer
weight × Flatbed
```

### Geographic Features

Pickup and delivery latitude/longitude are included as standardized geographic features.

For December inference, where coordinates are not supplied, coordinates are deterministically derived from the city-to-coordinate mappings learned from the development data.

---

## Leakage Prevention

Leakage was treated as a key consideration throughout development.

The solution follows these principles:

* `load_id` is excluded from modeling
* Validation target values are never used during feature engineering or model fitting
* Imputation statistics are calculated from training data
* Geographic scaling parameters are calculated from training data
* Route-based features were only evaluated using training-period information
* Features unavailable in the December inference data are not used in the final model
* Calendar features were tested but removed after degrading out-of-time validation performance

---

## Model Selection

Several approaches were evaluated through controlled experiments, including:

* Median baseline
* Distance-only linear regression
* Polynomial distance features
* Feature-engineered linear regression
* Gradient boosting
* Route-based features
* Extreme-rate modeling
* Log-target regression
* Calendar features
* Ridge regression

The final model is a **Linear Regression model with engineered features**.

The simpler model was retained because the tested nonlinear and more complex approaches did not provide sufficient improvement on the chronological holdout.

---

## Final Model

The final feature set contains:

```text
distance
distance²
imputed weight
reefer indicator
flatbed indicator
distance × reefer
distance × flatbed
weight × reefer
weight × flatbed
pickup latitude
pickup longitude
delivery latitude
delivery longitude
```

The final model is intentionally interpretable and easy to reproduce.

---

## Internal Validation

On the primary chronological holdout:

```text
Training   → January–August 2025
Validation → September–October 2025
```

the final feature set achieved:

| Metric |  Result |
| ------ | ------: |
| MAE    | $116.53 |
| RMSE   | $633.07 |
| R²     |  0.8279 |

These are **internal model-development metrics** and are not the official Spotter evaluation score.

---

## Error Analysis

The model performs substantially better on typical freight loads than on rare extreme-rate observations.

The error analysis showed that a small number of unusually high-rate loads contribute disproportionately to squared error.

Several approaches were tested specifically for these extreme observations, but they did not improve overall out-of-time performance enough to replace the final model.

This remains an important limitation of the current solution.

---

## December Scenario

The December input contains 31 rows representing the same shipment characteristics:

```text
Pickup    → Lexington
Delivery  → Fort Wayne
Distance  → 360 miles
Equipment → Dry Van
Weight    → 32,000
```

Only the date changes across December 1–31, 2025.

Calendar features were tested during model development but were removed because they degraded chronological validation performance.

Therefore, the final model produces the same prediction for all 31 December rows because the remaining predictive inputs are identical.

The generated December predictions are validated using the supplied scorer and used to produce the required prediction chart.

---

## Repository Structure

```text
spotter-ml-assessment/
│
├── train_model.py
├── score.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── experiments/
│   ├── baseline.py
│   ├── experiment_01.py
│   ├── experiment_02.py
│   ├── experiment_03.py
│   ├── experiment_04.py
│   ├── experiment_05.py
│   ├── experiment_06_tree_rich.py
│   ├── experiment_07_two_stage.py
│   ├── experiment_08_robustness.py
│   ├── experiment_09_calendar.py
│   ├── experiment_10_ridge.py
│   ├── experiment_log_target.py
│   └── experiment_route_features.py
│
└── analysis/
    ├── error_analysis.py
    ├── extreme_rate_analysis.py
    └── inference_audit.py
```

The assessment data is intentionally excluded from version control.

---

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Place the provided assessment data inside the `data/` directory.

Expected input files:

```text
data/
├── train-test.csv
├── validation.csv
├── december-chart-inputs.csv
└── validation-predictions-template.csv
```

---

## Run

### Train and Generate Predictions

```bash
python train_model.py
```

This generates:

```text
validation_predictions.csv
december_predictions.csv
```

### Validate Outputs

Run the supplied scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

The scorer validates the prediction files and generates:

```text
scorer_results/candidate_december.png
```

---

## Output Format

The final validation file contains exactly:

```text
load_id,predicted_rate
```

with one prediction for every validation load.

The December prediction file contains the required December input fields together with:

```text
predicted_rate
```

---

## Limitations

The main limitation is the treatment of rare extreme-rate observations. These loads are difficult to estimate accurately from the available predictors and have a disproportionate effect on RMSE.

The final model also intentionally avoids features that are unavailable in the December inference scenario.

The hidden evaluation metric used by Spotter is not available during development, so the reported metrics represent internal chronological validation only.

---

## Key Design Principles

```text
Temporal Validation
        ↓
Leakage Prevention
        ↓
Inference Compatibility
        ↓
Controlled Experiments
        ↓
Interpretable Features
        ↓
Reproducible Pipeline
```

The final solution prioritizes a model that is **measurable, explainable, reproducible, and compatible with the actual inference inputs**.

```

