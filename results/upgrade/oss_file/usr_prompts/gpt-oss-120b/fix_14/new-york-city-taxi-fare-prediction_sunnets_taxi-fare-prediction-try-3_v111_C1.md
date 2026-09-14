# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict the fare amount for a taxi ride given the pickup and dropoff locations.

## Metric
Root mean-squared error.

## Submission Format
For each `key` in the test set, you must predict a value for the `fare_amount` variable. The file should contain a header and have the following format:

```
key,fare_amount
2015-01-27 13:08:24.0000002,11.00
2015-02-27 13:08:24.0000002,12.05
2015-03-27 13:08:24.0000002,11.23
2015-04-27 13:08:24.0000002,14.17
2015-05-27 13:08:24.0000002,15.12
etc
```

## Dataset
- **train.csv** - Input features and target `fare_amount` values for the training set (about 55M rows).
- **test.csv** - Input features for the test set (about 10K rows). Your goal is to predict `fare_amount` for each row.
- **sample_submission.csv** - a sample submission file in the correct format (columns `key` and `fare_amount`). This file 'predicts' `fare_amount` to be $`11.35` for all rows, which is the mean `fare_amount` from the training set.

### Data fields
**ID**

- **key** - Unique `string` identifying each row in both the training and test sets. Comprised of **pickup_datetime** plus a unique integer, but this doesn't matter, it should just be used as a unique ID field.Required in your submission CSV. Not necessarily needed in the training set, but could be useful to simulate a 'submission file' while doing cross-validation within the training set.

**Features**

- **pickup_datetime** - `timestamp` value indicating when the taxi ride started.
- **pickup_longitude** - `float` for longitude coordinate of where the taxi ride started.
- **pickup_latitude** - `float` for latitude coordinate of where the taxi ride started.
- **dropoff_longitude** - `float` for longitude coordinate of where the taxi ride ended.
- **dropoff_latitude** - `float` for latitude coordinate of where the taxi ride ended.
- **passenger_count** - `integer` indicating the number of passengers in the taxi ride.

**Target**

- **fare_amount** - `float` dollar amount of the cost of the taxi ride. This value is only in the training set; this is what you are predicting in the test set and it is required in your submission CSV.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        input/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
```

-> data/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Target score

4.210338263743533

# 6. Current score

4.8016

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.5943) has done: 'The script is optimized by cutting the GradientBoostingRegressor’s tree count in half (from 200 to 100), which roughly halves training time while keeping the same model type, loss, and depth, preserving correctness. No other logic or data handling is altered.'
- What this solution (achieved 5.57545) has done: 'I add a few inexpensive features that often help taxi‑fare models—weekday, Euclidean distance, and keep the existing Manhattan distance. Then I restore the GradientBoostingRegressor to a slightly larger ensemble (150 trees) while keeping the other hyper‑parameters unchanged. These changes keep the original pipeline intact, only extend feature engineering and modestly increase model capacity, which should lower the validation RMSE toward the target value.'
- What this solution (achieved 5.91636) has done: 'The update keeps the exact data‑preprocessing and feature engineering, but replaces the standard `GradientBoostingRegressor` with the histogram‑based `HistGradientBoostingRegressor`, which is much faster on large numeric datasets while preserving the same gradient‑boosting logic and using identical hyper‑parameters (trees, depth, learning rate).  The rest of the pipeline—including cleaning, scaling, log‑target transformation, validation, and submission generation—remains unchanged, so the predictions stay equivalent up to negligible floating‑point differences and the script now finishes well within the 600‑second limit.'
- What this solution (achieved 4.80967) has done: 'The fix adds missing imports, corrects the dtype typo, implements the required preprocessing utilities (clean, time/coordinate/distance feature generators), defines a simple `output_submission` helper, and ensures the pipeline runs end‑to‑end producing a valid CSV file. These changes resolve the NameError failures and enable a functional model while keeping the original architecture and feature set intact.'
- What this solution (achieved 4.83442) has done: 'I add a couple of cheap distance‑difference features (absolute latitude and longitude deltas) that often help taxi‑fare models, and I slightly increase the number of boosting iterations while lowering the learning rate to let the model fit those new signals better. These changes keep the same overall pipeline and model type, but give the validation RMSE a modest improvement, moving it closer to the target value.'
- What this solution (achieved 4.75813) has done: 'I keep the overall model and training approach unchanged, but add a few inexpensive time‑based features (day of month and weekend flag) and a log‑transform of the haversine distance, drop the unnecessary MinMax scaling (tree‑based models work well on raw values), and slightly boost model capacity (more trees, a bit deeper, smaller learning rate). These tweaks are lightweight, preserve the original pipeline, and are aimed at lowering the RMSE toward the target.'
- What this solution (achieved 4.77271) has done: 'I add cheap cyclic hour features (sin / cos) to give the model a better sense of daily patterns, and slightly increase model capacity by using more trees, a deeper depth, and a smaller learning rate. These adjustments keep the original pipeline and model type intact while aiming to lower the validation RMSE closer to the target.'
- What this solution (achieved 4.8016) has done: 'I add inexpensive Euclidean‑distance features (and a log‑scaled version) to give the model extra geographic signal, and slightly increase the tree capacity while using a smaller learning rate. These tweaks keep the original pipeline and model type unchanged, but are expected to lower the validation RMSE, moving the score closer to the target.'

# 9. Code solution

## === cell 0
TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

DATASET_SIZE = 800000  # ~0.8 M rows fits comfortably in memory




## === cell 1
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import HistGradientBoostingRegressor

dtype_spec = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop rows with missing values and non‑positive fares."""
    df = df.dropna()
    df = df[df["fare_amount"] > 0]
    return df.reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, weekday and month from pickup_datetime."""
    dt = pd.to_datetime(df["pickup_datetime"])
    df["pickup_hour"] = dt.dt.hour.astype(np.int8)
    df["pickup_weekday"] = dt.dt.weekday.astype(np.int8)
    df["pickup_month"] = dt.dt.month.astype(np.int8)
    return df


def add_extra_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Additional cheap time features: day of month and weekend flag."""
    dt = pd.to_datetime(df["pickup_datetime"])
    df["pickup_day"] = dt.dt.day.astype(np.int8)
    df["is_weekend"] = (dt.dt.weekday >= 5).astype(np.int8)
    return df


def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic representation of hour to capture daily periodicity."""
    hour = df["pickup_hour"].astype(np.float32)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    return df


def add_coordinate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder – keep existing coordinates unchanged."""
    return df


def add_distances_features(df: pd.DataFrame) -> pd.DataFrame:
    """Manhattan distance (in degrees) between pickup and dropoff."""
    df["manhattan"] = (
        (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
        + (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    ).astype(np.float32)
    return df


def add_haversine(df: pd.DataFrame) -> pd.DataFrame:
    """Great‑circle (haversine) distance in km."""
    R = 6371.0  # Earth radius in km
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    dlat = lat2 - lat1
    dlon = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    df["haversine"] = (R * c).astype(np.float32)
    return df


def add_euclidean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Euclidean distance (degrees) and its log‑scaled version."""
    lat_diff = (df["pickup_latitude"] - df["dropoff_latitude"]).astype(np.float32)
    lon_diff = (df["pickup_longitude"] - df["dropoff_longitude"]).astype(np.float32)
    df["euclidean"] = np.sqrt(lat_diff**2 + lon_diff**2).astype(np.float32)
    df["log_euclidean"] = np.log1p(df["euclidean"]).astype(np.float32)
    return df


def add_log_haversine(df: pd.DataFrame) -> pd.DataFrame:
    """Log‑scaled haversine distance – often easier for trees to split."""
    df["log_haversine"] = np.log1p(df["haversine"]).astype(np.float32)
    return df


def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Absolute latitude and longitude differences – cheap extra signals."""
    df["abs_lat_diff"] = (
        (df["pickup_latitude"] - df["dropoff_latitude"]).abs().astype(np.float32)
    )
    df["abs_lon_diff"] = (
        (df["pickup_longitude"] - df["dropoff_longitude"]).abs().astype(np.float32)
    )
    return df


def output_submission(
    raw_test: pd.DataFrame,
    prediction: np.ndarray,
    id_column: str,
    prediction_column: str,
    file_name: str,
):
    """Write submission file with required columns and .csv suffix."""
    sub = raw_test[[id_column]].copy()
    sub[prediction_column] = prediction
    if not file_name.lower().endswith(".csv"):
        file_name = f"{file_name}.csv"
    sub.to_csv(file_name, index=False)
    print(f"Submission saved to {file_name}")


train_df = pd.read_csv(TRAIN_PATH, nrows=DATASET_SIZE, dtype=dtype_spec)
test_df = pd.read_csv(TEST_PATH, dtype=dtype_spec)

print(f"Loaded train rows: {len(train_df)}, test rows: {len(test_df)}")

train_df = clean(train_df)

train_df = add_time_features(train_df)
test_df = add_time_features(test_df)

train_df = add_extra_time_features(train_df)
test_df = add_extra_time_features(test_df)

train_df = add_cyclic_time_features(train_df)
test_df = add_cyclic_time_features(test_df)

train_df = add_coordinate_features(train_df)
test_df = add_coordinate_features(test_df)

train_df = add_distances_features(train_df)
test_df = add_distances_features(test_df)

train_df = add_haversine(train_df)
test_df = add_haversine(test_df)

train_df = add_euclidean_features(train_df)  # new feature
test_df = add_euclidean_features(test_df)  # new feature

train_df = add_log_haversine(train_df)
test_df = add_log_haversine(test_df)

train_df = add_diff_features(train_df)
test_df = add_diff_features(test_df)

train_df = train_df.drop(columns=["pickup_datetime"])
test_df = test_df.drop(columns=["pickup_datetime"])




## === cell 2
target_col = "fare_amount"
y = train_df[target_col].values
X = train_df.drop(columns=[target_col, "key"]).values
test_keys = test_df["key"].values
X_test = test_df.drop(columns=["key"]).values

X = X.astype(np.float32)
X_test = X_test.astype(np.float32)




## === cell 3
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

model = HistGradientBoostingRegressor(
    max_iter=1000,  # a bit more trees for capacity
    learning_rate=0.015,  # smaller steps for finer fitting
    max_depth=12,  # slightly deeper trees
    random_state=42,
)

model.fit(X_train, y_train_log)

val_pred = np.expm1(model.predict(X_val))
val_rmse = np.sqrt(((val_pred - y_val) ** 2).mean())
print(f"Validation RMSE: {val_rmse:.4f}")




## === cell 4
test_pred = np.expm1(model.predict(X_test))

output_submission(
    raw_test=pd.DataFrame({"key": test_keys}),
    prediction=test_pred,
    id_column="key",
    prediction_column="fare_amount",
    file_name=SUBMISSION_NAME,
)
