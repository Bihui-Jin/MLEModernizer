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

3.617832926497142

# 6. Current score

4.80409

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.97499) has done: 'Implemented fixes to load data without the missing feather module, corrected datetime parsing, ensured the dataframe df is defined before use, added necessary feature engineering (distance, coordinate differences, hour), trained a RandomForestRegressor with modest parameters on a sampled subset, evaluated RMSE on a validation split, and generated a proper submission CSV named submission.csv with the required columns key and fare_amount. All cells now run sequentially without errors.'
- What this solution (achieved 4.80242) has done: 'I add a few inexpensive feature engineering steps (year, month, weekday) and train the model on a log‑transformed target to better capture the skew of fare amounts. I also increase the number of trees and limit depth modestly, which should lower the validation RMSE toward the target without changing the overall model type or pipeline logic.'
- What this solution (achieved 4.80409) has done: 'The changes limit the depth of each decision tree in the RandomForest (setting `max_depth=20`) which dramatically cuts the amount of work per tree while keeping the same model type and training procedure. The rest of the pipeline—including feature engineering, log‑target handling, and prediction logic—remains unchanged, so the validation metric and final submission stay comparable. This reduction in computational cost prevents the 10‑minute timeout and allows the script to finish well within the 600‑second limit.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt




## === cell 1
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 6367 * 2 * np.arcsin(np.sqrt(a)) * 0.62137  # km to miles




## === cell 2
train_path = os.path.join(
    "..", "input", "new-york-city-taxi-fare-prediction", "train.csv"
)
usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_map = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int16,
}
df = pd.read_csv(
    train_path,
    nrows=500_000,
    usecols=usecols,
    dtype=dtype_map,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)

df = df[
    (df.passenger_count > 0)
    & (df.dropoff_latitude != 0)
    & (df.pickup_latitude != 0)
    & (df.dropoff_longitude != 0)
    & (df.pickup_longitude != 0)
    & (df.fare_amount > 2)
    & (df.fare_amount < 100)
]

df["hour"] = df["pickup_datetime"].dt.hour.astype(np.int8)
df["year"] = df["pickup_datetime"].dt.year.astype(np.int16)
df["month"] = df["pickup_datetime"].dt.month.astype(np.int8)
df["weekday"] = df["pickup_datetime"].dt.weekday.astype(np.int8)
df.drop(columns=["pickup_datetime"], inplace=True)

df["distance"] = haversine_np(
    df["pickup_longitude"],
    df["pickup_latitude"],
    df["dropoff_longitude"],
    df["dropoff_latitude"],
).astype(np.float32)

df["diff_lat"] = (
    (df["dropoff_latitude"] - df["pickup_latitude"]).abs().astype(np.float32)
)
df["diff_long"] = (
    (df["dropoff_longitude"] - df["pickup_longitude"]).abs().astype(np.float32)
)

df.dropna(inplace=True)




## === cell 3
feature_cols = [
    "diff_lat",
    "diff_long",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "distance",
    "passenger_count",
    "hour",
    "year",
    "month",
    "weekday",
]
X = df[feature_cols].astype(np.float32)
y = df["fare_amount"].astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_log = np.log1p(y_train).astype(np.float32)
y_val_log = np.log1p(y_val).astype(np.float32)

rf = RandomForestRegressor(
    n_estimators=800,
    max_depth=20,  # <<< added depth limit for performance
    max_features=0.8,
    n_jobs=-1,
    random_state=42,
    verbose=0,
)

rf.fit(X_train.values, y_train_log.values)

val_pred_log = rf.predict(X_val.values)
val_pred = np.expm1(val_pred_log)

rmse = np.sqrt(metrics.mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {rmse:.4f}")




## === cell 4
test_path = os.path.join(
    "..", "input", "new-york-city-taxi-fare-prediction", "test.csv"
)
test = pd.read_csv(
    test_path,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype={
        "pickup_longitude": np.float32,
        "pickup_latitude": np.float32,
        "dropoff_longitude": np.float32,
        "dropoff_latitude": np.float32,
        "passenger_count": np.int16,
    },
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)

test_id = test["key"].copy()  # preserve order for submission

test["hour"] = test["pickup_datetime"].dt.hour.astype(np.int8)
test["year"] = test["pickup_datetime"].dt.year.astype(np.int16)
test["month"] = test["pickup_datetime"].dt.month.astype(np.int8)
test["weekday"] = test["pickup_datetime"].dt.weekday.astype(np.int8)
test.drop(columns=["pickup_datetime"], inplace=True)

test["distance"] = haversine_np(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
).astype(np.float32)

test["diff_lat"] = (
    (test["dropoff_latitude"] - test["pickup_latitude"]).abs().astype(np.float32)
)
test["diff_long"] = (
    (test["dropoff_longitude"] - test["pickup_longitude"]).abs().astype(np.float32)
)

test_features = test[feature_cols].astype(np.float32)

test_pred_log = rf.predict(test_features.values)
test_pred = np.expm1(test_pred_log)

submission = pd.DataFrame({"key": test_id, "fare_amount": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
