# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

np.random.seed(42)




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
    nrows=800_000,
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
df["is_weekend"] = (df["weekday"] >= 5).astype(np.int8)

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7).astype(np.float32)
df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7).astype(np.float32)

df.drop(columns=["pickup_datetime"], inplace=True)

df["distance"] = haversine_np(
    df["pickup_longitude"],
    df["pickup_latitude"],
    df["dropoff_longitude"],
    df["dropoff_latitude"],
).astype(np.float32)

df["distance_sq"] = (df["distance"] ** 2).astype(np.float32)
df["log_distance"] = np.log1p(df["distance"]).astype(np.float32)

df["diff_lat"] = (
    (df["dropoff_latitude"] - df["pickup_latitude"]).abs().astype(np.float32)
)
df["diff_long"] = (
    (df["dropoff_longitude"] - df["pickup_longitude"]).abs().astype(np.float32)
)

df["total_diff"] = (df["diff_lat"] + df["diff_long"]).astype(np.float32)

df.dropna(inplace=True)




## === cell 3
feature_cols = [
    "diff_lat",
    "diff_long",
    "total_diff",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "distance",
    "distance_sq",
    "log_distance",  # new feature
    "passenger_count",
    "hour",
    "hour_sin",  # new cyclical features
    "hour_cos",
    "year",
    "month",
    "month_sin",
    "month_cos",
    "weekday",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
]
X = df[feature_cols].astype(np.float32)
y = df["fare_amount"].astype(np.float32)

X_np = X.to_numpy()
y_np = y.to_numpy()

X_train, X_val, y_train, y_val = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42
)

y_train_log = np.log1p(y_train).astype(np.float32)
y_val_log = np.log1p(y_val).astype(np.float32)

rf = RandomForestRegressor(
    n_estimators=400,  # more trees for better accuracy
    max_depth=30,
    max_features=0.8,
    max_samples=0.9,  # use more data per tree
    n_jobs=-1,
    random_state=42,
    verbose=0,
)

rf.fit(X_train, y_train_log)

val_pred_log = rf.predict(X_val)
val_pred = np.expm1(val_pred_log)

rmse = np.sqrt(metrics.mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {rmse:.4f}")

val_bias = np.mean(val_pred - y_val)
print(f"Validation bias (to be subtracted from test predictions): {val_bias:.6f}")




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
test["is_weekend"] = (test["weekday"] >= 5).astype(np.int8)

test["hour_sin"] = np.sin(2 * np.pi * test["hour"] / 24).astype(np.float32)
test["hour_cos"] = np.cos(2 * np.pi * test["hour"] / 24).astype(np.float32)
test["month_sin"] = np.sin(2 * np.pi * test["month"] / 12).astype(np.float32)
test["month_cos"] = np.cos(2 * np.pi * test["month"] / 12).astype(np.float32)
test["weekday_sin"] = np.sin(2 * np.pi * test["weekday"] / 7).astype(np.float32)
test["weekday_cos"] = np.cos(2 * np.pi * test["weekday"] / 7).astype(np.float32)

test.drop(columns=["pickup_datetime"], inplace=True)

test["distance"] = haversine_np(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
).astype(np.float32)

test["distance_sq"] = (test["distance"] ** 2).astype(np.float32)
test["log_distance"] = np.log1p(test["distance"]).astype(np.float32)

test["diff_lat"] = (
    (test["dropoff_latitude"] - test["pickup_latitude"]).abs().astype(np.float32)
)
test["diff_long"] = (
    (test["dropoff_longitude"] - test["pickup_longitude"]).abs().astype(np.float32)
)

test["total_diff"] = (test["diff_lat"] + test["diff_long"]).astype(np.float32)

test_features = test[feature_cols].astype(np.float32).to_numpy()

test_pred_log = rf.predict(test_features)
test_pred = np.expm1(test_pred_log)

test_pred_corrected = np.clip(test_pred - val_bias, a_min=0.0, a_max=None)

submission = pd.DataFrame({"key": test_id, "fare_amount": test_pred_corrected})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
