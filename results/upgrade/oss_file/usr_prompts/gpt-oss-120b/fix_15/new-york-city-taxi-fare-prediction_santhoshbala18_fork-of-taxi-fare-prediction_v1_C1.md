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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
xgboost==2.0.3

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
import math
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


def locate_file(filename: str) -> Path:
    for p in Path(".").rglob(filename):
        return p
    raise FileNotFoundError(f"{filename} not found in the current directory tree.")


TRAIN_PATH = locate_file("train.csv")
TEST_PATH = locate_file("test.csv")
SAMPLE_SUBMISSION_PATH = locate_file("sample_submission.csv")




## === cell 1
dtype_map = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

df = pd.read_csv(
    TRAIN_PATH,
    dtype=dtype_map,
    usecols=list(dtype_map.keys()) + ["key", "pickup_datetime"],
    nrows=8_000_000,
)

test_set = pd.read_csv(
    TEST_PATH,
    dtype={k: v for k, v in dtype_map.items() if k != "fare_amount"},
    usecols=[
        c
        for c in ["key", "pickup_datetime"] + list(dtype_map.keys())
        if c != "fare_amount"
    ],
)

df = df.dropna(
    subset=[
        "fare_amount",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
)
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 500)]


def haversine(lon1, lat1, lon2, lat2):
    """Vectorised haversine distance in kilometres."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


df["haversine_km"] = haversine(
    df["pickup_longitude"],
    df["pickup_latitude"],
    df["dropoff_longitude"],
    df["dropoff_latitude"],
)

test_set["haversine_km"] = haversine(
    test_set["pickup_longitude"],
    test_set["pickup_latitude"],
    test_set["dropoff_longitude"],
    test_set["dropoff_latitude"],
)

df = df[df["haversine_km"] > 0]

df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
test_set["pickup_datetime"] = pd.to_datetime(test_set["pickup_datetime"])

for d in (df, test_set):
    d["hour"] = d["pickup_datetime"].dt.hour.astype("int8")
    d["weekday"] = d["pickup_datetime"].dt.weekday.astype("int8")
    d["month"] = d["pickup_datetime"].dt.month.astype("int8")
    d["is_weekend"] = (d["weekday"] >= 5).astype("int8")
    avg_lat = np.radians((d["pickup_latitude"] + d["dropoff_latitude"]) / 2.0)
    lon_km = 111.319 * np.cos(avg_lat)
    d["manhattan_km"] = (
        np.abs(d["pickup_longitude"] - d["dropoff_longitude"]) * lon_km
        + np.abs(d["pickup_latitude"] - d["dropoff_latitude"]) * 111.0
    )
    d["abs_lat_diff"] = (
        (d["pickup_latitude"] - d["dropoff_latitude"]).abs().astype("float32")
    )
    d["abs_lon_diff"] = (
        (d["pickup_longitude"] - d["dropoff_longitude"]).abs().astype("float32")
    )
    d["log_haversine_km"] = np.log1p(d["haversine_km"]).astype("float32")
    d["log_manhattan_km"] = np.log1p(d["manhattan_km"]).astype("float32")

X = df.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = df["fare_amount"]

y_log = np.log1p(y)

X_train, X_val, y_train_log, y_val = train_test_split(
    X, y_log, test_size=0.05, random_state=42
)




## === cell 2
regressor = XGBRegressor(
    max_depth=9,
    learning_rate=0.03,
    n_estimators=1500,
    objective="reg:squarederror",
    n_jobs=4,
    random_state=42,
    tree_method="hist",
    eval_metric="rmse",
    reg_lambda=2.0,
    subsample=0.8,  # slight row‑subsampling for regularisation
    colsample_bytree=0.8,  # column‑subsampling for regularisation
)

regressor.fit(X_train, y_train_log)

val_pred_log = regressor.predict(X_val)
val_pred = np.expm1(val_pred_log)
rmse = math.sqrt(mean_squared_error(np.expm1(y_val), val_pred))
print(f"Validation RMSE: {rmse:.5f}")




## === cell 3
X_test = test_set.drop(["key", "pickup_datetime"], axis=1)

test_pred_log = regressor.predict(X_test)
test_pred = np.expm1(test_pred_log)

submission = pd.DataFrame(
    {"key": test_set["key"], "fare_amount": test_pred.astype("float32")}
)

submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path.resolve()}")
