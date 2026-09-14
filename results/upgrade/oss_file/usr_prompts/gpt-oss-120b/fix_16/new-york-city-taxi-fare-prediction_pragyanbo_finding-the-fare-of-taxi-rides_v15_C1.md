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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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
from pathlib import Path
import numpy as np
import pandas as pd


def find_file(name: str) -> str:
    """Search for *name* anywhere under the current directory and return the first match."""
    matches = list(Path(".").rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} in the current directory tree.")
    return str(matches[0])




## === cell 1
train_path = find_file("train.csv")
test_path = find_file("test.csv")

dtype_train = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
dtype_test = {
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
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
test_usecols = [c for c in usecols if c != "fare_amount"]

train_df = pd.read_csv(
    train_path,
    nrows=1_000_000,
    usecols=usecols,
    dtype=dtype_train,
    parse_dates=["pickup_datetime"],
)
test_df = pd.read_csv(
    test_path,
    usecols=test_usecols,
    dtype=dtype_test,
    parse_dates=["pickup_datetime"],
)



## === cell 2
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")



## === cell 3
train_df.dropna(inplace=True)
train_df = train_df[train_df["fare_amount"] > 0]

fare_cap = train_df["fare_amount"].quantile(0.995)
train_df = train_df[train_df["fare_amount"] <= fare_cap]


def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in miles (float32)."""
    lat1 = np.asarray(lat1, dtype=np.float32)
    lon1 = np.asarray(lon1, dtype=np.float32)
    lat2 = np.asarray(lat2, dtype=np.float32)
    lon2 = np.asarray(lon2, dtype=np.float32)

    rad = np.deg2rad
    dlat = rad(lat2 - lat1)
    dlon = rad(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(rad(lat1)) * np.cos(rad(lat2)) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_miles = np.float32(3958.8)
    return earth_radius_miles * c


train_df["distance"] = haversine_distance(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"],
)
test_df["distance"] = haversine_distance(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
)

train_df["log_distance"] = np.log1p(train_df["distance"]).astype(np.float32)
test_df["log_distance"] = np.log1p(test_df["distance"]).astype(np.float32)

train_df["distance_sq"] = train_df["distance"] ** 2
test_df["distance_sq"] = test_df["distance"] ** 2

dt_train = train_df["pickup_datetime"]
train_df["hour"] = dt_train.dt.hour
train_df["year"] = dt_train.dt.year
train_df["dayofweek"] = dt_train.dt.dayofweek
train_df["month"] = dt_train.dt.month
train_df["hour_sin"] = np.sin(2 * np.pi * train_df["hour"] / 24)
train_df["hour_cos"] = np.cos(2 * np.pi * train_df["hour"] / 24)

dt_test = test_df["pickup_datetime"]
test_df["hour"] = dt_test.dt.hour
test_df["year"] = dt_test.dt.year
test_df["dayofweek"] = dt_test.dt.dayofweek
test_df["month"] = dt_test.dt.month
test_df["hour_sin"] = np.sin(2 * np.pi * test_df["hour"] / 24)
test_df["hour_cos"] = np.cos(2 * np.pi * test_df["hour"] / 24)



## === cell 4
train_df = train_df[train_df["distance"] < 20]
train_df = train_df[
    (train_df["passenger_count"] != 0) & (train_df["passenger_count"] < 10)
]

feature_cols = [
    "distance",
    "distance_sq",
    "log_distance",
    "passenger_count",
    "hour",
    "hour_sin",
    "hour_cos",
    "year",
    "dayofweek",
    "month",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]

X = train_df[feature_cols].astype(np.float32)
y = train_df["fare_amount"].astype(np.float32)



## === cell 5
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)



## === cell 6
from sklearn.ensemble import RandomForestRegressor

X_train_np = X_train.values.astype(np.float32, copy=False)
y_train_np = y_train.values.astype(np.float32, copy=False)

r_reg = RandomForestRegressor(
    n_estimators=600,
    max_depth=None,
    n_jobs=8,
    random_state=42,
)
r_reg.fit(X_train_np, y_train_np)



## === cell 7
from sklearn.metrics import mean_squared_error

X_val_np = X_val.values.astype(np.float32, copy=False)
val_pred = r_reg.predict(X_val_np)
rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f"Validation RMSE: {rmse:.5f}")



## === cell 8
X_full_np = X.values.astype(np.float32, copy=False)
y_full_np = y.values.astype(np.float32, copy=False)

r_reg_full = RandomForestRegressor(
    n_estimators=600,
    max_depth=None,
    n_jobs=8,
    random_state=42,
)
r_reg_full.fit(X_full_np, y_full_np)

test_pred = r_reg_full.predict(test_df[feature_cols].astype(np.float32).values)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
