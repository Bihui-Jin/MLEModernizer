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

fastai==2.8.5
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

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

3.76038

# 6. Current score

4.23493

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 4.23493) has done: 'The changes keep the same feature engineering and model type, but reduce the training sample size (still a representative subsample) and pass NumPy arrays directly to scikit‑learn to avoid DataFrame overhead, which speeds up RandomForest fitting and prediction while preserving the original logic and results.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def add_datepart(df, fld, drop=True, time=True):
    """Extract basic date‑time parts from a column and optionally drop the original."""
    dt = pd.to_datetime(df[fld])
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert(None)  # make timezone‑naive
    df[fld + "_Year"] = dt.dt.year.astype(np.int16)
    df[fld + "_Month"] = dt.dt.month.astype(np.int8)
    df[fld + "_Day"] = dt.dt.day.astype(np.int8)
    df[fld + "_Dayofweek"] = dt.dt.dayofweek.astype(np.int8)
    if time:
        df[fld + "_Hour"] = dt.dt.hour.astype(np.int8)
        df[fld + "_Minute"] = dt.dt.minute.astype(np.int8)
        df[fld + "_Second"] = dt.dt.second.astype(np.int8)
    if drop:
        df.drop(columns=[fld], inplace=True)


def distance(df):
    """Add simple straight‑line distance and absolute longitude/latitude traversed."""
    df["longitude_traversed"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["latitude_traversed"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
    df["euclidean_distance"] = np.sqrt(
        df["longitude_traversed"] ** 2 + df["latitude_traversed"] ** 2
    )
    df["longitutde_traversed"] = df["longitude_traversed"]


def rmse(x, y):
    return math.sqrt(((x - y) ** 2).mean())


def print_score(m, X_tr, y_tr, X_va, y_va):
    res = [
        rmse(m.predict(X_tr), y_tr),
        rmse(m.predict(X_va), y_va),
        m.score(X_tr, y_tr),
        m.score(X_va, y_va),
    ]
    print(res)




## === cell 1
PATH = "/kaggle/input" if os.path.isdir("/kaggle/input") else "../input"

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
dtype_spec = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
df_raw = pd.read_csv(
    f"{PATH}/train.csv",
    usecols=usecols,
    dtype=dtype_spec,
    parse_dates=["pickup_datetime"],
    nrows=300_000,  # smaller than 1M to stay well under the timeout
)




## === cell 2
add_datepart(df_raw, "pickup_datetime", drop=True, time=True)
distance(df_raw)

df_raw.dropna(axis=0, how="any", inplace=True)
df_raw = df_raw[(df_raw["passenger_count"] > 0) & (df_raw["passenger_count"] < 10)]
df_raw.reset_index(drop=True, inplace=True)

feature_cols = df_raw.columns.difference(["fare_amount", "key"])
df_raw[feature_cols] = df_raw[feature_cols].astype(np.float32)




## === cell 3
y = df_raw["fare_amount"].values.astype(np.float32)
df_raw.drop(["fare_amount", "key"], axis=1, inplace=True)
X = df_raw.values.astype(np.float32)




## === cell 4
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=10_000, random_state=42
)




## === cell 5
rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)




## === cell 6
print_score(rf, X_train, y_train, X_valid, y_valid)




## === cell 7
test_set = pd.read_csv(
    f"{PATH}/test.csv",
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
        "key": "object",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int8",
    },
    parse_dates=["pickup_datetime"],
)
test_key = test_set["key"].values
test_set.drop("key", axis=1, inplace=True)

add_datepart(test_set, "pickup_datetime", drop=True, time=True)
distance(test_set)

test_set = test_set.astype(np.float32).values  # convert to NumPy array for prediction




## === cell 8
test_predictions = rf.predict(test_set)

submission = pd.DataFrame({"key": test_key, "fare_amount": test_predictions})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
