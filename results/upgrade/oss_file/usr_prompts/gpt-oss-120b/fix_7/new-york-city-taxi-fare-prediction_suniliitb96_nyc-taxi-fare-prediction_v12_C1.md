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

3.93524

# 6. Current score

5.52833

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.52833) has done: 'I make the data‑path resolution robust by checking several common locations so the script can locate train.csv and test.csv in this environment. I also stop filtering the test set (bounds and trip‑distance cuts) because dropping rows changes the required submission size; keeping all test rows ensures a correctly‑shaped submission. These minimal changes let the notebook run end‑to‑end and write a valid taxi_fare_submission.csv while preserving the original modelling logic.'
- What this solution (achieved 5.52833) has done: 'I add a simple outlier filter to remove unusually high fares (keeping only fares ≤ 200) which reduces the impact of extreme values on RMSE, and I modestly increase the GradientBoostingRegressor complexity (more trees and a slightly higher learning rate) to let the model capture more patterns from the data. These minimal changes keep the original pipeline intact while aiming to lower the validation RMSE toward the target.'

# 9. Code solution

## === cell 0
import timeit
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import math
import os

KMS_PER_RADIAN = 6371.0088
JFK_GEO_LOCATION = (40.6413, -73.7781)
LGR_GEO_LOCATION = (40.7769, -73.8740)
EWR_GEO_LOCATION = (40.6895, -74.1745)

MAX_TRAINING_SIZE = 100_000  # keep a manageable subset
EPS_IN_KM = 0.5
MIN_SAMPLES_CLUSTER = 500
RADIUS_VICINITY_AIRPORTS = 1.0
THRESHOLD_TRIP_DISTANCE = 25.0
THRESHOLD_TRIP_FARE_RATE = 50.0


def resolve_path(filename):
    """
    Return an existing path for ``filename``.
    Tries the originally‑specified relative path and several fall‑backs
    (e.g. ./data/, /kaggle/input/). Raises FileNotFoundError if none exist.
    """
    candidates = [
        filename,
        os.path.join("data", os.path.basename(filename)),
        os.path.join("/kaggle/input", os.path.basename(filename)),
        os.path.join(".", os.path.basename(filename)),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Unable to locate {filename}")




## === cell 1
def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return KMS_PER_RADIAN * 2 * math.asin(math.sqrt(a))




## === cell 2
train_path = resolve_path("../input/train.csv")
test_path = resolve_path("../input/test.csv")

df_train = pd.read_csv(
    train_path, nrows=MAX_TRAINING_SIZE, parse_dates=["pickup_datetime"]
)
df_test = pd.read_csv(test_path, parse_dates=["pickup_datetime"])

test_key = df_test["key"].copy()
df_train.drop(columns=["key"], inplace=True)
df_test.drop(columns=["key"], inplace=True)




## === cell 3
df_train = df_train[df_train.fare_amount >= 0]
df_train = df_train.dropna()
df_train = df_train[(df_train.passenger_count > 0) & (df_train.passenger_count < 7)]

BB = (-74.5, -72.8, 40.5, 41.8)


def within_bounds(df):
    return (
        (df["pickup_longitude"] >= BB[0])
        & (df["pickup_longitude"] <= BB[1])
        & (df["pickup_latitude"] >= BB[2])
        & (df["pickup_latitude"] <= BB[3])
    )


df_train = df_train[within_bounds(df_train)]




## === cell 4
def add_trip_distance(df):
    df["trip_distance"] = df.apply(
        lambda row: haversine(
            (row["pickup_latitude"], row["pickup_longitude"]),
            (row["dropoff_latitude"], row["dropoff_longitude"]),
        ),
        axis=1,
    )
    return df


df_train = add_trip_distance(df_train)
df_test = add_trip_distance(df_test)

df_train = df_train[df_train.trip_distance < THRESHOLD_TRIP_DISTANCE]




## === cell 5
def add_datetime_features(df):
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day"] = df["pickup_datetime"].dt.day
    df["month"] = df["pickup_datetime"].dt.month
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["year"] = df["pickup_datetime"].dt.year
    return df


df_train = add_datetime_features(df_train)
df_test = add_datetime_features(df_test)




## === cell 6
def add_airport_features(df):
    for name, loc in zip(
        ["jfk", "lgr", "ewr"], [JFK_GEO_LOCATION, LGR_GEO_LOCATION, EWR_GEO_LOCATION]
    ):
        df[f"pickup_distance_to_{name}"] = df.apply(
            lambda row: haversine(
                (row["pickup_latitude"], row["pickup_longitude"]), loc
            ),
            axis=1,
        )
        df[f"drop_distance_to_{name}"] = df.apply(
            lambda row: haversine(
                (row["dropoff_latitude"], row["dropoff_longitude"]), loc
            ),
            axis=1,
        )
    return df


df_train = add_airport_features(df_train)
df_test = add_airport_features(df_test)




## === cell 7
df_train["trip_rate"] = df_train["fare_amount"] / df_train["trip_distance"].replace(
    0, 0.2
)
df_train = df_train[df_train.trip_rate < THRESHOLD_TRIP_FARE_RATE]

df_train = df_train[df_train.fare_amount <= 200]




## === cell 8
cols_to_drop_train = [
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "trip_rate",  # keep only for potential analysis; drop before modeling
]
df_train = df_train.drop(columns=cols_to_drop_train)

cols_to_drop_test = [c for c in cols_to_drop_train if c != "trip_rate"]
df_test = df_test.drop(columns=cols_to_drop_test)

test_key = test_key.loc[df_test.index].reset_index(drop=True)
df_test = df_test.reset_index(drop=True)




## === cell 9
y = df_train["fare_amount"]
X = df_train.drop(columns=["fare_amount"])

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=42)




## === cell 10
try:
    import xgboost as xgb

    xgboost_available = True
except Exception:
    xgboost_available = False

if xgboost_available:
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)

    params = {
        "max_depth": 8,
        "eta": 0.03,
        "subsample": 1,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "verbosity": 0,
    }

    evals = [(dval, "validation")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        early_stopping_rounds=10,
        evals=evals,
        verbose_eval=False,
    )
else:
    model = GradientBoostingRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=8, random_state=42
    )
    model.fit(x_train, y_train)




## === cell 11
if xgboost_available:
    preds_val = model.predict(dval)
else:
    preds_val = model.predict(x_val)

val_rmse = mean_squared_error(y_val, preds_val, squared=False)
print(f"Validation RMSE: {val_rmse:.4f}")




## === cell 12
if xgboost_available:
    dtest = xgb.DMatrix(df_test)
    test_pred = model.predict(dtest)
else:
    test_pred = model.predict(df_test)




## === cell 13
submission = pd.DataFrame({"key": test_key, "fare_amount": np.round(test_pred, 2)})
output_path = "taxi_fare_submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
print(submission.head())
