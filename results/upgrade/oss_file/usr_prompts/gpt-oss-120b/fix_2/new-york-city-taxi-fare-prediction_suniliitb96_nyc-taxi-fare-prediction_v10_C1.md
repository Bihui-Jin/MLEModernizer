# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.95939

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import timeit
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

try:
    import xgboost as xgb

    _XGB_AVAILABLE = True
except Exception:  # pragma: no cover
    _XGB_AVAILABLE = False


def haversine(coord1, coord2):
    """
    Calculate the great-circle distance between two (lat, lon) points.
    """
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0088 * c  # Earth radius in km


def locate_file(relative_path):
    """
    Return the first existing path among a set of common data directories.
    """
    candidates = [
        relative_path,
        os.path.join("data", relative_path),
        os.path.join("input", relative_path),
        os.path.join("..", "input", relative_path),
        os.path.join("..", relative_path),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"Unable to locate {relative_path}")




## === cell 1
MAX_TRAINING_SIZE = 1_000_00  # 100 000 rows for quick dev
EPS_IN_KM = 0.5
MIN_SAMPLES_CLUSTER = 500
RADIUS_VICINITY_AIRPORTS = 1.0
THERSHOLD_TRIP_FARE_RATE = 50.0
KMS_PER_RADIAN = 6371.0088

JFK_GEO_LOCATION = (40.6413, -73.7781)
LGR_GEO_LOCATION = (40.7769, -73.8740)
EWR_GEO_LOCATION = (40.6895, -74.1745)




## === cell 2
train_path = locate_file("train.csv")
test_path = locate_file("test.csv")
sample_sub_path = locate_file("sample_submission.csv")

df_train = pd.read_csv(
    train_path, nrows=MAX_TRAINING_SIZE, parse_dates=["pickup_datetime"]
)
df_test = pd.read_csv(test_path, parse_dates=["pickup_datetime"])

test_key = df_test["key"].copy()
df_train.drop(columns=["key"], inplace=True)
df_test.drop(columns=["key"], inplace=True)




## === cell 3
df_train = df_train[df_train["fare_amount"] >= 0]
df_train = df_train.dropna()
df_train = df_train[~df_train["passenger_count"].isin([0, 7, 8, 9, 10])]
BB = (-74.5, -72.8, 40.5, 41.8)


def within_bb(df):
    return (
        (df["pickup_longitude"] >= BB[0])
        & (df["pickup_longitude"] <= BB[1])
        & (df["pickup_latitude"] >= BB[2])
        & (df["pickup_latitude"] <= BB[3])
        & (df["dropoff_longitude"] >= BB[0])
        & (df["dropoff_longitude"] <= BB[1])
        & (df["dropoff_latitude"] >= BB[2])
        & (df["dropoff_latitude"] <= BB[3])
    )


df_train = df_train[within_bb(df_train)]
df_test = df_test[within_bb(df_test)]


def add_distance_features(df):
    df["trip_distance"] = df.apply(
        lambda row: haversine(
            (row["pickup_latitude"], row["pickup_longitude"]),
            (row["dropoff_latitude"], row["dropoff_longitude"]),
        ),
        axis=1,
    )
    return df


df_train = add_distance_features(df_train)
df_test = add_distance_features(df_test)

df_train = df_train[df_train["trip_distance"] < 25.0]


def add_datetime_features(df):
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day"] = df["pickup_datetime"].dt.day
    df["month"] = df["pickup_datetime"].dt.month
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["year"] = df["pickup_datetime"].dt.year
    return df


df_train = add_datetime_features(df_train)
df_test = add_datetime_features(df_test)

drop_cols = [
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
df_train = df_train.drop(columns=drop_cols)
df_test = df_test.drop(columns=drop_cols)




## === cell 4
y = df_train["fare_amount"]
X = df_train.drop(columns=["fare_amount"])

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.01, random_state=0)

if _XGB_AVAILABLE:
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        "max_depth": 8,
        "eta": 0.03,
        "subsample": 1,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "verbosity": 0,
    }
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        evals=[(dval, "val")],
        early_stopping_rounds=10,
        verbose_eval=False,
    )
else:
    model = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=8, random_state=0
    )
    model.fit(X_tr, y_tr)

if _XGB_AVAILABLE:
    preds_val = model.predict(dval)
else:
    preds_val = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, preds_val))
print(f"Validation RMSE: {val_rmse:.4f}")




## === cell 5
if _XGB_AVAILABLE:
    dtest = xgb.DMatrix(df_test)
    test_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
else:
    test_pred = model.predict(df_test)

submission = pd.DataFrame({"key": test_key, "fare_amount": np.round(test_pred, 2)})
submission_path = "taxi_fare_submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission.head())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/642372482.py in <cell line: 0>()
      2 if _XGB_AVAILABLE:
      3     dtest = xgb.DMatrix(df_test)
----> 4     test_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
      5 else:
      6     test_pred = model.predict(df_test)

AttributeError: 'Booster' object has no attribute 'best_ntree_limit'
