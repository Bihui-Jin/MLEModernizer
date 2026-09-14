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

3.11

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
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
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

3.99076

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os
import gc
import datetime as dt

import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

RANDOM_STATE = 42

DATA_DIR = "/kaggle/input" if os.path.exists("/kaggle/input") else "/kaggle/data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = os.path.join(
        DATA_DIR, "new-york-city-taxi-fare-prediction", "train.csv"
    )
if not os.path.exists(TEST_PATH):
    TEST_PATH = os.path.join(DATA_DIR, "new-york-city-taxi-fare-prediction", "test.csv")

assert os.path.exists(TRAIN_PATH), f"train.csv not found at {TRAIN_PATH}"
assert os.path.exists(TEST_PATH), f"test.csv not found at {TEST_PATH}"



## === cell 1

N_TRAIN = 2_000_000  # bounded subset for speed + better score than tiny samples
DTYPE_MAP = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

n_total_guess = 55_423_856  # provided in prompt
keep_prob = min(1.0, N_TRAIN / n_total_guess)
rng = np.random.default_rng(RANDOM_STATE)


def _skiprow(i: int) -> bool:
    if i == 0:
        return False
    return rng.random() > keep_prob


train = pd.read_csv(
    TRAIN_PATH,
    usecols=usecols_train,
    dtype=DTYPE_MAP,
    parse_dates=["pickup_datetime"],
    skiprows=_skiprow,
)

test = pd.read_csv(
    TEST_PATH,
    usecols=usecols_test,
    dtype={k: v for k, v in DTYPE_MAP.items() if k != "fare_amount"},
    parse_dates=["pickup_datetime"],
)

print("Train shape (sampled):", train.shape)
print("Test shape:", test.shape)




## === cell 2
def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pickup_hour"] = out["pickup_datetime"].dt.hour.astype("int16")
    out["pickup_dayofweek"] = out["pickup_datetime"].dt.dayofweek.astype("int16")
    out["pickup_month"] = out["pickup_datetime"].dt.month.astype("int16")
    out["pickup_year"] = out["pickup_datetime"].dt.year.astype("int16")

    out["haversine_km"] = haversine_km(
        out["pickup_latitude"].values,
        out["pickup_longitude"].values,
        out["dropoff_latitude"].values,
        out["dropoff_longitude"].values,
    ).astype("float32")

    out["abs_lon_diff"] = np.abs(
        out["pickup_longitude"] - out["dropoff_longitude"]
    ).astype("float32")
    out["abs_lat_diff"] = np.abs(
        out["pickup_latitude"] - out["dropoff_latitude"]
    ).astype("float32")

    out["passenger_count"] = (
        out["passenger_count"].clip(lower=0, upper=8).astype("int16")
    )
    return out


def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out["fare_amount"].notna()]
    out = out[out["fare_amount"] > 0]
    out = out[out["fare_amount"] < 500]  # remove extreme outliers

    out = out[
        (out["pickup_longitude"].between(-74.5, -72.8))
        & (out["dropoff_longitude"].between(-74.5, -72.8))
        & (out["pickup_latitude"].between(40.5, 41.8))
        & (out["dropoff_latitude"].between(40.5, 41.8))
    ]

    out = out[
        ~(
            (out["pickup_longitude"] == out["dropoff_longitude"])
            & (out["pickup_latitude"] == out["dropoff_latitude"])
            & (out["fare_amount"] > 10)
        )
    ]
    return out


train = clean_train(train)
train_feat = add_features(train)
test_feat = add_features(test)

target = train_feat["fare_amount"].astype("float32")
train_keys = train_feat["key"]
test_keys = test_feat["key"]

drop_cols = ["key", "pickup_datetime", "fare_amount"]
X = train_feat.drop(columns=[c for c in drop_cols if c in train_feat.columns])
X_test = test_feat.drop(
    columns=[c for c in ["key", "pickup_datetime"] if c in test_feat.columns]
)

print("Features:", list(X.columns))
print("Cleaned train shape:", X.shape)

del train, train_feat
gc.collect()



## === cell 3
X_train, X_valid, y_train, y_valid = train_test_split(
    X, target, test_size=0.1, random_state=RANDOM_STATE
)

model = lgbm.LGBMRegressor(
    objective="regression",
    n_estimators=1200,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    verbose=200,
)

valid_pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, valid_pred, squared=False)
print("Local validation RMSE:", rmse)

del X_train, X_valid, y_train, y_valid, valid_pred
gc.collect()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3174318497.py in <cell line: 0>()
     18 )
     19 
---> 20 model.fit(
     21     X_train,
     22     y_train,

TypeError: LGBMRegressor.fit() got an unexpected keyword argument 'verbose'

## === cell 4
test_pred = model.predict(X_test)
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame(
    {"key": test_keys, "fare_amount": test_pred.astype("float32")}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/4163027324.py in <cell line: 0>()
      1 # Change rationale: ensure submission is valid (correct columns, aligned to test keys, .csv filename).
----> 2 test_pred = model.predict(X_test)
      3 # Minimal post-process: fares can't be negative; clipping avoids invalid outputs that hurt RMSE
      4 test_pred = np.clip(test_pred, 0, None)
      5 

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in predict(self, X, raw_score, start_iteration, num_iteration, pred_leaf, pred_contrib, validate_features, **kwargs)
   1104         """Docstring is set after definition, using a template."""
   1105         if not self.__sklearn_is_fitted__():
-> 1106             raise LGBMNotFittedError("Estimator not fitted, call fit before exploiting the model.")
   1107         if not isinstance(X, (pd_DataFrame, dt_DataTable)):
   1108             X = _LGBMValidateData(

NotFittedError: Estimator not fitted, call fit before exploiting the model.
