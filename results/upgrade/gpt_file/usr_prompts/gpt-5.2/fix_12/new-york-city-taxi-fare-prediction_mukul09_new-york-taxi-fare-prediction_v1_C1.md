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
import time
import itertools
import numpy as np
import pandas as pd

from xgboost import XGBRegressor, DMatrix
import xgboost as xgb
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings("ignore")

np.random.seed(0)



## === cell 1
TRAIN_PATH = "../input/train.csv"
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
dtypes_train = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

TARGET_N = 500_000
STRIDE = 110  # empirically ~55M/110 ~= 500k


def _skiprows():
    kept = 0
    i = 1
    while True:
        if (i - 1) % STRIDE == 0 and kept < TARGET_N:
            kept += 1
        else:
            yield i
        i += 1


train_data = pd.read_csv(
    TRAIN_PATH,
    usecols=usecols_train,
    dtype=dtypes_train,
    skiprows=_skiprows(),
    engine="c",
)
if len(train_data) > TARGET_N:
    train_data = train_data.iloc[:TARGET_N].copy()



## === cell 2
TEST_PATH = "../input/test.csv"
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtypes_test = {
    "key": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
test_data = pd.read_csv(
    TEST_PATH,
    usecols=usecols_test,
    dtype=dtypes_test,
    engine="c",
)



## === cell 3
pass



## === cell 4
train_data.dropna(axis=0, inplace=True)



## === cell 5
pass



## === cell 6
m = (train_data["fare_amount"] > 0) & (train_data["fare_amount"] <= 250)
m &= (train_data["passenger_count"] <= 6) & (train_data["passenger_count"] > 0)
m &= (
    train_data["pickup_latitude"].between(40.5, 41.0)
    & train_data["dropoff_latitude"].between(40.5, 41.0)
    & train_data["pickup_longitude"].between(-74.3, -73.6)
    & train_data["dropoff_longitude"].between(-74.3, -73.6)
)
train_data = train_data.loc[m].copy()



## === cell 7
pass




## === cell 8
def distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return distance along great radius between pickup and dropoff coordinates.
    """
    R_earth = 6371.0
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * R_earth * np.arcsin(np.sqrt(a))


def date_time_info(data):
    dt_series = pd.to_datetime(data["pickup_datetime"], utc=True, errors="coerce")
    data["pickup_datetime"] = dt_series
    dt = dt_series.dt
    data["hour"] = dt.hour.astype("int16")
    data["day"] = dt.day.astype("int16")
    data["month"] = dt.month.astype("int16")
    data["weekday"] = dt.weekday.astype("int16")
    data["year"] = dt.year.astype("int16")
    return data


train_data = date_time_info(train_data)

p_lat = train_data["pickup_latitude"].to_numpy()
p_lon = train_data["pickup_longitude"].to_numpy()
d_lat = train_data["dropoff_latitude"].to_numpy()
d_lon = train_data["dropoff_longitude"].to_numpy()
train_data["distance"] = distance(p_lat, p_lon, d_lat, d_lon).astype("float32")



## === cell 9
train_data.dropna(subset=["pickup_datetime"], inplace=True)

dist_np = train_data["distance"].to_numpy()
m = np.isfinite(dist_np) & (dist_np > 0) & (dist_np <= 100)
train_data = train_data.loc[m].copy()

train_data.drop(["key", "pickup_datetime"], axis=1, inplace=True)



## === cell 10
pass



## === cell 11
test_data = date_time_info(test_data)

p_lat_t = test_data["pickup_latitude"].to_numpy()
p_lon_t = test_data["pickup_longitude"].to_numpy()
d_lat_t = test_data["dropoff_latitude"].to_numpy()
d_lon_t = test_data["dropoff_longitude"].to_numpy()
test_data["distance"] = distance(p_lat_t, p_lon_t, d_lat_t, d_lon_t).astype("float32")

test_key_all = test_data["key"].copy()
x_pred_all = test_data.drop(columns=["key", "pickup_datetime"])



## === cell 12
y = np.log1p(train_data["fare_amount"].astype(float)).to_numpy()
X = train_data.drop(["fare_amount"], axis=1)

X_np = np.ascontiguousarray(X.to_numpy(dtype=np.float32, copy=False))
x_pred_np = np.ascontiguousarray(x_pred_all.to_numpy(dtype=np.float32, copy=False))

cv_split = KFold(n_splits=5, shuffle=True, random_state=0)




## === cell 13
def rmse_in_dollars_from_log(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0.0)
    return np.sqrt(mean_squared_error(y_true, y_pred))




## === cell 14
pass



## === cell 15
params = {
    "max_depth": [7, 8],
    "learning_rate": [0.03],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 5],
    "reg_lambda": [1.0],
    "n_estimators": [1500, 2000],
    "objective": ["reg:squarederror"],
    "verbosity": [0],
}

param_list = list(ParameterGrid(params))

splits = [(tr_idx, va_idx) for tr_idx, va_idx in cv_split.split(X_np, y)]

dtrain = xgb.DMatrix(X_np, label=y)
folds = splits  # xgb.cv accepts list of (train_idx, test_idx)

best_score = -np.inf  # greater is better for our negative RMSE scorer
best_params = None

t0 = time.time()

for p in param_list:
    xgb_params = {
        "max_depth": int(p["max_depth"]),
        "eta": float(p["learning_rate"]),
        "subsample": float(p["subsample"]),
        "colsample_bytree": float(p["colsample_bytree"]),
        "min_child_weight": float(p["min_child_weight"]),
        "lambda": float(p["reg_lambda"]),
        "objective": p["objective"],
        "verbosity": int(p["verbosity"]),
        "eval_metric": "rmse",
        "seed": 0,
        "nthread": -1,
    }
    num_boost_round = int(p["n_estimators"])

    cv_res = xgb.cv(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        folds=folds,
        metrics=("rmse",),
        seed=0,
        shuffle=False,  # folds already define the split
        verbose_eval=False,
        as_pandas=True,
    )

    rmse_log = float(cv_res["test-rmse-mean"].iloc[-1])

    fold_scores = []
    for tr_idx, va_idx in folds:
        dtr = dtrain.slice(tr_idx)
        dva = dtrain.slice(va_idx)

        booster = xgb.train(
            params=xgb_params,
            dtrain=dtr,
            num_boost_round=num_boost_round,
            verbose_eval=False,
        )
        pred_va = booster.predict(dva)
        score = -rmse_in_dollars_from_log(y[va_idx], pred_va)
        fold_scores.append(score)

    mean_score = float(np.mean(fold_scores))
    if mean_score > best_score:
        best_score = mean_score
        best_params = p

best_xgb_params = {
    "max_depth": int(best_params["max_depth"]),
    "eta": float(best_params["learning_rate"]),
    "subsample": float(best_params["subsample"]),
    "colsample_bytree": float(best_params["colsample_bytree"]),
    "min_child_weight": float(best_params["min_child_weight"]),
    "lambda": float(best_params["reg_lambda"]),
    "objective": best_params["objective"],
    "verbosity": int(best_params["verbosity"]),
    "eval_metric": "rmse",
    "seed": 0,
    "nthread": -1,
}
best_num_boost_round = int(best_params["n_estimators"])

booster_final = xgb.train(
    params=best_xgb_params,
    dtrain=dtrain,
    num_boost_round=best_num_boost_round,
    verbose_eval=False,
)

dtest = xgb.DMatrix(x_pred_np)
pred_log_all = booster_final.predict(dtest)

prediction_all = np.expm1(pred_log_all)
prediction_all = np.maximum(prediction_all, 0.0)

fallback_fare = float(np.expm1(np.nanmedian(y)))
prediction_all = np.where(np.isfinite(prediction_all), prediction_all, fallback_fare)

submission = pd.DataFrame({"key": test_key_all, "fare_amount": prediction_all})
submission.to_csv("taxi_fare_submission.csv", index=False)



## === cell 16
pass
