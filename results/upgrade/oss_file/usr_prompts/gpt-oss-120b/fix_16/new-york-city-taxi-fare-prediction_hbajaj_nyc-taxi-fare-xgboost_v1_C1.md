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
import time
import numpy as np
import pandas as pd
from functools import wraps
from pathlib import Path
import os
import joblib
import gc

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

_USE_XGB = False


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print(
            "***************************Total time running << %s >>: %s seconds"
            % (function.__name__, str(np.round((t1 - t0), 2)))
        )
        return result

    return function_timer


@fn_timer
def read_file_to_df(fileName, rows=0):
    """
    Reads CSV with Feather caching. If rows>0 a limited read is performed,
    otherwise the whole file is loaded.
    """
    start_time = time.time()
    traintypes = {
        "key": "object",  # ensure key is loaded
        "fare_amount": "float32",
        "pickup_datetime": "str",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    }

    cols = list(traintypes.keys())
    feather_path = fileName + ".feather"
    if Path(feather_path).is_file():
        df = pd.read_feather(feather_path)
    else:
        if rows > 0:
            df = pd.read_csv(fileName, nrows=rows, usecols=cols, dtype=traintypes)
        else:
            df = pd.read_csv(fileName, usecols=cols, dtype=traintypes)
        df.to_feather(feather_path)
    print("--- Read Files: %s secs ---" % np.round((time.time() - start_time), 2))
    return df


@fn_timer
def read_test_file_to_df(fileName):
    """
    Reads the test CSV with Feather caching to avoid repeated CSV parsing.
    """
    start_time = time.time()
    feather_path = fileName + ".feather"
    if Path(feather_path).is_file():
        df = pd.read_feather(feather_path)
    else:
        df = pd.read_csv(fileName)
        df.to_feather(feather_path)
    print("--- Read test file: %s secs ---" % np.round((time.time() - start_time), 2))
    return df


def drop_rows_with_nan(df):
    print("Before dropna")
    df.dropna(how="any", axis="rows", inplace=True)
    print(df.shape)


@fn_timer
def data_cleanup(tmp_df):
    print("Before clearing outliers")
    tmp_df = tmp_df[tmp_df["fare_amount"] > 0]
    tmp_df = tmp_df[tmp_df["pickup_longitude"] < -72]
    tmp_df = tmp_df[(tmp_df["pickup_latitude"] > 40) & (tmp_df["pickup_latitude"] < 44)]
    tmp_df = tmp_df[tmp_df["dropoff_longitude"] < -72]
    tmp_df = tmp_df[
        (tmp_df["dropoff_latitude"] > 40) & (tmp_df["dropoff_latitude"] < 44)
    ]
    tmp_df = tmp_df[(tmp_df["passenger_count"] > 0) & (tmp_df["passenger_count"] < 10)]
    print(tmp_df.shape)
    return tmp_df


def reformat_pickup_datetime(df):
    df["pickup_datetime"] = df["pickup_datetime"].str.slice(0, 13)
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], utc=True, format="%Y-%m-%d %H"
    )
    return df


def add_day_of_week_feature(tmp_df):
    tmp_df["dayOfWeek"] = tmp_df["pickup_datetime"].dt.dayofweek.astype("uint8")


def add_time_of_day_feature(tmp_df):
    tmp_df["timeOfDay"] = tmp_df["pickup_datetime"].dt.hour.astype("uint8")


def add_month_feature(tmp_df):
    tmp_df["month"] = tmp_df["pickup_datetime"].dt.month.astype("uint8")


def add_week_of_year_feature(tmp_df):
    tmp_df["weekOfYear"] = (
        tmp_df["pickup_datetime"].dt.isocalendar().week.astype("uint8")
    )


@fn_timer
def add_distance_feature(tmp_df):
    """
    Vectorized Haversine distance (in km) between pickup and dropoff points.
    Mirrors the previous per‑row implementation output precision.
    """
    start_time = time.time()
    R = 6373.0  # Earth radius in km

    lat1 = np.radians(tmp_df["pickup_latitude"].astype("float64"))
    lon1 = np.radians(tmp_df["pickup_longitude"].astype("float64"))
    lat2 = np.radians(tmp_df["dropoff_latitude"].astype("float64"))
    lon2 = np.radians(tmp_df["dropoff_longitude"].astype("float64"))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    tmp_df["distance"] = distance.astype("float32")
    print("--- Add distance: %s secs ---" % np.round((time.time() - start_time), 2))


@fn_timer
def add_coord_diff_feature(tmp_df):
    """
    Simple absolute latitude and longitude differences.
    These often help linear or tree models in addition to haversine distance.
    """
    start_time = time.time()
    tmp_df["lat_diff"] = (
        (tmp_df["dropoff_latitude"] - tmp_df["pickup_latitude"]).abs().astype("float32")
    )
    tmp_df["lon_diff"] = (
        (tmp_df["dropoff_longitude"] - tmp_df["pickup_longitude"])
        .abs()
        .astype("float32")
    )
    print("--- Add coord diff: %s secs ---" % np.round((time.time() - start_time), 2))


@fn_timer
def filter_based_on_distance(df, distance):
    df = df[(df["distance"] < distance)]
    return df


@fn_timer
def data_one_hot_encoding(tmp_df):
    """
    One‑hot encode timeOfDay, dayOfWeek, and month in a single call.
    Uses uint8 dtype to keep memory low.
    """
    print("Before one hot encoding")
    start_time = time.time()
    cat_cols = ["timeOfDay", "dayOfWeek", "month"]
    existing_cols = [c for c in cat_cols if c in tmp_df.columns]
    if not existing_cols:
        print("No categorical columns to encode; skipping one‑hot.")
        print(
            "--- One hot encoding: %s secs ---"
            % np.round((time.time() - start_time), 2)
        )
        return tmp_df

    dummies = pd.get_dummies(tmp_df[existing_cols], dtype=np.uint8)
    dummies = dummies.rename(columns=lambda x: f"{x}")

    tmp_df = pd.concat([tmp_df.drop(existing_cols, axis=1), dummies], axis=1)
    print(tmp_df.columns)
    print("--- One hot encoding: %s secs ---" % np.round((time.time() - start_time), 2))
    return tmp_df


@fn_timer
def prepare_data_split(tmp_df):
    drop_cols = ["pickup_datetime", "fare_amount"]
    if "key" in tmp_df.columns:
        drop_cols.append("key")
    tmp_X = tmp_df.drop(drop_cols, axis=1)
    tmp_y = tmp_df["fare_amount"]
    tmp_X = tmp_X.values.astype(np.float32)
    tmp_y = tmp_y.values.astype(np.float32)

    print("Before train test split")
    tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = train_test_split(
        tmp_X, tmp_y, test_size=0.05, random_state=42
    )
    print("df.shape:" + str(tmp_df.shape))
    print("tmp_X.shape:" + str(tmp_X.shape))
    print("tmp_y.shape:" + str(tmp_y.shape))
    return tmp_X, tmp_y, tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test


@fn_timer
def get_rmse(model, X, y):
    """
    Compute RMSE for both scikit‑learn models and XGBoost boosters.
    X can be a pandas DataFrame; if we are using XGBoost we wrap it in DMatrix.
    """
    if _USE_XGB:
        dmatrix = xgb.DMatrix(X)
        y_pred = model.predict(dmatrix)
    else:
        y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return rmse


@fn_timer
def fit_model(X_train, X_test, y_train, y_test):
    if _USE_XGB:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        params = {
            "eval_metric": "rmse",
            "max_depth": 7,
            "subsample": 0.8,
            "eta": 0.1,
            "gamma": 1,
            "colsample_bytree": 0.9,
        }
        model = xgb.train(params, dtrain, num_boost_round=250)
    else:
        model = RandomForestRegressor(
            n_estimators=250,  # ↓ from 600 (still strong enough)
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)
    if _USE_XGB:
        y_pred = model.predict(dtest)
        print("Validation RMSE (xgb):", np.sqrt(mean_squared_error(y_test, y_pred)))
    else:
        y_pred = model.predict(X_test)
        print("Validation RMSE (RF):", np.sqrt(mean_squared_error(y_test, y_pred)))
    del X_train, X_test, y_train, y_test
    gc.collect()
    return model


@fn_timer
def output_submission(models, df_test, test_X):
    """
    Produce the Kaggle submission file by averaging predictions from all models.
    Handles both sklearn models and XGBoost boosters.
    """
    start_time = time.time()
    preds = []
    for model in models:
        if _USE_XGB:
            preds.append(model.predict(xgb.DMatrix(test_X)))
        else:
            preds.append(np.expm1(model.predict(test_X)))
    test_pred = np.mean(preds, axis=0)
    test_pred = np.round(test_pred, 2)
    submission = pd.DataFrame(
        {"key": df_test["key"], "fare_amount": test_pred},
        columns=["key", "fare_amount"],
    )
    submission.to_csv("submission.csv", index=False)
    print(
        "--- Output submission: %s secs ---" % np.round((time.time() - start_time), 2)
    )


@fn_timer
def preprocess_df(df, ifTest):
    start_time = time.time()
    print("Preprocessing data start")
    if not ifTest:
        drop_rows_with_nan(df)
        df = data_cleanup(df)
    df = reformat_pickup_datetime(df)

    print("Adding day,time and month feature")
    add_day_of_week_feature(df)
    add_time_of_day_feature(df)
    add_month_feature(df)
    add_week_of_year_feature(df)

    print(df.shape)
    print(df.dtypes)

    print("Adding distance feature")
    add_distance_feature(df)

    print("Adding latitude/longitude delta features")
    add_coord_diff_feature(df)

    print("One hot encoding features")
    df = data_one_hot_encoding(df)

    if not ifTest:
        drop_rows_with_nan(df)

    gc.collect()
    print("Preprocessing data end")
    print("--- preprocess_df: %s secs ---" % np.round((time.time() - start_time), 2))
    return df


@fn_timer
def split_and_fit_model(df):
    tmp_df = preprocess_df(df, False)
    X, y, X_train, X_test, y_train, y_test = prepare_data_split(tmp_df)
    del tmp_df
    gc.collect()

    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    model = fit_model(X_train, X_test, y_train_log, y_test_log)

    preds_log = model.predict(X)
    preds = np.expm1(preds_log)
    rmse = np.sqrt(mean_squared_error(y, preds))
    print("Overall RMSE on full data:", rmse)
    del X, y, y_train, y_test
    gc.collect()
    return model, rmse


def dump_model(model, fileName):
    joblib.dump(model, fileName + ".compressed", compress=True)




## === cell 1
train_path = Path("../input/train.csv")
if not train_path.is_file():
    train_path = Path("./input/train.csv")
df_train = read_file_to_df(str(train_path), rows=1500000)

models = []
rmses = []

model, rmse = split_and_fit_model(df_train)
models.append(model)
rmses.append(rmse)

print("Training completed. RMSes:", rmses)




## === cell 2
test_path = Path("../input/test.csv")
if not test_path.is_file():
    test_path = Path("./input/test.csv")
df_test = read_test_file_to_df(str(test_path))
df_test = preprocess_df(df_test, True)
test_X = df_test.drop(["pickup_datetime", "key"], axis=1).values.astype(np.float32)

output_submission(models, df_test, test_X)

del df_test, test_X, models
