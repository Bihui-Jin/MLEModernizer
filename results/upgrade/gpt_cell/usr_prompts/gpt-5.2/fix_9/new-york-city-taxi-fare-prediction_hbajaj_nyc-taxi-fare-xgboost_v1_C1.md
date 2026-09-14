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

3.63072

# 6. Current score

5.39776

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.14242) has done: 'You don’t currently have a Kaggle score because the pipeline likely fails (or silently produces misaligned features) due to the `.feather` caching dependency and, more importantly, inconsistent one-hot columns between train and test (test can miss hours/months/days, and train can miss some too), which break XGBoost or degrade predictions. I make minimal, metric-relevant changes: (1) ensure both train and test get a fixed one-hot schema for time/day/month so train/test columns always match, (2) clamp negative predictions to 0 (fares can’t be negative; this usually improves RMSE slightly), and (3) make the train/test split deterministic for stability without changing the core approach. The model, features, and training loop remain the same; we only stabilize feature generation and submission validity so you can get a score and move toward the target.'
- What this solution (achieved 7.07335) has done: 'Your current gap to target is large (7.14242 vs 3.63072, lower is better), so we need a small but meaningful improvement without changing the overall XGBoost approach. The biggest score drag here is using a geometric-mean “stacking” over just one model, which is unnecessary and can distort predictions; we output the single model’s predictions directly. Next, we align the model’s objective with the RMSE metric by using `reg:squarederror` (same loss family, but correct explicit objective), and we train on the full cleaned batch (not a 95/5 split) while still printing a holdout RMSE for sanity—this reduces underfitting without changing architecture. Finally, we keep your fixed one-hot schema alignment and non-negative clipping to maintain submission validity and stability.'
- What this solution (achieved 7.82091) has done: 'Your score is far above the target (7.07 vs 3.63, lower is better), so we need a small-but-real modeling improvement while keeping the same XGBoost + feature-engineering pipeline. The biggest low-risk gain here is to add a few standard taxi-fare features (absolute lat/lon deltas and simple Manhattan distance) alongside your existing haversine distance; this preserves the core logic and usually cuts RMSE substantially. I also fix the RMSE calculation bug (it currently compares swapped arguments) so the printed RMSE reflects reality, without affecting training. Finally, I keep your fixed one-hot schema alignment and non-negative clipping so the submission stays valid and stable.'
- What this solution (achieved 5.62826) has done: 'We move your RMSE down toward the 3.63 target with the smallest changes that keep the same XGBoost + feature-engineering pipeline. The biggest score drag still present is the extremely slow/approximate row-wise haversine (string formatting + apply), so we replace it with an equivalent vectorized haversine computation (same feature, much more accurate numerically and consistent), which typically gives a meaningful RMSE drop without changing model logic. We also add the standard NYC bounding-box cleanup for test rows (clip to NaN then fill with 0 via reindex) to prevent pathological distances from exploding predictions. Finally, we keep your fixed one-hot schema and non-negative clipping, and we write `submission.csv` exactly as required.'
- What this solution (achieved 5.39776) has done: 'We keep your exact XGBoost setup and existing features, but make two minimal, metric-relevant improvements that typically reduce RMSE: (1) compute a more realistic “trip distance” by scaling your current haversine distance to kilometers (the current 6373 radius is in km but NYC fares correlate better to miles; we add a derived `distance_km` feature while retaining the original `distance` to preserve core logic), and (2) add a simple “airport proximity” signal (distance to JFK/LGA/EWR for pickup/dropoff) which is a standard low-risk improvement for this competition and doesn’t change the training approach. We also ensure train/test feature columns stay perfectly aligned by generating the schema once and reindexing both train and test to it (your test already does this; we make training use the same schema too). These are small feature additions and column-alignment stability changes only; the model/loop/objective remain unchanged and a valid `submission.csv` is still produced.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import glob
import time
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from math import sin, cos, sqrt, atan2, radians
from sklearn import metrics  # evaluating models
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from datetime import datetime

try:
    import joblib
except Exception:
    from sklearn.externals import joblib

from functools import wraps
import xgboost as xgb
import scipy
from sklearn.metrics import mean_squared_error

print(os.listdir("../input"))

os.makedirs("./model", exist_ok=True)




## === cell 1
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
def read_file_to_df(fileName, rows):
    start_time = time.time()
    traintypes = {
        "fare_amount": "float32",
        "pickup_datetime": "str",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
        "key": "str",
    }

    cols = list(traintypes.keys())

    if rows > 0:
        df = pd.read_csv(fileName, usecols=cols, dtype=traintypes, nrows=rows)
    else:
        df = pd.read_csv(fileName, usecols=cols, dtype=traintypes)

    print("--- Read Files: %s secs ---" % np.round((time.time() - start_time), 2))
    return df




## === cell 2
def read_test_file_to_df(fileName):
    df = pd.read_csv(fileName)
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
    tmp_df["weekOfYear"] = tmp_df["pickup_datetime"].dt.weekofyear.astype("uint8")


@fn_timer
def add_distance_feature(tmp_df):
    start_time = time.time()
    R = 6373.0  # original constant kept

    lat1 = np.radians(tmp_df["pickup_latitude"].astype("float64"))
    lon1 = np.radians(tmp_df["pickup_longitude"].astype("float64"))
    lat2 = np.radians(tmp_df["dropoff_latitude"].astype("float64"))
    lon2 = np.radians(tmp_df["dropoff_longitude"].astype("float64"))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    dist = (R * c).astype("float32")
    tmp_df["distance"] = dist  # preserve original feature name/semantics

    tmp_df["distance_km"] = (6371.0088 * c).astype("float32")

    print("--- Add distance: %s secs ---" % np.round((time.time() - start_time), 2))


@fn_timer
def add_manhattan_and_deltas_features(tmp_df):
    start_time = time.time()
    tmp_df["abs_lon_diff"] = np.abs(
        tmp_df["pickup_longitude"] - tmp_df["dropoff_longitude"]
    ).astype("float32")
    tmp_df["abs_lat_diff"] = np.abs(
        tmp_df["pickup_latitude"] - tmp_df["dropoff_latitude"]
    ).astype("float32")
    tmp_df["manhattan_distance"] = (
        tmp_df["abs_lon_diff"] + tmp_df["abs_lat_diff"]
    ).astype("float32")
    print(
        "--- Add manhattan/deltas: %s secs ---"
        % np.round((time.time() - start_time), 2)
    )


@fn_timer
def add_airport_distance_features(tmp_df):
    start_time = time.time()

    def _haversine_km(lat, lon, alat, alon):
        lat1 = np.radians(lat.astype("float64"))
        lon1 = np.radians(lon.astype("float64"))
        lat2 = np.radians(np.float64(alat))
        lon2 = np.radians(np.float64(alon))
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return (6371.0088 * c).astype("float32")

    airports = {
        "JFK": (40.6413, -73.7781),
        "LGA": (40.7769, -73.8740),
        "EWR": (40.6895, -74.1745),
    }

    for code, (alat, alon) in airports.items():
        tmp_df[f"pickup_dist_{code}_km"] = _haversine_km(
            tmp_df["pickup_latitude"], tmp_df["pickup_longitude"], alat, alon
        )
        tmp_df[f"dropoff_dist_{code}_km"] = _haversine_km(
            tmp_df["dropoff_latitude"], tmp_df["dropoff_longitude"], alat, alon
        )

    print(
        "--- Add airport distances: %s secs ---"
        % np.round((time.time() - start_time), 2)
    )


@fn_timer
def filter_based_on_distance(df, distance):
    return df[(df["distance"] < distance)]


@fn_timer
def data_one_hot_encoding(tmp_df):
    print("Before one hot encoding")
    start_time = time.time()

    tmp_df["timeOfDay"] = pd.Categorical(
        tmp_df["timeOfDay"], categories=list(range(0, 24))
    )
    tmp_df["dayOfWeek"] = pd.Categorical(
        tmp_df["dayOfWeek"], categories=list(range(0, 7))
    )
    tmp_df["month"] = pd.Categorical(tmp_df["month"], categories=list(range(1, 13)))

    tmp_df = pd.concat(
        [tmp_df, pd.get_dummies(tmp_df["timeOfDay"], prefix="timeOfDay")], axis=1
    )
    tmp_df.drop(["timeOfDay"], axis=1, inplace=True)

    tmp_df = pd.concat(
        [tmp_df, pd.get_dummies(tmp_df["dayOfWeek"], prefix="dayOfWeek")], axis=1
    )
    tmp_df.drop(["dayOfWeek"], axis=1, inplace=True)

    tmp_df = pd.concat(
        [tmp_df, pd.get_dummies(tmp_df["month"], prefix="month")], axis=1
    )
    tmp_df.drop(["month"], axis=1, inplace=True)

    print(tmp_df.columns)
    print("--- One hot encoding: %s secs ---" % np.round((time.time() - start_time), 2))
    return tmp_df


@fn_timer
def prepare_data_split(tmp_df):
    tmp_X = tmp_df.drop(["pickup_datetime", "fare_amount", "key"], axis=1)
    tmp_y = tmp_df["fare_amount"]
    print("Before train test split")
    tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = train_test_split(
        tmp_X, tmp_y, test_size=0.05, random_state=42
    )
    print("df.shape:" + str(tmp_df.shape))
    print("tmp_X.shape:" + str(tmp_X.shape))
    print("tmp_y.shape:" + str(tmp_y.shape))
    return tmp_X, tmp_y, tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test


@fn_timer
def get_rmse(model, data, output):
    dtrain = xgb.DMatrix(data, label=output)
    y_pred = model.predict(dtrain)
    rmse = np.sqrt(metrics.mean_squared_error(output, y_pred))
    return rmse


@fn_timer
def fit_xgboost_model(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 7,
        "subsample": 0.8,
        "eta": 0.1,
        "gamma": 1,
        "colsample_bytree": 0.9,
        "seed": 42,
    }
    print("fit_xgboost_model")
    model = xgb.train(params, dtrain, num_boost_round=250)

    y_pred = model.predict(dtest)
    y_train_pred = model.predict(dtrain)

    print(np.sqrt(mean_squared_error(y_test, y_pred)))
    print(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    return model


@fn_timer
def output_submission(model, df_test, test_X):
    start_time = time.time()
    test_pred = model.predict(xgb.DMatrix(test_X))
    test_pred = np.maximum(test_pred, 0.0)
    submission = pd.DataFrame(
        {"key": df_test.key, "fare_amount": test_pred}, columns=["key", "fare_amount"]
    )
    submission.to_csv("submission.csv", index=False)
    print(
        "--- Output submission: %s secs ---" % np.round((time.time() - start_time), 2)
    )


@fn_timer
def output_submission_stacking(df_test, test_d):
    test_X = xgb.DMatrix(test_d)
    rmse_df = pd.DataFrame()
    for model in models:
        test_pred = model.predict(test_X)
        test_pred = np.maximum(test_pred, 0.0)
        rmse_df = pd.concat([rmse_df, pd.DataFrame(test_pred)], axis=1)
        del model

    submission = pd.DataFrame(
        {"key": df_test.key, "fare_amount": scipy.stats.mstats.gmean(rmse_df, axis=1)},
        columns=["key", "fare_amount"],
    )
    submission.to_csv("submission.csv", index=False)


def _clip_test_coordinates_inplace(df):
    for col, lo, hi in [
        ("pickup_longitude", -80.0, -70.0),
        ("dropoff_longitude", -80.0, -70.0),
        ("pickup_latitude", 35.0, 45.0),
        ("dropoff_latitude", 35.0, 45.0),
    ]:
        bad = (df[col] < lo) | (df[col] > hi)
        if bad.any():
            df.loc[bad, col] = np.nan


@fn_timer
def preprocess_df(df, ifTest):
    start_time = time.time()

    print("Preprocessing data start")
    if ifTest == False:
        drop_rows_with_nan(df)
        df = data_cleanup(df)
    else:
        _clip_test_coordinates_inplace(df)

    df = reformat_pickup_datetime(df)
    print("Adding day,time and month feature")
    add_day_of_week_feature(df)
    add_time_of_day_feature(df)
    add_month_feature(df)
    print(df.shape)
    print(df.dtypes)
    df.head()
    print("Adding distance feature")
    add_distance_feature(df)

    add_manhattan_and_deltas_features(df)

    add_airport_distance_features(df)

    print("One hot encoding features")
    if ifTest == False:
        df = filter_based_on_distance(df, 100)
    df = data_one_hot_encoding(df)
    if ifTest == False:
        drop_rows_with_nan(df)
    print("Preprocessing data end")
    print("--- preprocess_df: %s secs ---" % np.round((time.time() - start_time), 2))
    return df


@fn_timer
def split_and_fit_model(df, train_feature_columns=None):
    tmp_df = preprocess_df(df, False)
    print("Shape of tmp_df: {}".format(tmp_df.shape))
    print("Info of tmp_df: {}".format(tmp_df.info()))

    X, y, X_train, X_test, y_train, y_test = prepare_data_split(tmp_df)
    if train_feature_columns is not None:
        X = X.reindex(columns=train_feature_columns, fill_value=0)
        X_train = X_train.reindex(columns=train_feature_columns, fill_value=0)
        X_test = X_test.reindex(columns=train_feature_columns, fill_value=0)

    model = fit_xgboost_model(X_train, X_test, y_train, y_test)

    dfull = xgb.DMatrix(X, label=y)
    params_full = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 7,
        "subsample": 0.8,
        "eta": 0.1,
        "gamma": 1,
        "colsample_bytree": 0.9,
        "seed": 42,
    }
    model_full = xgb.train(params_full, dfull, num_boost_round=250)

    rmse = get_rmse(model_full, X, y)
    print("rmse:" + str(rmse))

    del tmp_df, X, y, X_train, X_test, y_train, y_test, model, dfull
    return model_full, rmse


def dump_model(model, fileName):
    joblib.dump(model, fileName + ".compressed", compress=True)




## === cell 3
reader = pd.read_csv("../input/train.csv", header=0, iterator=True)

models = []
rmses = []
TRAIN_FEATURE_COLUMNS = None

pklFileName = "./model/modelFile"
for i in range(0, 1):
    print("##################### Batch start num {} #####################".format(i))
    batchSize = 1_00_000
    df = reader.get_chunk(batchSize)

    tmp_df_for_schema = preprocess_df(df.copy(), False)
    TRAIN_FEATURE_COLUMNS = tmp_df_for_schema.drop(
        ["pickup_datetime", "fare_amount", "key"], axis=1
    ).columns
    del tmp_df_for_schema

    model, rmse = split_and_fit_model(df, train_feature_columns=TRAIN_FEATURE_COLUMNS)
    models.append(model)
    rmses.append(rmse)
    del df
    print("##################### Batch end #####################")

reader.close()
del reader



## === cell 4
df_test = read_test_file_to_df("../input/test.csv")
df_test = preprocess_df(df_test, True)

test_X = df_test.drop(["pickup_datetime", "key"], axis=1)
test_X = test_X.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)

output_submission(models[0], df_test, test_X)

del df_test
del test_X
