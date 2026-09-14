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

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O

import os
import time
from math import sin, cos, sqrt, atan2, radians
from functools import wraps

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import scipy

try:
    import joblib
except Exception:
    from sklearn.externals import joblib  # pragma: no cover

INPUT_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
print("Input dir exists:", os.path.exists(INPUT_DIR))
print("Files:", sorted(os.listdir(INPUT_DIR))[:20])




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

    from pathlib import Path

    my_file = Path(fileName + ".feather")
    if my_file.is_file():
        df = pd.read_feather(fileName + ".feather")
    else:
        if rows and rows > 0:
            df = pd.read_csv(fileName, usecols=cols, dtype=traintypes, nrows=rows)
        else:
            df = pd.read_csv(fileName, usecols=cols, dtype=traintypes)
        df.to_feather(fileName + ".feather")
    print("--- Read Files: %s secs ---" % np.round((time.time() - start_time), 2))
    return df


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
    df["pickup_datetime"] = df["pickup_datetime"].astype(str).str.slice(0, 13)
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


def distance_between_two_points(row):
    R = 6373.0

    lat1 = radians(row["pickup_latitude"])
    lon1 = radians(row["pickup_longitude"])
    lat2 = radians(row["dropoff_latitude"])
    lon2 = radians(row["dropoff_longitude"])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return "{0:.3f}".format(distance)


@fn_timer
def add_distance_feature(tmp_df):
    start_time = time.time()
    tmp_df["distance"] = (
        tmp_df.apply(distance_between_two_points, axis=1)
        .apply(pd.to_numeric)
        .astype("float32")
    )
    print("--- Add distance: %s secs ---" % np.round((time.time() - start_time), 2))


@fn_timer
def filter_based_on_distance(df, distance):
    df = df[(df["distance"] < distance)]
    return df


@fn_timer
def data_one_hot_encoding(tmp_df):
    print("Before one hot encoding")
    start_time = time.time()
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
    rmse = np.sqrt(metrics.mean_squared_error(y_pred, output))
    return rmse


@fn_timer
def fit_xgboost_model(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    params = {
        "eval_metric": "rmse",
        "max_depth": 7,
        "subsample": 0.8,
        "eta": 0.1,
        "gamma": 1,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "seed": 42,
    }
    print("fit_xgboost_model")
    model = xgb.train(params, dtrain, num_boost_round=250)

    y_pred = model.predict(dtest)
    y_train_pred = model.predict(dtrain)

    print("Holdout RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
    return model


@fn_timer
def output_submission_stacking(df_test, test_d, models):
    test_X = xgb.DMatrix(test_d)

    rmse_df = pd.DataFrame()
    for m in models:
        test_pred = m.predict(test_X)
        test_pred = np.round(test_pred, 2)
        rmse_df = pd.concat([rmse_df, pd.DataFrame(test_pred)], axis=1)

    submission = pd.DataFrame(
        {
            "key": df_test.key,
            "fare_amount": np.round(scipy.stats.mstats.gmean(rmse_df, axis=1), 2),
        },
        columns=["key", "fare_amount"],
    )
    submission.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission.shape)


@fn_timer
def preprocess_df(df, ifTest):
    start_time = time.time()

    print("Preprocessing data start")
    if ifTest is False:
        drop_rows_with_nan(df)
        df = data_cleanup(df)

    df = reformat_pickup_datetime(df)

    print("Adding day,time and month feature")
    add_day_of_week_feature(df)
    add_time_of_day_feature(df)
    add_month_feature(df)

    print("Adding distance feature")
    add_distance_feature(df)

    if ifTest is False:
        df = filter_based_on_distance(df, 100)

    print("One hot encoding features")
    df = data_one_hot_encoding(df)

    if ifTest is False:
        drop_rows_with_nan(df)

    print("Preprocessing data end")
    print("--- preprocess_df: %s secs ---" % np.round((time.time() - start_time), 2))
    return df


@fn_timer
def split_and_fit_model(df):
    tmp_df = preprocess_df(df, False)
    print("Shape of tmp_df: {}".format(tmp_df.shape))

    X, y, X_train, X_test, y_train, y_test = prepare_data_split(tmp_df)

    model = fit_xgboost_model(X_train, X_test, y_train, y_test)
    rmse = get_rmse(model, X, y)
    print("rmse:" + str(rmse))
    return model, rmse, list(X.columns)


def dump_model(model, fileName):
    os.makedirs(os.path.dirname(fileName), exist_ok=True)
    joblib.dump(model, fileName + ".compressed", compress=True)




## === cell 2
train_path = os.path.join(INPUT_DIR, "train.csv")
reader = pd.read_csv(train_path, header=0, iterator=True)

models = []
rmses = []
feature_columns = None

for i in range(0, 1):
    print("##################### Batch start num {} #####################".format(i))
    batchSize = 100_000
    df = reader.get_chunk(batchSize)

    model, rmse, feat_cols = split_and_fit_model(df)
    models.append(model)
    rmses.append(rmse)
    feature_columns = feat_cols  # columns used by training (after OHE)

    del df
    print("##################### Batch end #####################")

reader.close()
del reader

print("RMSEs:", rmses)



## === cell 3
test_path = os.path.join(INPUT_DIR, "test.csv")
df_test_raw = read_test_file_to_df(test_path)

df_test = preprocess_df(df_test_raw, True)
test_X = df_test.drop(["pickup_datetime", "key"], axis=1)

if feature_columns is None:
    raise RuntimeError("feature_columns is None; training did not run correctly.")
test_X = test_X.reindex(columns=feature_columns, fill_value=0)

output_submission_stacking(df_test, test_X, models)

sub = pd.read_csv("submission.csv")
print(sub.head())
print(sub.shape)
print(sub.columns.tolist())
assert sub.columns.tolist() == ["key", "fare_amount"]
assert len(sub) == len(df_test)

del df_test_raw, df_test, test_X, sub
