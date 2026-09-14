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

# 8. Previous improvement plans

- What this solution (achieved 5.16136) has done: 'Implemented a fix in the prediction step to ensure the submission file contains predictions for **all** test rows.  
The original code filtered `df_test`, causing a length mismatch with the original `test_key`.  
Now we:
1. Preserve the original row indices after filtering.  
2. Build a full‑length prediction array, inserting model outputs at the retained indices.  
3. Fill missing predictions with the training mean fare (a sensible default).  
4. Produce a correctly sized submission CSV.'
- What this solution (achieved 5.08455) has done: 'I increase the amount of training data used, cap extreme fare values, add a simple Manhattan‑distance feature, and give the GradientBoostingRegressor a few more trees with a slightly lower learning rate. These modest tweaks keep the original model and pipeline intact while improving prediction quality and moving the RMSE closer to the target.'
- What this solution (achieved 5.08455) has done: 'I add a small amount of regularization (subsample 0.8 and max_features “sqrt”) and increase the number of trees to give the GradientBoostingRegressor a bit more capacity while reducing over‑fit. I also clip any negative predictions to 0, which removes impossible fare values and typically lowers RMSE. These tweaks keep the overall pipeline unchanged but should move the validation RMSE closer to the target.'
- What this solution (achieved 5.18393) has done: 'I add a log‑transform of the target (fare_amount) to make the distribution more Gaussian, which usually improves GradientBoosting performance, and I also expose the trip distance in miles as an extra feature. These small, targeted changes keep the original modelling pipeline intact while helping the validation RMSE move closer to the target score.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

_XGB_AVAILABLE = False
try:
    import xgboost as xgb

    _XGB_AVAILABLE = True
except Exception:
    _XGB_AVAILABLE = False

MAX_TRAINING_SIZE = 1_000_000  # use more data for a better model (was 500k)
EPS_IN_KM = 0.5
MIN_SAMPLES_CLUSTER = 500
RADIUS_VICINITY_AIRPORTS = 1.0
THERSHOLD_TRIP_FARE_RATE = 50.0
KMS_PER_RADIAN = 6371.0088

JFK_GEO_LOCATION = (40.6413, -73.7781)
LGR_GEO_LOCATION = (40.7769, -73.8740)
EWR_GEO_LOCATION = (40.6895, -74.1745)


def haversine(coord1, coord2):
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return KMS_PER_RADIAN * c


base_dir = "data"
train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")

df_train = pd.read_csv(
    train_path,
    nrows=MAX_TRAINING_SIZE,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)

df_test = pd.read_csv(
    test_path,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)

test_key = df_test["key"].values.copy()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3493933634.py in <cell line: 0>()
     43 
     44 # Load a subset for faster experimentation
---> 45 df_train = pd.read_csv(
     46     train_path,
     47     nrows=MAX_TRAINING_SIZE,

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'data/train.csv'

## === cell 1
def add_distance_features(df):
    df["trip_distance"] = df.apply(
        lambda row: haversine(
            (row["pickup_latitude"], row["pickup_longitude"]),
            (row["dropoff_latitude"], row["dropoff_longitude"]),
        ),
        axis=1,
    )
    df["trip_distance_miles"] = df["trip_distance"] * 0.621371
    df["manhattan_distance"] = np.abs(
        df["pickup_latitude"] - df["dropoff_latitude"]
    ) + np.abs(df["pickup_longitude"] - df["dropoff_longitude"])
    df["manhattan_distance_miles"] = df["manhattan_distance"] * 0.621371
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


def add_interaction_features(df):
    df["hour_tripdist"] = df["hour"] * df["trip_distance"]
    return df


df_train = add_interaction_features(df_train)
df_test = add_interaction_features(df_test)

drop_cols = [
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
df_train = df_train.drop(columns=drop_cols)
df_test = df_test.drop(columns=drop_cols)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/73735975.py in <cell line: 0>()
     15 
     16 
---> 17 df_train = add_distance_features(df_train)
     18 df_test = add_distance_features(df_test)
     19 

NameError: name 'df_train' is not defined

## === cell 2
y = df_train["fare_amount"]
y_log = np.log1p(y)  # log(1 + fare)

X = df_train.drop(columns=["fare_amount"])

X_tr, X_val, y_tr, y_val = train_test_split(X, y_log, test_size=0.01, random_state=0)

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
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        max_features="sqrt",
        random_state=0,
    )
    model.fit(X_tr, y_tr)

if _XGB_AVAILABLE:
    preds_val_log = model.predict(dval)
else:
    preds_val_log = model.predict(X_val)

preds_val = np.expm1(preds_val_log)
preds_val = np.maximum(preds_val, 0)

val_rmse = np.sqrt(
    mean_squared_error(y_val, np.log1p(preds_val))
)  # evaluate in log‑space
print(f"Validation RMSE (log‑space): {val_rmse:.4f}")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3455736649.py in <cell line: 0>()
----> 1 y = df_train["fare_amount"]
      2 y_log = np.log1p(y)  # log(1 + fare)
      3 
      4 X = df_train.drop(columns=["fare_amount"])
      5 

NameError: name 'df_train' is not defined

## === cell 3
if _XGB_AVAILABLE:
    dtest = xgb.DMatrix(df_test)
    predict_kwargs = {}
    if hasattr(model, "best_ntree_limit"):
        predict_kwargs["ntree_limit"] = model.best_ntree_limit
    test_pred_log = model.predict(dtest, **predict_kwargs)
else:
    test_pred_log = model.predict(df_test)

test_pred = np.expm1(test_pred_log)
test_pred = np.maximum(test_pred, 0)
test_pred = np.clip(test_pred, 0, 200)

full_pred = test_pred  # length already matches test_key

mean_fare = y.mean()
full_pred = np.where(np.isnan(full_pred), mean_fare, full_pred)

submission = pd.DataFrame({"key": test_key, "fare_amount": np.round(full_pred, 2)})
submission_path = "taxi_fare_submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4232355095.py in <cell line: 0>()
      1 if _XGB_AVAILABLE:
----> 2     dtest = xgb.DMatrix(df_test)
      3     predict_kwargs = {}
      4     if hasattr(model, "best_ntree_limit"):
      5         predict_kwargs["ntree_limit"] = model.best_ntree_limit

NameError: name 'df_test' is not defined
