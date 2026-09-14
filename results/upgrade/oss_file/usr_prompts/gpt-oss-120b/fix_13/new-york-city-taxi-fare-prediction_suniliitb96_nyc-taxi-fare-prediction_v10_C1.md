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

MAX_TRAINING_SIZE = 1_000_000  # use up to 1M rows for faster experimentation
KMS_PER_RADIAN = 6371.0088


def haversine(coord1, coord2):
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return KMS_PER_RADIAN * c


base_dirs = [
    "data",
    "kaggle/data",
    "kaggle/input",
    "kaggle/working",
    "working",
    ".",
]
train_path = None
test_path = None

for base in base_dirs:
    possible_train = [
        os.path.join(base, "train.csv"),
        os.path.join(base, "new-york-city-taxi-fare-prediction", "train.csv"),
        os.path.join(base, "input", "train.csv"),
        os.path.join(base, "input", "new-york-city-taxi-fare-prediction", "train.csv"),
    ]
    possible_test = [
        os.path.join(base, "test.csv"),
        os.path.join(base, "new-york-city-taxi-fare-prediction", "test.csv"),
        os.path.join(base, "input", "test.csv"),
        os.path.join(base, "input", "new-york-city-taxi-fare-prediction", "test.csv"),
    ]
    for p in possible_train:
        if os.path.exists(p):
            train_path = p
            break
    for p in possible_test:
        if os.path.exists(p):
            test_path = p
            break
    if train_path and test_path:
        break

if train_path is None or test_path is None:
    raise FileNotFoundError(
        "Unable to locate train.csv or test.csv in expected directories."
    )

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

df_train = df_train.drop(columns=["key"])
df_test = df_test.drop(columns=["key"])




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

    df["dist_per_passenger"] = df["trip_distance"] / df["passenger_count"].replace(
        0, np.nan
    )
    df["dist_per_passenger"].fillna(df["trip_distance"], inplace=True)

    df["is_weekend"] = df["pickup_datetime"].dt.weekday.isin([5, 6]).astype(int)

    hour = df["pickup_datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

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
    df["hour_manhattan"] = df["hour"] * df["manhattan_distance"]
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
        n_estimators=3500,
        learning_rate=0.008,
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

val_rmse = np.sqrt(mean_squared_error(y_val, np.log1p(preds_val)))
print(f"Validation RMSE (log‑space): {val_rmse:.4f}")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/2772125582.py in <cell line: 0>()
      7 
      8 if _XGB_AVAILABLE:
----> 9     dtrain = xgb.DMatrix(X_tr, label=y_tr)
     10     dval = xgb.DMatrix(X_val, label=y_val)
     11     params = {

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in __init__(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)
    867         self.handle = handle
    868 
--> 869         self.set_info(
    870             label=label,
    871             weight=weight,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in set_info(self, label, weight, base_margin, group, qid, label_lower_bound, label_upper_bound, feature_names, feature_types, feature_weights)
    930 
    931         if label is not None:
--> 932             self.set_label(label)
    933         if weight is not None:
    934             self.set_weight(weight)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in set_label(self, label)
   1068         from .data import dispatch_meta_backend
   1069 
-> 1070         dispatch_meta_backend(self, label, "label", "float")
   1071 
   1072     def set_weight(self, weight: ArrayLike) -> None:

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in dispatch_meta_backend(matrix, data, name, dtype)
   1223         return
   1224     if _is_pandas_series(data):
-> 1225         _meta_from_pandas_series(data, name, dtype, handle)
   1226         return
   1227     if _is_dlpack(data):

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _meta_from_pandas_series(data, name, dtype, handle)
    543         data = data.to_dense()  # type: ignore
    544     assert len(data.shape) == 1 or data.shape[1] == 0 or data.shape[1] == 1
--> 545     _meta_from_numpy(data, name, dtype, handle)
    546 
    547 

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _meta_from_numpy(data, field, dtype, handle)
   1157         raise ValueError("Masked array is not supported.")
   1158     interface_str = _array_interface(data)
-> 1159     _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle, c_str(field), interface_str))
   1160 
   1161 

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _check_call(ret)
    280     """
    281     if ret != 0:
--> 282         raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    283 
    284 

XGBoostError: [02:26:40] /workspace/src/data/data.cc:507: Check failed: valid: Label contains NaN, infinity or a value too large.
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3588ca) [0x7f372e8978ca]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x38a21d) [0x7f372e8c921d]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x38ab51) [0x7f372e8c9b51]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGDMatrixSetInfoFromInterface+0xb0) [0x7f372e69d3a0]
  [bt] (4) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7f37a1effe2e]
  [bt] (5) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7f37a1efc493]
  [bt] (6) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7f37a1f0f4d8]
  [bt] (7) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7f37a1f0ec8e]
  [bt] (8) /usr/bin/python3(_PyObject_MakeTpCall+0x27c) [0x52f85c]



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

full_pred = test_pred
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
/tmp/ipykernel_11/2386282464.py in <cell line: 0>()
      2     dtest = xgb.DMatrix(df_test)
      3     predict_kwargs = {}
----> 4     if hasattr(model, "best_ntree_limit"):
      5         predict_kwargs["ntree_limit"] = model.best_ntree_limit
      6     test_pred_log = model.predict(dtest, **predict_kwargs)

NameError: name 'model' is not defined
