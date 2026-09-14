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

3.93524

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

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
train_path = "../input/train.csv"
test_path = "../input/test.csv"

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
mask = (
    (df_train.pickup_longitude >= BB[0])
    & (df_train.pickup_longitude <= BB[1])
    & (df_train.pickup_latitude >= BB[2])
    & (df_train.pickup_latitude <= BB[3])
    & (df_test.pickup_longitude >= BB[0])
    & (df_test.pickup_longitude <= BB[1])
    & (df_test.pickup_latitude >= BB[2])
    & (df_test.pickup_latitude <= BB[3])
)
df_train = df_train[mask[: len(df_train)]]
df_test = df_test[mask[len(df_train) :]]




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
IndexingError                             Traceback (most recent call last)
/tmp/ipykernel_11/1703657479.py in <cell line: 0>()
     16     & (df_test.pickup_latitude <= BB[3])
     17 )
---> 18 df_train = df_train[mask[: len(df_train)]]
     19 df_test = df_test[mask[len(df_train) :]]
     20 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4091         # Do we have a (boolean) 1d indexer?
   4092         if com.is_bool_indexer(key):
-> 4093             return self._getitem_bool_array(key)
   4094 
   4095         # We are left with two options: a single key, and a collection of keys,

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _getitem_bool_array(self, key)
   4147         # check_bool_indexer will throw exception if Series key cannot
   4148         # be reindexed to match DataFrame rows
-> 4149         key = check_bool_indexer(self.index, key)
   4150 
   4151         if key.all():

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in check_bool_indexer(index, key)
   2660         indexer = result.index.get_indexer_for(index)
   2661         if -1 in indexer:
-> 2662             raise IndexingError(
   2663                 "Unalignable boolean Series provided as "
   2664                 "indexer (index of the boolean Series and of "

IndexingError: Unalignable boolean Series provided as indexer (index of the boolean Series and of the indexed object do not match).

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
df_test = df_test[df_test.trip_distance < THRESHOLD_TRIP_DISTANCE]




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



## === cell 8
cols_to_drop = [
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "trip_rate",  # keep only for potential analysis; drop before modeling
]
df_train = df_train.drop(columns=cols_to_drop)
df_test = df_test.drop(columns=cols_to_drop)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2665429679.py in <cell line: 0>()
      9 ]
     10 df_train = df_train.drop(columns=cols_to_drop)
---> 11 df_test = df_test.drop(columns=cols_to_drop)
     12 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   5579                 weight  1.0     0.8
   5580         """
-> 5581         return super().drop(
   5582             labels=labels,
   5583             axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   4786         for axis, labels in axes.items():
   4787             if labels is not None:
-> 4788                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
   4789 
   4790         if inplace:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _drop_axis(self, labels, axis, level, errors, only_slice)
   4828                 new_axis = axis.drop(labels, level=level, errors=errors)
   4829             else:
-> 4830                 new_axis = axis.drop(labels, errors=errors)
   4831             indexer = axis.get_indexer(new_axis)
   4832 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in drop(self, labels, errors)
   7068         if mask.any():
   7069             if errors != "ignore":
-> 7070                 raise KeyError(f"{labels[mask].tolist()} not found in axis")
   7071             indexer = indexer[~mask]
   7072         return self.delete(indexer)

KeyError: "['trip_rate'] not found in axis"

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
        n_estimators=500, learning_rate=0.03, max_depth=8, random_state=42
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
    test_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
else:
    test_pred = model.predict(df_test)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3045304631.py in <cell line: 0>()
      1 # Predict on the original test set
      2 if xgboost_available:
----> 3     dtest = xgb.DMatrix(df_test)
      4     test_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
      5 else:

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in __init__(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)
    855             return
    856 
--> 857         handle, feature_names, feature_types = dispatch_data_backend(
    858             data,
    859             missing=self.missing,

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in dispatch_data_backend(data, missing, threads, feature_names, feature_types, enable_categorical, data_split_mode)
   1087         data = pd.DataFrame(data)
   1088     if _is_pandas_df(data):
-> 1089         return _from_pandas_df(
   1090             data, enable_categorical, missing, threads, feature_names, feature_types
   1091         )

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _from_pandas_df(data, enable_categorical, missing, nthread, feature_names, feature_types)
    520     feature_types: Optional[FeatureTypes],
    521 ) -> DispatchedDataBackendReturnType:
--> 522     data, feature_names, feature_types = _transform_pandas_df(
    523         data, enable_categorical, feature_names, feature_types
    524     )

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _transform_pandas_df(data, enable_categorical, feature_names, feature_types, meta, meta_type)
    488             or is_pa_ext_dtype(dtype)
    489         ):
--> 490             _invalid_dataframe_dtype(data)
    491         if is_pa_ext_dtype(dtype):
    492             pyarrow_extension = True

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _invalid_dataframe_dtype(data)
    306     type_err = "DataFrame.dtypes for data must be int, float, bool or category."
    307     msg = f"""{type_err} {_ENABLE_CAT_ERR} {err}"""
--> 308     raise ValueError(msg)
    309 
    310 

ValueError: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, The experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:pickup_datetime: datetime64[ns, UTC]

## === cell 13
submission = pd.DataFrame({"key": test_key, "fare_amount": np.round(test_pred, 2)})
output_path = "taxi_fare_submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
print(submission.head())

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3747436901.py in <cell line: 0>()
      1 # Build submission file
----> 2 submission = pd.DataFrame({"key": test_key, "fare_amount": np.round(test_pred, 2)})
      3 output_path = "taxi_fare_submission.csv"
      4 submission.to_csv(output_path, index=False)
      5 print(f"Submission written to {output_path}")

NameError: name 'test_pred' is not defined
