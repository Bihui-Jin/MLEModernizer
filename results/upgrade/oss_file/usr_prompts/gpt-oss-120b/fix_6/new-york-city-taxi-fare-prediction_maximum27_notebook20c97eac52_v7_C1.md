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

3.12

# 3. Installed packages

cudf-polars-cu12==25.6.0
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
polars==1.25.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

3.60484

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import polars as pl
import numpy as np

train_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
test_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")

num_rows = train_df.height
random_indices = np.random.choice(num_rows, size=10000000, replace=False)
train_df = train_df[random_indices]
train_df = train_df.drop_nulls()



## === cell 1
train = train_df
test = test_df



## === cell 2
import polars as pl
import numpy as np


def preprocess(df):
    df = df.with_columns(
        pl.col("pickup_datetime")
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S UTC")
        .alias("pickup_datetime")
    )
    df = df.with_columns(
        [
            pl.col("pickup_datetime").dt.year().alias("pickup_year"),
            pl.col("pickup_datetime").dt.month().alias("pickup_month"),
            pl.col("pickup_datetime").dt.day().alias("pickup_day"),
            pl.col("pickup_datetime").dt.hour().alias("pickup_hour"),
            pl.col("pickup_datetime").dt.minute().alias("pickup_minute"),
            pl.col("pickup_datetime").dt.second().alias("pickup_second"),
            pl.col("pickup_datetime").dt.weekday().alias("pickup_weekday"),
        ]
    )
    return df


def haversine(df):
    """Add realistic haversine distance (km) between pickup and dropoff."""
    R = 6371.0  # Earth radius in km
    lat1 = pl.col("pickup_latitude") * (np.pi / 180)
    lat2 = pl.col("dropoff_latitude") * (np.pi / 180)
    dlat = (pl.col("dropoff_latitude") - pl.col("pickup_latitude")) * (np.pi / 180)
    dlon = (pl.col("dropoff_longitude") - pl.col("pickup_longitude")) * (np.pi / 180)

    a = ((dlat / 2).sin()).pow(2) + (
        lat1.cos() * lat2.cos() * ((dlon / 2).sin()).pow(2)
    )
    distance = 2 * R * pl.atan2(pl.sqrt(a), pl.sqrt(1 - a))
    return df.with_columns(distance.alias("distance"))


def drop_encoding(df):
    drop_columns = ["key", "pickup_datetime", "pickup_minute", "pickup_second"]
    return df.drop(drop_columns)


def cycling_encoding(df):
    df = df.with_columns(
        [
            (pl.col("pickup_month") * 2 * np.pi / 12).sin().alias("pickup_month_sin"),
            (pl.col("pickup_month") * 2 * np.pi / 12).cos().alias("pickup_month_cos"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).sin().alias("pickup_hour_sin"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).cos().alias("pickup_hour_cos"),
        ]
    )
    return df


def one_hot_encoding(df):
    one_hot_cols = ["pickup_year", "pickup_day"]
    return df.to_dummies(one_hot_cols)


def drop_outliner(df):
    """Remove extreme geographic and passenger‑count values."""
    return (
        df.filter((pl.col("pickup_latitude") >= 40) & (pl.col("pickup_latitude") < 41))
        .filter(
            (pl.col("pickup_longitude") >= -74) & (pl.col("pickup_longitude") < -73)
        )
        .filter(
            (pl.col("dropoff_longitude") >= -74) & (pl.col("dropoff_longitude") < -73)
        )
        .filter((pl.col("dropoff_latitude") >= 40) & (pl.col("dropoff_latitude") < 41))
        .filter((pl.col("passenger_count") >= 1) & (pl.col("passenger_count") < 6))
        .filter(pl.col("fare_amount") < 10000)
    )


train = preprocess(train)
train = drop_outliner(train)  # filter outliers
train = haversine(train)  # realistic distance feature
train = drop_encoding(train)
train = cycling_encoding(train)
train = one_hot_encoding(train)

test = preprocess(test)
test_key = test["key"].to_pandas().values  # keep for submission
test = haversine(test)
test = drop_encoding(test)
test = cycling_encoding(test)
test = one_hot_encoding(test)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3646024863.py in <cell line: 0>()
     78 train = preprocess(train)
     79 train = drop_outliner(train)  # filter outliers
---> 80 train = haversine(train)  # realistic distance feature
     81 train = drop_encoding(train)
     82 train = cycling_encoding(train)

/tmp/ipykernel_55/3646024863.py in haversine(df)
     34         lat1.cos() * lat2.cos() * ((dlon / 2).sin()).pow(2)
     35     )
---> 36     distance = 2 * R * pl.atan2(pl.sqrt(a), pl.sqrt(1 - a))
     37     return df.with_columns(distance.alias("distance"))
     38 

/usr/local/lib/python3.11/dist-packages/polars/__init__.py in __getattr__(name)
    443 
    444     msg = f"module {__name__!r} has no attribute {name!r}"
--> 445     raise AttributeError(msg)

AttributeError: module 'polars' has no attribute 'atan2'

## === cell 3
train_feature_cols = [c for c in train.columns if c != "fare_amount"]
for col in train_feature_cols:
    if col not in test.columns:
        test = test.with_column(pl.lit(0).alias(col))
test = test.select(train_feature_cols)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1203993415.py in <cell line: 0>()
      3 for col in train_feature_cols:
      4     if col not in test.columns:
----> 5         test = test.with_column(pl.lit(0).alias(col))
      6 # Ensure same order
      7 test = test.select(train_feature_cols)

AttributeError: 'DataFrame' object has no attribute 'with_column'

## === cell 4
import polars as pl

float64_cols_train = [c for c, t in zip(train.columns, train.dtypes) if t == pl.Float64]
train = train.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols_train])

float64_cols_test = [c for c, t in zip(test.columns, test.dtypes) if t == pl.Float64]
test = test.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols_test])

numeric_types = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}
train = train.select(
    [pl.col(c) for c in train.columns if train.schema[c] in numeric_types]
)
test = test.select([pl.col(c) for c in test.columns if test.schema[c] in numeric_types])



## === cell 5
import warnings

warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = train.drop(["fare_amount"]).to_pandas()
y = train["fare_amount"].to_pandas()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "regression",
    "metric": "rmse",
    "device": "cpu",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)

y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"rmse:{rmse}")
model_name = "lgbm"



## === cell 6
test_pd = test.to_pandas()
sub_pred = bst.predict(test_pd)
exp_num = "base"
submission = pd.DataFrame({"key": test_key, "fare_amount": sub_pred})
submission.to_csv(f"submission_{model_name}_{exp_num}.csv", index=False)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
LightGBMError                             Traceback (most recent call last)
/tmp/ipykernel_55/1707216365.py in <cell line: 0>()
      1 test_pd = test.to_pandas()
----> 2 sub_pred = bst.predict(test_pd)
      3 exp_num = "base"
      4 submission = pd.DataFrame({"key": test_key, "fare_amount": sub_pred})
      5 submission.to_csv(f"submission_{model_name}_{exp_num}.csv", index=False)

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in predict(self, data, start_iteration, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, validate_features, **kwargs)
   4765             else:
   4766                 num_iteration = -1
-> 4767         return predictor.predict(
   4768             data=data,
   4769             start_iteration=start_iteration,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in predict(self, data, start_iteration, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, validate_features)
   1202             )
   1203         elif isinstance(data, np.ndarray):
-> 1204             preds, nrow = self.__pred_for_np2d(
   1205                 mat=data,
   1206                 start_iteration=start_iteration,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in __pred_for_np2d(self, mat, start_iteration, num_iteration, predict_type)
   1359             return preds, nrow
   1360         else:
-> 1361             return self.__inner_predict_np2d(
   1362                 mat=mat,
   1363                 start_iteration=start_iteration,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in __inner_predict_np2d(self, mat, start_iteration, num_iteration, predict_type, preds)
   1305             raise ValueError("Wrong length of pre-allocated predict array")
   1306         out_num_preds = ctypes.c_int64(0)
-> 1307         _safe_call(
   1308             _LIB.LGBM_BoosterPredictForMat(
   1309                 self._handle,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _safe_call(ret)
    311     """
    312     if ret != 0:
--> 313         raise LightGBMError(_LIB.LGBM_GetLastError().decode("utf-8"))
    314 
    315 

LightGBMError: The number of features in data (5) is not the same as it was in training data (12).
You can set ``predict_disable_shape_check=true`` to discard this error, but please be aware what you are doing.

## === cell 7
import matplotlib.pyplot as plt

feature_importance_df = pd.DataFrame()
feature_importance_df["feature"] = X.columns
feature_importance_df["importance"] = bst.feature_importance()
feature_importance_df = feature_importance_df.sort_values(
    by="importance", ascending=False
).head(15)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["feature"], feature_importance_df["importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.show()



## === cell 8
train.head()
