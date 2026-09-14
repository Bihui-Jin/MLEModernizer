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

4.377

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

import matplotlib.pyplot as plt
import seaborn as sns
import plotly


import plotly.offline as offline
import plotly.graph_objs as go

offline.init_notebook_mode()

import datetime as dt
import lightgbm as lgbm

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import io
import os
import gc



## === cell 1
BASE_DIRS = [
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/data/new-york-city-taxi-fare-prediction",
    "/kaggle/input/new-york-city-taxi-fare-prediction",
]


def find_file(filename: str) -> str:
    for d in BASE_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    if os.path.exists(filename):
        return filename
    raise FileNotFoundError(f"Could not find {filename} in {BASE_DIRS} or CWD")


train_path = find_file("train.csv")
test_path = find_file("test.csv")
sample_path = find_file("sample_submission.csv")

train_path, test_path, sample_path



## === cell 2
usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

NROWS = 2_000_000

train_df = pd.read_csv(
    train_path,
    usecols=usecols,
    nrows=NROWS,
    parse_dates=["pickup_datetime"],
)

test_df = pd.read_csv(
    test_path,
    parse_dates=["pickup_datetime"],
)

train_df.shape, test_df.shape




## === cell 3
def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().copy()
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 300)]
    df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)]

    df = df[
        (df["pickup_longitude"].between(-75, -72))
        & (df["dropoff_longitude"].between(-75, -72))
        & (df["pickup_latitude"].between(40, 42))
        & (df["dropoff_latitude"].between(40, 42))
    ]
    return df


train_df = clean_train(train_df)
train_df.shape




## === cell 4
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0088 * c
    return km


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dtcol = df["pickup_datetime"]
    df["pickup_year"] = dtcol.dt.year.astype(np.int16)
    df["pickup_month"] = dtcol.dt.month.astype(np.int8)
    df["pickup_day"] = dtcol.dt.day.astype(np.int8)
    df["pickup_hour"] = dtcol.dt.hour.astype(np.int8)
    df["pickup_weekday"] = dtcol.dt.weekday.astype(np.int8)

    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()

    df["haversine_km"] = haversine_np(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    ).astype(np.float32)

    df["manhattan_km"] = (df["abs_lon_diff"] + df["abs_lat_diff"]) * 111.0

    df = df.drop(columns=["pickup_datetime"])
    return df


train_feat = add_features(train_df)
test_feat = add_features(test_df)

target = train_feat["fare_amount"].astype(np.float32)
train_feat = train_feat.drop(columns=["fare_amount"])

feature_cols = train_feat.columns.tolist()
len(feature_cols), feature_cols[:10]



## === cell 5
X_train, X_valid, y_train, y_valid = train_test_split(
    train_feat, target, test_size=0.1, random_state=42
)

model = lgbm.LGBMRegressor(
    objective="regression",
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

valid_pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, valid_pred, squared=False)
rmse



## === cell 6
model.fit(train_feat, target)
test_pred = model.predict(test_feat)

test_pred = np.clip(test_pred, 0, None)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1014337210.py in <cell line: 0>()
      1 # Train on full sampled training data and predict test
      2 model.fit(train_feat, target)
----> 3 test_pred = model.predict(test_feat)
      4 
      5 # CHANGE (score/stability): clip negatives; fares can't be negative and negatives harm RMSE.

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in predict(self, X, raw_score, start_iteration, num_iteration, pred_leaf, pred_contrib, validate_features, **kwargs)
   1142         predict_params["num_threads"] = self._process_n_jobs(predict_params["num_threads"])
   1143 
-> 1144         return self._Booster.predict(  # type: ignore[union-attr]
   1145             X,
   1146             raw_score=raw_score,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in predict(self, data, start_iteration, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, validate_features, **kwargs)
   4765             else:
   4766                 num_iteration = -1
-> 4767         return predictor.predict(
   4768             data=data,
   4769             start_iteration=start_iteration,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in predict(self, data, start_iteration, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, validate_features)
   1156 
   1157         if isinstance(data, pd_DataFrame):
-> 1158             data = _data_from_pandas(
   1159                 data=data,
   1160                 feature_name="auto",

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _data_from_pandas(data, feature_name, categorical_feature, pandas_categorical)
    866 
    867     return (
--> 868         _pandas_to_numpy(data, target_dtype=target_dtype),
    869         feature_name,
    870         categorical_feature,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _pandas_to_numpy(data, target_dtype)
    812     target_dtype: "np.typing.DTypeLike",
    813 ) -> np.ndarray:
--> 814     _check_for_bad_pandas_dtypes(data.dtypes)
    815     try:
    816         # most common case (no nullable dtypes)

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _check_for_bad_pandas_dtypes(pandas_dtypes_series)
    803     ]
    804     if bad_pandas_dtypes:
--> 805         raise ValueError(
    806             f"pandas dtypes must be int, float or bool.\nFields with bad pandas dtypes: {', '.join(bad_pandas_dtypes)}"
    807         )

ValueError: pandas dtypes must be int, float or bool.
Fields with bad pandas dtypes: key: object

## === cell 7
sub = pd.read_csv(sample_path)
sub["fare_amount"] = test_pred.astype(np.float32)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

sub.head(), out_path, sub.shape

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/239593492.py in <cell line: 0>()
      2 sub = pd.read_csv(sample_path)
      3 # Ensure alignment: test has same order as sample_submission in this competition.
----> 4 sub["fare_amount"] = test_pred.astype(np.float32)
      5 
      6 # Write submission

NameError: name 'test_pred' is not defined
