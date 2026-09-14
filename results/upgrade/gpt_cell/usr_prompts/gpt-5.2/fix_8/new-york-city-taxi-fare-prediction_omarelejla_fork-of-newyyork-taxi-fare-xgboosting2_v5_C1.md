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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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
import sys
import subprocess

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "protobuf>=3.20.3,<5"]
)

import tensorflow as tf

print(tf.__version__)



## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
)  # kept (not used) to preserve core structure

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

TRAIN_NROWS = 200_000
RANDOM_SEED = 42

USECOLS = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
DTYPES = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

TOTAL_TRAIN_ROWS = 55_423_856  # as provided in prompt
rng = np.random.RandomState(RANDOM_SEED)

n_blocks = 8
block_size = max(
    50_000, (TRAIN_NROWS * 2) // n_blocks
)  # oversample then downsample deterministically
max_start = max(1, TOTAL_TRAIN_ROWS - block_size - 1)

blocks = []
for _ in range(n_blocks):
    start = int(rng.randint(1, max_start))  # skip header row (0); data rows start at 1
    skip = range(1, start + 1)
    df_block = pd.read_csv(
        train_iop_path,
        usecols=USECOLS,
        dtype=DTYPES,
        nrows=block_size,
        skiprows=skip,
    )
    blocks.append(df_block)

dataset_train = pd.concat(blocks, axis=0, ignore_index=True)
dataset_train = dataset_train.drop_duplicates(subset=["key"], keep="first").set_index(
    "key"
)

if len(dataset_train) > TRAIN_NROWS:
    dataset_train = dataset_train.sample(n=TRAIN_NROWS, random_state=RANDOM_SEED)
elif len(dataset_train) < TRAIN_NROWS:
    need = TRAIN_NROWS - len(dataset_train)
    topup = pd.read_csv(
        train_iop_path,
        usecols=USECOLS,
        dtype=DTYPES,
        nrows=need,
    ).set_index("key")
    dataset_train = pd.concat([dataset_train, topup], axis=0)
    dataset_train = dataset_train[~dataset_train.index.duplicated(keep="first")]
    if len(dataset_train) > TRAIN_NROWS:
        dataset_train = dataset_train.sample(n=TRAIN_NROWS, random_state=RANDOM_SEED)

dataset_test = pd.read_csv(
    test_iop_path,
    nrows=10000,
    index_col="key",
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int8",
    },
)

print("train shape:", dataset_train.shape)
print("test shape:", dataset_test.shape)



## === cell 2
print("dataset_train old size", len(dataset_train))

dataset_train = dataset_train.dropna()

dataset_train = dataset_train[
    (dataset_train["fare_amount"] > 0) & (dataset_train["fare_amount"] < 500)
]
dataset_train = dataset_train[
    (dataset_train["passenger_count"] > 0) & (dataset_train["passenger_count"] <= 6)
]

dataset_train = dataset_train[
    (dataset_train["pickup_longitude"].between(-74.5, -72.8))
    & (dataset_train["dropoff_longitude"].between(-74.5, -72.8))
    & (dataset_train["pickup_latitude"].between(40.5, 41.8))
    & (dataset_train["dropoff_latitude"].between(40.5, 41.8))
]

dataset_train = dataset_train[dataset_train.dropoff_longitude != 0]

print("dataset_train new size", len(dataset_train))
dataset_train.head(5)



## === cell 3
print("dataset_test size (kept intact for submission alignment)", len(dataset_test))
dataset_test.head(5)




## === cell 4
def get_year(pickup_date):
    return pickup_date.year


def get_month(pickup_date):
    return pickup_date.month


def get_day(pickup_date):
    return pickup_date.day


def get_hour(pickup_date):
    return pickup_date.hour




## === cell 5
from datetime import datetime as dt
import warnings


def preparedataset2(datasetname):
    warnings.filterwarnings("ignore")

    df = datasetname.copy()

    dt_series = pd.to_datetime(
        df["pickup_datetime"].astype(str).str.replace("UTC", "", regex=False),
        errors="coerce",
    )

    df["pickup_year"] = dt_series.dt.year.fillna(0).astype(np.int16)
    df["pickup_month"] = dt_series.dt.month.fillna(0).astype(np.int8)
    df["pickup_day"] = dt_series.dt.day.fillna(0).astype(np.int8)
    df["pickup_hour"] = dt_series.dt.hour.fillna(0).astype(np.int8)

    df["x_dis"] = df["dropoff_longitude"] - df["pickup_longitude"]
    df["y_dis"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["dis"] = np.sqrt(
        (df["dropoff_longitude"] - df["pickup_longitude"]) ** 2
        + (df["dropoff_latitude"] - df["pickup_latitude"]) ** 2
    )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in [
        "pickup_year",
        "pickup_month",
        "pickup_day",
        "pickup_hour",
        "x_dis",
        "y_dis",
        "dis",
    ]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    df = df.drop(["pickup_datetime"], axis=1)
    df = df.drop(["pickup_longitude"], axis=1)
    df = df.drop(["dropoff_latitude"], axis=1)
    df = df.drop(["dropoff_longitude"], axis=1)
    df = df.drop(["pickup_latitude"], axis=1)

    return df




## === cell 6
from datetime import datetime as dt
import warnings

warnings.filterwarnings("ignore")


def preparedataset(datasetname):
    datasetname = datasetname.copy()
    datasetname["pickup_year"] = 0
    datasetname["pickup_month"] = 0
    datasetname["pickup_day"] = 0
    datasetname["pickup_hour"] = 0
    datasetname["dis"] = 0
    datasetname["x_dis"] = 0
    datasetname["y_dis"] = 0

    for k in range(len(datasetname.index)):
        datetime = dt.strptime(
            datasetname["pickup_datetime"][k].replace("UTC", ""), "%Y-%m-%d %H:%M:%S "
        )
        datasetname["pickup_year"][k] = datetime.year
        datasetname["pickup_month"][k] = datetime.month
        datasetname["pickup_day"][k] = datetime.day
        datasetname["pickup_hour"][k] = datetime.hour

    datasetname["x_dis"] = (
        datasetname["dropoff_longitude"] - datasetname["pickup_longitude"]
    )
    datasetname["y_dis"] = (
        datasetname["dropoff_latitude"] - datasetname["pickup_latitude"]
    )
    datasetname["dis"] = (
        (datasetname["dropoff_longitude"] - datasetname["pickup_longitude"]) ** 2
        + (datasetname["dropoff_latitude"] - datasetname["pickup_latitude"]) ** 2
    ) ** 0.5
    datasetname = datasetname.drop(["pickup_datetime"], axis=1)
    datasetname = datasetname.drop(["pickup_longitude"], axis=1)
    datasetname = datasetname.drop(["dropoff_latitude"], axis=1)
    datasetname = datasetname.drop(["dropoff_longitude"], axis=1)
    datasetname = datasetname.drop(["pickup_latitude"], axis=1)

    return datasetname




## === cell 7
df = preparedataset2(dataset_train)
df.head(5)



## === cell 8
test_df = preparedataset2(dataset_test)
test_df.head(5)



## === cell 9
from sklearn.model_selection import train_test_split

y = df.fare_amount
X = df.drop("fare_amount", axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, random_state=42, test_size=0.2
)



## === cell 10
import warnings

warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

result = {}
best_istemator = 0
best_learing_rate = 0
best_rmse = 1e18

best_max_depth = 0
best_subsample = 0
best_colsample = 0
best_min_child_weight = 0

N_JOBS = max(1, (os.cpu_count() or 4) - 1)

from joblib import Parallel, delayed

lrs = [
    x / 100 for x in range(15, 41, 5)
]  # 0.15..0.40 (slightly narrower for stability)
nss = list(range(300, 651, 50))  # 300..650
max_depths = [6, 8]
subsamples = [0.8, 1.0]
colsamples = [0.8, 1.0]
min_child_weights = [1, 5]

param_grid = [
    (lr, ns, md, ss, cs, mcw)
    for lr in lrs
    for ns in nss
    for md in max_depths
    for ss in subsamples
    for cs in colsamples
    for mcw in min_child_weights
]


def _fit_eval_one(lr, ns, md, ss, cs, mcw):
    model = XGBRegressor(
        n_estimators=ns,
        learning_rate=lr,
        max_depth=md,
        subsample=ss,
        colsample_bytree=cs,
        min_child_weight=mcw,
        n_jobs=1,  # avoid nested parallelism when joblib is parallelizing
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    return (ns, lr, md, ss, cs, mcw, rmse)


evaluations = Parallel(n_jobs=N_JOBS, backend="threading", prefer="threads")(
    delayed(_fit_eval_one)(lr, ns, md, ss, cs, mcw)
    for (lr, ns, md, ss, cs, mcw) in param_grid
)

for ns, lr, md, ss, cs, mcw, rmse in evaluations:
    result[(ns, lr, md, ss, cs, mcw)] = rmse
    if rmse < best_rmse:
        best_rmse = rmse
        best_istemator = ns
        best_learing_rate = lr
        best_max_depth = md
        best_subsample = ss
        best_colsample = cs
        best_min_child_weight = mcw
        print("better found")
        print(ns, lr, md, ss, cs, mcw, rmse)

my_model_2 = XGBRegressor(
    n_estimators=best_istemator,
    learning_rate=best_learing_rate,
    max_depth=best_max_depth,
    subsample=best_subsample,
    colsample_bytree=best_colsample,
    min_child_weight=best_min_child_weight,
    n_jobs=N_JOBS,
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist",
)
my_model_2.fit(X_train, y_train)

predictions_2 = my_model_2.predict(X_valid)
rmse_2 = mean_squared_error(y_valid, predictions_2, squared=False)

print("best_istemator:", best_istemator)
print("best_learing_rate:", best_learing_rate)
print("best_max_depth:", best_max_depth)
print("best_subsample:", best_subsample)
print("best_colsample:", best_colsample)
print("best_min_child_weight:", best_min_child_weight)
print("Validation RMSE:", rmse_2)



## === cell 11
from xgboost import XGBRegressor

my_model_2 = XGBRegressor(
    n_estimators=best_istemator if best_istemator else 650,
    learning_rate=best_learing_rate if best_learing_rate else 0.2,
    max_depth=best_max_depth if best_max_depth else 8,
    subsample=best_subsample if best_subsample else 0.8,
    colsample_bytree=best_colsample if best_colsample else 0.8,
    min_child_weight=best_min_child_weight if best_min_child_weight else 1,
    n_jobs=N_JOBS,
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist",
)
my_model_2.fit(X, y)

test_preds = my_model_2.predict(test_df)

output = pd.DataFrame({"key": test_df.index, "fare_amount": test_preds})
output.to_csv("submission.csv", index=False)

print("Wrote submission.csv with rows:", len(output))
print(output.head())
