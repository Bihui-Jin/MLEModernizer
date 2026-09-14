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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"


def read_train_multi_chunks(path, total_rows=500_000, seed=42):
    rng = np.random.default_rng(seed)

    n_chunks = 5
    rows_per = total_rows // n_chunks

    with open(path, "rb") as f:
        n_lines = sum(1 for _ in f)
    n_rows = n_lines - 1  # excluding header

    max_start = max(1, n_rows - rows_per - 1)
    starts = rng.integers(low=1, high=max_start, size=n_chunks)

    parts = []
    for s in starts:
        part = pd.read_csv(
            path,
            skiprows=range(1, int(s) + 1),
            nrows=rows_per,
            index_col="key",
        )
        parts.append(part)

    df = pd.concat(parts, axis=0)
    df = df.sample(frac=1.0, random_state=seed)
    return df


dataset_train = read_train_multi_chunks(train_iop_path, total_rows=500_000, seed=42)
dataset_test = pd.read_csv(test_iop_path, nrows=10_000, index_col="key")



## === cell 1
print("dataset_train old size", len(dataset_train))

dataset_train = dataset_train.dropna(
    subset=[
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
)

dataset_train = dataset_train[
    (dataset_train["fare_amount"] > 0) & (dataset_train["fare_amount"] < 250)
]

num_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
for c in num_cols:
    dataset_train[c] = pd.to_numeric(dataset_train[c], errors="coerce")

dataset_train["passenger_count"] = dataset_train["passenger_count"].fillna(1).clip(1, 6)

for c, lo, hi in [
    ("pickup_longitude", -74.5, -72.8),
    ("dropoff_longitude", -74.5, -72.8),
    ("pickup_latitude", 40.5, 41.8),
    ("dropoff_latitude", 40.5, 41.8),
]:
    med = pd.to_numeric(dataset_train[c], errors="coerce").median()
    dataset_train[c] = dataset_train[c].fillna(med).clip(lo, hi)

print("new size", len(dataset_train))
dataset_train.head(5)



## === cell 2
print("dataset_test old size", len(dataset_test))

print("new size (kept unchanged for submission)", len(dataset_test))
dataset_test.head(5)




## === cell 3
def get_year(pickup_date):
    return pickup_date.year


def get_month(pickup_date):
    return pickup_date.month


def get_day(pickup_date):
    return pickup_date.day


def get_hour(pickup_date):
    return pickup_date.hour




## === cell 4
from datetime import datetime as dt
import warnings


def _haversine_km(lon1, lat1, lon2, lat2):
    """
    Keep same core feature `dis`, but compute it as haversine distance in km for better physical meaning.
    """
    lon1 = np.radians(lon1.astype(np.float64))
    lat1 = np.radians(lat1.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    R = 6371.0
    return R * c


def preparedataset2(datasetname: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized datetime parsing + distance features. Core feature set unchanged.
    """
    warnings.filterwarnings("ignore")

    ds = datasetname.copy()

    ds["pickup_datetime"] = pd.to_datetime(
        ds["pickup_datetime"], errors="coerce", utc=True
    )
    ds["pickup_year"] = ds["pickup_datetime"].dt.year
    ds["pickup_month"] = ds["pickup_datetime"].dt.month
    ds["pickup_day"] = ds["pickup_datetime"].dt.day
    ds["pickup_hour"] = ds["pickup_datetime"].dt.hour

    ds["x_dis"] = ds["dropoff_longitude"] - ds["pickup_longitude"]
    ds["y_dis"] = ds["dropoff_latitude"] - ds["pickup_latitude"]

    ds["dis"] = _haversine_km(
        ds["pickup_longitude"].to_numpy(),
        ds["pickup_latitude"].to_numpy(),
        ds["dropoff_longitude"].to_numpy(),
        ds["dropoff_latitude"].to_numpy(),
    )

    ds = ds.drop(
        [
            "pickup_datetime",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
            "pickup_latitude",
        ],
        axis=1,
    )

    return ds




## === cell 5
from datetime import datetime as dt
import warnings

warnings.filterwarnings("ignore")


def preparedataset(datasetname):
    datasetname["pickup_year"] = 0
    datasetname["pickup_month"] = 0
    datasetname["pickup_day"] = 0
    datasetname["pickup_hour"] = 0
    datasetname["dis"] = 0
    datasetname["x_dis"] = 0
    datasetname["y_dis"] = 0

    datasetname.head()

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




## === cell 6
df = preparedataset2(dataset_train)

df = df.dropna()

if "dis" in df.columns:
    df = df[df["dis"].between(0.2, 40)].copy()

if "passenger_count" in df.columns:
    df = df[df["passenger_count"].between(1, 6)].copy()

if "dis" in df.columns and "fare_amount" in df.columns:
    d = df["dis"].to_numpy(dtype=np.float64)
    f = df["fare_amount"].to_numpy(dtype=np.float64)
    fpk = f / np.maximum(d, 0.2)
    mask = (fpk >= 1.0) & (fpk <= 60.0)
    df = df.loc[mask].copy()

df.head(5)



## === cell 7
test_df_raw = dataset_test.copy()

num_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
for c in num_cols:
    test_df_raw[c] = pd.to_numeric(test_df_raw[c], errors="coerce")

test_df_raw["passenger_count"] = test_df_raw["passenger_count"].fillna(1).clip(1, 6)

for c, lo, hi in [
    ("pickup_longitude", -74.5, -72.8),
    ("dropoff_longitude", -74.5, -72.8),
    ("pickup_latitude", 40.5, 41.8),
    ("dropoff_latitude", 40.5, 41.8),
]:
    med = pd.to_numeric(dataset_train[c], errors="coerce").median()
    test_df_raw[c] = test_df_raw[c].fillna(med).clip(lo, hi)

test_df = preparedataset2(test_df_raw)

for c in ["pickup_year", "pickup_month", "pickup_day", "pickup_hour"]:
    if c in test_df.columns:
        fill_val = pd.to_numeric(df[c], errors="coerce").median()
        test_df[c] = test_df[c].fillna(fill_val)

test_df.head(5)



## === cell 8
from sklearn.model_selection import train_test_split

y = df.fare_amount
y_log = np.log1p(y)

X = df.drop("fare_amount", axis=1)

test_df = test_df.reindex(columns=X.columns)
for c in X.columns:
    if test_df[c].isna().any():
        test_df[c] = test_df[c].fillna(pd.to_numeric(X[c], errors="coerce").median())

_dt = pd.to_datetime(
    dataset_train.loc[df.index, "pickup_datetime"], errors="coerce", utc=True
)
order = _dt.sort_values().index

X_sorted = X.loc[order]
y_sorted_log = y_log.loc[order]
y_sorted = y.loc[order]

split = int(len(X_sorted) * 0.8)
X_train, X_valid = X_sorted.iloc[:split], X_sorted.iloc[split:]
y_train_log, y_valid_log = y_sorted_log.iloc[:split], y_sorted_log.iloc[split:]
y_valid = y_sorted.iloc[split:]



## === cell 9
import warnings

warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

result = {}
best_params = None
best_rmse = 1e18

for lr in [0.10, 0.15, 0.20]:
    for ns in [500, 750, 1000]:
        for md in [6, 8]:
            for subs in [0.8, 1.0]:
                for col in [0.8, 1.0]:
                    for mcw in [5, 10]:
                        for ra in [0.0, 0.1]:
                            my_model = XGBRegressor(
                                n_estimators=ns,
                                learning_rate=lr,
                                max_depth=md,
                                subsample=subs,
                                colsample_bytree=col,
                                min_child_weight=mcw,
                                reg_alpha=ra,
                                reg_lambda=1.0,
                                objective="reg:squarederror",
                                eval_metric="rmse",
                                n_jobs=4,
                                random_state=42,
                            )
                            my_model.fit(X_train, y_train_log)

                            pred_valid = np.expm1(my_model.predict(X_valid))
                            pred_valid = np.clip(pred_valid, 0, 250)
                            rmse = mean_squared_error(
                                y_valid, pred_valid, squared=False
                            )

                            params_key = (ns, lr, md, subs, col, mcw, ra)
                            result[params_key] = rmse

                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_params = {
                                    "n_estimators": ns,
                                    "learning_rate": lr,
                                    "max_depth": md,
                                    "subsample": subs,
                                    "colsample_bytree": col,
                                    "min_child_weight": mcw,
                                    "reg_alpha": ra,
                                }
                                print("better found")
                                print(best_params, "rmse:", rmse)

my_model_2 = XGBRegressor(
    n_estimators=best_params["n_estimators"],
    learning_rate=best_params["learning_rate"],
    max_depth=best_params["max_depth"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    min_child_weight=best_params["min_child_weight"],
    reg_alpha=best_params["reg_alpha"],
    reg_lambda=1.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    n_jobs=4,
    random_state=42,
)
my_model_2.fit(X_train, y_train_log)
predictions_2 = np.expm1(my_model_2.predict(X_valid))
predictions_2 = np.clip(predictions_2, 0, 250)
rmse_2 = mean_squared_error(y_valid, predictions_2, squared=False)

print("best_params:", best_params)
print(
    "Validation RMSE (best model fit on train fold, evaluated on valid fold):", rmse_2
)



## === cell 10
from xgboost import XGBRegressor

final_model = XGBRegressor(
    n_estimators=best_params["n_estimators"],
    learning_rate=best_params["learning_rate"],
    max_depth=best_params["max_depth"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    min_child_weight=best_params["min_child_weight"],
    reg_alpha=best_params["reg_alpha"],
    reg_lambda=1.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    n_jobs=4,
    random_state=42,
)

final_model.fit(X, y_log)

test_preds = np.expm1(final_model.predict(test_df))
test_preds = np.clip(test_preds, 0, 250)

output = pd.DataFrame({"key": dataset_test.index, "fare_amount": test_preds})
output.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", output.shape)
print(output.head())
