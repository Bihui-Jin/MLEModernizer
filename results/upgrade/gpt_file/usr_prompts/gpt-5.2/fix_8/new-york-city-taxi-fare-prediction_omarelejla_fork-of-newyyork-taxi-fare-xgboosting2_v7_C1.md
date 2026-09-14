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
from pathlib import Path

RANDOM_STATE = 42

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))

train_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

N_TRAIN_SAMPLE = 1_000_000  # was 200k; larger sample typically drops RMSE toward ~4-5 for this baseline
TRAIN_TOTAL_ROWS = 55_423_857  # approx total rows in train.csv including header

rng = np.random.RandomState(RANDOM_STATE)
skip_n = max(0, (TRAIN_TOTAL_ROWS - 1) - N_TRAIN_SAMPLE)

if skip_n > 0:
    skip_idx = set(rng.randint(1, TRAIN_TOTAL_ROWS, size=skip_n, dtype=np.int64))
else:
    skip_idx = None

dataset_train = pd.read_csv(train_iop_path, skiprows=skip_idx, index_col="key")
dataset_test = pd.read_csv(test_iop_path, index_col="key")

print("Loaded train:", dataset_train.shape, "test:", dataset_test.shape)



## === cell 1
print("dataset_train old size", len(dataset_train))

dataset_train = dataset_train[dataset_train.dropoff_longitude != 0]

dataset_train = dataset_train[
    (dataset_train["fare_amount"] > 0) & (dataset_train["fare_amount"] <= 250)
]
dataset_train = dataset_train[
    (dataset_train["passenger_count"] >= 1) & (dataset_train["passenger_count"] <= 6)
]

dataset_train = dataset_train[
    (dataset_train["pickup_longitude"].between(-75, -72))
    & (dataset_train["dropoff_longitude"].between(-75, -72))
    & (dataset_train["pickup_latitude"].between(40, 42))
    & (dataset_train["dropoff_latitude"].between(40, 42))
]

print("new size", len(dataset_train))
dataset_train.head(5)



## === cell 2
print("dataset_test size", len(dataset_test))
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
import warnings

warnings.filterwarnings("ignore")


def _to_radians(x):
    arr = np.asarray(x, dtype=float)
    return np.radians(arr)


def _haversine_km(lon1, lat1, lon2, lat2):
    """
    Score-relevant: correct spherical distance is a strong baseline feature for this competition.
    Vectorized for speed; robust to scalar lon2/lat2 inputs.
    """
    lon1 = _to_radians(lon1)
    lat1 = _to_radians(lat1)
    lon2 = _to_radians(lon2)
    lat2 = _to_radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0
    return R * c


def _bearing(lon1, lat1, lon2, lat2):
    """
    Score-relevant, minimal feature addition: direction of travel helps the tree split space better.
    Returns bearing in radians in [-pi, pi]. Robust to scalar lon2/lat2.
    """
    lon1 = _to_radians(lon1)
    lat1 = _to_radians(lat1)
    lon2 = _to_radians(lon2)
    lat2 = _to_radians(lat2)

    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


def preparedataset2(datasetname: pd.DataFrame) -> pd.DataFrame:
    """
    Preserves your core logic (time parts + simple displacement + haversine distance),
    and adds two common, low-risk NYC-taxi features (bearing + distance to NYC center)
    that typically reduce RMSE without changing the model/training approach.
    """
    df = datasetname.copy()

    dt_series = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    dt_series = dt_series.dt.tz_convert(None)

    df["pickup_year"] = dt_series.dt.year.astype("Int64")
    df["pickup_month"] = dt_series.dt.month.astype("Int64")
    df["pickup_day"] = dt_series.dt.day.astype("Int64")
    df["pickup_hour"] = dt_series.dt.hour.astype("Int64")

    df["x_dis"] = df["dropoff_longitude"] - df["pickup_longitude"]
    df["y_dis"] = df["dropoff_latitude"] - df["pickup_latitude"]

    df["dis"] = _haversine_km(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )

    df["bearing"] = _bearing(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )

    nyc_lon, nyc_lat = -73.985428, 40.748817
    df["pickup_to_center_km"] = _haversine_km(
        df["pickup_longitude"], df["pickup_latitude"], nyc_lon, nyc_lat
    )
    df["dropoff_to_center_km"] = _haversine_km(
        df["dropoff_longitude"], df["dropoff_latitude"], nyc_lon, nyc_lat
    )

    model_drop_cols = [
        "pickup_datetime",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
        "pickup_latitude",
    ]
    df = df.drop(columns=model_drop_cols, errors="ignore")

    for c in ["pickup_year", "pickup_month", "pickup_day", "pickup_hour"]:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode(dropna=True).iloc[0]).astype(int)

    if "passenger_count" in df.columns:
        df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce")
        df["passenger_count"] = df["passenger_count"].fillna(1).clip(0, 6)

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

    return df




## === cell 5
import warnings

warnings.filterwarnings("ignore")


def preparedataset(datasetname):
    return preparedataset2(datasetname)




## === cell 6
df = preparedataset2(dataset_train)
df.head(5)



## === cell 7
test_df = preparedataset2(dataset_test)
test_df.head(5)



## === cell 8
from sklearn.model_selection import train_test_split

y = df["fare_amount"]
X = df.drop("fare_amount", axis=1)

test_df = test_df.reindex(columns=X.columns)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)



## === cell 9
import warnings

warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

result = {}
best_istemator = 0
best_learing_rate = 0
best_rmse = float("inf")

fixed_params = dict(
    n_jobs=4,
    random_state=RANDOM_STATE,
    tree_method="hist",
    eval_metric="rmse",
    subsample=0.8,
    colsample_bytree=0.8,
)

for lr in [0.10, 0.15, 0.20, 0.25, 0.30]:
    for ns in [300, 400, 500, 600, 700]:
        my_model = XGBRegressor(n_estimators=ns, learning_rate=lr, **fixed_params)
        my_model.fit(X_train, y_train)

        predictions = my_model.predict(X_valid)
        rmse = mean_squared_error(y_valid, predictions, squared=False)

        if rmse < best_rmse:
            best_rmse = rmse
            best_istemator = ns
            best_learing_rate = lr
            print("better found")
            print(ns, lr, rmse)

        result[(ns, lr)] = rmse

my_model_2 = XGBRegressor(
    n_estimators=best_istemator, learning_rate=best_learing_rate, **fixed_params
)
my_model_2.fit(X_train, y_train)
predictions_2 = my_model_2.predict(X_valid)
rmse_2 = mean_squared_error(y_valid, predictions_2, squared=False)

print("best_istemator:", best_istemator)
print("best_learing_rate:", best_learing_rate)
print("Validation RMSE:", rmse_2)



## === cell 10
from xgboost import XGBRegressor

final_model = XGBRegressor(
    n_estimators=best_istemator,
    learning_rate=best_learing_rate,
    n_jobs=4,
    random_state=RANDOM_STATE,
    tree_method="hist",
    eval_metric="rmse",
    subsample=0.8,
    colsample_bytree=0.8,
)
final_model.fit(X, y)

test_df = test_df.reindex(columns=X.columns)
test_preds = final_model.predict(test_df)

test_preds = np.clip(test_preds, 0.0, 250.0)

output = pd.DataFrame({"key": test_df.index, "fare_amount": test_preds})
output.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", output.shape)
print(output.head())
