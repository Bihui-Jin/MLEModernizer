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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os
import random
import pathlib

np.random.seed(42)
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

from sklearnex import patch_sklearn

patch_sklearn()

print(os.listdir("../input"))




## === cell 1
use_cols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "key",
]
dtypes = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
    "key": object,
}
train = pd.read_csv(
    "../input/train.csv",
    usecols=use_cols,
    dtype=dtypes,
    nrows=800_000,
    low_memory=False,
)
test = pd.read_csv(
    "../input/test.csv",
    usecols=[c for c in use_cols if c != "fare_amount"],
    dtype={k: v for k, v in dtypes.items() if k != "fare_amount"},
    low_memory=False,
)




## === cell 2
mask_not_na = train.notna().all(axis=1)
mask_nonzero = (train != 0).all(axis=1)
train = train.loc[mask_not_na & mask_nonzero].reset_index(drop=True)




## === cell 3
def engineer_features(df):
    dt_series = pd.to_datetime(df["pickup_datetime"], utc=True, errors="coerce")
    df["year"] = dt_series.dt.year.astype(np.int16)
    df["month"] = dt_series.dt.month.astype(np.int8)
    df["weekday"] = dt_series.dt.weekday.astype(np.int8)
    df["hour"] = dt_series.dt.hour.astype(np.int8)

    hour = df["hour"].astype(np.float32).values
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)

    month = df["month"].astype(np.float32).values
    df["month_sin"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * month / 12).astype(np.float32)

    df["dayofyear"] = dt_series.dt.dayofyear.astype(np.int16)
    doy = df["dayofyear"].astype(np.float32).values
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365).astype(np.float32)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365).astype(np.float32)

    df.drop(columns=["pickup_datetime"], inplace=True)

    lon1 = np.radians(df["pickup_longitude"].values.astype(np.float32))
    lon2 = np.radians(df["dropoff_longitude"].values.astype(np.float32))
    lat1 = np.radians(df["pickup_latitude"].values.astype(np.float32))
    lat2 = np.radians(df["dropoff_latitude"].values.astype(np.float32))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = 6371.0  # Earth radius in km
    df["haversine_km"] = (R * c).astype(np.float32)

    df["abs_lon_diff"] = np.abs(
        df["pickup_longitude"] - df["dropoff_longitude"]
    ).astype(np.float32)
    df["abs_lat_diff"] = np.abs(df["pickup_latitude"] - df["dropoff_latitude"]).astype(
        np.float32
    )

    return df


train = engineer_features(train)
test = engineer_features(test)




## === cell 4
print("Train shape after engineering:", train.shape)
print("Test shape after engineering :", test.shape)




## === cell 5
train = train[train["fare_amount"] > 0].copy()
train["fare_amount"] = np.log1p(train["fare_amount"]).astype(np.float32)

feature_cols = [c for c in train.columns if c not in ("fare_amount", "key")]
X = train[feature_cols].astype(np.float32, copy=False)
y = train["fare_amount"]




## === cell 6
from sklearn.model_selection import train_test_split

X_np = X.to_numpy(dtype=np.float32, copy=False)
y_np = y.to_numpy(dtype=np.float32, copy=False)

X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X_np, y_np, test_size=0.25, random_state=42
)




## === cell 7
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=2500,
    max_features=0.8,
    max_depth=25,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_np, y_train_np)




## === cell 8
from sklearn.metrics import mean_squared_error

val_pred_log = rf.predict(X_val_np)
val_pred = np.expm1(val_pred_log)
val_true = np.expm1(y_val_np)
rmse = np.sqrt(mean_squared_error(val_true, val_pred))
print(f"Validation RMSE (log‑target back‑transformed): {rmse:.5f}")




## === cell 9
test_features = test.drop(columns=["key"]).astype(np.float32, copy=False)
test_pred = np.expm1(rf.predict(test_features.values))

submission = pd.DataFrame({"key": test["key"], "fare_amount": test_pred})
submission.to_csv("NYCtaxiFare_prediction.csv", index=False)
print("Submission saved to NYCtaxiFare_prediction.csv")
submission.head()
