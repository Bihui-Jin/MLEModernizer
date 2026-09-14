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
import numpy as np
import pandas as pd
import xgboost as xgb

np.random.seed(0)



## === cell 1
TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"

train_usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
test_dtypes = {
    "key": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

_csv_engine = "pyarrow"
_datetime_format = "%Y-%m-%d %H:%M:%S.%f"
try:
    train = pd.read_csv(
        TRAIN_PATH,
        nrows=5_000_000,
        usecols=train_usecols,
        dtype=train_dtypes,
        engine=_csv_engine,
        parse_dates=["pickup_datetime"],
        date_format=_datetime_format,
    )
    test = pd.read_csv(
        TEST_PATH,
        usecols=test_usecols,
        dtype=test_dtypes,
        engine=_csv_engine,
        parse_dates=["pickup_datetime"],
        date_format=_datetime_format,
    )
except Exception:
    train = pd.read_csv(
        TRAIN_PATH,
        nrows=5_000_000,
        usecols=train_usecols,
        dtype=train_dtypes,
        parse_dates=["pickup_datetime"],
        infer_datetime_format=True,
    )
    test = pd.read_csv(
        TEST_PATH,
        usecols=test_usecols,
        dtype=test_dtypes,
        parse_dates=["pickup_datetime"],
        infer_datetime_format=True,
    )




## === cell 2
def handle_date(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True, cache=True)

    hour = dt.dt.hour.to_numpy(dtype=np.float32, na_value=np.nan)
    month = dt.dt.month.to_numpy(dtype=np.float32, na_value=np.nan)
    year = dt.dt.year.to_numpy(dtype=np.float32, na_value=np.nan)
    day_of_year = dt.dt.dayofyear.to_numpy(dtype=np.float32, na_value=np.nan)
    weekday = dt.dt.weekday.to_numpy(dtype=np.float32, na_value=np.nan)
    quarter = dt.dt.quarter.to_numpy(dtype=np.float32, na_value=np.nan)
    day_of_month = dt.dt.day.to_numpy(dtype=np.float32, na_value=np.nan)

    iso = dt.dt.isocalendar()
    week = iso.week.to_numpy(dtype=np.float32, na_value=np.nan)

    df["hour_of_day"] = hour
    df["week"] = week
    df["week_of_year"] = week
    df["month"] = month
    df["year"] = year
    df["day_of_year"] = day_of_year
    df["weekday"] = weekday
    df["quarter"] = quarter
    df["day_of_month"] = day_of_month

    df = df.drop("pickup_datetime", axis=1)
    return df


def handle_distance(df):
    plat = df["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
    plon = df["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)

    pickup_lat = np.radians(plat)
    pickup_lon = np.radians(plon)
    dropoff_lat = np.radians(dlat)
    dropoff_lon = np.radians(dlon)

    dlat_r = dropoff_lat - pickup_lat
    dlon_r = dropoff_lon - pickup_lon

    a = np.sin(dlat_r / 2.0) ** 2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * (
        np.sin(dlon_r / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    df["distance_travelled"] = (earth_radius_km * c).astype("float32")

    df["abs_lon_diff"] = np.abs(dlon - plon).astype("float32")
    df["abs_lat_diff"] = np.abs(dlat - plat).astype("float32")

    return df


def clean_up_train(train_df):
    train_df = train_df.dropna()

    fa = train_df["fare_amount"].to_numpy(copy=False)
    pc = train_df["passenger_count"].to_numpy(copy=False)
    plon = train_df["pickup_longitude"].to_numpy(copy=False)
    dlon = train_df["dropoff_longitude"].to_numpy(copy=False)
    plat = train_df["pickup_latitude"].to_numpy(copy=False)
    dlat = train_df["dropoff_latitude"].to_numpy(copy=False)
    dist = train_df["distance_travelled"].to_numpy(copy=False)

    m = np.ones(train_df.shape[0], dtype=bool)

    m &= fa > 0
    m &= (pc > 0) & (pc < 7)

    m &= (plon >= -180) & (plon <= 180) & (dlon >= -180) & (dlon <= 180)
    m &= (plat >= -90) & (plat <= 90) & (dlat >= -90) & (dlat <= 90)

    m &= (plon >= -74.5) & (plon <= -72.5) & (dlon >= -74.5) & (dlon <= -72.5)
    m &= (plat >= 40.0) & (plat <= 41.8) & (dlat >= 40.0) & (dlat <= 41.8)

    m &= (dist >= 0.0) & (dist <= 200.0)
    m &= (fa >= 2.5) & (fa <= 250.0)

    m &= ~((dist < 0.05) & (fa > 3.5))

    train_df = train_df.loc[m]

    d_lo, d_hi = train_df["distance_travelled"].quantile([0.001, 0.999])
    train_df = train_df[train_df["distance_travelled"].between(d_lo, d_hi)]

    q_lo, q_hi = train_df["fare_amount"].quantile([0.001, 0.999])
    train_df = train_df[train_df["fare_amount"].between(q_lo, q_hi)]

    return train_df


def get_feature_cols(test_df):
    return [c for c in test_df.columns if c != "key"]




## === cell 3
train = handle_date(train)
test = handle_date(test)

train = handle_distance(train)
test = handle_distance(test)

train = clean_up_train(train)



## === cell 4
from sklearn.model_selection import train_test_split

FEATURE_COLS = get_feature_cols(test)

X = train[FEATURE_COLS]
y = train["fare_amount"].astype("float32")

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=0
)



## === cell 5
y_train_np = y_train.to_numpy(dtype=np.float32, copy=False)
y_valid_np = y_valid.to_numpy(dtype=np.float32, copy=False)

y_train_log = np.log1p(y_train_np)
y_valid_log = np.log1p(y_valid_np)

lo, hi = np.quantile(y_train_log, [0.001, 0.999])
mask = (y_train_log >= lo) & (y_train_log <= hi)

X_train_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.float32, copy=False))
X_valid_np = np.ascontiguousarray(X_valid.to_numpy(dtype=np.float32, copy=False))

X_train_f = X_train_np[mask]
y_train_log_f = y_train_log[mask]
y_valid_log_np = y_valid_log

base_params = dict(
    max_depth=4,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0,
    tree_method="hist",
    min_child_weight=5,
    reg_lambda=1.0,
    n_jobs=-1,
    verbosity=0,
)

dtrain_es = xgb.QuantileDMatrix(X_train_f, label=y_train_log_f)
dvalid_es = xgb.QuantileDMatrix(X_valid_np, label=y_valid_log_np, ref=dtrain_es)

model_es = xgb.train(
    params=base_params,
    dtrain=dtrain_es,
    num_boost_round=4000,
    evals=[(dvalid_es, "validation")],
    early_stopping_rounds=80,
    verbose_eval=False,
)

best_n = int(getattr(model_es, "best_iteration", 0))
best_n = max(best_n, 100)

y_np = y.to_numpy(dtype=np.float32, copy=False)
y_full_log = np.log1p(y_np)
lo_full, hi_full = np.quantile(y_full_log, [0.001, 0.999])
mask_full = (y_full_log >= lo_full) & (y_full_log <= hi_full)

X_full_np = np.ascontiguousarray(X.to_numpy(dtype=np.float32, copy=False))
X_full_f = X_full_np[mask_full]
y_full_log_f = y_full_log[mask_full]

dtrain_full = xgb.QuantileDMatrix(X_full_f, label=y_full_log_f)

model = xgb.train(
    params=base_params,
    dtrain=dtrain_full,
    num_boost_round=best_n,
    evals=[],
    verbose_eval=False,
)

test_np = np.ascontiguousarray(
    test[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
)
dtest = xgb.DMatrix(test_np)
pred_log = model.predict(dtest)
pred = np.expm1(pred_log)
pred = np.clip(pred, 0, None)

submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": pred}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)
print(submission.head(20))
