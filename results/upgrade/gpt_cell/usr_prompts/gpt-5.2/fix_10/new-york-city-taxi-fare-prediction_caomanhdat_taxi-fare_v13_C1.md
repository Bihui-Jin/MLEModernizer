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
import numpy as np
import pandas as pd

import os
import xgboost as xgb

print(os.listdir("../input"))



## === cell 1
train = pd.read_csv("../input/train.csv", nrows=5_000_000)
test = pd.read_csv("../input/test.csv")



## === cell 2
train.head()



## === cell 3
test.head()



## === cell 4
train.describe()



## === cell 5
test.describe()



## === cell 6
train.isnull().sum()



## === cell 7
test.isnull().sum()




## === cell 8
def handle_date(df):
    s = df["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
    df["pickup_datetime"] = pd.to_datetime(
        s, errors="coerce", infer_datetime_format=True
    )

    df["hour_of_day"] = df["pickup_datetime"].dt.hour.astype("float32")

    iso_week = df["pickup_datetime"].dt.isocalendar().week.astype("float32")
    df["week"] = iso_week
    df["week_of_year"] = iso_week

    df["month"] = df["pickup_datetime"].dt.month.astype("float32")
    df["year"] = df["pickup_datetime"].dt.year.astype("float32")
    df["day_of_year"] = df["pickup_datetime"].dt.dayofyear.astype("float32")
    df["weekday"] = df["pickup_datetime"].dt.weekday.astype("float32")
    df["quarter"] = df["pickup_datetime"].dt.quarter.astype("float32")
    df["day_of_month"] = df["pickup_datetime"].dt.day.astype("float32")

    df = df.drop("pickup_datetime", axis=1)
    return df


def handle_distance(df):
    pickup_lat = np.radians(df["pickup_latitude"].astype(float))
    pickup_lon = np.radians(df["pickup_longitude"].astype(float))
    dropoff_lat = np.radians(df["dropoff_latitude"].astype(float))
    dropoff_lon = np.radians(df["dropoff_longitude"].astype(float))

    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon

    a = np.sin(dlat / 2.0) ** 2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    df["distance_travelled"] = (earth_radius_km * c).astype("float32")

    df["abs_lon_diff"] = (
        df["dropoff_longitude"].astype("float32")
        - df["pickup_longitude"].astype("float32")
    ).abs()
    df["abs_lat_diff"] = (
        df["dropoff_latitude"].astype("float32")
        - df["pickup_latitude"].astype("float32")
    ).abs()

    return df


def clean_up_train(train_df):
    train_df = train_df.dropna()

    train_df = train_df[train_df["fare_amount"] > 0]
    train_df = train_df[train_df["passenger_count"] > 0]
    train_df = train_df[train_df["passenger_count"] < 7]

    train_df = train_df[train_df["pickup_longitude"].between(-180, 180)]
    train_df = train_df[train_df["dropoff_longitude"].between(-180, 180)]
    train_df = train_df[train_df["pickup_latitude"].between(-90, 90)]
    train_df = train_df[train_df["dropoff_latitude"].between(-90, 90)]

    train_df = train_df[train_df["pickup_longitude"].between(-74.5, -72.5)]
    train_df = train_df[train_df["dropoff_longitude"].between(-74.5, -72.5)]
    train_df = train_df[train_df["pickup_latitude"].between(40.0, 41.8)]
    train_df = train_df[train_df["dropoff_latitude"].between(40.0, 41.8)]

    train_df = train_df[train_df["distance_travelled"].between(0.0, 200.0)]
    train_df = train_df[train_df["fare_amount"].between(2.5, 250.0)]

    train_df = train_df[
        ~((train_df["distance_travelled"] < 0.05) & (train_df["fare_amount"] > 3.5))
    ]

    d_lo, d_hi = train_df["distance_travelled"].quantile([0.001, 0.999])
    train_df = train_df[train_df["distance_travelled"].between(d_lo, d_hi)]

    q_lo, q_hi = train_df["fare_amount"].quantile([0.001, 0.999])
    train_df = train_df[train_df["fare_amount"].between(q_lo, q_hi)]

    return train_df


def get_feature_cols(test_df):
    return [c for c in test_df.columns if c != "key"]




## === cell 9
train = handle_date(train)
test = handle_date(test)

train = handle_distance(train)
test = handle_distance(test)

train = clean_up_train(train)
train.describe()



## === cell 10
from sklearn.model_selection import train_test_split

FEATURE_COLS = get_feature_cols(test)

X = train[FEATURE_COLS]
y = train["fare_amount"].astype("float32")

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=0
)



## === cell 11
y_train_log = np.log1p(y_train.astype("float32"))
y_valid_log = np.log1p(y_valid.astype("float32"))

lo, hi = np.quantile(y_train_log, [0.001, 0.999])
mask = (y_train_log >= lo) & (y_train_log <= hi)

X_train_f = X_train.loc[mask]
y_train_log_f = y_train_log.loc[mask]

base_params = dict(
    max_depth=4,  # was 2
    learning_rate=0.08,  # slightly smaller to balance deeper trees
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0,
    tree_method="hist",
    min_child_weight=5,
    reg_lambda=1.0,
)

model_es = xgb.XGBRegressor(
    n_estimators=4000,  # upper bound; best_iteration chosen via early stopping
    **base_params,
)

model_es.fit(
    X_train_f,
    y_train_log_f,
    eval_set=[(X_valid, y_valid_log)],
    verbose=False,
    early_stopping_rounds=80,
)

best_n = int(getattr(model_es, "best_iteration", model_es.n_estimators))
best_n = max(best_n, 100)  # safety: ensure non-trivial model size

y_full_log = np.log1p(y.astype("float32"))
lo_full, hi_full = np.quantile(y_full_log, [0.001, 0.999])
mask_full = (y_full_log >= lo_full) & (y_full_log <= hi_full)

X_full_f = X.loc[mask_full]
y_full_log_f = y_full_log.loc[mask_full]

model = xgb.XGBRegressor(
    n_estimators=best_n,
    **base_params,
)
model.fit(X_full_f, y_full_log_f)

pred_log = model.predict(test[FEATURE_COLS])
pred = np.expm1(pred_log)

pred = np.clip(pred, 0, None)

submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": pred}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)
submission.head(20)
