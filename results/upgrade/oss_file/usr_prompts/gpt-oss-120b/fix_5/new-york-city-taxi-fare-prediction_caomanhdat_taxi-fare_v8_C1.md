# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

6.974

# 6. Current score

5.12497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5.12497) has done: 'Add Intel® Extension for Scikit‑Learn to accelerate the RandomForest without changing its hyper‑parameters or any preprocessing steps. The patch is applied before importing `RandomForestRegressor`, giving the same model semantics but a much faster fit.'

# 9. Code solution

## === cell 0
from sklearnex import patch_sklearn

patch_sklearn()  # must run before any sklearn import

import numpy as np
import pandas as pd
import os

print(os.listdir("../input"))




## === cell 1
usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_dict = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train = pd.read_csv(
    "../input/train.csv",
    usecols=usecols,
    dtype=dtype_dict,
    nrows=6000000,
    low_memory=False,
)
test = pd.read_csv(
    "../input/test.csv",
    usecols=[c for c in usecols if c != "fare_amount"],
    dtype=dtype_dict,
    low_memory=False,
)




## === cell 2
def handle_date(df):
    df["pickup_datetime"] = df["pickup_datetime"].str.replace(" UTC", "", regex=False)
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    df["hour_of_day"] = df["pickup_datetime"].dt.hour.astype("int8")
    df["week"] = df["pickup_datetime"].dt.isocalendar().week.astype("int16")
    df["month"] = df["pickup_datetime"].dt.month.astype("int8")
    df["year"] = df["pickup_datetime"].dt.year.astype("int16")
    df["day_of_year"] = df["pickup_datetime"].dt.dayofyear.astype("int16")
    df["weekday"] = df["pickup_datetime"].dt.weekday.astype("int8")
    df["quarter"] = df["pickup_datetime"].dt.quarter.astype("int8")
    df["day_of_month"] = df["pickup_datetime"].dt.day.astype("int8")
    df.drop(columns=["pickup_datetime"], inplace=True)
    return df


train = handle_date(train)
test = handle_date(test)




## === cell 3
def clean_up_train(df):
    df = df.dropna()
    df = df[df["fare_amount"] > 0]
    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] < 7)]
    return df


train = clean_up_train(train)




## === cell 4
def cleanup_out_of_zone(train_df, test_df):
    min_pickup_long = test_df["pickup_longitude"].min()
    max_pickup_long = test_df["pickup_longitude"].max()
    min_dropoff_lat = test_df["dropoff_latitude"].min()
    max_dropoff_lat = test_df["dropoff_latitude"].max()
    mask = (
        (train_df["pickup_longitude"] >= min_pickup_long)
        & (train_df["pickup_longitude"] <= max_pickup_long)
        & (train_df["dropoff_latitude"] >= min_dropoff_lat)
        & (train_df["dropoff_latitude"] <= max_dropoff_lat)
    )
    return train_df[mask]


train = cleanup_out_of_zone(train, test)




## === cell 5
def get_samples_output(df):
    X = df.drop(columns=["key", "fare_amount"])
    y = df["fare_amount"]
    return X, y


samples_train, samples_label = get_samples_output(train)




## === cell 6
from sklearn.ensemble import (
    RandomForestRegressor,
)  # uses the patched (accelerated) version

X_np = samples_train.values.astype(np.float32, copy=False)
y_np = samples_label.values.astype(np.float32, copy=False)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=0,
    n_jobs=-1,
    bootstrap=True,
)

rf.fit(X_np, y_np)

test_features = test.drop(columns=["key"])
test_np = test_features.values.astype(np.float32, copy=False)
preds = rf.predict(test_np)

submission = pd.DataFrame({"key": test["key"], "fare_amount": preds})
submission.to_csv("submission.csv", index=False)
submission.head(20)
