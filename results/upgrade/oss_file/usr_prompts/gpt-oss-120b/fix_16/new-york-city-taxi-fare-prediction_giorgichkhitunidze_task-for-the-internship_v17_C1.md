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

3.10

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
seaborn==0.12.2
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
tf_keras==2.18.0
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

np.random.seed(42)




## === cell 1
train_df = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=2000000
)
test_df = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")




## === cell 2
train_df.isnull().sum()




## === cell 3
train_df.dropna(axis=0, subset=["dropoff_longitude", "dropoff_latitude"], inplace=True)
train_df.reset_index(drop=True, inplace=True)




## === cell 4
pd.set_option("display.float_format", lambda x: "%.5f" % x)
train_df.describe()




## === cell 5
print("Number of observations out of valid range in coordinate columns:")
print(
    "pickup_longitude:",
    (train_df.pickup_longitude < -180).sum() + (train_df.pickup_longitude > 180).sum(),
)
print(
    "pickup_latitude :",
    (train_df.pickup_latitude < -90).sum() + (train_df.pickup_latitude > 90).sum(),
)
print(
    "dropoff_longitude:",
    (train_df.dropoff_longitude < -180).sum()
    + (train_df.dropoff_longitude > 180).sum(),
)
print(
    "dropoff_latitude :",
    (train_df.dropoff_latitude < -90).sum() + (train_df.dropoff_latitude > 90).sum(),
)




## === cell 6
coord_mask = (
    (train_df.pickup_longitude < -180)
    | (train_df.pickup_longitude > 180)
    | (train_df.pickup_latitude < -90)
    | (train_df.pickup_latitude > 90)
    | (train_df.dropoff_longitude < -180)
    | (train_df.dropoff_longitude > 180)
    | (train_df.dropoff_latitude < -90)
    | (train_df.dropoff_latitude > 90)
)
train_df = train_df[~coord_mask].reset_index(drop=True)




## === cell 7
train_df.describe()




## === cell 8
swap_idx = train_df[train_df.pickup_longitude >= 40].index
train_df.loc[swap_idx, ["dropoff_longitude", "dropoff_latitude"]] = train_df.loc[
    swap_idx, ["dropoff_latitude", "dropoff_longitude"]
].values
train_df.loc[swap_idx, ["pickup_longitude", "pickup_latitude"]] = train_df.loc[
    swap_idx, ["pickup_latitude", "pickup_longitude"]
].values




## === cell 9
geo_mask = (
    (train_df.pickup_longitude < -75)
    | (train_df.pickup_longitude > -72)
    | (train_df.dropoff_longitude < -75)
    | (train_df.dropoff_longitude > -72)
    | (train_df.pickup_latitude < 40)
    | (train_df.pickup_latitude > 42)
    | (train_df.dropoff_latitude < 40)
    | (train_df.dropoff_latitude > 42)
)
train_df = train_df[~geo_mask].reset_index(drop=True)




## === cell 10
train_df.describe()




## === cell 11
train_df.passenger_count.value_counts()




## === cell 12
train_df = train_df.drop(train_df[train_df.passenger_count == 0].index, axis=0)




## === cell 13
train_df.fare_amount.sort_values(ascending=False)




## === cell 14
train_df = train_df.drop(train_df[train_df.fare_amount <= 0].index, axis=0)
train_df = train_df[train_df.fare_amount < 200]  # cap high fares
train_df["fare_amount"].sort_values(ascending=False)




## === cell 15
test_df.isna().sum()




## === cell 16
test_df.describe()




## === cell 17
train_df.dtypes




## === cell 18
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])




## === cell 19
def date_splitter(df):
    df["Year"] = df["pickup_datetime"].dt.year
    df["Month"] = df["pickup_datetime"].dt.month
    df["Day"] = df["pickup_datetime"].dt.day
    df["Weekday"] = df["pickup_datetime"].dt.dayofweek
    df["Hour"] = df["pickup_datetime"].dt.hour


date_splitter(train_df)
date_splitter(test_df)

train_df.drop(["pickup_datetime"], axis=1, inplace=True)
test_df.drop(["pickup_datetime"], axis=1, inplace=True)




## === cell 20
import math


def haversine_distance(df):
    coord = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]
    phi1, lambda1, phi2, lambda2 = [df[i] * math.pi / 180.0 for i in coord]
    R = 6371
    dPhi = phi2 - phi1
    dLambda = lambda2 - lambda1
    a = (
        np.sin(dPhi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dLambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["Distance"] = R * c

    df["Delta_Lat"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
    df["Delta_Lon"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["Is_Weekend"] = (df["Weekday"] >= 5).astype(int)
    df["Dist_per_passenger"] = df["Distance"] / (df["passenger_count"] + 1)

    df["Log_Distance"] = np.log1p(df["Distance"])
    df["Is_Night"] = ((df["Hour"] >= 0) & (df["Hour"] <= 5)).astype(int)

    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)


haversine_distance(train_df)
haversine_distance(test_df)




## === cell 21
train_df.Distance.sort_values()




## === cell 22
pass




## === cell 23
pass




## === cell 24
pass




## === cell 25
pass




## === cell 26
pass




## === cell 27
pass




## === cell 28
pass




## === cell 29
features = [col for col in train_df.columns if col not in ["key", "fare_amount"]]
target = "fare_amount"




## === cell 30
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor




## === cell 31
X_train, X_val, y_train, y_val = train_test_split(
    train_df[features], train_df[target], test_size=0.30, random_state=42
)

train_X = X_train.to_numpy(np.float32, copy=False)
val_X = X_val.to_numpy(np.float32, copy=False)
test_X = test_df[features].to_numpy(np.float32, copy=False)




## === cell 32
y_train_log = np.log1p(y_train).astype(np.float32)
y_val_log = np.log1p(y_val).astype(np.float32)

log_model = XGBRegressor(
    n_estimators=8000,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    n_jobs=os.cpu_count(),
    random_state=42,
    reg_lambda=1.0,
    tree_method="hist",
    predictor="cpu_predictor",
    max_bin=256,
    single_precision_histogram=True,
)

log_model.fit(
    train_X,
    y_train_log,
    eval_set=[(val_X, y_val_log)],
    eval_metric="rmse",
    early_stopping_rounds=800,
    verbose=False,
)

val_pred_log = log_model.predict(val_X)
val_pred_log_back = np.expm1(val_pred_log)
val_rmse_log = mean_squared_error(y_val, val_pred_log_back, squared=False)

raw_model = XGBRegressor(
    n_estimators=6000,
    max_depth=6,  # shallower trees to reduce over‑fit
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    n_jobs=os.cpu_count(),
    random_state=42,
    reg_lambda=2.0,  # stronger L2 regularisation
    reg_alpha=0.0,
    tree_method="hist",
    predictor="cpu_predictor",
    max_bin=256,
    single_precision_histogram=True,
)

raw_model.fit(
    train_X,
    y_train.values,
    eval_set=[(val_X, y_val.values)],
    eval_metric="rmse",
    early_stopping_rounds=300,
    verbose=False,
)

val_pred_raw = raw_model.predict(val_X)
val_rmse_raw = mean_squared_error(y_val, val_pred_raw, squared=False)

print(f"Validation RMSE (log‑model): {val_rmse_log:.5f}")
print(f"Validation RMSE (raw‑model): {val_rmse_raw:.5f}")

if val_rmse_raw < val_rmse_log:
    best_model = raw_model
    test_pred = best_model.predict(test_X)
else:
    best_model = log_model
    test_pred_log = best_model.predict(test_X)
    test_pred = np.expm1(test_pred_log)

print(f"Selected model validation RMSE: {min(val_rmse_log, val_rmse_raw):.5f}")




## === cell 33
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission.to_csv("taxi_fare_submission.csv", index=False)
