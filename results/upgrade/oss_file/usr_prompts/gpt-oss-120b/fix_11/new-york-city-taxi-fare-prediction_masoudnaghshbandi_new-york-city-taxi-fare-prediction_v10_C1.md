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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
import os, numpy as np, pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import xgboost as xgb

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
dtype_map = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=5_000_000,
    dtype=dtype_map,
    usecols=list(dtype_map.keys()) + ["pickup_datetime"],
)
test = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/test.csv",
    dtype=dtype_map,
    usecols=list(dtype_map.keys()) + ["pickup_datetime"],
)



## === cell 2
train.dropna(inplace=True)

mask = (
    (train["pickup_longitude"] != 0)
    & (train["pickup_latitude"] != 0)
    & (train["dropoff_longitude"] != 0)
    & (train["dropoff_latitude"] != 0)
    & train["passenger_count"].between(1, 5)
    & train["pickup_longitude"].between(-75, -72)
    & train["pickup_latitude"].between(40, 42)
    & train["dropoff_longitude"].between(-75, -72)
    & train["dropoff_latitude"].between(40, 42)
    & train["fare_amount"].between(0, 200)
)
train = train[mask]


def haversine(lon1, lat1, lon2, lat2):
    """Vectorised haversine distance in kilometres."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


train["distance"] = haversine(
    train["pickup_longitude"],
    train["pickup_latitude"],
    train["dropoff_longitude"],
    train["dropoff_latitude"],
)

train["manhattan"] = np.abs(
    train["pickup_longitude"] - train["dropoff_longitude"]
) + np.abs(train["pickup_latitude"] - train["dropoff_latitude"])

train["log_distance"] = np.log1p(train["distance"])
train["log_manhattan"] = np.log1p(train["manhattan"])

train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"])
train["hour"] = train["pickup_datetime"].dt.hour
train["minute"] = train["pickup_datetime"].dt.minute
train["dayofweek"] = train["pickup_datetime"].dt.dayofweek
train["month"] = train["pickup_datetime"].dt.month

train["hour_sin"] = np.sin(2 * np.pi * train["hour"] / 24)
train["hour_cos"] = np.cos(2 * np.pi * train["hour"] / 24)
train["minute_sin"] = np.sin(2 * np.pi * train["minute"] / 60)
train["minute_cos"] = np.cos(2 * np.pi * train["minute"] / 60)

train["is_weekend"] = train["dayofweek"].isin([5, 6]).astype(int)



## === cell 3
X = train.drop(["fare_amount", "key", "pickup_datetime"], axis=1)
y = np.log1p(train["fare_amount"])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=12)



## === cell 4
X_train_arr = X_train.astype(np.float32).values
X_val_arr = X_val.astype(np.float32).values

dtrain = xgb.DMatrix(X_train_arr, label=y_train)
dval = xgb.DMatrix(X_val_arr, label=y_val)



## === cell 5
params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.015,
    "max_depth": 12,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 0.5,
    "tree_method": "hist",
    "eval_metric": "rmse",
    "seed": 123,
    "max_bin": 128,  # speed‑up without changing model capacity
    "nthread": 4,
}
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dval, "validation")],
    early_stopping_rounds=50,
    verbose_eval=False,
)



## === cell 6
y_val_pred_log = bst.predict(dval)
y_val_pred = np.expm1(y_val_pred_log)
rmse = np.sqrt(MSE(np.expm1(y_val), y_val_pred))
print(f"RMSE : {rmse:.5f}")



## === cell 7
test_clean = test.copy()
test_features = test_clean.drop(["key", "pickup_datetime"], axis=1)

test_features["distance"] = haversine(
    test_features["pickup_longitude"],
    test_features["pickup_latitude"],
    test_features["dropoff_longitude"],
    test_features["dropoff_latitude"],
)

test_features["manhattan"] = np.abs(
    test_features["pickup_longitude"] - test_features["dropoff_longitude"]
) + np.abs(test_features["pickup_latitude"] - test_features["dropoff_latitude"])

test_features["log_distance"] = np.log1p(test_features["distance"])
test_features["log_manhattan"] = np.log1p(test_features["manhattan"])

test_features["pickup_datetime"] = pd.to_datetime(test_clean["pickup_datetime"])
test_features["hour"] = test_features["pickup_datetime"].dt.hour
test_features["minute"] = test_features["pickup_datetime"].dt.minute
test_features["dayofweek"] = test_features["pickup_datetime"].dt.dayofweek
test_features["month"] = test_features["pickup_datetime"].dt.month

test_features["hour_sin"] = np.sin(2 * np.pi * test_features["hour"] / 24)
test_features["hour_cos"] = np.cos(2 * np.pi * test_features["hour"] / 24)
test_features["minute_sin"] = np.sin(2 * np.pi * test_features["minute"] / 60)
test_features["minute_cos"] = np.cos(2 * np.pi * test_features["minute"] / 60)

test_features["is_weekend"] = test_features["dayofweek"].isin([5, 6]).astype(int)

test_features = test_features.drop(["pickup_datetime"], axis=1)

test_features_arr = test_features.astype(np.float32).values
dtest = xgb.DMatrix(test_features_arr)



## === cell 8
test_pred_log = bst.predict(dtest)
test_pred = np.expm1(test_pred_log)

test_pred = np.clip(test_pred, 0, 200)



## === cell 9
submission = pd.DataFrame({"key": test_clean["key"], "fare_amount": test_pred})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with shape:", submission.shape)
