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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error as MSE




## === cell 1
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # Earth radius in km




## === cell 2
dtype_train = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
    "key": "object",
}
dtype_test = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
    "key": "object",
}
train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

train = pd.read_csv(
    train_path,
    dtype=dtype_train,
    parse_dates=["pickup_datetime"],
    nrows=2_000_000,
)
test = pd.read_csv(test_path, dtype=dtype_test, parse_dates=["pickup_datetime"])

for df in (train, test):
    df["hour"] = df["pickup_datetime"].dt.hour.astype("int8")
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek.astype("int8")
    df["month"] = df["pickup_datetime"].dt.month.astype("int8")




## === cell 3
train = train.dropna(how="any", axis="rows")
geo_mask = (
    (train["pickup_longitude"] >= -75)
    & (train["pickup_longitude"] <= -72)
    & (train["pickup_latitude"] >= 40)
    & (train["pickup_latitude"] <= 42)
    & (train["dropoff_longitude"] >= -75)
    & (train["dropoff_longitude"] <= -72)
    & (train["dropoff_latitude"] >= 40)
    & (train["dropoff_latitude"] <= 42)
)
train = train.loc[geo_mask]

mask = (
    (train["pickup_longitude"] != 0)
    & (train["pickup_latitude"] != 0)
    & (train["dropoff_longitude"] != 0)
    & (train["dropoff_latitude"] != 0)
    & (train["passenger_count"] > 0)
    & (train["passenger_count"] <= 5)
)
train = train.loc[mask]




## === cell 4
def add_features(df):
    df["distance"] = haversine_np(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )
    df["manhattan_distance"] = np.abs(
        df["pickup_longitude"] - df["dropoff_longitude"]
    ) + np.abs(df["pickup_latitude"] - df["dropoff_latitude"])
    df["log_distance"] = np.log1p(df["distance"])
    df["distance_per_passenger"] = df["distance"] / df["passenger_count"].replace(0, 1)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["timestamp"] = df["pickup_datetime"].astype("int64") // 1_000_000_000


add_features(train)
add_features(test)




## === cell 5
train = train[(train["fare_amount"] >= 1) & (train["fare_amount"] <= 200)]

train.drop(columns=["key", "pickup_datetime"], inplace=True)
test_keys = test["key"].values
test.drop(columns=["key", "pickup_datetime"], inplace=True)




## === cell 6
X, y_raw = train.drop("fare_amount", axis=1), train["fare_amount"]
y = np.log1p(y_raw)  # log‑transform target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=12)

X_train = X_train.to_numpy(dtype=np.float32, copy=False)
X_val = X_val.to_numpy(dtype=np.float32, copy=False)




## === cell 7
xgb_r = xgb.XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    n_estimators=12000,  # increased from 8000
    learning_rate=0.005,  # lower learning rate for finer updates
    max_depth=10,  # deeper trees
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    gamma=0.1,
    seed=123,
    tree_method="hist",
    max_bin=256,
    n_jobs=4,
)




## === cell 8
xgb_r.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=100,  # allow a few more rounds before stopping
    verbose=False,
)




## === cell 9
y_pred_log = xgb_r.predict(X_val)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_val)
rmse = np.sqrt(MSE(y_true, y_pred))
print(f"Validation RMSE : {rmse:.5f}")




## === cell 10
test_scaled = test.to_numpy(dtype=np.float32, copy=False)




## === cell 11
new_pred_log = xgb_r.predict(test_scaled)
new_pred = np.expm1(new_pred_log)
new_pred = np.maximum(new_pred, 0)  # non‑negative
new_pred = np.clip(new_pred, 1, 200)  # realistic fare bounds




## === cell 12
submission = pd.DataFrame({"key": test_keys, "fare_amount": new_pred})




## === cell 13
submission.head()




## === cell 14
submission.to_csv("submission.csv", index=False)




## === cell 15
print("Submission file 'submission.csv' created with", len(submission), "rows.")
