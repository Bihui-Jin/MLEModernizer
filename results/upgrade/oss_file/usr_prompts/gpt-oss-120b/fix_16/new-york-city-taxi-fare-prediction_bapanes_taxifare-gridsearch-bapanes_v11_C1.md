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

No external packages required in the script and installed.

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def haversine(lon1, lat1, lon2, lat2):
    """Vectorised haversine distance in kilometres."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    km = 6371.0 * 2 * np.arcsin(np.sqrt(a))
    return km


TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

train_usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
numeric_dtype = {
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.float32,
}
df_train = pd.read_csv(
    TRAIN_PATH,
    usecols=train_usecols,
    dtype=numeric_dtype,
    nrows=1_000_000,  # keep original sample size
    low_memory=False,
)

numeric_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
df_train[numeric_cols] = df_train[numeric_cols].astype(np.float32)




## === cell 1
df_train["distance"] = haversine(
    df_train["pickup_longitude"],
    df_train["pickup_latitude"],
    df_train["dropoff_longitude"],
    df_train["dropoff_latitude"],
)
df_train["distance"] = np.clip(df_train["distance"], 0, 100)

df_train["log_distance"] = np.log1p(df_train["distance"])

df_train["pickup_datetime"] = pd.to_datetime(
    df_train["pickup_datetime"], errors="coerce"
)
df_train["hour"] = df_train["pickup_datetime"].dt.hour
df_train["weekday"] = df_train["pickup_datetime"].dt.weekday

df_train["hour_sin"] = np.sin(2 * np.pi * df_train["hour"] / 24)
df_train["hour_cos"] = np.cos(2 * np.pi * df_train["hour"] / 24)
df_train["weekday_sin"] = np.sin(2 * np.pi * df_train["weekday"] / 7)
df_train["weekday_cos"] = np.cos(2 * np.pi * df_train["weekday"] / 7)

feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
    "log_distance",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]

df_train = df_train[(df_train["fare_amount"] > 0) & (df_train["fare_amount"] < 200)]
df_train = df_train.dropna(subset=feature_cols + ["fare_amount"])

df_train[feature_cols] = df_train[feature_cols].astype(np.float32)

X = df_train[feature_cols].values
y = np.log1p(df_train["fare_amount"].values.astype(np.float32))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




## === cell 2
rf = RandomForestRegressor(
    n_estimators=300,  # increase from 200 for better averaging
    max_depth=None,
    max_features=1.0,  # consider all features at each split
    min_samples_leaf=2,
    max_samples=0.9,  # each tree sees 90 % of the data (more information than 80 %)
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)




## === cell 3
val_pred_log = rf.predict(X_val)
val_pred = np.expm1(val_pred_log)
val_pred = np.clip(val_pred, 0, None)

rmse = mean_squared_error(np.expm1(y_val), val_pred, squared=False)
print(f"Validation RMSE: {rmse:.4f}")




## === cell 4
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
df_test = pd.read_csv(
    TEST_PATH,
    usecols=test_usecols,
    dtype=numeric_dtype,
    low_memory=False,
)

df_test["distance"] = haversine(
    df_test["pickup_longitude"],
    df_test["pickup_latitude"],
    df_test["dropoff_longitude"],
    df_test["dropoff_latitude"],
)
df_test["distance"] = np.clip(df_test["distance"], 0, 100)
df_test["log_distance"] = np.log1p(df_test["distance"])

df_test["pickup_datetime"] = pd.to_datetime(df_test["pickup_datetime"], errors="coerce")
df_test["hour"] = df_test["pickup_datetime"].dt.hour
df_test["weekday"] = df_test["pickup_datetime"].dt.weekday

df_test["hour_sin"] = np.sin(2 * np.pi * df_test["hour"] / 24)
df_test["hour_cos"] = np.cos(2 * np.pi * df_test["hour"] / 24)
df_test["weekday_sin"] = np.sin(2 * np.pi * df_test["weekday"] / 7)
df_test["weekday_cos"] = np.cos(2 * np.pi * df_test["weekday"] / 7)

X_test = df_test[feature_cols].astype(np.float32).fillna(0).values

test_pred_log = rf.predict(X_test)
test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({"key": df_test["key"], "fare_amount": test_pred})
submission.to_csv("submission_file.csv", index=False)
print("Submission file 'submission_file.csv' written.")
