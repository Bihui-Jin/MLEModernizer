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
import os
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, asin

try:
    import tensorflow as tf
except Exception:
    tf = None


def distance(arr):
    """
    arr: numpy array with columns
        [pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude]
    Returns: 1‑D array of distances (km) for each row.
    """
    lon1, lat1, lon2, lat2 = arr.T
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c  # Earth radius in kilometers
    return km


def add_datepart(df, field_name, drop=False):
    """
    For column `field_name` (string) in df, create new columns:
    Year, Month, Weekofyear, Day, Dayofweek, Dayofyear, Hour, Elapsed
    The new columns are named `field_name<Part>` (e.g., pickup_datetimeYear).
    If drop=True the original column is removed.
    """
    fld = df[field_name]
    if not np.issubdtype(fld.dtype, np.datetime64):
        fld = pd.to_datetime(fld, errors="coerce")
    df[field_name + "Year"] = fld.dt.year
    df[field_name + "Month"] = fld.dt.month
    df[field_name + "Weekofyear"] = fld.dt.isocalendar().week.astype(int)
    df[field_name + "Day"] = fld.dt.day
    df[field_name + "Dayofweek"] = fld.dt.weekday
    df[field_name + "Dayofyear"] = fld.dt.dayofyear
    df[field_name + "Hour"] = fld.dt.hour
    df[field_name + "Elapsed"] = fld.astype("int64") // 10**9
    if drop:
        df.drop(columns=[field_name], inplace=True)




## === cell 1
train_path = "../input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "../input/new-york-city-taxi-fare-prediction/test.csv"

df_train = pd.read_csv(train_path, nrows=2_000_000)
df_test = pd.read_csv(test_path)




## === cell 2
df_train["Herv_Dist"] = distance(
    df_train[
        ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    ].values.astype(np.float64)
)
df_test["Herv_Dist"] = distance(
    df_test[
        ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    ].values.astype(np.float64)
)

add_datepart(df_train, "pickup_datetime", drop=True)
add_datepart(df_test, "pickup_datetime", drop=True)

df_train.fillna(0, inplace=True)
df_test.fillna(0, inplace=True)

y_train = df_train["fare_amount"].values.reshape(-1, 1)

feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "pickup_datetimeYear",
    "pickup_datetimeMonth",
    "pickup_datetimeWeekofyear",
    "pickup_datetimeDay",
    "pickup_datetimeDayofweek",
    "pickup_datetimeDayofyear",
    "pickup_datetimeHour",
    "pickup_datetimeElapsed",
    "Herv_Dist",
]
x_train_unscl = df_train[feature_cols].values
x_test_unscl = df_test[feature_cols].values

if tf is not None:
    mu = x_train_unscl.mean(axis=0)
    sigma = x_train_unscl.std(axis=0) + 1e-8
    x_train = (x_train_unscl - mu) / sigma
    x_test = (x_test_unscl - mu) / sigma
else:
    x_train = x_train_unscl
    x_test = x_test_unscl




## === cell 3
if tf is not None:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(x_train.shape[1],)),
            tf.keras.layers.Dense(
                2000,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
            tf.keras.layers.Dense(
                1000,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
            tf.keras.layers.Dense(
                500,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
            tf.keras.layers.Dense(
                250,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
            tf.keras.layers.Dense(
                125,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
            tf.keras.layers.Dense(
                50,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
            tf.keras.layers.Dense(
                25,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
            tf.keras.layers.Dense(
                10,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mse")
    model.fit(x_train, y_train, epochs=5, batch_size=1024, verbose=0)
else:
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    )
    model.fit(x_train, y_train.ravel())




## === cell 4
if tf is not None:
    y_pred_test = model.predict(x_test).reshape(-1, 1)
else:
    y_pred_test = model.predict(x_test).reshape(-1, 1)

y_pred_test = np.clip(y_pred_test, a_min=0, a_max=None)

submission = pd.DataFrame({"key": df_test["key"], "fare_amount": y_pred_test.ravel()})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"submission saved to {submission_path}, shape: {submission.shape}")
