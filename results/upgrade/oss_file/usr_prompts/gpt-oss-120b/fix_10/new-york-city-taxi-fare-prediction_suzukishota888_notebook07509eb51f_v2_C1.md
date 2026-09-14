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

3.11

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
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
cols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
train_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    usecols=cols,
    nrows=800_000,  # increased sample size
)

test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    usecols=test_usecols,
)




## === cell 2
train_df["hour"] = pd.to_datetime(train_df["pickup_datetime"]).dt.hour
test_df["hour"] = pd.to_datetime(test_df["pickup_datetime"]).dt.hour
train_df["weekday"] = pd.to_datetime(train_df["pickup_datetime"]).dt.weekday
test_df["weekday"] = pd.to_datetime(test_df["pickup_datetime"]).dt.weekday

train_df["is_weekend"] = (train_df["weekday"] >= 5).astype(np.int8)
test_df["is_weekend"] = (test_df["weekday"] >= 5).astype(np.int8)

train_df["hour_sin"] = np.sin(2 * np.pi * train_df["hour"] / 24).astype(np.float32)
train_df["hour_cos"] = np.cos(2 * np.pi * train_df["hour"] / 24).astype(np.float32)
test_df["hour_sin"] = np.sin(2 * np.pi * test_df["hour"] / 24).astype(np.float32)
test_df["hour_cos"] = np.cos(2 * np.pi * test_df["hour"] / 24).astype(np.float32)




## === cell 3
train_df.dropna(
    subset=[
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_longitude",
        "pickup_latitude",
    ],
    inplace=True,
)

train_df = train_df[(train_df["fare_amount"] >= 2.5) & (train_df["fare_amount"] < 100)]

train_df = train_df[
    (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] <= 6)
]

lon_min, lon_max = -74.5, -73.0
lat_min, lat_max = 40.5, 42.0
train_df = train_df[
    (train_df["pickup_longitude"] >= lon_min)
    & (train_df["pickup_longitude"] <= lon_max)
    & (train_df["dropoff_longitude"] >= lon_min)
    & (train_df["dropoff_longitude"] <= lon_max)
    & (train_df["pickup_latitude"] >= lat_min)
    & (train_df["pickup_latitude"] <= lat_max)
    & (train_df["dropoff_latitude"] >= lat_min)
    & (train_df["dropoff_latitude"] <= lat_max)
]




## === cell 4
test_keys = test_df["key"].copy()

train_df = train_df.drop(["key", "pickup_datetime"], axis=1)
test_df = test_df.drop(["key", "pickup_datetime"], axis=1)




## === cell 5
def haversine(lat1, lon1, lat2, lon2):
    R = np.float32(6371.0)
    lat1, lat2 = np.radians(lat1.astype(np.float32)), np.radians(
        lat2.astype(np.float32)
    )
    dlat = lat2 - lat1
    dlon = np.radians(lon2.astype(np.float32) - lon1.astype(np.float32))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


train_df["distance"] = haversine(
    train_df["pickup_latitude"].values,
    train_df["pickup_longitude"].values,
    train_df["dropoff_latitude"].values,
    train_df["dropoff_longitude"].values,
)
train_df["lat_diff"] = np.abs(
    train_df["pickup_latitude"] - train_df["dropoff_latitude"]
)
train_df["lon_diff"] = np.abs(
    train_df["pickup_longitude"] - train_df["dropoff_longitude"]
)

test_df["distance"] = haversine(
    test_df["pickup_latitude"].values,
    test_df["pickup_longitude"].values,
    test_df["dropoff_latitude"].values,
    test_df["dropoff_longitude"].values,
)
test_df["lat_diff"] = np.abs(test_df["pickup_latitude"] - test_df["dropoff_latitude"])
test_df["lon_diff"] = np.abs(test_df["pickup_longitude"] - test_df["dropoff_longitude"])




## === cell 6
train = train_df
test = test_df




## === cell 7
X = train.drop("fare_amount", axis=1).astype(np.float32).values
y = train["fare_amount"].values.astype(np.float32)
X_test = test.astype(np.float32).values




## === cell 8
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)




## === cell 9
model = RandomForestRegressor(
    n_estimators=600,  # a bit more trees for stronger ensemble
    max_depth=30,
    random_state=0,
    n_jobs=-1,
    min_samples_leaf=1,
)
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"validation RMSE: {rmse:.5f}")




## === cell 10
sub = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)
sub["fare_amount"] = model.predict(X_test)
sub.to_csv("submission.csv", index=False)
print("submission.csv written")
