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
import os, warnings

warnings.filterwarnings("ignore")
print(os.listdir("../input"))




## === cell 1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error




## === cell 2
dtype_dict = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
train = pd.read_csv(
    "../input/train.csv",
    nrows=4_000_000,  # use more rows for better learning
    usecols=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype=dtype_dict,
    low_memory=False,
)




## === cell 3
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] < 200)]
train = train.loc[(train["pickup_longitude"] > -150) & (train["pickup_longitude"] < 0)]
train = train.loc[(train["pickup_latitude"] > 0) & (train["pickup_latitude"] < 80)]
train = train.loc[
    (train["dropoff_longitude"] > -150) & (train["dropoff_longitude"] < 0)
]
train = train.loc[(train["dropoff_latitude"] > 0) & (train["dropoff_latitude"] < 80)]
train = train.loc[train["passenger_count"] <= 8]




## === cell 4
test = pd.read_csv("../input/test.csv")
test_keys = test["key"].values




## === cell 5
def haversine(lon1, lat1, lon2, lat2):
    """Great‑circle distance in kilometres."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def add_distance(df):
    df["distance"] = haversine(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )


add_distance(train)
add_distance(test)


train["lat_diff"] = train["dropoff_latitude"] - train["pickup_latitude"]
test["lat_diff"] = test["dropoff_latitude"] - test["pickup_latitude"]
train["lon_diff"] = train["dropoff_longitude"] - train["pickup_longitude"]
test["lon_diff"] = test["dropoff_longitude"] - test["pickup_longitude"]
train["abs_lat_diff"] = train["lat_diff"].abs()
test["abs_lat_diff"] = test["lat_diff"].abs()
train["abs_lon_diff"] = train["lon_diff"].abs()
test["abs_lon_diff"] = test["lon_diff"].abs()




## === cell 6
train_dt = pd.to_datetime(train["pickup_datetime"])
test_dt = pd.to_datetime(test["pickup_datetime"])

train["hour"] = train_dt.dt.hour
test["hour"] = test_dt.dt.hour

train["weekday"] = train_dt.dt.weekday
test["weekday"] = test_dt.dt.weekday

train["month"] = train_dt.dt.month
test["month"] = test_dt.dt.month

train["is_weekend"] = (train["weekday"] >= 5).astype(np.int8)
test["is_weekend"] = (test["weekday"] >= 5).astype(np.int8)

train["log_hour"] = np.log1p(train["hour"]).astype(np.float32)
test["log_hour"] = np.log1p(test["hour"]).astype(np.float32)

train["log_distance"] = np.log1p(train["distance"])
test["log_distance"] = np.log1p(test["distance"])

train["distance_sq"] = train["distance"] ** 2
test["distance_sq"] = test["distance"] ** 2

train["hour_sin"] = np.sin(2 * np.pi * train["hour"] / 24)
train["hour_cos"] = np.cos(2 * np.pi * train["hour"] / 24)
test["hour_sin"] = np.sin(2 * np.pi * test["hour"] / 24)
test["hour_cos"] = np.cos(2 * np.pi * test["hour"] / 24)

train["dist_hour"] = train["distance"] * train["hour"]
test["dist_hour"] = test["distance"] * test["hour"]

train["passenger_distance"] = train["distance"] * train["passenger_count"]
test["passenger_distance"] = test["distance"] * test["passenger_count"]

train["dist_per_hour"] = train["distance"] / (train["hour"] + 1)
test["dist_per_hour"] = test["distance"] / (test["hour"] + 1)

train["dist_per_passenger"] = train["distance"] / (train["passenger_count"] + 1)
test["dist_per_passenger"] = test["distance"] / (test["passenger_count"] + 1)

train["log_passenger_count"] = np.log1p(train["passenger_count"])
test["log_passenger_count"] = np.log1p(test["passenger_count"])

cols_to_drop = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
train.drop(columns=cols_to_drop, inplace=True)
test.drop(columns=cols_to_drop, inplace=True)




## === cell 7
train.dropna(inplace=True)
test.dropna(inplace=True)




## === cell 8
train = train.loc[train["fare_amount"] <= 100]  # cap extreme fares




## === cell 9
y_original = train["fare_amount"].values.astype(np.float32)
y = np.log1p(y_original).astype(np.float32)

train.drop(columns=["fare_amount"], inplace=True)

X = train.values.astype(np.float32)
X_test = test.values.astype(np.float32)




## === cell 10
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)




## === cell 11
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.01, random_state=42)




## === cell 12
def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))




## === cell 13
model = HistGradientBoostingRegressor(
    max_iter=5000,  # more trees for better fit
    learning_rate=0.003,  # finer step size
    max_depth=15,  # deeper trees to capture interactions
    min_samples_leaf=2,  # allow smaller leaves
    random_state=42,
)

model.fit(X_tr, y_tr)

val_pred_log = model.predict(X_val)
val_pred = np.expm1(val_pred_log)
val_true = np.expm1(y_val)
val_rmse = rmse(val_true, val_pred)
print(f"Validation RMSE: {val_rmse:.4f}")

model.fit(X, y)




## === cell 14
preds_log = model.predict(X_test).flatten()
preds = np.expm1(preds_log)
preds = np.where(preds < 0, 0, preds)




## === cell 15
submission = pd.DataFrame({"key": test_keys, "fare_amount": preds})
submission.to_csv("submission.csv", index=False)




## === cell 16
print("Submission written. Files in current directory:")
print(os.listdir("."))
