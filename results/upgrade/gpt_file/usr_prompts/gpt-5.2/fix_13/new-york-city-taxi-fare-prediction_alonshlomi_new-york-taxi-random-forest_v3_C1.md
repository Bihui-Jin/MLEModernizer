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

3.12

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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

BASE_INPUT = "/kaggle/input"
DATA_DIR = os.path.join(BASE_INPUT, "new-york-city-taxi-fare-prediction")

print("BASE_INPUT exists:", os.path.exists(BASE_INPUT))
print("Available under /kaggle/input:", os.listdir(BASE_INPUT)[:20])
print("Using DATA_DIR:", DATA_DIR)
print("DATA_DIR listing (head):", os.listdir(DATA_DIR)[:20])

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

print("train.csv exists:", os.path.exists(train_path))
print("test.csv exists:", os.path.exists(test_path))
print("sample_submission.csv exists:", os.path.exists(sample_sub_path))



## === cell 1
NROWS = 2_000_000  # keep the same pragmatic cap as the provided solution

usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

dtype_train = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
    "key": "string",
    "pickup_datetime": "string",
}
dtype_test = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
    "key": "string",
    "pickup_datetime": "string",
}

train = pd.read_csv(train_path, nrows=NROWS, usecols=usecols_train, dtype=dtype_train)
test = pd.read_csv(test_path, usecols=usecols_test, dtype=dtype_test)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()



## === cell 2
train.dropna(inplace=True)

numeric_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
for c in numeric_cols:
    med = float(train[c].median())
    test[c] = test[c].astype("float32")
    test[c] = test[c].fillna(med)

test["passenger_count"] = test["passenger_count"].fillna(1).astype("int16")
test["pickup_datetime"] = test["pickup_datetime"].fillna("2009-01-01 00:00:00 UTC")

train = train[(train["fare_amount"] > 0) & (train["fare_amount"] < 500)]
train = train[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 6)]

coord_bounds = {
    "pickup_longitude": (-74.5, -72.8),
    "dropoff_longitude": (-74.5, -72.8),
    "pickup_latitude": (40.5, 41.8),
    "dropoff_latitude": (40.5, 41.8),
}
for col, (lo, hi) in coord_bounds.items():
    train = train[(train[col] >= lo) & (train[col] <= hi)]

print("Train shape after cleaning:", train.shape)
print("Test shape (kept all rows):", test.shape)




## === cell 3
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points on Earth (in km).
    Vectorized for NumPy arrays / pandas Series.
    """
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c




## === cell 4
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], errors="coerce", utc=True
)
train.dropna(subset=["pickup_datetime"], inplace=True)

p_lat = train["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
p_lon = train["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
d_lat = train["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
d_lon = train["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)

train["distance"] = haversine_distance(p_lat, p_lon, d_lat, d_lon).astype(np.float32)
train["abs_lat_diff"] = np.abs(d_lat - p_lat).astype(np.float32)
train["abs_lon_diff"] = np.abs(d_lon - p_lon).astype(np.float32)
train["manhattan"] = (train["abs_lat_diff"] + train["abs_lon_diff"]).astype(np.float32)

NYC_LAT, NYC_LON = 40.7128, -74.0060
train["pickup_to_center_km"] = haversine_distance(
    p_lat, p_lon, NYC_LAT, NYC_LON
).astype(np.float32)
train["dropoff_to_center_km"] = haversine_distance(
    d_lat, d_lon, NYC_LAT, NYC_LON
).astype(np.float32)
train["center_delta_km"] = (
    train["dropoff_to_center_km"] - train["pickup_to_center_km"]
).astype(np.float32)

dt = train["pickup_datetime"].dt
train["day_of_week"] = dt.dayofweek.astype(np.int8)
train["month"] = dt.month.astype(np.int8)
train["hour"] = dt.hour.astype(np.int8)

train = train[train["distance"] > 0].copy()
train = train[~((train["fare_amount"] < 2.5) & (train["distance"] > 0.5))].copy()
train = train[~((train["fare_amount"] < 3.5) & (train["distance"] > 1.0))].copy()
train = train[~((train["fare_amount"] < 5.0) & (train["distance"] > 2.0))].copy()
train = train[train["distance"] <= 100].copy()

train = train[~((train["distance"] < 0.05) & (train["fare_amount"] > 10.0))].copy()

train = train[(train["distance"] >= 0.05) & (train["distance"] <= 80.0)].copy()
fare_per_km = (train["fare_amount"] / np.maximum(train["distance"], 0.001)).astype(
    np.float32
)
train = train[(fare_per_km >= 0.5) & (fare_per_km <= 80.0)].copy()
del fare_per_km

dist = train["distance"].to_numpy(dtype=np.float64, copy=False)
fare = train["fare_amount"].to_numpy(dtype=np.float64, copy=False)
A = np.vstack([dist, np.ones_like(dist)]).T
coef, _, _, _ = np.linalg.lstsq(A, fare, rcond=None)
fare_hat = A @ coef
resid = fare - fare_hat
mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
keep = np.abs(resid) <= (12.0 * mad)
train = train.loc[keep].copy()

train.head()



## === cell 5
test["passenger_count"] = test["passenger_count"].clip(1, 6).astype("int16")

p_lat = test["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
p_lon = test["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
d_lat = test["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
d_lon = test["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)

test["distance"] = haversine_distance(p_lat, p_lon, d_lat, d_lon).astype(np.float32)
test["abs_lat_diff"] = np.abs(d_lat - p_lat).astype(np.float32)
test["abs_lon_diff"] = np.abs(d_lon - p_lon).astype(np.float32)
test["manhattan"] = (test["abs_lat_diff"] + test["abs_lon_diff"]).astype(np.float32)

NYC_LAT, NYC_LON = 40.7128, -74.0060
test["pickup_to_center_km"] = haversine_distance(p_lat, p_lon, NYC_LAT, NYC_LON).astype(
    np.float32
)
test["dropoff_to_center_km"] = haversine_distance(
    d_lat, d_lon, NYC_LAT, NYC_LON
).astype(np.float32)
test["center_delta_km"] = (
    test["dropoff_to_center_km"] - test["pickup_to_center_km"]
).astype(np.float32)

test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], errors="coerce", utc=True
)

dt_test = test["pickup_datetime"].dt
test["day_of_week"] = dt_test.dayofweek
test["month"] = dt_test.month
test["hour"] = dt_test.hour

dow_mode = int(train["day_of_week"].mode(dropna=True).iloc[0])
month_mode = int(train["month"].mode(dropna=True).iloc[0])
hour_mode = int(train["hour"].mode(dropna=True).iloc[0])

test["day_of_week"] = test["day_of_week"].fillna(dow_mode).astype(np.int8)
test["month"] = test["month"].fillna(month_mode).astype(np.int8)
test["hour"] = test["hour"].fillna(hour_mode).astype(np.int8)

test.head()



## === cell 6
feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
    "abs_lat_diff",
    "abs_lon_diff",
    "manhattan",
    "pickup_to_center_km",
    "dropoff_to_center_km",
    "center_delta_km",
    "day_of_week",
    "month",
    "hour",
]

X = train[feature_cols]
y = train["fare_amount"]

print("X shape:", X.shape, "y shape:", y.shape)



## === cell 7
random_forest = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    oob_score=True,  # enables oob_prediction_
    bootstrap=True,  # explicit to guarantee OOB availability
)

random_forest.fit(X, y)

oob = getattr(random_forest, "oob_prediction_", None)
if oob is not None:
    mask = np.isfinite(oob)
    rmse = mean_squared_error(y.to_numpy()[mask], oob[mask], squared=False)
    print(f"OOB RMSE (proxy for validation): {rmse:.6f}")
else:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    rf_tmp = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    rf_tmp.fit(X_train, y_train)
    y_pred = rf_tmp.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"Validation RMSE: {rmse:.6f}")



## === cell 8
X_test = test[feature_cols]
y_pred_test = random_forest.predict(X_test)

y_min = float(y.min())
y_max = float(y.max())
y_pred_test = np.clip(y_pred_test, y_min, y_max)
y_pred_test = np.maximum(y_pred_test, 0)

submission = pd.DataFrame({"key": test["key"].astype(str), "fare_amount": y_pred_test})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print("Expected test rows:", pd.read_csv(test_path, usecols=["key"]).shape[0])
submission.head()
