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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



## === cell 1
train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
dtype_map = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
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
data = pd.read_csv(
    train_path,
    nrows=1_000_000,
    dtype=dtype_map,
    parse_dates=["pickup_datetime"],
    usecols=usecols,
    low_memory=False,
)



## === cell 2
mean_dropoff_longitude = data["dropoff_longitude"].mean()
mean_dropoff_latitude = data["dropoff_latitude"].mean()
data["dropoff_longitude"] = data["dropoff_longitude"].fillna(mean_dropoff_longitude)
data["dropoff_latitude"] = data["dropoff_latitude"].fillna(mean_dropoff_latitude)



## === cell 3
data = data[data["fare_amount"] <= 500]
data = data[(data["passenger_count"] >= 1) & (data["passenger_count"] <= 7)]

ny_lat_min, ny_lat_max = 40.4774, 40.9176
ny_lon_min, ny_lon_max = -74.2591, -73.7004
data = data[
    (data["pickup_latitude"].between(ny_lat_min, ny_lat_max))
    & (data["pickup_longitude"].between(ny_lon_min, ny_lon_max))
    & (data["dropoff_latitude"].between(ny_lat_min, ny_lat_max))
    & (data["dropoff_longitude"].between(ny_lon_min, ny_lon_max))
]



## === cell 4
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
testData = pd.read_csv(
    test_path,
    parse_dates=["pickup_datetime"],
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int8",
    },
    low_memory=False,
)



## === cell 5
testData["dropoff_longitude"] = testData["dropoff_longitude"].fillna(
    mean_dropoff_longitude
)
testData["dropoff_latitude"] = testData["dropoff_latitude"].fillna(
    mean_dropoff_latitude
)




## === cell 6
def haversine_vectorized(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # Earth radius in km


data["haversine_distance"] = haversine_vectorized(
    data["pickup_latitude"],
    data["pickup_longitude"],
    data["dropoff_latitude"],
    data["dropoff_longitude"],
)

testData["haversine_distance"] = haversine_vectorized(
    testData["pickup_latitude"],
    testData["pickup_longitude"],
    testData["dropoff_latitude"],
    testData["dropoff_longitude"],
)

data = data[(data["haversine_distance"] > 0.05) & (data["haversine_distance"] < 500)]



## === cell 7
data["year"] = data["pickup_datetime"].dt.year
data["month"] = data["pickup_datetime"].dt.month
data["hour_of_day"] = data["pickup_datetime"].dt.hour
data["day_of_week"] = data["pickup_datetime"].dt.dayofweek
data["is_night_time"] = (
    (data["hour_of_day"] >= 21) | (data["hour_of_day"] < 6)
).astype(int)

testData["year"] = testData["pickup_datetime"].dt.year
testData["month"] = testData["pickup_datetime"].dt.month
testData["hour_of_day"] = testData["pickup_datetime"].dt.hour
testData["day_of_week"] = testData["pickup_datetime"].dt.dayofweek
testData["is_night_time"] = (
    (testData["hour_of_day"] >= 21) | (testData["hour_of_day"] < 6)
).astype(int)



## === cell 8
public_holidays = [
    (1, 1),
    (1, 15),
    (2, 12),
    (2, 19),
    (5, 27),
    (6, 19),
    (7, 4),
    (9, 2),
    (10, 14),
    (11, 5),
    (11, 11),
    (11, 28),
    (12, 25),
]
holiday_set = set(public_holidays)

data_md = pd.MultiIndex.from_arrays(
    [data["pickup_datetime"].dt.month, data["pickup_datetime"].dt.day]
)
data["is_public_holiday"] = data_md.isin(holiday_set).astype(int)

test_md = pd.MultiIndex.from_arrays(
    [testData["pickup_datetime"].dt.month, testData["pickup_datetime"].dt.day]
)
testData["is_public_holiday"] = test_md.isin(holiday_set).astype(int)



## === cell 9
airports = [(-73.7789, 40.6413), (-73.8740, 40.7769), (-74.1811, 40.6925)]

airport_lons = np.array([lon for lon, lat in airports])
airport_lats = np.array([lat for lon, lat in airports])


def near_airport_vectorized(lat_series, lon_series):
    lat1 = np.radians(lat_series.to_numpy()[:, None])  # (n,1)
    lon1 = np.radians(lon_series.to_numpy()[:, None])  # (n,1)
    lat2 = np.radians(airport_lats)  # (3,)
    lon2 = np.radians(airport_lons)  # (3,)

    dlon = lon2 - lon1  # broadcast to (n,3)
    dlat = lat2 - lat1
    a = np.sin(dlon / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlat / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = 6371 * c  # (n,3) km
    min_dist = distances.min(axis=1)
    return (min_dist < 5).astype(int)


data["pickup_near_airport"] = near_airport_vectorized(
    data["pickup_latitude"], data["pickup_longitude"]
)
data["dropoff_near_airport"] = near_airport_vectorized(
    data["dropoff_latitude"], data["dropoff_longitude"]
)

testData["pickup_near_airport"] = near_airport_vectorized(
    testData["pickup_latitude"], testData["pickup_longitude"]
)
testData["dropoff_near_airport"] = near_airport_vectorized(
    testData["dropoff_latitude"], testData["dropoff_longitude"]
)



## === cell 10
data["abs_lat_diff"] = (data["dropoff_latitude"] - data["pickup_latitude"]).abs()
data["abs_lon_diff"] = (data["dropoff_longitude"] - data["pickup_longitude"]).abs()

testData["abs_lat_diff"] = (
    testData["dropoff_latitude"] - testData["pickup_latitude"]
).abs()
testData["abs_lon_diff"] = (
    testData["dropoff_longitude"] - testData["pickup_longitude"]
).abs()



## === cell 11
features = [
    "passenger_count",
    "haversine_distance",
    "pickup_near_airport",
    "dropoff_near_airport",
    "hour_of_day",
    "day_of_week",
    "year",
    "month",
    "is_public_holiday",
    "is_night_time",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "abs_lat_diff",
    "abs_lon_diff",
]

X = data[features].astype("float32")
y = data["fare_amount"].astype("float32")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)



## === cell 12
rf_model = RandomForestRegressor(
    n_estimators=300, max_depth=25, random_state=42, n_jobs=-1
)
rf_model.fit(X_train.values, y_train.values)

y_val_pred = rf_model.predict(X_val.values)
rmse = mean_squared_error(y_val, y_val_pred, squared=False)
print("Validation RMSE:", rmse)



## === cell 13
X_test = testData[features].astype("float32")
test_pred = rf_model.predict(X_test.values)

test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({"key": testData["key"], "fare_amount": test_pred})
submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' written.")
