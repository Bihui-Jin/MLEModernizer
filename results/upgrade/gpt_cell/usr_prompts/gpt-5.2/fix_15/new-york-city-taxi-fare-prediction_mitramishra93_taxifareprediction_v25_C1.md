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

try:
    from sklearnex import patch_sklearn  # type: ignore

    patch_sklearn()
except Exception:
    pass

from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print(os.listdir("../input"))



## === cell 1
train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
test_dtypes = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

NROWS_TRAIN = 2_500_000

train_path = "../input/train.csv"
total_train_rows = 55_423_856  # rows excluding header
rng = np.random.RandomState(RANDOM_STATE)
keep_rows = set(
    rng.choice(
        np.arange(1, total_train_rows + 1), size=NROWS_TRAIN, replace=False
    ).tolist()
)
skiprows = lambda i: (i != 0) and (i not in keep_rows)

train_df = pd.read_csv(
    train_path,
    skiprows=skiprows,
    dtype=train_dtypes,
    parse_dates=["pickup_datetime"],
)

test_df = pd.read_csv(
    "../input/test.csv",
    dtype=test_dtypes,
    parse_dates=["pickup_datetime"],
)

train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], utc=True, errors="coerce"
)
test_df["pickup_datetime"] = pd.to_datetime(
    test_df["pickup_datetime"], utc=True, errors="coerce"
)



## === cell 2
_ = (train_df.shape, test_df.shape)



## === cell 3
train_df.dropna(axis=0, how="any", inplace=True)



## === cell 4
for df in (train_df, test_df):
    if "passenger_count" in df.columns:
        pc = df["passenger_count"].to_numpy(copy=False)
        if (pc == 0).any():
            df.loc[pc == 0, "passenger_count"] = np.uint8(1)

fare = train_df["fare_amount"].to_numpy()
pc = train_df["passenger_count"].to_numpy()
pl = train_df["pickup_latitude"].to_numpy()
p_lon = train_df["pickup_longitude"].to_numpy()

mask = (
    (fare >= 0)
    & (pc >= 1)
    & (pc <= 6)
    & (pl >= -90)
    & (pl <= 90)
    & (p_lon >= -180)
    & (p_lon <= 180)
)
train_df = train_df.loc[mask]



## === cell 5
for df in (train_df, test_df):
    dt = df["pickup_datetime"].dt
    df["date"] = dt.day.astype("uint8")
    df["month"] = dt.month.astype("uint8")
    df["day_of_week"] = dt.dayofweek.astype("uint8")
    df["hour"] = dt.hour.astype("uint8")
    df["year"] = dt.year.astype("uint16")




## === cell 6
def add_sphere_distance(df):
    R = np.float32(6367.0)
    lat1 = np.radians(df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False))
    lat2 = np.radians(df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False))
    dlat = lat2 - lat1
    dlon = np.radians(
        df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
        - df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    )
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    df["S_Distance"] = (R * c).astype("float32", copy=False)


add_sphere_distance(train_df)
add_sphere_distance(test_df)



## === cell 7
mask_bad = (
    (train_df["pickup_latitude"].to_numpy() == 0)
    & (train_df["pickup_longitude"].to_numpy() == 0)
    & (train_df["dropoff_latitude"].to_numpy() != 0)
    & (train_df["dropoff_longitude"].to_numpy() != 0)
    & (train_df["fare_amount"].to_numpy() == 0)
)
if mask_bad.any():
    train_df = train_df.loc[~mask_bad]

mask_zero = (train_df["S_Distance"].to_numpy() == 0) & (
    train_df["fare_amount"].to_numpy() == 0
)
if mask_zero.any():
    train_df = train_df.loc[~mask_zero]

rush_hour_mask = (
    (train_df["hour"].to_numpy() >= 6)
    & (train_df["hour"].to_numpy() <= 20)
    & (train_df["day_of_week"].to_numpy() >= 1)
    & (train_df["day_of_week"].to_numpy() <= 5)
    & (train_df["S_Distance"].to_numpy() == 0)
    & (train_df["fare_amount"].to_numpy() < 2.5)
)
if rush_hour_mask.any():
    train_df = train_df.loc[~rush_hour_mask]



## === cell 8
nyc_mask_train = (
    train_df["pickup_longitude"].between(-74.3, -73.6)
    & train_df["dropoff_longitude"].between(-74.3, -73.6)
    & train_df["pickup_latitude"].between(40.4, 41.0)
    & train_df["dropoff_latitude"].between(40.4, 41.0)
)
train_df = train_df.loc[nyc_mask_train]

nyc_mask_test = (
    test_df["pickup_longitude"].between(-74.3, -73.6)
    & test_df["dropoff_longitude"].between(-74.3, -73.6)
    & test_df["pickup_latitude"].between(40.4, 41.0)
    & test_df["dropoff_latitude"].between(40.4, 41.0)
)



## === cell 9
train_df = train_df.loc[
    (train_df["fare_amount"] >= 2.5)
    & (train_df["fare_amount"] <= 250.0)
    & (train_df["S_Distance"] >= 0.0)
    & (train_df["S_Distance"] <= 100.0)
]

sd = train_df["S_Distance"].to_numpy(dtype=np.float32, copy=False)
fa = train_df["fare_amount"].to_numpy(dtype=np.float32, copy=False)

near_zero_bad = (sd < np.float32(0.3)) & (fa > np.float32(12.0))

rate = fa / np.maximum(sd, np.float32(1e-3))
rate_bad = rate > np.float32(50.0)

low_rate_long_bad = (sd > np.float32(10.0)) & (rate < np.float32(0.8))

bad = near_zero_bad | rate_bad | low_rate_long_bad
if bad.any():
    train_df = train_df.loc[~bad]



## === cell 10
for df in (train_df, test_df):
    p_lon = df["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
    d_lon = df["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)
    p_lat = df["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
    d_lat = df["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)

    dlon = (p_lon - d_lon).astype("float32", copy=False)
    dlat = (p_lat - d_lat).astype("float32", copy=False)

    abs_dlon = np.abs(dlon, dtype=np.float32)
    abs_dlat = np.abs(dlat, dtype=np.float32)

    df["abs_dlon"] = abs_dlon
    df["abs_dlat"] = abs_dlat
    df["manhattan_dist"] = (abs_dlon + abs_dlat).astype("float32", copy=False)

    df["euclidean_deg_dist"] = np.sqrt(
        abs_dlon * abs_dlon + abs_dlat * abs_dlat
    ).astype("float32", copy=False)

    lat1 = np.radians(p_lat.astype(np.float64, copy=False))
    lat2 = np.radians(d_lat.astype(np.float64, copy=False))
    dlon_rad = np.radians((d_lon - p_lon).astype(np.float64, copy=False))
    y = np.sin(dlon_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon_rad)
    df["bearing"] = np.arctan2(y, x).astype("float32", copy=False)

    df["distance_km"] = df["S_Distance"].astype("float32", copy=False)



## === cell 11
test_keys = test_df["key"].copy()

train_df = train_df.drop(["key", "pickup_datetime"], axis=1)
test_features = test_df.drop(["key", "pickup_datetime"], axis=1)

x_train_df = train_df.loc[:, train_df.columns != "fare_amount"]
y_train = train_df["fare_amount"].to_numpy()

x_train_df = x_train_df.sort_index(axis=1)
test_features = test_features.reindex(columns=x_train_df.columns)

x_train = np.ascontiguousarray(x_train_df.to_numpy(dtype=np.float32, copy=False))
x_test = np.ascontiguousarray(test_features.to_numpy(dtype=np.float32, copy=False))



## === cell 12
rg = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
rg.fit(x_train, y_train)
y_predict = rg.predict(x_test)
y_predict = np.clip(y_predict, 0.0, None).astype("float32")



## === cell 13
submission = pd.read_csv("../input/sample_submission.csv")

default_fare = float(np.mean(y_train))

pred_full = pd.Series(y_predict, index=test_df.index, dtype="float32")
pred_full.loc[~nyc_mask_test] = np.float32(default_fare)
pred_full = pred_full.fillna(np.float32(default_fare))

submission["fare_amount"] = pred_full.values.astype("float32")
submission.to_csv("submission_1.csv", index=False)
submission.head(10)
