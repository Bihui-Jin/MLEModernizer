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

np.random.seed(0)

INPUT_DIR_CANDIDATES = ["../input", "/kaggle/input", "/kaggle/data"]
INPUT_DIR = None
for p in INPUT_DIR_CANDIDATES:
    if os.path.isdir(p):
        INPUT_DIR = p
        break
if INPUT_DIR is None:
    INPUT_DIR = "../input"

print("Using INPUT_DIR:", INPUT_DIR)
print(os.listdir(INPUT_DIR))



## === cell 1
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

if not os.path.exists(train_path):
    alt = os.path.join(INPUT_DIR, "labels.csv")
    if os.path.exists(alt):
        train_path = alt

if not os.path.exists(test_path):
    nested_test = os.path.join(
        INPUT_DIR, "new-york-city-taxi-fare-prediction", "test.csv"
    )
    if os.path.exists(nested_test):
        test_path = nested_test

if not os.path.exists(train_path):
    nested_train = os.path.join(
        INPUT_DIR, "new-york-city-taxi-fare-prediction", "train.csv"
    )
    nested_labels = os.path.join(
        INPUT_DIR, "new-york-city-taxi-fare-prediction", "labels.csv"
    )
    if os.path.exists(nested_train):
        train_path = nested_train
    elif os.path.exists(nested_labels):
        train_path = nested_labels

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

dtype_common = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
dtype_train = dict(dtype_common)
dtype_train["fare_amount"] = "float32"

read_csv_train_kwargs = dict(low_memory=False, engine="c")

read_csv_test_kwargs = {}
try:
    import pyarrow  # noqa: F401

    read_csv_test_kwargs["engine"] = "pyarrow"
except Exception:
    read_csv_test_kwargs["engine"] = "c"
    read_csv_test_kwargs["low_memory"] = False

print("Resolved train_path:", train_path)
print("Resolved test_path :", test_path)

train = pd.read_csv(
    train_path,
    nrows=5000000,
    usecols=usecols_train,
    dtype=dtype_train,
    **read_csv_train_kwargs,
)
test = pd.read_csv(
    test_path,
    usecols=usecols_test,
    dtype=dtype_common,
    **read_csv_test_kwargs,
)

print("Train shape:", train.shape, "Test shape:", test.shape)
print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())




## === cell 2
def handle_date_inplace(df):
    s = df["pickup_datetime"]
    s = s.astype("string").str.replace(" UTC", "", regex=False)

    dt = pd.to_datetime(s, errors="coerce", cache=True)

    dti = dt.dt
    df["hour_of_day"] = dti.hour.astype("int16")
    df["month"] = dti.month.astype("int16")
    df["year"] = dti.year.astype("int16")
    df["day_of_year"] = dti.dayofyear.astype("int16")
    iso = dti.isocalendar()
    week = iso.week.astype("int16")
    df["week"] = week
    df["week_of_year"] = week
    df["weekday"] = dti.weekday.astype("int16")
    df["quarter"] = dti.quarter.astype("int16")
    df["day_of_month"] = dti.day.astype("int16")

    df.drop(columns=["pickup_datetime"], inplace=True)
    return df


train = handle_date_inplace(train)
test = handle_date_inplace(test)




## === cell 3
def handle_distance_inplace(df):
    plon = df["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
    plat = df["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)

    lon_dist = np.abs(plon - dlon).astype(np.float32, copy=False)
    lat_dist = np.abs(plat - dlat).astype(np.float32, copy=False)

    df["distance_travelled"] = np.sqrt(
        lon_dist * lon_dist + lat_dist * lat_dist
    ).astype(np.float32, copy=False)

    R = np.float32(6371.0)
    deg2rad = np.float32(np.pi / 180.0)

    lat1 = plat * deg2rad
    lat2 = dlat * deg2rad
    dlat_r = lat2 - lat1
    dlon_r = (dlon - plon) * deg2rad

    half = np.float32(0.5)
    sin_dlat = np.sin(dlat_r * half)
    sin_dlon = np.sin(dlon_r * half)
    a = sin_dlat * sin_dlat + np.cos(lat1) * np.cos(lat2) * (sin_dlon * sin_dlon)
    df["haversine_km"] = (np.float32(2.0) * R * np.arcsin(np.sqrt(a))).astype(
        np.float32, copy=False
    )

    df["manhattan_approx"] = (lon_dist + lat_dist).astype(np.float32, copy=False)

    df["bearing"] = np.arctan2(dlon_r, dlat_r).astype(np.float32, copy=False)

    def _haversine_to_point_km(lon, lat, lon0, lat0):
        lon0 = np.float32(lon0)
        lat0 = np.float32(lat0)
        lat_r = lat * deg2rad
        lon_r = lon * deg2rad
        lat0_r = lat0 * deg2rad
        lon0_r = lon0 * deg2rad

        dlatp = lat_r - lat0_r
        dlonp = lon_r - lon0_r
        sin_dlatp = np.sin(dlatp * half)
        sin_dlonp = np.sin(dlonp * half)
        a0 = sin_dlatp * sin_dlatp + np.cos(lat_r) * np.cos(lat0_r) * (
            sin_dlonp * sin_dlonp
        )
        return (np.float32(2.0) * R * np.arcsin(np.sqrt(a0))).astype(
            np.float32, copy=False
        )

    NYC_LON, NYC_LAT = -73.985428, 40.748817  # Midtown-ish (Empire State Building)
    JFK_LON, JFK_LAT = -73.7781, 40.6413
    LGA_LON, LGA_LAT = -73.8740, 40.7769
    EWR_LON, EWR_LAT = -74.1745, 40.6895

    df["pickup_dist_nyc_center_km"] = _haversine_to_point_km(
        plon, plat, NYC_LON, NYC_LAT
    )
    df["dropoff_dist_nyc_center_km"] = _haversine_to_point_km(
        dlon, dlat, NYC_LON, NYC_LAT
    )

    df["pickup_dist_jfk_km"] = _haversine_to_point_km(plon, plat, JFK_LON, JFK_LAT)
    df["dropoff_dist_jfk_km"] = _haversine_to_point_km(dlon, dlat, JFK_LON, JFK_LAT)

    df["pickup_dist_lga_km"] = _haversine_to_point_km(plon, plat, LGA_LON, LGA_LAT)
    df["dropoff_dist_lga_km"] = _haversine_to_point_km(dlon, dlat, LGA_LON, LGA_LAT)

    df["pickup_dist_ewr_km"] = _haversine_to_point_km(plon, plat, EWR_LON, EWR_LAT)
    df["dropoff_dist_ewr_km"] = _haversine_to_point_km(dlon, dlat, EWR_LON, EWR_LAT)

    return df


train = handle_distance_inplace(train)
test = handle_distance_inplace(test)




## === cell 4
def clean_up_train(train):
    train = train.dropna()

    fare = train["fare_amount"].to_numpy(dtype=np.float32, copy=False)
    pc = train["passenger_count"].to_numpy(dtype=np.int16, copy=False)
    plon = train["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
    dlon = train["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)
    plat = train["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
    dlat = train["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)
    hv = train["haversine_km"].to_numpy(dtype=np.float32, copy=False)

    mask = (
        (fare > 0)
        & (pc > 0)
        & (pc < 7)
        & (plon >= -74.3)
        & (plon <= -73.7)
        & (dlon >= -74.3)
        & (dlon <= -73.7)
        & (plat >= 40.5)
        & (plat <= 41.0)
        & (dlat >= 40.5)
        & (dlat <= 41.0)
        & (hv > 0.05)
        & (hv < 80.0)
        & (fare < 200)
    )

    mask &= (
        (np.abs(plon) > 0.001)
        & (np.abs(dlon) > 0.001)
        & (np.abs(plat) > 0.001)
        & (np.abs(dlat) > 0.001)
    )

    same_loc = (plon == dlon) & (plat == dlat)
    mask &= ~same_loc

    mask &= fare <= (2.5 + 10.0 * hv)

    denom_km = np.maximum(hv, np.float32(0.2))
    fare_per_km = fare / denom_km
    mask &= (fare_per_km >= 0.5) & (fare_per_km <= 60.0)

    return train.loc[mask]


train = clean_up_train(train)
print("Cleaned train shape:", train.shape)



## === cell 5
feature_cols = test.columns.drop("key")
feature_cols = list(feature_cols)  # stable order

X_train = train.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)
y = train["fare_amount"].to_numpy(dtype=np.float64, copy=False)
X_test = test.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)

np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

print("X_train:", X_train.shape, "X_test:", X_test.shape, "y:", y.shape)



## === cell 6
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=120,
    max_depth=14,
    min_samples_leaf=1,
    random_state=0,
    n_jobs=-1,
)

y_train_log = np.log1p(y)
model.fit(X_train, y_train_log)

preds_log = model.predict(X_test)
preds = np.expm1(preds_log)

preds = np.clip(preds, 0.0, None)

submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": preds}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)

print(submission.head(20))
print("Wrote submission.csv with shape:", submission.shape)
