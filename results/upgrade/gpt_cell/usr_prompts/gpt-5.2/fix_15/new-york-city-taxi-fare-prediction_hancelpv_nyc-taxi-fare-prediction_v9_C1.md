# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

3.99125

# 6. Current score

4.89959

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.20754) has done: 'The timeout is dominated by fitting a default `RandomForestRegressor` on 1,000,000 rows with one-hot encoded time features, which is far too expensive in 600s. I keep the exact same feature engineering and the same RandomForest model/fit/predict flow, but (1) accelerate the pipeline by using Intel’s scikit-learn extensions (already installed) and parallel training, and (2) avoid unnecessary work and memory blowups by using categoricals + separate `get_dummies` on train/test with aligned columns (equivalent to concatenation for one-hot) and by using faster vectorized operations for the week-of-month feature. These changes preserve evaluation semantics and model accuracy (no sampling, no reduced trees, no early stopping), while substantially reducing overhead and enabling faster training with full CPU usage.'
- What this solution (achieved 7.33803) has done: 'Diagnosis: The crash happens because `y` is a NumPy array created before any row filtering that changes `train`’s index (multiple boolean filters preserve original row labels). After `train_X.dropna(...)`, `train_X.index` contains original (non-0..n-1) labels, but `y[...]` is being indexed positionally as a NumPy array, causing an out-of-bounds `IndexError`. The correct alignment is to keep `y` indexed the same way as `train_X` (by label), or reset indices consistently.  

Patch summary: In cell 12 only, rebuild `y` as a pandas Series aligned to `train_X` by index (using `train.loc[train_X.index, "fare_amount"]` after the final `dropna`), then convert back to a NumPy array so downstream semantics stay the same. This avoids positional indexing with stale labels while keeping model inputs/outputs unchanged.  

Updated cells: Cell 12 only (minimal edit around the failing `y = y[...]` line).  

Compatibility notes for cell k+1: `train_X` and `test_X` remain DataFrames; `y` remains a 1D NumPy array; variables `x` and `x_test` in cell 13 continue to work unchanged.  

Assumptions: `train_X`’s index is a subset of `train`’s index (true because `train_X` is derived from `train` and then filtered by `dropna`).'
- What this solution (achieved 5.22914) has done: 'Your current RMSE (7.33803) is worse than the target (3.99125), so we should improve performance with minimal, safe changes that preserve the same RandomForest training flow and the same feature set concept. The biggest likely cause of poor score here is a distribution mismatch: you train only on rides inside a tight NYC bounding box but you do not apply the same bounding box filtering to the test set, which pushes the model to extrapolate badly for out-of-box rows. I apply the identical NYC bounding-box mask to the test set and then fill predictions for the excluded test rows with the median training fare (a robust constant baseline), which typically reduces RMSE while keeping the same model/fit/predict semantics. I also clip predictions to a reasonable range (>=0 and <=250) consistent with your training filters to avoid extreme errors that hurt RMSE.'
- What this solution (achieved 4.89899) has done: 'Your current RMSE (5.22914) is worse than the target (3.99125), so we should make a small, low-risk improvement without changing the overall RandomForest + engineered features approach. The biggest gap is that the model never sees absolute pickup/dropoff locations (only deltas), which hurts it because NYC fares depend strongly on where the trip starts/ends; adding these raw coordinates as features is a minimal change and typically yields a meaningful RMSE drop. To keep behavior stable and avoid training/test feature mismatch, we include the four coordinate columns in the feature list before one-hot encoding (they remain numeric). Everything else (data size, filters, model type, fit/predict flow, submission writing) stays the same.'
- What this solution (achieved 4.89142) has done: 'Your current RMSE (4.89899) is worse than the target (3.99125), so we should make a small improvement while keeping the same RandomForest + (geo + time + one-hot) pipeline. The biggest low-risk gain is to add a single, widely-used NYC Taxi feature: Haversine (great-circle) distance, computed from the same coordinates you already use; this preserves the overall feature-engineering approach and typically reduces error versus using only degree deltas. To keep behavior stable, we just append this numeric feature to the existing feature list and leave the model/training flow unchanged. Submission writing and paths remain the same.'
- What this solution (achieved 4.87611) has done: 'Your current RMSE (4.89142) is worse than the target (3.99125), so we should make a small, low-risk improvement without changing the RandomForest model or the overall feature-engineering flow. The biggest likely remaining error source is that the model doesn’t have strong “trip length” signal in the same unit as fares; we add one additional numeric geo feature: approximate straight-line distance in miles derived from your existing Haversine distance (no new data, same semantics). We also add a minimal “airport-ish” indicator using the same coordinates (binary flags) to help capture large systematic fare patterns with tiny feature additions. Everything else (row count, filters, one-hot time handling, model type/fit/predict, submission format/path) stays the same.'
- What this solution (achieved 4.89871) has done: 'To move RMSE down toward your 3.99125 target (current 4.87611; lower is better) without changing the core RandomForest + engineered geo/time + one-hot pipeline, I make two small, standard NYC-taxi cleaning tweaks that reduce label noise in the 1M training rows: remove rows where pickup/dropoff are identical (near-zero distance rides often contain noisy fares) and filter extreme speed outliers using your existing Haversine miles and parsed datetime (keeps typical trips, drops implausible ones). These are minimal additions that preserve the same modeling approach while improving training data quality. I also ensure the datetime parsing used for the speed filter is consistent with your existing `add_time_features` parsing (UTC, coercing errors), and keep submission generation unchanged.'
- What this solution (achieved 4.89704) has done: 'Your current RMSE (4.89871) is worse than the target (3.99125), so we should improve but keep the same RandomForest + geo/time + one-hot pipeline. The most effective minimal change here is to add two standard, lightweight numeric features that keep the same feature-engineering “type”: (1) `log1p(haversine_miles)` to help the model fit fare vs distance nonlinearity and (2) `bearing` (trip direction), which is known to capture systematic route/bridge/tunnel effects in NYC. These are simple deterministic transformations from your existing coordinates, don’t alter the model/fit/predict flow, and typically reduce RMSE materially. Submission writing, paths, row filtering logic, and the RandomForest configuration remain unchanged.'
- What this solution (achieved 4.89683) has done: 'Your current RMSE (4.89704; lower is better) is meaningfully worse than the target (3.99125), so we should make a small, safe improvement without changing the RandomForest training flow. The biggest remaining low-risk gap is that the model lacks a strong “base fare + distance rate + time-of-day” signal; adding a single deterministic numeric feature `estimated_fare` (a simple NYC-like formula using your existing haversine miles and time-derived night/rush flags) typically reduces RMSE while keeping the same model, loss, and fit/predict semantics. This is just additional feature engineering (no new data, no tuning loops), and it’s applied identically to train and test before one-hot encoding. Everything else (filters, bbox handling + fallback, RandomForestRegressor, submission writing) stays unchanged.'
- What this solution (achieved 4.91663) has done: 'Your current RMSE (4.89683) is still above the target (3.99125), so we should improve accuracy with the smallest changes that keep the same RandomForest + engineered geo/time + one-hot pipeline. The biggest low-risk gain here is to remove a known, systematic label-noise pattern: rows where the pickup or dropoff coordinates are exactly (0,0), which are erroneous in this dataset and hurt the model. I also apply the same coordinate validity filters you already apply to train (lat/long ranges) to the test set before the bbox split, and route any invalid/out-of-box test rows to the existing fallback prediction, preserving your current “in-bbox model + fallback” semantics. Everything else (features, model type, fit/predict flow, submission format/path) remains unchanged.'
- What this solution (achieved 4.89959) has done: 'Your RMSE (4.91663; lower is better) is still above the target (3.99125), so we should nudge performance down with very small, safe changes that keep the same RandomForest + engineered geo/time + one-hot pipeline. The least invasive likely win is to fix a subtle train/test distribution mismatch: you filter training to NYC bbox but do not add any “location prior” inside the bbox; adding distances to Midtown (a common baseline point) gives the forest a simple global-location signal without changing the model or training loop. Second, your current `align(..., join="outer")` introduces train-only dummy columns that are always-zero at test time (and vice versa), which can slightly hurt splits; switching to `join="inner"` keeps only shared columns and typically improves generalization while preserving the exact one-hot approach. Finally, we use the in-bbox training median (not the global median after multiple filters) consistently as fallback to reduce large errors on out-of-bbox rows.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import math

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

SEED = 42
np.random.seed(SEED)


def resolve_path(fname: str) -> str:
    candidates = [
        os.path.join("../input", fname),
        os.path.join("/kaggle/input", fname),
        os.path.join("/kaggle/data", fname),
        os.path.join("/kaggle/data/new-york-city-taxi-fare-prediction", fname),
        os.path.join("/kaggle/input/new-york-city-taxi-fare-prediction", fname),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


print("Resolved train:", resolve_path("train.csv"))
print("Resolved test :", resolve_path("test.csv"))
print("Resolved sample:", resolve_path("sample_submission.csv"))



## === cell 1
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

cols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]



## === cell 2
train_path = resolve_path("train.csv")
test_path = resolve_path("test.csv")
sample_path = resolve_path("sample_submission.csv")

train = pd.read_csv(train_path, nrows=1000000, usecols=cols, dtype=types)
test = pd.read_csv(
    test_path,
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
)
samp = pd.read_csv(sample_path)



## === cell 3
train.dropna(how="any", axis="rows", inplace=True)
train = train[train.fare_amount > 0]
train = train[train.fare_amount < 250]  # common robust cap for this dataset
train = train[train["passenger_count"] <= 6]
train = train[train["passenger_count"] > 0]



## === cell 4
latitude_mask_pickup = (train.pickup_latitude > -90) & (train.pickup_latitude < 90)
train = train[latitude_mask_pickup]

latitude_mask_dropoff = (train.dropoff_latitude > -90) & (train.dropoff_latitude < 90)
train = train[latitude_mask_dropoff]



## === cell 5
longitude_mask_pickup = (train.pickup_longitude > -180) & (train.pickup_longitude < 180)
train = train[longitude_mask_pickup]

longitude_mask_dropoff = (train.dropoff_longitude > -180) & (
    train.dropoff_longitude < 180
)
train = train[longitude_mask_dropoff]



## === cell 6
nyc_lon_min, nyc_lon_max = -74.3, -73.7
nyc_lat_min, nyc_lat_max = 40.5, 41.0

bbox = (
    (train.pickup_longitude.between(nyc_lon_min, nyc_lon_max))
    & (train.dropoff_longitude.between(nyc_lon_min, nyc_lon_max))
    & (train.pickup_latitude.between(nyc_lat_min, nyc_lat_max))
    & (train.dropoff_latitude.between(nyc_lat_min, nyc_lat_max))
)
train = train[bbox]



## === cell 7
y = train.fare_amount.values
test_id = test.key

train_X = train.drop(["fare_amount"], axis=1)
test_X = test.copy()




## === cell 8
def week_num_vec(day_series: pd.Series) -> pd.Series:
    bins = [0, 7, 14, 21, 28, 31]
    labels = ["first", "second", "third", "fourth", "fifth"]
    return pd.cut(day_series, bins=bins, labels=labels, include_lowest=True, right=True)




## === cell 9
def add_geo_features(data):
    data["abs_diff_longitude"] = (data.dropoff_longitude - data.pickup_longitude).abs()
    data["abs_diff_latitude"] = (data.dropoff_latitude - data.pickup_latitude).abs()

    data["manhattan_distance"] = data["abs_diff_longitude"] + data["abs_diff_latitude"]

    data["squared_long"] = np.power(data["abs_diff_longitude"], 2)
    data["squared_lat"] = np.power(data["abs_diff_latitude"], 2)

    data["euclid_disance"] = np.sqrt(data["squared_long"] + data["squared_lat"])

    r = 6371.0  # Earth radius (km)
    lat1 = np.deg2rad(data["pickup_latitude"].to_numpy(dtype=np.float64, copy=False))
    lon1 = np.deg2rad(data["pickup_longitude"].to_numpy(dtype=np.float64, copy=False))
    lat2 = np.deg2rad(data["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False))
    lon2 = np.deg2rad(data["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    data["haversine_km"] = (r * c).astype("float32")

    data["haversine_miles"] = (data["haversine_km"] * np.float32(0.621371)).astype(
        "float32"
    )

    data["log_haversine_miles"] = np.log1p(
        data["haversine_miles"].to_numpy(dtype=np.float32, copy=False)
    ).astype("float32")

    y_b = np.sin(dlon) * np.cos(lat2)
    x_b = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y_b, x_b)  # radians in [-pi, pi]
    data["bearing"] = bearing.astype("float32")

    pu_lat = data["pickup_latitude"]
    pu_lon = data["pickup_longitude"]
    do_lat = data["dropoff_latitude"]
    do_lon = data["dropoff_longitude"]

    jfk_pu = (pu_lat.between(40.621, 40.67)) & (pu_lon.between(-73.835, -73.75))
    jfk_do = (do_lat.between(40.621, 40.67)) & (do_lon.between(-73.835, -73.75))

    lga_pu = (pu_lat.between(40.76, 40.78)) & (pu_lon.between(-73.90, -73.86))
    lga_do = (do_lat.between(40.76, 40.78)) & (do_lon.between(-73.90, -73.86))

    ewr_pu = (pu_lat.between(40.68, 40.71)) & (pu_lon.between(-74.20, -74.16))
    ewr_do = (do_lat.between(40.68, 40.71)) & (do_lon.between(-74.20, -74.16))

    data["pickup_airport"] = (jfk_pu | lga_pu | ewr_pu).astype("uint8")
    data["dropoff_airport"] = (jfk_do | lga_do | ewr_do).astype("uint8")

    mid_lat = np.float32(40.7580)
    mid_lon = np.float32(-73.9855)
    data["pickup_midtown_manhattan"] = (
        (data["pickup_latitude"] - mid_lat).abs()
        + (data["pickup_longitude"] - mid_lon).abs()
    ).astype("float32")
    data["dropoff_midtown_manhattan"] = (
        (data["dropoff_latitude"] - mid_lat).abs()
        + (data["dropoff_longitude"] - mid_lon).abs()
    ).astype("float32")

    return data




## === cell 10
def add_time_features(data):
    dt = pd.to_datetime(data.pickup_datetime, errors="coerce", utc=True)

    data["hour"] = dt.dt.hour.astype("int16")
    data["day_of_week"] = dt.dt.day_name()
    dom = dt.dt.day.astype("int16")
    data["week_of_month"] = week_num_vec(dom)
    data["month"] = dt.dt.month.astype("int16")
    data["year"] = dt.dt.year.astype("int16")

    data["hour"] = data["hour"].astype("string")
    data["month"] = data["month"].astype("string")
    data["year"] = data["year"].astype("string")

    data["day_of_week"] = data["day_of_week"].astype("category")
    data["week_of_month"] = data["week_of_month"].astype("category")

    return data




## === cell 11
train_X = add_geo_features(train_X)
test_X = add_geo_features(test_X)



## === cell 12
bad_zero_coord_train = (
    (train_X["pickup_longitude"] == 0) & (train_X["pickup_latitude"] == 0)
) | ((train_X["dropoff_longitude"] == 0) & (train_X["dropoff_latitude"] == 0))
train_X = train_X.loc[~bad_zero_coord_train].copy()
train = train.loc[train_X.index].copy()

zero_trip_mask = (train_X["abs_diff_longitude"] == 0) & (
    train_X["abs_diff_latitude"] == 0
)
train_X = train_X.loc[~zero_trip_mask].copy()
train = train.loc[train_X.index].copy()

_dt = pd.to_datetime(train_X["pickup_datetime"], errors="coerce", utc=True)
valid_dt_mask = _dt.notna()
train_X = train_X.loc[valid_dt_mask].copy()
train = train.loc[train_X.index].copy()
_dt = _dt.loc[train_X.index]

dist_mask = train_X["haversine_miles"].between(0, 60)
train_X = train_X.loc[dist_mask].copy()
train = train.loc[train_X.index].copy()



## === cell 13
test_valid = (
    (test_X["passenger_count"].between(1, 6))
    & (test_X["pickup_latitude"].between(-90, 90))
    & (test_X["dropoff_latitude"].between(-90, 90))
    & (test_X["pickup_longitude"].between(-180, 180))
    & (test_X["dropoff_longitude"].between(-180, 180))
)
test_not_zero_coord = ~(
    ((test_X["pickup_longitude"] == 0) & (test_X["pickup_latitude"] == 0))
    | ((test_X["dropoff_longitude"] == 0) & (test_X["dropoff_latitude"] == 0))
)

test_valid_mask = test_valid & test_not_zero_coord

test_bbox = (
    (test_X.pickup_longitude.between(nyc_lon_min, nyc_lon_max))
    & (test_X.dropoff_longitude.between(nyc_lon_min, nyc_lon_max))
    & (test_X.pickup_latitude.between(nyc_lat_min, nyc_lat_max))
    & (test_X.dropoff_latitude.between(nyc_lat_min, nyc_lat_max))
)

test_model_mask = test_bbox & test_valid_mask

test_in_bbox = test_X.loc[test_model_mask].copy()
test_out_bbox_idx = test_X.index[~test_model_mask]



## === cell 14
train_X = add_time_features(train_X)
test_in_bbox = add_time_features(test_in_bbox)

train_X = train_X.dropna(axis=0, how="any")
y = train.loc[train_X.index, "fare_amount"].to_numpy()

fallback_value = float(np.median(y))


def add_estimated_fare_feature(df: pd.DataFrame) -> pd.DataFrame:
    hour_num = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype("int16")

    is_night = ((hour_num >= 20) | (hour_num <= 6)).astype("int8")
    is_rush = (
        ((hour_num >= 7) & (hour_num <= 10)) | ((hour_num >= 16) & (hour_num <= 19))
    ).astype("int8")

    miles = df["haversine_miles"].astype("float32")
    pc = df["passenger_count"].astype("float32")
    est = (
        np.float32(2.5)
        + np.float32(2.0) * miles
        + np.float32(0.25) * (pc - np.float32(1.0)).clip(lower=np.float32(0.0))
        + np.float32(0.5) * is_night.astype("float32")
        + np.float32(0.35) * is_rush.astype("float32")
    ).astype("float32")

    df["estimated_fare"] = est
    return df


train_X = add_estimated_fare_feature(train_X)
test_in_bbox = add_estimated_fare_feature(test_in_bbox)

features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "euclid_disance",
    "manhattan_distance",
    "haversine_km",
    "haversine_miles",
    "log_haversine_miles",
    "bearing",
    "pickup_airport",
    "dropoff_airport",
    "pickup_midtown_manhattan",
    "dropoff_midtown_manhattan",
    "estimated_fare",
]

train_X = train_X[features]
test_in_bbox = test_in_bbox[features]

train_X = pd.get_dummies(train_X)
test_in_bbox = pd.get_dummies(test_in_bbox)

train_X, test_in_bbox = train_X.align(test_in_bbox, join="inner", axis=1)



## === cell 15
x = train_X
x_test = test_in_bbox



## === cell 16
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1, random_state=SEED)



## === cell 17
pass



## === cell 18
pass



## === cell 19
model.fit(x, y)



## === cell 20
pass



## === cell 21
pass



## === cell 22
test_pred_in_bbox = model.predict(x_test)

test_pred_full = np.empty(shape=(len(test_X),), dtype=np.float32)
test_pred_full[:] = np.float32(fallback_value)
test_pred_full[test_model_mask.to_numpy()] = test_pred_in_bbox.astype(
    np.float32, copy=False
)

test_pred_full = np.clip(test_pred_full, 0.0, 250.0)

sub = pd.DataFrame()
sub["key"] = test_id
sub["fare_amount"] = test_pred_full
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
