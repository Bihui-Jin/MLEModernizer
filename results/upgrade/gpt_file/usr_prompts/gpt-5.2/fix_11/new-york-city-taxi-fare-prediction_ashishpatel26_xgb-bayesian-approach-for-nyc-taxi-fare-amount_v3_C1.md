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
geopy==2.4.1
lightgbm==4.6.0
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

# 5. Target score

4.04952

# 6. Current score

4.78834

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.37025) has done: 'I fix the runtime errors by (1) replacing the removed `geopy.distance.VincentyDistance` with a fast vectorized Haversine distance computation (same “distance feature” intent, but compatible and much faster), and (2) ensuring LightGBM only receives numeric feature columns (your current `key`, `pickup_datetime`, `key2` objects are causing the `bad pandas dtypes` crash). I also update the deprecated datetime accessors (`week` / `weekofyear`) to pandas-2 compatible `isocalendar().week` while keeping the same time-based feature set. Finally, I make sure the pipeline trains, predicts on test, and writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 7.99583) has done: 'You currently don’t have a measured Kaggle score for this exact script, so the safest way to move toward the target RMSE is to make small, proven feature tweaks and training settings that typically reduce error without changing the overall approach. I keep the same LightGBM regressor and the same overall feature engineering intent, but add two very lightweight numeric features (Manhattan distance in km and a straight-line bearing proxy) and ensure the train/test filtering is consistent (apply the same NYC bounding-box filter to test to avoid distribution mismatch). I also switch the LightGBM training to use the validated `best_iteration_` for the final fit via `early_stopping` on the validation split (this doesn’t change the model family or loss; it just prevents overfitting and usually improves RMSE). Finally, I guarantee the submission aligns exactly to the original test keys by predicting on an unfiltered copy of test and only using filters for feature sanity checks.'
- What this solution (achieved 5.07649) has done: 'Your current RMSE (7.99583) is far from the target (4.04952), so we need a modest-but-real modeling lift while keeping your LightGBM + engineered-features pipeline intact. The biggest issue hurting score is likely a train/test distribution mismatch caused by training on a filtered NYC/clean subset but predicting on unfiltered `test_raw` rows with imputed zeros; instead, we apply the exact same feature construction and light sanity filtering to `test_raw` while still producing predictions for every test key. Next, we use the validation-derived `best_iteration_` (via LightGBM early stopping) and then refit on all training data with `n_estimators=best_iteration_` to reduce overfitting without changing model family or loss. Finally, we make sure training and submission use identical numeric feature preparation, and we keep the submission format unchanged (`key,fare_amount`) and always 9914 rows.'
- What this solution (achieved 8.15713) has done: 'We’re still worse than the target (RMSE 5.07649 vs 4.04952, lower is better), so we should cautiously improve without changing the overall LightGBM + engineered-features approach. The biggest score drag remaining is the “fallback-to-mean for out-of-bbox rows” in the submission step, which creates unnatural constant predictions for those rows; we instead apply the *same* light sanity filters used for training (distance and fare_per_km-inspired bounds) directly as features (no row dropping) and remove the constant fallback, keeping predictions per-row. We also increase the training sample size (from 1,000,000 to 2,000,000) since this is the same model/training loop and is typically the smallest legitimate lift toward better RMSE within the 600s budget. Finally, we align train/test numeric clipping for distance to avoid extreme values dominating splits, while keeping the same feature set and LightGBM objective.'
- What this solution (achieved 8.18503) has done: 'To move your RMSE down toward the 4.049 target (current 8.157, lower is better) with minimal disruption, I’m keeping your LightGBM + engineered-distance/time-features core intact and focusing on fixing a key train/test mismatch: you currently *drop* many rows from `test` (bbox/passenger filters) before training feature stats, but then you predict on `test_raw` (unfiltered) without applying the same bbox sanity logic, causing extreme/out-of-distribution coordinates to produce bad distance features and large errors. I stop filtering `test` for bbox (keep only minimal passenger_count cleanup), and instead apply the bbox validity as an explicit numeric feature (`in_nyc_bbox`) plus safe coordinate clipping for both train/test/submission feature creation (no row dropping), so the model can learn to downweight those cases rather than generating wild distances. I also make train/test datetime handling consistent (derive all time features from the same parsed column) and keep the rest (model params, early stopping, refit) unchanged to preserve evaluation semantics and runtime.'
- What this solution (achieved 7.60488) has done: 'To move RMSE down toward the 4.04952 target (current 8.18503, lower is better) without changing your core LightGBM + distance/time-features approach, I’m making the train/test feature pipeline more consistent and removing a likely major score drag: training drops out-of-bbox rows but inference includes them, creating distribution shift. I keep your existing engineered features and model, but (1) apply the same `in_nyc_bbox` consistency to training by filtering to bbox (as is standard for this competition), (2) remove the aggressive `fare_per_km` filter that can delete too many legitimate rides and harm generalization, and (3) cap extreme distances with a `log1p(distance)` feature to stabilize large outliers while keeping all existing features intact. These are minimal, metric-aligned changes that typically reduce RMSE substantially while preserving the overall logic and runtime constraints, and the script still writes a valid `submission.csv` with 9914 rows and the required columns.'
- What this solution (achieved 4.78834) has done: 'Your current RMSE (7.60488) is far above the target (4.04952), so we should make a small, metric-aligned improvement without changing the LightGBM + engineered distance/time-features core. The biggest low-risk lift here is to remove train/test mismatch created by training only on `in_nyc_bbox==1` while still predicting for all test rows (including out-of-bbox), by keeping all rows and letting `in_nyc_bbox` act as a feature rather than a hard filter. Second, we add one standard NYC Taxi feature with minimal disruption: distance to NYC center for pickup/dropoff (helps handle spatial priors and reduces error noticeably). Finally, we keep the same early-stopping + refit flow but make the train/validation split deterministic and slightly more representative by stratifying on `in_nyc_bbox` (no new training approach, just a safer split).'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("fivethirtyeight")

import geopy.distance  # kept to preserve original core intent, though we will not use deprecated Vincenty

import os
import gc

from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from lightgbm import early_stopping
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

INPUT_DIR_CANDIDATES = [
    "../input",  # original
    "/kaggle/input",  # standard Kaggle
    "/kaggle/data",  # provided in this environment listing
]
INPUT_DIR = None
for d in INPUT_DIR_CANDIDATES:
    if os.path.exists(d):
        INPUT_DIR = d
        break
if INPUT_DIR is None:
    INPUT_DIR = "../input"

print("Using INPUT_DIR =", INPUT_DIR)
if os.path.exists(INPUT_DIR):
    print("Top-level input contents:", os.listdir(INPUT_DIR)[:20])




## === cell 1
def load_Data():
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_path):
        train_path = os.path.join(
            INPUT_DIR, "new-york-city-taxi-fare-prediction", "train.csv"
        )
    if not os.path.exists(test_path):
        test_path = os.path.join(
            INPUT_DIR, "new-york-city-taxi-fare-prediction", "test.csv"
        )

    train = pd.read_csv(train_path, nrows=2_000_000, low_memory=True)
    test = pd.read_csv(test_path, low_memory=True)  # test is small anyway
    return train, test




## === cell 2
train, test = load_Data()



## === cell 3
train.head(5)



## === cell 4
train.describe()



## === cell 5
train.info()



## === cell 6
train.isnull().sum()



## === cell 7
req_train = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
train = train.dropna(subset=req_train).copy()

req_test = [
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_raw = test.copy()
test = test.dropna(subset=req_test).copy()



## === cell 8
train["key2"] = pd.to_datetime(train["pickup_datetime"], errors="coerce", utc=True)
test["key2"] = pd.to_datetime(test["pickup_datetime"], errors="coerce", utc=True)
test_raw["key2"] = pd.to_datetime(
    test_raw["pickup_datetime"], errors="coerce", utc=True
)

train = train.dropna(subset=["key2"]).copy()
test = test.dropna(subset=["key2"]).copy()
train.info()



## === cell 9
train["fare_amount"].plot(kind="box")



## === cell 10
gc.collect()
train.describe()



## === cell 11
print(
    "% of fares above 25$ - {:0.2f}".format(
        train[train["fare_amount"] > 25]["key"].count() * 100 / train["key"].count()
    )
)
print(
    "% of fares above 50$ - {:0.2f}".format(
        train[train["fare_amount"] > 50]["key"].count() * 100 / train["key"].count()
    )
)
print(
    "% of fares above 100$ - {:0.2f}".format(
        train[train["fare_amount"] > 100]["key"].count() * 100 / train["key"].count()
    )
)
print(
    "% of fares below 0$ - {:0.2f}".format(
        train[train["fare_amount"] < 0]["key"].count() * 100 / train["key"].count()
    )
)



## === cell 12
fig, axarr = plt.subplots(2, 2, figsize=(20, 10))

train[~(train["fare_amount"] > 25)]["fare_amount"].plot(kind="box", ax=axarr[0][0])
train[~(train["fare_amount"] > 50)]["fare_amount"].plot(kind="box", ax=axarr[0][1])
train[~(train["fare_amount"] > 100)]["fare_amount"].plot(kind="box", ax=axarr[1][0])
train[~(train["fare_amount"] < 0)]["fare_amount"].plot(kind="box", ax=axarr[1][1])



## === cell 13
train["passenger_count"].plot(kind="box")



## === cell 14
print(
    "Count of invalid pickup latitude",
    train[(train["pickup_latitude"] > 90) | (train["pickup_latitude"] < -90)][
        "pickup_latitude"
    ].count(),
)
print(
    "Count of invalid dropoff latitude",
    train[(train["dropoff_latitude"] > 90) | (train["dropoff_latitude"] < -90)][
        "dropoff_latitude"
    ].count(),
)
print(
    "Count of invalid pickup longitude",
    train[(train["pickup_longitude"] > 180) | (train["pickup_longitude"] < -180)][
        "pickup_longitude"
    ].count(),
)
print(
    "Count of invalid dropoff longitude",
    train[(train["dropoff_longitude"] > 180) | (train["dropoff_longitude"] < -180)][
        "dropoff_longitude"
    ].count(),
)



## === cell 15
print(
    "Count of invalid pickup latitude",
    test[(test["pickup_latitude"] > 90) | (test["pickup_latitude"] < -90)][
        "pickup_latitude"
    ].count(),
)
print(
    "Count of invalid dropoff latitude",
    test[(test["dropoff_latitude"] > 90) | (test["dropoff_latitude"] < -90)][
        "dropoff_latitude"
    ].count(),
)
print(
    "Count of invalid pickup longitude",
    test[(test["pickup_longitude"] > 180) | (test["pickup_longitude"] < -180)][
        "pickup_longitude"
    ].count(),
)
print(
    "Count of invalid dropoff longitude",
    test[(test["dropoff_longitude"] > 180) | (test["dropoff_longitude"] < -180)][
        "dropoff_longitude"
    ].count(),
)



## === cell 16
nyc_min_lon, nyc_max_lon = -74.5, -72.8
nyc_min_lat, nyc_max_lat = 40.0, 41.8


def in_nyc_bbox(df):
    return (
        df["pickup_longitude"].between(nyc_min_lon, nyc_max_lon)
        & df["dropoff_longitude"].between(nyc_min_lon, nyc_max_lon)
        & df["pickup_latitude"].between(nyc_min_lat, nyc_max_lat)
        & df["dropoff_latitude"].between(nyc_min_lat, nyc_max_lat)
    ).astype("int8")


train = train[(train["fare_amount"] > 0) & (train["fare_amount"] <= 250)].copy()
train = train[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 6)].copy()

train = train[~((train["pickup_latitude"] > 90) | (train["pickup_latitude"] < -90))]
train = train[~((train["dropoff_latitude"] > 90) | (train["dropoff_latitude"] < -90))]
train = train[~((train["pickup_longitude"] > 180) | (train["pickup_longitude"] < -180))]
train = train[
    ~((train["dropoff_longitude"] > 180) | (train["dropoff_longitude"] < -180))
].copy()

test = test.copy()
test["passenger_count"] = (
    pd.to_numeric(test["passenger_count"], errors="coerce").fillna(1).clip(1, 6)
)

train["in_nyc_bbox"] = in_nyc_bbox(train)
test["in_nyc_bbox"] = in_nyc_bbox(test)





## === cell 17
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0088 * c  # mean Earth radius in km


def manhattan_km(lon1, lat1, lon2, lat2):
    lat1r = np.radians(lat1.astype(float))
    lat2r = np.radians(lat2.astype(float))
    dlat = np.abs(lat2r - lat1r)
    dlon = np.abs(np.radians(lon2.astype(float)) - np.radians(lon1.astype(float)))
    mean_lat = 0.5 * (lat1r + lat2r)
    km_per_rad = 6371.0088
    return km_per_rad * dlat + km_per_rad * np.cos(mean_lat) * dlon


def bearing_sin_cos(lon1, lat1, lon2, lat2):
    lon1r = np.radians(lon1.astype(float))
    lat1r = np.radians(lat1.astype(float))
    lon2r = np.radians(lon2.astype(float))
    lat2r = np.radians(lat2.astype(float))
    dlon = lon2r - lon1r
    y = np.sin(dlon) * np.cos(lat2r)
    x = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    brng = np.arctan2(y, x)  # [-pi, pi]
    return np.sin(brng), np.cos(brng)


NYC_CENTER_LON, NYC_CENTER_LAT = -73.985428, 40.748817  # Midtown Manhattan approx



## === cell 18
for df in (train, test):
    for c, lo, hi in [
        ("pickup_longitude", -180.0, 180.0),
        ("dropoff_longitude", -180.0, 180.0),
        ("pickup_latitude", -90.0, 90.0),
        ("dropoff_latitude", -90.0, 90.0),
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].clip(lo, hi)

train["distance"] = haversine_km(
    train["pickup_longitude"].values,
    train["pickup_latitude"].values,
    train["dropoff_longitude"].values,
    train["dropoff_latitude"].values,
)



## === cell 19
train = train[train["distance"] > 0.01].copy()
gc.collect()



## === cell 20
train["abs_lon_diff"] = (train["pickup_longitude"] - train["dropoff_longitude"]).abs()
train["abs_lat_diff"] = (train["pickup_latitude"] - train["dropoff_latitude"]).abs()

train["manhattan"] = manhattan_km(
    train["pickup_longitude"].values,
    train["pickup_latitude"].values,
    train["dropoff_longitude"].values,
    train["dropoff_latitude"].values,
)
b_sin, b_cos = bearing_sin_cos(
    train["pickup_longitude"].values,
    train["pickup_latitude"].values,
    train["dropoff_longitude"].values,
    train["dropoff_latitude"].values,
)
train["bearing_sin"] = b_sin
train["bearing_cos"] = b_cos
del b_sin, b_cos

train["log_distance"] = np.log1p(train["distance"].clip(lower=0))

train["pickup_center_dist"] = haversine_km(
    train["pickup_longitude"].values,
    train["pickup_latitude"].values,
    np.full(len(train), NYC_CENTER_LON),
    np.full(len(train), NYC_CENTER_LAT),
)
train["dropoff_center_dist"] = haversine_km(
    train["dropoff_longitude"].values,
    train["dropoff_latitude"].values,
    np.full(len(train), NYC_CENTER_LON),
    np.full(len(train), NYC_CENTER_LAT),
)



## === cell 21
train.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 22
test["distance"] = haversine_km(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
)



## === cell 23
test["abs_lon_diff"] = (test["pickup_longitude"] - test["dropoff_longitude"]).abs()
test["abs_lat_diff"] = (test["pickup_latitude"] - test["dropoff_latitude"]).abs()

test["manhattan"] = manhattan_km(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
)
b_sin, b_cos = bearing_sin_cos(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
)
test["bearing_sin"] = b_sin
test["bearing_cos"] = b_cos
del b_sin, b_cos

test["log_distance"] = np.log1p(test["distance"].clip(lower=0))

test["pickup_center_dist"] = haversine_km(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    np.full(len(test), NYC_CENTER_LON),
    np.full(len(test), NYC_CENTER_LAT),
)
test["dropoff_center_dist"] = haversine_km(
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
    np.full(len(test), NYC_CENTER_LON),
    np.full(len(test), NYC_CENTER_LAT),
)



## === cell 24
test.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 25
print(
    "% of trips above 25 KM - {:0.2f}".format(
        train[train["distance"] > 25]["key"].count() * 100 / train["key"].count()
    )
)



## === cell 26
train["year"] = train["key2"].dt.year
train["month"] = train["key2"].dt.month
train["day"] = train["key2"].dt.day
train["day of week"] = train["key2"].dt.weekday
train["hour"] = train["key2"].dt.hour
train["week"] = train["key2"].dt.isocalendar().week.astype("int16")
train["day_of_year"] = train["key2"].dt.dayofyear
train["week_of_year"] = train["key2"].dt.isocalendar().week.astype("int16")
train["quarter"] = train["key2"].dt.quarter



## === cell 27
train.columns



## === cell 28
test["year"] = test["key2"].dt.year
test["month"] = test["key2"].dt.month
test["day"] = test["key2"].dt.day
test["day of week"] = test["key2"].dt.weekday
test["hour"] = test["key2"].dt.hour
test["week"] = test["key2"].dt.isocalendar().week.astype("int16")
test["day_of_year"] = test["key2"].dt.dayofyear
test["week_of_year"] = test["key2"].dt.isocalendar().week.astype("int16")
test["quarter"] = test["key2"].dt.quarter



## === cell 29
column_list = [
    "passenger_count",
    "distance",
    "log_distance",
    "abs_lon_diff",
    "abs_lat_diff",
    "manhattan",
    "bearing_sin",
    "bearing_cos",
    "in_nyc_bbox",
    "pickup_center_dist",
    "dropoff_center_dist",
    "year",
    "month",
    "day",
    "day of week",
    "hour",
    "week",
    "day_of_year",
    "week_of_year",
    "quarter",
]
y_train_ = train["fare_amount"]
X_train_ = train[column_list].copy()

X_test = test[column_list].copy()

for c in column_list:
    X_train_[c] = pd.to_numeric(X_train_[c], errors="coerce").fillna(0)
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce").fillna(0)

X_train_["distance"] = X_train_["distance"].clip(0, 50)
X_test["distance"] = X_test["distance"].clip(0, 50)
X_train_["manhattan"] = X_train_["manhattan"].clip(0, 80)
X_test["manhattan"] = X_test["manhattan"].clip(0, 80)

X_train_["pickup_center_dist"] = X_train_["pickup_center_dist"].clip(0, 200)
X_train_["dropoff_center_dist"] = X_train_["dropoff_center_dist"].clip(0, 200)
X_test["pickup_center_dist"] = X_test["pickup_center_dist"].clip(0, 200)
X_test["dropoff_center_dist"] = X_test["dropoff_center_dist"].clip(0, 200)

X_train_.shape, y_train_.shape, X_test.shape



## === cell 30
X_train, X_val, y_train, y_val = train_test_split(
    X_train_, y_train_, test_size=0.1, random_state=42, stratify=X_train_["in_nyc_bbox"]
)



## === cell 31
X_train.shape, X_val.shape, y_train.shape, y_val.shape



## === cell 32
lgb = LGBMRegressor(
    boosting_type="gbdt",
    class_weight=None,
    colsample_bytree=0.9,
    learning_rate=0.1,
    max_depth=8,
    min_child_samples=55,
    min_child_weight=0.001,
    min_split_gain=0.1,
    n_estimators=500,
    n_jobs=-1,
    num_leaves=45,
    objective="regression",
    random_state=42,  # score-stabilizing; does not change core logic
    reg_alpha=5.0,
    reg_lambda=3.0,
    subsample=1.0,
    subsample_for_bin=200000,
    subsample_freq=1,
)



## === cell 33
import time

t0 = time.time()
lgb.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
)
print("Fit time (s):", round(time.time() - t0, 2))
print("best_iteration_:", getattr(lgb, "best_iteration_", None))



## === cell 34
t0 = time.time()
pred = lgb.predict(X_val, num_iteration=lgb.best_iteration_)
print("Predict time (s):", round(time.time() - t0, 2))



## === cell 35
print(r2_score(y_val, pred))
print(np.sqrt(mean_squared_error(y_val, pred)))



## === cell 36
best_n_estimators = (
    int(lgb.best_iteration_) if getattr(lgb, "best_iteration_", None) else 500
)

lgb_final = LGBMRegressor(
    boosting_type="gbdt",
    class_weight=None,
    colsample_bytree=0.9,
    learning_rate=0.1,
    max_depth=8,
    min_child_samples=55,
    min_child_weight=0.001,
    min_split_gain=0.1,
    n_estimators=best_n_estimators,
    n_jobs=-1,
    num_leaves=45,
    objective="regression",
    random_state=42,
    reg_alpha=5.0,
    reg_lambda=3.0,
    subsample=1.0,
    subsample_for_bin=200000,
    subsample_freq=1,
)

t0 = time.time()
lgb_final.fit(X_train_, y_train_)
print("Full fit time (s):", round(time.time() - t0, 2))
print("Using n_estimators:", best_n_estimators)




## === cell 37
def display_importances(feature_importance_df_, doWorst=False, n_feat=50):
    if not doWorst:
        cols = (
            feature_importance_df_[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:n_feat]
            .index
        )
    else:
        cols = (
            feature_importance_df_[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[-n_feat:]
            .index
        )

    mean_imp = (
        feature_importance_df_[["feature", "importance"]].groupby("feature").mean()
    )
    df_2_neglect = mean_imp[mean_imp["importance"] < 1e-3]
    print("The list of features with ~0 importance: ")
    print(df_2_neglect.index.values.tolist())
    del mean_imp, df_2_neglect

    best_features = feature_importance_df_.loc[
        feature_importance_df_.feature.isin(cols)
    ]

    plt.figure(figsize=(8, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features.sort_values(by="importance", ascending=False),
    )
    plt.title("LightGBM Features")
    plt.tight_layout()
    plt.savefig("lgbm_importances.png")


importance_df = pd.DataFrame()
importance_df["feature"] = column_list
importance_df["importance"] = lgb_final.feature_importances_
display_importances(feature_importance_df_=importance_df, n_feat=20)



## === cell 38
test_sub = test_raw.copy()

for col in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]:
    test_sub[col] = pd.to_numeric(test_sub[col], errors="coerce")

test_sub["passenger_count"] = test_sub["passenger_count"].fillna(1).clip(1, 6)

missing_dt = test_sub["key2"].isna()
if missing_dt.any():
    test_sub.loc[missing_dt, "key2"] = pd.Timestamp("2015-01-01", tz="UTC")

for c, lo, hi in [
    ("pickup_longitude", -180.0, 180.0),
    ("dropoff_longitude", -180.0, 180.0),
    ("pickup_latitude", -90.0, 90.0),
    ("dropoff_latitude", -90.0, 90.0),
]:
    test_sub[c] = test_sub[c].clip(lo, hi)

test_sub["in_nyc_bbox"] = in_nyc_bbox(test_sub)

test_sub["distance"] = haversine_km(
    test_sub["pickup_longitude"].values,
    test_sub["pickup_latitude"].values,
    test_sub["dropoff_longitude"].values,
    test_sub["dropoff_latitude"].values,
)
test_sub["abs_lon_diff"] = (
    test_sub["pickup_longitude"] - test_sub["dropoff_longitude"]
).abs()
test_sub["abs_lat_diff"] = (
    test_sub["pickup_latitude"] - test_sub["dropoff_latitude"]
).abs()
test_sub["manhattan"] = manhattan_km(
    test_sub["pickup_longitude"].values,
    test_sub["pickup_latitude"].values,
    test_sub["dropoff_longitude"].values,
    test_sub["dropoff_latitude"].values,
)
b_sin, b_cos = bearing_sin_cos(
    test_sub["pickup_longitude"].values,
    test_sub["pickup_latitude"].values,
    test_sub["dropoff_longitude"].values,
    test_sub["dropoff_latitude"].values,
)
test_sub["bearing_sin"] = b_sin
test_sub["bearing_cos"] = b_cos
del b_sin, b_cos

test_sub["log_distance"] = np.log1p(test_sub["distance"].clip(lower=0))

test_sub["pickup_center_dist"] = haversine_km(
    test_sub["pickup_longitude"].values,
    test_sub["pickup_latitude"].values,
    np.full(len(test_sub), NYC_CENTER_LON),
    np.full(len(test_sub), NYC_CENTER_LAT),
)
test_sub["dropoff_center_dist"] = haversine_km(
    test_sub["dropoff_longitude"].values,
    test_sub["dropoff_latitude"].values,
    np.full(len(test_sub), NYC_CENTER_LON),
    np.full(len(test_sub), NYC_CENTER_LAT),
)

test_sub["year"] = test_sub["key2"].dt.year
test_sub["month"] = test_sub["key2"].dt.month
test_sub["day"] = test_sub["key2"].dt.day
test_sub["day of week"] = test_sub["key2"].dt.weekday
test_sub["hour"] = test_sub["key2"].dt.hour
test_sub["week"] = test_sub["key2"].dt.isocalendar().week.astype("int16")
test_sub["day_of_year"] = test_sub["key2"].dt.dayofyear
test_sub["week_of_year"] = test_sub["key2"].dt.isocalendar().week.astype("int16")
test_sub["quarter"] = test_sub["key2"].dt.quarter

X_sub = test_sub[column_list].copy()
for c in column_list:
    X_sub[c] = pd.to_numeric(X_sub[c], errors="coerce").fillna(0)

X_sub["distance"] = X_sub["distance"].clip(0, 50)
X_sub["manhattan"] = X_sub["manhattan"].clip(0, 80)
X_sub["pickup_center_dist"] = X_sub["pickup_center_dist"].clip(0, 200)
X_sub["dropoff_center_dist"] = X_sub["dropoff_center_dist"].clip(0, 200)

y_pred = lgb_final.predict(X_sub)
y_pred = np.maximum(y_pred.astype(float), 0.0)

submission = pd.DataFrame({"key": test_sub["key"], "fare_amount": y_pred})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Submission columns:", submission.columns.tolist())
print("Any NaN preds:", int(np.isnan(y_pred).sum()))
