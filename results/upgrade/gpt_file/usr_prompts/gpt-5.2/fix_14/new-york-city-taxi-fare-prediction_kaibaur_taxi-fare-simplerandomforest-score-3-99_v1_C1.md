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

3.11

# 3. Installed packages

geopandas==0.14.4
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
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

4.377

# 6. Current score

5.57701

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.60476) has done: 'I fix the runtime error by ensuring the non-numeric `key` column is never passed into LightGBM at prediction time (it’s present in `test_feat` but not in `train_feat`). I also make the feature-building step explicitly align test features to the training feature columns, which prevents silent column-order/column-set mismatches. Finally, I keep the same model and training logic, and ensure the pipeline always writes a valid `submission.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 6.17297) has done: 'Your current RMSE (6.60) is worse than the target (4.377), so we should make the smallest legitimate changes that usually improve this classic NYC Taxi baseline without changing the model type or training loop. The biggest gain for minimal risk is to add a few standard, cheap geospatial/time features (straight-line distance, bearing-like directionality, and “NYC bounding box” cleanup) while keeping LightGBM and the same fit/predict flow. I also ensure train/test get identical feature columns, and I keep the submission-writing logic intact. These changes typically move RMSE meaningfully downward while preserving your core approach.'
- What this solution (achieved 8.24514) has done: 'We keep your LightGBM regressor and the same fit/predict flow, but make a few minimal feature/cleaning tweaks that are well-known to reduce RMSE on this competition. Specifically: (1) add a couple of cheap, standard geospatial features (pickup/dropoff distance to NYC center and a simple “jfk-ish”/“ewr-ish” flag) without changing the modeling approach; (2) slightly improve training cleaning by removing obviously-bad zero-coordinate points (which otherwise inject noise); and (3) ensure identical feature columns between train/test as you already do. These changes should improve generalization enough to move RMSE downward toward your 4.377 target while staying within the same core logic.'
- What this solution (achieved 5.92126) has done: 'Your current RMSE (8.245) is far worse than the target (4.377), so the smallest reliable improvement is to fix a likely feature/scale bug rather than change the model. The main issue is that `manhattan_km` is computed from degree deltas using a constant 111 for both lat and lon; longitude degrees must be scaled by `cos(latitude)`, otherwise distances are badly distorted and the model generalizes poorly. I replace `manhattan_km` with a latitude-aware km approximation (still the same feature idea: “Manhattan distance”), keep all other core logic intact, and also add a very small, standard training cleanup to drop rides with near-zero distance but non-trivial fare (a common noise source). This preserves your LightGBM training flow and submission format, but should move RMSE downward toward the target.'
- What this solution (achieved 5.3761) has done: 'We need to move RMSE down from 5.921 toward 4.377 (lower is better), so we make minimal, well-known improvements without changing the model type or training flow. The biggest safe gain is to fix target skew and outliers by training on `log1p(fare_amount)` and inverting with `expm1` at prediction time (same regressor, same fit/predict loop, same metric semantics but typically much better RMSE here). We also add one standard geospatial feature (`pickup_dropoff_manhattan_km` already exists, but we add a simple `haversine_km_sq` and `manhattan_km_sq` to help nonlinear scaling without changing core logic) and apply the same NYC bounding-box cleanup to test features only via clipping (not dropping rows) to avoid distribution mismatch. Submission writing stays identical and still outputs `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.40003) has done: 'To move RMSE down from 5.3761 toward the 4.377 target (lower is better) without changing your model/training loop, the smallest reliable gain is to remove a bit more label noise/outliers that disproportionately hurt RMSE in this competition. I add two very standard, minimal training-only filters: drop rides with near-zero distance (haversine) but non-trivial fare, and drop implausible “too-fast” trips by computing an approximate trip speed using the existing timestamps and haversine distance. I also add a single cyclical encoding for hour-of-day (sin/cos) to better model time effects while keeping the same LightGBM regressor and fit/predict flow. Submission writing and feature alignment remain unchanged.'
- What this solution (achieved 5.59758) has done: 'We need to move RMSE down from 5.40003 toward 4.377 (lower is better), so the smallest reliable gain without changing your model/training loop is to fix a subtle but important inconsistency: you apply the “near-zero distance but non-trivial fare” filter using `haversine_km` computed while `fare_amount` is still present, then later transform the target—this is fine—but you never apply the “too-fast trip” filter you described (it’s computed but not used), leaving label-noise that hurts RMSE. I minimally add a standard speed-based training-only filter using your already-available `pickup_datetime` and `haversine_km` (no new model logic), and I also ensure we don’t accidentally leak time info misaligned with filtered rows by computing duration from the original `train_df` before feature-drop and then aligning via index. Everything else (features, LightGBM settings, log1p target, submission writing) stays the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 5.23367) has done: 'To move RMSE down from 5.59758 toward 4.377 (lower is better) with minimal disruption, I keep your LightGBM + log1p target and the same fit/predict flow, but fix a key feature mismatch: you currently clip test coordinates to the NYC box while dropping out-of-box rows in train, which creates an artificial “edge pile-up” in test that the model never sees in train. I instead apply the same bounding-box cleanup to test by snapping out-of-box coordinates to the nearest boundary *only when they are out of bounds*, and I add a single binary “was_clipped” indicator per coordinate so the model can learn that these are lower-confidence rows (this is still just feature engineering, not a model change). I also make the speed/noise filter actually use trip duration (seconds) rather than the fare/distance proxy, which is a minimal correctness fix that typically reduces label noise and improves RMSE. Submission writing remains unchanged and still produces `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.60962) has done: 'Your RMSE (5.23367) is worse than the target (4.377), so we should make the smallest change that typically improves generalization without changing the model type or training loop. The biggest issue in your current pipeline is the “speed_kmh” filter: it uses `diff()` between consecutive rows’ pickup times (not the trip duration), which is essentially random with respect to each ride and can drop many valid rows while keeping noise—this hurts RMSE. I remove that incorrect speed filter and replace it with a minimal, standard, per-row outlier filter that uses only already-available features (distance + fare) to remove unrealistic fare-per-km extremes. Everything else (LightGBM regressor, log1p target, feature set, submission writing) stays the same, and the script still writes `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.60962) has done: 'Your current RMSE (5.60962) is worse than the target (4.377), so we should make a small, reliable improvement without changing the LightGBM model or the overall fit/predict flow. The biggest low-risk gain here is to fix a train/test inconsistency: you create clipping flags only for test, but those columns don’t exist in train, so they get dropped by reindexing and the model can’t use them—either remove them or (better) add the same flags to train without clipping. I also make the outlier filtering use the filtered `fare`/`dist` vectors (currently `fare_per_km` is computed from stale `fare`/`dist` after you subset once), which can silently keep bad rows and hurt RMSE. These changes preserve your core logic (same features, same model, same log1p target, same submission format) while typically moving RMSE downward toward your target.'
- What this solution (achieved 5.60962) has done: 'We need to move RMSE down from 5.60962 toward the 4.377 target (lower is better), so the smallest reliable gains are (1) fixing a train/test inconsistency where you clip coordinates for both train and test but only clip `passenger_count` for test, and (2) preventing obvious test-time outliers (zeros/extreme coords) from turning into pathological distances by applying the same “non-zero coords” cleanup via clipping+flags rather than dropping test rows. I keep your LightGBM regressor, log1p target, fit/predict flow, and all existing engineered features, but make the preprocessing consistent and slightly less noisy. I also ensure the feature columns remain perfectly aligned and the script still writes a valid `submission.csv` with `key,fare_amount`. These changes typically reduce noise and improve generalization without changing the modeling approach.'
- What this solution (achieved 5.57701) has done: 'Your RMSE (5.60962) is still worse than the target (4.377), so we should make a small, reliable improvement without changing the LightGBM model or the overall fit/predict flow. The biggest minimal win here is to reduce label noise by adding one standard NYC Taxi cleaning step: removing implausible “long-distance but tiny fare” and “very large fare-per-km” rows using the already-computed `haversine_km` (this complements your existing outlier filters). I also add two very common, cheap geospatial interaction features (`jfk_trip` and `ewr_trip`) derived from your existing airport distance features, which usually helps the model capture airport flat-fee behavior without changing architecture. Everything else (log1p target, features, model params, submission writing) stays the same, and it still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly

import plotly.offline as offline
import plotly.graph_objs as go

offline.init_notebook_mode()

import datetime as dt
import lightgbm as lgbm

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import io
import os
import gc



## === cell 1
BASE_DIRS = [
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/data/new-york-city-taxi-fare-prediction",
    "/kaggle/input/new-york-city-taxi-fare-prediction",
]


def find_file(filename: str) -> str:
    for d in BASE_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    if os.path.exists(filename):
        return filename
    raise FileNotFoundError(f"Could not find {filename} in {BASE_DIRS} or CWD")


train_path = find_file("train.csv")
test_path = find_file("test.csv")
sample_path = find_file("sample_submission.csv")

train_path, test_path, sample_path



## === cell 2
usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

NROWS = 2_000_000

train_df = pd.read_csv(
    train_path,
    usecols=usecols,
    nrows=NROWS,
    parse_dates=["pickup_datetime"],
)

test_df = pd.read_csv(
    test_path,
    parse_dates=["pickup_datetime"],
)

train_df.shape, test_df.shape




## === cell 3
def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().copy()
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 300)]
    df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)]

    df = df[
        (df["pickup_longitude"] != 0)
        & (df["pickup_latitude"] != 0)
        & (df["dropoff_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
    ]

    df = df[
        (df["pickup_longitude"].between(-74.3, -73.7))
        & (df["dropoff_longitude"].between(-74.3, -73.7))
        & (df["pickup_latitude"].between(40.5, 41.0))
        & (df["dropoff_latitude"].between(40.5, 41.0))
    ]
    return df


train_df = clean_train(train_df)
train_df.shape




## === cell 4
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0088 * c
    return km


def _clip_with_flag(s: pd.Series, lo: float, hi: float, name: str):
    clipped = s.clip(lo, hi)
    flag = (clipped != s).astype(np.int8)
    return clipped, flag.rename(f"{name}_was_clipped")


def add_features(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.copy()

    df["passenger_count"] = df["passenger_count"].clip(1, 6)

    for c in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]:
        if c in df.columns:
            zero_flag = (df[c] == 0).astype(np.int8)
            df[f"{c}_was_zero"] = zero_flag

    df["pickup_longitude"], plon_flag = _clip_with_flag(
        df["pickup_longitude"], -74.3, -73.7, "pickup_longitude"
    )
    df["dropoff_longitude"], dlon_flag = _clip_with_flag(
        df["dropoff_longitude"], -74.3, -73.7, "dropoff_longitude"
    )
    df["pickup_latitude"], plat_flag = _clip_with_flag(
        df["pickup_latitude"], 40.5, 41.0, "pickup_latitude"
    )
    df["dropoff_latitude"], dlat_flag = _clip_with_flag(
        df["dropoff_latitude"], 40.5, 41.0, "dropoff_latitude"
    )

    df["pickup_was_clipped"] = (plon_flag | plat_flag).astype(np.int8)
    df["dropoff_was_clipped"] = (dlon_flag | dlat_flag).astype(np.int8)
    df["any_coord_was_clipped"] = (
        df["pickup_was_clipped"] | df["dropoff_was_clipped"]
    ).astype(np.int8)

    dtcol = df["pickup_datetime"]

    df["pickup_year"] = dtcol.dt.year.astype(np.int16)
    df["pickup_month"] = dtcol.dt.month.astype(np.int8)
    df["pickup_day"] = dtcol.dt.day.astype(np.int8)
    df["pickup_hour"] = dtcol.dt.hour.astype(np.int8)
    df["pickup_weekday"] = dtcol.dt.weekday.astype(np.int8)

    hour = df["pickup_hour"].astype(np.float32).values
    df["pickup_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    df["pickup_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)

    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()

    df["haversine_km"] = haversine_np(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    ).astype(np.float32)

    mean_lat_rad = np.radians(
        ((df["pickup_latitude"].values + df["dropoff_latitude"].values) * 0.5).astype(
            np.float64
        )
    )
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(mean_lat_rad)
    df["manhattan_km"] = (
        df["abs_lat_diff"].values * km_per_deg_lat
        + df["abs_lon_diff"].values * km_per_deg_lon
    ).astype(np.float32)

    df["lon_diff"] = (df["dropoff_longitude"] - df["pickup_longitude"]).astype(
        np.float32
    )
    df["lat_diff"] = (df["dropoff_latitude"] - df["pickup_latitude"]).astype(np.float32)
    df["euclidean_deg"] = np.sqrt(
        df["abs_lon_diff"] ** 2 + df["abs_lat_diff"] ** 2
    ).astype(np.float32)
    df["bearing"] = np.arctan2(df["lon_diff"].values, df["lat_diff"].values).astype(
        np.float32
    )

    df["haversine_per_passenger"] = (df["haversine_km"] / df["passenger_count"]).astype(
        np.float32
    )

    df["haversine_km_sq"] = (df["haversine_km"] ** 2).astype(np.float32)
    df["manhattan_km_sq"] = (df["manhattan_km"] ** 2).astype(np.float32)

    NYC_CENTER_LON, NYC_CENTER_LAT = -73.985428, 40.748817  # Manhattan-ish
    JFK_LON, JFK_LAT = -73.778139, 40.641311
    EWR_LON, EWR_LAT = -74.174462, 40.689531

    df["pickup_dist_center_km"] = haversine_np(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        np.full(len(df), NYC_CENTER_LON),
        np.full(len(df), NYC_CENTER_LAT),
    ).astype(np.float32)

    df["dropoff_dist_center_km"] = haversine_np(
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
        np.full(len(df), NYC_CENTER_LON),
        np.full(len(df), NYC_CENTER_LAT),
    ).astype(np.float32)

    df["pickup_dist_jfk_km"] = haversine_np(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        np.full(len(df), JFK_LON),
        np.full(len(df), JFK_LAT),
    ).astype(np.float32)
    df["dropoff_dist_jfk_km"] = haversine_np(
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
        np.full(len(df), JFK_LON),
        np.full(len(df), JFK_LAT),
    ).astype(np.float32)

    df["pickup_dist_ewr_km"] = haversine_np(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        np.full(len(df), EWR_LON),
        np.full(len(df), EWR_LAT),
    ).astype(np.float32)
    df["dropoff_dist_ewr_km"] = haversine_np(
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
        np.full(len(df), EWR_LON),
        np.full(len(df), EWR_LAT),
    ).astype(np.float32)

    df["near_jfk"] = (
        (df["pickup_dist_jfk_km"] < 2.0) | (df["dropoff_dist_jfk_km"] < 2.0)
    ).astype(np.int8)
    df["near_ewr"] = (
        (df["pickup_dist_ewr_km"] < 2.0) | (df["dropoff_dist_ewr_km"] < 2.0)
    ).astype(np.int8)

    df["jfk_trip"] = (
        (df["pickup_dist_jfk_km"] < 2.0) | (df["dropoff_dist_jfk_km"] < 2.0)
    ).astype(np.int8)
    df["ewr_trip"] = (
        (df["pickup_dist_ewr_km"] < 2.0) | (df["dropoff_dist_ewr_km"] < 2.0)
    ).astype(np.int8)

    drop_cols = ["pickup_datetime"]
    if "key" in df.columns:
        drop_cols.append("key")
    df = df.drop(columns=drop_cols)
    return df


train_feat = add_features(train_df, is_train=True)
test_feat = add_features(test_df, is_train=False)

if "fare_amount" in train_feat.columns:
    dist = train_feat["haversine_km"].astype(np.float32)
    fare = train_feat["fare_amount"].astype(np.float32)

    mask_bad = (dist < 0.05) & (fare > 7.0)
    train_feat = train_feat.loc[~mask_bad].copy()

    dist = train_feat["haversine_km"].astype(np.float32)
    fare = train_feat["fare_amount"].astype(np.float32)
    fare_per_km = fare / (dist + 0.3)  # +0.3 reduces blow-up for short trips
    train_feat = train_feat.loc[fare_per_km.between(1.0, 80.0)].copy()

    dist = train_feat["haversine_km"].astype(np.float32)
    fare = train_feat["fare_amount"].astype(np.float32)
    train_feat = train_feat.loc[~((dist > 20.0) & (fare < 5.0))].copy()

    fare_per_km = fare / (dist + 0.3)
    train_feat = train_feat.loc[fare_per_km <= 60.0].copy()

if "haversine_km" in train_feat.columns:
    train_feat = train_feat.loc[train_feat["haversine_km"] <= 60.0].copy()

target = np.log1p(train_feat["fare_amount"].astype(np.float32))
train_feat = train_feat.drop(columns=["fare_amount"])

feature_cols = train_feat.columns.tolist()
test_feat = test_feat.reindex(columns=feature_cols)

len(feature_cols), feature_cols[:12]



## === cell 5
X_train, X_valid, y_train, y_valid = train_test_split(
    train_feat, target, test_size=0.1, random_state=42
)

model = lgbm.LGBMRegressor(
    objective="regression",
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

valid_pred_log = model.predict(X_valid)
valid_pred = np.expm1(valid_pred_log)
y_valid_orig = np.expm1(y_valid)
rmse = mean_squared_error(y_valid_orig, valid_pred, squared=False)
rmse



## === cell 6
model.fit(train_feat, target)
test_pred_log = model.predict(test_feat)

test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, 0, None)

test_pred[:5], test_pred.shape



## === cell 7
sub = pd.read_csv(sample_path)

if (
    ("key" in sub.columns)
    and (len(sub) == len(test_df))
    and sub["key"].astype(str).equals(test_df["key"].astype(str))
):
    sub["fare_amount"] = test_pred.astype(np.float32)
else:
    sub = pd.DataFrame(
        {"key": test_df["key"].astype(str), "fare_amount": test_pred.astype(np.float32)}
    )

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

sub.head(), out_path, sub.shape
