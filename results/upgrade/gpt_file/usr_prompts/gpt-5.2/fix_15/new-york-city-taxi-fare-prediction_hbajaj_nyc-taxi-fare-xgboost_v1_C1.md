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

3.63072

# 6. Current score

5.66463

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.04125) has done: 'Your code likely isn’t producing a valid Kaggle score because it either (a) fail when trying to write `.feather` (missing optional dependency) or (b) produce inconsistent one-hot columns between train/test, silently hurting predictions. I make two minimal, score-relevant fixes: ensure the feather caching never breaks execution (fallback to CSV-only if feather isn’t supported), and force one-hot encoding to use a fixed set of categories (hours 0–23, weekdays 0–6, months 1–12) so train/test feature columns are consistent without relying on reindex to guess missing dummies. These changes preserve your core model/training logic and should reliably generate a valid `submission.csv` while improving RMSE toward your target by reducing train/test feature mismatch. I also ensure predictions are non-negative (fares can’t be negative), which typically improves RMSE slightly without changing the model.'
- What this solution (achieved 7.73351) has done: 'Your current RMSE (7.04) is far from the target (3.63), so we need a small but meaningful correctness/performance fix without changing the model/training approach. The biggest score drag here is that the haversine `distance` feature is being computed as a *string rounded to 0.001 km* and then cast back to numeric, which destroys important signal; changing it to return a float preserves the same feature but with proper precision. I’m also making the train/test datetime parsing consistent by explicitly using the same truncation+UTC parsing for test (already done) and keeping everything else unchanged. These minimal changes should improve generalization and move RMSE substantially toward your target while preserving the same core logic, model, and training loop.'
- What this solution (achieved 6.64727) has done: 'We need to move your RMSE down from 7.73351 toward 3.63072 (lower is better), so we should make a small but score-impactful correction without changing the core modeling approach (XGBoost regressor + same features). The biggest remaining score drag is that your haversine distance is computed row-by-row with `apply`, which is both slow and can introduce subtle dtype/object issues; we compute the exact same haversine feature vectorized in float32, preserving semantics but improving numerical consistency and letting you safely increase the training batch size within the same single-batch training loop. Increasing the batch from 100k to 500k keeps the same training approach/params but reduces variance and usually improves RMSE materially on this competition. Finally, we stop rounding individual model predictions and the geometric mean to 2 decimals (Kaggle scores on raw floats; rounding adds avoidable error), while still clipping to non-negative fares.'
- What this solution (achieved 7.48783) has done: 'Your current RMSE (6.64727) is still far above the target (3.63072), so we should make a small, correctness-focused change that typically yields a large gain without changing your model/training approach. The biggest remaining issue is a train/test distribution mismatch from using random row splits: taxi fares drift over time, so we split chronologically (by `pickup_datetime`) to better match the test distribution and reduce overfitting, while keeping the same XGBoost setup and features. To keep semantics identical, we only change how the holdout split is formed (still 5% holdout, same feature set, same training loop), and we keep submission generation the same. This should move RMSE materially downward toward your target without any architecture or loss changes.'
- What this solution (achieved 6.41861) has done: 'Your RMSE (7.48783) is far worse than the target (3.63072), so we should make a small but high-impact correctness fix rather than tune the model. The biggest remaining score drag is that the model is trained on only a single 500k-row chunk from a 55M-row dataset; keeping the same one-chunk training approach but increasing the chunk size (within memory) usually yields a large RMSE drop on this competition. To keep runtime reasonable and semantics consistent, I also ensure the DMatrix uses `nthread` and keep everything else (features, params, num_boost_round, preprocessing, submission format) unchanged. This should move RMSE materially downward toward the target band without changing your core logic.'
- What this solution (achieved 5.59803) has done: 'Your current RMSE (6.41861, lower is better) is still far from the target (3.63072), so the smallest meaningful improvement is to train on more data without changing the model, features, or training loop. I increase the single training chunk size (still one batch, same preprocessing and XGBoost params) to reduce variance and better capture the full fare distribution, which typically yields a large RMSE drop on this competition. I also cap extreme `fare_amount` outliers during cleanup (keeping the same logic, just adding an upper bound) because a tiny fraction of huge fares can disproportionately hurt RMSE and model fit. Everything else (feature engineering, OHE scheme, distance, time split, submission writing) stays the same.'
- What this solution (achieved 5.58222) has done: 'We need to move RMSE down from 5.59803 toward 3.63072 (lower is better), so we make small, score-relevant corrections without changing the core XGBoost approach, features, or training loop. The biggest likely remaining drag is a coordinate consistency/validity issue: the cleanup filters only one-sided longitude bounds and doesn’t enforce realistic NYC bounding boxes for both pickup and dropoff, leaving bad rows that harm fit; we add symmetric lon/lat bounds and ensure dropoff bounds are applied consistently. We also remove rows with zero distance (often bad GPS/duplicates) after computing distance, which typically improves RMSE while preserving the same feature set. Finally, we keep everything else unchanged, including single-batch training, parameters, OHE categories, and submission formatting.'
- What this solution (achieved 5.58222) has done: 'Your current pipeline is likely failing to yield a Kaggle score because it filters rows out of the test set (by lat/lon/passengers and by distance), which makes the submission row count not match `sample_submission` and gets rejected. The smallest score-relevant fix is to keep all test rows, only doing safe imputation (dropna->fill) and feature generation, and then always output predictions for every `key` in the original test file. To keep the model semantics unchanged, I leave the training cleanup as-is and only adjust test preprocessing and submission creation to preserve all rows, filling missing/invalid feature rows with neutral values. This should produce a valid `submission.csv` and improve RMSE versus a broken/invalid submission without changing your model or feature definitions.'
- What this solution (achieved 5.58222) has done: 'We need to reduce RMSE from 5.58222 toward 3.63072 (lower is better), so the smallest high-impact change is to remove a known train/test mismatch: the model currently never sees the `passenger_count` feature because it’s accidentally excluded by `usecols` in `read_file_to_df`, while it exists in test, harming generalization. I fix `read_file_to_df` to include `passenger_count` and (to preserve core logic) keep everything else the same, including the same XGBoost params, features, one-batch training, and preprocessing. I also make the holdout RMSE calculation consistent (swap arguments in `mean_squared_error`) without affecting training. These changes are directly score-relevant and should move RMSE materially down toward your target without changing the modeling approach.'
- What this solution (achieved 5.71475) has done: 'We need to move RMSE down from 5.58222 toward 3.63072 (lower is better), so the smallest meaningful change is to improve feature signal without changing the overall XGBoost approach or training loop. Your current feature set is missing the classic NYC taxi baseline feature: straight-line (Euclidean) distance in lat/lon degrees; adding it is a minimal, high-impact, metric-aligned improvement that preserves the same modeling pipeline (still one-batch XGBoost on engineered numeric features + OHE time features). I also make the test preprocessing more robust by filling invalid passenger counts (0/NaN) to 1 (common default) so test feature distributions match training cleanup constraints without dropping any test rows. Everything else—data reading, cleanup, time-based split, XGBoost params/rounds, stacking logic, and submission format—stays the same.'
- What this solution (achieved 5.70032) has done: 'You’re still far from the target RMSE (5.71 vs 3.63, lower is better), so we should make one small, high-impact *correctness/feature-signal* change without altering the XGBoost training approach or feature set style. The classic missing baseline here is the NYC “center” effect: fares correlate strongly with how close pickup/dropoff are to Manhattan; adding distances-to-center uses the same raw coordinates you already use and typically yields a large RMSE drop. I add two numeric features (`pickup_manhattan_dist`, `dropoff_manhattan_dist`) computed with the same vectorized haversine function, keep all existing preprocessing/training/stacking intact, and ensure the train/test feature columns remain aligned through your existing `reindex`. This is minimal code change, preserves core logic, and is directly score-relevant for RMSE.'
- What this solution (achieved 5.66463) has done: 'Your RMSE (5.70) is still far above the target (3.63), so we need a small but high-impact feature signal fix without changing the XGBoost training approach or overall pipeline. The biggest remaining gap in this classic competition is that fares correlate strongly with *directional* movement and approximate *city-block (“Manhattan”) distance*, not just straight-line distance; adding these is minimal, uses the same coordinates you already process, and typically yields a sizable RMSE reduction. I add two numeric features: `manhattan` (|Δlat|+|Δlon|) and `bearing` (trip direction), then keep the same preprocessing flow, one-hot scheme, training loop, stacking, and submission format. This should move the score materially down toward your target while preserving core logic and keeping runtime within limits.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O

import os
import time
from math import sin, cos, sqrt, atan2, radians
from functools import wraps

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import scipy

try:
    import joblib
except Exception:
    from sklearn.externals import joblib  # pragma: no cover

INPUT_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
print("Input dir exists:", os.path.exists(INPUT_DIR))
print("Files:", sorted(os.listdir(INPUT_DIR))[:20])


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print(
            "***************************Total time running << %s >>: %s seconds"
            % (function.__name__, str(np.round((t1 - t0), 2)))
        )
        return result

    return function_timer


@fn_timer
def read_file_to_df(fileName, rows):
    """
    Change (score-relevant, minimal): ensure passenger_count is included in train reads.
    Previously it was omitted from usecols due to cols = list(traintypes.keys()).
    This created a train/test feature mismatch (model never saw passenger_count),
    which typically hurts RMSE on this competition.
    """
    start_time = time.time()
    traintypes = {
        "fare_amount": "float32",
        "pickup_datetime": "str",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
        "key": "str",
    }

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

    from pathlib import Path

    feather_path = Path(fileName + ".feather")
    if feather_path.is_file():
        try:
            df = pd.read_feather(str(feather_path))
            print("Loaded cached feather:", str(feather_path))
        except Exception as e:
            print("Feather read failed, falling back to CSV. Error:", repr(e))
            if rows and rows > 0:
                df = pd.read_csv(fileName, usecols=cols, dtype=traintypes, nrows=rows)
            else:
                df = pd.read_csv(fileName, usecols=cols, dtype=traintypes)
    else:
        if rows and rows > 0:
            df = pd.read_csv(fileName, usecols=cols, dtype=traintypes, nrows=rows)
        else:
            df = pd.read_csv(fileName, usecols=cols, dtype=traintypes)
        try:
            df.to_feather(str(feather_path))
            print("Wrote cached feather:", str(feather_path))
        except Exception as e:
            print("Feather write skipped (not supported). Error:", repr(e))

    print("--- Read Files: %s secs ---" % np.round((time.time() - start_time), 2))
    return df


def read_test_file_to_df(fileName):
    df = pd.read_csv(fileName)
    return df


def drop_rows_with_nan(df):
    print("Before dropna")
    df.dropna(how="any", axis="rows", inplace=True)
    print(df.shape)


@fn_timer
def data_cleanup(tmp_df):
    """
    Tight coordinate sanity checks to reduce bad GPS rows that harm RMSE.
    """
    print("Before clearing outliers")
    tmp_df = tmp_df[(tmp_df["fare_amount"] > 0) & (tmp_df["fare_amount"] <= 250)]

    tmp_df = tmp_df[
        (tmp_df["pickup_longitude"] > -75) & (tmp_df["pickup_longitude"] < -72)
    ]
    tmp_df = tmp_df[
        (tmp_df["dropoff_longitude"] > -75) & (tmp_df["dropoff_longitude"] < -72)
    ]

    tmp_df = tmp_df[(tmp_df["pickup_latitude"] > 40) & (tmp_df["pickup_latitude"] < 42)]
    tmp_df = tmp_df[
        (tmp_df["dropoff_latitude"] > 40) & (tmp_df["dropoff_latitude"] < 42)
    ]

    tmp_df = tmp_df[(tmp_df["passenger_count"] > 0) & (tmp_df["passenger_count"] < 10)]
    print(tmp_df.shape)
    return tmp_df


def reformat_pickup_datetime(df):
    df["pickup_datetime"] = df["pickup_datetime"].astype(str).str.slice(0, 13)
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], utc=True, format="%Y-%m-%d %H", errors="coerce"
    )
    return df


def add_day_of_week_feature(tmp_df):
    tmp_df["dayOfWeek"] = tmp_df["pickup_datetime"].dt.dayofweek.astype("uint8")


def add_time_of_day_feature(tmp_df):
    tmp_df["timeOfDay"] = tmp_df["pickup_datetime"].dt.hour.astype("uint8")


def add_month_feature(tmp_df):
    tmp_df["month"] = tmp_df["pickup_datetime"].dt.month.astype("uint8")


def add_week_of_year_feature(tmp_df):
    tmp_df["weekOfYear"] = (
        tmp_df["pickup_datetime"].dt.isocalendar().week.astype("uint8")
    )


def distance_between_two_points(row):
    R = 6373.0

    lat1 = radians(row["pickup_latitude"])
    lon1 = radians(row["pickup_longitude"])
    lat2 = radians(row["dropoff_latitude"])
    lon2 = radians(row["dropoff_longitude"])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return np.float32(distance)


@fn_timer
def add_distance_feature(tmp_df):
    """
    Vectorized haversine distance feature (same definition).
    """
    start_time = time.time()
    R = np.float32(6373.0)

    lat1 = np.radians(tmp_df["pickup_latitude"].astype("float32").to_numpy())
    lon1 = np.radians(tmp_df["pickup_longitude"].astype("float32").to_numpy())
    lat2 = np.radians(tmp_df["dropoff_latitude"].astype("float32").to_numpy())
    lon2 = np.radians(tmp_df["dropoff_longitude"].astype("float32").to_numpy())

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = (R * c).astype("float32")

    tmp_df["distance"] = dist
    print("--- Add distance: %s secs ---" % np.round((time.time() - start_time), 2))


@fn_timer
def add_euclidean_feature(tmp_df):
    """
    Add straight-line (euclidean) distance in degree-space.
    """
    start_time = time.time()
    dlat = (
        tmp_df["dropoff_latitude"].astype("float32")
        - tmp_df["pickup_latitude"].astype("float32")
    ).to_numpy()
    dlon = (
        tmp_df["dropoff_longitude"].astype("float32")
        - tmp_df["pickup_longitude"].astype("float32")
    ).to_numpy()
    tmp_df["euclidean"] = np.sqrt(dlat * dlat + dlon * dlon).astype("float32")
    print("--- Add euclidean: %s secs ---" % np.round((time.time() - start_time), 2))


def _haversine_km_vec(lat1, lon1, lat2, lon2):
    """
    Reusable vectorized haversine used to compute distances to a fixed NYC center point.
    """
    R = np.float32(6373.0)
    lat1 = np.radians(lat1.astype("float32").to_numpy())
    lon1 = np.radians(lon1.astype("float32").to_numpy())
    lat2 = np.radians(lat2.astype("float32"))
    lon2 = np.radians(lon2.astype("float32"))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return (R * c).astype("float32")


@fn_timer
def add_manhattan_center_features(tmp_df):
    """
    Add distances from pickup/dropoff to a fixed Manhattan/NYC center point.
    """
    start_time = time.time()
    center_lat = np.float32(40.7580)
    center_lon = np.float32(-73.9855)

    tmp_df["pickup_manhattan_dist"] = _haversine_km_vec(
        tmp_df["pickup_latitude"],
        tmp_df["pickup_longitude"],
        center_lat,
        center_lon,
    )
    tmp_df["dropoff_manhattan_dist"] = _haversine_km_vec(
        tmp_df["dropoff_latitude"],
        tmp_df["dropoff_longitude"],
        center_lat,
        center_lon,
    )
    print(
        "--- Add manhattan center dists: %s secs ---"
        % np.round((time.time() - start_time), 2)
    )


@fn_timer
def add_manhattan_and_bearing_features(tmp_df):
    """
    Change (score-relevant, minimal): add two classic baseline signals without changing
    the model/training approach:
      - manhattan: |Δlat| + |Δlon| (proxy for city-block travel)
      - bearing: trip direction (atan2 of lon/lat deltas, in radians)
    These are cheap to compute and typically reduce RMSE materially on this competition.
    """
    start_time = time.time()
    dlat = (
        tmp_df["dropoff_latitude"].astype("float32")
        - tmp_df["pickup_latitude"].astype("float32")
    ).to_numpy()
    dlon = (
        tmp_df["dropoff_longitude"].astype("float32")
        - tmp_df["pickup_longitude"].astype("float32")
    ).to_numpy()

    tmp_df["manhattan"] = (np.abs(dlat) + np.abs(dlon)).astype("float32")
    tmp_df["bearing"] = np.arctan2(dlon, dlat).astype("float32")

    print(
        "--- Add manhattan+bearing: %s secs ---"
        % np.round((time.time() - start_time), 2)
    )


@fn_timer
def filter_based_on_distance(df, distance):
    df = df[(df["distance"] < distance)]
    return df


@fn_timer
def data_one_hot_encoding(tmp_df):
    """
    Enforce fixed categories for OHE so train/test columns are consistent.
    """
    print("Before one hot encoding")
    start_time = time.time()

    tmp_df["timeOfDay"] = pd.Categorical(
        tmp_df["timeOfDay"], categories=list(range(24))
    )
    tmp_df["dayOfWeek"] = pd.Categorical(tmp_df["dayOfWeek"], categories=list(range(7)))
    tmp_df["month"] = pd.Categorical(tmp_df["month"], categories=list(range(1, 13)))

    tmp_df = pd.concat(
        [tmp_df, pd.get_dummies(tmp_df["timeOfDay"], prefix="timeOfDay")], axis=1
    )
    tmp_df.drop(["timeOfDay"], axis=1, inplace=True)

    tmp_df = pd.concat(
        [tmp_df, pd.get_dummies(tmp_df["dayOfWeek"], prefix="dayOfWeek")], axis=1
    )
    tmp_df.drop(["dayOfWeek"], axis=1, inplace=True)

    tmp_df = pd.concat(
        [tmp_df, pd.get_dummies(tmp_df["month"], prefix="month")], axis=1
    )
    tmp_df.drop(["month"], axis=1, inplace=True)

    print("--- One hot encoding: %s secs ---" % np.round((time.time() - start_time), 2))
    return tmp_df


@fn_timer
def prepare_data_split(tmp_df):
    tmp_df = tmp_df.sort_values("pickup_datetime").reset_index(drop=True)

    tmp_X = tmp_df.drop(["pickup_datetime", "fare_amount", "key"], axis=1)
    tmp_y = tmp_df["fare_amount"]

    print("Before train test split (time-based)")
    n = len(tmp_df)
    split_idx = int(n * (1.0 - 0.05))
    tmp_X_train = tmp_X.iloc[:split_idx]
    tmp_y_train = tmp_y.iloc[:split_idx]
    tmp_X_test = tmp_X.iloc[split_idx:]
    tmp_y_test = tmp_y.iloc[split_idx:]

    print("df.shape:" + str(tmp_df.shape))
    print("tmp_X.shape:" + str(tmp_X.shape))
    print("tmp_y.shape:" + str(tmp_y.shape))
    return tmp_X, tmp_y, tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test


@fn_timer
def get_rmse(model, data, output):
    dtrain = xgb.DMatrix(data, label=output, nthread=4)
    y_pred = model.predict(dtrain)
    rmse = np.sqrt(metrics.mean_squared_error(output, y_pred))
    return rmse


@fn_timer
def fit_xgboost_model(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train, nthread=4)
    dtest = xgb.DMatrix(X_test, nthread=4)

    params = {
        "eval_metric": "rmse",
        "max_depth": 7,
        "subsample": 0.8,
        "eta": 0.1,
        "gamma": 1,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "seed": 42,
        "nthread": 4,
    }
    print("fit_xgboost_model")
    model = xgb.train(params, dtrain, num_boost_round=250)

    y_pred = model.predict(dtest)
    y_train_pred = model.predict(dtrain)

    print("Holdout RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
    return model


@fn_timer
def output_submission_stacking(df_test_keys, test_d, models):
    test_X = xgb.DMatrix(test_d, nthread=4)

    rmse_df = pd.DataFrame(index=df_test_keys.index)
    for m in models:
        test_pred = m.predict(test_X)
        test_pred = np.clip(test_pred, 0, None)
        rmse_df = pd.concat(
            [rmse_df, pd.DataFrame(test_pred, index=df_test_keys.index)], axis=1
        )

    submission = pd.DataFrame(
        {
            "key": df_test_keys.values,
            "fare_amount": np.asarray(scipy.stats.mstats.gmean(rmse_df, axis=1)).astype(
                "float32"
            ),
        },
        columns=["key", "fare_amount"],
    )
    submission.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission.shape)


@fn_timer
def preprocess_df(df, ifTest):
    start_time = time.time()

    print("Preprocessing data start")
    if ifTest is False:
        drop_rows_with_nan(df)
        df = data_cleanup(df)
    else:
        df = df.dropna(how="all", axis="rows")
        if "passenger_count" in df.columns:
            df["passenger_count"] = pd.to_numeric(
                df["passenger_count"], errors="coerce"
            )
            df["passenger_count"] = (
                df["passenger_count"].fillna(1).clip(lower=1, upper=9).astype("uint8")
            )

    df = reformat_pickup_datetime(df)

    if ifTest is True:
        dt_fill = pd.Timestamp("2010-01-01 00:00:00", tz="UTC")
        df["pickup_datetime"] = df["pickup_datetime"].fillna(dt_fill)

    print("Adding day,time and month feature")
    add_day_of_week_feature(df)
    add_time_of_day_feature(df)
    add_month_feature(df)

    print("Adding distance feature")
    add_distance_feature(df)

    add_euclidean_feature(df)

    add_manhattan_center_features(df)

    add_manhattan_and_bearing_features(df)

    if ifTest is False:
        df = df[df["distance"] > 0]
        df = filter_based_on_distance(df, 100)
    else:
        df["distance"] = df["distance"].replace([np.inf, -np.inf], np.nan)
        df["distance"] = df["distance"].fillna(np.float32(0.0))
        df["distance"] = df["distance"].clip(
            lower=np.float32(0.0), upper=np.float32(100.0)
        )
        for c in ["euclidean", "manhattan"]:
            df[c] = (
                df[c]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(np.float32(0.0))
                .clip(lower=np.float32(0.0))
            )
        for c in ["pickup_manhattan_dist", "dropoff_manhattan_dist"]:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(np.float32(0.0))
            df[c] = df[c].clip(lower=np.float32(0.0))
        df["bearing"] = (
            df["bearing"].replace([np.inf, -np.inf], np.nan).fillna(np.float32(0.0))
        )

    print("One hot encoding features")
    df = data_one_hot_encoding(df)

    if ifTest is False:
        drop_rows_with_nan(df)
    else:
        df = df.fillna(0)

    print("Preprocessing data end")
    print("--- preprocess_df: %s secs ---" % np.round((time.time() - start_time), 2))
    return df


@fn_timer
def split_and_fit_model(df):
    tmp_df = preprocess_df(df, False)
    print("Shape of tmp_df: {}".format(tmp_df.shape))

    X, y, X_train, X_test, y_train, y_test = prepare_data_split(tmp_df)

    model = fit_xgboost_model(X_train, X_test, y_train, y_test)
    rmse = get_rmse(model, X, y)
    print("rmse:" + str(rmse))
    return model, rmse, list(X.columns)


def dump_model(model, fileName):
    os.makedirs(os.path.dirname(fileName), exist_ok=True)
    joblib.dump(model, fileName + ".compressed", compress=True)




## === cell 1
train_path = os.path.join(INPUT_DIR, "train.csv")
reader = pd.read_csv(train_path, header=0, iterator=True)

models = []
rmses = []
feature_columns = None

for i in range(0, 1):
    print("##################### Batch start num {} #####################".format(i))
    batchSize = 5_000_000
    df = reader.get_chunk(batchSize)

    model, rmse, feat_cols = split_and_fit_model(df)
    models.append(model)
    rmses.append(rmse)
    feature_columns = feat_cols  # columns used by training (after OHE)

    del df
    print("##################### Batch end #####################")

reader.close()
del reader

print("RMSEs:", rmses)



## === cell 2
test_path = os.path.join(INPUT_DIR, "test.csv")
df_test_raw = read_test_file_to_df(test_path)

test_keys = df_test_raw["key"].copy()

df_test = preprocess_df(df_test_raw, True)
test_X = df_test.drop(["pickup_datetime", "key"], axis=1)

if feature_columns is None:
    raise RuntimeError("feature_columns is None; training did not run correctly.")
test_X = test_X.reindex(columns=feature_columns, fill_value=0)

output_submission_stacking(test_keys, test_X, models)

sub = pd.read_csv("submission.csv")
print(sub.head())
print(sub.shape)
print(sub.columns.tolist())
assert sub.columns.tolist() == ["key", "fare_amount"]
assert len(sub) == len(df_test_raw)

del df_test_raw, df_test, test_X, sub, test_keys
