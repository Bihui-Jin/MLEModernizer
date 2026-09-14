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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

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

4.41621

# 6. Current score

12.02557

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 12.88621) has done: 'I fix the runtime blockers while preserving your model and feature logic: (1) remove Keras/TF import issues that trigger the `MessageFactory` protobuf error by switching to `tf_keras` (already installed) with the same Sequential/Dense architecture, and update the optimizer API to the modern `Adam(learning_rate=...)`. (2) Fix the NYC water-mask cleaning step so it works offline in Kaggle by downloading the PNG via `urllib` + `PIL` (instead of `plt.imread(url)`), keeping the same mask logic and filtering behavior. (3) Correct a few logic/runtime issues that were causing very poor score or crashes (time feature parsing, late_night/night conditions, and evaluating/predicting with scaled arrays), then ensure a valid `submissiontry_water.csv` with columns `key,fare_amount` is written.'
- What this solution (achieved 6.03614) has done: 'I fix the two runtime blockers: (1) avoid importing TensorFlow (which triggers the protobuf `MessageFactory.GetPrototype` crash in this environment) while keeping the same `tf_keras` Sequential/Dense model, and (2) make the NYC water-mask step work offline by loading the mask from a local file if present and otherwise skipping the water-filter cleanly (score-neutral vs crashing). I also correct the dataset paths to the actual Kaggle-mounted files you listed (`/kaggle/input/...`) so the notebook runs end-to-end. These changes preserve your model architecture, features, training loop, and evaluation semantics, but ensure a valid `submissiontry_water.csv` is always produced.'
- What this solution (achieved 15.14185) has done: 'We need to fix the immediate runtime crash caused by importing `tf_keras`/protobuf (`MessageFactory.GetPrototype`) while keeping the same Keras Sequential/Dense architecture and training loop. The minimal robust fix in this Kaggle environment is to switch to the standalone `keras` (Keras 3) backend configured to use NumPy (no TensorFlow import), which avoids the protobuf crash and preserves model semantics closely enough for this baseline. I also correct the train/test file paths to the actual listed dataset location (`/kaggle/input/new-york-city-taxi-fare-prediction/...`) so the code can read data reliably. Finally, to nudge RMSE toward the target without changing the core model, I ensure the NYC mask step remains non-blocking and add a small, score-safe prediction calibration: clip to non-negative (already present) and cap extreme fares to the training-cleaning max (50) to reduce occasional large errors.'
- What this solution (achieved 6.31405) has done: 'I fix the runtime blockers so the notebook trains and predicts end-to-end by avoiding the Keras NumPy backend (which doesn’t implement `fit`) and avoiding the protobuf `MessageFactory.GetPrototype` crash by using the installed `tf_keras` package instead of `keras`/TensorFlow imports. I keep your exact model architecture, features, cleaning logic, scaling, and training loop semantics intact; changes are limited to import/backend configuration and a small safety cast to ensure arrays are proper NumPy dtypes for `evaluate/predict`. This should both make the pipeline run and improve RMSE versus the current broken/degenerate run (15.14) by actually training the neural net. The submission writing stays the same and always produce `submissiontry_water.csv` with `key,fare_amount`.'
- What this solution (achieved 7.69812) has done: 'I fix the immediate runtime crash (`MessageFactory` protobuf issue) by avoiding any TensorFlow/tf_keras import at module import time and instead using scikit-learn’s `MLPRegressor` with the same “dense ReLU stack” training semantics (feed-forward MLP trained with Adam), which runs reliably in this Kaggle environment. I keep your existing data cleaning and feature engineering intact, but optimize the time-feature creation by replacing slow row-wise `apply` with vectorized boolean logic (same semantics, much faster and less error-prone). I also preserve your MinMax scaling and keep the same clipping of predictions to `[0, 50]`, ensuring the submission is written with exact required columns and a `.csv` suffix. These changes should both unblock execution and improve RMSE toward the target by actually training a stable neural net on your engineered features.'
- What this solution (achieved 5.25487) has done: 'You’re currently worse than the target (RMSE 7.698 → 4.416), so we should make small, low-risk improvements that reduce large errors without changing the overall approach. I keep your exact feature engineering and the same sklearn MLP “dense ReLU stack”, but (1) train on the full cleaned training split instead of only half the sampled data (your current `train_test_split(..., test_size=0.50)` discards half the data), and (2) align train/test cleaning by applying the same `clean()` to the Kaggle test set before feature generation so train/test distributions match better. Finally, I make the submission key alignment robust by predicting on the cleaned test rows and then merging back onto the original `sample_submission` keys (any dropped rows get a safe fill value), preventing accidental row-mismatch issues that can silently hurt RMSE.'
- What this solution (achieved 5.38067) has done: 'Your current RMSE (5.25487) is worse than the target (4.41621), so we should make a small, low-risk improvement that reduces large errors while keeping your exact MLP model and engineered features. The biggest issue is that you drop `passenger_count` even though it is predictive and already cleaned; adding it back keeps the same core approach but typically improves RMSE for this competition. I also ensure the feature columns are aligned between train/validation/test/Kaggle-test after cleaning (same columns, same order) to avoid subtle train/test mismatch that can hurt score. Everything else (cleaning logic, time/distance features, MinMax scaling, MLPRegressor setup, clipping, and submission merge-by-key) stays the same.'
- What this solution (achieved 12.02518) has done: 'Your current RMSE (5.38067) is worse than the target (4.41621), so we should make the smallest “distribution/feature” improvements without changing the model class, layer sizes, or training loop. The biggest low-risk gain here is to replace the rough Euclidean-on-degrees distance with a proper haversine distance in km (keeping your existing distance feature, just making it physically meaningful), and to add the standard NYC-taxi baseline features “bearing” and “abs lat/lon deltas” that are derived only from the same coordinates. These are minimal feature-engineering changes that typically reduce large errors for this competition while keeping everything else (cleaning, scaling, MLPRegressor params, clipping, submission merge-by-key) identical. I also keep the submission alignment logic unchanged to ensure a valid CSV is always produced.'
- What this solution (achieved 12.02518) has done: 'Your current RMSE (12.02518) is far worse than the target (4.41621), so we need a small change that clearly fixes a likely “silent data/feature mismatch” rather than tuning the model. The main issue is that you read `train.csv` without the `key` column, but later `clean()` calls `remove_datapoints_from_water()`, which uses `df[idx]`; with a missing `key`, this boolean indexing can still work but may produce subtle misalignment and (more importantly) your train cleaning is not guaranteed to be identical to test cleaning. I make the minimal fix: read `key` for train as well and then explicitly drop it only right before scaling, ensuring water-filtering and downstream feature generation run on the exact same schema for train/val/test. This preserves your model, features, and training loop, but typically improves RMSE significantly by preventing accidental column/row misalignment during cleaning/feature steps. The submission writing and key alignment remain unchanged and still produce `submissiontry_water.csv`.'
- What this solution (achieved 12.02505) has done: 'We make the smallest changes that address the most likely cause of the sudden RMSE blow-up (12.0): train/validation/test splits are currently done *before* cleaning, so after row-filtering you end up training and evaluating on different distributions and (worse) a potentially “easier/harder” holdout that doesn’t reflect the real test set. We instead clean the sampled training data once, then split into train/holdout/validation on the already-cleaned data (same semantics, same features/model), which typically reduces error substantially without changing the model. We also ensure the same column dtypes are used for test reading (avoids subtle parsing differences for passenger_count and coordinates) and keep the submission key alignment logic unchanged.'
- What this solution (achieved 12.02557) has done: 'Your current RMSE (12.025) is far worse than the target (4.416), so the most likely issue is not “model capacity” but a silent bug/semantic mismatch. I make two minimal, directly score-relevant fixes: (1) correct the airport/landmark filtering logic that currently removes almost all normal NYC rides due to using `!=` with `&` instead of removing only the exact coordinate matches, and (2) ensure the cleaned test set is used consistently (including dropping invalid datetimes) so train/test feature distributions match. Everything else (same MLPRegressor architecture/params, same scaling, same feature set, same submission merge-by-key) stays unchanged.'

# 9. Code solution

## === cell 0
import os
import io
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(1)

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SAMPLE_SUB_PATH = (
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 1000
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

print("Train path exists:", os.path.exists(TRAIN_PATH), TRAIN_PATH)
print("Test path exists:", os.path.exists(TEST_PATH), TEST_PATH)
print("Sample sub path exists:", os.path.exists(SAMPLE_SUB_PATH), SAMPLE_SUB_PATH)



## === cell 1
from urllib.request import urlopen
from PIL import Image


def _load_nyc_mask_cached(cache_path="nyc_mask.png"):
    """
    Kaggle internet is often disabled and the previous URL may 404.
    Keep identical mask logic when a local cached mask exists; otherwise return None and skip filtering.
    """
    candidate_paths = [
        cache_path,
        "/kaggle/input/nyc_mask.png",
        "/kaggle/working/nyc_mask.png",
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                return plt.imread(p)[:, :, 0] > 0.9
            except Exception:
                pass

    url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
    try:
        with urlopen(url) as f:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
        img.save(cache_path)
        return plt.imread(cache_path)[:, :, 0] > 0.9
    except Exception as e:
        print(
            "WARNING: Could not load NYC mask (offline/404). Skipping water filtering. Error:",
            repr(e),
        )
        return None


def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)
    nyc_mask = _load_nyc_mask_cached()

    if nyc_mask is None:
        return df

    pickup_x, pickup_y = lonlat_to_xy(
        df.pickup_longitude,
        df.pickup_latitude,
        nyc_mask.shape[1],
        nyc_mask.shape[0],
        BB,
    )
    dropoff_x, dropoff_y = lonlat_to_xy(
        df.dropoff_longitude,
        df.dropoff_latitude,
        nyc_mask.shape[1],
        nyc_mask.shape[0],
        BB,
    )

    pickup_x = np.clip(pickup_x, 0, nyc_mask.shape[1] - 1)
    dropoff_x = np.clip(dropoff_x, 0, nyc_mask.shape[1] - 1)
    pickup_y = np.clip(pickup_y, 0, nyc_mask.shape[0] - 1)
    dropoff_y = np.clip(dropoff_y, 0, nyc_mask.shape[0] - 1)

    idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]
    return df[idx]


def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=False)
    before_dt = len(df)
    df = df.loc[dt.notna()].copy()
    print(
        " New size after dropping invalid pickup_datetime: %d (dropped %d)"
        % (len(df), before_dt - len(df))
    )

    if "fare_amount" in df.columns:
        df = df[
            (df["dropoff_longitude"] != df["pickup_longitude"])
            & (df["dropoff_latitude"] != df["pickup_latitude"])
        ]
        print(" New size after removing same long lat: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    ]
    print(" New size after removing 0 long lat: %d" % len(df))

    MinMax = (-74.5, -72.8, 40.5, 41.8)
    df = df[
        (MinMax[0] <= df["pickup_longitude"]) & (df["pickup_longitude"] <= MinMax[1])
    ]
    df = df[
        (MinMax[0] <= df["dropoff_longitude"]) & (df["dropoff_longitude"] <= MinMax[1])
    ]
    df = df[(MinMax[2] <= df["pickup_latitude"]) & (df["pickup_latitude"] <= MinMax[3])]
    df = df[
        (MinMax[2] <= df["dropoff_latitude"]) & (df["dropoff_latitude"] <= MinMax[3])
    ]
    print(" New size after only NYC: %d" % len(df))

    if "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)

    def _remove_exact_coord(df_, lat, lon, prefix):
        before = len(df_)
        pickup_match = (df_["pickup_latitude"] == lat) & (
            df_["pickup_longitude"] == lon
        )
        dropoff_match = (df_["dropoff_latitude"] == lat) & (
            df_["dropoff_longitude"] == lon
        )
        df_ = df_.loc[~(pickup_match | dropoff_match)].copy()
        print(
            " New size after removing %s exact coord: %d (dropped %d)"
            % (prefix, len(df_), before - len(df_))
        )
        return df_

    df = _remove_exact_coord(df, nyc_coord[0], nyc_coord[1], "nyc_center")
    df = _remove_exact_coord(df, fk_coord[0], fk_coord[1], "jfk")
    df = _remove_exact_coord(df, ewr_coord[0], ewr_coord[1], "ewr")
    df = _remove_exact_coord(df, lga_coord[0], lga_coord[1], "lga")
    df = _remove_exact_coord(df, sol_coord[0], sol_coord[1], "sol")

    print("Old size: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=False)
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday
    df["pickup_datetime"] = df["pickup_datetime"].astype(str)

    hour = df["hour"]
    weekday = df["weekday"]

    df["night"] = (((hour > 20) | (hour < 6)) & (weekday < 5)).astype(np.int8)
    df["late_night"] = (((hour <= 3) | (hour >= 23))).astype(np.int8)
    df["rush_hour"] = (((hour >= 16) & (hour <= 20)) & (weekday < 5)).astype(np.int8)
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["latdiff"] = lat1 - lat2
    df["londiff"] = lon1 - lon2
    df["abs_latdiff"] = np.abs(df["latdiff"])
    df["abs_londiff"] = np.abs(df["londiff"])
    return df


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1 = np.radians(lat1.astype(np.float64))
    lon1 = np.radians(lon1.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return (R * c).astype(np.float32)


def _bearing_rad(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(np.float64))
    lon1 = np.radians(lon1.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x).astype(np.float32)


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)

    df["distance"] = _haversine_km(lat1, lon1, lat2, lon2)

    df["bearing"] = _bearing_rad(lat1, lon1, lat2, lon2)
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows=", len(df))


def plot_loss_accuracy_rmse(history):
    print("plot_loss_accuracy_rmse skipped (no Keras History object with sklearn).")




## === cell 2
datatypes = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

test_dtypes = {
    "key": "str",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

trainKaggle = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    dtype=datatypes,
    usecols=[0, 1, 2, 3, 4, 5, 6, 7],  # key + fare + 6 features
)
testKaggle_raw = pd.read_csv(
    TEST_PATH,
    dtype=test_dtypes,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

trainKaggle.head(), testKaggle_raw.head(), sample_sub.head()



## === cell 3
testKaggle = testKaggle_raw.copy()
train_all = trainKaggle.copy()

print("train_all clean (before split)")
train_all = clean(train_all)

train_all, test_df = train_test_split(train_all, test_size=0.20, random_state=1)
train_df, validation_df = train_test_split(train_all, test_size=0.10, random_state=1)

print("testKaggle clean")
testKaggle_cleaned = clean(testKaggle.copy())



## === cell 4
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("validation_df Size %d" % len(validation_df))
print("test_df Size %d" % len(test_df))



## === cell 5
train_df.describe()



## === cell 6
test_df.describe()



## === cell 7
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)

print("testKaggle_cleaned add_time_features")
testKaggle_cleaned = add_time_features(testKaggle_cleaned)



## === cell 8
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)

print("testKaggle_cleaned add_coordinate_features")
testKaggle_cleaned = add_coordinate_features(testKaggle_cleaned)



## === cell 9
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)

print("testKaggle_cleaned add_distances_features")
testKaggle_cleaned = add_distances_features(testKaggle_cleaned)

print("Done with Adding features")



## === cell 10
dropped_columns = ["pickup_datetime", "key"]

train_df = train_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

testKaggle_clean_features = testKaggle_cleaned.drop(["pickup_datetime", "key"], axis=1)

print("Done with dropped_columns")
train_df.head()



## === cell 11
train_labels = train_df["fare_amount"].values.astype(np.float32)
validation_labels = validation_df["fare_amount"].values.astype(np.float32)
test_labels = test_df["fare_amount"].values.astype(np.float32)

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")
print(train_df.shape, validation_df.shape, test_df.shape)



## === cell 12
feature_cols = list(train_df.columns)
validation_df = validation_df[feature_cols]
test_df = test_df[feature_cols]
testKaggle_clean_features = testKaggle_clean_features[feature_cols]

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df).astype(np.float32)
validation_df_scaled = scaler.transform(validation_df).astype(np.float32)
test_scaled = scaler.transform(test_df).astype(np.float32)
testKaggle_scaled = scaler.transform(testKaggle_clean_features).astype(np.float32)

train_labels = np.asarray(train_labels, dtype=np.float32)
validation_labels = np.asarray(validation_labels, dtype=np.float32)
test_labels = np.asarray(test_labels, dtype=np.float32)




## === cell 13
def rmse_np(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))




## === cell 14
model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 8),
    activation="relu",
    solver="adam",
    alpha=0.0,
    batch_size=BATCH_SIZE,
    learning_rate_init=LEARNING_RATE,
    max_iter=EPOCHS,
    shuffle=True,
    random_state=1,
    early_stopping=False,
    validation_fraction=0.0,
    n_iter_no_change=EPOCHS + 1,
    verbose=True,
)

print("Dataset size (nrows read): %s" % DATASET_SIZE)
print("Epochs (max_iter): %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % feature_cols)

model.fit(train_df_scaled, train_labels)

history = {"loss_curve_": getattr(model, "loss_curve_", None)}



## === cell 15
plot_loss_accuracy_rmse(history)



## === cell 16
pred_test = model.predict(test_scaled).astype(np.float32)
rmse_holdout = rmse_np(test_labels, pred_test)
print("Holdout RMSE:", rmse_holdout)



## === cell 17
predictionKaggle_clean = (
    model.predict(testKaggle_scaled).astype(np.float32).reshape(-1, 1)
)
predictionKaggle_clean = np.clip(predictionKaggle_clean, 0.0, 50.0)

pred_df = pd.DataFrame(
    {
        "key": testKaggle_cleaned["key"].values,
        "fare_amount": predictionKaggle_clean.reshape(-1),
    }
)

submission = sample_sub.merge(pred_df, on="key", how="left", suffixes=("", "_pred"))
if "fare_amount_pred" in submission.columns:
    submission["fare_amount"] = submission["fare_amount_pred"].fillna(
        submission["fare_amount"]
    )
    submission = submission.drop(columns=["fare_amount_pred"])
else:
    submission["fare_amount"] = submission["fare_amount"].fillna(11.35)

submission["fare_amount"] = submission["fare_amount"].astype(np.float32).clip(0.0, 50.0)

submission.to_csv(SUBMISSION_NAME, index=False)
print("Output complete:", SUBMISSION_NAME, "rows=", len(submission))



## === cell 18
print("Example preds (holdout):", pred_test[:5])
print("Example test labels:", test_labels[:5])
print(
    "Submission file exists:",
    os.path.exists(SUBMISSION_NAME),
    "size bytes:",
    os.path.getsize(SUBMISSION_NAME) if os.path.exists(SUBMISSION_NAME) else None,
)
print("Submission head:")
print(pd.read_csv(SUBMISSION_NAME).head())
