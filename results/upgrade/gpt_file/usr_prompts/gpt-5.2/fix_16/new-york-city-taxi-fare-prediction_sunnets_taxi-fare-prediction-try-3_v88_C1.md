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

4.31009

# 6. Current score

6.38375

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 111.90148) has done: 'I fix the crashes caused by (1) `plt.imread()` trying to read a URL (not supported, and internet may be blocked), (2) the Keras 3 optimizer API change (`optimizers.adam` no longer exists), and (3) an optional model-plot import that doesn’t exist in this environment. I keep your model and training loop intact, but make the water-mask step safely optional/fallback so cleaning doesn’t delete everything or error, and ensure the code always reaches submission writing. I also correct two time-feature boolean bugs (`late_night`, `night`) that currently make those features incorrect (even if you don’t use them later, this is score-positive and minimal). Finally, I update paths to the provided Kaggle dataset location and ensure the submission is written as a valid `.csv` with the required columns.'
- What this solution (achieved 5.55562) has done: 'I fix the immediate runtime crash happening before your own code runs by forcing the pure-Python protobuf implementation (this resolves the `MessageFactory.GetPrototype` AttributeError seen with some tf/keras/protobuf combos on Kaggle). Then I fix the biggest logic issue causing the very poor RMSE: your model is trained on a *different cleaned/train-split distribution* than the Kaggle test set, but you scale Kaggle test using a scaler fitted on that cleaned split without applying the same cleaning to Kaggle test; I apply the exact same `clean()` pipeline to `testKaggle` before feature engineering and scaling so features are aligned. Finally, I make submission creation robust to any rows dropped by cleaning by starting from `sample_submission.csv` and merging predictions by `key`, ensuring a valid submission with all required keys and `.csv` suffix.'
- What this solution (achieved 98.67914) has done: 'I fix the TensorFlow/Keras/protobuf crash by setting the protobuf env vars before any TF/Keras import and by avoiding the problematic default `MessageFactory` path. Then I make the Kaggle-test cleaning step score-safe by not dropping rows from `test.csv` (dropping test rows forces mean-filling later, which hurts RMSE), while keeping the exact same training cleaning logic and model/training loop intact. Finally, I ensure feature engineering produces no NaNs (from bad datetimes) and keep submission creation aligned to `sample_submission.csv` keys so the output is always valid and complete.'
- What this solution (achieved 121.96193) has done: 'I fix the immediate runtime crash caused by the protobuf/TensorFlow-Keras incompatibility by ensuring the environment variables are set before *any* keras/tf import and by avoiding importing `tf_keras` when it fails. Then I keep your exact model/training loop intact but switch to the built-in `tensorflow` + `tf.keras` backend (same layers/optimizer/loss/metrics) so the notebook runs end-to-end reliably in Kaggle. Finally, I keep your current “non-dropping” test cleaning and submission merge-by-key logic, but make the inference scaling robust to any column-order mismatch (a common silent bug that can severely hurt RMSE). These changes are execution/stability fixes and a small correctness fix for feature alignment, expected to improve score toward the target without changing the modeling approach.'
- What this solution (achieved 7.78002) has done: 'I fix the crash in the very first cell caused by a known TensorFlow/Keras + protobuf incompatibility by forcing the pure-Python protobuf implementation and disabling the C++ one *before* any TF import. Then I keep your exact model/training pipeline but ensure the runtime stays within Kaggle’s limits by preventing extremely slow/blocked operations (the remote water mask) from triggering long timeouts. Finally, to move RMSE strongly toward the target (current 121.96 → target 4.31, lower is better), I correct the submission prediction scale issue that typically causes huge errors: enforce non-negative fares and clip extreme predictions to the same target range used in training cleaning (0–50), which is a minimal, metric-aligned post-processing step.'
- What this solution (achieved 8.18471) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by pinning the pure-Python protobuf implementation and also forcing the legacy Python protos path before importing TensorFlow (this is the root runtime blocker). Then I keep your exact model/training pipeline intact but fix a key feature-alignment bug: `testKaggle_clean` is currently filled with zeros **after** reindexing, which can silently wipe whole columns if they were missing/invalid; I instead impute missing values in `testKaggle_clean` using the **training-feature medians** (score-positive and consistent with your scaling). Finally, I make submission writing robust and deterministic (ensuring float dtype, correct columns, and row count), without changing your architecture, epochs, batch size, optimizer, or loss.'
- What this solution (achieved 6.8187) has done: 'I fix the protobuf/TensorFlow crash by moving the protobuf environment variables to the very top and importing TensorFlow only after that, plus adding a safe fallback to `tf_keras` if the crash still occurs in this environment. Then I correct a subtle but score-impacting preprocessing mismatch: the Kaggle test set currently uses medians computed on *unscaled* features but is passed through a scaler fitted on the training set—this is correct, but we also need to ensure the Kaggle test columns are strictly numeric and ordered identically *before* any NaN filling and scaling to avoid silent dtype/object contamination. Finally, I keep your model/training exactly the same, but I make the submission writing deterministic and guaranteed complete (all keys, correct columns, `.csv` suffix) without changing evaluation semantics.'
- What this solution (achieved 7.74861) has done: 'I fix the immediate runtime blocker (`MessageFactory.GetPrototype` protobuf issue) by forcing the pure-Python protobuf runtime and legacy/proto API *before* any TensorFlow/Keras import, and by explicitly importing `google.protobuf` early to ensure the env vars take effect. I also make the custom `rmse` metric import backend from the active Keras backend (tf.keras vs tf_keras) to avoid backend mismatches at runtime. Finally, to move RMSE down toward your target without changing the model/training loop, I fix a small but score-relevant preprocessing mismatch by applying the same “clip passenger_count to [1,6]” logic to train/validation/test splits as is already done for Kaggle test, keeping features consistent between train and inference.'
- What this solution (achieved 6.53962) has done: 'I fix the immediate runtime blocker in the first cell caused by an incompatible protobuf runtime by removing the forced pure-Python protobuf settings and instead forcing TensorFlow to use its bundled protobuf implementation (this is the standard Kaggle-safe workaround for the `MessageFactory.GetPrototype` error). I keep your model, training loop, and feature pipeline intact, but add a tiny, metric-aligned calibration step on predictions (shift by the validation-set residual mean) to reduce RMSE toward the target without changing architecture/training. I also make submission writing robust and deterministic (still using sample_submission merge-by-key) and keep the `.csv` suffix and required columns. All other logic is preserved.'
- What this solution (achieved 8.19089) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* any TensorFlow/Keras import (this is the most reliable workaround for the `MessageFactory.GetPrototype` error in Kaggle’s mixed TF/Keras environments). Then I keep your model/training loop intact, but make one minimal score-positive correction: apply the same numeric/median imputation used for Kaggle test using **training medians** (not test medians) so inference preprocessing matches training distribution more closely. Finally, I keep your submission merge-by-key logic but ensure the script always writes a valid `.csv` with the required columns even if anything upstream produces NaNs.'
- What this solution (achieved 8.17666) has done: 'I fix the root runtime blocker by removing the incompatible TensorFlow import path (protobuf `_message` ImportError) and switching to scikit-learn’s `MLPRegressor` while keeping your core “dense MLP on MinMax-scaled engineered features” approach and train/validation split semantics intact. I also make the Kaggle test preprocessing fully consistent with training (same time + manhattan features, same column order, numeric coercion, and train-median imputation) so inference doesn’t silently mis-scale inputs. Finally, I guarantee a valid `.csv` submission is always written with exactly the required `key,fare_amount` columns and all keys from `sample_submission.csv`, and keep the existing minimal calibration (validation residual mean shift + clipping to [0, 50]) because it is metric-aligned for RMSE.'
- What this solution (achieved 7.99599) has done: 'Your current RMSE (8.17666) is far above the target (4.31009), so we should improve generalization with the smallest changes that don’t alter the overall pipeline (engineered features → MinMax scaling → dense MLP regression). The biggest score lever that stays within your core logic is using a more appropriate scaler for an MLP (StandardScaler) and adding a single additional distance feature (haversine) alongside your existing manhattan distance. Both changes are minimal, keep the same training loop/model family, and typically reduce RMSE substantially for this competition without touching the rest of your cleaning or submission alignment. I also keep your current validation residual-mean calibration + clipping, since it’s metric-aligned and already stable.'
- What this solution (achieved 6.38375) has done: 'We need to move your RMSE down (lower is better) from 7.996 toward 4.31, so we should improve generalization without changing the overall pipeline (clean → time/distance features → scaling → MLPRegressor). The largest issue still consistent with “minimal changes” is that `MLPRegressor` with `max_iter=EPOCHS` is currently not doing “100 epochs” over your data; it is a solver-iteration limit and often stops underfit or unstable—switching to `warm_start=True` and doing a fixed number of `.fit()` passes keeps the same model/training approach but makes training behave like true epochs. Second, we log-transform the target (`fare_amount`) for training and invert predictions; this is a standard, metric-aligned change for heavy-tailed fares and typically reduces RMSE substantially while keeping the same regressor and loss (still squared error, just in transformed space). Finally, we make test-time clipping consistent with the inverse-transform and keep your existing residual-mean calibration (applied in fare space) so the submission stays stable and valid.'
- What this solution (achieved 6.38375) has done: 'Your RMSE (6.38375, lower is better) is still above the target (4.31009), so we should make the smallest preprocessing/training corrections that typically improve generalization without changing your model family or loop structure. I keep your MLPRegressor, feature set, scaler, log-target option, and warm_start epoch loop intact, but fix one score-hurting mismatch: the Kaggle test cleaning currently imputes its own medians (distribution shift vs train); instead we impute test using the **training-feature medians** consistently. I also fix the calibration sign: residual mean `(pred - true)` should be **subtracted from predictions**, but your current code subtracts a negative when underpredicting and can amplify bias; we apply the correction in the correct direction while preserving the same “mean residual” calibration idea. Finally, I keep submission merging-by-key and clipping, ensuring a valid CSV is always produced.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

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
DATASET_SIZE = 800000

USE_LOG_TARGET = True

print("Using TRAIN_PATH:", TRAIN_PATH)
print("Using TEST_PATH:", TEST_PATH)
print("Using SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("Will write submission to:", os.path.abspath(SUBMISSION_NAME))



## === cell 1
import urllib.request
from PIL import Image


def remove_datapoints_from_water(df):
    """
    Tries to fetch an NYC land mask; if unavailable (offline), skip water filtering.
    """

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)
    url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"

    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            nyc_mask_img = Image.open(resp)
            nyc_mask = np.array(nyc_mask_img)[:, :, 0] > 0.9
    except Exception as e:
        print(
            "Warning: could not load NYC water mask (offline or blocked). Skipping water filtering. Error:",
            repr(e),
        )
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
    print(" New size after NYC lang lot: %d" % len(df))

    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["dropoff_longitude"] != 0)]
    df = df[(df["dropoff_latitude"] != 0)]
    print(" New size after lang lot > 0: %d" % len(df))

    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)]
    print(" New size after lang - lot > 0.001: %d" % len(df))

    print(" New size after only NYC: %d" % len(df))
    if "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    if "passenger_count" in df.columns:
        df["passenger_count"] = df["passenger_count"].clip(lower=1, upper=6)

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)

    df = df[
        (nyc_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != nyc_coord[0])
    ]
    df = df[
        (nyc_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != nyc_coord[0])
    ]
    print(" New size after NY airport: %d" % len(df))

    df = df[
        (fk_coord[1] != df["pickup_longitude"]) & (df["pickup_latitude"] != fk_coord[0])
    ]
    df = df[
        (fk_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != fk_coord[0])
    ]
    print(" New size after jfk airport: %d" % len(df))

    df = df[
        (ewr_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != ewr_coord[0])
    ]
    df = df[
        (ewr_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != ewr_coord[0])
    ]
    print(" New size after ewr airport: %d" % len(df))

    df = df[
        (lga_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != lga_coord[0])
    ]
    df = df[
        (lga_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != lga_coord[0])
    ]
    print(" New size after lgr airport: %d" % len(df))

    df = df[
        (sol_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != sol_coord[0])
    ]
    df = df[
        (sol_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != sol_coord[0])
    ]
    print(" New size after sol removed: %d" % len(df))

    print("Old size: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def clean_test_no_drop(df):
    print(" Old size (test): %d" % len(df))
    df = df.copy()

    num_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "passenger_count" in df.columns:
        df["passenger_count"] = df["passenger_count"].fillna(1)
        df["passenger_count"] = df["passenger_count"].clip(lower=1, upper=6)

    print(" New size (test) after non-dropping clean: %d" % len(df))
    return df


def late_night(row):
    return 1 if (row["hour"] <= 3) else 0


def night(row):
    return 1 if ((row["hour"] > 20 or row["hour"] < 6) and (row["weekday"] < 5)) else 0


def rush_hour(row):
    return (
        1
        if ((row["hour"] <= 20) and (row["hour"] >= 16) and (row["weekday"] < 5))
        else 0
    )


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=False
    )
    df["year"] = df["pickup_datetime"].dt.year.fillna(0).astype(np.int16)
    df["month"] = df["pickup_datetime"].dt.month.fillna(0).astype(np.int8)
    df["day"] = df["pickup_datetime"].dt.day.fillna(0).astype(np.int8)
    df["hour"] = df["pickup_datetime"].dt.hour.fillna(0).astype(np.int8)
    df["weekday"] = df["pickup_datetime"].dt.weekday.fillna(0).astype(np.int8)
    return df


def add_coordinate_features(df):
    _ = df["pickup_latitude"]
    _ = df["dropoff_latitude"]
    _ = df["pickup_longitude"]
    _ = df["dropoff_longitude"]
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["haversine_km"] = haversine_km(lat1, lon1, lat2, lon2).astype(np.float32)
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))


def plot_loss_accuracy_rmse(history):
    print(
        "plot_loss_accuracy_rmse skipped (sklearn model does not provide Keras History)."
    )




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
trainKaggle = pd.read_csv(
    TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=usecols_train
)
testKaggle = pd.read_csv(
    TEST_PATH, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
)

print("Loaded trainKaggle:", trainKaggle.shape, "testKaggle:", testKaggle.shape)



## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000].copy()



## === cell 4
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## === cell 5
train_df.describe()



## === cell 6
test_df.describe()



## === cell 7
print("train_df clean")
train_df = clean(train_df)
test_df = clean(test_df)

print("testKaggle clean (non-dropping)")
testKaggle = clean_test_no_drop(testKaggle)



## === cell 8
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 9
train_df.describe()



## === cell 10
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features Disabled!")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 11
train_df.describe()



## === cell 12
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## === cell 13
train_df.describe()



## === cell 14
try:
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "weekday")
except Exception as e:
    print("Plotting skipped:", repr(e))



## === cell 15
dropped_columns = ["pickup_datetime"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")



## === cell 16
train_df.shape



## === cell 17
train_df.describe()



## === cell 18
test_df.describe()



## === cell 19
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 20
train_df_main = train_df
validation_df_main = validation_df



## === cell 21
validation_df.describe()



## === cell 22
train_labels = train_df["fare_amount"].values.astype(np.float32)
validation_labels = validation_df["fare_amount"].values.astype(np.float32)
test_labels = test_df["fare_amount"].values.astype(np.float32)

if USE_LOG_TARGET:
    train_labels_model = np.log1p(train_labels).astype(np.float32)
    validation_labels_model = np.log1p(validation_labels).astype(np.float32)
    test_labels_model = np.log1p(test_labels).astype(np.float32)
else:
    train_labels_model = train_labels
    validation_labels_model = validation_labels
    test_labels_model = test_labels

drop_target_and_id = ["fare_amount", "key"]
train_df = train_df.drop(drop_target_and_id, axis=1)
validation_df = validation_df.drop(drop_target_and_id, axis=1)
test_df = test_df.drop(drop_target_and_id, axis=1)

print("Done with Labels")



## === cell 23
test_labels



## === cell 24
train_df.describe()



## === cell 25
validation_df.describe()



## === cell 26
test_df.describe()



## === cell 27
train_feature_cols = list(train_df.columns)


def _coerce_numeric(df, cols):
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


train_df = _coerce_numeric(train_df, train_feature_cols)
validation_df = _coerce_numeric(validation_df, train_feature_cols)
test_df = _coerce_numeric(test_df, train_feature_cols)

testKaggle_clean = testKaggle_clean.reindex(columns=train_feature_cols)
testKaggle_clean = _coerce_numeric(testKaggle_clean, train_feature_cols)

train_feature_medians = train_df.median(numeric_only=True)

train_df = train_df.fillna(train_feature_medians)
validation_df = validation_df.fillna(train_feature_medians)
test_df = test_df.fillna(train_feature_medians)

testKaggle_clean = testKaggle_clean.fillna(train_feature_medians).fillna(0)

scaler = preprocessing.StandardScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)

test_scaled




## === cell 28
def rmse_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _inverse_target(x):
    x = np.asarray(x, dtype=np.float64)
    if USE_LOG_TARGET:
        return np.expm1(x)
    return x




## === cell 29
model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 8),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=BATCH_SIZE,
    learning_rate_init=LEARNING_RATE,
    max_iter=1,  # one solver iteration per outer loop "epoch"
    warm_start=True,  # continue training across .fit() calls
    shuffle=True,
    random_state=1,
    early_stopping=False,  # do NOT introduce early stopping
    verbose=False,  # keep runtime output smaller; training semantics unchanged
)

print("Dataset size:", DATASET_SIZE)
print("Epochs (outer loop):", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Batch size:", BATCH_SIZE)
print("Input dimension:", train_df_scaled.shape[1])
print("Features used:", list(train_df.columns))

for ep in range(EPOCHS):
    model.fit(train_df_scaled, train_labels_model)
    if (ep + 1) % 10 == 0:
        val_pred_ep = _inverse_target(model.predict(validation_df_scaled))
        print(
            f"Epoch {ep+1}/{EPOCHS} - Validation RMSE:",
            rmse_np(validation_labels, val_pred_ep),
        )



## === cell 30
print("Skipping model_to_dot visualization (module not available in this environment).")



## === cell 31
print("Skipping Keras history plots (sklearn model).")



## === cell 32
train_pred = _inverse_target(model.predict(train_df_scaled))
print("Train RMSE:", rmse_np(train_labels, train_pred))



## === cell 33
val_pred = _inverse_target(model.predict(validation_df_scaled))
print("Validation RMSE:", rmse_np(validation_labels, val_pred))



## === cell 34
test_pred = _inverse_target(model.predict(test_scaled))
print("Test RMSE:", rmse_np(test_labels, test_pred))



## === cell 35
validation_predictions = np.asarray(val_pred, dtype=np.float32).flatten()

plt.scatter(validation_labels, validation_predictions, s=2, alpha=0.3)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot(
    [validation_predictions.min(), validation_predictions.max()],
    [validation_predictions.min(), validation_predictions.max()],
    "k--",
    lw=2,
)



## === cell 36
test_predictions = np.asarray(test_pred, dtype=np.float32).flatten()

plt.scatter(test_labels, test_predictions, s=2, alpha=0.3)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot(
    [test_predictions.min(), test_predictions.max()],
    [test_predictions.min(), test_predictions.max()],
    "k--",
    lw=2,
)



## === cell 37
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_labels[np.argmax(test_predictions)])
test_df.iloc[np.argmax(test_predictions)]



## === cell 38
print(np.argmin(test_predictions))
print(test_predictions[np.argmin(test_predictions)])
print(test_labels[np.argmin(test_predictions)])
test_df.iloc[np.argmin(test_predictions)]



## === cell 39
fig, ax = plt.subplots()
ax.scatter(test_labels, test_predictions, s=2, alpha=0.3)
ax.plot(
    [test_labels.min(), test_labels.max()],
    [test_labels.min(), test_labels.max()],
    "k--",
    lw=2,
)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()



## === cell 40
plt.figure(figsize=(20, 10))
plt.plot(validation_labels[:1000])
plt.plot(validation_predictions[:1000])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()

validation_df_scaled[52]
validation_df.iloc[52]



## === cell 41
plt.figure(figsize=(20, 10))
plt.plot(test_labels[:100])
plt.plot(test_predictions[:100])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()



## === cell 42
error = validation_predictions - validation_labels
plt.hist(error, bins=100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## === cell 43
error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## === cell 44
print(len(error))

errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
print(len(errorGreaterZero))

plt.hist(errorGreaterZero, bins=100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## === cell 45
predictionKaggle_model = model.predict(testKaggle_scaled)
predictionKaggle = (
    _inverse_target(predictionKaggle_model).reshape(-1, 1).astype(np.float32)
)



## === cell 46
val_residual_mean = float(np.mean(validation_predictions - validation_labels))
print("Validation residual mean (pred - true):", val_residual_mean)

predictionKaggle = predictionKaggle - np.float32(val_residual_mean)

predictionKaggle = np.clip(predictionKaggle, 0.0, 50.0)

pred_df = pd.DataFrame(
    {"key": testKaggle["key"].values, "fare_amount": predictionKaggle.flatten()}
)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sub = sample_sub.merge(pred_df, on="key", how="left", suffixes=("_sample", ""))

if sub["fare_amount"].isna().any():
    fill_value = float(sample_sub["fare_amount"].mean())
    sub["fare_amount"] = sub["fare_amount"].fillna(fill_value)

sub["fare_amount"] = pd.to_numeric(sub["fare_amount"], errors="coerce").astype(
    np.float32
)
sub["fare_amount"] = (
    sub["fare_amount"]
    .fillna(float(sample_sub["fare_amount"].mean()))
    .astype(np.float32)
)

sub = sub[["key", "fare_amount"]]
sub.to_csv(SUBMISSION_NAME, index=False)
print("Output complete:", SUBMISSION_NAME, "rows:", len(sub))

print("Submission head:")
print(pd.read_csv(SUBMISSION_NAME).head())
