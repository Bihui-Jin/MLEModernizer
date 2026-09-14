# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

4.94576

# 6. Current score

6.26674

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 319.32785) has done: 'I fix the two issues that prevent the notebook from running end-to-end: (1) the `MessageFactory.GetPrototype` import crash by avoiding the incompatible `tf_keras` package and using `tensorflow.keras` instead, and (2) the scaler failure caused by a leftover string `key` column in the training features (so we drop `key` before scaling, just like you already do for `testKaggle_clean`). These are execution/stability fixes and keep your model architecture, features, and training loop intact. After that, the pipeline successfully train, predict on the real Kaggle test set, and write a valid `submissiontry_water.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 119.97671) has done: 'I fix the TensorFlow/Keras import crash causing `MessageFactory.GetPrototype` by removing the hard dependency on `tensorflow` and instead using the already-installed `tf_keras` package (and forcing the legacy `.h5` save format for the checkpoint). Then I address the huge RMSE (319) by fixing a logic/data issue that makes the model train on a different feature set than it uses for Kaggle test prediction: currently `key` is present in `train_df` during splitting/cleaning and only dropped later, which can silently create inconsistent columns and scaling behavior; we enforce an explicit, identical feature column list for train/validation/test/testKaggle before scaling. These changes preserve your model architecture, feature engineering, loss, and training loop, but ensure train-time and predict-time feature alignment so the model can generalize and the Kaggle submission score moves toward the target band. The script still run end-to-end and write a valid `submissiontry_water.csv` with `key,fare_amount`.'
- What this solution (achieved 15.15512) has done: 'I fix the runtime crash in the first cell by preventing the known `protobuf`/`tf_keras` `MessageFactory.GetPrototype` incompatibility: we stop importing `tf_keras` and instead use the already-installed standalone `keras` package (same Sequential/Dense/BatchNorm APIs, same training loop). Then I make one minimal, score-relevant correction that preserves your core approach but avoids obviously-wrong predictions on Kaggle: clip final fare predictions to a reasonable non-negative range (the training cleaning already restricts fares to (0, 50], so this is consistent and typically reduces RMSE). Finally, I keep the submission writing unchanged but add a small safety check to ensure the submission has the required columns and row count.'
- What this solution (achieved 5.54589) has done: 'I fix the current runtime blockers by (1) avoiding the `keras` import path that triggers the protobuf `MessageFactory.GetPrototype` crash in this environment and using the already-installed `tf_keras` package instead, and (2) updating the custom `rmse` metric to use backend-agnostic TensorFlow ops (so it no longer calls missing `keras.backend.sqrt`). These are execution fixes that keep your model architecture, features, and training loop the same. To move RMSE toward your target, I also make one minimal, score-relevant correction: apply the exact same `clean()` filtering to the Kaggle test set before feature generation (previously it wasn’t cleaned at all), then predict only on the cleaned rows and merge back to all keys using a robust default for dropped rows; this typically reduces large errors from invalid coordinates without changing the model itself. The script still write a valid `submissiontry_water.csv` with the required `key,fare_amount` columns and correct row count.'
- What this solution (achieved 5.95869) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by removing the incompatible `tensorflow`/`tf_keras` import combination and using the standalone `keras` package consistently (same Sequential/Dense/BatchNorm model, same training loop). Then I make one minimal score-improving correction that preserves your approach: compute the scaler from the same columns for train/validation/test/Kaggle-test by explicitly dropping `key` everywhere and reindexing Kaggle features to `train_df`’s columns before scaling (prevents subtle column mismatch and improves generalization). Finally, I keep your current Kaggle-test cleaning + merge-back logic, but ensure predictions are always finite and clipped, and always write a valid `submissiontry_water.csv` with the required `key,fare_amount` columns and row count.'
- What this solution (achieved 5.83073) has done: 'I fix the current runtime crash (`MessageFactory.GetPrototype`) by removing the incompatible standalone `keras` + `tensorflow` combination and instead using the already-installed `tf_keras` consistently (same Sequential/Dense/BatchNorm architecture and training loop). Then I make one minimal, score-relevant correction: the code currently *drops rows from the Kaggle test set* via `clean()`, which can distort predictions by forcing a fallback fare for many rows; instead, we only apply lightweight feature-safe cleaning (NaN handling) to test, generate features for all rows, and keep the model’s training cleaning unchanged. Finally, I keep the submission-writing logic intact but ensure feature columns are aligned and the output CSV is always valid with the required `key,fare_amount` columns and exact row count.'
- What this solution (achieved 7.83235) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by removing the incompatible TensorFlow import and rewriting the custom `rmse` metric to use `tf_keras.backend` ops only, keeping your model, loss, and training loop unchanged. Then I make one minimal, score-relevant correction that preserves your current feature engineering: apply the same lightweight test cleaning (`dropna`) *and* enforce identical feature columns by reindexing Kaggle test features to the train feature list before scaling/predicting. Finally, I keep the submission logic intact but add a small safety guard to ensure predictions align 1:1 with the raw test keys and the output CSV is valid.'
- What this solution (achieved 15.17575) has done: 'I fix the runtime crash happening at import time by removing the TensorFlow/tf_keras/protobuf incompatibility and using the already-installed standalone `keras` package consistently for model, callbacks, and backend ops. This is a minimal execution fix that keeps your exact model architecture, features, loss, optimizer, and training loop intact, but allows the notebook to run end-to-end. To move RMSE toward your target (lower is better) with minimal semantic change, I also make the prediction post-processing consistent with your training cleaning by clipping fares to the same (0, 50] range (already partially done) and ensuring any NaNs from datetime parsing are handled deterministically. The script still write a valid `submissiontry_water.csv` with the required `key,fare_amount` columns and the exact test row count.'
- What this solution (achieved 6.17794) has done: 'I fix the current runtime blockers by (1) preventing the protobuf `MessageFactory.GetPrototype` crash at import time by using the already-installed `tf_keras` package consistently (instead of standalone `keras`), and (2) fixing the custom `rmse` metric to use backend-agnostic ops (`tf_keras.backend`) that exist in this environment. These changes are execution/stability fixes and keep your model architecture, loss, optimizer, feature engineering, and training loop intact. To move RMSE toward your target (lower is better) without changing the core approach, I also stop using the `"accuracy"` metric (which is invalid for regression and can create misleading training behavior) while keeping the regression loss/labels unchanged. The script still train end-to-end and write a valid `submissiontry_water.csv` with `key,fare_amount` and the exact test row count.'
- What this solution (achieved 15.21651) has done: 'I fix the import-time `MessageFactory.GetPrototype` crash by avoiding the incompatible `tf_keras`/protobuf combination and using the installed standalone `keras` package consistently (same Sequential/Dense/BatchNorm model and same training loop). Then, to move RMSE down toward your target with minimal semantic change, I (a) use a slightly larger training sample size (still read as `nrows`, not changing the approach) and (b) add one additional distance feature (haversine) alongside your existing coordinate/time/manhattan features, which keeps the same feature-engineering pattern but usually improves taxi-fare accuracy. Finally, I keep the existing submission-writing logic but add small guards to ensure feature alignment and that the submission CSV is always valid (`key,fare_amount`, correct row count).'
- What this solution (achieved 7.85205) has done: 'I fix the two runtime blockers so training can actually start: the protobuf `MessageFactory.GetPrototype` crash caused by importing standalone `keras`, and the broken custom `rmse` metric that calls missing backend ops in this environment. The minimal safe fix is to consistently use the already-installed `tf_keras` (Keras 2.x API) and implement `rmse` with `tf_keras.backend` ops that exist. These changes preserve your model architecture, features, and training loop semantics, but allow the notebook to run end-to-end and produce a valid `submissiontry_water.csv`. With training no longer broken, the score should move down from ~15 toward your target band simply because the model train correctly instead of failing/using invalid metrics.'
- What this solution (achieved 6.01922) has done: 'I fix the import-time `MessageFactory.GetPrototype` crash by removing the `tf_keras` dependency and using `tensorflow.keras` consistently (same Sequential/Dense/BatchNorm model, optimizer, callbacks, and training loop). This is the earliest blocker preventing any training/inference and is score-neutral aside from allowing the model to actually run. I also make the RMSE metric use TensorFlow ops directly to avoid backend incompatibilities, while keeping the loss/targets identical. Finally, I keep your feature pipeline intact but add a tiny safety alignment: ensure the Kaggle test features have the exact same columns/dtypes as train before scaling so predictions are well-formed and the submission is valid.'
- What this solution (achieved 6.26674) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by avoiding TensorFlow imports in this environment and using the already-installed `tf_keras` API consistently for model/ops/callbacks (this keeps your model architecture and training loop the same but makes it run). Then I make one minimal, score-relevant correction that preserves your feature pipeline: ensure the Kaggle test features are generated for *all* test rows (no row-dropping) and are aligned to the exact train feature columns before scaling/predicting, to prevent silent column/NaN issues that typically worsen RMSE. Finally, I keep your existing prediction clipping consistent with training outlier filtering and guarantee a valid `submissiontry_water.csv` with the required `key,fare_amount` columns and correct row count.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tf_keras as tfk
import tensorflow as tf  # used for @tf.function and tensor ops; tf import is generally OK without keras bindings

from tf_keras.models import Sequential
from tf_keras.layers import Dense, BatchNormalization
from tf_keras.callbacks import ModelCheckpoint
from tf_keras import optimizers, regularizers

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"

if not os.path.exists(TRAIN_PATH):
    alt_train = "/kaggle/input/train.csv"
    alt_test = "/kaggle/input/test.csv"
    if os.path.exists(alt_train):
        TRAIN_PATH = alt_train
    if os.path.exists(alt_test):
        TEST_PATH = alt_test

SUBMISSION_NAME = "submissiontry_water.csv"  # must end with .csv

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001

DATASET_SIZE = 200000

np.random.seed(1)
tf.random.set_seed(1)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def remove_datapoints_from_water(df):
    """
    Original intent: filter trips whose pickup/dropoff are on land using an NYC mask image.
    Bugfix: Kaggle kernels typically have no internet; matplotlib can't imread(URL).
    Minimal fix: try to load a local mask if present; otherwise skip this filter.
    """

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    local_mask_path = "nyc_mask-74.5_-72.8_40.5_41.8.png"
    if not os.path.exists(local_mask_path):
        return df

    nyc_mask = plt.imread(local_mask_path)[:, :, 0] > 0.9

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

    if "passenger_count" in df.columns:
        df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
        print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

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
    df2 = remove_datapoints_from_water(df)
    print("New size: %d" % len(df2))
    print(" New size: %d" % len(df2))
    return df2


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    )
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["latdiff"] = (lat1 - lat2).abs()
    df["londiff"] = (lon1 - lon2).abs()
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    return df


def add_haversine_feature(df):
    lat1 = np.radians(df["pickup_latitude"].astype("float64"))
    lon1 = np.radians(df["pickup_longitude"].astype("float64"))
    lat2 = np.radians(df["dropoff_latitude"].astype("float64"))
    lon2 = np.radians(df["dropoff_longitude"].astype("float64"))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R_km = 6371.0
    df["haversine_km"] = (R_km * c).astype("float32")
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df_out = pd.DataFrame(
        {
            id_column: raw_test[id_column].values,
            prediction_column: np.asarray(prediction).reshape(-1),
        }
    )
    df_out.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df_out))


@tf.function
def rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=-1))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    if "rmse" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["rmse"])
        plt.plot(history.history["val_rmse"])
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper right")
        plt.show()


def clean_test_only(df):
    df = df.copy()
    df = df.dropna(how="any", axis="rows")
    return df




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

trainKaggle = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    dtype=datatypes,
    usecols=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)

testKaggle = pd.read_csv(
    TEST_PATH,
    dtype={k: v for k, v in datatypes.items() if k != "fare_amount"},
)



## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]



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

print("testKaggle clean (test-only: keep all rows, avoid heavy filtering)")
testKaggle_raw = testKaggle.copy()
testKaggle_cleaned = clean_test_only(testKaggle)



## === cell 8
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle_cleaned = add_time_features(testKaggle_cleaned)



## === cell 9
train_df.describe()



## === cell 10
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features Disabled!")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle_cleaned = add_coordinate_features(testKaggle_cleaned)



## === cell 11
train_df.describe()



## === cell 12
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle_cleaned = add_distances_features(testKaggle_cleaned)

print("train_df add_haversine_feature")
train_df = add_haversine_feature(train_df)
print("test_df add_haversine_feature")
test_df = add_haversine_feature(test_df)
print("testKaggle add_haversine_feature")
testKaggle_cleaned = add_haversine_feature(testKaggle_cleaned)

print("Done with Adding features")



## === cell 13
train_df.describe()



## === cell 14
if len(train_df) >= 2000:
    plot = train_df.iloc[:2000].plot.scatter("latdiff", "londiff")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "weekday")



## === cell 15
dropped_columns = ["pickup_datetime"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

testKaggle_features = testKaggle_cleaned.drop(dropped_columns + ["key"], axis=1)

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
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

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
for _df_name, _df in [
    ("train_df", train_df),
    ("validation_df", validation_df),
    ("test_df", test_df),
]:
    if "key" in _df.columns:
        _df.drop(["key"], axis=1, inplace=True)

common_cols = list(train_df.columns)

testKaggle_features = testKaggle_features.reindex(columns=common_cols)

train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
validation_df = validation_df.replace([np.inf, -np.inf], np.nan).fillna(0)
test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)
testKaggle_features = testKaggle_features.replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_features)



## === cell 28
test_scaled



## === cell 29
checkpoint = ModelCheckpoint(
    filepath="my_model.h5", verbose=1, save_best_only=True, save_weights_only=False
)

model = Sequential()
model.add(
    Dense(
        256,
        activation="linear",
        input_dim=train_df_scaled.shape[1],
        activity_regularizer=regularizers.l1(0.01),
    )
)
model.add(BatchNormalization())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(32, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(8, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(loss="mean_squared_error", optimizer=adam, metrics=["mae", rmse, "mse"])

print("Dataset size: %s" % DATASET_SIZE)
print("Epochs: %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % list(train_df.columns))
model.summary()

history = model.fit(
    x=train_df_scaled,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[checkpoint],
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)



## === cell 30
pass



## === cell 31
plot_loss_accuracy_rmse(history)



## === cell 32
score = model.evaluate(train_df_scaled, train_labels, verbose=1)
print(score)
print("train mean_squared_error:", score[0])
print("train mae:", score[1])
print("train rmse:", score[2])
print("train mse:", score[3])



## === cell 33
score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print(score)
print("Validation mean_squared_error:", score[0])
print("Validation mae:", score[1])
print("Validation rmse:", score[2])
print("Validation mse:", score[3])



## === cell 34
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print("Test mean_squared_error:", score[0])
print("Test mae:", score[1])
print("Test rmse:", score[2])
print("Test mse:", score[3])



## === cell 35
validation_predictions = model.predict(validation_df_scaled).flatten()

plt.scatter(validation_labels, validation_predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot(
    [validation_predictions.min(), validation_predictions.max()],
    [validation_predictions.min(), validation_predictions.max()],
    "k--",
    lw=4,
)



## === cell 36
test_predictions = model.predict(test_scaled).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot(
    [test_predictions.min(), test_predictions.max()],
    [test_predictions.min(), test_predictions.max()],
    "k--",
    lw=4,
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
ax.scatter(test_labels, test_predictions)
ax.plot(
    [test_labels.min(), test_labels.max()],
    [test_labels.min(), test_labels.max()],
    "k--",
    lw=4,
)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()



## === cell 40
plt.figure(figsize=(20, 10))
plt.plot(validation_labels[:100])
plt.plot(validation_predictions[:100])
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
errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
print(len(error))
print(len(errorGreaterZero))

plt.hist(errorGreaterZero, bins=100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## === cell 45
testKaggle_full = clean_test_only(testKaggle_raw.copy())
testKaggle_full = add_time_features(testKaggle_full)
testKaggle_full = add_coordinate_features(testKaggle_full)
testKaggle_full = add_distances_features(testKaggle_full)
testKaggle_full = add_haversine_feature(testKaggle_full)

testKaggle_full_features = testKaggle_full.drop(["pickup_datetime", "key"], axis=1)
testKaggle_full_features = testKaggle_full_features.reindex(columns=common_cols)
testKaggle_full_features = testKaggle_full_features.replace(
    [np.inf, -np.inf], np.nan
).fillna(0)
testKaggle_full_scaled = scaler.transform(testKaggle_full_features)

predictionKaggle = model.predict(testKaggle_full_scaled, batch_size=128, verbose=1)
predictionKaggle = np.asarray(predictionKaggle).reshape(-1)

predictionKaggle = np.where(np.isfinite(predictionKaggle), predictionKaggle, np.nan)
fallback_fare = float(np.clip(np.mean(train_labels), 0.01, 50.0))
predictionKaggle = np.nan_to_num(predictionKaggle, nan=fallback_fare)
predictionKaggle = np.clip(predictionKaggle, 0.01, 50.0).astype("float32")

if len(predictionKaggle) != len(testKaggle_raw):
    print(
        "Warning: prediction length mismatch; adjusting with fallback to match raw test length."
    )
    fixed = np.full(len(testKaggle_raw), fallback_fare, dtype="float32")
    n = min(len(predictionKaggle), len(fixed))
    fixed[:n] = predictionKaggle[:n]
    predictionKaggle = fixed

sub = pd.DataFrame(
    {"key": testKaggle_raw["key"].values, "fare_amount": predictionKaggle}
)
sub.to_csv(SUBMISSION_NAME, index=False)
print("Wrote submission to:", os.path.abspath(SUBMISSION_NAME), "rows:", len(sub))

sub2 = pd.read_csv(SUBMISSION_NAME)
assert list(sub2.columns) == [
    "key",
    "fare_amount",
], f"Bad submission columns: {sub2.columns}"
assert len(sub2) == len(
    testKaggle_raw
), f"Bad submission length: {len(sub2)} vs {len(testKaggle_raw)}"
print(sub2.head())
