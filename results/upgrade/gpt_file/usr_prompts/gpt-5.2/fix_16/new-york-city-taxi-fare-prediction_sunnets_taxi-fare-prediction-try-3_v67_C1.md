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

4.70167

# 6. Current score

508.11941

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 35.07842) has done: 'I fix the environment/runtime crash caused by a protobuf/Keras import interaction by removing the unnecessary `keras` package import and using `tf_keras` only (same model logic). Then I fix the scaler failure by ensuring all remaining columns are numeric: keep `pickup_datetime` out of the feature matrix and also drop the string `key` column from train/valid/test feature frames (it was accidentally left in). Finally, I keep the training/inference pipeline the same but ensure the submission is written as a valid `key,fare_amount` CSV with the correct row alignment and non-negative fares.'
- What this solution (achieved 68.87497) has done: 'I fix the immediate runtime crash in the first cell caused by an incompatibility between `tf_keras` and the installed `protobuf` by switching the imports to use `tensorflow.keras` instead (same Sequential/Dense/BatchNorm model and training loop). I keep the exact feature engineering, cleaning, scaling, and training procedure unchanged, only touching imports and a small metric compatibility detail so the script runs end-to-end. This should also restore normal model training behavior (instead of crashing), which is necessary to move the RMSE score down toward the target. Finally, I keep the submission writing logic intact and ensure it always produces the required `key,fare_amount` CSV.'
- What this solution (achieved 6.71082) has done: 'I fix the runtime crash in the very first cell by forcing TensorFlow to use the pure-Python protobuf implementation (this avoids the `MessageFactory.GetPrototype` incompatibility seen in some Kaggle TF/protobuf builds) while keeping the exact same model, features, and training loop. Then I make the data loading paths robust to both `/kaggle/input/train.csv` style and the common nested competition folder layout, so the notebook always finds the CSVs. Finally, I keep submission generation identical but ensure it always writes a proper `key,fare_amount` CSV with correct alignment and non-negative fares as you already intended.'
- What this solution (achieved 248.81923) has done: 'I fix the immediate runtime crash coming from a protobuf/TensorFlow incompatibility by ensuring the protobuf “python” implementation is forced **before** any TensorFlow import (and by avoiding importing the standalone `keras` package). Then I keep your model and training loop intact but correct a key logic issue that hurts RMSE: you are training on an arbitrary 80k slice of the CSV (chronologically biased); switching to a reproducible random sample of 80k rows from the full train file typically improves generalization without changing the core approach. Finally, I keep the same feature engineering/cleaning/scaling and submission writing, ensuring the produced `submissiontry_water.csv` is valid and aligned with `test.csv`.'
- What this solution (achieved 52.15022) has done: 'I fix the immediate runtime crash (`MessageFactory` has no `GetPrototype`) by avoiding TensorFlow import paths that trigger the incompatible protobuf implementation; instead, I use the already-installed `tf_keras` package (same Keras API/core model code) and keep the protobuf env forcing at the very top. I also fix a major data-loading bug that is destroying your training data quality: the current random-sampling logic builds a massive `skiprows` list (tens of millions of ints), which can silently break/timeout and yield corrupted/biased sampling—this is a direct reason your RMSE blew up to ~248. Finally, I keep your feature engineering, cleaning, scaling, model architecture, and training loop intact, while making the train sampling memory-safe and deterministic and ensuring a valid `submissiontry_water.csv` is always written with `key,fare_amount`.'
- What this solution (achieved 122.84247) has done: 'You’re currently crashing immediately on import due to an incompatibility between `tf_keras` and the protobuf version in this Kaggle image (`MessageFactory.GetPrototype`). The minimal stable fix is to stop importing `tf_keras` and instead use `tensorflow.keras` (same Sequential/Dense/BatchNorm architecture, same optimizer/loss/training loop), while keeping the protobuf env vars set before TensorFlow import to avoid the common TF/protobuf issue. To move RMSE down toward your target without changing the modeling approach, I also make the “random sample” actually come from across the full 55M-row file (memory-safe) rather than only from the first chunk, which is heavily biased and is a common reason for very poor leaderboard RMSE. All other feature engineering, cleaning, scaling, and submission formatting remain the same, and the script always write a valid `.csv` submission with `key,fare_amount`.'
- What this solution (achieved 248.68545) has done: 'You’re failing immediately due to a TensorFlow/protobuf binary incompatibility, so I avoid importing TensorFlow altogether and switch to the already-installed `tf_keras` backend (same Keras Sequential/Dense/BatchNorm model, same optimizer/loss/fit loop). This fix is execution-blocking and should also restore normal training behavior, which move RMSE down substantially from the current ~122 toward your target. I also make the train sampling more stable (guarantee enough rows are collected and avoid pathological undersampling) without changing the “random sample from full train” approach. Finally, I keep submission formatting identical but ensure the output is always a valid `key,fare_amount` CSV aligned to `test.csv`.'
- What this solution (achieved 588.50903) has done: 'I fix the execution-blocking protobuf/TensorFlow issue by removing the `tf_keras` import path that triggers `MessageFactory.GetPrototype` and switching to `tensorflow.keras` (same Sequential/Dense/BatchNorm model, same loss/optimizer/training loop). I also ensure the protobuf “python” implementation env vars are set before importing TensorFlow, which is the minimal stability change for this Kaggle image. To move RMSE down from ~248 toward the 4.70 target without changing the modeling approach, I fix the train sampling so it reliably draws a true random sample across the full 55M-row file (the current keep_prob heuristic can easily under/over-sample and yield a poor-quality tiny pool). Finally, I keep the same feature engineering/cleaning/scaling and still write a valid `key,fare_amount` submission CSV aligned to `test.csv`.'
- What this solution (achieved 23.40126) has done: 'I fix the execution-blocking protobuf/TensorFlow crash (`MessageFactory` missing `GetPrototype`) by switching the deep learning stack to the already-installed `tf_keras` package and avoiding importing `tensorflow` entirely, while keeping the exact same Sequential/Dense/BatchNorm architecture, compile settings, and `.fit()` loop. I also keep the existing feature engineering and sampling logic intact, but make the data path resolution robust to your actual `/kaggle/data/...` layout so the script always finds `train.csv`/`test.csv`. Finally, I ensure the submission is always written as a valid `key,fare_amount` CSV with the correct row alignment and non-negative fares (same as your intended post-processing). These changes are minimal, unblock runtime, and should move RMSE down dramatically from the current broken run.'
- What this solution (achieved 1129.04845) has done: 'I fix the immediate runtime crash caused by importing `tf_keras` in this Kaggle image (protobuf `MessageFactory.GetPrototype` error) by switching the model imports to `tensorflow.keras` while keeping the exact same Sequential/Dense/BatchNorm architecture, optimizer/loss, and training loop. I also keep the protobuf “python” implementation environment variables set before TensorFlow import for stability. To move RMSE down toward your target without changing the modeling approach, I correct the training sampling logic so it truly samples across the full `train.csv` stream (instead of taking the first `take` rows of every chunk), which is a minimal change but materially improves generalization. Submission writing and column alignment remain unchanged, ensuring a valid `key,fare_amount` CSV is produced.'
- What this solution (achieved 212.84444) has done: 'I fix the execution-blocking TensorFlow/protobuf incompatibility that’s causing the `MessageFactory.GetPrototype` crash by avoiding the TensorFlow import path entirely and using the already-installed `tf_keras` backend (same Sequential/Dense/BatchNorm model, same optimizer/loss/fit loop). I keep your feature engineering, cleaning, scaling, sampling, and training loop intact, only adjusting imports and the backend reference used by the custom RMSE metric so it runs end-to-end. This should also move your score down dramatically from the broken ~1129 RMSE because the model actually train instead of failing at import/runtime. Submission writing remains the same and still produce a valid `key,fare_amount` CSV named `submissiontry_water.csv`.'
- What this solution (achieved 1831.77087) has done: 'I fix the execution-blocking protobuf/`tf_keras` import crash by switching to `tensorflow.keras` while keeping the exact same Sequential(Dense/BatchNorm) architecture, optimizer/loss, and `.fit()` loop. Then I make sure the custom RMSE metric uses the TensorFlow backend consistently so it compiles cleanly in this environment. Finally, I keep your sampling/feature engineering/cleaning/scaling unchanged, and ensure the submission file is always written as a valid `key,fare_amount` CSV with the same row alignment as `test.csv` (and non-negative fares as you intended). These changes should restore real training (instead of crashing), which is necessary to move RMSE down dramatically toward the 4.70 target.'
- What this solution (achieved 610.67403) has done: 'We need to fix the execution-blocking TensorFlow/protobuf crash in the very first cell (`MessageFactory` missing `GetPrototype`) while keeping the same Keras model, training loop, and features. The safest minimal change in this environment is to avoid importing TensorFlow entirely and use the already-installed `tf_keras` package with the exact same `Sequential/Dense/BatchNormalization` architecture and compile/fit semantics. I also make the RMSE metric backend come from the same Keras package to prevent mixed-backend issues, and keep all data handling and submission writing unchanged so a valid `submissiontry_water.csv` is always produced. These fixes should restore real training/inference and dramatically reduce RMSE from the current broken ~1831 toward your target band.'
- What this solution (achieved 508.11941) has done: 'We fix the immediate execution blocker (`MessageFactory.GetPrototype` crash) by avoiding the incompatible `tf_keras` import path and using `tensorflow.keras` instead, while keeping the exact same model architecture, loss, optimizer, and training loop. We also keep the protobuf “python” implementation env vars set before importing TensorFlow for stability in this Kaggle image. To move RMSE down toward the 4.70 target (from ~610), we keep your existing full-file reservoir sampling but fix its logic so it truly samples uniformly across the entire 55M-row file (your current version mistakenly never increments `seen` during sampling, which severely biases the sample and can destroy score). All other feature engineering, cleaning, scaling, and submission formatting remain the same, and the script always write a valid `submissiontry_water.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import optimizers, regularizers, backend


def _resolve_path(*candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[0]


TRAIN_PATH = _resolve_path(
    "/kaggle/input/train.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/data/train.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/train.csv",
)
TEST_PATH = _resolve_path(
    "/kaggle/input/test.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/data/test.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/test.csv",
)

SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

RANDOM_SAMPLE_FROM_FULL_TRAIN = True
SAMPLE_RANDOM_SEED = 1

np.random.seed(1)
try:
    tf.random.set_seed(1)
except Exception:
    pass

print("Using TRAIN_PATH:", TRAIN_PATH)
print("Using TEST_PATH:", TEST_PATH)
print("TensorFlow version:", getattr(tf, "__version__", "unknown"))
print("Keras version:", getattr(keras, "__version__", "unknown"))
print("Keras backend:", backend.backend())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def remove_datapoints_from_water(df):
    return df


def late_night(row):
    return 1 if (row["hour"] <= 3) else 0


def night(row):
    return 1 if (row["hour"] > 20 and row["weekday"] < 5) else 0


def rush_hour(row):
    return (
        1
        if ((row["hour"] <= 20) and (row["hour"] >= 16) and (row["weekday"] < 5))
        else 0
    )


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    dt = dt.fillna(pd.Timestamp("2009-01-01", tz="UTC"))

    df["year"] = dt.dt.year.astype(np.int16)
    df["month"] = dt.dt.month.astype(np.int8)
    df["day"] = dt.dt.day.astype(np.int8)
    df["hour"] = dt.dt.hour.astype(np.int8)
    df["weekday"] = dt.dt.weekday.astype(np.int8)

    df["pickup_datetime"] = dt.astype(str)

    df["night"] = df.apply(lambda x: night(x), axis=1).astype(np.int8)
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1).astype(np.int8)
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1).astype(np.int8)
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
    df["distance"] = np.sqrt(np.abs(lon1 - lon2) ** 2 + np.abs(lat1 - lat2) ** 2)
    return df


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


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {
            id_column: raw_test[id_column].values,
            prediction_column: np.asarray(prediction).reshape(-1),
        }
    )
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows=", len(df))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 6))
    plt.plot(history.history.get("loss", []))
    plt.plot(history.history.get("val_loss", []))
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()

    if "rmse_metric" in history.history:
        plt.figure(figsize=(20, 6))
        plt.plot(history.history.get("rmse_metric", []))
        plt.plot(history.history.get("val_rmse_metric", []))
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper right")
        plt.show()




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

if RANDOM_SAMPLE_FROM_FULL_TRAIN:
    rng = np.random.RandomState(SAMPLE_RANDOM_SEED)
    chunksize = 200_000
    usecols = list(datatypes.keys())

    reservoir = None
    seen = 0  # total rows processed so far across all chunks
    for chunk in pd.read_csv(
        TRAIN_PATH,
        dtype=datatypes,
        usecols=usecols,
        chunksize=chunksize,
    ):
        chunk_len = len(chunk)
        if reservoir is None:
            reservoir = chunk.head(0).copy()

        for i in range(chunk_len):
            if len(reservoir) < DATASET_SIZE:
                reservoir = pd.concat(
                    [reservoir, chunk.iloc[[i]]], axis=0, ignore_index=True
                )
            else:
                j = rng.randint(0, seen + i + 1)
                if j < DATASET_SIZE:
                    reservoir.iloc[j] = chunk.iloc[i]

        seen += chunk_len

    if reservoir is None or len(reservoir) < DATASET_SIZE:
        raise RuntimeError(
            f"Sampling failed: collected {0 if reservoir is None else len(reservoir)} rows, expected {DATASET_SIZE}."
        )

    trainKaggle = reservoir.reset_index(drop=True)
else:
    trainKaggle = pd.read_csv(TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes)

testKaggle = pd.read_csv(
    TEST_PATH, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
)



## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]



## === cell 4
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## === cell 5
train_df.describe(include="all")



## === cell 6
test_df.describe(include="all")



## === cell 7
print("train_df clean")
train_df = clean(train_df)
print("test_df clean")
test_df = clean(test_df)



## === cell 8
train_df.describe()



## === cell 9
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 10
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 11
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## === cell 12
_ = train_df.iloc[:2000].plot.scatter("latdiff", "londiff")



## === cell 13
dropped_columns = ["pickup_datetime", "key"]  # keep passenger_count
train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(["pickup_datetime", "key"], axis=1)

print("Done with dropped_columns")



## === cell 14
train_df.shape



## === cell 15
train_df.describe()



## === cell 16
test_df.describe()



## === cell 17
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 18
train_labels = train_df["fare_amount"].values.astype("float32")
validation_labels = validation_df["fare_amount"].values.astype("float32")
test_labels = test_df["fare_amount"].values.astype("float32")

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 19
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)




## === cell 20
def rmse_metric(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))




## === cell 21
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
model.compile(
    loss="mean_squared_error", optimizer=adam, metrics=["mae", rmse_metric, "mse"]
)

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
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)



## === cell 22
plot_loss_accuracy_rmse(history)



## === cell 23
score = model.evaluate(train_df_scaled, train_labels, verbose=1)
print("Train metrics:", dict(zip(model.metrics_names, score)))



## === cell 24
score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print("Validation metrics:", dict(zip(model.metrics_names, score)))



## === cell 25
score = model.evaluate(test_scaled, test_labels, verbose=1)
print("Test metrics:", dict(zip(model.metrics_names, score)))



## === cell 26
validation_predictions = model.predict(validation_df_scaled, verbose=0).flatten()
plt.scatter(validation_labels, validation_predictions, s=5, alpha=0.3)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
lims = [
    min(validation_labels.min(), validation_predictions.min()),
    max(validation_labels.max(), validation_predictions.max()),
]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)



## === cell 27
test_predictions = model.predict(test_scaled, verbose=0).flatten()
plt.scatter(test_labels, test_predictions, s=5, alpha=0.3)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
lims = [
    min(test_labels.min(), test_predictions.min()),
    max(test_labels.max(), test_predictions.max()),
]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)



## === cell 28
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1).reshape(
    -1
)
predictionKaggle = np.maximum(predictionKaggle, 0.0)



## === cell 29
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
print(pd.read_csv(SUBMISSION_NAME).head())
print("Submission saved to:", os.path.abspath(SUBMISSION_NAME))
