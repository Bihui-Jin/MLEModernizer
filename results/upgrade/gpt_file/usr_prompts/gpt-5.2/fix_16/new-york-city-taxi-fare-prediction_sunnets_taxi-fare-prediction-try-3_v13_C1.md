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

4.15617

# 6. Current score

6.45089

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.84669) has done: 'I fix the immediate runtime/import issues caused by mixing `keras` (Keras 3) with incompatible backend/optimizer APIs, while keeping the same dense-network architecture and training loop. I also fix the NYC “water mask” cleaning step to work offline in Kaggle (no URL reads) by loading the mask locally if present and otherwise safely skipping that filter. Finally, I correct a couple of logic bugs in the time-based feature functions (conditions that were always true) and ensure the script always writes a valid `key,fare_amount` submission CSV at the end.'
- What this solution (achieved 6.5266) has done: 'I fix the crash in the first cell caused by an incompatible protobuf/TensorFlow import path by switching from `tf_keras` to the standard `tensorflow.keras` API available in Kaggle, while keeping the exact same model architecture, optimizer, and training loop. I also correct the train CSV reading bug where `fare_amount` was accidentally excluded via `usecols`, which prevents proper label handling and can silently harm training/score. Finally, I make paths robust to Kaggle’s `/kaggle/input/...` layout while preserving your filenames and ensure the submission CSV is always written with `key,fare_amount`. These changes are minimal, unblock end-to-end execution, and should improve RMSE by restoring proper supervision and stable Keras imports.'
- What this solution (achieved 6.1469) has done: 'I fix the TensorFlow/protobuf crash by forcing the Python protobuf implementation before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in many Kaggle TF/keras setups. Then I correct a couple of cleaning logic bugs where the “airport removal” filters were unintentionally removing far too many rows because they used `!=` comparisons on each coordinate separately instead of excluding only exact coordinate matches. Finally, I keep the same model and training loop, but increase effective training signal by safely using a larger training sample size (still manageable within the time limit) and ensure the submission is always written as a valid `key,fare_amount` CSV.'
- What this solution (achieved 6965.21806) has done: 'I fix the TensorFlow/protobuf crash by enforcing a compatible protobuf version before importing TensorFlow (Kaggle often ships protobuf>=4 which triggers `MessageFactory.GetPrototype` errors with TF on py3.7). Then I correct two data/feature issues that hurt RMSE without changing the model/training loop: (1) the “same long/lat” cleaning currently removes too many valid rides because it requires both lon and lat to be equal; it should remove rows only when both pickup and dropoff coordinates are identical, and (2) the time-feature functions have always-false/too-restrictive conditions (e.g., `night`) and are applied row-wise; I keep the same features but compute them vectorized and with correct logic. These are minimal, semantics-preserving fixes and should move RMSE down toward the target while keeping architecture/training unchanged, and the script still always write a valid `key,fare_amount` CSV.'
- What this solution (achieved 15.1502) has done: 'I make two minimal score-relevant fixes while keeping your model, loss, and training loop unchanged: (1) stop cleaning the real Kaggle test set so the submission keeps exactly the required 9914 keys in the original order (cleaning currently drops rows and yields an invalid/misaligned submission), and (2) remove the accidental “water mask inversion” bug so, when a local mask exists, it correctly filters out water points instead of keeping only water-to-water rides. These changes preserve your feature set and architecture but improve label/row alignment and data quality, which should move RMSE down toward your target while guaranteeing a valid `key,fare_amount` CSV.'
- What this solution (achieved 4.428676392854537e+26) has done: 'Your current RMSE (15.1502) is far worse than the target (4.15617), so we should make a small, score-relevant change that improves generalization without changing your model architecture or training loop. The biggest issue is that you’re evaluating/training on raw-dollar fares with MSE, which heavily over-weights large fares; a minimal, common fix for this competition is to train on `log1p(fare_amount)` and then invert with `expm1` at prediction time (same network, same loss, same loop). I keep all feature engineering and cleaning as-is, but apply the log transform only to the labels (and invert for holdout metric and submission), which typically drops RMSE substantially toward your target. I also clip negative/NaN predictions after inversion to ensure a valid submission.'
- What this solution (achieved 6.31829) has done: 'Your current RMSE is astronomically bad because the model is trained to predict `log1p(fare_amount)` but several input features can still become NaN/invalid due to `pickup_datetime` parsing failures (year/month/day/hour/weekday become NaN) and are then silently turned into zeros; this can lead to extreme miscalibration and exploding predictions after `expm1`. I make two minimal, score-relevant fixes while preserving your architecture/training loop: (1) ensure time features are always finite by filling datetime-derived NaNs with safe defaults and keeping them numeric, and (2) add a simple upper clip on the final fare predictions (e.g., 0–500) to prevent a few extreme outputs from destroying RMSE. These changes don’t alter the model design or loss, but they directly address the runaway `expm1` issue that produces absurd errors and should move RMSE down dramatically toward your target band. The script still run end-to-end and write a valid `key,fare_amount` submission CSV with the same paths/names.'
- What this solution (achieved 5.82936) has done: 'Your current RMSE (6.318) is worse than the target (4.156), so we should make small, score-relevant fixes that improve generalization without changing your model architecture or training loop. The biggest leak in performance is that the Kaggle test set is *not cleaned at all* while the training data is heavily cleaned, causing a train/test distribution mismatch; I apply the same non-label-dependent cleaning to `testKaggle` but in a way that preserves the exact original row order and count by predicting only on the “kept” rows and filling dropped rows with a safe baseline. To further reduce RMSE without changing the model, I also fix scaling leakage by fitting the scaler on `train_df` only (not including validation), and use the model’s training-label mean as the baseline prediction for any filtered/invalid rows. These are minimal changes, keep your core logic intact (same features, same network, same loss/training loop), and should move RMSE down toward your target.'
- What this solution (achieved 6.45089) has done: 'You’re currently worse than the target (RMSE 5.829 vs 4.156, lower is better), so the smallest safe move is to reduce train/test mismatch and stabilize the target scale without changing the network, loss, or training loop. I (1) apply the same non-label-dependent cleaning to the holdout `test_df` using your existing keep-mask pattern so the holdout distribution matches training/validation, and (2) consistently train/evaluate/predict in the same target space by using `log1p` for the holdout labels too (then invert with `expm1` only for RMSE and submission). These changes keep the same architecture, optimizer, epochs, and features, but remove a major inconsistency that typically harms RMSE. Submission writing remains identical and still preserves Kaggle test row order/count.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 4:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<4"]
            )
            import importlib

            importlib.invalidate_caches()
    except Exception as e:
        print("Warning: protobuf compatibility step encountered:", repr(e))


_ensure_protobuf_compatible()

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers


def _resolve_kaggle_path(filename):
    candidates = [
        os.path.join("..", "input", filename),
        os.path.join("/kaggle/input", filename),
        os.path.join("/kaggle/input/new-york-city-taxi-fare-prediction", filename),
        os.path.join("/kaggle/data", filename),
        os.path.join("/kaggle/data/new-york-city-taxi-fare-prediction", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


TRAIN_PATH = _resolve_kaggle_path("train.csv")
TEST_PATH = _resolve_kaggle_path("test.csv")
SUBMISSION_NAME = "submissiontry_water.csv"
SUBMISSION_NAME_KAGGLE = "submission.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001

DATASET_SIZE = 1000000

ACTIVITY_L1 = 0.001

np.random.seed(1)
tf.random.set_seed(1)

print("Using TRAIN_PATH:", TRAIN_PATH)
print("Using TEST_PATH:", TEST_PATH)
print("DATASET_SIZE:", DATASET_SIZE)
print("ACTIVITY_L1:", ACTIVITY_L1)



## === cell 1
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



## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.20, random_state=1)



## === cell 3
testKaggle.head()



## === cell 4
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 5
train_df.head()



## === cell 6
test_df.head()



## === cell 7
validation_df.head()




## === cell 8
def clean(df):
    has_fare = "fare_amount" in df.columns

    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    df = df[
        ~(
            (df["dropoff_longitude"] == df["pickup_longitude"])
            & (df["dropoff_latitude"] == df["pickup_latitude"])
        )
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

    if has_fare:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))
    else:
        print(
            " New size after removing outliers: %d (skipped; no fare_amount column)"
            % len(df)
        )

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)  # Statue of Liberty

    df = df[
        ~(
            (df["pickup_longitude"] == nyc_coord[1])
            & (df["pickup_latitude"] == nyc_coord[0])
        )
    ]
    df = df[
        ~(
            (df["dropoff_longitude"] == nyc_coord[1])
            & (df["dropoff_latitude"] == nyc_coord[0])
        )
    ]
    print(" New size after NY airport: %d" % len(df))

    df = df[
        ~(
            (df["pickup_longitude"] == fk_coord[1])
            & (df["pickup_latitude"] == fk_coord[0])
        )
    ]
    df = df[
        ~(
            (df["dropoff_longitude"] == fk_coord[1])
            & (df["dropoff_latitude"] == fk_coord[0])
        )
    ]
    print(" New size after jfk airport: %d" % len(df))

    df = df[
        ~(
            (df["pickup_longitude"] == ewr_coord[1])
            & (df["pickup_latitude"] == ewr_coord[0])
        )
    ]
    df = df[
        ~(
            (df["dropoff_longitude"] == ewr_coord[1])
            & (df["dropoff_latitude"] == ewr_coord[0])
        )
    ]
    print(" New size after ewr airport: %d" % len(df))

    df = df[
        ~(
            (df["pickup_longitude"] == lga_coord[1])
            & (df["pickup_latitude"] == lga_coord[0])
        )
    ]
    df = df[
        ~(
            (df["dropoff_longitude"] == lga_coord[1])
            & (df["dropoff_latitude"] == lga_coord[0])
        )
    ]
    print(" New size after lgr airport: %d" % len(df))

    df = df[
        ~(
            (df["pickup_longitude"] == sol_coord[1])
            & (df["pickup_latitude"] == sol_coord[0])
        )
    ]
    df = df[
        ~(
            (df["dropoff_longitude"] == sol_coord[1])
            & (df["dropoff_latitude"] == sol_coord[0])
        )
    ]
    print(" New size after sol removed: %d" % len(df))

    print("Old size: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def remove_datapoints_from_water(df):
    """
    Load a local nyc mask file if present; otherwise, skip this filter.

    Keep land-only points when mask exists: drop rows where either endpoint is water.
    """

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)
    mask_filename = "nyc_mask-74.5_-72.8_40.5_41.8.png"

    possible_paths = [
        mask_filename,
        os.path.join("/kaggle/input", mask_filename),
        os.path.join("/kaggle/working", mask_filename),
        os.path.join("/kaggle/input/new-york-city-taxi-fare-prediction", mask_filename),
    ]
    mask_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if mask_path is None:
        return df

    nyc_mask = plt.imread(mask_path)
    if nyc_mask.ndim == 3:
        nyc_mask = nyc_mask[:, :, 0]
    nyc_mask = nyc_mask > 0.9  # True means "water"

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

    pickup_is_water = nyc_mask[pickup_y, pickup_x]
    dropoff_is_water = nyc_mask[dropoff_y, dropoff_x]

    keep_idx = ~(pickup_is_water | dropoff_is_water)
    return df[keep_idx]


def late_night(row):
    return 1 if (row["hour"] <= 3 or row["hour"] >= 20) else 0


def night(row):
    return (
        1 if ((row["weekday"] < 5) and (row["hour"] <= 6 or row["hour"] >= 20)) else 0
    )


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

    df["year"] = dt.dt.year.astype("float32")
    df["month"] = dt.dt.month.astype("float32")
    df["day"] = dt.dt.day.astype("float32")
    df["hour"] = dt.dt.hour.astype("float32")
    df["weekday"] = dt.dt.weekday.astype("float32")

    df["year"] = df["year"].fillna(2010.0).astype("float32")
    df["month"] = df["month"].fillna(1.0).astype("float32")
    df["day"] = df["day"].fillna(1.0).astype("float32")
    df["hour"] = df["hour"].fillna(0.0).astype("float32")
    df["weekday"] = df["weekday"].fillna(0.0).astype("float32")

    df["pickup_datetime"] = dt.astype(str)

    hour = df["hour"].astype("int16")
    weekday = df["weekday"].astype("int16")

    df["night"] = (((weekday < 5) & ((hour <= 6) | (hour >= 20)))).astype("uint8")
    df["late_night"] = (((hour <= 3) | (hour >= 20))).astype("uint8")
    df["rush_hour"] = (((weekday < 5) & (hour >= 16) & (hour <= 20))).astype("uint8")
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["latdiff"] = lat1 - lat2
    df["londiff"] = lon1 - lon2
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = np.sqrt(np.abs(lon1 - lon2) ** 2 + np.abs(lat1 - lat2) ** 2)
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {id_column: raw_test[id_column].values, prediction_column: prediction}
    )
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))
    print("Columns:", list(df.columns))


def plot_loss_accuracy(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()


def clean_keep_mask(df):
    """
    Return a boolean mask (len(df)) indicating which rows would survive clean(df),
    but without dropping rows (so we can preserve submission alignment).
    """
    mask = pd.Series(True, index=df.index)

    mask &= df.notna().all(axis=1)

    mask &= ~(
        (df["dropoff_longitude"] == df["pickup_longitude"])
        & (df["dropoff_latitude"] == df["pickup_latitude"])
    )

    mask &= (
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    )

    MinMax = (-74.5, -72.8, 40.5, 41.8)
    mask &= (MinMax[0] <= df["pickup_longitude"]) & (
        df["pickup_longitude"] <= MinMax[1]
    )
    mask &= (MinMax[0] <= df["dropoff_longitude"]) & (
        df["dropoff_longitude"] <= MinMax[1]
    )
    mask &= (MinMax[2] <= df["pickup_latitude"]) & (df["pickup_latitude"] <= MinMax[3])
    mask &= (MinMax[2] <= df["dropoff_latitude"]) & (
        df["dropoff_latitude"] <= MinMax[3]
    )

    mask &= (df["passenger_count"] > 0) & (df["passenger_count"] <= 6)

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)

    for coord in [nyc_coord, fk_coord, ewr_coord, lga_coord, sol_coord]:
        mask &= ~(
            (df["pickup_longitude"] == coord[1]) & (df["pickup_latitude"] == coord[0])
        )
        mask &= ~(
            (df["dropoff_longitude"] == coord[1]) & (df["dropoff_latitude"] == coord[0])
        )

    BB = (-74.5, -72.8, 40.5, 41.8)
    mask_filename = "nyc_mask-74.5_-72.8_40.5_41.8.png"
    possible_paths = [
        mask_filename,
        os.path.join("/kaggle/input", mask_filename),
        os.path.join("/kaggle/working", mask_filename),
        os.path.join("/kaggle/input/new-york-city-taxi-fare-prediction", mask_filename),
    ]
    mask_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if mask_path is not None:
        nyc_mask = plt.imread(mask_path)
        if nyc_mask.ndim == 3:
            nyc_mask = nyc_mask[:, :, 0]
        nyc_mask = nyc_mask > 0.9  # True means "water"

        def lonlat_to_xy(longitude, latitude, dx, dy, BB_):
            return (dx * (longitude - BB_[0]) / (BB_[1] - BB_[0])).astype("int"), (
                dy - dy * (latitude - BB_[2]) / (BB_[3] - BB_[2])
            ).astype("int")

        pickup_x, pickup_y = lonlat_to_xy(
            df["pickup_longitude"],
            df["pickup_latitude"],
            nyc_mask.shape[1],
            nyc_mask.shape[0],
            BB,
        )
        dropoff_x, dropoff_y = lonlat_to_xy(
            df["dropoff_longitude"],
            df["dropoff_latitude"],
            nyc_mask.shape[1],
            nyc_mask.shape[0],
            BB,
        )

        pickup_x = np.clip(pickup_x, 0, nyc_mask.shape[1] - 1)
        dropoff_x = np.clip(dropoff_x, 0, nyc_mask.shape[1] - 1)
        pickup_y = np.clip(pickup_y, 0, nyc_mask.shape[0] - 1)
        dropoff_y = np.clip(dropoff_y, 0, nyc_mask.shape[0] - 1)

        pickup_is_water = nyc_mask[pickup_y, pickup_x]
        dropoff_is_water = nyc_mask[dropoff_y, dropoff_x]
        mask &= ~(pickup_is_water | dropoff_is_water)

    return mask.values.astype(bool)




## === cell 9
validation_df.head()



## === cell 10
print("train_df clean")
train_df = clean(train_df).reset_index(drop=True)
print("validation_df clean")
validation_df = clean(validation_df).reset_index(drop=True)

print("test_df clean: applying keep-mask cleaning (preserves holdout alignment)")
test_df_keep_mask = clean_keep_mask(test_df)
print("test_df keep rows:", int(test_df_keep_mask.sum()), "out of", len(test_df))

print("testKaggle clean: applying keep-mask cleaning (preserves alignment)")
testKaggle_keep_mask = clean_keep_mask(testKaggle)
print(
    "testKaggle keep rows:", int(testKaggle_keep_mask.sum()), "out of", len(testKaggle)
)

train_df.describe()

print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)

print("test_df add_time_features")
test_df = add_time_features(test_df)

print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 11
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 12
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## === cell 13
dropped_columns = ["passenger_count", "pickup_datetime"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")



## === cell 14
train_labels = np.log1p(train_df["fare_amount"].values.astype("float64"))
validation_labels = np.log1p(validation_df["fare_amount"].values.astype("float64"))
test_labels_log = np.log1p(test_df["fare_amount"].values.astype("float64"))
test_labels = test_df["fare_amount"].values.astype(
    "float64"
)  # kept for RMSE in $ space

train_df = train_df.drop(["fare_amount", "key"], axis=1)
validation_df = validation_df.drop(["fare_amount", "key"], axis=1)
test_df = test_df.drop(["fare_amount", "key"], axis=1)

print(
    "Done with Labels (log1p on train/validation/holdout; expm1 only for metrics/submission)"
)



## === cell 15
train_df.shape



## === cell 16
test_df.shape



## === cell 17
validation_df.shape




## === cell 18
def _finite_row_mask(X_df):
    return np.isfinite(X_df.to_numpy(dtype="float64")).all(axis=1)


def _sanitize_numeric_df(df):
    df2 = df.copy()
    for c in df2.columns:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df2 = df2.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df2


train_df = _sanitize_numeric_df(train_df)
validation_df = _sanitize_numeric_df(validation_df)
test_df = _sanitize_numeric_df(test_df)
testKaggle_clean = _sanitize_numeric_df(testKaggle_clean)

holdout_mask = test_df_keep_mask & _finite_row_mask(test_df)
print(
    "Holdout rows total:",
    len(test_df),
    "kept+finite-feature rows:",
    int(holdout_mask.sum()),
)

scaler = preprocessing.MinMaxScaler()
scaler.fit(train_df)

train_df_scaled = scaler.transform(train_df)
validation_df_scaled = scaler.transform(validation_df)

test_scaled = scaler.transform(test_df.loc[holdout_mask])
testKaggle_scaled_kept = scaler.transform(testKaggle_clean.loc[testKaggle_keep_mask])


def rmse_np(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))




## === cell 19
model = Sequential()
model.add(
    Dense(
        256,
        activation="relu",
        input_dim=train_df_scaled.shape[1],
        activity_regularizer=regularizers.l1(ACTIVITY_L1),
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
model.compile(loss="mean_squared_error", optimizer=adam, metrics=["mae"])

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



## === cell 20
plot_loss_accuracy(history)



## === cell 21
prediction_holdout_log = model.predict(test_scaled, batch_size=128, verbose=1)



## === cell 22
test_labels.shape



## === cell 23
predictionKaggle_log_kept = model.predict(
    testKaggle_scaled_kept, batch_size=128, verbose=1
)



## === cell 24
MAX_FARE_CAP = 500.0

prediction_holdout = np.expm1(np.asarray(prediction_holdout_log).reshape(-1))
prediction_holdout = np.where(np.isfinite(prediction_holdout), prediction_holdout, 0.0)
prediction_holdout = np.clip(prediction_holdout, 0.0, MAX_FARE_CAP)

metric = rmse_np(test_labels[holdout_mask], prediction_holdout)



## === cell 25
print(metric)



## === cell 26
baseline_fare = float(np.expm1(np.mean(train_labels)))
baseline_fare = float(np.clip(baseline_fare, 0.0, MAX_FARE_CAP))
print("Baseline fare used for filtered Kaggle test rows:", baseline_fare)

predictionKaggle_kept = np.expm1(np.asarray(predictionKaggle_log_kept).reshape(-1))
predictionKaggle_kept = np.where(
    np.isfinite(predictionKaggle_kept), predictionKaggle_kept, baseline_fare
)
predictionKaggle_kept = np.clip(predictionKaggle_kept, 0.0, MAX_FARE_CAP)

predictionKaggle = np.full(
    shape=(len(testKaggle),), fill_value=baseline_fare, dtype="float64"
)
predictionKaggle[testKaggle_keep_mask] = predictionKaggle_kept

assert len(predictionKaggle) == len(testKaggle), (
    "Prediction length must match test rows for a valid submission. "
    f"pred={len(predictionKaggle)} test={len(testKaggle)}"
)

output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
output_submission(
    testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME_KAGGLE
)



## === cell 27
print(predictionKaggle[:10])
