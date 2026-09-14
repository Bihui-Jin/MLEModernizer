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

4.93166

# 6. Current score

5.49725

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 69.5726) has done: 'I (1) fix the environment/runtime import error by switching to `tf_keras` (compatible with the Kaggle TensorFlow stack) while keeping the same model architecture and training loop. I (2) fix the NYC water-mask cleaning step by loading the mask via `urllib`+PIL (no internet assumption: falls back gracefully if unavailable) so cleaning no longer crashes. I (3) correct the optimizer API (`Adam(learning_rate=...)`) and remove/disable the model-plot cell that imports a missing module. Finally, I (4) fix the datetime parsing format and the buggy night/late_night logic (currently always true), which is a minimal feature-correctness change expected to improve RMSE toward your target without changing the core approach.'
- What this solution (achieved 56.6639) has done: 'I fix the immediate runtime/import failure by avoiding the protobuf “MessageFactory.GetPrototype” crash that happens when importing `tf_keras` in this Kaggle image, switching to `tensorflow.keras` while keeping the exact same model architecture, optimizer, and training loop. I also correct the file paths to the actual provided dataset location (`/kaggle/input/...`) so the code reliably finds train/test on Kaggle. Finally, I keep the feature engineering and cleaning logic intact, but add a tiny safety guard so the water-mask step defaults off cleanly without breaking, and ensure the submission is always written as a valid `.csv` with the required columns.'
- What this solution (achieved 164.27002) has done: 'I fix the runtime crash caused by an incompatible protobuf version when importing TensorFlow by forcing the pure‑Python protobuf implementation before TensorFlow is imported; this keeps your model/training logic unchanged but makes the notebook runnable end‑to‑end. I also fix the broken debug MSE line by using a compatible MSE function (so it no longer errors) without affecting training or submission outputs. Finally, I correct the Kaggle input file paths to the actual dataset location you listed (`/kaggle/input/new-york-city-taxi-fare-prediction/...`) so the code always finds the CSVs, and ensure the submission is written as a valid `.csv` with `key,fare_amount`.'
- What this solution (achieved 42.82831) has done: 'I fix the runtime crash happening at `import tensorflow` by pinning the protobuf Python implementation and forcing a compatible protobuf version before TensorFlow loads, which is the root cause of the `MessageFactory.GetPrototype` error in this Kaggle image. I also fix a data-loading logic bug where `fare_amount` wasn’t actually read (because `usecols` excluded it), which makes training labels wrong and can severely harm RMSE. Finally, I keep the model/training loop intact, but ensure test/train columns align deterministically and the submission is always written as `key,fare_amount` to a `.csv` file.'
- What this solution (achieved 943.37285) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by avoiding the fragile runtime protobuf downgrade and forcing the pure‑Python protobuf implementation *before* importing TensorFlow, which is the minimal change that unblocks execution in this Kaggle image. I also fix a major data/label mismatch: your training split is made from `trainKaggle` that currently drops the `key` column, while your test set keeps it; this causes feature columns to misalign and can badly inflate RMSE—so I include `key` in the train read (without using it as a feature) to keep schemas consistent. Finally, I make `clean()` safe for test/validation frames by only applying the `fare_amount` outlier filter when that column exists, preventing accidental KeyErrors and keeping inference stable, while leaving the model architecture/training loop unchanged.'
- What this solution (achieved 104.29584) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by switching the environment flag to use the C++ protobuf implementation (the current forced pure-Python implementation triggers this failure in Kaggle’s TF build). I also correct a critical data-loading bug where `test.csv` is read with a dtype dict that includes `fare_amount` (a non-existent column in test), which can silently mis-parse or fail depending on pandas version and leads to broken inference/score. These two changes unblock execution end-to-end and restore proper train/test schema handling without changing the model architecture, training loop, or feature engineering semantics. The script still write a valid `key,fare_amount` submission CSV to the same filename.'
- What this solution (achieved 5.63939) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure‑Python protobuf implementation *before* importing TensorFlow, which is the most reliable way to run in this Kaggle image. Then I fix a major logic bug that is inflating RMSE: the label (`fare_amount`) is being dropped before it’s extracted in cell 18, causing either a KeyError or incorrect training labels depending on execution state; I extract labels first, then drop columns. Finally, I keep your model/training loop and feature engineering intact, but ensure the submission is always written as a valid `.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 6.31696) has done: 'I fix the hard runtime failure at TensorFlow import by switching the protobuf implementation from the unavailable C++ extension (`cpp`, which triggers the `_message` import error) to the pure-Python implementation before importing TensorFlow. This unblocks the entire notebook so later cells can run and produce a valid `submissiontry_water.csv` file. I also keep all existing feature engineering, cleaning, model architecture, and training loop intact, only adding small safety guards (e.g., ensure required columns exist after cleaning) so the pipeline completes reliably. Finally, I ensure the submission is always written with the required `key,fare_amount` columns and a `.csv` suffix.'
- What this solution (achieved 5.49725) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by ensuring a protobuf version compatible with the Kaggle TF build is installed at runtime and by not forcing the pure‑Python protobuf implementation (which is what triggers this crash here). This is a minimal environment/runtime fix that preserves your model, training loop, and feature engineering unchanged, and it let the notebook run end-to-end and write a valid `submissiontry_water.csv`. I also keep the existing deterministic seeds and paths intact, and add only a small, safe fallback so the run does not fail if the pip install cannot execute (in which case the original error would still surface). No score-tuning changes are introduced beyond restoring correct execution, so score changes should come only from the pipeline actually running reliably.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None

    target = "protobuf==3.20.3"
    try:
        if pb_ver is None or tuple(int(x) for x in pb_ver.split(".")[:2]) >= (4, 21):
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", target]
            )
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf"):
                    del sys.modules[m]
    except Exception as e:
        print(
            "Warning: could not ensure protobuf compatibility via pip. Reason:", repr(e)
        )


_ensure_protobuf_compatible()

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import backend

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

np.random.seed(1)
tf.random.set_seed(1)

print("Using TensorFlow:", tf.__version__)
print("Train path exists:", os.path.exists(TRAIN_PATH))
print("Test path exists:", os.path.exists(TEST_PATH))



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

usecols = [
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
    TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=usecols
)

test_dtypes = {k: v for k, v in datatypes.items() if k != "fare_amount"}
testKaggle = pd.read_csv(TEST_PATH, dtype=test_dtypes)

print("Loaded trainKaggle columns:", list(trainKaggle.columns))
print("Loaded testKaggle columns:", list(testKaggle.columns))



## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)



## === cell 3
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 4
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("validation_df Size %d" % len(validation_df))
print("test_df Size %d" % len(test_df))



## === cell 5
train_df.describe()



## === cell 6
validation_df.describe()



## === cell 7
test_df.describe()




## === cell 8
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
    print(" New size after only NYC: %d" % len(df))

    if "fare_amount" in df.columns:
        df = df[(0.99 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))
    else:
        print(" Skipping fare_amount outlier filter (no fare_amount column).")

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)  # Statue of Liberty

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


def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)
    url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"

    try:
        import urllib.request
        from PIL import Image

        with urllib.request.urlopen(url, timeout=10) as resp:
            img = Image.open(resp)
            nyc_mask = (np.array(img)[:, :, 0] / 255.0) > 0.9

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
    except Exception as e:
        print("Warning: could not apply water-mask filter (skipping). Reason:", repr(e))
        return df


def late_night(row):
    h = row["hour"]
    return 1 if (h <= 3) else 0


def night(row):
    h = row["hour"]
    wd = row["weekday"]
    return 1 if ((h > 20) or (h < 6)) and (wd < 5) else 0


def rush_hour(row):
    h = row["hour"]
    wd = row["weekday"]
    return 1 if (16 <= h <= 20) and (wd < 5) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["year"] = dt.dt.year.astype("int16")
    df["month"] = dt.dt.month.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["weekday"] = dt.dt.weekday.astype("int8")
    df["pickup_datetime"] = dt.astype(str)

    df["night"] = df.apply(lambda x: night(x), axis=1).astype("int8")
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1).astype("int8")
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1).astype("int8")
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


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    pred = np.asarray(prediction).reshape(-1)
    df = pd.DataFrame({id_column: raw_test[id_column].values, prediction_column: pred})
    if not file_name.lower().endswith(".csv"):
        file_name = file_name + ".csv"
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))
    print(df.head())


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()

    if "accuracy" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["accuracy"])
        plt.plot(history.history.get("val_accuracy", []))
        plt.title("Model accuracy (not meaningful for regression)")
        plt.ylabel("Accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper right")
        plt.show()

    if "rmse" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["rmse"])
        plt.plot(history.history.get("val_rmse", []))
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper right")
        plt.show()




## === cell 9
print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)

print("test_df clean")
test_df = clean(test_df)

testKaggle_raw = testKaggle.copy()
print("testKaggle clean")
testKaggle_clean_rows = clean(testKaggle)

if len(testKaggle_clean_rows) == 0:
    print(
        "Warning: testKaggle_clean_rows is empty after cleaning; falling back to raw testKaggle."
    )
    testKaggle_clean_rows = testKaggle.copy()



## === cell 10
train_df.describe()



## === cell 11
validation_df.describe()



## === cell 12
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle_clean_rows add_time_features")
testKaggle_clean_rows = add_time_features(testKaggle_clean_rows)



## === cell 13
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle_clean_rows add_coordinate_features")
testKaggle_clean_rows = add_coordinate_features(testKaggle_clean_rows)



## === cell 14
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle_clean_rows add_distances_features")
testKaggle_clean_rows = add_distances_features(testKaggle_clean_rows)
print("Done with Adding features")



## === cell 15
train_df.describe()



## === cell 16
validation_df.describe()



## === cell 17
dropped_columns = [
    "passenger_count",
    "pickup_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "key",  # ensure key is never used as a feature
]

train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(dropped_columns + ["fare_amount"], axis=1)
validation_df = validation_df.drop(dropped_columns + ["fare_amount"], axis=1)
test_df = test_df.drop(dropped_columns + ["fare_amount"], axis=1)

testKaggle_clean = testKaggle_clean_rows.drop(dropped_columns, axis=1)

print("Done with dropped_columns + Labels")



## === cell 18
train_df.shape



## === cell 19
test_df.shape



## === cell 20
validation_df.shape



## === cell 21
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)




## === cell 22
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))




## === cell 23
model = Sequential()
model.add(
    Dense(
        512,
        activation="relu",
        input_dim=train_df_scaled.shape[1],
        activity_regularizer=regularizers.l1(0.01),
    )
)
model.add(BatchNormalization())
model.add(Dense(216, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(32, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(16, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(8, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    loss="mean_squared_error", optimizer=adam, metrics=["mae", "accuracy", rmse]
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



## === cell 24
print(
    "Model visualization skipped (keras.utils.vis_utils not available in this environment)."
)



## === cell 25
plot_loss_accuracy_rmse(history)



## === cell 26
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle_clean = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## === cell 27
mse_debug = tf.reduce_mean(
    tf.math.squared_difference(
        tf.convert_to_tensor(test_labels[:1000], dtype=tf.float32),
        tf.convert_to_tensor(prediction[:1000].reshape(-1), dtype=tf.float32),
    )
).numpy()
print("Debug MSE (first 1000):", float(mse_debug))



## === cell 28
pred_clean_series = pd.Series(
    predictionKaggle_clean.reshape(-1), index=testKaggle_clean_rows["key"].values
)
fill_value = float(pred_clean_series.mean()) if len(pred_clean_series) else 11.35

pred_full = testKaggle_raw[["key"]].copy()
pred_full["fare_amount"] = (
    pred_full["key"].map(pred_clean_series).fillna(fill_value).astype("float32")
)

if not SUBMISSION_NAME.lower().endswith(".csv"):
    SUBMISSION_NAME = SUBMISSION_NAME + ".csv"
pred_full.to_csv(SUBMISSION_NAME, index=False)
print("Output complete:", SUBMISSION_NAME, "rows:", len(pred_full))
print(pred_full.head())



## === cell 29
print(prediction[:5].reshape(-1))
print(test_labels[:5])
