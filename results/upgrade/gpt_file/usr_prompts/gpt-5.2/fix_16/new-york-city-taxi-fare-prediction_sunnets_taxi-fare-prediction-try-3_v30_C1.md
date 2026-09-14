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

4.20589

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.21055) has done: 'I fix the runtime/import issues caused by using Keras 3 (optimizer API changes and removed visualization utility), and make the NYC “water mask” cleaning step work offline (no URL fetch) by falling back safely when the mask can’t be loaded. I also correct the datetime parsing format so it parses the provided strings reliably, and fix a couple of feature-creation logic bugs (night/late_night conditions) that currently add noisy/incorrect signals and hurt RMSE. Finally, I ensure the script reads from the actual Kaggle input path you have (`/kaggle/input/...`) and always writes a valid `key,fare_amount` submission CSV.'
- What this solution (achieved 45.84444) has done: 'I fix the two runtime blockers preventing training: the protobuf `MessageFactory.GetPrototype` crash caused by importing `keras` (Keras 3) in this Python 3.7 Kaggle image, and the custom `rmse()` metric crash because `keras.backend` in Keras 3 no longer exposes `sqrt/mean/square` as used here. To keep your model architecture/training loop intact, I switch the imports to `tf_keras` (available in your environment) which matches the older Keras API your code expects, and I implement `rmse` via `tf.math` so it works reliably. I also ensure the input paths resolve to the provided `/kaggle/input/new-york-city-taxi-fare-prediction/...` files (your current `/kaggle/input/train.csv` path doesn’t exist in this dataset layout) and that the script always writes a valid `submissiontry_water.csv` with `key,fare_amount`.'
- What this solution (achieved 12.79023) has done: 'I fix the runtime crash happening before any training by preventing the protobuf/Keras import conflict that triggers `MessageFactory.GetPrototype` under this Python 3.7 image. To keep your model/training logic intact, I switch TensorFlow/Keras imports to the stable `tf.compat.v1` graph mode configuration and disable eager/protobuf fast paths that commonly cause this exact error, while still using `tf_keras` as you intended. I also correct one data-reading bug that currently drops the `key` column from the training sample via positional `usecols`, which can silently misalign columns depending on CSV parsing; this is score-relevant because it can shift features/labels. Finally, I ensure the submission is always written as a valid `key,fare_amount` CSV with the expected row count.'
- What this solution (achieved 50.10199) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by avoiding the incompatible `tensorflow` import path in this Python 3.7 Kaggle image and instead using the stable `tf_keras` backend (`tensorflow.compat.v2`) with safe environment flags set before any TF-related import. I also fix a score-relevant data bug where the `key` column is not loaded for train/test (needed for correct submission alignment) and ensure the test set used for local evaluation is cleaned the same way as training/validation to avoid a distribution mismatch that hurts RMSE. Finally, I keep your model architecture/training loop intact and ensure a valid `key,fare_amount` CSV is always written to `submissiontry_water.csv`.'
- What this solution (achieved 8.99015) has done: 'I fix the runtime crash in the first cell caused by the protobuf/TensorFlow import conflict by forcing the pure-Python protobuf implementation *before* any TensorFlow-related import and by avoiding `tensorflow.compat.v2` in favor of a safer `tensorflow` import path that works with `tf_keras` in this Kaggle image. I also remove the `accuracy` metric (which is classification-only and destabilizes training logs/gradients for regression) while keeping the same model, loss, data, and training loop; this is a minimal, score-relevant calibration fix that should move RMSE down toward your target. Finally, I ensure the submission is always written with the required `key,fare_amount` columns and a `.csv` suffix to `/kaggle/working/`.'
- What this solution (achieved 5.89651) has done: 'I fix the runtime crash happening in the first cell (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by ensuring TensorFlow uses the pure-Python protobuf implementation and by importing `tf_keras` (and `tensorflow`) in a safer order with a small compatibility shim for newer `protobuf` versions. This is a correctness/stability fix that unblocks training/inference without changing your model architecture, loss, or feature pipeline. I also keep your existing submission-writing logic intact so it always produces a valid `/kaggle/working/submissiontry_water.csv` with `key,fare_amount`. No score-tuning changes are introduced beyond making the pipeline run reliably end-to-end.'
- What this solution (achieved 6.09271) has done: 'I make two minimal, score-relevant corrections that commonly cause inflated RMSE in this competition without changing your model/training loop: (1) fix the “remove identical pickup/dropoff” filter to remove rows where *both* coordinates are identical (your current `&` removes too many valid short trips), and (2) handle `pickup_datetime` parsing consistently by using `utc=True` then converting to naive time to avoid timezone/coercion artifacts that add noise to time features. These changes should improve generalization and move RMSE down toward your target while keeping everything else (features, scaling, architecture, loss, epochs, batch size) the same. The script still run end-to-end and write `/kaggle/working/submissiontry_water.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf

try:
    tf.config.experimental.enable_tensor_float_32_execution(False)
except Exception:
    pass

from tf_keras.models import Sequential
from tf_keras.layers import Dense, BatchNormalization
from tf_keras import optimizers, regularizers

np.random.seed(1)
tf.random.set_seed(1)

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001

DATASET_SIZE = 250000

USE_LOG_TARGET = False

if not os.path.exists(TRAIN_PATH):
    alt = "/kaggle/input/train.csv"
    if os.path.exists(alt):
        TRAIN_PATH = alt
    else:
        TRAIN_PATH = "../input/train.csv"

if not os.path.exists(TEST_PATH):
    alt = "/kaggle/input/test.csv"
    if os.path.exists(alt):
        TEST_PATH = alt
    else:
        TEST_PATH = "../input/test.csv"

print("TRAIN_PATH:", TRAIN_PATH)
print("TEST_PATH:", TEST_PATH)

if not SUBMISSION_NAME.lower().endswith(".csv"):
    SUBMISSION_NAME = SUBMISSION_NAME + ".csv"
SUBMISSION_PATH = os.path.join("/kaggle/working", SUBMISSION_NAME)
print("SUBMISSION_PATH:", SUBMISSION_PATH)



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
    dtype={
        "key": "str",
        "pickup_datetime": "str",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
)



## === cell 2
train_df, validation_df = train_test_split(trainKaggle, test_size=0.10, random_state=1)



## === cell 3
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("validation_df Size %d" % len(validation_df))



## === cell 4
train_df.describe()



## === cell 5
validation_df.describe()




## === cell 6
def remove_datapoints_from_water(df):
    """
    Score-relevant improvement (minimal): if the NYC water mask image is not available offline,
    fall back to a deterministic heuristic that removes points very likely to be in water/harbor
    while keeping core logic intact. This avoids the previous no-op that left noisy coastal/water points,
    which can inflate RMSE.
    """

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        x = (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int")
        y = (dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])).astype("int")
        return x, y

    BB = (-74.5, -72.8, 40.5, 41.8)

    local_mask_paths = [
        "/kaggle/input/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/working/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "nyc_mask-74.5_-72.8_40.5_41.8.png",
    ]

    mask_path = None
    for p in local_mask_paths:
        if os.path.exists(p):
            mask_path = p
            break

    if mask_path is None:
        lat_p = df["pickup_latitude"]
        lat_d = df["dropoff_latitude"]
        lon_p = df["pickup_longitude"]
        lon_d = df["dropoff_longitude"]

        waterish = (
            (lat_p < 40.60) | (lat_d < 40.60) | (lon_p > -73.65) | (lon_d > -73.65)
        )
        return df[~waterish]

    nyc_mask = plt.imread(mask_path)
    if nyc_mask.ndim == 3:
        nyc_mask = nyc_mask[:, :, 0]
    nyc_mask = nyc_mask > 0.9

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

    in_water = nyc_mask[pickup_y, pickup_x] | nyc_mask[dropoff_y, dropoff_x]
    return df[~in_water]


def _near_point_mask(lon, lat, center_lon, center_lat, radius_deg):
    return ((lon - center_lon) ** 2 + (lat - center_lat) ** 2) <= (radius_deg**2)


def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    df = df[
        ~(
            (df["dropoff_longitude"] == df["pickup_longitude"])
            & (df["dropoff_latitude"] == df["pickup_latitude"])
        )
    ]
    print(" New size after removing same pickup/dropoff point: %d" % len(df))

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

    print(" Skipping landmark point-noise removal (was overly aggressive).")

    print("Old size: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)

    if "fare_amount" in df.columns:
        valid = dt.notna()
        if valid.sum() != len(df):
            df = df.loc[valid].copy()
            dt = dt.loc[valid]
    else:
        dt = dt.fillna(pd.Timestamp("2010-01-01", tz="UTC"))

    dt = dt.dt.tz_convert(None)

    df["year"] = dt.dt.year.astype("float32")
    df["month"] = dt.dt.month.astype("float32")
    df["day"] = dt.dt.day.astype("float32")
    df["hour"] = dt.dt.hour.astype("float32")
    df["weekday"] = dt.dt.weekday.astype("float32")

    df["pickup_datetime"] = dt.astype(str)

    weekday = df["weekday"]
    hour = df["hour"]

    df["night"] = ((weekday < 5) & ((hour >= 20) | (hour <= 6))).astype("uint8")
    df["late_night"] = (hour <= 3).astype("uint8")
    df["rush_hour"] = ((weekday < 5) & (hour >= 16) & (hour <= 20)).astype("uint8")

    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["latdiff"] = (lat1 - lat2).astype("float32")
    df["londiff"] = (lon1 - lon2).astype("float32")
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2).astype("float32")
    df["distance"] = np.sqrt(
        np.abs(df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
        + np.abs(df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
    ).astype("float32")
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {
            id_column: raw_test[id_column].astype(str),
            prediction_column: prediction.reshape(-1),
        }
    )
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()

    if "rmse" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["rmse"])
        plt.plot(history.history["val_rmse"])
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper right")
        plt.show()




## === cell 7
print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)

print("testKaggle clean")
testKaggle = clean(testKaggle)



## === cell 8
train_df.describe()



## === cell 9
validation_df.describe()



## === cell 10
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 11
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 12
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)
print("Done with Adding features")



## === cell 13
train_df.describe()



## === cell 14
validation_df.describe()



## === cell 15
dropped_columns = ["passenger_count", "pickup_datetime"]

train_df = train_df.drop(dropped_columns + ["key"], axis=1)
validation_df = validation_df.drop(dropped_columns + ["key"], axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")



## === cell 16
for _df in (train_df, validation_df, testKaggle_clean):
    for c in _df.columns:
        if c != "fare_amount":
            if _df[c].dtype.kind in "fc":
                _df[c] = _df[c].fillna(0.0)
            else:
                _df[c] = _df[c].fillna(0)

train_labels = train_df["fare_amount"].values.astype("float32")
validation_labels = validation_df["fare_amount"].values.astype("float32")

if USE_LOG_TARGET:
    train_labels = np.log1p(train_labels)
    validation_labels = np.log1p(validation_labels)

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 17
train_df.shape



## === cell 18
validation_df.shape




## === cell 19
def fit_clip_bounds(df, q_low=0.001, q_high=0.999):
    bounds = {}
    for c in df.columns:
        if df[c].dtype.kind in "fc":
            lo = float(df[c].quantile(q_low))
            hi = float(df[c].quantile(q_high))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                continue
            bounds[c] = (lo, hi)
    return bounds


def apply_clip_bounds(df, bounds):
    df = df.copy()
    for c, (lo, hi) in bounds.items():
        if c in df.columns:
            df[c] = df[c].clip(lower=lo, upper=hi)
    return df


clip_bounds = fit_clip_bounds(train_df, q_low=0.001, q_high=0.999)
train_df_clip = apply_clip_bounds(train_df, clip_bounds)
validation_df_clip = apply_clip_bounds(validation_df, clip_bounds)
testKaggle_clip = apply_clip_bounds(testKaggle_clean, clip_bounds)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df_clip)
validation_df_scaled = scaler.transform(validation_df_clip)
testKaggle_scaled = scaler.transform(testKaggle_clip)




## === cell 20
def rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=-1))




## === cell 21
model = Sequential()
model.add(
    Dense(
        256,
        activation="relu",
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

model.compile(loss="mean_squared_error", optimizer=adam, metrics=["mae", rmse])

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
print(
    "Model visualization skipped (keras.utils.vis_utils not available / not needed for submission)."
)



## === cell 23
plot_loss_accuracy_rmse(history)



## === cell 24
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## === cell 25
predictionKaggle = predictionKaggle.reshape(-1).astype("float32")
if USE_LOG_TARGET:
    predictionKaggle = np.expm1(predictionKaggle)

predictionKaggle = np.maximum(predictionKaggle, 0.0).reshape(-1, 1)
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_PATH)



## === cell 26
print("Wrote submission to:", SUBMISSION_PATH)
print("Submission preview:")
print(pd.read_csv(SUBMISSION_PATH).head())
