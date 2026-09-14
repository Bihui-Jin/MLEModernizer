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

4.84169

# 6. Current score

8.07435

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 220.92615) has done: 'I fix the runtime crash by forcing Keras 3 to use the TensorFlow backend (the error you see commonly comes from protobuf/TensorFlow initialization issues when the backend isn’t set cleanly). I also remove the external `plt.imread("https://...")` dependency in the water-mask cleaner (Kaggle has no internet), and fix a couple of logic/IO issues that would otherwise break training (missing `fare_amount` due to wrong `usecols`, and cleaning code that computes `train = ...` but returns the unfiltered `df`). These are minimal, correctness-focused changes that keep your model and feature logic intact and ensure a valid `submissiontry_water.csv` is written.'
- What this solution (achieved 2352.73797) has done: 'I fix the TensorFlow/Keras initialization crash (`MessageFactory.GetPrototype`) by importing TensorFlow first and forcing the pure-Python protobuf implementation before Keras loads, which is the common Kaggle fix for this exact protobuf mismatch. I also correct three time-feature logic bugs (`late_night`, `night`, `rush_hour`) that currently make those engineered features nearly constant/incorrect and are a direct cause of the extremely poor RMSE; this is a minimal change that preserves your feature set and model/training loop but makes the intended features actually work. Finally, I ensure the training CSV read includes the `key` column only where needed (submission uses test `key` already) and keep paths/outputs unchanged so a valid `submissiontry_water.csv` is always written end-to-end.'
- What this solution (achieved 2947.80079) has done: 'I fix the protobuf/TensorFlow/Keras crash by setting the environment variables before *any* TensorFlow/Keras-related import and by avoiding importing `keras.backend` (which can trigger the same protobuf path) since it is unused for training. I also correct the file paths to the actual Kaggle dataset location you listed (`/kaggle/input/...`) so the script runs end-to-end in the Kaggle environment without missing-file errors. These changes are runtime/stability fixes and do not alter your model architecture, features, training loop, or loss, so the score impact should come only from the code actually running correctly (and not failing before training/prediction). The submission still be written as `submissiontry_water.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 175.42634) has done: 'I fix the TensorFlow/Keras initialization crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before any protobuf/tensorflow/keras import* and by importing `google.protobuf` early, which is the standard Kaggle-safe workaround for this mismatch. I also correct the training CSV `usecols` to include `key`, because your submission writer needs the test `key` and the train key is harmless but prevents downstream mismatches if you later reuse utilities; this is score-neutral. Finally, I make submission predictions strictly 1D float values (Keras can return shape `(n,1)`), preventing occasional CSV formatting/type issues without changing the model or training loop.'
- What this solution (achieved 567.48066) has done: 'I fix the runtime crash caused by the protobuf `MessageFactory.GetPrototype` mismatch by forcing TensorFlow to use the pure-Python protobuf implementation *and* disabling the C++ implementation before any protobuf/TensorFlow/Keras import, which is the common Kaggle-stable workaround for this exact error. I also make the backend selection deterministic by importing `tensorflow` before importing `keras`, while keeping your model, features, and training loop unchanged. Finally, I keep the submission writer as-is but ensure the pipeline always reaches CSV creation end-to-end.'
- What this solution (achieved 1115.75726) has done: 'I fix the protobuf crash by ensuring we don’t import `google.protobuf` directly (that import is what triggers the `MessageFactory.GetPrototype` attribute error in this Kaggle image) and by forcing the pure-Python protobuf implementation before TensorFlow/Keras load. Then I make the time-derived binary features (`night`, `late_night`, `rush_hour`) vectorized and logically consistent so they aren’t nearly-constant/incorrect, which is a direct cause of the very high RMSE while preserving the same feature intent. I also keep your model/training loop unchanged, but ensure the submission writes `key,fare_amount` with a 1D float array and a `.csv` suffix.'
- What this solution (achieved 51.41664) has done: 'I fix the protobuf/TensorFlow/Keras initialization crash that prevents the notebook from running by ensuring the pure-Python protobuf runtime is selected before any TensorFlow/Keras import and by importing `tf_keras` (the Kaggle-installed TensorFlow-compatible Keras) instead of standalone `keras` which is what typically triggers this specific `MessageFactory.GetPrototype` error. I keep your model architecture, feature engineering, training loop, and submission-writing logic unchanged. I also make the backend/env setup occur at the very top (before any other imports) so it’s actually effective in the Kaggle runtime. This should restore end-to-end execution and produce a valid `submissiontry_water.csv`, and because your previous good scores depended on the pipeline actually training, it should move RMSE dramatically down toward the target again.'
- What this solution (achieved 6.04744) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime and importing `tf_keras`/TensorFlow in a safe order before any other heavy imports that can indirectly load protobuf. I also make the dataset path robust to both `/kaggle/input/...` layouts you listed so the code always finds the CSVs without changing any modeling logic. Finally, I keep your feature engineering/model/training loop identical, only adding a small safety clamp to prevent negative fare predictions (which can severely hurt RMSE) while still writing a valid `submissiontry_water.csv`.'
- What this solution (achieved 7.82675) has done: 'I fix the protobuf/TensorFlow initialization crash by making the protobuf env vars take effect before any TensorFlow-related import and by importing `tf_keras` (TensorFlow-bundled Keras) in a safer order that avoids the `MessageFactory.GetPrototype` issue in this Kaggle image. I keep your model, features, training loop, and submission logic unchanged, but I add a small, score-improving post-process that matches the common competition constraint: clip predictions to a reasonable fare range (0–50), consistent with your training outlier filter (this usually reduces RMSE versus only clipping negatives). I also keep the dataset path logic and ensure the script always writes `submissiontry_water.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 5.85942) has done: 'We fix the runtime crash (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf implementation *before anything that might import protobuf*, and by explicitly importing `google.protobuf` early (this avoids the mismatched C++ protobuf path that triggers the attribute error in some Kaggle images). We also ensure TensorFlow uses a single thread configuration for stability in the Kaggle container (score-neutral) and keep your model/feature/training logic identical. Finally, we keep the submission formatting exactly as required (`key,fare_amount`) and guarantee predictions are a flat float array clipped to the same reasonable range you already intended.'
- What this solution (achieved 25.61528) has done: 'We fix the immediate runtime crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by avoiding the fragile `google.protobuf` import path and instead forcing TensorFlow-bundled Keras (`tf_keras`) to be used safely, with protobuf env vars set before any TF/Keras import. This is a minimal stability change that keeps your feature engineering, model architecture, training loop, and loss exactly the same. To nudge RMSE toward the target, we also clip predictions only at the low end (no negative fares) and not cap at 50, because the test set can legitimately contain higher fares and the hard upper clip can inflate RMSE when those occur. The script still run end-to-end and always write `submissiontry_water.csv` with `key,fare_amount`.'
- What this solution (achieved 7.78336) has done: 'We fix the immediate protobuf/TensorFlow initialization crash by selecting the pure-Python protobuf runtime before any TF/Keras import and by importing TensorFlow first, then using `tf.keras` (which avoids the fragile standalone `keras`/protobuf path that triggers `MessageFactory.GetPrototype`). This is a runtime/stability fix that keeps your model architecture, features, training loop, and loss unchanged. To move RMSE down toward the target (lower is better) with minimal semantic impact, we also apply the same upper-bound clipping you already used during training cleaning (`fare_amount <= 50`) to the test predictions (in addition to the existing non-negative clamp), which typically reduces RMSE for this competition given the train-time target range. The script still run end-to-end and write `submissiontry_water.csv` with `key,fare_amount`.'
- What this solution (achieved 8.07435) has done: 'I fix the runtime crash caused by the protobuf/TensorFlow/Keras incompatibility by forcing the pure-Python protobuf implementation and importing TensorFlow in a safe order before any other imports that can indirectly load protobuf. This is strictly a stability fix and does not change your model, features, training loop, or loss. I also make the `matplotlib` import lazy inside the plotting/water-mask functions so it can’t trigger the protobuf path early, while keeping the same behavior. Finally, I keep your existing prediction shaping/clipping and ensure the submission CSV is always written as `submissiontry_water.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_C"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import tensorflow as tf  # noqa: F401

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

_CANDIDATE_BASES = [
    "/kaggle/input/new-york-city-taxi-fare-prediction",
    "/kaggle/input/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction",
]
_BASE = next(
    (p for p in _CANDIDATE_BASES if os.path.exists(os.path.join(p, "train.csv"))),
    _CANDIDATE_BASES[0],
)

TRAIN_PATH = os.path.join(_BASE, "train.csv")
TEST_PATH = os.path.join(_BASE, "test.csv")
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
DATASET_SIZE = 60000

np.random.seed(1)
tf.random.set_seed(1)

print("Using TRAIN_PATH:", TRAIN_PATH)
print("Using TEST_PATH :", TEST_PATH)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def remove_datapoints_from_water(df):
    """
    Kaggle notebooks have no internet; if the mask image isn't present locally,
    skip water-mask filtering (score-neutral stability behavior).
    """
    local_mask_path = "nyc_mask-74.5_-72.8_40.5_41.8.png"
    if not os.path.exists(local_mask_path):
        return df

    import matplotlib.pyplot as plt

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)
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
    print(" New size after only NYC: %d" % len(df))

    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0)]
    print(" New size after removing passenger_count > 0: %d" % len(df))

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


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["year"] = dt.dt.year.astype("float32")
    df["month"] = dt.dt.month.astype("float32")
    df["day"] = dt.dt.day.astype("float32")
    df["hour"] = dt.dt.hour.astype("float32")
    df["weekday"] = dt.dt.weekday.astype("float32")

    hour = dt.dt.hour.fillna(0).astype(np.int16)
    weekday = dt.dt.weekday.fillna(0).astype(np.int16)

    df["late_night"] = (hour <= 3).astype("uint8")
    df["night"] = (((hour >= 20) | (hour <= 6)) & (weekday < 5)).astype("uint8")
    df["rush_hour"] = ((hour >= 16) & (hour <= 20) & (weekday < 5)).astype("uint8")

    df = df.drop("pickup_datetime", axis=1)
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
    pred = np.asarray(prediction).reshape(-1).astype(np.float32)
    pred = np.clip(pred, 0.0, 50.0)

    df = pd.DataFrame({id_column: raw_test[id_column].values, prediction_column: pred})
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))


def plot_loss_accuracy(history):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
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

train = pd.read_csv(
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
test = pd.read_csv(
    TEST_PATH, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
)

train = clean(train)

train = add_time_features(train)
test = add_time_features(test)

train = add_coordinate_features(train)
test = add_coordinate_features(test)

train = add_distances_features(train)
test = add_distances_features(test)

dropped_columns = ["passenger_count"]

train_clean = train.drop(dropped_columns + ["key"], axis=1)
test_clean = test.drop(dropped_columns + ["key"], axis=1)

train_df, validation_df = train_test_split(train_clean, test_size=0.10, random_state=1)

train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_clean)



## === cell 3
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
model.compile(loss="mse", optimizer=adam, metrics=["mae"])

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

plot_loss_accuracy(history)

prediction = model.predict(test_scaled, batch_size=128, verbose=1)

output_submission(test, prediction, "key", "fare_amount", SUBMISSION_NAME)
