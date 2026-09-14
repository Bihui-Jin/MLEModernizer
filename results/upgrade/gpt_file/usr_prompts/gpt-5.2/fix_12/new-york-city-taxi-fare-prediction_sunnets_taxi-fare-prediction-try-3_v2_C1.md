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

4.71823

# 6. Current score

5.81569

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 389.22137) has done: 'I fix the runtime crash by forcing Keras to use the TensorFlow-backed `tf_keras` package (your environment’s `keras==3` can trigger the protobuf `MessageFactory.GetPrototype` error). I also remove the external URL image dependency in the water-mask filter (no internet on Kaggle) while keeping the rest of the cleaning and feature logic intact. I correct a couple of obvious logic bugs in the time-feature flags (conditions that always evaluate true) and make datetime parsing robust to the dataset’s actual format. Finally, I fix the data loading `usecols` so `fare_amount` is included (it was being dropped), and ensure a valid `submissiontry_water.csv` is written with `key,fare_amount`.'
- What this solution (achieved 6.01542) has done: 'I fix the immediate runtime crash caused by an incompatible protobuf/TensorFlow/Keras import combination by importing TensorFlow first and forcing the legacy `tf_keras` backend deterministically. Then I fix a major data logic issue that drives the huge RMSE: the training CSV is being loaded without the `key` column but later the pipeline expects `key` to exist (and the train/test feature columns become misaligned); I load `key` for train and ensure both train and test drop/keep the same columns. Finally, I keep your model/training loop intact but make sure the submission is written with the correct `key,fare_amount` columns and row alignment.'
- What this solution (achieved 282.92907) has done: 'I fix the immediate protobuf/Keras crash by preventing `keras==3` from being imported implicitly and by forcing the TensorFlow-backed `tf_keras` implementation via `sys.modules` before any Keras-related imports. I also correct the Kaggle file paths (`../input/train.csv` doesn’t exist here) to use the provided `/kaggle/input/...` paths so the script can actually read the data. To improve RMSE toward your target without changing the model/training loop, I fix a logic error in the airport/landmark filtering: the current `!=` with `&` removes almost nothing (it should remove rows that match both lat and lon), so switching to a proper “not (lat==A and lon==B)” filter should reduce label noise. Finally, I keep the submission format identical (`key,fare_amount`) and ensure the output CSV is always written.'
- What this solution (achieved 10.73543) has done: 'The crash happens before any training because the environment’s protobuf version is incompatible with TensorFlow’s default (C++) protobuf implementation, triggering `MessageFactory.GetPrototype`. The minimal robust fix is to force protobuf to use the pure-Python implementation **before importing TensorFlow**, which avoids that AttributeError without changing your model or training logic. I also add a safe fallback for train/test paths (both `/kaggle/input/...` and `/kaggle/input/new-york-city-taxi-fare-prediction/...`) so the notebook runs reliably in this dataset layout. Everything else (cleaning, features, network, training loop, and submission format) is kept intact to move RMSE down simply by making the run stable and consistent.'
- What this solution (achieved 5.55912) has done: 'You’re still crashing before training because the protobuf “python” fallback env var must be set *before any protobuf/TensorFlow-related import happens*, and Kaggle can import those indirectly very early. I make that fix by moving the environment-variable setup to the absolute top and forcing pure-Python protobuf via both env vars and a safe import order, while keeping your model/training loop unchanged. Then, to improve RMSE toward the 4.72 target without changing architecture or training semantics, I fix a subtle but impactful cleaning bug: filtering “same long/lat” currently drops most valid trips (it requires both lat *and* lon to differ); it should only drop rows where both pickup and dropoff coordinates are identical. Finally, I keep the submission format identical and ensure the CSV is always written correctly.'
- What this solution (achieved 79.1861) has done: 'I fix the runtime crash by forcing protobuf to use the pure-Python implementation in the most robust way (setting both env vars before any TensorFlow/Keras import) and by importing `google.protobuf` before importing TensorFlow. This keeps your model/training loop unchanged while preventing the `MessageFactory.GetPrototype` incompatibility in this Kaggle environment. I also make the Keras backend forcing more deterministic (ensuring any accidental `keras==3` imports can’t sneak in) without altering architecture or training semantics. Finally, I keep the same data/feature pipeline and ensure the submission CSV is always written as `key,fare_amount` with the expected filename.'
- What this solution (achieved 117.77723) has done: 'You’re crashing immediately due to the protobuf `MessageFactory.GetPrototype` incompatibility happening before TensorFlow fully initializes, so we force the pure‑Python protobuf implementation and import order at the very top in a more defensive way. To also move RMSE strongly toward your target (current 79 is far off), the minimal “core-logic-preserving” fix is to stop scaling the label column by accident: right now `MinMaxScaler` is fit on a dataframe that still contains `key` (a string), which can corrupt/derail training; we explicitly ensure only numeric feature columns are scaled and passed to the model. We also make sure the feature columns are identical and ordered for train/validation/test to prevent silent misalignment. Finally, we keep the same network, optimizer, epochs, and training loop, and still write `key,fare_amount` to `submissiontry_water.csv`.'
- What this solution (achieved 212.60746) has done: 'I fix the immediate crash caused by the protobuf/TensorFlow mismatch by setting the required environment variables before any protobuf/TensorFlow-related import and by avoiding an early `google.protobuf` import (which can lock in the C++ implementation). Then I keep your exact data/feature/model/training logic unchanged, but add a small defensive fallback: if TensorFlow still fails to import in this Kaggle image, the script automatically retry with the pure-Python protobuf settings in a fresh subprocess so you still get a valid submission CSV. Finally, I ensure the submission is always written as `submissiontry_water.csv` with the required `key,fare_amount` columns and correct row alignment.'
- What this solution (achieved 6.07088) has done: 'I fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf runtime **before anything can import TensorFlow/protobuf**, and remove the risky “rerun subprocess” logic (it can’t help if the kernel already loaded the C++ protobuf). Then I keep your exact data cleaning/feature engineering/model/training loop, but make one minimal, score-relevant correction: fit the MinMaxScaler on the same cleaned numeric feature set you actually train on, and ensure test rows with NaNs are handled consistently (fill with train medians) so predictions aren’t corrupted by NaNs/infs. Finally, I ensure the submission is always written as a valid `key,fare_amount` CSV with the correct row alignment.'
- What this solution (achieved 19.10662) has done: 'I fix the protobuf/TensorFlow crash by forcing the pure‑Python protobuf runtime as early and as defensively as possible (before any TensorFlow-related import can occur), which addresses the `MessageFactory.GetPrototype` error. To move RMSE toward your target without changing the model/training loop, I correct one score-hurting data issue: the current cleaning step accidentally *removes* rows exactly at major landmarks/airports, which are often valid and informative; instead we should remove rows only when coordinates are outside NYC bounds/zeros/same-point, and keep those landmark rows. Finally, I keep feature extraction, scaling, model architecture, epochs, and submission format the same, ensuring a valid `submissiontry_water.csv` is written.'
- What this solution (achieved 5.81569) has done: 'We fix the protobuf/TensorFlow crash by forcing pure-Python protobuf *and* preventing an early `google.protobuf` import that can lock the C++ implementation before TensorFlow loads. This is a runtime-only change (no modeling change) and should make the notebook run end-to-end reliably in this Kaggle image. Then we keep your exact data cleaning, feature engineering, model, and training loop unchanged, and ensure the submission CSV is always written as `submissiontry_water.csv` with `key,fare_amount`. With execution unblocked, your existing pipeline should train properly and improve RMSE substantially versus the current crash/instability behavior.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("TF_KERAS", "1")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split


import tensorflow as tf  # import after env vars are set
import tf_keras as keras

sys.modules["keras"] = keras
sys.modules["keras.api._v2.keras"] = keras

from tf_keras.models import Sequential
from tf_keras.layers import Dense, BatchNormalization
from tf_keras import optimizers, regularizers
import tf_keras.backend as K


def _resolve_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("None of the candidate paths exist: %s" % (candidates,))


TRAIN_PATH = _resolve_path(
    "/kaggle/input/train.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
)
TEST_PATH = _resolve_path(
    "/kaggle/input/test.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
)
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
DATASET_SIZE = 60000

np.random.seed(1)
tf.random.set_seed(1)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def remove_datapoints_from_water(df):
    """
    Original implementation attempted to read a mask image from a URL (no internet on Kaggle).
    Keep pipeline behavior stable by skipping this filter when the mask is unavailable.
    """
    return df


def _drop_exact_coord(df, lat_col, lon_col, lat_val, lon_val):
    """
    Drop rows where BOTH latitude and longitude match the point.
    """
    mask = ~((df[lat_col] == lat_val) & (df[lon_col] == lon_val))
    return df[mask]


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
    print(" New size after removing same point trips: %d" % len(df))

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

    print("Old size: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df




## === cell 2
def late_night(row):
    h = row["hour"]
    return 1 if (h <= 3 or h >= 22) else 0


def night(row):
    h = row["hour"]
    wd = row["weekday"]
    return 1 if ((h >= 20 or h <= 6) and wd < 5) else 0


def rush_hour(row):
    h = row["hour"]
    wd = row["weekday"]
    return 1 if (16 <= h <= 20 and wd < 5) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    )
    df = df.dropna(subset=["pickup_datetime"])
    df["year"] = df["pickup_datetime"].dt.year.astype("int16")
    df["month"] = df["pickup_datetime"].dt.month.astype("int8")
    df["day"] = df["pickup_datetime"].dt.day.astype("int8")
    df["hour"] = df["pickup_datetime"].dt.hour.astype("int8")
    df["weekday"] = df["pickup_datetime"].dt.weekday.astype("int8")
    df["night"] = df.apply(lambda x: night(x), axis=1).astype("int8")
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1).astype("int8")
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1).astype("int8")
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
    df = pd.DataFrame(
        {
            id_column: raw_test[id_column].values,
            prediction_column: prediction.reshape(-1),
        }
    )
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows=", len(df))


def plot_loss_accuracy(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()




## === cell 3
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

train = clean(train)

train = add_time_features(train)
test = add_time_features(test)

train = add_coordinate_features(train)
test = add_coordinate_features(test)

train = add_distances_features(train)
test = add_distances_features(test)

dropped_columns = ["passenger_count"]

train_clean = train.drop(dropped_columns, axis=1)
test_clean = test.drop(dropped_columns, axis=1)

feature_cols = [c for c in train_clean.columns if c not in ["fare_amount", "key"]]
train_features = train_clean[feature_cols].copy()
test_features = test_clean[feature_cols].copy()

train_df, validation_df = train_test_split(
    train_clean[["fare_amount"] + feature_cols], test_size=0.10, random_state=1
)

train_labels = train_df["fare_amount"].values.astype(np.float32)
validation_labels = validation_df["fare_amount"].values.astype(np.float32)

train_df = train_df[feature_cols].copy()
validation_df = validation_df[feature_cols].copy()

train_df = train_df.apply(pd.to_numeric, errors="coerce")
validation_df = validation_df.apply(pd.to_numeric, errors="coerce")
test_features = test_features.apply(pd.to_numeric, errors="coerce")

train_nan_mask = train_df.notna().all(axis=1)
val_nan_mask = validation_df.notna().all(axis=1)

if (~train_nan_mask).any():
    train_df = train_df.loc[train_nan_mask].copy()
    train_labels = train_labels[train_nan_mask.values]
if (~val_nan_mask).any():
    validation_df = validation_df.loc[val_nan_mask].copy()
    validation_labels = validation_labels[val_nan_mask.values]

train_medians = train_df.median(axis=0)
test_features = test_features.replace([np.inf, -np.inf], np.nan)
test_features = test_features.fillna(train_medians)

validation_df = validation_df.replace([np.inf, -np.inf], np.nan).fillna(train_medians)
train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(train_medians)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df.values)
validation_df_scaled = scaler.transform(validation_df.values)
test_scaled = scaler.transform(test_features.values)




## === cell 4
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


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

plot_loss_accuracy(history)

prediction = model.predict(test_scaled, batch_size=128, verbose=1)
prediction = np.maximum(prediction, 0.0).astype(np.float32)

output_submission(test, prediction, "key", "fare_amount", SUBMISSION_NAME)
print("Wrote submission to:", os.path.abspath(SUBMISSION_NAME))
