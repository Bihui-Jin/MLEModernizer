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

4.85575

# 6. Current score

12.795

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1091.41409) has done: 'I fix the runtime crash by ensuring Keras uses the TensorFlow-backed `tf_keras` package (Kaggle’s environment can trigger protobuf/MessageFactory issues with standalone `keras==3.x`). I also fix path issues by pointing `TRAIN_PATH/TEST_PATH` to the actual `/kaggle/input/...` locations, and remove the internet dependency in `remove_datapoints_from_water` (Kaggle notebooks have no outbound internet), while keeping the cleaning logic otherwise intact. Finally, I repair a couple of logical bugs in the time-feature flags so they behave as intended (previous conditions were always true), which should improve RMSE without changing the overall modeling approach. The script run end-to-end and write a valid `submissiontry_water.csv` with columns `key,fare_amount`.'
- What this solution (achieved 13.76488) has done: 'I fix the Keras/protobuf crash by importing `tf_keras` first and forcing protobuf to use the pure-Python implementation (this is the common root cause of the `MessageFactory.GetPrototype` error in Kaggle). I also fix the training/test CSV loading so that the `key` column is retained for test (your current code drops it, which breaks submission alignment) and so that train/test use consistent columns. Finally, I make prediction output 1D numeric values and clip to a reasonable non-negative range to avoid pathological submissions that explode RMSE, without changing the model or training procedure.'
- What this solution (achieved 14.37951) has done: 'I fix the protobuf/Keras crash by forcing the pure-Python protobuf implementation *before* any TensorFlow/tf_keras import and by clearing any pre-imported protobuf modules, which is the root cause of the `MessageFactory.GetPrototype` error. I also correct the data loading bug where `clean()` expects `fare_amount` but the training read currently omits it, which can silently break training/labels and hurt RMSE. Finally, I correct the Kaggle input paths to the actual competition directory so the notebook reliably finds the CSVs, while keeping the model, features, and training loop unchanged so the score moves toward the target mainly through correctness.'
- What this solution (achieved 6.0238) has done: 'I fix the protobuf/Keras crash by avoiding the standalone protobuf-generated Keras 3 stack entirely and using `tensorflow`’s bundled `tf.keras` (which is stable on Kaggle) while keeping the same Sequential architecture, optimizer, epochs, and feature pipeline. I also ensure `pickup_datetime` parsing can’t fail on missing/invalid values by filling NaT-derived fields safely, preventing downstream dtype errors. Finally, I keep the submission alignment intact (using `test["key"]`) and guarantee the output is a 1D numeric array written to a `.csv` file with the required header.'
- What this solution (achieved 14.04536) has done: 'I fix the `MessageFactory.GetPrototype` crash by forcing protobuf to use the pure-Python implementation before importing TensorFlow/Keras, which is the stable workaround in Kaggle’s environment. I also correct the input file paths to the actual dataset location under `/kaggle/input/new-york-city-taxi-fare-prediction/`, so the CSVs are found reliably. To move RMSE toward your target, I stop dropping `passenger_count` (it’s a useful predictive feature and removing it hurts score) while keeping the model architecture, training loop, and loss unchanged. Finally, I ensure the submission is written as a valid `.csv` with `key,fare_amount` and aligned row order.'
- What this solution (achieved 5.79319) has done: 'I fix the `MessageFactory.GetPrototype` crash by forcing protobuf to use the pure-Python implementation *before* TensorFlow is imported and by clearing any already-imported `google.protobuf` modules, which is the reliable Kaggle workaround without changing your model/training logic. I also correct a data-loading bug where the training CSV read drops the `key` column (harmless) but more importantly can drop `fare_amount` from downstream alignment if columns are later manipulated; we keep `fare_amount` explicitly and ensure feature columns are consistent between train/validation/test. To move RMSE toward your target with minimal semantic change, I fix the coordinate “same long/lat” filter to remove only true duplicates (currently it removes too aggressively when only one coordinate matches), which improves data quality and model fit without altering architecture or training loop. Finally, I ensure the submission is written to a `.csv` file with exact `key,fare_amount` columns and correct row alignment.'
- What this solution (achieved 13.53272) has done: 'I fix the protobuf/Keras crash by forcing the pure-Python protobuf runtime before TensorFlow import and by avoiding `tensorflow.keras` (which can still trip the `MessageFactory.GetPrototype` issue in this Kaggle environment); instead I use the already-installed `tf_keras` backend, keeping the exact same model architecture and training loop. I also correct the NYC landmark filters in `clean()` where the boolean logic currently removes too many rows (it uses `&` where it should only remove rows that match BOTH latitude and longitude); this is a minimal data-quality fix that should improve RMSE toward your target without changing the modeling approach. Finally, I ensure the training CSV read includes `key` (harmless but keeps train/test column parity) and keep submission writing unchanged while guaranteeing the output is a valid `key,fare_amount` CSV.'
- What this solution (achieved 5.85831) has done: 'I fix the protobuf/Keras crash by avoiding the TensorFlow import path that triggers `MessageFactory.GetPrototype` in this environment and instead using `tf_keras` only (same model/training logic). I also speed up and stabilize time-feature creation by vectorizing the `night/late_night/rush_hour` flags (same semantics, but avoids slow `.apply` that can time out). Finally, I make sure the submission is always written as a valid `key,fare_amount` CSV with predictions aligned to the original test `key` order.'
- What this solution (achieved 13.49798) has done: 'I fix the protobuf/Keras crash by ensuring the pure-Python protobuf runtime is selected *before* any TensorFlow/tf_keras import, and by clearing any already-imported protobuf modules so the setting actually takes effect. I also add small determinism settings (seeds and deterministic ops) to stabilize training behavior in Kaggle without changing the model, features, loss, or training loop semantics. Finally, I keep submission writing identical but ensure the file always ends with `.csv` and predictions are 1D aligned to `test["key"]`, so the notebook reliably produces a valid submission file end-to-end.'
- What this solution (achieved 12.795) has done: 'I fix the runtime crash by switching the standalone `tf_keras` import to TensorFlow’s bundled `tf.keras`, which avoids the protobuf `MessageFactory.GetPrototype` issue in this Kaggle environment while keeping your exact model architecture and training loop intact. I keep all feature engineering, scaling, and cleaning logic the same, only adjusting imports/seeding to be compatible with TF. I also ensure the submission is always written as a valid `.csv` with `key,fare_amount` and predictions are 1D aligned to `test["key"]` (your current code already does this, so only minor robustness is added). These changes should both unblock execution and move RMSE back toward your target range by restoring stable training/inference.'
- What this solution (achieved 12.795) has done: 'I fix the TensorFlow/Keras import crash (`MessageFactory.GetPrototype`) by forcing protobuf to use the pure-Python implementation *before* importing TensorFlow and by clearing any already-imported protobuf modules so the setting actually applies. This is a runtime-only stabilization change and keeps your model architecture, training loop, and feature engineering intact. I also keep your data paths and submission writing the same, only adding a small safeguard to ensure the output is a valid 1D numeric `fare_amount` aligned to `test["key"]`. These changes should make the notebook run end-to-end and, by restoring stable TF execution, move RMSE back toward your target band compared to the current broken/unstable state.'

# 9. Code solution

## === cell 0
import os
import sys
import random

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
for _m in list(sys.modules.keys()):
    if _m.startswith("google.protobuf"):
        del sys.modules[_m]

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import optimizers, regularizers, backend as K

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
try:
    tf.random.set_seed(SEED)
except Exception:
    pass

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001
DATASET_SIZE = 60000


def remove_datapoints_from_water(df):
    return df


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


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)

    df["year"] = dt.dt.year.fillna(0).astype("int16")
    df["month"] = dt.dt.month.fillna(0).astype("int8")
    df["day"] = dt.dt.day.fillna(0).astype("int8")
    df["hour"] = dt.dt.hour.fillna(0).astype("int8")
    df["weekday"] = dt.dt.weekday.fillna(0).astype("int8")

    hour = df["hour"].astype("int16")
    weekday = df["weekday"].astype("int16")

    df["night"] = (((hour >= 20) | (hour <= 6)) & (weekday < 5)).astype("int8")
    df["late_night"] = ((hour <= 3) | (hour >= 23)).astype("int8")
    df["rush_hour"] = ((hour >= 16) & (hour <= 20) & (weekday < 5)).astype("int8")

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
    if len(pred) != len(raw_test):
        raise ValueError(
            f"Prediction length {len(pred)} != test length {len(raw_test)}"
        )
    df = pd.DataFrame({id_column: raw_test[id_column].values, prediction_column: pred})
    if not file_name.endswith(".csv"):
        file_name = file_name + ".csv"
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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

dropped_columns = ["key"]

train_clean = train.drop(dropped_columns, axis=1)
test_clean = test.drop(dropped_columns, axis=1)

train_df, validation_df = train_test_split(train_clean, test_size=0.10, random_state=1)

train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_clean)




## === cell 2
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
prediction = np.asarray(prediction).reshape(-1)
prediction = np.clip(prediction, 0.0, 100.0)

output_submission(test, prediction, "key", "fare_amount", SUBMISSION_NAME)
