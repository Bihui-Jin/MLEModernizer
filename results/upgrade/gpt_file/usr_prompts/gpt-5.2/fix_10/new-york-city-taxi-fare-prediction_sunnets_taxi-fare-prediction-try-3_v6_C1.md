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

4.57724

# 6. Current score

7.9584

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 432.95527) has done: 'I fix the crash that prevents training by removing the external URL image dependency in `remove_datapoints_from_water()` (Kaggle has no internet, and the current failure bubbles up as a protobuf/keras initialization error). I also correct path handling to use the actual provided `/kaggle/input/...` locations so the data loads reliably, and fix Keras 3 optimizer creation (`Adam(learning_rate=...)` instead of deprecated `optimizers.adam(lr=...)`). Finally, I correct the time parsing logic (the current format string is incompatible with the dataset) and fix the `late_night`/`night` conditions (they currently always/never trigger incorrectly), keeping the same features and model architecture while ensuring the script runs end-to-end and writes a valid `submissiontry_water.csv`.'
- What this solution (achieved 163.27774) has done: 'I fix the initial TensorFlow/Keras import crash by switching to the Kaggle-provided `tf_keras` backend (compatible with Keras 3 packages in this environment) while keeping the exact same model architecture and training loop. Then I fix the `KeyError: ['key'] not in index` by ensuring the `key` column is dropped from the training features (it was mistakenly kept), so train/validation/test feature columns align deterministically. Finally, I keep all paths unchanged, ensure the pipeline runs end-to-end, and always writes a valid `submissiontry_water.csv` with `key,fare_amount` columns.'
- What this solution (achieved 120.52414) has done: 'I fix the import-time crash (`MessageFactory` / protobuf incompatibility) by forcing the pure-Python protobuf implementation before importing `tf_keras`, which is the minimal change to unblock the whole pipeline. Then I make sure the training and test paths resolve correctly in this dataset layout by falling back to the known `/kaggle/input/new-york-city-taxi-fare-prediction/...` directory if `/kaggle/input/train.csv` is not present. Finally, to move RMSE strongly toward the target without changing the model architecture or training loop, I correct the feature prep bug where `passenger_count` was being dropped (it’s a legitimate signal) and keep the rest of the pipeline identical, still writing a valid `submissiontry_water.csv` with `key,fare_amount`.'
- What this solution (achieved 46.90842) has done: 'The crash is caused by forcing protobuf’s C++ implementation (`cpp`), which is incompatible with the installed protobuf build in this Kaggle image; switching to the pure-Python implementation fixes the `cannot import name '_message'` import error and unblocks `tf_keras`/TensorFlow. Once that import succeeds, the downstream `NameError`s (missing `TRAIN_PATH`, `Sequential`, `history`, etc.) also disappear because cell 0 execute fully. I also make the train/test path resolution robust to both `/kaggle/input/*.csv` and the competition subfolder (unchanged semantics), and keep the same model/feature logic so scoring behavior only changes from “no submission” to a valid baseline. Finally, the script always write `submissiontry_water.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 7.96775) has done: 'We fix the protobuf/Keras import crash by removing the incompatible protobuf override and instead using the Kaggle-stable `tf_keras`/TensorFlow import path without forcing a specific protobuf implementation. Then we keep your exact feature engineering and model architecture/training loop, but improve RMSE substantially (toward your 4.57724 target) with a minimal, metric-aligned post-processing step: clip negative and extreme fares to the same valid range used during training cleanup (0–50). Finally, we make sure the submission is always written as a valid `.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 7.9584) has done: 'I fix the current runtime crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by forcing protobuf to use the pure-Python implementation before importing `tf_keras`, which is a minimal, well-known Kaggle workaround for protobuf/TensorFlow incompatibilities. I keep your model architecture, training loop, features, and post-processing the same, only ensuring the import stack is stable and deterministic. I also keep the existing robust train/test path fallback and ensure the submission is written as a valid `.csv` with the required `key,fare_amount` columns. These changes are execution/stability fixes and should be score-neutral relative to your current 7.96775 submission behavior (once it runs).'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

try:
    import tf_keras as tf
    from tf_keras.models import Sequential
    from tf_keras.layers import Dense, BatchNormalization
    from tf_keras import optimizers, regularizers
    import tf_keras.backend as K
except Exception as e:
    raise RuntimeError(
        "Failed to import tf_keras in this Kaggle environment. Original error:\n" f"{e}"
    )

TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
DATASET_SIZE = 60000

os.environ["PYTHONHASHSEED"] = "1"
np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)

print("Using TRAIN_PATH:", TRAIN_PATH)
print("Using TEST_PATH :", TEST_PATH)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
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


def remove_datapoints_from_water(df):
    return df


def late_night(row):
    h = int(row["hour"])
    return 1 if (h <= 3 or h >= 22) else 0


def night(row):
    h = int(row["hour"])
    wd = int(row["weekday"])
    return 1 if (h > 20 and wd < 5) else 0


def rush_hour(row):
    h = int(row["hour"])
    wd = int(row["weekday"])
    return 1 if ((h <= 20) and (h >= 16) and (wd < 5)) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["year"] = dt.dt.year.astype("int16")
    df["month"] = dt.dt.month.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["weekday"] = dt.dt.weekday.astype("int8")

    df = df.drop("pickup_datetime", axis=1)

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


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    pred = np.asarray(prediction).reshape(-1)
    if len(pred) != len(raw_test):
        raise ValueError(
            f"Prediction length {len(pred)} != test length {len(raw_test)}"
        )

    df = pd.DataFrame({id_column: raw_test[id_column].values, prediction_column: pred})
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
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


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))




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

train = pd.read_csv(TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes)
test = pd.read_csv(TEST_PATH, dtype={"key": "str", "pickup_datetime": "str"})

train = clean(train)

train = add_time_features(train)
test = add_time_features(test)

train = add_coordinate_features(train)
test = add_coordinate_features(test)

train = add_distances_features(train)
test = add_distances_features(test)

dropped_columns = ["key"]
train_clean = train.drop(dropped_columns, axis=1)
test_clean = test.drop(["key"], axis=1)

train_df, validation_df = train_test_split(train_clean, test_size=0.10, random_state=1)

train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)

train_df = train_df.sort_index(axis=1)
validation_df = validation_df[train_df.columns]
test_clean = test_clean[train_df.columns]

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_clean)

print(
    "Train features shape:",
    train_df_scaled.shape,
    "Validation:",
    validation_df_scaled.shape,
    "Test:",
    test_scaled.shape,
)



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



## === cell 4
plot_loss_accuracy(history)



## === cell 5
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
prediction = np.clip(np.asarray(prediction).reshape(-1), 0.0, 50.0)



## === cell 6
output_submission(test, prediction, "key", "fare_amount", SUBMISSION_NAME)
