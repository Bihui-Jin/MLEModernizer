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

12.14823

# 6. Current score

11064.39293

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 490.62633) has done: 'I fix the runtime blockers by (1) removing the URL-based water-mask filter (no internet/URL reading in Kaggle) while keeping the rest of your cleaning logic intact, (2) switching the optimizer creation to the Keras 3 API (`optimizers.Adam`) without changing the model/training loop, and (3) making `pd.to_datetime` parsing robust to the actual NYC Taxi timestamp format. I also drop the broken model-plot cell (nonessential) and fix the loss debug call that used mismatched slice shapes. Finally, I ensure the script writes a valid `submissiontry_water.csv` with columns `key,fare_amount` from test predictions.'
- What this solution (achieved 721.71365) has done: 'I fix the immediate runtime crash caused by an incompatible protobuf runtime (the `MessageFactory.GetPrototype` error) by forcing the pure‑Python protobuf implementation before importing anything that transitively imports protobuf. Then I correct two logic bugs in your time-feature flags (`late_night` and `night`) that currently evaluate to “always true/always false” patterns and can destroy model signal, which should move RMSE down substantially toward your target without changing the model architecture or training loop. Finally, I make the train/test CSV reading robust to the provided Kaggle paths (your current `../input/...` paths don’t match your listed filesystem) while keeping filenames and submission format unchanged.'
- What this solution (achieved 1654.43063) has done: 'I fix the protobuf-related import crash by forcing the pure-Python protobuf implementation *and* the matching runtime version before importing `tf_keras`, which is the root cause of the `MessageFactory.GetPrototype` error in this environment. I also correct the dataset paths to match your actual filesystem (`/kaggle/data/...`) so the script reliably reads train/test without silent path issues. To move RMSE strongly toward your target (lower is better) with minimal core-logic change, I ensure the model trains on the intended full feature set by correctly including the needed columns in `read_csv` (your current `usecols` accidentally drops pickup/dropoff coordinates), and I remove the irrelevant `accuracy` metric for regression (score-neutral but avoids misleading training). The rest of the feature engineering, model architecture, training loop, and submission writing remain the same, and the script always write a valid `submissiontry_water.csv`.'
- What this solution (achieved 709.68378) has done: 'I fix the single biggest reason your RMSE is astronomically far from the target: you train the model on `log1p(fare_amount)` but submit predictions back in dollars (`expm1`), while the Kaggle metric is RMSE in dollars. That mismatch makes the model optimize the wrong objective scale, yielding very poor public scores. With minimal change and without altering your model architecture or training loop, I train on raw `fare_amount` instead (and remove the expm1/log1p post-processing) so the loss aligns with the evaluation metric. I also add a small safety clip for negative predictions (valid fares are non-negative) and keep the submission writer unchanged so you always get a valid CSV.'
- What this solution (achieved 11064.39293) has done: 'You don’t yet have a Kaggle score because you haven’t submitted, but your current debug RMSE being extremely large is typically caused by (a) training on a non-representative subsample created by `skiprows` randomness and (b) feeding the network rows that still contain invalid/misaligned coordinates due to the airport/landmark “!= AND !=” filters (those conditions barely remove anything). To move RMSE down substantially toward the target while keeping your model/training loop intact, I make the train sampling deterministic and uniform (read a contiguous block instead of random `skiprows`) and I fix those coordinate “removal” filters to actually exclude points that match the landmark coordinates (using OR). I also keep the submission writer identical but add one safeguard to ensure test cleaning never drops rows (so your submission always has exactly 9914 keys). These are minimal changes that preserve the overall pipeline and should materially reduce error without changing the model architecture or loss.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pbver
    except Exception:
        pbver = None

    target = "protobuf==3.20.3"
    need_install = False
    if pbver is None:
        need_install = True
    else:
        try:
            major = int(pbver.split(".")[0])
            if major >= 4:
                need_install = True
        except Exception:
            need_install = True

    if need_install:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", target])


_ensure_protobuf_compatible()

import random
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tf_keras.callbacks import EarlyStopping
from tf_keras import optimizers
from tf_keras import regularizers
from tf_keras import backend

SEED = 1
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

TRAIN_PATH = "/kaggle/data/train.csv"
TEST_PATH = "/kaggle/data/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001

DATASET_SIZE = 500000

SKIPROWS_STEP = 5

print("Using keras backend:", keras.__name__)



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

train_usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

_effective_nrows = int(DATASET_SIZE)
trainKaggle = pd.read_csv(
    TRAIN_PATH,
    nrows=_effective_nrows,
    dtype=datatypes,
    usecols=train_usecols,
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

print("Read train rows:", len(trainKaggle), "target sample:", DATASET_SIZE)
print("Read test rows:", len(testKaggle))




## === cell 2
def clean(df, is_train=True):
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

    if is_train and "fare_amount" in df.columns:
        df = df[(0.99 < df["fare_amount"]) & (df["fare_amount"] <= 250)]
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
        | (df["pickup_latitude"] != nyc_coord[0])
    ]
    df = df[
        (nyc_coord[1] != df["dropoff_longitude"])
        | (df["dropoff_latitude"] != nyc_coord[0])
    ]
    print(" New size after NY airport: %d" % len(df))

    df = df[
        (fk_coord[1] != df["pickup_longitude"]) | (df["pickup_latitude"] != fk_coord[0])
    ]
    df = df[
        (fk_coord[1] != df["dropoff_longitude"])
        | (df["dropoff_latitude"] != fk_coord[0])
    ]
    print(" New size after jfk airport: %d" % len(df))

    df = df[
        (ewr_coord[1] != df["pickup_longitude"])
        | (df["pickup_latitude"] != ewr_coord[0])
    ]
    df = df[
        (ewr_coord[1] != df["dropoff_longitude"])
        | (df["dropoff_latitude"] != ewr_coord[0])
    ]
    print(" New size after ewr airport: %d" % len(df))

    df = df[
        (lga_coord[1] != df["pickup_longitude"])
        | (df["pickup_latitude"] != lga_coord[0])
    ]
    df = df[
        (lga_coord[1] != df["dropoff_longitude"])
        | (df["dropoff_latitude"] != lga_coord[0])
    ]
    print(" New size after lgr airport: %d" % len(df))

    df = df[
        (sol_coord[1] != df["pickup_longitude"])
        | (df["pickup_latitude"] != sol_coord[0])
    ]
    df = df[
        (sol_coord[1] != df["dropoff_longitude"])
        | (df["dropoff_latitude"] != sol_coord[0])
    ]
    print(" New size after sol removed: %d" % len(df))

    print("Old size: %d" % len(df))
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df.copy().reset_index(drop=True)


def late_night(row):
    return 1 if (row["hour"] <= 3) else 0


def night(row):
    return (
        1 if ((row["hour"] >= 20 or row["hour"] <= 5) and (row["weekday"] < 5)) else 0
    )


def rush_hour(row):
    if ((row["hour"] <= 20) and (row["hour"] >= 16)) and (row["weekday"] < 5):
        return 1
    else:
        return 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(
        df["pickup_datetime"], infer_datetime_format=True, errors="coerce", utc=False
    )
    dt = dt.fillna(pd.Timestamp("2010-01-01 00:00:00"))
    df["pickup_datetime"] = dt
    df["year"] = df["pickup_datetime"].dt.year.astype("int16")
    df["month"] = df["pickup_datetime"].dt.month.astype("int8")
    df["day"] = df["pickup_datetime"].dt.day.astype("int8")
    df["hour"] = df["pickup_datetime"].dt.hour.astype("int8")
    df["weekday"] = df["pickup_datetime"].dt.weekday.astype("int8")

    hour = df["hour"]
    weekday = df["weekday"]

    df["night"] = (((hour >= 20) | (hour <= 5)) & (weekday < 5)).astype("int8")
    df["late_night"] = (hour <= 3).astype("int8")
    df["rush_hour"] = (((hour <= 20) & (hour >= 16)) & (weekday < 5)).astype("int8")

    df["pickup_datetime"] = df["pickup_datetime"].astype(str)
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
    df["distance"] = np.sqrt(
        np.abs(df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
        + np.abs(df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
    )
    return df


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


def distanceP(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv((file_name), index=False)
    print("Output complete:", file_name, "rows:", len(df))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    plt.figure(figsize=(20, 10))
    if "rmse" in history.history and "val_rmse" in history.history:
        plt.plot(history.history["rmse"])
        plt.plot(history.history["val_rmse"])
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper right")
        plt.show()




## === cell 3
print("trainKaggle clean (before splitting)")
trainKaggle = clean(trainKaggle, is_train=True)



## === cell 4
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)



## === cell 5
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 6
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("validation_df Size %d" % len(validation_df))
print("test_df Size %d" % len(test_df))



## === cell 7
train_df.describe()



## === cell 8
validation_df.describe()



## === cell 9
test_df.describe()



## === cell 10
print("testKaggle clean (test-safe: no row dropping)")
_testKaggle_keys = testKaggle[["key"]].copy()
testKaggle_cleaned = clean(testKaggle, is_train=False)
if len(testKaggle_cleaned) != len(testKaggle):
    print(
        "Warning: test cleaning dropped rows (%d -> %d). Reverting to unfiltered test for submission alignment."
        % (len(testKaggle), len(testKaggle_cleaned))
    )
    testKaggle = testKaggle.copy()
else:
    testKaggle = testKaggle_cleaned



## === cell 11
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 12
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 13
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## === cell 14
train_df.describe()



## === cell 15
validation_df.describe()



## === cell 16
dropped_columns = ["passenger_count", "pickup_datetime"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)

testKaggle_for_submit = testKaggle[["key"]].copy()
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

if "key" in train_df.columns:
    train_df = train_df.drop(["key"], axis=1)
if "key" in test_df.columns:
    test_df = test_df.drop(["key"], axis=1)
if "key" in validation_df.columns:
    validation_df = validation_df.drop(["key"], axis=1)

print("Done with dropped_columns")



## === cell 17
train_labels = train_df["fare_amount"].values.astype("float32")
validation_labels = validation_df["fare_amount"].values.astype("float32")
test_labels = test_df["fare_amount"].values.astype("float32")

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 18
test_df = test_df[train_df.columns]
validation_df = validation_df[train_df.columns]
testKaggle_clean = testKaggle_clean[train_df.columns]

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

print("Dataset size (requested sample): %s" % DATASET_SIZE)
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
    shuffle=False,
)



## === cell 24
print("Model visualization skipped (keras.utils.vis_utils not available).")



## === cell 25
plot_loss_accuracy_rmse(history)



## === cell 26
prediction = model.predict(test_scaled, batch_size=128, verbose=1).astype("float32")
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1).astype(
    "float32"
)

prediction = np.clip(prediction, 0.0, None)
predictionKaggle = np.clip(predictionKaggle, 0.0, None)



## === cell 27
m = min(1000, len(test_labels), len(prediction))
mse_dbg = float(
    np.mean((test_labels[:m].reshape(-1) - prediction[:m].reshape(-1)) ** 2)
)
rmse_dbg = float(np.sqrt(mse_dbg))
print("Debug RMSE on held-out test_df (dollars, first %d): %.5f" % (m, rmse_dbg))



## === cell 28
if len(testKaggle_for_submit) != len(predictionKaggle):
    raise ValueError(
        "Submission alignment error: len(keys)=%d but len(preds)=%d"
        % (len(testKaggle_for_submit), len(predictionKaggle))
    )
output_submission(
    testKaggle_for_submit, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME
)



## === cell 29
idx = min(10000, len(prediction) - 1)
print("Sample prediction (fare):", float(prediction[idx]))
print("Sample label (fare):", float(test_labels[idx]))
