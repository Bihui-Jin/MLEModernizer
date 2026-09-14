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

4.59075

# 6. Current score

5.8895

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.25184) has done: 'I fix the two root runtime blockers: the protobuf/Keras import crash in the first cell and the timezone-aware datetime conversion that breaks `add_time_features`. Then I make the feature engineering consistent across train/validation/test (the scaler error comes from missing time columns in validation/test because feature creation previously failed), without changing the model architecture or training loop. Finally, I fix the data-paths to match your Kaggle filesystem and ensure the submission writer produces a valid `key,fare_amount` CSV. These changes are score-neutral aside from allowing the intended pipeline to run end-to-end.'
- What this solution (achieved 5.88933) has done: 'I fix the Keras/protobuf import crash by switching to the installed `tf_keras` backend (TensorFlow Keras) while keeping the exact same model architecture and training loop. I also fix the custom RMSE metric by implementing it with TensorFlow ops instead of `keras.api.backend`, which is missing `sqrt` in this environment. To move the score toward your target (lower RMSE), I correct the `night`/`late_night` logical conditions which currently always return 1 and therefore add noisy, incorrect features; this is a minimal logic fix in feature engineering that should substantially reduce error without changing the overall approach. Finally, I keep the same output path/name but ensure the submission is always written as a valid `key,fare_amount` CSV.'
- What this solution (achieved 68.87975) has done: 'I fix the two runtime blockers preventing an end-to-end run and submission: (1) the protobuf/TensorFlow import crash by removing the forced pure-python protobuf implementation, and (2) the scaler failure caused by leaving the string `key` column inside the model feature matrices. I keep the model architecture, optimizer, loss, epochs, and training loop unchanged, and only adjust the data prep to ensure train/validation/test all have identical numeric feature columns. Finally, I ensure the script always writes a valid `key,fare_amount` submission CSV with the required name and columns.'
- What this solution (achieved 21.47216) has done: 'I fix the immediate runtime crash coming from an incompatible protobuf/TensorFlow combination by forcing the pure-Python protobuf implementation before importing TensorFlow (this is a minimal environment fix and does not change your model/logic). Then I correct the two time-of-day feature functions (`night` and `late_night`) whose current boolean conditions make them almost always 1, which adds strong noise and is the most likely cause of the very high RMSE; this keeps the same feature set and pipeline but fixes the intended logic. Finally, I keep the same training loop and submission writing, just ensuring the submission file is always produced as `key,fare_amount` with the `.csv` suffix.'
- What this solution (achieved 69.93894) has done: 'I fix the immediate runtime blocker in the first cell by removing the protobuf pure-Python override that is triggering the `MessageFactory.GetPrototype` AttributeError in this Kaggle environment, while keeping the same TensorFlow + `tf_keras` model, training loop, and features. I also make the input paths robust by falling back to `/kaggle/input/...` if the currently set dataset path doesn’t exist (score-neutral, prevents file-not-found). Finally, I add a small safety check to ensure the feature columns are aligned across train/validation/test before scaling (prevents subtle column-mismatch bugs and keeps semantics identical) and keep the submission writer producing a valid `key,fare_amount` CSV.'
- What this solution (achieved 74.32566) has done: 'I fix the runtime crash in the first cell caused by an incompatible TensorFlow/protobuf combination by switching the TensorFlow import to a safe, installed backend (`tf_keras`) without changing the model architecture or training loop. I also ensure the script doesn’t fail early from that import, so it can complete training, inference, and always write a valid `key,fare_amount` submission CSV with a `.csv` suffix. No feature set, scaling, loss, epochs, or optimizer settings are changed, so this should be score-improving only insofar as it restores the intended pipeline to run reliably end-to-end.'
- What this solution (achieved 74.35762) has done: 'I fix the runtime crash in the first cell by removing the protobuf pure-Python override that’s incompatible with the Kaggle TensorFlow/protobuf build and triggers `MessageFactory.GetPrototype` errors. I keep your exact model architecture, optimizer, epochs, batch size, training loop, and feature set, but I make `add_time_features` robust to invalid datetimes so it can’t produce NaNs that later break scaling/training. To move the RMSE down toward your target with a minimal, semantics-preserving fix, I correct the `night` feature logic (it currently returns 1 for every hour due to an always-true condition), which otherwise injects strong noise. Finally, I keep the same submission writing but ensure the output is always a valid `key,fare_amount` CSV with a `.csv` suffix.'
- What this solution (achieved 74.37025) has done: 'I fix the TensorFlow/protobuf import crash in the first cell by avoiding the direct `tensorflow` import and using the installed `tf_keras` (TensorFlow Keras) backend consistently, since the crash happens before any model code runs. I keep the exact model architecture, optimizer, epochs, batch size, and training loop unchanged, and only adjust imports plus the custom `rmse` metric to use `tf_keras.backend` so it works without `tensorflow` being imported. I also remove the incorrect extra train/test split that was training on only 40% of the sampled data (a logic bug that severely hurts RMSE) while preserving the same validation split procedure. Finally, I keep the same submission writer but ensure the output is always written as a valid `key,fare_amount` CSV with a `.csv` suffix.'
- What this solution (achieved 69.37789) has done: 'I fix the immediate runtime crash in the first cell caused by the protobuf/TensorFlow backend incompatibility by explicitly forcing the pure-Python protobuf implementation *before* importing `tf_keras`, which is the minimal environment-only change needed to run end-to-end. I keep your model architecture, optimizer, epochs, batch size, training loop, and overall feature pipeline unchanged, but I remove the accidental second train/test split that was shrinking your effective training set and hurting RMSE substantially. I also make the RMSE metric use TensorFlow ops via `tf_keras` (so it works reliably in this environment) and ensure the submission is always written as a valid `key,fare_amount` CSV with a `.csv` suffix.'
- What this solution (achieved 5.8895) has done: 'I fix the runtime crash happening before any training by removing the protobuf pure-Python override that’s incompatible with the TensorFlow stack in this Kaggle image, while keeping the same `tf_keras` model/loop. I also fix a critical logic bug where `test_df` is incorrectly set to a copy of `validation_df` (so you’re “testing” on validation instead of holding out a real test split), which heavily inflates RMSE; this keeps the same splitting approach but makes it correct. Finally, I make the custom RMSE metric use TensorFlow ops (available via `tf_keras`) instead of backend calls that can be missing, and keep submission writing unchanged but guaranteed to produce a valid `key,fare_amount` CSV.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, BatchNormalization
from tf_keras import regularizers
from tf_keras.optimizers import Adam

import tensorflow as tf

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/input/test.csv"

SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

np.random.seed(1)
try:
    keras.utils.set_random_seed(1)
except Exception:
    pass

print("Using TRAIN_PATH:", TRAIN_PATH)
print("Using TEST_PATH:", TEST_PATH)



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
train_df = trainKaggle.copy()
train_df, temp_df = train_test_split(train_df, test_size=0.20, random_state=1)
validation_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=1)



## === cell 3
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("validation_df Size %d" % len(validation_df))
print("test_df Size %d" % len(test_df))



## === cell 4
train_df.describe()



## === cell 5
validation_df.describe()



## === cell 6
test_df.describe()
pass




## === cell 7
def remove_datapoints_from_water(df):
    return df.copy()


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

    df = df[(0.99 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    print(" New size after removing outliers: %d" % len(df))

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


def late_night(row):
    h = int(row["hour"])
    return 1 if (h <= 3 or h >= 23) else 0


def night(row):
    h = int(row["hour"])
    return 1 if (h >= 20 or h <= 6) else 0


def rush_hour(row):
    return (
        1
        if ((row["hour"] <= 20) and (row["hour"] >= 16) and (row["weekday"] < 5))
        else 0
    )


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    if dt.isna().any():
        fill_val = dt.dropna().median()
        if pd.isna(fill_val):
            fill_val = pd.Timestamp("2010-01-01")
        dt = dt.fillna(fill_val)

    df["year"] = dt.dt.year.astype("int16")
    df["month"] = dt.dt.month.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["weekday"] = dt.dt.weekday.astype("int8")

    df["pickup_datetime"] = dt.astype(str)

    df["night"] = df.apply(lambda x: night(x), axis=1)
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1)
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1)
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
    df = pd.DataFrame({id_column: raw_test[id_column].values, prediction_column: pred})
    if not file_name.lower().endswith(".csv"):
        file_name = file_name + ".csv"
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows=", len(df))
    return file_name


def rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=-1))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history.get("loss", []))
    plt.plot(history.history.get("val_loss", []))
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()

    if "accuracy" in history.history and "val_accuracy" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper right")
        plt.show()

    if "rmse" in history.history and "val_rmse" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["rmse"])
        plt.plot(history.history["val_rmse"])
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper right")
        plt.show()




## === cell 8
print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)



## === cell 9
train_df.describe()



## === cell 10
validation_df.describe()



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
print("test_df add_coordinate_features Disabled!")
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
dropped_columns = ["key", "passenger_count", "pickup_datetime"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns, axis=1)

print("Done with dropped_columns")



## === cell 17
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 18
train_df.shape



## === cell 19
test_df.shape



## === cell 20
validation_df.shape



## === cell 21
train_cols = list(train_df.columns)
validation_df = validation_df.reindex(columns=train_cols)
test_df = test_df.reindex(columns=train_cols)
testKaggle_clean = testKaggle_clean.reindex(columns=train_cols)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)



## === cell 22
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
model.add(Dense(256, activation="relu"))
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

adam = Adam(learning_rate=LEARNING_RATE)
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



## === cell 23
print(
    "Model visualization skipped (keras.utils.vis_utils not available in this environment)."
)



## === cell 24
plot_loss_accuracy_rmse(history)



## === cell 25
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## === cell 26
predictionKaggle = np.clip(predictionKaggle, 0.0, 500.0)



## === cell 27
SUBMISSION_NAME = output_submission(
    testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME
)



## === cell 28
print(np.asarray(predictionKaggle[:10]).reshape(-1))
print("Wrote submission to:", SUBMISSION_NAME)
print(
    "Submission exists:",
    os.path.exists(SUBMISSION_NAME),
    "bytes:",
    os.path.getsize(SUBMISSION_NAME) if os.path.exists(SUBMISSION_NAME) else None,
)
