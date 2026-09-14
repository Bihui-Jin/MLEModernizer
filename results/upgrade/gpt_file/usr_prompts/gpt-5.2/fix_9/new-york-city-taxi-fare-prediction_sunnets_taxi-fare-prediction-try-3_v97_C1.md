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

4.1558

# 6. Current score

None

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 108.22836) has done: 'I (1) fix the Keras 3 incompatibilities causing early import/optimizer/visualization crashes by switching to `tf_keras` (same Keras API/semantics) and using the correct `Adam(learning_rate=...)` call, (2) fix the NYC “water mask” cleaning step so it works offline (no URL fetch) by using a local cached mask if available and otherwise skipping that filter (to keep the pipeline running), and (3) correct a couple of time-feature logic bugs that were unintentionally turning “late night/night” into always-true/always-false conditions. These changes keep your model architecture/training loop intact, but should materially improve RMSE versus the current ~15.5 by preventing broken cleaning and bad temporal flags, and ensure a valid `submissiontry_water.csv` is produced. Paths remain unchanged and the script writes a `.csv` with `key,fare_amount`.'
- What this solution (achieved 33.28518) has done: 'I fix two execution blockers: the protobuf/TensorFlow import crash at startup and the MinMaxScaler failure caused by a leftover string `key` column in the feature matrix. To keep core modeling logic intact, I only adjust imports/env vars to make TensorFlow/tf_keras load reliably in Kaggle, and I drop `key` from train/val/test feature frames right before scaling (while keeping it in `testKaggle` for submission). These changes unblock training/inference end-to-end and ensure a valid `submissiontry_water.csv` with `key,fare_amount` is written. No architecture, loss, optimizer type, or training loop changes are introduced.'
- What this solution (achieved 36.01057) has done: 'I fix the startup crash in the first cell caused by a protobuf/TensorFlow incompatibility by ensuring we import `tf_keras` first (and only import TensorFlow after), and by soft-disabling the pure-Python protobuf fallback that triggers the `MessageFactory.GetPrototype` error in newer protobuf versions. Then, to move RMSE substantially toward the target without changing the model/training loop, I correct the dataset paths to the actual Kaggle input location and add the missing distance features (haversine) that are already defined but never used, which is a minimal feature-engineering fix consistent with your existing approach. Finally, I keep the submission writing logic intact and ensure the output CSV has the required `key,fare_amount` columns and a `.csv` filename.'
- What this solution (achieved 15.14834) has done: 'I fix the runtime crash caused by importing TensorFlow/tf_keras in an environment where TensorFlow can’t load due to a protobuf binary mismatch, by switching the Keras imports to the standalone `keras` package (same Sequential/Dense API and training loop semantics). I also fix the “TRAIN_PATH not defined” cascade by ensuring the initial setup cell executes and by using the correct Kaggle file paths that exist in your environment. To keep the pipeline stable end-to-end, I make the plotting cells non-fatal (so missing display/Graphviz won’t stop submission generation) and ensure the submission writer always produces `key,fare_amount` with a `.csv` suffix. These changes are execution/stability fixes; they do not change your model architecture or training approach, and they should allow you to obtain a much better RMSE than “not yielded” by actually producing a valid submission.'
- What this solution (achieved 57.05224) has done: 'I fix the two hard runtime blockers: the protobuf/TensorFlow-related `MessageFactory.GetPrototype` crash (by switching back to `tf_keras`, which is installed and avoids the standalone-keras/TensorFlow protobuf mismatch in this environment) and the custom `rmse` metric crash (by implementing RMSE via `tf_keras.backend`/TensorFlow ops that exist). These changes keep your model architecture, optimizer type, loss, and training loop the same, but allow training/inference to run end-to-end again. To nudge RMSE materially toward the 4.1558 target without changing the modeling approach, I also ensure the script reads from the correct Kaggle input paths that exist and keeps the same feature set/cleaning flow. Finally, I keep the submission writer intact and guarantee a valid `key,fare_amount` CSV is produced.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tf_keras as keras
from tf_keras.models import Sequential, load_model
from tf_keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras import optimizers, regularizers, backend

np.random.seed(1)
try:
    keras.utils.set_random_seed(1)
except Exception:
    pass

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 1000
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

USE_LOG_TARGET = True




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    candidate_paths = [
        "../input/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "../input/new-york-city-taxi-fare-prediction/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/input/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/input/new-york-city-taxi-fare-prediction/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "./nyc_mask-74.5_-72.8_40.5_41.8.png",
    ]
    mask_path = next((p for p in candidate_paths if os.path.exists(p)), None)

    if mask_path is None:
        return df

    nyc_mask = plt.imread(mask_path)[:, :, 0] > 0.9

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
    print(" New size after NYC lang lot: %d" % len(df))

    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["dropoff_longitude"] != 0)]
    df = df[(df["dropoff_latitude"] != 0)]
    print(" New size after lang lot > 0: %d" % len(df))

    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)]
    print(" New size after lang - lot > 0.001: %d" % len(df))

    print(" New size after only NYC: %d" % len(df))
    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0)]
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
        df["pickup_datetime"], utc=True, errors="coerce"
    )
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["minute"] = df["pickup_datetime"].dt.minute
    df["second"] = df["pickup_datetime"].dt.second
    df["weekday"] = df["pickup_datetime"].dt.weekday

    df["late_night"] = df.apply(late_night, axis=1)
    df["night"] = df.apply(night, axis=1)
    df["rush_hour"] = df.apply(rush_hour, axis=1)
    return df


def add_coordinate_features(df):
    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
    df["sum_lon"] = df["pickup_longitude"] + df["dropoff_longitude"]
    df["sum_lat"] = df["pickup_latitude"] + df["dropoff_latitude"]
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["haversine_miles"] = distance(lat1, lon1, lat2, lon2)
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
    if not file_name.lower().endswith(".csv"):
        file_name = file_name + ".csv"
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows=", len(df))


def plot_loss_accuracy_rmse(history):
    try:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper right")
        plt.show()

        if "rmse" in history.history:
            plt.figure(figsize=(20, 10))
            plt.plot(history.history["rmse"])
            plt.plot(history.history.get("val_rmse", []))
            plt.title("Model rmse")
            plt.ylabel("rmse")
            plt.xlabel("epoch")
            plt.legend(["train", "test"], loc="upper right")
            plt.show()
    except Exception as e:
        print("Plotting skipped:", repr(e))




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
trainKaggle = pd.read_csv(
    TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=train_usecols
)
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
train_df.describe()




## === cell 6
test_df.describe()




## === cell 7
print("train_df clean")
train_df = clean(train_df)
test_df = clean(test_df)




## === cell 8
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)




## === cell 9
train_df.describe()




## === cell 10
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features Disabled!")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)




## === cell 11
train_df.describe()




## === cell 12
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)
print("Done with Adding features")




## === cell 13
train_df.describe()




## === cell 14
try:
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")
except Exception as e:
    print("EDA plots skipped:", repr(e))




## === cell 15
dropped_columns = ["pickup_datetime"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")




## === cell 16
train_df.shape




## === cell 17
train_df.describe()




## === cell 18
test_df.describe()




## === cell 19
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)




## === cell 20
train_df_main = train_df
validation_df_main = validation_df




## === cell 21
validation_df.describe()




## === cell 22
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

if USE_LOG_TARGET:
    train_labels = np.log1p(train_labels).astype(np.float32)
    validation_labels = np.log1p(validation_labels).astype(np.float32)
    test_labels = np.log1p(test_labels).astype(np.float32)

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")




## === cell 23
test_labels




## === cell 24
train_df.describe()




## === cell 25
validation_df.describe()




## === cell 26
test_df.describe()




## === cell 27
for _df_name, _df in [
    ("train_df", train_df),
    ("validation_df", validation_df),
    ("test_df", test_df),
]:
    if "key" in _df.columns:
        _df.drop(["key"], axis=1, inplace=True)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)




## === cell 28
test_scaled




## === cell 29
def rmse(y_true, y_pred):
    y_true = backend.cast(y_true, "float32")
    y_pred = backend.cast(y_pred, "float32")
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))




## === cell 30
checkpoint = ModelCheckpoint(filepath="my_model.keras", verbose=1, save_best_only=True)

model = Sequential()
model.add(
    Dense(
        256,
        activation="linear",
        input_dim=train_df_scaled.shape[1],
        activity_regularizer=regularizers.l1(0.01),
    )
)
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="linear"))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mean_squared_error", optimizer=adam, metrics=["mae", rmse, "mse"])

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
    callbacks=[checkpoint],
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)




## === cell 31
pass




## === cell 32
try:
    from IPython.display import SVG
    from tf_keras.utils import model_to_dot

    SVG(model_to_dot(model).create(prog="dot", format="svg"))
except Exception as e:
    print("Model visualization skipped:", repr(e))




## === cell 33
plot_loss_accuracy_rmse(history)




## === cell 34
score = model.evaluate(train_df_scaled, train_labels, verbose=1)
print(score)
print("train mean_squared_error:", score[0])
print("train mae:", score[1])
print("train rmse:", score[2])
print("train mse:", score[3])




## === cell 35
score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print(score)
print("Validation mean_squared_error:", score[0])
print("Validation mae:", score[1])
print("Validation rmse:", score[2])
print("Validation mse:", score[3])




## === cell 36
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print("Test mean_squared_error:", score[0])
print("Test mae:", score[1])
print("Test rmse:", score[2])
print("Test mse:", score[3])




## === cell 37
validation_predictions = model.predict(validation_df_scaled).flatten()

try:
    plt.scatter(validation_labels, validation_predictions)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.axis("equal")
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot(
        [validation_predictions.min(), validation_predictions.max()],
        [validation_predictions.min(), validation_predictions.max()],
        "k--",
        lw=4,
    )
    plt.show()
except Exception as e:
    print("Validation scatter skipped:", repr(e))




## === cell 38
test_predictions = model.predict(test_scaled).flatten()

try:
    plt.scatter(test_labels, test_predictions)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.axis("equal")
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot(
        [test_predictions.min(), test_predictions.max()],
        [test_predictions.min(), test_predictions.max()],
        "k--",
        lw=4,
    )
    plt.show()
except Exception as e:
    print("Test scatter skipped:", repr(e))




## === cell 39
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_labels[np.argmax(test_predictions)])
test_df.iloc[np.argmax(test_predictions)]




## === cell 40
print(np.argmin(test_predictions))
print(test_predictions[np.argmin(test_predictions)])
print(test_labels[np.argmin(test_predictions)])
test_df.iloc[np.argmin(test_predictions)]




## === cell 41
try:
    fig, ax = plt.subplots()
    ax.scatter(test_labels, test_predictions)
    ax.plot(
        [test_labels.min(), test_labels.max()],
        [test_labels.min(), test_labels.max()],
        "k--",
        lw=4,
    )
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    plt.show()
except Exception as e:
    print("Measured vs Predicted plot skipped:", repr(e))




## === cell 42
try:
    plt.figure(figsize=(20, 10))
    plt.plot(validation_labels[:100])
    plt.plot(validation_predictions[:100])
    plt.title("Prediction vs Actual")
    plt.ylabel("Fare Amount")
    plt.xlabel("Transaction")
    plt.legend(["Actual", "prediction"], loc="upper right")
    plt.show()
except Exception as e:
    print("Validation series plot skipped:", repr(e))




## === cell 43
try:
    plt.figure(figsize=(20, 10))
    plt.plot(test_labels[:100])
    plt.plot(test_predictions[:100])
    plt.title("Prediction vs Actual")
    plt.ylabel("Fare Amount")
    plt.xlabel("Transaction")
    plt.legend(["Actual", "prediction"], loc="upper right")
    plt.show()
except Exception as e:
    print("Test series plot skipped:", repr(e))




## === cell 44
error = validation_predictions - validation_labels
try:
    plt.hist(error, bins=100)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
    plt.show()
except Exception as e:
    print("Validation error hist skipped:", repr(e))




## === cell 45
error = test_predictions - test_labels
try:
    plt.hist(error, bins=50)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
    plt.show()
except Exception as e:
    print("Test error hist skipped:", repr(e))




## === cell 46
print(len(error))
errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
print(len(errorGreaterZero))
try:
    plt.hist(errorGreaterZero, bins=100)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
    plt.show()
except Exception as e:
    print("Filtered error hist skipped:", repr(e))




## === cell 47
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)




## === cell 48
predictionKaggle = np.asarray(predictionKaggle).reshape(-1, 1)
predictionKaggle = np.where(np.isfinite(predictionKaggle), predictionKaggle, 0.0)

if USE_LOG_TARGET:
    predictionKaggle = np.expm1(predictionKaggle)

predictionKaggle = np.clip(predictionKaggle, 0.0, None)

output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
print("Wrote submission to:", SUBMISSION_NAME)
