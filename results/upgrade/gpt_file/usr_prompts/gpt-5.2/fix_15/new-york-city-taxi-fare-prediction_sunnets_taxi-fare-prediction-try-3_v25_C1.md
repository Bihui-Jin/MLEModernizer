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

5.60311

# 6. Current score

None

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.24492) has done: 'I fix the runtime errors caused by (1) Keras 3 optimizer API changes, (2) attempting to load a remote NYC land/water mask (no internet / `plt.imread` can’t read URLs), and (3) a few logic/shape issues that currently prevent the notebook from completing and writing a valid CSV. To improve RMSE toward your target, I keep the same model/training loop but correct two feature bugs (`night`/`late_night` conditions and datetime parsing) and ensure coordinate/distance features are actually applied consistently. I also make the file paths robust for Kaggle (`/kaggle/input/...`) while preserving your intended filenames and output submission schema. Finally, I remove/guard visualization-only code that fails under this environment and fix the MSE debug line that uses mismatched slice sizes.'
- What this solution (achieved 876.7039) has done: 'I fix the runtime error caused by Keras 3 backend API changes by rewriting the custom `rmse` metric to use `keras.ops`, which is the supported backend-agnostic math layer. I also make the initial Keras import robust to the protobuf-related `MessageFactory.GetPrototype` crash by preferring `tf_keras` (available in your environment) while keeping the same model, layers, optimizer, and training loop semantics. Finally, I guard plotting so it doesn’t crash when training fails, and ensure the submission is always written with the required `key,fare_amount` columns and a `.csv` suffix.'
- What this solution (achieved 268.42615) has done: 'I fix the early crash (`MessageFactory.GetPrototype`) by making the imports consistently use `tf_keras` (and its submodules) instead of mixing `tf_keras` with standalone `keras`, which is what triggers the protobuf incompatibility in this environment. I also correct the train/test file path to match your provided dataset location (`/kaggle/input/new-york-city-taxi-fare-prediction/...`) so the code reliably reads the intended files. To move RMSE strongly toward your target (current is catastrophically bad), I make two minimal feature-logic fixes that preserve your approach: compute `night` correctly and make datetime parsing robust (so time features aren’t mostly NaN/garbage). Finally, I keep the model/training loop intact and ensure the submission is written as a valid `.csv` with `key,fare_amount`.'
- What this solution (achieved 224.8797) has done: 'I fix the protobuf crash causing `MessageFactory.GetPrototype` by avoiding the standalone `tensorflow` import and using `tf_keras.backend` for the custom RMSE metric (this keeps the same loss/metric semantics but prevents the environment mismatch). I also correct the path candidates to include the actual Kaggle mount you listed (`/kaggle/data/...`) so the code reliably reads the dataset. To move RMSE strongly toward your target (current score is catastrophically high), I fix a major logic bug in the cleaning step where “airport removal” is implemented as an equality check (it currently drops almost nothing); switching to a small-radius “near airport” filter preserves your intended logic but makes the cleaned training data consistent and improves generalization. Finally, I keep the same model/training loop and ensure the submission CSV is written with the required `key,fare_amount` columns and a `.csv` suffix.'
- What this solution (achieved 123.98317) has done: 'I fix the startup crash (`MessageFactory.GetPrototype`) by avoiding `tf_keras` imports entirely and using `tensorflow.keras`, which is the stable backend in Kaggle for this competition. I keep your exact model architecture, loss, and training loop intact, but make the custom `rmse` metric backend-safe using TensorFlow ops. I also make the debug MSE line use TF/Keras APIs consistently to prevent shape/type issues, and ensure the submission is always written as `key,fare_amount` to a `.csv` filename. These changes are execution/stability fixes and should also move the RMSE dramatically down from the current broken run.'
- What this solution (achieved 133.68328) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by avoiding `tensorflow`/`tensorflow.keras` entirely and consistently using the already-installed `tf_keras` stack (including optimizer, layers, losses) so the notebook runs end-to-end in this Kaggle environment. I keep your exact model architecture, loss, and training loop, only updating the custom `rmse` metric and the debug MSE calculation to use `tf_keras.backend` ops to prevent backend/type mismatches. I also ensure the train/test CSV paths resolve to the provided `/kaggle/input/...` or `/kaggle/data/...` locations and that the submission is always written as a valid `.csv` with columns `key,fare_amount`. These changes are primarily stability fixes and should move RMSE substantially down from the current broken run toward your target.'
- What this solution (achieved 41.40529) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by avoiding the incompatible `tf_keras` stack in this environment and switching imports to the stable `tensorflow.keras` API while keeping your model, optimizer, and training loop unchanged. I also make the custom `rmse` metric backend-safe using TensorFlow ops (same RMSE semantics) and keep the debug MSE computation consistent with TensorFlow tensors to prevent shape/type issues. To move the RMSE strongly toward your target (current score is catastrophically high), I make one minimal but high-impact data IO fix: ensure the training read includes the `key` column so later feature/submit alignment doesn’t silently break, and I ensure the submission is always written as a valid `.csv` with `key,fare_amount`. All other core feature engineering and the network architecture/training settings remain the same.'
- What this solution (achieved 60.58413) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by removing the incompatible TensorFlow import and running the exact same Keras model/training loop using the already-installed `tf_keras` stack consistently. I also make the RMSE metric and the debug MSE computation use `tf_keras.backend` ops to avoid cross-backend tensor/type issues. These changes are execution/stability fixes (core model, features, training settings remain the same) and should also bring RMSE down from the currently broken run by ensuring training/inference actually completes and the submission is generated correctly.'
- What this solution (achieved 145.19184) has done: 'I fix the startup crash caused by importing `tensorflow` in this Kaggle environment (the `MessageFactory.GetPrototype` protobuf error) by removing that optional TensorFlow seed block and relying on NumPy seeding only, keeping the same `tf_keras` model/training logic. Then I make one minimal, score-improving correction that preserves your intended semantics: ensure `night()` uses the correct boolean grouping so it doesn’t incorrectly flag almost all early-morning rides as “night” regardless of weekday (a known high-impact feature bug for this dataset). Finally, I keep the rest of the pipeline unchanged and ensure the submission is written as a valid `.csv` with `key,fare_amount`.'
- What this solution (achieved 18.28001) has done: 'I fix the immediate crash in cell 1 caused by importing `tf_keras` in this Kaggle environment (protobuf `MessageFactory.GetPrototype` incompatibility) by switching the Keras stack imports to the stable `tensorflow.keras` API, while keeping the same model architecture, loss, optimizer type, and training loop. I also make the custom `rmse` metric use TensorFlow ops so it works reliably with `tensorflow.keras` tensors. Additionally, I correct the `night()` feature logic to avoid incorrectly labeling most early-morning rides as “night” on weekends (this is a small, intended-semantics bug fix that should significantly improve RMSE toward your target). Finally, I keep the submission writing unchanged but ensure it always produces a valid `.csv` with `key,fare_amount`.'
- What this solution (achieved 182.44127) has done: 'I fix the hard crash coming from importing TensorFlow (protobuf `MessageFactory.GetPrototype` incompatibility) by switching the Keras stack to the already-installed `tf_keras` consistently, and re-implement the RMSE metric using `tf_keras.backend` ops so compilation/training works end-to-end. I keep the same model architecture, loss, optimizer type, training loop, and feature set, but replace the slow/bug-prone row-wise `.apply()` time-feature flags with equivalent vectorized logic (same semantics) so the intended time features actually compute correctly and faster, which should improve RMSE toward your target. I also ensure the script always writes a valid `key,fare_amount` submission CSV with the requested filename. These changes are directly aimed at unblocking execution and improving feature correctness without changing the core approach.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

TRAIN_CANDIDATES = [
    "/kaggle/data/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/data/train.csv",
    "/kaggle/input/train.csv",
    "../input/new-york-city-taxi-fare-prediction/train.csv",
    "../input/train.csv",
]
TEST_CANDIDATES = [
    "/kaggle/data/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/data/test.csv",
    "/kaggle/input/test.csv",
    "../input/new-york-city-taxi-fare-prediction/test.csv",
    "../input/test.csv",
]


def _pick_existing(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


TRAIN_PATH = _pick_existing(TRAIN_CANDIDATES)
TEST_PATH = _pick_existing(TEST_CANDIDATES)

SUBMISSION_NAME = "submissiontry_water.csv"
if not SUBMISSION_NAME.lower().endswith(".csv"):
    SUBMISSION_NAME = SUBMISSION_NAME + ".csv"

BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

np.random.seed(1)
try:
    tf.random.set_seed(1)
except Exception:
    pass

print("Using TRAIN_PATH:", TRAIN_PATH)
print("Using TEST_PATH :", TEST_PATH)
print("Will write submission:", SUBMISSION_NAME)



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

usecols_train = [
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
    TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=usecols_train
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
def remove_datapoints_from_water(df):
    return df


def late_night(row):
    h = int(row["hour"])
    return 1 if (h <= 3 or h >= 22) else 0


def night(row):
    h = int(row["hour"])
    wd = int(row["weekday"])
    return 1 if ((h >= 20) or (h <= 6)) and (wd < 5) else 0


def rush_hour(row):
    h = int(row["hour"])
    wd = int(row["weekday"])
    return 1 if (16 <= h <= 20) and (wd < 5) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def _deg_distance_approx(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


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

    fk_coord = (40.639722, -73.778889)  # JFK
    ewr_coord = (40.6925, -74.168611)  # EWR
    lga_coord = (40.77725, -73.872611)  # LGA
    sol_coord = (40.6892, -74.0445)  # Statue of Liberty

    r = 0.03

    def _not_near(coord_lat, coord_lon, lat_col, lon_col):
        d = _deg_distance_approx(df[lat_col], df[lon_col], coord_lat, coord_lon)
        return d > r

    before = len(df)
    df = df[_not_near(fk_coord[0], fk_coord[1], "pickup_latitude", "pickup_longitude")]
    df = df[
        _not_near(fk_coord[0], fk_coord[1], "dropoff_latitude", "dropoff_longitude")
    ]
    print(" New size after jfk airport: %d (dropped %d)" % (len(df), before - len(df)))

    before = len(df)
    df = df[
        _not_near(ewr_coord[0], ewr_coord[1], "pickup_latitude", "pickup_longitude")
    ]
    df = df[
        _not_near(ewr_coord[0], ewr_coord[1], "dropoff_latitude", "dropoff_longitude")
    ]
    print(" New size after ewr airport: %d (dropped %d)" % (len(df), before - len(df)))

    before = len(df)
    df = df[
        _not_near(lga_coord[0], lga_coord[1], "pickup_latitude", "pickup_longitude")
    ]
    df = df[
        _not_near(lga_coord[0], lga_coord[1], "dropoff_latitude", "dropoff_longitude")
    ]
    print(" New size after lga airport: %d (dropped %d)" % (len(df), before - len(df)))

    before = len(df)
    df = df[
        _not_near(sol_coord[0], sol_coord[1], "pickup_latitude", "pickup_longitude")
    ]
    df = df[
        _not_near(sol_coord[0], sol_coord[1], "dropoff_latitude", "dropoff_longitude")
    ]
    print(" New size after sol removed: %d (dropped %d)" % (len(df), before - len(df)))

    print("Old size: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def clean_test(df):
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

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    print("Old size: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)

    year = dt.dt.year.fillna(0).astype("int16")
    month = dt.dt.month.fillna(0).astype("int8")
    day = dt.dt.day.fillna(0).astype("int8")
    hour = dt.dt.hour.fillna(0).astype("int8")
    weekday = dt.dt.weekday.fillna(0).astype("int8")

    df["year"] = year
    df["month"] = month
    df["day"] = day
    df["hour"] = hour
    df["weekday"] = weekday
    df["pickup_datetime"] = df["pickup_datetime"].astype(str)

    h = hour.astype("int16")
    wd = weekday.astype("int16")

    df["night"] = (((h >= 20) | (h <= 6)) & (wd < 5)).astype("int8")
    df["late_night"] = ((h <= 3) | (h >= 22)).astype("int8")
    df["rush_hour"] = ((h >= 16) & (h <= 20) & (wd < 5)).astype("int8")
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
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper right")
    plt.show()

    if "accuracy" in history.history and "val_accuracy" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper right")
        plt.show()

    if "rmse" in history.history and "val_rmse" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["rmse"])
        plt.plot(history.history["val_rmse"])
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper right")
        plt.show()




## === cell 9
print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)

print("testKaggle clean (no fare filtering)")
testKaggle = clean_test(testKaggle)



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
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 13
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 14
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

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
    "key",
]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns, axis=1)

print("Done with dropped_columns")



## === cell 18
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 19
train_df.shape



## === cell 20
test_df.shape



## === cell 21
validation_df.shape



## === cell 22
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)

testKaggle_clean = testKaggle_clean.reindex(columns=train_df.columns)
testKaggle_scaled = scaler.transform(testKaggle_clean)




## === cell 23
def rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=-1))




## === cell 24
model = Sequential()
model.add(
    Dense(
        512,
        activation="relu",
        input_dim=train_df_scaled.shape[1],
        activity_regularizer=regularizers.l1(0.01),
    )
)
model.add(Dropout(0.5))
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



## === cell 25
print(
    "Model trained. (Visualization skipped if matplotlib backend/headless issues occur.)"
)



## === cell 26
if "history" in globals() and history is not None:
    try:
        plot_loss_accuracy_rmse(history)
    except Exception as e:
        print("Plot skipped due to error:", repr(e))



## === cell 27
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## === cell 28
n = min(len(test_labels), len(prediction), 1000)
y_true_dbg = np.asarray(test_labels[:n], dtype="float32")
y_pred_dbg = np.asarray(prediction[:n].reshape(-1), dtype="float32")
debug_mse_val = float(np.mean((y_pred_dbg - y_true_dbg) ** 2))
print("Debug MSE (first %d): %s" % (n, debug_mse_val))

predictionKaggle = np.asarray(predictionKaggle).reshape(-1)
predictionKaggle = np.nan_to_num(
    predictionKaggle, nan=11.35, posinf=11.35, neginf=11.35
)
predictionKaggle = np.clip(predictionKaggle, 0.0, None).astype(np.float32)

output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)

print("Sample predictions:", predictionKaggle[:5])
print("Submission file exists:", os.path.exists(SUBMISSION_NAME), "->", SUBMISSION_NAME)
