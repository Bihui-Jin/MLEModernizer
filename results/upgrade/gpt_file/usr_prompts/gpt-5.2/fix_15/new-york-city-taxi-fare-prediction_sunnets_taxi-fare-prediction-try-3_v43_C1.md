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

4.55419

# 6. Current score

None

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 212.78392) has done: 'I fix the runtime import error by switching from the legacy `keras` package to `tf_keras`, which is compatible in this Kaggle environment and avoids the protobuf `MessageFactory.GetPrototype` crash. I also fix the timezone-aware datetime conversion bug in `add_time_features` by localizing to timezone-naive before casting, keeping the same time feature logic. Finally, I fix the scaler “feature names mismatch” error by ensuring all splits (train/validation/test/testKaggle) receive identical feature engineering before dropping columns and scaling, so training, evaluation, and submission generation complete and produce a valid `submissiontry_water.csv`.'
- What this solution (achieved 12.62339) has done: 'I fix the immediate runtime crash in the first cell by making the Keras imports robust: prefer `tf_keras` (as intended) but fall back to `tensorflow.keras` when the protobuf/keras stack still triggers `MessageFactory.GetPrototype`. Then I correct a major logic issue causing the extremely poor RMSE: the code currently trains on engineered features but the `testKaggle_clean` dataframe is created from the *raw* test set (missing engineered columns), and the train/test split is reversed (training on 40% and “testing” on 50%). Finally, I keep the same model and training loop, but ensure identical feature engineering on all splits and fix the split proportions so training uses the majority of the sampled data; this should move the score substantially toward the target without changing the core approach.'
- What this solution (achieved 100.69101) has done: 'I fix the runtime crash coming from the protobuf/keras stack by removing the fallback import of `tensorflow` entirely and using `tf_keras` only (which is already installed and avoids the `MessageFactory.GetPrototype` error). I also fix the input data paths to point to the actual Kaggle files you listed (`/kaggle/input/new-york-city-taxi-fare-prediction/train.csv` and `test.csv`), since the current `../input/*.csv` paths are not valid here and can trigger unexpected behavior. Finally, I keep the model/training/feature logic the same, but make the submission writing more robust by flattening predictions to 1D to avoid shape/column issues and ensure a valid `*.csv` is always produced.'
- What this solution (achieved 174.35999) has done: 'I fix the immediate runtime crash caused by an incompatibility between `tf_keras` and the installed `protobuf` version by switching the model imports to `tensorflow.keras`, which is the stable stack in Kaggle’s TF environment for this competition. I also correct the data path to the actual mounted directory (`/kaggle/input/new-york-city-taxi-fare-prediction/`) so the script reliably reads the intended files. To move RMSE toward the target (lower is better) without changing the model/feature core logic, I fix the submission index alignment by ensuring the test set `key` is preserved exactly and predictions are written 1-to-1 in the same order. All other training, feature engineering, and network architecture remain unchanged.'
- What this solution (achieved 108.21343) has done: 'I fix the runtime crash caused by the protobuf/TensorFlow import stack by avoiding `tensorflow`/`tensorflow.keras` entirely and using the already-installed `tf_keras` backend, which prevents the `MessageFactory.GetPrototype` error in this environment. Since your current RMSE (174) is far from the target (4.55), I also apply two minimal, competition-standard data quality fixes that don’t change the model architecture/training loop: include the `key` column when reading the test set (so submission alignment is guaranteed), and remove extreme fare outliers + obvious coordinate outliers before training (while keeping the same feature engineering and scaling pipeline). These changes are directly tied to correctness and RMSE reduction on this dataset and should move the score substantially toward the target band. The script still write `submissiontry_water.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 203.08709) has done: 'I fix the immediate crash in cell 1 caused by the protobuf/Keras incompatibility by avoiding the `tf_keras` stack and using the stable `tensorflow.keras` API when available (falling back to `tf_keras` only if TF import fails). This is a runtime fix only and keeps your exact model architecture, loss, training loop, and feature engineering unchanged. I also make the random seeding robust across both backends to preserve determinism. Everything else (data paths, cleaning rules, engineered features, scaling, training/evaluation, and submission formatting) is left intact so the pipeline runs end-to-end and writes a valid `submissiontry_water.csv`.'
- What this solution (achieved 65.1839) has done: 'I fix the runtime crash in the first cell caused by the protobuf/TensorFlow import path by avoiding `tensorflow` entirely and using the already-installed `tf_keras` backend consistently (this is the same model API but prevents the `MessageFactory.GetPrototype` error). I also remove the incorrect `dtype` mapping for `pickup_datetime` in the test read (the test file has no `fare_amount`, and forcing `pickup_datetime` to a numeric dtype can silently coerce to NaNs and destroy time features), while keeping the same columns and feature engineering. Finally, I ensure all numeric feature columns are coerced to numeric after feature engineering (guarding against string contamination from datetime handling) so scaling/training/prediction remain stable and the submission writes correctly.'

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
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import backend

_KERAS_BACKEND = "tensorflow.keras"

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 1000
EPOCHS = 30
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

np.random.seed(1)
tf.random.set_seed(1)

if not os.path.exists(TRAIN_PATH):
    alt_train = "/kaggle/input/train.csv"
    if os.path.exists(alt_train):
        TRAIN_PATH = alt_train

if not os.path.exists(TEST_PATH):
    alt_test = "/kaggle/input/test.csv"
    if os.path.exists(alt_test):
        TEST_PATH = alt_test

print("Keras backend:", _KERAS_BACKEND)
print("Using TRAIN_PATH:", TRAIN_PATH, "exists:", os.path.exists(TRAIN_PATH))
print("Using TEST_PATH:", TEST_PATH, "exists:", os.path.exists(TEST_PATH))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
datatypes_train = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

datatypes_test = {
    "key": "str",
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
    dtype=datatypes_train,
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
    dtype=datatypes_test,
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



## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.20, random_state=1)



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
    h = row["hour"]
    return 1 if (h <= 3 or h >= 22) else 0


def night(row):
    h = row["hour"]
    wd = row["weekday"]
    return 1 if (h >= 20 or h <= 5) and (wd < 5) else 0


def rush_hour(row):
    h = row["hour"]
    wd = row["weekday"]
    return 1 if (16 <= h <= 20) and (wd < 5) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def _remove_near_coord(df, lat_col, lon_col, coord_lat, coord_lon, eps=0.002):
    return df[
        ~(
            (df[lat_col].between(coord_lat - eps, coord_lat + eps))
            & (df[lon_col].between(coord_lon - eps, coord_lon + eps))
        )
    ]


def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    if "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 250)]
        print(" New size after removing extreme fare outliers: %d" % len(df))

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
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)

    df = _remove_near_coord(
        df, "pickup_latitude", "pickup_longitude", nyc_coord[0], nyc_coord[1], eps=0.002
    )
    df = _remove_near_coord(
        df,
        "dropoff_latitude",
        "dropoff_longitude",
        nyc_coord[0],
        nyc_coord[1],
        eps=0.002,
    )
    print(" New size after NY coord filter: %d" % len(df))

    df = _remove_near_coord(
        df, "pickup_latitude", "pickup_longitude", fk_coord[0], fk_coord[1], eps=0.002
    )
    df = _remove_near_coord(
        df, "dropoff_latitude", "dropoff_longitude", fk_coord[0], fk_coord[1], eps=0.002
    )
    print(" New size after jfk coord filter: %d" % len(df))

    df = _remove_near_coord(
        df, "pickup_latitude", "pickup_longitude", ewr_coord[0], ewr_coord[1], eps=0.002
    )
    df = _remove_near_coord(
        df,
        "dropoff_latitude",
        "dropoff_longitude",
        ewr_coord[0],
        ewr_coord[1],
        eps=0.002,
    )
    print(" New size after ewr coord filter: %d" % len(df))

    df = _remove_near_coord(
        df, "pickup_latitude", "pickup_longitude", lga_coord[0], lga_coord[1], eps=0.002
    )
    df = _remove_near_coord(
        df,
        "dropoff_latitude",
        "dropoff_longitude",
        lga_coord[0],
        lga_coord[1],
        eps=0.002,
    )
    print(" New size after lga coord filter: %d" % len(df))

    df = _remove_near_coord(
        df, "pickup_latitude", "pickup_longitude", sol_coord[0], sol_coord[1], eps=0.002
    )
    df = _remove_near_coord(
        df,
        "dropoff_latitude",
        "dropoff_longitude",
        sol_coord[0],
        sol_coord[1],
        eps=0.002,
    )
    print(" New size after sol coord filter: %d" % len(df))

    print("Old size: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    dt = dt.dt.tz_localize(None)

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday

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
    pred_1d = np.asarray(prediction).reshape(-1).astype(np.float32)
    keys = raw_test[id_column].astype(str).values
    if len(keys) != len(pred_1d):
        raise ValueError(
            f"Submission length mismatch: keys={len(keys)} preds={len(pred_1d)}"
        )
    df = pd.DataFrame({id_column: keys, prediction_column: pred_1d})
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df), "cols:", list(df.columns))


def rmse(y_true, y_pred):
    sq = backend.square(y_pred - y_true)
    return backend.sqrt(backend.mean(sq, axis=-1))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 8))
    plt.plot(history.history.get("loss", []))
    plt.plot(history.history.get("val_loss", []))
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper right")
    plt.show()

    plt.figure(figsize=(20, 8))
    if "rmse" in history.history and "val_rmse" in history.history:
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

print("test_df clean")
test_df = clean(test_df)

print("testKaggle clean")
testKaggle = clean(testKaggle)



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
def coerce_numeric_features(df, exclude_cols):
    for c in df.columns:
        if c in exclude_cols:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


numeric_exclude = {"key", "pickup_datetime"}  # will be dropped later
train_df = coerce_numeric_features(train_df, numeric_exclude)
validation_df = coerce_numeric_features(validation_df, numeric_exclude)
test_df = coerce_numeric_features(test_df, numeric_exclude)
testKaggle = coerce_numeric_features(testKaggle, numeric_exclude)



## === cell 18
dropped_columns = ["passenger_count", "pickup_datetime"]

train_df = train_df.drop(dropped_columns + ["key"], axis=1)
test_df = test_df.drop(dropped_columns + ["key"], axis=1)
validation_df = validation_df.drop(dropped_columns + ["key"], axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")



## === cell 19
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 20
train_df.shape



## === cell 21
test_df.shape



## === cell 22
validation_df.shape



## === cell 23
feature_cols = list(train_df.columns)
validation_df = validation_df[feature_cols]
test_df = test_df[feature_cols]
testKaggle_clean = testKaggle_clean[feature_cols]

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)



## === cell 24
model = Sequential()
model.add(
    Dense(
        256,
        activation="linear",
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
model.compile(
    loss="mean_squared_error", optimizer=adam, metrics=["mae", "accuracy", rmse, "mse"]
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



## === cell 25
print(
    "Model visualization skipped (keras.utils.vis_utils not available in this environment)."
)



## === cell 26
plot_loss_accuracy_rmse(history)



## === cell 27
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print("Test loss:", score[0])
print("Test mae:", score[1])



## === cell 28
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## === cell 29
predictionKaggle = np.maximum(predictionKaggle, 0.0)



## === cell 30
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)



## === cell 31
idx = min(10000, len(prediction) - 1)
print(prediction[idx])
print(test_labels[idx])
