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

4.35348

# 6. Current score

7.93827

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.39169) has done: 'I fix the runtime errors caused by (1) incorrect Kaggle file paths, (2) trying to read an NYC land/water mask from a URL (no internet / unsupported by `plt.imread`), and (3) Keras 3 API changes (`optimizers.adam`, missing `vis_utils`). I keep your model architecture and training loop intact, but correct the optimizer construction and ensure the model is actually compiled and evaluated on the scaled features. To move RMSE toward your target (lower is better) with minimal semantic change, I also fix clear logic bugs in the time-feature helpers (they currently always return 1/0 incorrectly), and ensure datetime parsing works with the dataset’s format. Finally, the script always write a valid `submissiontry_water.csv` with columns `key,fare_amount`.'
- What this solution (achieved 351.18649) has done: 'I fix the runtime error by replacing the deprecated `keras.backend.sqrt/mean/square` usage with a Keras 3–compatible RMSE metric implemented via `tf.math`, which unblocks `model.fit()` and downstream cells (`history`, `evaluate`, `predict`). I also fix the initial import crash (`MessageFactory ... GetPrototype`) by avoiding the standalone `keras` package and using `tf_keras` (Kaggle’s stable TF-backed Keras) while keeping the exact same model architecture, optimizer, and training loop semantics. Finally, I correct the Kaggle data paths to the provided `/kaggle/data/...` locations so the script reliably reads data and always writes a valid `submissiontry_water.csv` with columns `key,fare_amount`. These changes are execution-critical and should also improve RMSE substantially versus a run that never trained correctly.'
- What this solution (achieved 1291.66504) has done: 'I fix the import/runtime crash in the first cell by avoiding the protobuf-triggering `tf_keras` import and instead using `tensorflow.keras`, which is stable on Kaggle and keeps your model, layers, and training loop unchanged. I also fix the file paths so `TRAIN_PATH/TEST_PATH` actually exist in your environment, which unblocks the rest of the pipeline and ensures a submission CSV is always written. To improve score toward your target (lower RMSE), I preserve all core modeling logic but make sure datetime parsing is robust and time-feature columns are generated consistently (no NaT-induced failures), and I clip negative predictions to 0 (fares can’t be negative), which is a legitimate metric-aligned post-processing step. Finally, I ensure the submission uses exactly `key,fare_amount` and writes `submissiontry_water.csv` in the working directory.'
- What this solution (achieved 214.88983) has done: 'I fix the TensorFlow/protobuf import crash by forcing use of the stable `tf_keras` backend (and disabling C++ protobuf implementation) before importing TensorFlow/Keras, which unblocks the whole pipeline. I also correct the Kaggle file paths to the ones that actually exist in your environment (`/kaggle/data/...`) so data loads reliably. To move RMSE down substantially toward your target without changing the model/training core, I fix the internal split bug that was discarding most of the held-out set (leading to effectively untrained/garbage scoring) and ensure scaling/prediction alignment stays consistent. Finally, I keep the submission writer unchanged but ensure it always outputs a valid `submissiontry_water.csv` with `key,fare_amount`.'
- What this solution (achieved 2203.82083) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by avoiding the standalone `tf_keras` package and using `tensorflow.keras`, which is the stable TF-backed Keras available on Kaggle; this is execution-critical and keeps your model architecture/training loop the same. I also correct the RMSE metric to return a scalar (batch-wise mean) so Keras metrics aggregate properly; the previous axis-based reduction can lead to inconsistent logging/evaluation. To move RMSE substantially toward your target (lower is better) without changing the core approach, I fix a major logic bug: you trained on scaled features but predicted Kaggle test using a scaler fit on train while `testKaggle_clean` had a different column set/order risk—so I enforce identical feature columns/order for train/val/test/Kaggle before scaling. Finally, I keep the same submission writer but ensure it always writes a valid `submissiontry_water.csv` with `key,fare_amount`.'
- What this solution (achieved 8.11793) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation and importing TensorFlow in a safer order; this is execution-critical and score-neutral. I also correct the data path constants to use the provided `/kaggle/input/...` locations (your environment shows files under `/kaggle/input/`, not `/kaggle/data/`), preventing silent file-missing issues. Finally, I keep your model/training loop and feature engineering intact, but add a small, metric-aligned post-processing step to clip overly large predictions to a reasonable upper bound (matching your training fare filter), which should substantially reduce RMSE from catastrophic outliers and move the score toward the target band.'
- What this solution (achieved 7.93827) has done: 'I fix the import-time TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation early and by importing the stable `tf_keras` backend instead of `tensorflow.keras`, which avoids the `MessageFactory.GetPrototype` error in this environment. I also correct the Kaggle file paths to the actual available `/kaggle/input/new-york-city-taxi-fare-prediction/...` files (with a safe fallback to `/kaggle/input/...`) so data loads reliably. These changes are execution-critical and score-neutral; your current high RMSE is largely explained by the script not running end-to-end. Finally, I keep your model, features, training loop, and submission format unchanged so the score should move down toward the target simply by training/predicting successfully.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, BatchNormalization
from tf_keras import optimizers, regularizers

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/input/test.csv"

SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

np.random.seed(1)
tf.random.set_seed(1)

print("Using TRAIN_PATH:", TRAIN_PATH, "exists:", os.path.exists(TRAIN_PATH))
print("Using TEST_PATH:", TEST_PATH, "exists:", os.path.exists(TEST_PATH))
print("TF version:", tf.__version__)
print("Keras backend:", keras.__name__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def remove_datapoints_from_water(df):
    return df


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
    return 1 if (row["hour"] <= 3) else 0


def night(row):
    return (
        1 if ((row["hour"] >= 20 or row["hour"] <= 5) and (row["weekday"] < 5)) else 0
    )


def rush_hour(row):
    return (
        1
        if ((row["hour"] <= 20) and (row["hour"] >= 16) and (row["weekday"] < 5))
        else 0
    )


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["year"] = dt.dt.year.fillna(0).astype("int16")
    df["month"] = dt.dt.month.fillna(0).astype("int8")
    df["day"] = dt.dt.day.fillna(0).astype("int8")
    df["hour"] = dt.dt.hour.fillna(0).astype("int8")
    df["weekday"] = dt.dt.weekday.fillna(0).astype("int8")
    df["pickup_datetime"] = dt.astype(str)

    df["night"] = df.apply(lambda x: night(x), axis=1).astype("int8")
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1).astype("int8")
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1).astype("int8")
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["latdiff"] = (lat1 - lat2).abs()
    df["londiff"] = (lon1 - lon2).abs()
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = np.sqrt(
        (df["pickup_longitude"] - df["dropoff_longitude"]).abs() ** 2
        + (df["pickup_latitude"] - df["dropoff_latitude"]).abs() ** 2
    )
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows=", len(df))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"])
        plt.legend(["train", "val"], loc="upper right")
    else:
        plt.legend(["train"], loc="upper right")
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()

    if "rmse_keras" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["rmse_keras"])
        if "val_rmse_keras" in history.history:
            plt.plot(history.history["val_rmse_keras"])
            plt.legend(["train", "val"], loc="upper right")
        else:
            plt.legend(["train"], loc="upper right")
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
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

trainKaggle = pd.read_csv(TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes)

testKaggle = pd.read_csv(
    TEST_PATH, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
)

print("Loaded trainKaggle:", trainKaggle.shape, "testKaggle:", testKaggle.shape)



## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
print("Internal split train_df:", train_df.shape, "test_df:", test_df.shape)



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



## === cell 8
train_df.describe()



## === cell 9
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 10
train_df.describe()



## === cell 11
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 12
train_df.describe()



## === cell 13
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)
print("Done with Adding features")



## === cell 14
train_df.describe()



## === cell 15
_ = train_df.iloc[:2000].plot.scatter("latdiff", "londiff")



## === cell 16
dropped_columns = ["passenger_count", "pickup_datetime"]  # keep core logic

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)
train_df = train_df.drop(["key"], axis=1)
test_df = test_df.drop(["key"], axis=1)

print("Done with dropped_columns")
print("train_df columns:", list(train_df.columns))
print("testKaggle_clean columns:", list(testKaggle_clean.columns))



## === cell 17
train_df.shape



## === cell 18
train_df.describe()



## === cell 19
test_df.describe()



## === cell 20
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 21
train_df.describe()



## === cell 22
validation_df.describe()



## === cell 23
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")
print("Feature count:", train_df.shape[1])



## === cell 24
feature_cols = list(train_df.columns)

validation_df = validation_df.reindex(columns=feature_cols)
test_df = test_df.reindex(columns=feature_cols)
testKaggle_clean = testKaggle_clean.reindex(columns=feature_cols)

missing_any = (
    validation_df.isna().any().any()
    or test_df.isna().any().any()
    or testKaggle_clean.isna().any().any()
)
if missing_any:
    validation_df = validation_df.fillna(0)
    test_df = test_df.fillna(0)
    testKaggle_clean = testKaggle_clean.fillna(0)

print("Aligned feature columns count:", len(feature_cols))



## === cell 25
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)




## === cell 26
def rmse_keras(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))




## === cell 27
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
    loss="mean_squared_error", optimizer=adam, metrics=["mae", rmse_keras, "mse"]
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



## === cell 28
print(
    "Model training complete; skipping model visualization (keras.utils.vis_utils not required)."
)



## === cell 29
plot_loss_accuracy_rmse(history)



## === cell 30
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print("Test loss:", score[0])



## === cell 31
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)

prediction = np.clip(prediction, 0.0, 50.0)
predictionKaggle = np.clip(predictionKaggle, 0.0, 50.0)




## === cell 32
def rmse_np(predictions, targets):
    predictions = np.asarray(predictions).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    return np.sqrt(((predictions - targets) ** 2).mean())




## === cell 33
Validation_prediction = model.predict(validation_df_scaled, batch_size=128, verbose=1)
Validation_prediction = np.clip(Validation_prediction, 0.0, 50.0)
rmse_val = rmse_np(Validation_prediction, validation_labels)
print("rmse error is: " + str(rmse_val))



## === cell 34
rmse_test = rmse_np(prediction, test_labels)
print("rmse error is: " + str(rmse_test))



## === cell 35
print("Max prediction:", float(np.max(prediction)))



## === cell 36
fig, ax = plt.subplots()
ax.scatter(test_labels, prediction.reshape(-1), s=5)
ax.plot(
    [test_labels.min(), test_labels.max()],
    [test_labels.min(), test_labels.max()],
    "k--",
    lw=2,
)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()



## === cell 37
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)



## === cell 38
idx = 10000
if len(prediction) > idx and len(test_labels) > idx:
    print(prediction[idx])
    print(test_labels[idx])
else:
    print("Not enough rows in internal test split to print index", idx)
