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

4.41293

# 6. Current score

None

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.26336) has done: 'I fix the runtime errors by (1) switching the Keras imports/optimizer construction to the Keras 3 API, (2) removing the unsupported URL-based water-mask filtering (which currently crashes) while keeping all other cleaning/feature logic intact, and (3) making the datetime parsing robust to the dataset’s timestamp format. I also correct the train/test CSV paths to the actual Kaggle-mounted `/kaggle/input/...` locations so the notebook runs end-to-end. Finally, I ensure the script always writes a valid `key,fare_amount` submission CSV and avoid the post-hoc loss call that currently errors due to mismatched array shapes.'
- What this solution (achieved 361.30795) has done: 'I fix the two runtime blockers so the notebook runs end-to-end: (1) avoid the environment-level `MessageFactory.GetPrototype` crash by forcing Keras to use the TensorFlow backend before importing `keras`, and (2) replace the deprecated/removed `keras.backend.sqrt/mean/square` usage with a TensorFlow-based RMSE metric compatible with Keras 3. I also correct the input file paths to the actual `/kaggle/input/new-york-city-taxi-fare-prediction/...` location in this environment so data loads reliably. To move the score toward your target (lower RMSE), I fix the time-feature boolean logic bugs (`late_night` and `night`) that currently always/never trigger incorrectly while keeping the same feature set and model architecture/training loop unchanged. Finally, I ensure the submission is always written as a valid `key,fare_amount` CSV.'
- What this solution (achieved 208.42212) has done: 'I fix the environment crash that happens before any training by pinning protobuf to the pure‑Python implementation and disabling C++ protobuf, which avoids the `MessageFactory.GetPrototype` error with TensorFlow/Keras in this Kaggle image. I also make the input CSV loading include the `key` column for train splits (it’s currently excluded) so later feature drops and any optional debugging don’t silently misalign, while keeping the same features used for modeling. Finally, I remove the incorrect `accuracy` metric (it is for classification and can interfere with regression logging/behavior) while leaving the model, loss, optimizer, training loop, and features unchanged; this is score-neutral and improves stability/clarity.'
- What this solution (achieved 476.11138) has done: 'I fix the environment crash that prevents TensorFlow from importing by switching protobuf to the pure-Python implementation (the current `"cpp"` setting triggers the `_message` import error). Then I keep the rest of your pipeline the same (data loading, cleaning, feature engineering, scaling, Keras model, training loop), but make it robust so later cells don’t cascade with `NameError`s by ensuring the earlier imports and variables always get defined. Finally, I ensure the submission is always written as a valid `key,fare_amount` CSV with a `.csv` suffix in the working directory. These changes are execution/stability fixes and should not negatively impact your intended modeling logic.'
- What this solution (achieved 72.28162) has done: 'I fix the TensorFlow/Keras import crash caused by the protobuf incompatibility (the `MessageFactory.GetPrototype` error) by forcing the pure-Python protobuf runtime *and* removing the conflicting bundled `google.protobuf` from `sys.modules` before importing TensorFlow/Keras. This is the root runtime blocker and let the pipeline run end-to-end and write a valid `.csv` submission. To move RMSE down toward your target with minimal semantic change, I also apply the exact same cleaning/feature-engineering pipeline to the `test_df` holdout split (it was previously left uncleaned, causing a distribution mismatch and very poor generalization). Everything else (model architecture, loss, optimizer, training loop, features used) remains unchanged.'
- What this solution (achieved 468.58541) has done: 'I fix the runtime crash happening before training by preventing the incompatible system protobuf runtime from being used when importing TensorFlow/Keras (this is the direct cause of the `MessageFactory.GetPrototype` failure in this environment). I do this with a minimal, standard Kaggle-safe approach: forcing the pure-Python protobuf implementation early and clearing any preloaded protobuf modules, then importing TensorFlow. I keep your model, loss, optimizer, training loop, and feature logic unchanged so the score movement comes only from the pipeline actually running reliably. Finally, I ensure the submission is always written as a valid `key,fare_amount` CSV with a `.csv` suffix in the working directory.'
- What this solution (achieved 539.99938) has done: 'I fix the TensorFlow/Keras import crash causing `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by forcing TensorFlow to use the pure‑Python protobuf runtime *and* preventing any preloaded protobuf modules from being imported before TensorFlow initializes. This is the earliest blocker (cell 0) and must be resolved for the pipeline to run at all. I keep the model architecture, features, training loop, and hyperparameters unchanged, and only add safe environment guards plus a small fallback to `tf_keras` if the standalone `keras` import still triggers the protobuf issue in this Kaggle image. This should restore end-to-end execution and produce a valid `submissiontry_water.csv`, and it should also materially reduce the RMSE versus the currently broken/unstable run by ensuring training actually happens reliably.'
- What this solution (achieved 504.73967) has done: 'The runtime is failing at the very first cell due to a known TensorFlow/protobuf incompatibility in this Kaggle image; the current “force python protobuf” guard isn’t sufficient. I fix this by using the most reliable mitigation: force the pure-Python protobuf runtime *and* ensure TensorFlow uses the legacy protobuf APIs, then import `tf_keras` directly (keeping your same Keras-style model code/architecture). This is a correctness/stability fix and unblock training/inference so you actually generate a valid `.csv` submission. Since your current RMSE is catastrophically high (likely from the run not training properly), these fixes should also move the score strongly toward your target without changing your model/features/training loop semantics.'

# 9. Code solution

## === cell 0
import os
import sys
import gc

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("TF_PROTOBUF_USE_CAPI", "0")

for m in list(sys.modules.keys()):
    if m.startswith(("google.protobuf", "protobuf")):
        del sys.modules[m]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, BatchNormalization
from tf_keras import regularizers

np.random.seed(1)
tf.random.set_seed(1)

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"  # must end with .csv

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"TRAIN_PATH not found: {TRAIN_PATH}")
if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"TEST_PATH not found: {TEST_PATH}")

print("Paths OK.")
print("TF version:", tf.__version__)
print("keras module:", keras.__name__)
print("keras version:", getattr(keras, "__version__", "unknown"))



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
    usecols=[0, 1, 2, 3, 4, 5, 6, 7],  # key + fare_amount + 6 feature cols
)
testKaggle = pd.read_csv(
    TEST_PATH, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
)

print(trainKaggle.head())
print(testKaggle.head())



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

    if "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))
    else:
        print(" fare_amount not present; skipping fare outlier filtering")

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
    print(
        "New size: %d (water-mask filtering skipped due to environment limitations)"
        % len(df)
    )

    print(" New size: %d" % len(df))
    return df


def late_night(row):
    h = int(row["hour"])
    return 1 if (h <= 3) or (h >= 22) else 0


def night(row):
    h = int(row["hour"])
    wd = int(row["weekday"])
    return 1 if ((h >= 20) or (h <= 6)) and (wd < 5) else 0


def rush_hour(row):
    if ((row["hour"] <= 20) and (row["hour"] >= 16)) and (row["weekday"] < 5):
        return 1
    else:
        return 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    dt = dt.fillna(pd.Timestamp("2009-01-01", tz="UTC"))
    df["year"] = dt.dt.year.astype(np.int16)
    df["month"] = dt.dt.month.astype(np.int8)
    df["day"] = dt.dt.day.astype(np.int8)
    df["hour"] = dt.dt.hour.astype(np.int8)
    df["weekday"] = dt.dt.weekday.astype(np.int8)

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
    df = pd.DataFrame(
        {
            id_column: raw_test[id_column].values,
            prediction_column: prediction.reshape(-1),
        }
    )
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    if not file_name.lower().endswith(".csv"):
        raise ValueError("Submission file must have .csv suffix")
    print(f"Output complete: {file_name} ({len(df)} rows)")


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history.get("val_loss", []))
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()

    if "rmse" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["rmse"])
        plt.plot(history.history.get("val_rmse", []))
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper right")
        plt.show()




## === cell 9
print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)

print("test_df clean (holdout)")
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
print("test_df add_coordinate_features Disabled!")
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
dropped_columns = ["passenger_count", "pickup_datetime"]  # keep as original

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")



## === cell 18
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount", "key"], axis=1)
validation_df = validation_df.drop(["fare_amount", "key"], axis=1)
test_df = test_df.drop(["fare_amount", "key"], axis=1)

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
testKaggle_scaled = scaler.transform(testKaggle_clean)




## === cell 23
def rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))




## === cell 24
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

adam = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

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
    shuffle=False,
)



## === cell 25
print(
    "Model visualization skipped (keras.utils.vis_utils not available in this environment)."
)



## === cell 26
plot_loss_accuracy_rmse(history)



## === cell 27
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## === cell 28
mse_sample = np.mean(
    (test_labels[: len(prediction)].reshape(-1) - prediction.reshape(-1)) ** 2
)
rmse_sample = float(np.sqrt(mse_sample))
print(f"Holdout RMSE (on test_df split): {rmse_sample:.4f}")



## === cell 29
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)



## === cell 30
idx = min(1000, len(prediction) - 1)
print("Example prediction:", float(prediction[idx]))
print("Example label:", float(test_labels[idx]))

gc.collect()
