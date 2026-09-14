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

4.6336

# 6. Current score

3.782275882247914e+24

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 35.64618) has done: 'I fixed the import errors (using tensorflow.keras instead of the standalone keras package), corrected the optimizer call, removed the failing URL‑based water‑mask step, and cleaned up unused visualization imports. These changes let the notebook run end‑to‑end, produce a trained model, and write a proper submission.csv file while keeping the original feature‑engineering and model architecture intact, which should lower the RMSE toward the target score.'
- What this solution (achieved 15.19503) has done: 'Implemented fixes to resolve import errors, correct datetime parsing, ensure feature‑engineering functions modify the actual DataFrames, and retain the useful `passenger_count` feature. These changes eliminate the protobuf crash, properly enrich the data with time and distance attributes, and preserve a predictive feature, moving the RMSE much closer to the target while keeping the original model architecture intact.'
- What this solution (achieved 5.66235) has done: 'Implemented three key fixes: switched all Keras imports to `tensorflow.keras` to resolve the protobuf backend errors, rewrote the custom RMSE metric using TensorFlow operations, and trained the model on the log‑transformed `fare_amount` (using `np.log1p`) then exponentiated predictions for evaluation and submission. These changes remove the runtime crashes, align the loss with a more appropriate target scale, and provide a noticeable RMSE improvement toward the target while preserving the original architecture and feature engineering.'
- What this solution (achieved 106.37852) has done: 'I remove the log‑transform of the target so the model learns fares directly (keeping the same architecture) and adjust the subsequent code that previously applied `expm1`. This aligns the loss with the evaluation metric and should lower the RMSE toward the target. I also raise the sample size to 200 000 rows for a more representative training set while staying within memory limits.'
- What this solution (achieved 15.23303) has done: 'Implemented two minimal fixes: (1) set the protobuf implementation environment variable before importing TensorFlow to stop the `MessageFactory` AttributeError, and (2) switched the model loss to `mean_squared_log_error` which better matches the fare distribution and nudges RMSE toward the target. No other logic or architecture was altered.'
- What this solution (achieved 493.29177) has done: 'I fixed the loss identifier to the correct TensorFlow name (`mean_squared_logarithmic_error`), reduced the L1 regularization strength (to avoid over‑penalising the model), and clipped the final predictions to non‑negative values before writing the Kaggle submission. These changes unblock model training, improve learning stability, and keep the core architecture unchanged, moving the RMSE closer to the target while still producing a valid `submissiontry_water.csv` file.'
- What this solution (achieved 5.9273) has done: 'Implemented a log‑transform of the target variable so the model trains on a smoother scale and then exponentiates predictions back to the original fare amount before evaluation and submission. Adjusted the loss to mean‑squared‑error (matching the transformed target) and updated the RMSE calculations to work on the de‑transformed predictions. These minimal changes keep the original architecture intact while substantially improving the RMSE toward the target.'
- What this solution (achieved 3.782275882247914e+24) has done: 'Implemented three key fixes: (1) added a haversine distance feature for better geographic representation, (2) corrected the train/validation split to avoid data leakage by training only on the true training subset, and (3) increased training epochs to allow the model more learning capacity. These minimal changes keep the original architecture intact while improving model performance and ensuring a valid submission CSV is created.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import optimizers, regularizers, backend as K

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 200  # increased epochs for better learning
LEARNING_RATE = 0.001
DATASET_SIZE = 200000  # increased sample size for better learning



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
    usecols=[1, 2, 3, 4, 5, 6, 7],  # exclude 'key' for training
)

testKaggle = pd.read_csv(TEST_PATH, dtype=datatypes)  # keep full test for submission



## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]  # keep a manageable size for quick checks




## === cell 3
def clean(df):
    """Basic data cleaning – drops NaNs, impossible coordinates and outliers."""
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        & (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    print(" New size after removing same long/lat: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    ]
    print(" New size after removing zero coordinates: %d" % len(df))

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
    print(" New size after NYC bounds filter: %d" % len(df))

    df = df[
        ((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)
        & ((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)
    ]
    print(" New size after minimum distance filter: %d" % len(df))

    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    print(" New size after fare amount filter: %d" % len(df))

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after passenger count filter: %d" % len(df))

    return df




## === cell 4
print("Cleaning training split")
train_df = clean(train_df)
print("Cleaning test split")
test_df = clean(test_df)




## === cell 5
def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["pickup_datetime"] = df["pickup_datetime"].astype(str)

    df["night"] = ((df["hour"] >= 20) & (df["hour"] <= 23)).astype(int)
    df["late_night"] = ((df["hour"] >= 0) & (df["hour"] <= 3)).astype(int)
    df["rush_hour"] = (
        (df["hour"] >= 16) & (df["hour"] <= 20) & (df["weekday"] < 5)
    ).astype(int)
    return df


def add_coordinate_features(df):
    df["latdiff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
    df["londiff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    return df


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def haversine(pickup_long, pickup_lat, dropoff_long, dropoff_lat):
    R = 6371.0
    phi1 = np.radians(pickup_lat)
    phi2 = np.radians(dropoff_lat)
    dphi = np.radians(dropoff_lat - pickup_lat)
    dlambda = np.radians(dropoff_long - pickup_long)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def add_distances_features(df):
    df["manhattan"] = manhattan(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["distance"] = np.sqrt(
        (df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
        + (df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
    )
    df["haversine"] = haversine(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )
    return df




## === cell 6
train_df = add_time_features(train_df)
train_df = add_coordinate_features(train_df)
train_df = add_distances_features(train_df)

test_df = add_time_features(test_df)
test_df = add_coordinate_features(test_df)
test_df = add_distances_features(test_df)

testKaggle = add_time_features(testKaggle)
testKaggle = add_coordinate_features(testKaggle)
testKaggle = add_distances_features(testKaggle)



## === cell 7
dropped_columns = ["pickup_datetime"]
train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(
    dropped_columns + ["key"], axis=1
)  # keep only features for prediction

train_set, val_set = train_test_split(train_df, test_size=0.10, random_state=1)

train_labels_log = np.log1p(train_set["fare_amount"].values)
val_labels_log = np.log1p(val_set["fare_amount"].values)

train_features = train_set.drop(["fare_amount"], axis=1).values
val_features = val_set.drop(["fare_amount"], axis=1).values
test_features = test_df.drop(["fare_amount"], axis=1).values
test_labels = test_df["fare_amount"].values  # original scale for later check

print(
    "Shapes -> train:",
    train_features.shape,
    "validation:",
    val_features.shape,
    "test:",
    test_features.shape,
)



## === cell 8
scaler = preprocessing.MinMaxScaler()
train_scaled = scaler.fit_transform(train_features)
validation_scaled = scaler.transform(val_features)
test_scaled = scaler.transform(test_features)
testKaggle_scaled = scaler.transform(testKaggle_clean)




## === cell 9
def rmse_tf(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))




## === cell 10
model = Sequential()
model.add(
    Dense(
        256,
        activation="linear",
        input_dim=train_scaled.shape[1],
        activity_regularizer=regularizers.l1(0.001),  # reduced regularization
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
model.add(Dense(1, activation="linear"))  # predicts log‑fare

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    loss="mse",  # regression on log‑transformed target
    optimizer=adam,
    metrics=["mae", rmse_tf, "mse"],
)

print("Model summary:")
model.summary()



## === cell 11
history = model.fit(
    x=train_scaled,
    y=train_labels_log,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(validation_scaled, val_labels_log),
    shuffle=True,
)



## === cell 12
val_pred_log = model.predict(validation_scaled).flatten()
val_pred = np.expm1(val_pred_log)  # back to original scale
val_rmse = np.sqrt(((val_pred - np.expm1(val_labels_log)) ** 2).mean())
print(f"Validation RMSE (original scale): {val_rmse:.4f}")



## === cell 13
test_predictions_log = model.predict(test_scaled).flatten()
test_predictions = np.expm1(test_predictions_log)
rmse_local = np.sqrt(((test_predictions - test_labels) ** 2).mean())
print(f"Local test RMSE (slice, original scale): {rmse_local:.4f}")



## === cell 14
predictionKaggle_log = model.predict(
    testKaggle_scaled, batch_size=128, verbose=1
).flatten()
predictionKaggle = np.expm1(predictionKaggle_log)
predictionKaggle = np.maximum(predictionKaggle, 0.0)

submission_df = pd.DataFrame(
    {"key": testKaggle["key"], "fare_amount": predictionKaggle}
)
submission_df.to_csv(SUBMISSION_NAME, index=False)
print(f"Submission file written to {SUBMISSION_NAME}")
