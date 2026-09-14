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

6.82552

# 6. Current score

348598211.48712

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1265.63524) has done: 'Implemented fixes to resolve import errors, URL image loading, optimizer usage, and RMSE calculation. Updated imports to use TensorFlow’s Keras, bypassed the water‑mask step, corrected optimizer instantiation, and rewrote the RMSE helper with NumPy. Adjusted dataset size and training epochs modestly to improve model performance while keeping the original architecture intact. The script now runs end‑to‑end and generates a correctly formatted CSV submission.'
- What this solution (achieved 1310.56841) has done: 'Implemented fixes to resolve the TensorFlow import error by switching to the standalone Keras package compatible with the environment, and corrected datetime parsing to robustly handle the input format. These changes enable the model to train and generate a proper CSV submission, moving the RMSE dramatically closer to the target score.'
- What this solution (achieved 486.76899) has done: 'I updated the imports to use `tensorflow.keras` to avoid the protobuf conflict, and I kept the passenger count feature (removing only the datetime column) because it provides useful information for fare prediction. These minimal changes fix the runtime error and should improve the RMSE, moving the score nearer to the target while preserving the original model architecture.'
- What this solution (achieved 205.04743) has done: 'I replace the direct keras imports with tensorflow.keras to avoid the protobuf error, fix the test‑set CSV column selection and dtype handling, and drop the unnecessary key column from the training splits so that the scaler sees identical feature sets. These minimal changes eliminate the runtime crashes, let the pipeline train and predict, and generate a correctly‑named CSV submission.'
- What this solution (achieved 153.95507) has done: 'I replace the standalone keras imports with tensorflow‑keras to avoid the protobuf error, and I stop dropping rows from the test set so the prediction array and the original test IDs have matching lengths. This fixes the runtime exception and produces a correctly‑sized submission CSV without altering the core model logic.'
- What this solution (achieved 708578.23105) has done: 'I replace the TensorFlow‑Keras imports with the standalone keras package to avoid the protobuf error, and I normalize the target fare amount (divide by its max value) before training and undo this scaling when evaluating and creating the submission. This fixes the runtime crash and should considerably lower the RMSE, moving the score toward the target while preserving the original model architecture.'
- What this solution (achieved 2987375.57175) has done: 'The fix replaces the standalone keras imports with tensorflow.keras to avoid the protobuf‐related crash, keeping the rest of the pipeline unchanged. This allows the model to train and generate a proper CSV submission, moving the RMSE dramatically closer to the target.'
- What this solution (achieved 348598211.48712) has done: 'I replace the TensorFlow‑Keras imports with the standalone keras package to avoid the protobuf‑related crash that stops the script from running. The rest of the pipeline remains unchanged, preserving the model architecture and data processing, so the code now execute end‑to‑end and generate a proper .csv submission.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras import optimizers, regularizers, backend as K


def locate_file(relative_path):
    candidates = [
        Path(relative_path),
        Path("/kaggle/input") / Path(relative_path).name,
        Path("/kaggle/working") / Path(relative_path).name,
        Path.cwd() / Path(relative_path),
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    raise FileNotFoundError(f"Unable to locate {relative_path}")


TRAIN_PATH = locate_file("kaggle/data/train.csv")
TEST_PATH = locate_file("kaggle/data/test.csv")
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 40  # keep original setting
LEARNING_RATE = 0.001
DATASET_SIZE = 200_000  # sample for quicker experimentation


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_dtypes = {
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
    dtype=train_dtypes,
    usecols=[0, 1, 2, 3, 4, 5, 6, 7],  # all columns, keep key for later
)

test_dtypes = {
    "key": "str",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
testKaggle = pd.read_csv(
    TEST_PATH,
    dtype=test_dtypes,
    usecols=[0, 1, 2, 3, 4, 5, 6],  # key + all feature columns
)


## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.20, random_state=1)
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)

print(f"train_df  size: {len(train_df)}")
print(f"validation_df size: {len(validation_df)}")
print(f"test_df  size: {len(test_df)}")
print(f"testKaggle size: {len(testKaggle)}")




## === cell 3
def clean(df):
    df = df.dropna(how="any", axis="rows")
    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        & (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    df = df[
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    ]
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
    if "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    return df


def late_night(row):
    return 1 if (row["hour"] <= 3) or (row["hour"] >= 22) else 0


def night(row):
    return 1 if (row["hour"] > 20) and (row["weekday"] < 5) else 0


def rush_hour(row):
    return 1 if (16 <= row["hour"] <= 20) and (row["weekday"] < 5) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["night"] = df.apply(night, axis=1)
    df["late_night"] = df.apply(late_night, axis=1)
    df["rush_hour"] = df.apply(rush_hour, axis=1)
    return df


def add_coordinate_features(df):
    df["latdiff"] = df["pickup_latitude"] - df["dropoff_latitude"]
    df["londiff"] = df["pickup_longitude"] - df["dropoff_longitude"]
    return df


def add_distances_features(df):
    df["manhattan"] = manhattan(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["distance"] = np.sqrt(
        np.square(df["pickup_longitude"] - df["dropoff_longitude"])
        + np.square(df["pickup_latitude"] - df["dropoff_latitude"])
    )
    return df


for name, df in [("train", train_df), ("validation", validation_df), ("test", test_df)]:
    df = clean(df)
    df = add_time_features(df)
    df = add_coordinate_features(df)
    df = add_distances_features(df)
    locals()[f"{name}_df"] = df  # overwrite with cleaned version

testKaggle_clean = testKaggle.copy()
testKaggle_clean = add_time_features(testKaggle_clean)
testKaggle_clean = add_coordinate_features(testKaggle_clean)
testKaggle_clean = add_distances_features(testKaggle_clean)
testKaggle_clean = testKaggle_clean.fillna(0)

print("Data cleaning and feature engineering completed.")


## === cell 4
drop_cols = ["pickup_datetime"]
train_df = train_df.drop(columns=drop_cols + ["key"])
validation_df = validation_df.drop(columns=drop_cols + ["key"])
test_df = test_df.drop(columns=drop_cols + ["key"])
testKaggle_clean = testKaggle_clean.drop(columns=drop_cols + ["key"])

fare_max = train_df["fare_amount"].max()
train_labels = (train_df["fare_amount"].values / fare_max).astype(np.float32)
validation_labels = (validation_df["fare_amount"].values / fare_max).astype(np.float32)
test_labels_original = test_df["fare_amount"].values.astype(
    np.float32
)  # keep original for metric

train_df = train_df.drop(columns=["fare_amount"])
validation_df = validation_df.drop(columns=["fare_amount"])
test_df = test_df.drop(columns=["fare_amount"])

print("Feature preparation complete.")
print(f"Target max value for scaling: {fare_max}")


## === cell 5
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


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

print(
    f"Training on {train_df_scaled.shape[0]} samples, {train_df_scaled.shape[1]} features"
)
model.summary()


## === cell 6
history = model.fit(
    x=train_df_scaled,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)


## === cell 7
prediction_internal_scaled = model.predict(test_scaled, batch_size=128, verbose=0)
prediction_internal = prediction_internal_scaled.ravel() * fare_max
metric = rmse(test_labels_original, prediction_internal)
print(f"RMSE on internal test split: {metric:.5f}")




## === cell 8
def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame({prediction_column: prediction.ravel()})
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print(f"Submission written to {file_name}")


predictionKaggle_scaled = model.predict(testKaggle_scaled, batch_size=128, verbose=0)
predictionKaggle = predictionKaggle_scaled.ravel() * fare_max
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)

print("Sample predictions (first 10):")
print(predictionKaggle[:10])
