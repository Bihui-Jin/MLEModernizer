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
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
        input/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
```

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> input/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> input/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> input/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> working/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Target score

4.41293

# 6. Current score

61866801.75932

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.20426) has done: 'Implemented fixes to resolve import errors, align feature columns for scaling, robust datetime parsing, and ensure predictions are valid. Adjusted imports to use the standalone keras library, dropped the target column from the internal test set before scaling, parsed dates without a strict format, and clipped negative fare predictions. The script now runs end‑to‑end and writes a proper submissiontry_water.csv file.'
- What this solution (achieved 6.22708) has done: 'I correct the import errors, point the data‑loading paths to the actual Kaggle input files, and adjust the Keras imports to use TensorFlow‑Keras, which resolves the protobuf AttributeError. The script now reads the training and test CSVs, performs the original cleaning, feature engineering, scaling, model building and training, and finally writes a properly formatted **submissiontry_water.csv** file. No core logic is changed, only the bugs that prevented execution are fixed, keeping the original methodology intact.'
- What this solution (achieved 15.25184) has done: 'I replace the TensorFlow‑Keras imports with the standalone Keras library to remove the protobuf error, and I slightly tune the model (reduce L1 regularisation, double the training epochs and add early stopping) to improve RMSE while keeping the original architecture unchanged. This ensures the script runs end‑to‑end and writes a proper submissiontry_water.csv file with a lower error score.'
- What this solution (achieved 5.92439) has done: 'The script failed due to an incompatibility when importing TensorFlow; we replace the TensorFlow‑Keras imports with the standalone keras package, add more informative distance and cyclical time features to improve model performance, and give the early‑stopping callback a larger patience so the network can train a bit longer.'
- What this solution (achieved 6.05393) has done: 'The fix switches to the TensorFlow‑Keras wrapper to avoid the protobuf import error and sets the protobuf implementation to the pure‑Python version. No core logic changes are made, so the model and feature engineering remain unchanged while the script can now run end‑to‑end and produce a proper CSV submission.'
- What this solution (achieved 5.81805) has done: 'The fix replaces the problematic `tf_keras` imports with the standard standalone Keras imports, eliminating the protobuf error that stopped the notebook from running. No other logic is changed, so the feature engineering, model, and submission steps remain intact, allowing the script to execute end‑to‑end and produce a valid CSV submission.'
- What this solution (achieved 15.25184) has done: 'Increasing the model capacity, removing the L1 regularizer, and giving Keras a pure‑NumPy backend fixes the protobuf import error while allowing the network to train longer (more epochs, larger patience). These minimal tweaks keep the original workflow but should lower the RMSE toward the target.'
- What this solution (achieved 61866801.75932) has done: 'The fix switches the Keras imports to TensorFlow‑Keras (the default backend that supports model.fit) and removes the environment setting that forced the NumPy backend, which caused the “fit not implemented” error. With TensorFlow as the backend the neural network can be trained, predictions are generated, and a properly formatted CSV submission file is written.'

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
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers, regularizers, backend as K

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/labels.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 200  # allow more epochs for better convergence
LEARNING_RATE = 0.0005
DATASET_SIZE = 500_000  # subsample for quicker runs



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
    usecols=[1, 2, 3, 4, 5, 6, 7],  # keep all except the first column (key)
)

testKaggle = pd.read_csv(TEST_PATH, dtype=datatypes)



## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)




## === cell 3
def remove_datapoints_from_water(df):
    try:
        import urllib.request
        from PIL import Image

        url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
        with urllib.request.urlopen(url) as resp:
            img = Image.open(resp)
            nyc_mask = np.array(img)[:, :, 0] > 0.9

        def lonlat_to_xy(longitude, latitude, dx, dy, BB):
            return (
                (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype(int),
                (dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])).astype(int),
            )

        BB = (-74.5, -72.8, 40.5, 41.8)
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
        idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]
        return df[idx]
    except Exception:
        return df


def clean(df):
    print("  Old size:", len(df))
    df = df.dropna(how="any", axis="rows")
    print("  After dropna:", len(df))
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
    bbox = (-74.5, -72.8, 40.5, 41.8)
    df = df[(bbox[0] <= df["pickup_longitude"]) & (df["pickup_longitude"] <= bbox[1])]
    df = df[(bbox[0] <= df["dropoff_longitude"]) & (df["dropoff_longitude"] <= bbox[1])]
    df = df[(bbox[2] <= df["pickup_latitude"]) & (df["pickup_latitude"] <= bbox[3])]
    df = df[(bbox[2] <= df["dropoff_latitude"]) & (df["dropoff_latitude"] <= bbox[3])]
    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]

    special = [
        (40.7141667, -74.0063889),
        (40.639722, -73.778889),
        (40.6925, -74.168611),
        (40.77725, -73.872611),
        (40.6892, -74.0445),
    ]
    for lat, lon in special:
        df = df[(df["pickup_latitude"] != lat) | (df["pickup_longitude"] != lon)]
        df = df[(df["dropoff_latitude"] != lat) | (df["dropoff_longitude"] != lon)]

    df = remove_datapoints_from_water(df)
    print("  Cleaned size:", len(df))
    return df


def late_night(row):
    return 1 if row["hour"] <= 3 else 0


def night(row):
    return 1 if (row["hour"] > 20 and row["weekday"] < 5) else 0


def rush_hour(row):
    return 1 if (16 <= row["hour"] <= 20 and row["weekday"] < 5) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["pickup_datetime"] = df["pickup_datetime"].astype(str)
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
        (df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
        + (df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
    )
    R = 6371.0
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    dlat = np.radians(df["dropoff_latitude"] - df["pickup_latitude"])
    dlon = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    df["haversine"] = 2 * R * np.arcsin(np.sqrt(a))
    return df


def add_cyclical_time_features(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print(f"Submission written to {file_name}")




## === cell 4
print("Cleaning training split")
train_df = clean(train_df)
print("Cleaning validation split")
validation_df = clean(validation_df)



## === cell 5
print("Adding time features")
train_df = add_time_features(train_df)
validation_df = add_time_features(validation_df)
test_df = add_time_features(test_df)
testKaggle = add_time_features(testKaggle)



## === cell 6
print("Adding coordinate features")
train_df = add_coordinate_features(train_df)
validation_df = add_coordinate_features(validation_df)
test_df = add_coordinate_features(test_df)
testKaggle = add_coordinate_features(testKaggle)



## === cell 7
print("Adding distance features")
train_df = add_distances_features(train_df)
validation_df = add_distances_features(validation_df)
test_df = add_distances_features(test_df)
testKaggle = add_distances_features(testKaggle)

print("Adding cyclical time features")
train_df = add_cyclical_time_features(train_df)
validation_df = add_cyclical_time_features(validation_df)
test_df = add_cyclical_time_features(test_df)
testKaggle = add_cyclical_time_features(testKaggle)

print("Feature engineering complete")



## === cell 8
drop_cols = ["pickup_datetime"]
train_df = train_df.drop(drop_cols, axis=1)
validation_df = validation_df.drop(drop_cols, axis=1)
test_df = test_df.drop(drop_cols, axis=1)

testKaggle_clean = testKaggle.drop(drop_cols + ["key"], axis=1)

test_df = test_df.drop(["fare_amount"], axis=1, errors="ignore")
print("Dropped unnecessary columns")



## === cell 9
train_labels = np.log1p(train_df["fare_amount"].values)
validation_labels = np.log1p(validation_df["fare_amount"].values)

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)

print("Separated labels (log‑transformed)")



## === cell 10
print("train shape:", train_df.shape)
print("validation shape:", validation_df.shape)
print("test shape:", test_df.shape)
print("testKaggle_clean shape:", testKaggle_clean.shape)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)
print("Scaling complete")



## === cell 11
model = Sequential()
model.add(
    Dense(
        512,
        activation="relu",
        input_dim=train_df_scaled.shape[1],
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
model.add(Dense(1))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mean_squared_error", optimizer=adam, metrics=["mae"])

print(f"Dataset size: {DATASET_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Input dimension: {train_df_scaled.shape[1]}")
print(f"Features used: {list(train_df.columns)}")
model.summary()

early_stop = EarlyStopping(patience=20, restore_best_weights=True)

history = model.fit(
    x=train_df_scaled,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
    callbacks=[early_stop],
)




## === cell 12
def plot_loss_accuracy(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Model loss (MSE)")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

    if "mae" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["mae"], label="train")
        plt.plot(history.history["val_mae"], label="val")
        plt.title("Model MAE")
        plt.ylabel("mae")
        plt.xlabel("epoch")
        plt.legend()
        plt.show()


plot_loss_accuracy(history)



## === cell 13
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)

prediction = np.expm1(prediction)
predictionKaggle = np.expm1(predictionKaggle)

prediction = np.maximum(prediction, 0)
predictionKaggle = np.maximum(predictionKaggle, 0)



## === cell 14
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)



## === cell 15
print("Sample prediction (first row):", prediction[0][0])
