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

4.25112

# 6. Current score

16.02111

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 307.80293) has done: 'I fixed the import errors, removed the faulty water‑mask loading, corrected the datetime parsing, fixed the optimizer call, cleaned up unused metrics, and added safe guards around optional visualisation code. These changes let the script run end‑to‑end and produce a valid `submissiontry_water.csv`, while keeping the original model architecture and feature engineering unchanged.'
- What this solution (achieved 15.31404) has done: 'The script failed because it imported TensorFlow‑specific Keras (`tensorflow.keras`), but the environment only provides the standalone Keras package (`keras` + `tf_keras`). Switching all Keras imports to the top‑level `keras` module resolves the import error and lets the pipeline run end‑to‑end, producing a proper CSV submission. No other logic is changed, preserving the original model and feature engineering while allowing the score to improve toward the target.'
- What this solution (achieved 702.15177) has done: 'Added a TensorFlow import and re‑implemented the custom RMSE metric using TensorFlow operations (tf.sqrt, tf.reduce_mean) which fixes the backend attribute error. This small change lets the model compile and train, defines `history`, and enables the subsequent plotting and prediction steps to run, producing a valid CSV submission. No other logic was altered, preserving the original workflow and feature engineering.'
- What this solution (achieved 15.26177) has done: 'The script failed because importing `tensorflow` triggered a protobuf `MessageFactory` error, and the custom RMSE metric relied on that import. We remove the direct TensorFlow import and replace the custom metric with Keras’s built‑in `RootMeanSquaredError`, which avoids the protobuf conflict while keeping the model architecture unchanged. The updated cells also drop the unused `rmse` function.'
- What this solution (achieved 91.53415) has done: 'We replace the TensorFlow‑based Keras imports with the `tf_keras` package to avoid the protobuf conflict, drop the overly strong L1 activity regularizer that was causing extreme under‑fitting, and clip the model’s predictions to non‑negative values (fares cannot be negative). These minimal fixes keep the original workflow intact while addressing the runtime error and nudging the RMSE toward the target.'
- What this solution (achieved 54.53186) has done: 'We keep the original architecture but fix three key issues that caused the huge RMSE: (1) the passenger count feature was dropped – we now retain it; (2) the target variable is now scaled with a Min‑Max scaler and predictions are inverse‑scaled before clipping, matching the feature scaling; (3) modestly increase training epochs for better convergence. These minimal edits keep the core logic intact while expectedly lowering the RMSE toward the target.'
- What this solution (achieved 6.10863) has done: 'The fix replaces the TensorFlow‑specific Keras imports (which raise a protobuf MessageFactory error) with the standalone `keras` package used in the environment, and updates the RMSE metric import accordingly. This resolves the import crash, lets the model train, and produces a proper CSV submission while preserving the original architecture and feature engineering.'
- What this solution (achieved 15.22784) has done: 'The script crashes because the standalone `keras` package pulls in TensorFlow protobuf code that isn’t compatible with the environment, leading to the `MessageFactory` error. Switching all Keras imports to the `tf_keras` wrapper avoids this conflict. I replace the import statements in cell 0 with `tf_keras` equivalents and adjust later references (metrics, optimizers, etc.) to use the same namespace, keeping the rest of the pipeline untouched so the model, feature engineering, and training remain identical while allowing the code to run end‑to‑end and produce a valid CSV submission.'
- What this solution (achieved 175.01346) has done: 'I fixed the TensorFlow/Keras compile error by importing the metric from the same `tf_keras` package and enabling eager execution with `run_eagerly=True`. I also removed the unnecessary Min‑Max scaling of the target variable, letting the model predict fares directly, and adjusted the prediction post‑processing accordingly. These changes let the notebook run end‑to‑end, produce a proper CSV submission, and should bring the RMSE much closer to the target.'
- What this solution (achieved 16.02111) has done: 'The fix adds proper scaling of the target variable `fare_amount` to stabilize training and improve RMSE, then inverses the scaling on predictions before creating the submission. This small change keeps the original model and feature engineering while bringing the score much closer to the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tf_keras.callbacks import EarlyStopping
from tf_keras import optimizers, regularizers, backend as K

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256  # smaller batch improves training stability
EPOCHS = 50  # a few more epochs for better convergence
LEARNING_RATE = 0.001
DATASET_SIZE = 80000



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
    TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=[1, 2, 3, 4, 5, 6, 7]
)
testKaggle = pd.read_csv(TEST_PATH)



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
    print(" New size after passenger count filter: %d" % len(df))

    print("Skipping water‑mask removal")
    return df


def late_night(row):
    return 1 if row["hour"] <= 3 else 0


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
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = np.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete")


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

    if "rmse" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["rmse"], label="train")
        plt.plot(history.history["val_rmse"], label="val")
        plt.title("Model RMSE")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend()
        plt.show()




## === cell 6
print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)



## === cell 7
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 8
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 9
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)
print("Done with Adding features")



## === cell 10
dropped_columns = ["pickup_datetime"]  # keep passenger_count as it is informative
train_df = train_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)
print("Done with dropped_columns")



## === cell 11
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values

target_scaler = preprocessing.MinMaxScaler()
train_labels_scaled = target_scaler.fit_transform(train_labels.reshape(-1, 1)).ravel()
validation_labels_scaled = target_scaler.transform(
    validation_labels.reshape(-1, 1)
).ravel()

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(
    ["fare_amount"], axis=1
)  # test set does not have target; this line keeps shape consistent
print("Done with Labels and target scaling")



## === cell 12
print("train shape:", train_df.shape)
print("validation shape:", validation_df.shape)
print("test shape:", test_df.shape)



## === cell 13
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)



## === cell 14
from tf_keras.metrics import RootMeanSquaredError  # use same tf_keras package

model = Sequential()
model.add(
    Dense(
        256,
        activation="relu",
        input_dim=train_df_scaled.shape[1],
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
    loss="mean_squared_error",
    optimizer=adam,
    metrics=["mae", RootMeanSquaredError()],
    run_eagerly=True,  # avoid distribution‑strategy issues
)

print("Dataset size:", DATASET_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Batch size:", BATCH_SIZE)
print("Input dimension:", train_df_scaled.shape[1])
print("Features used:", list(train_df.columns))
model.summary()
history = model.fit(
    x=train_df_scaled,
    y=train_labels_scaled,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(validation_df_scaled, validation_labels_scaled),
    shuffle=True,
)



## === cell 15
try:
    from IPython.display import SVG
    from tf_keras.utils import plot_model

    SVG(plot_model(model, show_shapes=True, dpi=60).create_svg())
except Exception as e:
    print("Model visualisation skipped:", e)



## === cell 16
plot_loss_accuracy_rmse(history)



## === cell 17
prediction_scaled = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle_scaled = model.predict(testKaggle_scaled, batch_size=128, verbose=1)

prediction = target_scaler.inverse_transform(prediction_scaled)
predictionKaggle = target_scaler.inverse_transform(predictionKaggle_scaled)

prediction = np.clip(prediction, 0, None)
predictionKaggle = np.clip(predictionKaggle, 0, None)



## === cell 18
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)



## === cell 19
print("Sample prediction vs true (if available):")
print(prediction[0])
