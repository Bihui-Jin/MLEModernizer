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

4.31009

# 6. Current score

7.83286

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.36891) has done: 'I replace the TensorFlow‑Keras imports with the standalone Keras package to avoid the protobuf error, adjust the `clean` function so it skips the fare‑amount filter when that column is missing (as in the test set), and remove the faulty call that tried to clean the test data before it existed. These minimal fixes let the notebook run end‑to‑end, produce a valid `.csv` submission, and keep the original model and feature engineering unchanged.'
- What this solution (achieved 419.37197) has done: 'I fixed the backend error by switching the custom RMSE metric to use `keras.ops`, added an early‑stopping callback to improve training without changing the model architecture, and increased the epoch count so the model can converge better, which should bring the RMSE closer to the target while keeping the original workflow intact.'
- What this solution (achieved 189.47225) has done: 'I fix the dtype mismatch that caused the test file read to fail and simplify the model by removing the heavy L1 regularizer and using a ReLU activation in the first layer. These minimal changes restore end‑to‑end execution and should bring the RMSE much closer to the target score while keeping the original workflow intact.'
- What this solution (achieved 184.93833) has done: 'The changes replace the problematic `keras` imports with the compatible `tensorflow.keras` stack (fixing the protobuf error), adjust the custom RMSE metric to use TensorFlow’s backend, and add a small dropout layer to improve generalisation without altering the overall model architecture. These fixes ensure the notebook runs fully, produces a proper `.csv` submission, and modestly moves the RMSE toward the target score.'
- What this solution (achieved 221.49587) has done: 'I added a small environment‑variable fix to avoid the protobuf MessageFactory error that stopped the notebook from running, and I guard the scaled feature arrays against possible NaNs that can cause exploding loss and huge RMSE values. These changes are minimal, keep the original model and workflow untouched, and should let the script execute fully while moving the validation RMSE much closer to the target.'
- What this solution (achieved 15.36061) has done: 'I replace the TensorFlow‑Keras imports with the standalone `keras` package to eliminate the protobuf MessageFactory error that stops the notebook from running. This change lets the model train normally, producing valid predictions and a proper `.csv` submission while preserving the original architecture and workflow.'
- What this solution (achieved 835.13125) has done: 'I fixed the backend‑compatibility issue by redefining the custom RMSE metric using `keras.ops` instead of the removed Keras backend functions, and I updated the callbacks to monitor `val_rmse` so training stops based on the proper metric. I also corrected the printed score indices to show the actual RMSE values. These minimal changes restore full execution, produce a valid `.csv` submission, and should bring the validation RMSE closer to the target.'
- What this solution (achieved 528.49576) has done: 'The fix switches to TensorFlow‑Keras (avoiding the protobuf import error), updates the metric‑printing indices so the true RMSE is shown, and keeps the original model and preprocessing untouched. These minimal changes let the notebook run end‑to‑end, produce a proper `.csv` submission, and move the reported score toward the target.'
- What this solution (achieved 712.92943) has done: 'I replace the TensorFlow Keras imports with the standalone keras package to avoid the protobuf `MessageFactory` error, keeping the rest of the workflow unchanged. This fixes the runtime failure and lets the model train and output a valid `.csv` submission, moving the RMSE toward the target.'
- What this solution (achieved 712.90755) has done: 'I replace the standalone `keras` imports with the TensorFlow‑Keras equivalents to avoid the protobuf “MessageFactory” error, alias the needed sub‑modules so the rest of the notebook can stay unchanged, and adjust the custom `rmse` function to use `tf.keras.ops`. These minimal fixes let the script run end‑to‑end, produce a proper `.csv` submission, and give the model a chance to train correctly, moving the RMSE toward the target value.'
- What this solution (achieved 80.49167) has done: 'The script failed because it mixed TensorFlow‑Keras (`tf.keras`) with the standalone `keras` package, which triggers a protobuf incompatibility (`MessageFactory` error). Switching all Keras imports to the TensorFlow‑Keras equivalents removes the conflict while keeping the original model, preprocessing, and metric logic unchanged, allowing the notebook to run end‑to‑end and produce a valid `.csv` submission.'
- What this solution (achieved 189.23629) has done: 'The fix switches all TensorFlow‑Keras imports to the native `keras` package (avoiding the protobuf MessageFactory error) and adds a modest learning‑rate schedule plus a slightly longer patience for early stopping. These changes let the notebook run end‑to‑end, produce a proper `.csv` submission and improve the model’s convergence, moving the RMSE much closer to the target while keeping the original architecture and feature engineering untouched.'
- What this solution (achieved 7.83286) has done: 'I fix the import issue, add a more informative distance feature (Haversine), and clip predictions to a realistic range before writing the submission. These changes keep the original model architecture and training loop intact while improving the data quality, which should lower the RMSE toward the target.'

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

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers, regularizers, backend as K
from keras import ops as Kops  # ops for sqrt, mean, square


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
    print(" New size after NYC long/lat bounds: %d" % len(df))

    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)]
    print(" New size after coordinate delta > 0.001: %d" % len(df))

    if "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after fare amount filter: %d" % len(df))

    df = df[(df["passenger_count"] > 0)]
    print(" New size after passenger count filter: %d" % len(df))

    df = remove_datapoints_from_water(df)
    print(" Final size after water mask step: %d" % len(df))
    return df


def remove_datapoints_from_water(df):
    return df


def late_night(row):
    return 1 if (row["hour"] <= 3) else 0


def night(row):
    return 1 if ((row["hour"] > 20) and (row["weekday"] < 5)) else 0


def rush_hour(row):
    return 1 if ((16 <= row["hour"] <= 20) and (row["weekday"] < 5)) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def haversine(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    R = 6371.0
    lat1 = np.radians(pickup_lat)
    lon1 = np.radians(pickup_long)
    lat2 = np.radians(dropoff_lat)
    lon2 = np.radians(dropoff_long)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    return df


def add_distances_features(df):
    df["manhattan"] = manhattan(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["haversine"] = haversine(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    pred = np.clip(prediction.squeeze(), 0, 50)
    df = pd.DataFrame(pred, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print(f"Output written to {file_name}")


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(history.history["rmse"], label="train rmse")
    plt.plot(history.history["val_rmse"], label="val rmse")
    plt.title("Model RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()


def rmse(y_true, y_pred):
    return Kops.sqrt(Kops.mean(Kops.square(y_pred - y_true)))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 1000
EPOCHS = 150  # a bit more epochs for better convergence
LEARNING_RATE = 0.001
DATASET_SIZE = 800_000  # rows from training file




## === cell 2
train_dtypes = {
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
test_dtypes = {
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
    usecols=[
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
    dtype=test_dtypes,
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




## === cell 3
train_df, validation_df = train_test_split(trainKaggle, test_size=0.10, random_state=1)

print("Cleaning training data...")
train_df = clean(train_df)
print("Cleaning validation data...")
validation_df = clean(validation_df)

testKaggle_clean = testKaggle.copy()




## === cell 4
print("Adding time features...")
train_df = add_time_features(train_df)
validation_df = add_time_features(validation_df)
testKaggle_clean = add_time_features(testKaggle_clean)

print("Adding distance features...")
train_df = add_distances_features(train_df)
validation_df = add_distances_features(validation_df)
testKaggle_clean = add_distances_features(testKaggle_clean)




## === cell 5
dropped_columns = ["pickup_datetime"]
train_df = train_df.drop(columns=dropped_columns)
validation_df = validation_df.drop(columns=dropped_columns)

testKaggle_clean_features = testKaggle_clean.drop(columns=dropped_columns + ["key"])




## === cell 6
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values

train_features = train_df.drop(columns=["fare_amount"])
validation_features = validation_df.drop(columns=["fare_amount"])




## === cell 7
scaler = preprocessing.MinMaxScaler()
train_scaled = scaler.fit_transform(train_features)
validation_scaled = scaler.transform(validation_features)
test_scaled = scaler.transform(testKaggle_clean_features)

train_scaled = np.nan_to_num(train_scaled)
validation_scaled = np.nan_to_num(validation_scaled)
test_scaled = np.nan_to_num(test_scaled)




## === cell 8
model = Sequential()
model.add(
    Dense(
        256,
        activation="relu",
        input_dim=train_scaled.shape[1],
    )
)
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    loss="mean_squared_error",
    optimizer=adam,
    metrics=["mae", rmse, "mse"],
)

print("Model summary:")
model.summary()




## === cell 9
checkpoint = ModelCheckpoint(
    filepath="my_model.keras",
    monitor="val_rmse",
    verbose=1,
    save_best_only=True,
    mode="min",
)
early_stop = EarlyStopping(
    monitor="val_rmse", patience=10, restore_best_weights=True, mode="min"
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_rmse",
    factor=0.5,
    patience=5,
    verbose=1,
    mode="min",
    min_lr=1e-6,
)

history = model.fit(
    x=train_scaled,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_scaled, validation_labels),
    callbacks=[checkpoint, early_stop, reduce_lr],
    shuffle=True,
    verbose=1,
)




## === cell 10
plot_loss_accuracy_rmse(history)




## === cell 11
train_score = model.evaluate(train_scaled, train_labels, verbose=0)
val_score = model.evaluate(validation_scaled, validation_labels, verbose=0)

print(f"Train RMSE: {train_score[2]:.4f}")
print(f"Validation RMSE: {val_score[2]:.4f}")




## === cell 12
test_predictions = model.predict(test_scaled, batch_size=128, verbose=1)

output_submission(
    raw_test=testKaggle,
    prediction=test_predictions,
    id_column="key",
    prediction_column="fare_amount",
    file_name=SUBMISSION_NAME,
)
