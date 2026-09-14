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

4.02509

# 6. Current score

35.60207

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 140.10896) has done: 'I fixed the import errors by switching to TensorFlow’s Keras API, replaced the custom rmse metric with the built‑in RootMeanSquaredError metric, removed the unnecessary model load (using the trained model directly), and ensured the submission CSV is written correctly. These changes unblock the training, allow metric calculation, and produce a valid submissiontry_water.csv file.'
- What this solution (achieved 37.45888) has done: 'I fixed the TensorFlow import error by setting the protobuf implementation before importing TensorFlow, corrected the dataset paths to point at the actual `labels.csv` and `test.csv` files, and adjusted the reading logic accordingly. With these fixes the pipeline runs end‑to‑end, trains the model, evaluates it, and writes a proper `submissiontry_water.csv` containing the required `key` and `fare_amount` columns.'
- What this solution (achieved 6.292490133145006e+30) has done: 'The main failure comes from importing the standalone keras package, which conflicts with the protobuf version used by TensorFlow. Switching to tf.keras removes the “MessageFactory” error and allows the whole pipeline to run, producing a proper submissiontry_water.csv file. No other logic changes are needed, so the core model and feature engineering stay intact.'
- What this solution (achieved 70.85182) has done: 'We train the model on the original fare amounts (no log‑transform) and adjust the prediction steps accordingly. This fixes the exploding‑value issue that produced an astronomically large RMSE and keeps the rest of the pipeline unchanged.'
- What this solution (achieved 54.64143) has done: 'The fix removes the unnecessary log‑transform of the target variable, trains the model directly on the original fare amounts, and adjusts the prediction steps accordingly. This aligns the training objective with the competition’s RMSE metric, eliminates the need for inverse‑transform steps, and ensures a valid submission CSV is written.'
- What this solution (achieved 35.60207) has done: 'The fix adds proper scaling of the target variable, inverses that scaling for predictions, and adjusts the imports. This keeps the original model architecture while enabling the network to learn more effectively, which should lower the RMSE toward the target score.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # new import

import tensorflow as tf
from tensorflow.keras import (
    models,
    layers,
    callbacks,
    optimizers,
    metrics,
    backend as K,
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
potential_paths = [
    "../input/new-york-city-taxi-fare-prediction",
    "./input/new-york-city-taxi-fare-prediction",
    "/kaggle/input/new-york-city-taxi-fare-prediction",
]
BASE_INPUT = next((p for p in potential_paths if os.path.isdir(p)), None)
if BASE_INPUT is None:
    raise FileNotFoundError(
        "Base input directory not found. Checked paths: " + ", ".join(potential_paths)
    )

TRAIN_PATH = os.path.join(BASE_INPUT, "labels.csv")
TEST_PATH = os.path.join(BASE_INPUT, "test.csv")
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 1000
EPOCHS = 150  # early stopping will cap
LEARNING_RATE = 0.001
DATASET_SIZE = 800_000  # number of rows to sample from training data



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
trainKaggle = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    dtype=datatypes,
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
    dtype=datatypes,
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
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10_000]




## === cell 4
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

    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["dropoff_longitude"] != 0)]
    df = df[(df["dropoff_latitude"] != 0)]

    print(" New size after removing residual zeros: %d" % len(df))

    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)]

    print(" New size after distance >0.001: %d" % len(df))

    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    print(" New size after fare outlier filter: %d" % len(df))

    df = df[(df["passenger_count"] > 0)]
    print(" New size after passenger count filter: %d" % len(df))

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)

    for lon, lat in [nyc_coord, fk_coord, ewr_coord, lga_coord, sol_coord]:
        df = df[(lon != df["pickup_longitude"]) & (lat != df["pickup_latitude"])]
        df = df[(lon != df["dropoff_longitude"]) & (lat != df["dropoff_latitude"])]

    print("Cleaning finished.")
    return df


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    return df


def add_coordinate_features(df):
    return df


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


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
    df = pd.DataFrame({prediction_column: prediction.squeeze()})
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print(f"Submission written to {file_name}")


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

    if "rmse" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["rmse"], label="train rmse")
        plt.plot(history.history["val_rmse"], label="val rmse")
        plt.title("Model RMSE")
        plt.ylabel("RMSE")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()




## === cell 5
print("Cleaning train_df")
train_df = clean(train_df)
print("Cleaning test_df")
test_df = clean(test_df)

print("Adding time features to train_df")
train_df = add_time_features(train_df)
print("Adding time features to test_df")
test_df = add_time_features(test_df)
print("Adding time features to testKaggle")
testKaggle = add_time_features(testKaggle)

print("Adding coordinate placeholder features")
train_df = add_coordinate_features(train_df)
test_df = add_coordinate_features(test_df)
testKaggle = add_coordinate_features(testKaggle)

print("Adding distance features")
train_df = add_distances_features(train_df)
test_df = add_distances_features(test_df)
testKaggle = add_distances_features(testKaggle)



## === cell 6
dropped_columns = ["pickup_datetime"]
train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)



## === cell 7
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)

train_labels_raw = train_df["fare_amount"].values
validation_labels_raw = validation_df["fare_amount"].values
test_labels_raw = test_df["fare_amount"].values

label_scaler = StandardScaler()
train_labels = label_scaler.fit_transform(train_labels_raw.reshape(-1, 1)).flatten()
validation_labels = label_scaler.transform(
    validation_labels_raw.reshape(-1, 1)
).flatten()
test_labels = label_scaler.transform(test_labels_raw.reshape(-1, 1)).flatten()

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)



## === cell 8
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)



## === cell 9
rmse_metric = metrics.RootMeanSquaredError(name="rmse")
checkpoint = callbacks.ModelCheckpoint(
    filepath="my_model.h5", verbose=1, save_best_only=True
)

model = models.Sequential()
model.add(layers.Dense(256, activation="relu", input_dim=train_df_scaled.shape[1]))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dense(1))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    loss="mean_squared_error",
    optimizer=adam,
    metrics=["mae", rmse_metric, "mse"],
)

print(f"Dataset size: {DATASET_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Input dimension: {train_df_scaled.shape[1]}")
print(f"Features used: {list(train_df.columns)}")
model.summary()

early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=12, restore_best_weights=True
)

history = model.fit(
    x=train_df_scaled,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[checkpoint, early_stop],
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)



## === cell 10
try:
    best_model = models.load_model("my_model.h5")
except Exception:
    best_model = model
    print("Using in‑memory trained model as best_model")



## === cell 11
plot_loss_accuracy_rmse(history)



## === cell 12
validation_pred_scaled = best_model.predict(validation_df_scaled).flatten()
validation_predictions = label_scaler.inverse_transform(
    validation_pred_scaled.reshape(-1, 1)
).flatten()
plt.scatter(validation_labels_raw, validation_predictions, alpha=0.5)
plt.xlabel("True fare")
plt.ylabel("Predicted fare")
plt.title("Validation: True vs Predicted")
plt.show()



## === cell 13
test_pred_scaled = best_model.predict(test_scaled).flatten()
test_predictions = label_scaler.inverse_transform(
    test_pred_scaled.reshape(-1, 1)
).flatten()
plt.scatter(test_labels_raw, test_predictions, alpha=0.5)
plt.xlabel("True fare")
plt.ylabel("Predicted fare")
plt.title("Internal Test: True vs Predicted")
plt.show()



## === cell 14
predictionKaggle_scaled = best_model.predict(
    testKaggle_scaled, batch_size=128, verbose=1
).flatten()
predictionKaggle = label_scaler.inverse_transform(
    predictionKaggle_scaled.reshape(-1, 1)
).flatten()
predictionKaggle = np.clip(predictionKaggle, a_min=0, a_max=None)
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
