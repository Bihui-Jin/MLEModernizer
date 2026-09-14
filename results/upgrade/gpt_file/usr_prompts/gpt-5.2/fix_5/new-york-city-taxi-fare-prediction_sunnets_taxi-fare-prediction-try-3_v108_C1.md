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

No external packages required in the script and installed.

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

4.031294871457426

# 6. Current score

36.43955

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 96.7295) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation early, which avoids the `MessageFactory.GetPrototype` error in Kaggle environments with protobuf/TensorFlow mismatches. Then I fix the `Invalid dtype: object` training error by ensuring all model input columns are numeric, dropping the non-numeric `key` column from train/val/test features, and aligning feature columns across train/val/testKaggle. Finally, I fix the inference shape mismatch (15 vs 14) by using a single shared feature column list and reindexing all splits to that list before scaling and prediction, so the submission CSV is produced correctly with `key,fare_amount`.'
- What this solution (achieved 14.55537) has done: 'I fix the TensorFlow/protobuf crash by switching to the supported workaround (forcing the pure-Python protobuf implementation *and* disabling the C++ implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`) before importing TensorFlow, and I add a safe fallback to a non-TF baseline so the notebook always produces a valid submission CSV even if TF still fails in the environment. To move the RMSE score strongly toward the target (lower is better) without changing your model architecture/training semantics, I correct a major scaling logic bug: you currently re-fit the `MinMaxScaler` separately for each column, effectively only using the last column’s scaling for all features; I replace it with a single scaler fit on the full feature matrix and applied consistently to train/val/test/Kaggle. I also clamp extreme/invalid predictions to a reasonable range (non-negative and capped) to avoid RMSE blow-ups from outliers, which is a minimal, metric-aligned post-processing step. All paths, feature engineering, and the Keras model/loss/training loop remain the same otherwise, and the script always write `submissiontry_water.csv`.'
- What this solution (achieved 36.43955) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *and* pre-importing `google.protobuf.message_factory` before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` AttributeError seen in this environment. To move RMSE substantially toward your target (lower is better) without changing your model/training loop, I remove the TF fallback path so the network actually trains instead of outputting near-constant predictions (which explains the very poor 14.55 score). I also ensure we load the checkpointed best model weights (`load_weights`) before predicting/evaluating, which is a minimal, metric-aligned fix that improves stability and score without altering architecture. All file paths, feature engineering, scaling approach, and the Keras model definition/training semantics remain the same, and the script always write a valid `submissiontry_water.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import google.protobuf.message_factory  # noqa: F401

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

SUBMISSION_NAME = "submissiontry_water.csv"

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

BATCH_SIZE = 1000
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

np.random.seed(1)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras import backend

tf.random.set_seed(1)




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
    print(" New size after NYC lang lot: %d" % len(df))

    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["dropoff_longitude"] != 0)]
    df = df[(df["dropoff_latitude"] != 0)]
    print(" New size after lang lot > 0: %d" % len(df))

    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)]
    print(" New size after lang - lot > 0.001: %d" % len(df))

    print(" New size after only NYC: %d" % len(df))

    if "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))

    if "passenger_count" in df.columns:
        df = df[(df["passenger_count"] > 0)]
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


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], utc=True, errors="coerce"
    )
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["minute"] = df["pickup_datetime"].dt.minute
    df["second"] = df["pickup_datetime"].dt.second
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
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv((file_name), index=False)
    print("Output complete ->", file_name, "rows:", len(df))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(history.history.get("rmse", []))
    plt.plot(history.history.get("val_rmse", []))
    plt.title("Model rmse")
    plt.ylabel("rmse")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()




## === cell 2
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

train_df_full = pd.read_csv(TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes_train)
testKaggle = pd.read_csv(TEST_PATH, dtype=datatypes_test)

train_df, test_df = train_test_split(train_df_full, test_size=0.10, random_state=1)



## === cell 3
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## === cell 4
print("train_df clean")
train_df = clean(train_df)
print("test_df clean")
test_df = clean(test_df)



## === cell 5
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)

train_df = train_df.dropna(subset=["pickup_datetime"])
test_df = test_df.dropna(subset=["pickup_datetime"])
testKaggle = testKaggle.dropna(subset=["pickup_datetime"])



## === cell 6
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 7
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## === cell 8
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")



## === cell 9
dropped_columns = ["pickup_datetime"]

train_df = train_df.drop(dropped_columns + ["key"], axis=1)
test_df = test_df.drop(dropped_columns + ["key"], axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")



## === cell 10
feature_cols = [c for c in train_df.columns if c != "fare_amount"]

train_df = train_df[["fare_amount"] + feature_cols].copy()
test_df = test_df[["fare_amount"] + feature_cols].copy()
testKaggle_clean = testKaggle_clean.reindex(columns=feature_cols).copy()

for c in feature_cols:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
    test_df[c] = pd.to_numeric(test_df[c], errors="coerce")
    testKaggle_clean[c] = pd.to_numeric(testKaggle_clean[c], errors="coerce")

train_df = train_df.dropna(subset=feature_cols + ["fare_amount"])
test_df = test_df.dropna(subset=feature_cols + ["fare_amount"])
testKaggle_clean = testKaggle_clean.fillna(0.0)

scaler = preprocessing.MinMaxScaler()
train_scaled_values = scaler.fit_transform(train_df[feature_cols].values)
test_scaled_values = scaler.transform(test_df[feature_cols].values)
kaggle_scaled_values = scaler.transform(testKaggle_clean[feature_cols].values)

train_df_scaled = pd.DataFrame(
    train_scaled_values, columns=feature_cols, index=train_df.index
)
test_df_scaled = pd.DataFrame(
    test_scaled_values, columns=feature_cols, index=test_df.index
)
testKaggle_scaled = pd.DataFrame(
    kaggle_scaled_values, columns=feature_cols, index=testKaggle_clean.index
)

train_df_scaled.insert(
    0, "fare_amount", train_df["fare_amount"].astype("float32").values
)
test_df_scaled.insert(0, "fare_amount", test_df["fare_amount"].astype("float32").values)



## === cell 11
train_df_scaled, validation_df_scaled = train_test_split(
    train_df_scaled, test_size=0.10, random_state=1
)

train_df_main = train_df_scaled
validation_df_main = validation_df_scaled

print(train_df_scaled.shape)
print(validation_df_scaled.shape)
print(test_df_scaled.shape)
print("Num features:", len(feature_cols))



## === cell 12
train_labels = train_df_scaled["fare_amount"].values.astype("float32")
validation_labels = validation_df_scaled["fare_amount"].values.astype("float32")
test_labels = test_df_scaled["fare_amount"].values.astype("float32")

train_df_scaled = train_df_scaled.drop(["fare_amount"], axis=1)
validation_df_scaled = validation_df_scaled.drop(["fare_amount"], axis=1)
test_df_scaled = test_df_scaled.drop(["fare_amount"], axis=1)

train_df_scaled = train_df_scaled.astype("float32")
validation_df_scaled = validation_df_scaled.astype("float32")
test_df_scaled = test_df_scaled.astype("float32")
testKaggle_scaled = testKaggle_scaled.astype("float32")

print("Done with Labels")
print(train_labels.shape)
print(validation_labels.shape)
print(test_labels.shape)




## === cell 13
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))




## === cell 14
checkpoint = ModelCheckpoint(
    filepath="my_model.keras",
    verbose=1,
    save_best_only=True,
    monitor="val_loss",
    mode="min",
)

model = Sequential()
model.add(
    Dense(
        256,
        activation="linear",
        input_dim=train_df_scaled.shape[1],
        activity_regularizer=regularizers.l1(0.01),
    )
)
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="linear"))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    loss="mean_squared_error",
    optimizer=adam,
    metrics=["mae", "accuracy", rmse, "mse"],
)

print("Dataset size: %s" % DATASET_SIZE)
print("Epochs: %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % feature_cols)
model.summary()

history = model.fit(
    x=train_df_scaled,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[checkpoint],
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)



## === cell 15
plot_loss_accuracy_rmse(history)



## === cell 16
if os.path.exists("my_model.keras"):
    try:
        model.load_weights("my_model.keras")
        print("Loaded best weights from my_model.keras")
    except Exception as e:
        print("WARNING: Could not load weights from my_model.keras:", repr(e))

score = model.evaluate(train_df_scaled, train_labels, verbose=1)
print(score)
print("train mean_squared_error:", score[0])
print("train mae:", score[1])
print("train accuracy:", score[2])
print("train rmse:", score[3])
print("train mse:", score[4])



## === cell 17
score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print(score)
print("Validation mean_squared_error:", score[0])
print("Validation mae:", score[1])
print("Validation accuracy:", score[2])
print("Validation rmse:", score[3])
print("Validation mse:", score[4])



## === cell 18
score = model.evaluate(test_df_scaled, test_labels, verbose=1)
print(score)
print("Test mean_squared_error:", score[0])
print("Test mae:", score[1])
print("Test accuracy:", score[2])
print("Test rmse:", score[3])
print("Test mse:", score[4])



## === cell 19
validation_predictions = model.predict(validation_df_scaled, verbose=0).flatten()

plt.scatter(validation_labels, validation_predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot(
    [validation_predictions.min(), validation_predictions.max()],
    [validation_predictions.min(), validation_predictions.max()],
    "k--",
    lw=4,
)



## === cell 20
test_predictions = model.predict(test_df_scaled, verbose=0).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot(
    [test_predictions.min(), test_predictions.max()],
    [test_predictions.min(), test_predictions.max()],
    "k--",
    lw=4,
)



## === cell 21
fig, ax = plt.subplots()
ax.scatter(test_labels, test_predictions)
ax.plot(
    [test_labels.min(), test_labels.max()],
    [test_labels.min(), test_labels.max()],
    "k--",
    lw=4,
)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()



## === cell 22
plt.figure(figsize=(20, 10))
plt.plot(validation_labels[:100])
plt.plot(validation_predictions[:100])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()



## === cell 23
plt.figure(figsize=(20, 10))
plt.plot(test_labels[:100])
plt.plot(test_predictions[:100])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()



## === cell 24
error = validation_predictions - validation_labels
plt.hist(error, bins=100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## === cell 25
error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## === cell 26
print(len(error))
errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
print(len(errorGreaterZero))
plt.hist(errorGreaterZero, bins=100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## === cell 27
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1).reshape(
    -1
)

predictionKaggle = np.asarray(predictionKaggle, dtype="float32")
predictionKaggle = np.clip(predictionKaggle, 0.0, 250.0)

output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
print("Wrote:", SUBMISSION_NAME)
