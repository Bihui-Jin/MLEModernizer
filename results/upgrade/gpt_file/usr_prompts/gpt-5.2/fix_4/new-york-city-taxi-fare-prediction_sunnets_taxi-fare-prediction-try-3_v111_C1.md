# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras import backend

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)
tf.random.set_seed(42)
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow:", tf.__version__)

TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 1000
EPOCHS = 100
LEARNING_RATE = 0.001

DATASET_SIZE = 2000000

assert os.path.exists(TRAIN_PATH), f"Missing {TRAIN_PATH}"
assert os.path.exists(TEST_PATH), f"Missing {TEST_PATH}"


def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    has_target = "fare_amount" in df.columns

    MinMax = (-74.5, -72.8, 40.5, 41.8)
    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)  # Statue of Liberty

    m = np.ones(len(df), dtype=bool)

    m &= (df["dropoff_longitude"].values != df["pickup_longitude"].values) & (
        df["dropoff_latitude"].values != df["pickup_latitude"].values
    )
    print(" New size after removing same long lat: %d" % int(m.sum()))

    plon = df["pickup_longitude"].values
    plat = df["pickup_latitude"].values
    dlon = df["dropoff_longitude"].values
    dlat = df["dropoff_latitude"].values
    m &= (dlon != 0) & (plon != 0) & (dlat != 0) & (plat != 0)
    print(" New size after removing 0 long lat: %d" % int(m.sum()))

    m &= (MinMax[0] <= plon) & (plon <= MinMax[1])
    m &= (MinMax[0] <= dlon) & (dlon <= MinMax[1])
    m &= (MinMax[2] <= plat) & (plat <= MinMax[3])
    m &= (MinMax[2] <= dlat) & (dlat <= MinMax[3])
    print(" New size after NYC lang lot: %d" % int(m.sum()))

    print(" New size after lang lot > 0: %d" % int(m.sum()))

    m &= np.abs(plat - dlat) > 0.001
    m &= np.abs(plon - dlon) > 0.001
    print(" New size after lang - lot > 0.001: %d" % int(m.sum()))
    print(" New size after only NYC: %d" % int(m.sum()))

    if has_target:
        fare = df["fare_amount"].values
        m &= (0 < fare) & (fare <= 50)
        print(" New size after removing outliers: %d" % int(m.sum()))
    else:
        print(" Skipping fare_amount outlier removal (no fare_amount column).")

    pc = df["passenger_count"].values
    m &= pc > 0
    print(" New size after removing 6=>passenger_count > 0 : %d" % int(m.sum()))

    m &= (plon != nyc_coord[1]) & (plat != nyc_coord[0])
    m &= (dlon != nyc_coord[1]) & (dlat != nyc_coord[0])
    print(" New size after NY airport: %d" % int(m.sum()))

    m &= (plon != fk_coord[1]) & (plat != fk_coord[0])
    m &= (dlon != fk_coord[1]) & (dlat != fk_coord[0])
    print(" New size after jfk airport: %d" % int(m.sum()))

    m &= (plon != ewr_coord[1]) & (plat != ewr_coord[0])
    m &= (dlon != ewr_coord[1]) & (dlat != ewr_coord[0])
    print(" New size after ewr airport: %d" % int(m.sum()))

    m &= (plon != lga_coord[1]) & (plat != lga_coord[0])
    m &= (dlon != lga_coord[1]) & (dlat != lga_coord[0])
    print(" New size after lgr airport: %d" % int(m.sum()))

    m &= (plon != sol_coord[1]) & (plat != sol_coord[0])
    m &= (dlon != sol_coord[1]) & (dlat != sol_coord[0])
    print(" New size after sol removed: %d" % int(m.sum()))

    df = df.loc[m].copy()
    print("Old size: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def remove_datapoints_from_water(df):
    return df


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    )
    df = df.dropna(subset=["pickup_datetime"])
    dt = df["pickup_datetime"].dt
    df["year"] = dt.year
    df["month"] = dt.month
    df["day"] = dt.day
    df["hour"] = dt.hour
    df["minute"] = dt.minute
    df["second"] = dt.second
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
    df = pd.DataFrame(
        {id_column: raw_test[id_column].values, prediction_column: prediction}
    )
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()

    if "rmse" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["rmse"])
        if "val_rmse" in history.history:
            plt.plot(history.history["val_rmse"])
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper right")
        plt.show()




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

train_usecols = list(datatypes_train.keys())
test_usecols = list(datatypes_test.keys())

train_df = pd.read_csv(
    TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes_train, usecols=train_usecols
)
testKaggle = pd.read_csv(TEST_PATH, dtype=datatypes_test, usecols=test_usecols)

test_df = None

print("Loaded train_df:", train_df.shape, "testKaggle:", testKaggle.shape)



## === cell 2
train_df, test_df = train_test_split(train_df, test_size=0.10, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Split train_df:", train_df.shape, "test_df:", test_df.shape)



## === cell 3
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## === cell 4
print("train_df clean")
train_df = clean(train_df)
print("test_df clean")
test_df = clean(test_df)

print("testKaggle clean")
testKaggle = clean(testKaggle)



## === cell 5
train_df = train_df.dropna()
test_df = test_df.dropna()
testKaggle = testKaggle.dropna()

print("After dropna:", train_df.shape, test_df.shape, testKaggle.shape)



## === cell 6
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 7
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 8
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## === cell 9
try:
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")
    plt.show()
except Exception as e:
    print("Plotting skipped due to:", repr(e))



## === cell 10
dropped_columns = ["pickup_datetime", "key"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(["pickup_datetime", "key"], axis=1)

print("Done with dropped_columns")
print("train_df cols:", train_df.columns.tolist())
print("test_df cols:", test_df.columns.tolist())
print("testKaggle_clean cols:", testKaggle_clean.columns.tolist())



## === cell 11
train_df_scaled = train_df.copy()
test_df_scaled = test_df.copy()
testKaggle_scaled = testKaggle_clean.copy()

scaler = preprocessing.MinMaxScaler()

scale_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "passenger_count",
    "manhattan",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "latdiff",
    "londiff",
]
scale_cols = [c for c in scale_cols if c in train_df_scaled.columns]

train_df_scaled[scale_cols] = scaler.fit_transform(train_df_scaled[scale_cols])
test_df_scaled[scale_cols] = scaler.transform(test_df_scaled[scale_cols])
if all(c in testKaggle_scaled.columns for c in scale_cols):
    testKaggle_scaled[scale_cols] = scaler.transform(testKaggle_scaled[scale_cols])

print(
    "Scaling done:",
    train_df_scaled.shape,
    test_df_scaled.shape,
    testKaggle_scaled.shape,
)



## === cell 12
train_df_scaled, validation_df_scaled = train_test_split(
    train_df_scaled, test_size=0.10, random_state=1
)



## === cell 13
train_df_main = train_df_scaled
validation_df_main = validation_df_scaled



## === cell 14
print(train_df_scaled.shape)
print(validation_df_scaled.shape)
print(test_df_scaled.shape)



## === cell 15
train_labels = train_df_scaled["fare_amount"].values.astype(np.float32)
validation_labels = validation_df_scaled["fare_amount"].values.astype(np.float32)
test_labels = test_df_scaled["fare_amount"].values.astype(np.float32)

train_df_scaled = train_df_scaled.drop(["fare_amount"], axis=1)
validation_df_scaled = validation_df_scaled.drop(["fare_amount"], axis=1)
test_df_scaled = test_df_scaled.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 16
print(train_labels.shape)
print(validation_labels.shape)
print(test_labels.shape)




## === cell 17
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))




## === cell 18
X_train = train_df_scaled.to_numpy(dtype=np.float32, copy=False)
X_val = validation_df_scaled.to_numpy(dtype=np.float32, copy=False)
X_test = test_df_scaled.to_numpy(dtype=np.float32, copy=False)

checkpoint = ModelCheckpoint(
    filepath="my_model.weights.h5",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
)

model = Sequential()
model.add(
    Dense(
        256,
        activation="linear",
        input_dim=X_train.shape[1],
        activity_regularizer=regularizers.l1(0.01),
    )
)
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="linear"))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mean_squared_error", optimizer=adam, metrics=["mae", rmse, "mse"])

print("Dataset size (initial read nrows): %s" % DATASET_SIZE)
print("Epochs: %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % X_train.shape[1])
print("Features used (post-drop): %s" % train_df_scaled.columns.tolist())
model.summary()

history = model.fit(
    x=X_train,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[checkpoint],
    validation_data=(X_val, validation_labels),
    shuffle=True,
)

try:
    model.load_weights("my_model.weights.h5")
    print("Loaded best checkpoint weights from my_model.weights.h5")
except Exception as e:
    print("Could not load checkpoint weights:", repr(e))



## === cell 19
try:
    from IPython.display import SVG
    from tensorflow.keras.utils import model_to_dot

    SVG(model_to_dot(model).create(prog="dot", format="svg"))
except Exception as e:
    print("Model visualization skipped due to:", repr(e))



## === cell 20
try:
    plot_loss_accuracy_rmse(history)
except Exception as e:
    print("Plot loss skipped due to:", repr(e))



## === cell 21
score = model.evaluate(X_train, train_labels, verbose=1)
print(score)
print("train mean_squared_error:", score[0])
print("train mae:", score[1])
print("train rmse:", score[2])
print("train mse:", score[3])



## === cell 22
score = model.evaluate(X_val, validation_labels, verbose=1)
print(score)
print("Validation mean_squared_error:", score[0])
print("Validation mae:", score[1])
print("Validation rmse:", score[2])
print("Validation mse:", score[3])



## === cell 23
score = model.evaluate(X_test, test_labels, verbose=1)
print(score)
print("Test mean_squared_error:", score[0])
print("Test mae:", score[1])
print("Test rmse:", score[2])
print("Test mse:", score[3])



## === cell 24
missing_cols = [
    c for c in train_df_scaled.columns if c not in testKaggle_scaled.columns
]
extra_cols = [c for c in testKaggle_scaled.columns if c not in train_df_scaled.columns]

if missing_cols:
    raise ValueError(f"Missing columns in testKaggle_scaled: {missing_cols}")
if extra_cols:
    testKaggle_scaled = testKaggle_scaled.drop(columns=extra_cols)

testKaggle_scaled = testKaggle_scaled[train_df_scaled.columns]
print("Aligned testKaggle_scaled:", testKaggle_scaled.shape)



## === cell 25
validation_predictions = model.predict(X_val, batch_size=2048, verbose=1).flatten()

try:
    plt.scatter(validation_labels, validation_predictions, s=2)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.axis("equal")
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot(
        [validation_predictions.min(), validation_predictions.max()],
        [validation_predictions.min(), validation_predictions.max()],
        "k--",
        lw=2,
    )
    plt.show()
except Exception as e:
    print("Validation scatter skipped due to:", repr(e))



## === cell 26
test_predictions = model.predict(X_test, batch_size=2048, verbose=1).flatten()

try:
    plt.scatter(test_labels, test_predictions, s=2)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.axis("equal")
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot(
        [test_predictions.min(), test_predictions.max()],
        [test_predictions.min(), test_predictions.max()],
        "k--",
        lw=2,
    )
    plt.show()
except Exception as e:
    print("Test scatter skipped due to:", repr(e))



## === cell 27
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_labels[np.argmax(test_predictions)])
print(test_df.iloc[np.argmax(test_predictions)])



## === cell 28
print(np.argmin(test_predictions))
print(test_predictions[np.argmin(test_predictions)])
print(test_labels[np.argmin(test_predictions)])
print(test_df.iloc[np.argmin(test_predictions)])



## === cell 29
try:
    fig, ax = plt.subplots()
    ax.scatter(test_labels, test_predictions, s=2)
    ax.plot(
        [test_labels.min(), test_labels.max()],
        [test_labels.min(), test_labels.max()],
        "k--",
        lw=2,
    )
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    plt.show()
except Exception as e:
    print("Measured vs Predicted plot skipped due to:", repr(e))



## === cell 30
try:
    plt.figure(figsize=(20, 10))
    plt.plot(validation_labels[:100])
    plt.plot(validation_predictions[:100])
    plt.title("Prediction vs Actual")
    plt.ylabel("Fare Amount")
    plt.xlabel("Transaction")
    plt.legend(["Actual", "prediction"], loc="upper right")
    plt.show()
except Exception as e:
    print("Validation series plot skipped due to:", repr(e))



## === cell 31
try:
    plt.figure(figsize=(20, 10))
    plt.plot(test_labels[:100])
    plt.plot(test_predictions[:100])
    plt.title("Prediction vs Actual")
    plt.ylabel("Fare Amount")
    plt.xlabel("Transaction")
    plt.legend(["Actual", "prediction"], loc="upper right")
    plt.show()
except Exception as e:
    print("Test series plot skipped due to:", repr(e))



## === cell 32
error = validation_predictions - validation_labels
try:
    plt.hist(error, bins=100)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
    plt.show()
except Exception as e:
    print("Validation error hist skipped due to:", repr(e))



## === cell 33
error = test_predictions - test_labels
try:
    plt.hist(error, bins=50)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
    plt.show()
except Exception as e:
    print("Test error hist skipped due to:", repr(e))



## === cell 34
print(len(error))
errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
print(len(errorGreaterZero))
try:
    plt.hist(errorGreaterZero, bins=100)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
    plt.show()
except Exception as e:
    print("ErrorGreaterZero hist skipped due to:", repr(e))



## === cell 35
X_kaggle = testKaggle_scaled.to_numpy(dtype=np.float32, copy=False)
predictionKaggle = model.predict(X_kaggle, batch_size=2048, verbose=1)



## === cell 36
print(predictionKaggle[:5].reshape(-1))



## === cell 37
predictionKaggle = np.maximum(predictionKaggle.reshape(-1).astype(np.float32), 0.0)
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
assert SUBMISSION_NAME.endswith(".csv") and os.path.exists(SUBMISSION_NAME)
print("Wrote submission:", SUBMISSION_NAME)
