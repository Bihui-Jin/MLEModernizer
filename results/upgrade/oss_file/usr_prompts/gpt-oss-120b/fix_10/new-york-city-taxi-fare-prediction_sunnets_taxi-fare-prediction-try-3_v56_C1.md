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

15.27794

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 479.2119) has done: 'The script now uses TensorFlow Keras (avoiding the protobuf import error), drops the non‑numeric `key` column before scaling, and fixes the data‑path handling. These changes unblock the pipeline, allow the scaler to work, and let the model train and produce a proper `submissiontry_water.csv` file with the required `key,fare_amount` columns.'
- What this solution (achieved 23.36162) has done: 'Cleaning up imports, fixing scaling mismatches, and correcting the Keras backend usage while keeping the core model unchanged. This resolves the protobuf import error, aligns test‑set features with the scaler, and provides a working RMSE metric so the pipeline runs end‑to‑end and creates a proper `submission.csv`.'
- What this solution (achieved 15.27794) has done: 'I replace the TensorFlow‑Keras imports with the standalone Keras (keras‑core) equivalents, adjust the custom RMSE metric to use Keras backend functions, and keep the rest of the pipeline unchanged. This removes the protobuf import error, lets the model train and evaluate, and ensures a proper `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras import Sequential
from keras.layers import Dense, BatchNormalization
from keras import regularizers, optimizers, callbacks, backend as K
from keras.metrics import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError

BASE_DIR = "/"

TRAIN_PATH = os.path.join(BASE_DIR, "kaggle", "data", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "kaggle", "data", "test.csv")

DATASET_SIZE = 50000

LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 20  # a few more epochs to improve validation RMSE
SUBMISSION_NAME = "submission.csv"


def remove_datapoints_from_water(df):
    try:
        import urllib.request
        from PIL import Image

        url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
        with urllib.request.urlopen(url) as resp:
            img = Image.open(resp)
            nyc_mask = np.array(img)[:, :, 0] > 0.9
        BB = (-74.5, -72.8, 40.5, 41.8)

        def lonlat_to_xy(lon, lat, dx, dy, BB):
            x = (dx * (lon - BB[0]) / (BB[1] - BB[0])).astype("int")
            y = (dy - dy * (lat - BB[2]) / (BB[3] - BB[2])).astype("int")
            return x, y

        pickup_x, pickup_y = lonlat_to_xy(
            df["pickup_longitude"].values,
            df["pickup_latitude"].values,
            nyc_mask.shape[1],
            nyc_mask.shape[0],
            BB,
        )
        dropoff_x, dropoff_y = lonlat_to_xy(
            df["dropoff_longitude"].values,
            df["dropoff_latitude"].values,
            nyc_mask.shape[1],
            nyc_mask.shape[0],
            BB,
        )
        idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]
        return df[idx]
    except Exception as e:
        print("Water‑mask removal skipped:", e)
        return df




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
    usecols=[0, 1, 2, 3, 4, 5, 6, 7],
)

testKaggle = pd.read_csv(
    TEST_PATH,
    dtype=datatypes,
    usecols=[0, 1, 2, 3, 4, 5, 6],
)




## === cell 2
def clean(df):
    print("  Old size:", len(df))
    df = df.dropna(how="any", axis="rows")
    print("  After dropna:", len(df))

    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        & (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    print("  After removing identical coords:", len(df))

    df = df[
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    ]
    print("  After removing zeros:", len(df))

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
    print("  After NYC bounds:", len(df))

    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    print("  After fare range filter:", len(df))

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print("  After passenger count filter:", len(df))

    airport_coords = {
        "nyc": (40.7141667, -74.0063889),
        "fk": (40.639722, -73.778889),
        "ewr": (40.6925, -74.168611),
        "lga": (40.77725, -73.872611),
        "sol": (40.6892, -74.0445),
    }
    for name, (lat, lon) in airport_coords.items():
        df = df[(df["pickup_longitude"] != lon) | (df["pickup_latitude"] != lat)]
        df = df[(df["dropoff_longitude"] != lon) | (df["dropoff_latitude"] != lat)]

    print("  After airport filters:", len(df))

    df = remove_datapoints_from_water(df)
    print("  After water mask:", len(df))
    return df


def late_night(row):
    return int(row["hour"] <= 3 or row["hour"] >= 22)


def night(row):
    return int(20 < row["hour"] <= 23 and row["weekday"] < 5)


def rush_hour(row):
    return int(16 <= row["hour"] <= 20 and row["weekday"] < 5)


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
    df["latdiff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
    df["londiff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
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
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {id_column: raw_test[id_column].values, prediction_column: prediction.ravel()}
    )
    df.to_csv(file_name, index=False)
    print("Submission written to:", file_name)


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Model loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

    if "rmse_keras" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["rmse_keras"], label="train")
        plt.plot(history.history["val_rmse_keras"], label="val")
        plt.title("Model RMSE")
        plt.xlabel("epoch")
        plt.legend()
        plt.show()


def rmse_keras(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))




## === cell 3
print("Cleaning train data")
train_df = clean(trainKaggle)

print("\nAdding time features")
train_df = add_time_features(train_df)
test_df = add_time_features(testKaggle.copy())

print("\nAdding coordinate features")
train_df = add_coordinate_features(train_df)
test_df = add_coordinate_features(test_df)

print("\nAdding distance features")
train_df = add_distances_features(train_df)
test_df = add_distances_features(test_df)

print("\nDropping non‑numeric / ID columns")
dropped_columns = ["key", "pickup_datetime"]
train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)



## === cell 4
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)

train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)

print("Shapes after split:")
print("  train:", train_df.shape, "labels:", train_labels.shape)
print("  validation:", validation_df.shape, "labels:", validation_labels.shape)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_df_scaled = scaler.transform(test_df)

print("Scaling completed")



## === cell 5
model = Sequential()
model.add(
    Dense(
        256,
        activation="relu",
        input_dim=train_df_scaled.shape[1],
        activity_regularizer=regularizers.l1(0.001),
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
    metrics=["mae", rmse_keras, "mse"],
)

print("Model summary:")
model.summary()



## === cell 6
early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

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



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3936378127.py in <cell line: 0>()
      1 early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
      2 
----> 3 history = model.fit(
      4     x=train_df_scaled,
      5     y=train_labels,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_55/4185155608.py in rmse_keras(y_true, y_pred)
    133 def rmse_keras(y_true, y_pred):
    134     # Keras backend implementation matching the original TF version
--> 135     return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
    136 
    137 

AttributeError: module 'keras.api.backend' has no attribute 'sqrt'

## === cell 7
plot_loss_accuracy_rmse(history)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2579130885.py in <cell line: 0>()
----> 1 plot_loss_accuracy_rmse(history)
      2 

NameError: name 'history' is not defined

## === cell 8
score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print("Evaluation on validation set (loss, mae, rmse, mse):", score)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/311578724.py in <cell line: 0>()
----> 1 score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
      2 print("Evaluation on validation set (loss, mae, rmse, mse):", score)
      3 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_55/4185155608.py in rmse_keras(y_true, y_pred)
    133 def rmse_keras(y_true, y_pred):
    134     # Keras backend implementation matching the original TF version
--> 135     return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
    136 
    137 

AttributeError: module 'keras.api.backend' has no attribute 'sqrt'

## === cell 9
prediction_val = model.predict(validation_df_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(test_df_scaled, batch_size=128, verbose=1)


def rmse_numpy(preds, targets):
    return np.sqrt(((preds.ravel() - targets) ** 2).mean())


rmse_val = rmse_numpy(prediction_val, validation_labels)
print("RMSE on validation split (numpy):", rmse_val)



## === cell 10
output_submission(
    raw_test=testKaggle,
    prediction=predictionKaggle,
    id_column="key",
    prediction_column="fare_amount",
    file_name=SUBMISSION_NAME,
)



## === cell 11
print("Sample validation prediction vs true:")
print(
    prediction_val[0].ravel(),
    validation_labels[0] if len(validation_labels) > 0 else "N/A",
)
