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

4.16348

# 6. Current score

5.76649

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.16922) has done: 'I fix the runtime errors caused by (1) Keras 3 API changes (optimizer naming/arguments and removed visualization utilities) and (2) trying to read a URL mask image inside `plt.imread`, which fails in Kaggle’s environment. To improve the RMSE toward your target with minimal core-logic change, I correct obvious feature-engineering bugs (the time-of-day indicator functions have always-true conditions) and stop accidentally training/evaluating on a random 50% split as if it were “test” (that split should be a validation set only). Finally, I ensure the script reads the `key` column for train/test alignment and always writes a valid `key,fare_amount` submission CSV.'
- What this solution (achieved 6.12509) has done: 'I fix the immediate runtime failures by (1) using the correct TensorFlow backend for the custom RMSE metric (Keras 3 no longer exposes `backend.sqrt`) and (2) avoiding the protobuf `MessageFactory.GetPrototype` crash by importing `tf_keras` (the TensorFlow-bundled Keras) instead of standalone `keras`. These changes keep your model architecture/training loop identical while making training/evaluation run end-to-end. I also keep the same file paths but add a safe fallback to `/kaggle/data/...` so it runs in your provided environment layout. Finally, the script always write a valid `key,fare_amount` submission CSV with the requested `.csv` suffix.'
- What this solution (achieved 5.80517) has done: 'I fix the `MessageFactory.GetPrototype` crash by making the runtime use the TensorFlow-bundled Keras (`tf_keras`) consistently and by forcing a protobuf implementation that avoids the known incompatibility in Kaggle images. Then I make one score-improving, core-logic-preserving adjustment: stop dropping `passenger_count` (it’s a key predictive feature for this competition and removing it commonly worsens RMSE). Finally, I add small stability guards (drop rows with invalid parsed datetimes after feature creation, align feature columns between train/test, and ensure the submission is always written with the required `key,fare_amount` columns and `.csv` suffix).'
- What this solution (achieved 5.95737) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing a compatible protobuf runtime mode before importing TensorFlow/Keras, and by pinning the pure‑python protobuf implementation via environment variables early. This is a runtime-only fix and does not change your model, features, training loop, or evaluation semantics. I also add a small safety guard to ensure the submission keys align exactly with the test rows after feature engineering and scaling, so the produced CSV is always valid. No score-tuning changes are introduced beyond restoring correct execution; once it runs, your existing logic should again achieve your previously observed RMSE range (and move toward the target vs. “no submission”).'
- What this solution (achieved 6.15795) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by forcing a compatible protobuf implementation before importing TensorFlow, and by switching to the Kaggle-compatible `tensorflow`-bundled Keras API (`tf.keras`) consistently. This is a runtime-only change and preserves your model architecture, training loop, and features. To nudge RMSE toward your target with minimal semantic change, I add one common-sense fare-specific feature (`abs` versions of lat/lon diffs) and keep everything else intact; this typically improves distance signal without changing the overall approach. Finally, I keep the same I/O paths and ensure a valid `key,fare_amount` submission CSV is always written.'
- What this solution (achieved 5.76649) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from running by forcing the pure‑python protobuf implementation early and by importing the TensorFlow-bundled Keras (`tf_keras`) consistently (this avoids the `MessageFactory.GetPrototype` error in many Kaggle images). I also add a small, score-improving but core-logic-preserving feature: a true geographic distance (Haversine) computed from the same lat/lon inputs you already use; this typically reduces RMSE vs. simple Euclidean-in-degrees without changing the model or training loop. Finally, I keep the existing paths/outputs but harden submission writing by ensuring feature alignment, numeric dtypes, and a valid `key,fare_amount` CSV is always produced.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, BatchNormalization
from tf_keras import optimizers, regularizers

np.random.seed(1)
tf.random.set_seed(1)

TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/data/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/data/test.csv"

SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

print("TRAIN_PATH:", TRAIN_PATH)
print("TEST_PATH:", TEST_PATH)
print("Will write:", SUBMISSION_NAME)




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

    print(" New size after only NYC: %d" % len(df))
    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    print(" New size after removing outliers: %d" % len(df))

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
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def late_night(row):
    return 1 if (row["hour"] <= 3) else 0


def night(row):
    return (
        1
        if (((row["hour"] >= 20) or (row["hour"] <= 5)) and (row["weekday"] < 5))
        else 0
    )


def rush_hour(row):
    return 1 if ((16 <= row["hour"] <= 20) and (row["weekday"] < 5)) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    )
    df["year"] = df["pickup_datetime"].dt.year.astype("Int64")
    df["month"] = df["pickup_datetime"].dt.month.astype("Int64")
    df["day"] = df["pickup_datetime"].dt.day.astype("Int64")
    df["hour"] = df["pickup_datetime"].dt.hour.astype("Int64")
    df["weekday"] = df["pickup_datetime"].dt.weekday.astype("Int64")
    df["pickup_datetime"] = df["pickup_datetime"].astype(str)

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

    df["abs_latdiff"] = np.abs(df["latdiff"])
    df["abs_londiff"] = np.abs(df["londiff"])
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = np.sqrt(np.abs(lon1 - lon2) ** 2 + np.abs(lat1 - lat2) ** 2)

    lat1r = np.radians(lat1.astype("float64"))
    lat2r = np.radians(lat2.astype("float64"))
    dlat = lat2r - lat1r
    dlon = np.radians(lon2.astype("float64") - lon1.astype("float64"))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    df["haversine_km"] = (6371.0 * c).astype("float32")

    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {
            id_column: raw_test[id_column].values,
            prediction_column: prediction.reshape(-1),
        }
    )
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows=", len(df))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 8))
    plt.plot(history.history["loss"])
    plt.plot(history.history.get("val_loss", []))
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()

    plt.figure(figsize=(20, 8))
    plt.plot(history.history.get("rmse", []))
    plt.plot(history.history.get("val_rmse", []))
    plt.title("Model rmse")
    plt.ylabel("rmse")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()




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
        "key",
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
    dtype={
        "key": "str",
        "pickup_datetime": "str",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
)




## === cell 3
train_df = trainKaggle.copy()

print("Kaggle test Size %d" % len(testKaggle))
print("Train Size %d" % len(train_df))




## === cell 4
train_df.describe()




## === cell 5
print("train_df clean")
train_df = clean(train_df)




## === cell 6
train_df.describe()




## === cell 7
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)

time_cols = ["year", "month", "day", "hour", "weekday"]
before = len(train_df)
train_df = train_df.dropna(subset=time_cols)
print("Dropped rows with invalid datetime:", before - len(train_df))




## === cell 8
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)




## === cell 9
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")




## === cell 10
dropped_columns = ["pickup_datetime"]

test_keys = testKaggle["key"].copy()

train_df = train_df.drop(dropped_columns + ["key"], axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

train_feature_cols = [c for c in train_df.columns if c != "fare_amount"]
testKaggle_clean = testKaggle_clean.reindex(columns=train_feature_cols)

testKaggle_clean = testKaggle_clean.astype("float32", errors="ignore")

print("Done with dropped_columns")
print("Train features shape:", train_df.shape)
print("Test features shape:", testKaggle_clean.shape)




## === cell 11
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)

train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)

print("Done with Labels")
print("Train X:", train_df.shape, "Train y:", train_labels.shape)
print("Val X:", validation_df.shape, "Val y:", validation_labels.shape)




## === cell 12
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)




## === cell 13
def rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=-1))




## === cell 14
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
model.compile(loss="mean_squared_error", optimizer=adam, metrics=["mae", rmse, "mse"])

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
    shuffle=True,
)




## === cell 15
plot_loss_accuracy_rmse(history)




## === cell 16
val_score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print("Validation metrics:", dict(zip(model.metrics_names, val_score)))




## === cell 17
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)
predictionKaggle = np.clip(predictionKaggle, 0.0, None)

assert len(predictionKaggle) == len(test_keys), (len(predictionKaggle), len(test_keys))

submission_df = pd.DataFrame(
    {"key": test_keys.values, "fare_amount": predictionKaggle.reshape(-1)}
)
submission_df.to_csv(SUBMISSION_NAME, index=False)
print("Wrote submission:", SUBMISSION_NAME)
print(submission_df.head())
print("Rows:", len(submission_df), "Columns:", submission_df.columns.tolist())
assert SUBMISSION_NAME.endswith(".csv")
assert submission_df.columns.tolist() == ["key", "fare_amount"]
assert len(submission_df) == len(testKaggle)
