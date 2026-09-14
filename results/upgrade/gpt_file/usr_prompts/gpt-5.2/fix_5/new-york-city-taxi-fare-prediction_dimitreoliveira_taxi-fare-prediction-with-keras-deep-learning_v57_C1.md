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

12.78702

# 6. Current score

2021.1194

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 2021.11148) has done: 'I fix the early import/runtime crash by switching from `tf_keras` to the standard `tensorflow.keras` API that is stable on Kaggle, while keeping the same model architecture and training loop. I also fix the feature-mismatch bug by preserving `key` in `test` during processing (so it doesn’t get dropped) and by ensuring `fare_amount` is not accidentally included in test features passed to the scaler. Finally, I make train/test feature columns align deterministically (same columns, same order) before scaling and prediction, so `test_scaled` is created successfully and a valid `submission.csv` is written.'
- What this solution (achieved 120.11285) has done: 'I fix the import/runtime crash in the first cell that prevents the notebook from running by avoiding TensorFlow (which is triggering the protobuf `MessageFactory.GetPrototype` error in this environment) and switching to the already-installed `tf_keras` backend. Then I correct a major feature-engineering logic bug that makes test-time binned features inconsistent with training (using `qcut` separately on train and test), which is a key reason the score is extremely bad; I fit bin edges on train and apply the same bins to both train and test while keeping the same engineered feature idea. Finally, I keep the model architecture/training loop intact, ensure train/test columns align deterministically, and write a valid `submission.csv` with columns `key,fare_amount`.'
- What this solution (achieved 2021.1194) has done: 'I fix the immediate runtime crash caused by the protobuf/TensorFlow stack by switching from `tf_keras` to the stable `keras` (Keras 3) API already installed in your environment, keeping the exact same Sequential architecture and training loop. I also make binning robust by ensuring `_apply_bins` never returns NaNs (values outside train bin edges be clipped into the edge bins), which should substantially improve RMSE without changing the overall feature idea. Finally, I keep train/test feature alignment deterministic and ensure the script always writes a valid `submission.csv` with the required `key,fare_amount` columns.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras import optimizers, regularizers

SEED = 1
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
try:
    keras.utils.set_random_seed(SEED)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def clean(df):
    df = df[(-76 <= df["pickup_longitude"]) & (df["pickup_longitude"] <= -72)]
    df = df[(-76 <= df["dropoff_longitude"]) & (df["dropoff_longitude"] <= -72)]
    df = df[(38 <= df["pickup_latitude"]) & (df["pickup_latitude"] <= 42)]
    df = df[(38 <= df["dropoff_latitude"]) & (df["dropoff_latitude"] <= 42)]
    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 250)]
    df = df[(df["dropoff_longitude"] != df["pickup_longitude"])]
    df = df[(df["dropoff_latitude"] != df["pickup_latitude"])]
    return df


def late_night(row):
    if (row["hour"] <= 6) or (row["hour"] >= 20):
        return 1
    else:
        return 0


def night(row):
    if ((row["hour"] <= 20) and (row["hour"] >= 16)) and (row["weekday"] < 5):
        return 1
    else:
        return 0


_BIN_EDGES = {}


def _fit_bin_edges(train_df, col, q=16):
    edges = pd.qcut(train_df[col], q, retbins=True, duplicates="drop")[1]
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.array([train_df[col].min(), train_df[col].max()])
    return edges


def _apply_bins(df, col, edges):
    x = df[col].astype("float32")
    lo = float(edges[0])
    hi = float(edges[-1])
    x = x.clip(lower=lo, upper=hi)
    b = pd.cut(x, bins=edges, labels=False, include_lowest=True)
    return b.astype("float32").fillna(0).astype("int32")


def process(df, fit_bins=False):
    key = df["key"] if "key" in df.columns else None

    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    )
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday

    df["night"] = df.apply(lambda x: night(x), axis=1)
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1)

    cols_to_bin = [
        "pickup_longitude",
        "dropoff_longitude",
        "pickup_latitude",
        "dropoff_latitude",
    ]

    global _BIN_EDGES
    if fit_bins:
        for c in cols_to_bin:
            _BIN_EDGES[c] = _fit_bin_edges(df, c, q=16)

    df["pickup_longitude_binned"] = _apply_bins(
        df, "pickup_longitude", _BIN_EDGES["pickup_longitude"]
    )
    df["dropoff_longitude_binned"] = _apply_bins(
        df, "dropoff_longitude", _BIN_EDGES["dropoff_longitude"]
    )
    df["pickup_latitude_binned"] = _apply_bins(
        df, "pickup_latitude", _BIN_EDGES["pickup_latitude"]
    )
    df["dropoff_latitude_binned"] = _apply_bins(
        df, "dropoff_latitude", _BIN_EDGES["dropoff_latitude"]
    )

    df = df.drop("pickup_datetime", axis=1)

    if key is not None:
        df["key"] = key.values
    return df


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_relevant_distances(df):
    ny = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)

    df["downtown_pickup_distance"] = manhattan(
        ny[1], ny[0], df["pickup_latitude"], df["pickup_longitude"]
    )
    df["downtown_dropoff_distance"] = manhattan(
        ny[1], ny[0], df["dropoff_latitude"], df["dropoff_longitude"]
    )
    df["jfk_pickup_distance"] = manhattan(
        jfk[1], jfk[0], df["pickup_latitude"], df["pickup_longitude"]
    )
    df["jfk_dropoff_distance"] = manhattan(
        jfk[1], jfk[0], df["dropoff_latitude"], df["dropoff_longitude"]
    )
    df["ewr_pickup_distance"] = manhattan(
        ewr[1], ewr[0], df["pickup_latitude"], df["pickup_longitude"]
    )
    df["ewr_dropoff_distance"] = manhattan(
        ewr[1], ewr[0], df["dropoff_latitude"], df["dropoff_longitude"]
    )
    df["lgr_pickup_distance"] = manhattan(
        lgr[1], lgr[0], df["pickup_latitude"], df["pickup_longitude"]
    )
    df["lgr_dropoff_distance"] = manhattan(
        lgr[1], lgr[0], df["dropoff_latitude"], df["dropoff_longitude"]
    )
    return df


def add_engineered(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    latdiff = lat1 - lat2
    londiff = lon1 - lon2
    euclidean = (latdiff**2 + londiff**2) ** 0.5

    df["latdiff"] = latdiff
    df["londiff"] = londiff
    df["euclidean"] = euclidean
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)

    df = pd.get_dummies(df, columns=["weekday"])
    df = pd.get_dummies(df, columns=["month"])
    return df




## === cell 2
def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print(
        "Output complete:",
        file_name,
        "shape=",
        df[[id_column, prediction_column]].shape,
    )


def plot_loss_accuracy(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper right")
    plt.show()




## === cell 3
TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SUBMISSION_NAME = "submission.csv"

BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 0.0001
DATASET_SIZE = 700000



## === cell 4
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

train = pd.read_csv(
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

test = pd.read_csv(
    TEST_PATH,
    dtype={k: v for k, v in datatypes.items() if k != "fare_amount"},
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



## === cell 5
train = clean(train)

train = process(train, fit_bins=True)
test = process(test, fit_bins=False)

train = add_relevant_distances(train)
test = add_relevant_distances(test)

train = add_engineered(train)
test = add_engineered(test)

feature_cols = [c for c in train.columns if c not in ["fare_amount"]]
if "key" not in feature_cols:
    feature_cols = ["key"] + feature_cols

all_feature_cols = sorted(set(feature_cols) | set(test.columns))
all_feature_cols = (["key"] if "key" in all_feature_cols else []) + [
    c for c in all_feature_cols if c != "key"
]

train = train.reindex(columns=(["fare_amount"] + all_feature_cols), fill_value=0)
test = test.reindex(columns=all_feature_cols, fill_value=0)

train = train.fillna(0)
test = test.fillna(0)



## === cell 6
dropped_columns = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]

train_clean = train.drop(dropped_columns, axis=1)
test_clean = test.drop(dropped_columns, axis=1)

train_clean.head(5)



## === cell 7
train_df, validation_df = train_test_split(train_clean, test_size=0.10, random_state=1)

train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values

train_df = train_df.drop(["fare_amount", "key"], axis=1, errors="ignore")
validation_df = validation_df.drop(["fare_amount", "key"], axis=1, errors="ignore")



## === cell 8
test_key = test_clean["key"].values if "key" in test_clean.columns else None
test_features = test_clean.drop(["key"], axis=1, errors="ignore")

test_features = test_features.reindex(columns=train_df.columns, fill_value=0)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_features)



## === cell 9
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
model.add(Dense(128, activation="relu", activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu", activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Dense(32, activation="relu", activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Dense(16, activation="relu", activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Dense(1))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mse", optimizer=adam, metrics=["mae"])



## === cell 10
print("Dataset size: %s" % DATASET_SIZE)
print("Epochs: %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % list(train_df.columns))



## === cell 11
model.summary()



## === cell 12
history = model.fit(
    x=train_df_scaled,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)



## === cell 13
plot_loss_accuracy(history)



## === cell 14
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
prediction = prediction.reshape(-1, 1)
prediction = np.clip(prediction, 0, None)



## === cell 15
raw_test = pd.read_csv(TEST_PATH, usecols=["key"])
output_submission(raw_test, prediction, "key", "fare_amount", SUBMISSION_NAME)
