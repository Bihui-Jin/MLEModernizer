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

14.59601

# 6. Current score

None

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 49.74386) has done: 'I fix the immediate protobuf/TensorFlow import crash by removing the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` override and forcing a safe import order so `tf_keras` can load its dependencies correctly in this Kaggle environment. I also adjust the cell numbering to start at 1 (your current script starts at cell 0) so the notebook-style execution is consistent. To ensure a valid submission is always produced, I add a small fallback that writes a sample-mean submission if training fails unexpectedly (this is score-neutral versus “no submission” and only activates on exception). No changes are made to the model architecture, feature engineering, training loop, or loss/metrics.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.pop("KERAS_BACKEND", None)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense
from tf_keras import optimizers
from tf_keras import regularizers
from tf_keras import backend
from tf_keras.callbacks import ModelCheckpoint

np.random.seed(1)
try:
    keras.utils.set_random_seed(1)
except Exception:
    pass

TRAIN_PATH_CANDIDATES = [
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/data/train.csv",
]
TEST_PATH_CANDIDATES = [
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/data/test.csv",
]


def _pick_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[-1]


TRAIN_PATH = _pick_existing_path(TRAIN_PATH_CANDIDATES)
TEST_PATH = _pick_existing_path(TEST_PATH_CANDIDATES)

SUBMISSION_NAME = "submissiontry_water.csv"  # must end with .csv

BATCH_SIZE = 1000
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

print("Using TRAIN_PATH:", TRAIN_PATH)
print("Using TEST_PATH :", TEST_PATH)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def remove_datapoints_from_water(df):
    """
    Original implementation relied on external resources (URL image mask).
    Kaggle has no internet; keep pipeline runnable by using identity filter.
    """
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
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
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
    pred = np.asarray(prediction).reshape(-1)
    df = pd.DataFrame({id_column: raw_test[id_column].values, prediction_column: pred})
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "shape=", df.shape)


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history.get("loss", []))
    plt.plot(history.history.get("val_loss", []))
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()

    plt.figure(figsize=(20, 10))
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

trainKaggle = pd.read_csv(TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes)
testKaggle = pd.read_csv(
    TEST_PATH, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
)



## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)

train_df.to_csv("trainMehrak.csv", index=False)
test_df = test_df[:10000].copy()
test_df.to_csv("testMehrak.csv", index=False)



## === cell 4
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## === cell 5
print("train_df clean")
train_df = clean(train_df)
print("test_df clean")
test_df = clean(test_df)



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
    print("Plotting skipped:", repr(e))



## === cell 10
dropped_columns = ["pickup_datetime", "key"]

train_df = train_df.drop(dropped_columns, axis=1, errors="ignore")
test_df = test_df.drop(dropped_columns, axis=1, errors="ignore")

testKaggle_features = testKaggle.drop(
    ["pickup_datetime", "key"], axis=1, errors="ignore"
)

print("Done with dropped_columns")



## === cell 11
feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "latdiff",
    "londiff",
    "manhattan",
]

for c in feature_cols:
    if c not in train_df.columns:
        raise KeyError(f"Missing feature in train_df: {c}")
    if c not in test_df.columns:
        raise KeyError(f"Missing feature in test_df: {c}")
    if c not in testKaggle_features.columns:
        raise KeyError(f"Missing feature in testKaggle_features: {c}")

train_df = train_df[feature_cols + ["fare_amount"]].copy()
test_df = test_df[feature_cols + ["fare_amount"]].copy()
testKaggle_features = testKaggle_features[feature_cols].copy()

for c in feature_cols:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce").astype("float32")
    test_df[c] = pd.to_numeric(test_df[c], errors="coerce").astype("float32")
    testKaggle_features[c] = pd.to_numeric(
        testKaggle_features[c], errors="coerce"
    ).astype("float32")

train_df["fare_amount"] = pd.to_numeric(
    train_df["fare_amount"], errors="coerce"
).astype("float32")
test_df["fare_amount"] = pd.to_numeric(test_df["fare_amount"], errors="coerce").astype(
    "float32"
)

train_df = train_df.dropna(axis=0, how="any")
test_df = test_df.dropna(axis=0, how="any")
testKaggle_features = testKaggle_features.dropna(axis=0, how="any")



## === cell 12
train_df_scaled = train_df.copy()
test_df_scaled = test_df.copy()
testKaggle_scaled = testKaggle_features.copy()

scalers = {}
for col in feature_cols:
    sc = preprocessing.MinMaxScaler()
    train_df_scaled[[col]] = sc.fit_transform(train_df[[col]])
    test_df_scaled[[col]] = sc.transform(test_df[[col]])
    testKaggle_scaled[[col]] = sc.transform(testKaggle_features[[col]])
    scalers[col] = sc

scaler_y = preprocessing.MinMaxScaler()
train_df_scaled[["fare_amount"]] = scaler_y.fit_transform(train_df[["fare_amount"]])
test_df_scaled[["fare_amount"]] = scaler_y.transform(test_df[["fare_amount"]])

train_df_scaled[feature_cols] = train_df_scaled[feature_cols].astype("float32")
test_df_scaled[feature_cols] = test_df_scaled[feature_cols].astype("float32")
testKaggle_scaled[feature_cols] = testKaggle_scaled[feature_cols].astype("float32")
train_df_scaled[["fare_amount"]] = train_df_scaled[["fare_amount"]].astype("float32")
test_df_scaled[["fare_amount"]] = test_df_scaled[["fare_amount"]].astype("float32")



## === cell 13
train_df_scaled, validation_df_scaled = train_test_split(
    train_df_scaled, test_size=0.10, random_state=1
)



## === cell 14
train_df_main = train_df_scaled
validation_df_main = validation_df_scaled



## === cell 15
print(train_df_scaled.shape)
print(validation_df_scaled.shape)
print(test_df_scaled.shape)



## === cell 16
train_labels = train_df_scaled["fare_amount"].values.astype("float32")
validation_labels = validation_df_scaled["fare_amount"].values.astype("float32")
test_labels = test_df_scaled["fare_amount"].values.astype("float32")

train_df_scaled = train_df_scaled.drop(["fare_amount"], axis=1)
validation_df_scaled = validation_df_scaled.drop(["fare_amount"], axis=1)
test_df_scaled = test_df_scaled.drop(["fare_amount"], axis=1)

train_df_scaled = train_df_scaled[feature_cols].copy()
validation_df_scaled = validation_df_scaled[feature_cols].copy()
test_df_scaled = test_df_scaled[feature_cols].copy()
testKaggle_scaled = testKaggle_scaled[feature_cols].copy()

train_df_scaled = train_df_scaled.astype("float32")
validation_df_scaled = validation_df_scaled.astype("float32")
test_df_scaled = test_df_scaled.astype("float32")
testKaggle_scaled = testKaggle_scaled.astype("float32")

print("Done with Labels")



## === cell 17
print(train_labels.shape)
print(validation_labels.shape)
print(test_labels.shape)



## === cell 18
train_df_scaled.shape



## === cell 19
validation_df_scaled.shape



## === cell 20
training_ok = True
try:
    checkpoint = ModelCheckpoint(
        filepath="my_model.keras", verbose=1, save_best_only=True
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
        loss="mean_squared_error", optimizer=adam, metrics=["mae", rmse, "mse"]
    )

    print("Dataset size: %s" % DATASET_SIZE)
    print("Epochs: %s" % EPOCHS)
    print("Learning rate: %s" % LEARNING_RATE)
    print("Batch size: %s" % BATCH_SIZE)
    print("Input dimension: %s" % train_df_scaled.shape[1])
    print("Features used: %s" % list(train_df_scaled.columns))
    model.summary()

    callbacks = []
    try:
        callbacks = [checkpoint]
    except Exception:
        callbacks = []

    history = model.fit(
        x=train_df_scaled,
        y=train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks,
        validation_data=(validation_df_scaled, validation_labels),
        shuffle=True,
    )
except Exception as e:
    training_ok = False
    print("Training failed, will create fallback submission. Error:", repr(e))
    history = None
    model = None



## === cell 21
try:
    if history is not None:
        plot_loss_accuracy_rmse(history)
except Exception as e:
    print("Training plots skipped:", repr(e))



## === cell 22
if training_ok:
    score = model.evaluate(train_df_scaled, train_labels, verbose=0)
    print(score)
    print("train mean_squared_error:", score[0])
    print("train mae:", score[1])
    print("train rmse:", score[2])
    print("train mse:", score[3])



## === cell 23
if training_ok:
    score = model.evaluate(validation_df_scaled, validation_labels, verbose=0)
    print(score)
    print("Validation mean_squared_error:", score[0])
    print("Validation mae:", score[1])
    print("Validation rmse:", score[2])
    print("Validation mse:", score[3])



## === cell 24
if training_ok:
    score = model.evaluate(test_df_scaled, test_labels, verbose=0)
    print(score)
    print("Test mean_squared_error:", score[0])
    print("Test mae:", score[1])
    print("Test rmse:", score[2])
    print("Test mse:", score[3])



## === cell 25
if training_ok:
    validation_predictions = model.predict(validation_df_scaled, verbose=0).flatten()
    try:
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
        plt.show()
    except Exception as e:
        print("Validation scatter skipped:", repr(e))



## === cell 26
if training_ok:
    test_predictions = model.predict(test_df_scaled, verbose=0).flatten()
    try:
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
        plt.show()
    except Exception as e:
        print("Test scatter skipped:", repr(e))



## === cell 27
if training_ok:
    print(np.argmax(test_predictions))
    print(test_predictions[np.argmax(test_predictions)])
    print(test_labels[np.argmax(test_predictions)])
    print(test_df.iloc[np.argmax(test_predictions)])



## === cell 28
if training_ok:
    print(np.argmin(test_predictions))
    print(test_predictions[np.argmin(test_predictions)])
    print(test_labels[np.argmin(test_predictions)])
    print(test_df.iloc[np.argmin(test_predictions)])



## === cell 29
if training_ok:
    try:
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
    except Exception as e:
        print("Measured vs predicted plot skipped:", repr(e))



## === cell 30
if training_ok:
    try:
        plt.figure(figsize=(20, 10))
        plt.plot(validation_labels[:100])
        plt.plot(validation_predictions[:100])
        plt.title("Prediction vs Actual")
        plt.ylabel("Fare Amount (scaled)")
        plt.xlabel("Transaction")
        plt.legend(["Actual", "prediction"], loc="upper right")
        plt.show()
    except Exception as e:
        print("Validation series plot skipped:", repr(e))



## === cell 31
if training_ok:
    try:
        plt.figure(figsize=(20, 10))
        plt.plot(test_labels[:100])
        plt.plot(test_predictions[:100])
        plt.title("Prediction vs Actual")
        plt.ylabel("Fare Amount (scaled)")
        plt.xlabel("Transaction")
        plt.legend(["Actual", "prediction"], loc="upper right")
        plt.show()
    except Exception as e:
        print("Test series plot skipped:", repr(e))



## === cell 32
if training_ok:
    error = validation_predictions - validation_labels
    try:
        plt.hist(error, bins=100)
        plt.xlabel("Prediction Error")
        _ = plt.ylabel("Count")
        plt.show()
    except Exception as e:
        print("Validation error hist skipped:", repr(e))



## === cell 33
if training_ok:
    error = test_predictions - test_labels
    try:
        plt.hist(error, bins=50)
        plt.xlabel("Prediction Error")
        _ = plt.ylabel("Count")
        plt.show()
    except Exception as e:
        print("Test error hist skipped:", repr(e))



## === cell 34
if training_ok:
    print(len(error))
    errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
    print(len(errorGreaterZero))
    try:
        plt.hist(errorGreaterZero, bins=100)
        plt.xlabel("Prediction Error")
        _ = plt.ylabel("Count")
        plt.show()
    except Exception as e:
        print("Filtered error hist skipped:", repr(e))



## === cell 35
testKaggle_scaled.head()



## === cell 36
if training_ok:
    predictionKaggle_scaled = model.predict(
        testKaggle_scaled, batch_size=128, verbose=1
    ).reshape(-1, 1)

    predictionKaggle = scaler_y.inverse_transform(predictionKaggle_scaled)
    predictionKaggle = np.clip(predictionKaggle, 0.0, None)
else:
    sample_path_candidates = [
        "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/new-york-city-taxi-fare-prediction/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
    sample_path = None
    for p in sample_path_candidates:
        if os.path.exists(p):
            sample_path = p
            break
    if sample_path is None:
        fallback_value = 11.35
    else:
        sample_sub = pd.read_csv(sample_path)
        fallback_value = float(sample_sub["fare_amount"].mean())
    predictionKaggle = np.full((len(testKaggle), 1), fallback_value, dtype="float32")



## === cell 37
assert "key" in testKaggle.columns, "testKaggle must contain 'key' for submission."
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)

sub = pd.read_csv(SUBMISSION_NAME)
print(sub.head())
print("Submission columns:", list(sub.columns), "rows:", len(sub))
assert list(sub.columns) == [
    "key",
    "fare_amount",
], "Submission must have columns: key,fare_amount"
assert SUBMISSION_NAME.endswith(".csv"), "Submission filename must end with .csv"
