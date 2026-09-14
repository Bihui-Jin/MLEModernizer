# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

4.26907

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            if hasattr(_message_factory, "GetMessageClass"):
                return _message_factory.GetMessageClass(descriptor)
            raise AttributeError("No compatible GetPrototype/GetMessageClass found")

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception as _e:
    print("protobuf patch skipped due to:", repr(_e))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import optimizers, regularizers
from keras import ops  # Keras 3-compatible math ops for custom metrics

np.random.seed(1)


def _resolve_path(*candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[-1]


TRAIN_PATH = _resolve_path(
    "/kaggle/input/train.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/data/train.csv",
)
TEST_PATH = _resolve_path(
    "/kaggle/input/test.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/data/test.csv",
)

SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH), TRAIN_PATH)
print("TEST_PATH exists:", os.path.exists(TEST_PATH), TEST_PATH)




## === cell 1
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
    df = df[(df["dropoff_longitude"] != 0)]
    df = df[(df["dropoff_latitude"] != 0)]
    print(" New size after lang lot > 0: %d" % len(df))

    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)]
    print(" New size after lang - lot > 0.001: %d" % len(df))

    if "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0)]
    print(" New size after removing passenger_count <= 0 : %d" % len(df))

    return df


def late_night(row):
    return 1 if (row["hour"] <= 3) else 0


def night(row):
    h = row["hour"]
    wd = row["weekday"]
    if wd < 5 and ((h >= 20) or (h <= 5)):
        return 1
    return 0


def rush_hour(row):
    if (16 <= row["hour"] <= 20) and (row["weekday"] < 5):
        return 1
    return 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["year"] = dt.dt.year.astype("float32")
    df["month"] = dt.dt.month.astype("float32")
    df["day"] = dt.dt.day.astype("float32")
    df["hour"] = dt.dt.hour.astype("float32")
    df["weekday"] = dt.dt.weekday.astype("float32")

    df["pickup_datetime"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

    df["night"] = df.apply(lambda x: night(x), axis=1).astype("uint8")
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1).astype("uint8")
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1).astype("uint8")
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["latdiff"] = (lat1 - lat2).abs()
    df["londiff"] = (lon1 - lon2).abs()
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = np.sqrt(
        (df["pickup_longitude"] - df["dropoff_longitude"]).abs() ** 2
        + (df["pickup_latitude"] - df["dropoff_latitude"]).abs() ** 2
    )
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {
            id_column: raw_test[id_column].values,
            prediction_column: np.asarray(prediction).reshape(-1),
        }
    )
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 6))
    plt.plot(history.history.get("loss", []))
    plt.plot(history.history.get("val_loss", []))
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()

    if "rmse" in history.history:
        plt.figure(figsize=(20, 6))
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

trainKaggle.head(), testKaggle.head()



## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]

print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## === cell 4
print("train_df clean")
train_df = clean(train_df)
test_df = clean(test_df)

print("testKaggle clean")
testKaggle = clean(testKaggle)



## === cell 5
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)

print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)

print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## === cell 6
try:
    _ = train_df.iloc[:2000].plot.scatter("latdiff", "londiff")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "weekday")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "night")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "late_night")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "rush_hour")
    plt.show()
except Exception as e:
    print("Skipping plots due to:", repr(e))



## === cell 7
dropped_columns = [
    "pickup_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_longitude",
    "dropoff_latitude",
]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")
print("Train columns:", list(train_df.columns))
print("TestKaggle_clean columns:", list(testKaggle_clean.columns))



## === cell 8
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)

train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")
print("Feature dim:", train_df.shape[1])



## === cell 9
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)


def rmse(y_true, y_pred):
    y_true = ops.cast(y_true, "float32")
    y_pred = ops.cast(y_pred, "float32")
    return ops.sqrt(ops.mean(ops.square(y_pred - y_true), axis=-1))




## === cell 10
checkpoint = ModelCheckpoint(filepath="my_model.keras", verbose=1, save_best_only=True)

model = Sequential()
model.add(
    Dense(
        256,
        activation="linear",
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
    callbacks=[checkpoint],
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)



## === cell 11
try:
    from keras.utils import model_to_dot
    from IPython.display import SVG

    SVG(model_to_dot(model).create(prog="dot", format="svg"))
except Exception as e:
    print("Skipping model graph due to:", repr(e))



## === cell 12
plot_loss_accuracy_rmse(history)



## === cell 13
score = model.evaluate(train_df_scaled, train_labels, verbose=1)
print(score)
print("train mean_squared_error:", score[0])
print("train mae:", score[1])
print("train rmse:", score[2])
print("train mse:", score[3])

score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print(score)
print("Validation mean_squared_error:", score[0])
print("Validation mae:", score[1])
print("Validation rmse:", score[2])
print("Validation mse:", score[3])

score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print("Test mean_squared_error:", score[0])
print("Test mae:", score[1])
print("Test rmse:", score[2])
print("Test mse:", score[3])



## === cell 14
validation_predictions = model.predict(validation_df_scaled, verbose=0).flatten()

plt.scatter(validation_labels, validation_predictions, s=5)
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



## === cell 15
test_predictions = model.predict(test_scaled, verbose=0).flatten()

plt.scatter(test_labels, test_predictions, s=5)
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



## === cell 16
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_labels[np.argmax(test_predictions)])
test_df.iloc[np.argmax(test_predictions)]



## === cell 17
print(np.argmin(test_predictions))
print(test_predictions[np.argmin(test_predictions)])
print(test_labels[np.argmin(test_predictions)])
test_df.iloc[np.argmin(test_predictions)]



## === cell 18
fig, ax = plt.subplots()
ax.scatter(test_labels, test_predictions, s=5)
ax.plot(
    [test_labels.min(), test_labels.max()],
    [test_labels.min(), test_labels.max()],
    "k--",
    lw=2,
)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()



## === cell 19
plt.figure(figsize=(20, 6))
plt.plot(validation_labels[:100])
plt.plot(validation_predictions[:100])
plt.title("Prediction vs Actual (validation)")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()



## === cell 20
plt.figure(figsize=(20, 6))
plt.plot(test_labels[:100])
plt.plot(test_predictions[:100])
plt.title("Prediction vs Actual (test split)")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()



## === cell 21
error = validation_predictions - validation_labels
plt.hist(error, bins=100)
plt.xlabel("Prediction Error (validation)")
_ = plt.ylabel("Count")
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel("Prediction Error (test split)")
_ = plt.ylabel("Count")
plt.show()



## === cell 22
errorGreaterZero = error[np.logical_or(error <= -1, error >= 1)]
print(len(error))
print(len(errorGreaterZero))

plt.hist(errorGreaterZero, bins=100)
plt.xlabel("Prediction Error |>=1| (test split)")
_ = plt.ylabel("Count")
plt.show()



## === cell 23
try:
    from keras.models import load_model

    if os.path.exists("my_model.keras"):
        model = load_model("my_model.keras", custom_objects={"rmse": rmse})
        print("Loaded best saved model from my_model.keras for submission predictions.")
    else:
        print("Checkpoint file my_model.keras not found; using current model weights.")
except Exception as e:
    print(
        "Could not load saved model; using current model weights. Reason:",
        repr(e),
    )

predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)
predictionKaggle = np.clip(predictionKaggle, 0, None)

output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
print("Submission preview:")
print(pd.read_csv(SUBMISSION_NAME).head())
print(
    "Submission file exists:",
    os.path.exists(SUBMISSION_NAME),
    "size(bytes):",
    os.path.getsize(SUBMISSION_NAME) if os.path.exists(SUBMISSION_NAME) else None,
)
