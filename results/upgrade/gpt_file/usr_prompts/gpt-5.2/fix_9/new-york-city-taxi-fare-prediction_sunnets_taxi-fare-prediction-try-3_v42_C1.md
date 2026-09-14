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

4.20178

# 6. Current score

7.82905

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 26.11342) has done: 'I fix the import/runtime crash in the first cell by making Keras/TensorFlow imports robust to the Kaggle environment so the notebook can start. Then I fix the timezone conversion bug in `add_time_features` by correctly removing timezone info before string conversion (keeping the same time-based features). Next I fix the scaler “feature names mismatch” by ensuring `train_df`, `validation_df`, `test_df`, and `testKaggle_clean` have identical feature columns in the same order before scaling. Finally, I ensure the submission is written to a valid `.csv` path (`/kaggle/working/...`) with the required `key,fare_amount` columns.'
- What this solution (achieved 26.23648) has done: 'I fix the early import crash by forcing the Keras backend to use `tf_keras` (installed) and setting environment variables to avoid the protobuf `MessageFactory.GetPrototype` issue that occurs when TensorFlow gets pulled in implicitly. Next, I fix the scaler crash by ensuring no string columns (especially `key`) remain in the feature matrix: after dropping `fare_amount`, we also drop `key` from train/validation/test feature frames, while keeping `key` only for submission. Finally, I make the downstream training/inference cells robust so they only rely on the now-defined scaled arrays and always write a valid `/kaggle/working/*.csv` submission with `key,fare_amount`.'
- What this solution (achieved 18.10401) has done: 'I fix the TensorFlow/protobuf crash by forcing TensorFlow to use the pure-Python protobuf implementation *before* any TF/keras import, which prevents the `MessageFactory.GetPrototype` error in this Kaggle environment. Then I make the Keras imports deterministic by preferring `tensorflow.keras` (and only falling back to `tf_keras`) to avoid backend mismatches that can silently degrade training/prediction. Finally, I make `rmse` and the sample-MSE computation robust across Keras versions (tensor vs numpy), without changing the model architecture, features, training loop, or submission format, so the notebook runs end-to-end and writes a valid `.csv`.'
- What this solution (achieved 24.90875) has done: 'I fix the protobuf/TensorFlow import crash that prevents the notebook from running by adding a safe monkey‑patch for the missing `MessageFactory.GetPrototype` method before importing TensorFlow/Keras. Then I correct a major logic bug in `clean()` where it filters on `fare_amount` even for the Kaggle test set (which has no `fare_amount`), by making that filter conditional; this is score-positive because it ensures all test rows get predictions instead of being accidentally dropped or erroring. Finally, I keep the existing model/training loop intact but ensure the submission is always produced with the required `key,fare_amount` columns and correct row count.'
- What this solution (achieved 6.05688) has done: 'Your RMSE is far above the target, so we need a score-improving fix with minimal disruption. The biggest issue is a feature mismatch: you convert `pickup_datetime` to string, then later drop it, so the model never sees time information; also the time indicator features (`night/late_night/rush_hour`) are computed via slow row-wise `apply` and can silently create NaNs when `pickup_datetime` parsing fails. I (1) keep your same feature set idea but stop overwriting `pickup_datetime` (leave it as datetime) and vectorize the three time-indicator features, (2) add the standard “haversine distance” feature alongside your existing distances (same feature-extraction approach, just one extra distance), and (3) clip predictions to a more realistic upper bound (50) consistent with your training outlier filter, which typically reduces RMSE on this competition.'
- What this solution (achieved 7.82905) has done: 'Your current RMSE (6.05688) is still worse than the target (4.20178), so we should make small, score-positive fixes without changing the overall model/training setup. The biggest low-risk gain here is to stop dropping `passenger_count` (it’s a core feature with real signal) and to add one standard, very cheap distance feature (`abs_latdiff + abs_londiff`) alongside your existing distance features. I also ensure coordinate features are consistently assigned (not relying on in-place side effects) and apply the same prediction clipping bound to the internal test split (just for a fairer sanity check; Kaggle submission already clips). Everything else (data size, model architecture, loss, training loop) stays the same, and the script still writes a valid submission CSV.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from google.protobuf import message_factory as _message_factory  # noqa: F401

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras  # noqa: F401
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
    from tensorflow.keras import optimizers
    from tensorflow.keras import regularizers
    from tensorflow.keras import backend

    _KERAS_BACKEND = "tensorflow.keras"
except Exception as e:
    import tf_keras as keras  # noqa: F401
    from tf_keras.models import Sequential
    from tf_keras.layers import Dense, Dropout, BatchNormalization, LSTM
    from tf_keras import optimizers
    from tf_keras import regularizers
    from tf_keras import backend

    _KERAS_BACKEND = "tf_keras (fallback due to tensorflow.keras import issue: %r)" % e

TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SUBMISSION_NAME = "/kaggle/working/submissiontry_water.csv"

BATCH_SIZE = 1000
EPOCHS = 30
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

np.random.seed(1)
print("Keras backend:", _KERAS_BACKEND)



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
    dtype={k: v for k, v in datatypes.items() if k != "fare_amount"},
)



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
train_df.describe()



## === cell 6
validation_df.describe()



## === cell 7
test_df.describe()




## === cell 8
def remove_datapoints_from_water(df):
    """
    FIX: Original code tried to download a NYC land/water mask from the internet,
    which is not available on Kaggle offline runtimes. To preserve core logic
    without external dependencies, we skip this step.
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
    print(" New size after only NYC: %d" % len(df))

    if "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))
    else:
        print(" Skipping fare_amount outlier filter (fare_amount not present)")

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


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["year"] = dt.dt.year.fillna(0).astype("int16")
    df["month"] = dt.dt.month.fillna(0).astype("int8")
    df["day"] = dt.dt.day.fillna(0).astype("int8")
    df["hour"] = dt.dt.hour.fillna(0).astype("int8")
    df["weekday"] = dt.dt.weekday.fillna(0).astype("int8")

    hour = df["hour"].astype("int16")
    weekday = df["weekday"].astype("int16")

    df["late_night"] = ((hour <= 3) | (hour >= 22)).astype("int8")
    df["night"] = (((hour >= 20) | (hour <= 6)) & (weekday < 5)).astype("int8")
    df["rush_hour"] = (((hour >= 16) & (hour <= 20)) & (weekday < 5)).astype("int8")
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["latdiff"] = lat1 - lat2
    df["londiff"] = lon1 - lon2
    return df


def _haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype("float64"))
    lat1 = np.radians(lat1.astype("float64"))
    lon2 = np.radians(lon2.astype("float64"))
    lat2 = np.radians(lat2.astype("float64"))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = np.sqrt(np.abs(lon1 - lon2) ** 2 + np.abs(lat1 - lat2) ** 2)

    df["haversine_km"] = _haversine_km(lon1, lat1, lon2, lat2).astype("float32")

    df["abs_latdiff_plus_abs_londiff"] = (
        np.abs(lat1 - lat2) + np.abs(lon1 - lon2)
    ).astype("float32")
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {id_column: raw_test[id_column].values, prediction_column: prediction}
    )
    df.to_csv(file_name, index=False)
    print("Output complete -> %s (%d rows)" % (file_name, len(df)))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    if "rmse" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["rmse"])
        plt.plot(history.history["val_rmse"])
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper right")
        plt.show()




## === cell 9
print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)

print("test_df clean")
test_df = clean(test_df)



## === cell 10
train_df.describe()



## === cell 11
validation_df.describe()



## === cell 12
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 13
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 14
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## === cell 15
train_df.describe()



## === cell 16
validation_df.describe()



## === cell 17
dropped_columns = ["pickup_datetime"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")



## === cell 18
train_labels = train_df["fare_amount"].values.astype("float32")
validation_labels = validation_df["fare_amount"].values.astype("float32")
test_labels = test_df["fare_amount"].values.astype("float32")

for _df_name, _df in [
    ("train_df", train_df),
    ("validation_df", validation_df),
    ("test_df", test_df),
]:
    if "key" not in _df.columns:
        raise KeyError(
            f"Expected 'key' column in {_df_name} before dropping; columns={list(_df.columns)}"
        )

train_df = train_df.drop(["fare_amount", "key"], axis=1)
validation_df = validation_df.drop(["fare_amount", "key"], axis=1)
test_df = test_df.drop(["fare_amount", "key"], axis=1)

print("Done with Labels")



## === cell 19
train_df.shape



## === cell 20
test_df.shape



## === cell 21
validation_df.shape



## === cell 22
feature_cols = list(train_df.columns)
validation_df = validation_df.reindex(columns=feature_cols)
test_df = test_df.reindex(columns=feature_cols)
testKaggle_clean = testKaggle_clean.reindex(columns=feature_cols)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)

print(
    "Scaled shapes:",
    train_df_scaled.shape,
    validation_df_scaled.shape,
    test_scaled.shape,
    testKaggle_scaled.shape,
)




## === cell 23
def rmse(y_true, y_pred):
    y_true = backend.cast(y_true, "float32")
    y_pred = backend.cast(y_pred, "float32")
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))




## === cell 24
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

model.compile(
    loss="mean_squared_error", optimizer=adam, metrics=["mae", "accuracy", rmse, "mse"]
)

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



## === cell 25
print("Model visualization skipped (vis_utils not available in this environment).")



## === cell 26
plot_loss_accuracy_rmse(history)



## === cell 27
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## === cell 28
prediction = np.clip(prediction.reshape(-1), 0.0, 50.0)
predictionKaggle = np.clip(predictionKaggle.reshape(-1), 0.0, 50.0)



## === cell 29
y_true_np = test_labels[:1000].astype("float32").reshape(-1)
y_pred_np = prediction[:1000].astype("float32").reshape(-1)
mse_sample = np.mean((y_pred_np - y_true_np) ** 2)
print("Sample MSE (first 1000):", float(mse_sample))



## === cell 30
if len(predictionKaggle) != len(testKaggle):
    raise ValueError(
        f"Prediction length mismatch: pred={len(predictionKaggle)} vs test={len(testKaggle)}"
    )



## === cell 31
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)



## === cell 32
print(
    "Example prediction:",
    float(prediction.reshape(-1)[min(10000, len(prediction) - 1)]),
)
print("Example true label:", float(test_labels[min(10000, len(test_labels) - 1)]))
print("Submission head:")
print(pd.read_csv(SUBMISSION_NAME).head())
print("Submission file written to:", SUBMISSION_NAME)
