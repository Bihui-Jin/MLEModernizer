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

8.05228

# 6. Current score

9.93663

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 9.93663) has done: 'I replace the TensorFlow Keras imports with the compatible tf_keras module, drop the non‑numeric “key” column before scaling, and fix the regularizer reference (use tf_keras.regularizers.l1_l2). These changes resolve the import error, allow the scaler to work on only numeric data, and let the model be built and trained so a proper CSV submission is generated.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tf_keras.models import Sequential
from tf_keras.layers import Dense, BatchNormalization
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping
from tf_keras import backend as K
from tf_keras import optimizers, regularizers
import warnings

warnings.filterwarnings("ignore")

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 80
LEARNING_RATE = 0.001
DATASET_SIZE = 80000




## === cell 1
def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))
    return df


def late_night(row):
    return 1 if (row["hour"] <= 3) else 0


def night(row):
    return 1 if (row["hour"] > 20 and row["weekday"] < 5) else 0


def rush_hour(row):
    return 1 if (16 <= row["hour"] <= 20 and row["weekday"] < 5) else 0


def manhattan(p_lat, p_lon, d_lat, d_lon):
    return np.abs(d_lat - p_lat) + np.abs(d_lon - p_lon)


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S %Z"
    )
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["night"] = df.apply(night, axis=1)
    df["late_night"] = df.apply(late_night, axis=1)
    df["rush_hour"] = df.apply(rush_hour, axis=1)
    df["pickup_datetime"] = df["pickup_datetime"].astype(str)
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
    df["euclidean"] = np.sqrt(
        (df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
        + (df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
    )
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {id_column: raw_test[id_column], prediction_column: prediction.squeeze()}
    )
    df.to_csv(file_name, index=False)
    print("Output complete: {}".format(file_name))


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))




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
testKaggle = pd.read_csv(TEST_PATH, dtype=datatypes)



## === cell 3
train_df, validation_df = train_test_split(trainKaggle, test_size=0.10, random_state=1)

train_df = clean(train_df)
validation_df = clean(validation_df)



## === cell 4
print("Adding time features...")
train_df = add_time_features(train_df)
validation_df = add_time_features(validation_df)
testKaggle = add_time_features(testKaggle)

print("Adding coordinate features...")
train_df = add_coordinate_features(train_df)
validation_df = add_coordinate_features(validation_df)
testKaggle = add_coordinate_features(testKaggle)

print("Adding distance features...")
train_df = add_distances_features(train_df)
validation_df = add_distances_features(validation_df)
testKaggle = add_distances_features(testKaggle)



## === cell 5
drop_cols = [
    "key",  # drop non‑numeric identifier
    "pickup_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
train_df = train_df.drop(columns=drop_cols)
validation_df = validation_df.drop(columns=drop_cols)
test_features = testKaggle.drop(columns=drop_cols + ["key"])



## === cell 6
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = (
    testKaggle["fare_amount"].values if "fare_amount" in testKaggle.columns else None
)

train_df = train_df.drop(columns=["fare_amount"])
validation_df = validation_df.drop(columns=["fare_amount"])



## === cell 7
scaler = preprocessing.MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
validation_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_features)



## === cell 8
adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model = Sequential(
    [
        Dense(
            256,
            activation="linear",
            input_dim=train_scaled.shape[1],
            activity_regularizer=regularizers.l1_l2(l1=0.01),
        ),
        BatchNormalization(),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dense(8, activation="relu"),
        BatchNormalization(),
        Dense(1),
    ]
)

model.compile(loss="mean_squared_error", optimizer=adam, metrics=["mae", rmse])

model.summary()



## === cell 9
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    "my_model.h5", monitor="val_loss", save_best_only=True, verbose=1
)

history = model.fit(
    train_scaled,
    train_labels,
    validation_data=(validation_scaled, validation_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint],
    verbose=1,
)



## === cell 10
val_score = model.evaluate(validation_scaled, validation_labels, verbose=0)
print("Validation RMSE:", val_score[2])  # index 2 corresponds to rmse metric



## === cell 11
test_predictions = model.predict(test_scaled, batch_size=128, verbose=1)
output_submission(testKaggle, test_predictions, "key", "fare_amount", SUBMISSION_NAME)
