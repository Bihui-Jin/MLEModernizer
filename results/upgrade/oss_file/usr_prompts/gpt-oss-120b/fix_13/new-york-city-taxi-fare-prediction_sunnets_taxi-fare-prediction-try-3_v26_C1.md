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

5.72184

# 6. Current score

None

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 122.33137) has done: 'I replace the TensorFlow‑based Keras imports with the pure Keras package to avoid the protobuf error, make the datetime parsing robust (remove the strict format string that caused NaT values), and add a fallback when the specified CSV paths are not found. These minimal fixes let the notebook run end‑to‑end and produce a correctly‑named *.csv* submission file, while keeping the original model and feature engineering unchanged.'
- What this solution (achieved 15.2385) has done: 'I replace the TensorFlow Keras imports with pure Keras to avoid the protobuf error, drop the target column from the internal test set before scaling (so the scaler sees the same feature set it was fitted on), and keep the rest of the pipeline unchanged. This fixes the runtime crashes and ensures a valid *.csv* submission is written, while preserving the original model and feature engineering.'
- What this solution (achieved 102.12449) has done: 'Implemented fixes and modest improvements:
- Imported TensorFlow and rewrote the custom RMSE metric using TensorFlow operations (the previous Keras backend lacked `sqrt`).
- Scaled the target variable with `StandardScaler` to help model training and inverse‑transformed predictions for correct submission values.
- Lightened L1 regularization (0.001) for better learning.
- Added an EarlyStopping callback to prevent over‑training.
- Adjusted training to use the scaled target values.'
- What this solution (achieved 105.37247) has done: 'I removed the TensorFlow dependency (which caused the import error) and replaced the custom TensorFlow‑based RMSE metric with the built‑in MAE metric.  
The target variable is now trained on its original scale (no StandardScaler ), so the model’s predictions can be written directly to the submission file.  
All other preprocessing, feature engineering and model architecture remain unchanged, ensuring the core logic is preserved while fixing the runtime crash and improving the evaluation score.'
- What this solution (achieved 150.4788) has done: 'Implemented robust imports using TensorFlow‑Keras to avoid the protobuf `MessageFactory` error, and added a fallback to pure Keras if TensorFlow is unavailable. Updated `add_time_features` to safely handle invalid timestamps by coercing errors, filling missing datetimes with a default value, and ensuring no NaNs remain in the newly created time columns. These changes fix the runtime crash and prevent NaN propagation that caused extreme RMSE, allowing the pipeline to run end‑to‑end and produce a valid submission CSV. The core model architecture and training logic remain unchanged.'
- What this solution (achieved 78.36806) has done: 'I add proper cleaning of the internal test split, scale the target variable with a StandardScaler and inverse‑transform predictions before saving. This fixes NaNs in the test data, improves training stability, and yields predictions on the original fare scale, moving the RMSE toward the target while keeping the original Keras architecture unchanged.'
- What this solution (achieved 581.15562) has done: 'Implemented a lightweight linear regression model and blended its predictions with the existing Keras network to boost accuracy. Added the necessary sklearn import, trained the regression on the same scaled features, and combined both model outputs (50/50 average) before creating the submission file. Also increased early‑stopping patience to allow the neural net a few more epochs for better convergence. This keeps the original pipeline intact while improving the RMSE toward the target score.'
- What this solution (achieved 471.36309) has done: 'I replace the TensorFlow‑Keras imports with pure Keras to eliminate the protobuf import error, and I train the LinearRegression model on the same scaled target values as the neural network. The linear‑regression predictions are therefore inverse‑scaled before blending, fixing the huge scale mismatch that caused the extreme RMSE.'

# 9. Code solution

## === cell 0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers, regularizers

from sklearn.linear_model import LinearRegression

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

TRAIN_PATH = "../input/labels.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 0.001
DATASET_SIZE = 80000  # subset for quick run




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


def read_csv_with_fallback(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except FileNotFoundError:
        alt_path = (
            f"/kaggle/input/new-york-city-taxi-fare-prediction/{os.path.basename(path)}"
        )
        return pd.read_csv(alt_path, **kwargs)


trainKaggle = read_csv_with_fallback(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    dtype=datatypes,
)

testKaggle = read_csv_with_fallback(TEST_PATH, dtype=datatypes)




## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)

train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)

print(
    f"Sizes -> train: {len(train_df)}, validation: {len(validation_df)}, internal test: {len(test_df)}"
)
print(f"Test set (Kaggle) size: {len(testKaggle)}")




## === cell 3
def clean(df):
    print(" Old size:", len(df))
    df = df.dropna(how="any", axis="rows")
    print(" After dropna:", len(df))

    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        & (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    print(" After removing identical coords:", len(df))

    df = df[
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    ]
    print(" After removing zero coords:", len(df))

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
    print(" After NYC filter:", len(df))

    df = df[(0.99 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    print(" After fare outlier filter:", len(df))

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" After passenger count filter:", len(df))

    return df


def remove_datapoints_from_water(df):
    return df  # placeholder – no water mask applied


def late_night(row):
    return 1 if (row["hour"] <= 3) else 0


def night(row):
    return 1 if ((row["hour"] > 20) and (row["weekday"] < 5)) else 0


def rush_hour(row):
    return 1 if (16 <= row["hour"] <= 20) and (row["weekday"] < 5) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["pickup_datetime"] = df["pickup_datetime"].fillna(pd.Timestamp("2000-01-01"))
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["pickup_datetime"] = df["pickup_datetime"].astype(str)
    df["night"] = df.apply(night, axis=1)
    df["late_night"] = df.apply(late_night, axis=1)
    df["rush_hour"] = df.apply(rush_hour, axis=1)
    return df


def add_coordinate_features(df):
    df["latdiff"] = df["pickup_latitude"] - df["dropoff_latitude"]
    df["londiff"] = df["pickup_longitude"] - df["dropoff_longitude"]
    return df


def add_distances_features(df):
    df["manhattan"] = manhattan(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    R = 6371.0  # Earth radius in km
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    dlat = lat2 - lat1
    dlon = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    df["haversine"] = 2 * R * np.arcsin(np.sqrt(a))

    df["distance"] = np.sqrt(
        (df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
        + (df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
    )
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    pred_series = pd.Series(np.clip(prediction, 0, None), name=prediction_column)
    df_out = pd.concat(
        [raw_test[id_column].reset_index(drop=True), pred_series], axis=1
    )
    df_out.to_csv(file_name, index=False)
    print(f"Submission written to {file_name}")


print("Cleaning training split")
train_df = clean(train_df)
print("Cleaning validation split")
validation_df = clean(validation_df)
print("Cleaning internal test split")
test_df = clean(test_df)




## === cell 4
print("Adding time features")
train_df = add_time_features(train_df)
validation_df = add_time_features(validation_df)
test_df = add_time_features(test_df)
testKaggle = add_time_features(testKaggle)

print("Adding coordinate features")
train_df = add_coordinate_features(train_df)
validation_df = add_coordinate_features(validation_df)
test_df = add_coordinate_features(test_df)
testKaggle = add_coordinate_features(testKaggle)

print("Adding distance features")
train_df = add_distances_features(train_df)
validation_df = add_distances_features(validation_df)
test_df = add_distances_features(test_df)
testKaggle = add_distances_features(testKaggle)




## === cell 5
dropped_columns = [
    "key",
    "pickup_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
train_df = train_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns, axis=1)

print("Dropped non‑numeric columns. Remaining features:", train_df.columns.tolist())

train_labels_raw = train_df["fare_amount"].values
validation_labels_raw = validation_df["fare_amount"].values

train_labels = np.log1p(train_labels_raw)
validation_labels = np.log1p(validation_labels_raw)

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)

target_scaler = preprocessing.StandardScaler()
train_labels_scaled = target_scaler.fit_transform(train_labels.reshape(-1, 1)).ravel()
validation_labels_scaled = target_scaler.transform(
    validation_labels.reshape(-1, 1)
).ravel()

print("Feature and target scaling complete.")




## === cell 6
lr_model = LinearRegression()
lr_model.fit(train_df_scaled, train_labels_scaled)

model = Sequential()
model.add(
    Dense(
        512,
        activation="relu",
        input_dim=train_df_scaled.shape[1],
        activity_regularizer=regularizers.l1(0.001),
    )
)
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(216, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(32, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(16, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(8, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mean_squared_error", optimizer=adam, metrics=["mae"])

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

print("Starting training...")
history = model.fit(
    x=train_df_scaled,
    y=train_labels_scaled,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(validation_df_scaled, validation_labels_scaled),
    shuffle=True,
    callbacks=[early_stop],
)

print("Predicting on internal test split (Keras)...")
pred_internal_scaled = model.predict(test_scaled, batch_size=128, verbose=0).ravel()
pred_internal_log = target_scaler.inverse_transform(
    pred_internal_scaled.reshape(-1, 1)
).ravel()
pred_internal = np.expm1(pred_internal_log)

print("Predicting on internal test split (Linear Regression)...")
pred_internal_lr_scaled = lr_model.predict(test_scaled)
pred_internal_lr_log = target_scaler.inverse_transform(
    pred_internal_lr_scaled.reshape(-1, 1)
).ravel()
pred_internal_lr = np.expm1(pred_internal_lr_log)

pred_internal_blend = 0.5 * pred_internal + 0.5 * pred_internal_lr

print("Predicting on Kaggle test set (Keras)...")
pred_kaggle_scaled = model.predict(testKaggle_scaled, batch_size=128, verbose=0).ravel()
pred_kaggle_log = target_scaler.inverse_transform(
    pred_kaggle_scaled.reshape(-1, 1)
).ravel()
pred_kaggle = np.expm1(pred_kaggle_log)

print("Predicting on Kaggle test set (Linear Regression)...")
pred_kaggle_lr_scaled = lr_model.predict(testKaggle_scaled)
pred_kaggle_lr_log = target_scaler.inverse_transform(
    pred_kaggle_lr_scaled.reshape(-1, 1)
).ravel()
pred_kaggle_lr = np.expm1(pred_kaggle_lr_log)

pred_kaggle_blend = 0.5 * pred_kaggle + 0.5 * pred_kaggle_lr




## === cell 7
output_submission(testKaggle, pred_kaggle_blend, "key", "fare_amount", SUBMISSION_NAME)
