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

4.82559

# 6. Current score

6.19245

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 569.76912) has done: 'I fixed the import conflict that caused the protobuf error by using the standalone keras package instead of tf.keras, added a robust helper to locate the CSV files regardless of the working directory, and corrected the test‑set loading (removed the nonexistent fare_amount dtype and specified the proper columns). These changes unblock the pipeline, ensure all variables are defined, and let the script finish with a valid submissiontry_water.csv file.'
- What this solution (achieved 27.23567) has done: 'I increase the training sample size, reduce the L1 regularization strength, train a bit longer, and clip the final predictions to a realistic fare range. These modest tweaks should lower the RMSE toward the target without changing the core modeling pipeline.'
- What this solution (achieved 6.24601) has done: 'Implemented fixes to unblock the pipeline and improve the RMSE:
- Removed the `keras` imports that caused protobuf errors and replaced them with a Scikit‑Learn `GradientBoostingRegressor`.
- Added the necessary Scikit‑Learn import.
- Simplified the model training step (no Keras history or plotting) while keeping the same feature engineering and scaling.
- Retained all preprocessing, cleaning, and feature creation logic, then trained the new model and generated the submission CSV.'
- What this solution (achieved 6.19245) has done: 'Implemented a log‑transform of the fare target and modestly strengthened the GradientBoostingRegressor (more trees, lower learning rate). The model now trains on `log1p(fare_amount)`, predictions are inverse‑transformed with `expm1`, and clipping is applied after back‑conversion. These tweaks keep the original preprocessing and feature engineering untouched while expectedly lowering the validation RMSE toward the target.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


def locate_file(filename: str) -> str:
    for path in Path(".").rglob(filename):
        if path.is_file():
            return str(path)
    raise FileNotFoundError(f"{filename} not found in the repository.")


TRAIN_PATH = locate_file("train.csv")
TEST_PATH = locate_file("test.csv")
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 50  # retained for compatibility; not used by sklearn
LEARNING_RATE = 0.001
DATASET_SIZE = 300000  # use a manageable subset for fast training


## === cell 1
dtypes_train = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
dtypes_test = {
    "key": "str",
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
    dtype=dtypes_train,
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
    dtype=dtypes_test,
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

train_df, temp_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
validation_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=1)

print(
    f"train size: {len(train_df)}, validation size: {len(validation_df)}, test size: {len(test_df)}"
)




## === cell 2
def remove_datapoints_from_water(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder – original code expected an external mask.
    We simply return the dataframe unchanged."""
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Cleaning – original size: {len(df)}")
    df = df.dropna()
    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        | (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    df = df[
        (
            df[
                [
                    "dropoff_longitude",
                    "pickup_longitude",
                    "dropoff_latitude",
                    "pickup_latitude",
                ]
            ]
            != 0
        ).all(axis=1)
    ]

    min_lon, max_lon, min_lat, max_lat = -74.5, -72.8, 40.5, 41.8
    df = df[
        (df["pickup_longitude"].between(min_lon, max_lon))
        & (df["dropoff_longitude"].between(min_lon, max_lon))
        & (df["pickup_latitude"].between(min_lat, max_lat))
        & (df["dropoff_latitude"].between(min_lat, max_lat))
    ]

    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 50)]
    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]

    df = remove_datapoints_from_water(df)
    print(f"Cleaned size: {len(df)}")
    return df


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S %Z", errors="coerce"
    )
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["night"] = ((df["hour"] <= 3) | (df["hour"] >= 0)).astype(int)
    df["late_night"] = ((df["hour"] <= 3) | (df["hour"] >= 0)).astype(int)
    df["rush_hour"] = (
        (df["hour"] >= 16) & (df["hour"] <= 20) & (df["weekday"] < 5)
    ).astype(int)
    return df


def add_coordinate_features(df: pd.DataFrame) -> pd.DataFrame:
    df["latdiff"] = df["pickup_latitude"] - df["dropoff_latitude"]
    df["londiff"] = df["pickup_longitude"] - df["dropoff_longitude"]
    return df


def add_distances_features(df: pd.DataFrame) -> pd.DataFrame:
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




## === cell 3
print("Cleaning train set")
train_df = clean(train_df)
print("Cleaning validation set")
validation_df = clean(validation_df)

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


## === cell 4
drop_cols = ["passenger_count", "pickup_datetime"]
train_df = train_df.drop(columns=drop_cols)
validation_df = validation_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)
testKaggle_clean = testKaggle.drop(columns=drop_cols + ["key"])

train_labels = np.log1p(train_df["fare_amount"].values)
validation_labels = np.log1p(validation_df["fare_amount"].values)

train_df = train_df.drop(columns=["fare_amount"])
validation_df = validation_df.drop(columns=["fare_amount"])
test_df = test_df.drop(columns=["fare_amount"])

scaler = preprocessing.MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
validation_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)

gbr = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    random_state=42,
)
gbr.fit(train_scaled, train_labels)

from sklearn.metrics import mean_squared_error

val_pred_log = gbr.predict(validation_scaled)
val_pred = np.expm1(val_pred_log)  # inverse of log1p
val_rmse = mean_squared_error(validation_labels, np.log1p(val_pred), squared=False)
print(f"Validation RMSE: {val_rmse:.4f}")


## === cell 5
prediction_log = gbr.predict(testKaggle_scaled)
predictionKaggle = np.expm1(prediction_log)
predictionKaggle = np.clip(predictionKaggle, 0.0, 200.0)
submission = pd.DataFrame({"key": testKaggle["key"], "fare_amount": predictionKaggle})
submission.to_csv(SUBMISSION_NAME, index=False)
print(f"Submission written to {SUBMISSION_NAME}")
