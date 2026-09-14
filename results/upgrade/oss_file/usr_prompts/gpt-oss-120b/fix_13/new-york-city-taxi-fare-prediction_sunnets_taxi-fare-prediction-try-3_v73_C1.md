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

4.67002

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 316.32145) has done: 'I fix the import errors, replace the broken water‑mask cleaning with a no‑op, correct the optimizer call, simplify the datetime parsing, and streamline the pipeline so the model can be compiled, trained and used to write a proper `submissiontry_water.csv`. These changes remove the runtime failures and let the regression model run, which should lower the RMSE toward the target value.'
- What this solution (achieved 15.41145) has done: 'The fix updates the TensorFlow‑Keras imports to use the native `keras` package (compatible with Keras 3) and adjusts the backend import accordingly. This resolves the `MessageFactory` import error, allowing the model to compile, train, and generate a proper CSV submission. No other logic is altered, keeping the original training pipeline intact.'
- What this solution (achieved 310.83733) has done: 'The fix adds TensorFlow import and rewrites the custom RMSE metric using `tf.sqrt` (the Keras backend no longer provides `sqrt`). This resolves the AttributeError that stopped training, allowing the model to train and evaluate properly, which should lower the validation RMSE toward the target.'
- What this solution (achieved 214.00557) has done: 'I replace the custom RMSE metric with the built‑in Keras RootMeanSquaredError (ensuring the correct value is reported) and adjust the printed validation score to use this metric. This fixes the earlier import‑related error and guarantees a proper RMSE computation without altering the core model or training logic.'
- What this solution (achieved 450.08427) has done: 'I removed the faulty TensorFlow import (which isn’t available in the environment) and switched to the native Keras‑3 RMSE metric. The script now imports `RootMeanSquaredError` directly from `keras.metrics` and uses it when compiling the model, eliminating the `MessageFactory` protobuf error while keeping all other logic unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor  # kept for fallback
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
DATASET_SIZE = 80000


def remove_datapoints_from_water(df):
    return df


def clean(df):
    """General cleaning used for training data.
    Works safely with test data that may not contain the target column."""
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    MinMax = (-74.5, -72.8, 40.5, 41.8)

    mask = (
        (df["dropoff_longitude"] != df["pickup_longitude"])
        & (df["dropoff_latitude"] != df["pickup_latitude"])
        & (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
        & (MinMax[0] <= df["pickup_longitude"])
        & (df["pickup_longitude"] <= MinMax[1])
        & (MinMax[0] <= df["dropoff_longitude"])
        & (df["dropoff_longitude"] <= MinMax[1])
        & (MinMax[2] <= df["pickup_latitude"])
        & (df["pickup_latitude"] <= MinMax[3])
        & (MinMax[2] <= df["dropoff_latitude"])
        & (df["dropoff_latitude"] <= MinMax[3])
        & ((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)
        & ((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)
    )

    if "fare_amount" in df.columns:
        mask &= (0 < df["fare_amount"]) & (df["fare_amount"] <= 50)

    mask &= (df["passenger_count"] > 0) & (df["passenger_count"] <= 6)

    df = df[mask]
    df = remove_datapoints_from_water(df)
    print(" Final cleaned size: %d" % len(df))
    return df


def late_night(row):
    return 1 if (row["hour"] <= 3 or row["hour"] >= 22) else 0


def night(row):
    return 1 if (20 < row["hour"] <= 23 and row["weekday"] < 5) else 0


def rush_hour(row):
    return 1 if (16 <= row["hour"] <= 20 and row["weekday"] < 5) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday

    df["night"] = ((df["hour"] > 20) & (df["hour"] <= 23) & (df["weekday"] < 5)).astype(
        int
    )
    df["late_night"] = ((df["hour"] <= 3) | (df["hour"] >= 22)).astype(int)
    df["rush_hour"] = (
        (df["hour"] >= 16) & (df["hour"] <= 20) & (df["weekday"] < 5)
    ).astype(int)
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


def add_haversine_feature(df):
    """
    Adds a haversine (great‑circle) distance column in kilometres.
    This inexpensive feature often improves fare prediction.
    """
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    df["haversine_km"] = earth_radius_km * c
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print(f"Submission written to {file_name}")




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




## === cell 2
testKaggle = clean(testKaggle)




## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.5, random_state=1)
test_df = test_df[:10000]  # keep a small hold‑out for quick checks




## === cell 4
print("Original train size:", len(trainKaggle))
print("Temp train size:", len(train_df))
print("Temp test size:", len(test_df))




## === cell 5
print("Cleaning training subset")
train_df = clean(train_df)
print("Cleaning test subset")
test_df = clean(test_df)




## === cell 6
print("Adding time features")
train_df = add_time_features(train_df)
test_df = add_time_features(test_df)
testKaggle = add_time_features(testKaggle)




## === cell 7
train_df = add_coordinate_features(train_df)
test_df = add_coordinate_features(test_df)
testKaggle = add_coordinate_features(testKaggle)

train_df = add_distances_features(train_df)
test_df = add_distances_features(test_df)
testKaggle = add_distances_features(testKaggle)

train_df = add_haversine_feature(train_df)
test_df = add_haversine_feature(test_df)
testKaggle = add_haversine_feature(testKaggle)




## === cell 8
drop_cols = ["pickup_datetime"]
train_df = train_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)
testKaggle_clean = testKaggle.drop(columns=drop_cols + ["key"])




## === cell 9
train_df, validation_df = train_test_split(train_df, test_size=0.1, random_state=1)




## === cell 10
train_labels = np.log1p(train_df["fare_amount"].values)
validation_labels = np.log1p(validation_df["fare_amount"].values)
test_labels = np.log1p(test_df["fare_amount"].values)

train_df = train_df.drop(columns=["fare_amount"])
validation_df = validation_df.drop(columns=["fare_amount"])
test_df = test_df.drop(columns=["fare_amount"])

print("Labels prepared (log‑transformed).")




## === cell 11
scaler = preprocessing.MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
validation_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)




## === cell 12
model = HistGradientBoostingRegressor(
    max_iter=5000,
    learning_rate=0.01,
    max_depth=6,
    random_state=42,
    early_stopping=False,  # ensure all trees are built
)
model.fit(train_scaled, train_labels)




## === cell 13
val_predictions_log = model.predict(validation_scaled)
val_predictions = np.expm1(val_predictions_log)

val_rmse_original = np.sqrt(
    np.mean((np.expm1(validation_labels) - val_predictions) ** 2)
)
print(f"Validation RMSE (original fare scale): {val_rmse_original:.4f}")

val_rmse_log = np.sqrt(np.mean((validation_labels - val_predictions_log) ** 2))
print(f"Validation RMSE (log‑space): {val_rmse_log:.4f}")




## === cell 14
test_predictions_log = model.predict(testKaggle_scaled)
test_predictions = np.expm1(test_predictions_log)  # convert back
output_submission(
    testKaggle,
    test_predictions,
    id_column="key",
    prediction_column="fare_amount",
    file_name=SUBMISSION_NAME,
)
