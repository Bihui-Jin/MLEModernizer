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

6.1977

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.22611) has done: 'The fixes add proper keras imports, implement the missing preprocessing functions, correct the label handling order, and ensure a valid CSV submission is written. This restores end‑to‑end execution and produces a submission file while keeping the original modeling approach unchanged.'
- What this solution (achieved 227.57656) has done: 'The changes fix the import errors by using tf.keras instead of the standalone Keras package, restore the missing `sqrt` function in the custom RMSE metric, correct the data paths to the Kaggle input directory, and keep the useful `passenger_count` feature (removing only the raw datetime column). These fixes allow the model to train, evaluate, and write a proper CSV submission while nudging the RMSE toward the target score.'
- What this solution (achieved 8.97037) has done: 'I fixed the import error by removing the TensorFlow/Keras dependencies and replaced the neural‑network model with a GradientBoostingRegressor from scikit‑learn, keeping the same preprocessing steps and feature engineering. The new code scales the data, trains the GBM, evaluates RMSE on the validation split, and writes a correctly formatted CSV submission. This restores end‑to‑end execution and should bring the RMSE close to the target while preserving the original workflow logic.'
- What this solution (achieved 8.74718) has done: 'The update reduces the GradientBoostingRegressor training cost by decreasing the number of trees while compensating with a higher learning rate, keeping the same model type and overall training approach. This cuts runtime well under the 600‑second limit without altering feature engineering, data handling, or prediction logic.'
- What this solution (achieved 6.1977) has done: 'The changes reduce the amount of data processed and ensure all numeric arrays are `float32`, which speeds up the heavy GradientBoostingRegressor training while keeping the same model type, features, and preprocessing logic. The dataset size constant is lowered from 500 k to 200 k rows (still a large sample) and the scaler is applied to `float32` data, preserving the original feature engineering and evaluation steps.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

BASE_PATH = "/kaggle/input"
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 200000




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

testKaggle = pd.read_csv(TEST_PATH)




## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.20, random_state=1)
test_df = test_df[:10000]  # keep the original 10k limit for debugging




## === cell 3
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with NaNs and obvious out‑of‑range values."""
    df = df.dropna()
    coord_mask = (
        (df["pickup_longitude"].between(-80, -70))
        & (df["pickup_latitude"].between(40, 45))
        & (df["dropoff_longitude"].between(-80, -70))
        & (df["dropoff_latitude"].between(40, 45))
    )
    other_mask = (
        (df["passenger_count"] > 0)
        & (df["passenger_count"] <= 6)
        & (df["fare_amount"] > 0)
        & (df["fare_amount"] < 500)
    )
    return df[coord_mask & other_mask].reset_index(drop=True)


print("train_df clean")
train_df = clean(train_df)
test_df = clean(test_df)




## === cell 4
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day of week, month, year and cyclical hour features."""
    dt = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)




## === cell 5
def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in kilometers."""
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


def add_coordinate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder – currently no extra coordinate features."""
    return df


def add_distances_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add haversine distance and its squared term."""
    df["distance_km"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["distance_km_sq"] = df["distance_km"] ** 2
    return df


def add_manhattan_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add a simple Manhattan‑like distance (absolute lat + lon deltas)."""
    df["manhattan_km"] = np.abs(
        df["dropoff_latitude"] - df["pickup_latitude"]
    ) + np.abs(df["dropoff_longitude"] - df["pickup_longitude"])
    return df


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

print("train_df add_manhattan_distance")
train_df = add_manhattan_distance(train_df)
print("test_df add_manhattan_distance")
test_df = add_manhattan_distance(test_df)
print("testKaggle add_manhattan_distance")
testKaggle = add_manhattan_distance(testKaggle)

print("Done with Adding features")




## === cell 6
train_split, validation_split = train_test_split(
    train_df, test_size=0.10, random_state=1
)

train_labels = train_split["fare_amount"].values
validation_labels = validation_split["fare_amount"].values

train_features = train_split.drop(["fare_amount"], axis=1).reset_index(drop=True)
validation_features = validation_split.drop(["fare_amount"], axis=1).reset_index(
    drop=True
)

test_labels = test_df["fare_amount"].values
test_features = test_df.drop(["fare_amount"], axis=1).reset_index(drop=True)




## === cell 7
dropped_columns = ["pickup_datetime"]

train_features = train_features.drop(dropped_columns, axis=1)
validation_features = validation_features.drop(dropped_columns, axis=1)
test_features = test_features.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")




## === cell 8
scaler = preprocessing.MinMaxScaler()
train_features_scaled = scaler.fit_transform(train_features.astype(np.float32))
validation_features_scaled = scaler.transform(validation_features.astype(np.float32))
test_features_scaled = scaler.transform(test_features.astype(np.float32))
testKaggle_scaled = scaler.transform(testKaggle_clean.astype(np.float32))




## === cell 9
gbr = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    random_state=42,
)

print("Training GradientBoostingRegressor...")
gbr.fit(train_features_scaled, train_labels)




## === cell 10
val_pred = gbr.predict(validation_features_scaled)
val_rmse = mean_squared_error(validation_labels, val_pred, squared=False)
print(f"Validation RMSE: {val_rmse:.5f}")




## === cell 11
predictionKaggle = gbr.predict(testKaggle_scaled)


def output_submission(
    raw_test: pd.DataFrame,
    prediction: np.ndarray,
    id_column: str,
    prediction_column: str,
    file_name: str,
):
    df = pd.DataFrame(
        {id_column: raw_test[id_column].values, prediction_column: prediction}
    )
    df.to_csv(file_name, index=False)
    print(f"Output complete: {file_name}")


output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
