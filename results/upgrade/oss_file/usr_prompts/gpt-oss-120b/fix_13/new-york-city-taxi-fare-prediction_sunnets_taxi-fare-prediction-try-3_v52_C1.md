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

4.22554

# 6. Current score

4.94272

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.65849) has done: 'I remove the broken TensorFlow imports and undefined helper functions, add proper preprocessing (cleaning, time‑based features, and haversine distance), and replace the unfinished neural‑network model with a lightweight GradientBoostingRegressor that can be trained quickly on the sampled data. The script now loads the data, creates features, evaluates RMSE on a validation split, predicts the test set, and writes a correctly formatted `submissiontry_water.csv` file.'
- What this solution (achieved 4.72752) has done: 'I remove unnecessary Min‑Max scaling (tree models don’t need it) and keep the raw numeric features, while slightly strengthening the GradientBoostingRegressor (more trees, deeper depth, higher subsample). These minimal tweaks keep the original workflow but are expected to lower the validation RMSE toward the target.'
- What this solution (achieved 4.74156) has done: 'I add cyclical hour and day‑of‑week features (sin / cos) to better capture temporal patterns, and increase the GradientBoostingRegressor’s number of trees modestly. These changes keep the overall pipeline unchanged while giving the model slightly richer information, which should lower the validation RMSE toward the target.'
- What this solution (achieved 4.75008) has done: 'I add a simple interaction feature `distance_per_passenger` (which captures cost per rider) and slightly increase the model capacity by using more trees with a lower learning rate. These minimal changes keep the original pipeline intact while giving the GradientBoostingRegressor a bit richer information, which should lower the validation RMSE toward the target.'
- What this solution (achieved 4.94272) has done: 'The update keeps the same preprocessing and model type but speeds up training by reducing the number of trees and tree depth, which cuts the computational work dramatically while leaving the overall gradient‑boosting approach unchanged. The reduced hyper‑parameters still use the same loss (squared error on log‑targets) and the same data‑scaling, so predictions remain comparable. All other cells stay identical.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from sklearnex import patch_sklearn

patch_sklearn()
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import os

np.random.seed(42)

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 1000  # kept for compatibility, not used
EPOCHS = 30  # kept for compatibility, not used
LEARNING_RATE = 0.001  # kept for compatibility, not used
DATASET_SIZE = 80000  # number of rows to read from training data




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
    TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=[0, 1, 2, 3, 4, 5, 6, 7]
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
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values, out‑of‑range coordinates,
    and implausible fares (if the column exists)."""
    df = df.dropna()
    lon_min, lon_max = -180, 180
    lat_min, lat_max = -90, 90
    coord_mask = (
        (df["pickup_longitude"].between(lon_min, lon_max))
        & (df["dropoff_longitude"].between(lon_min, lon_max))
        & (df["pickup_latitude"].between(lat_min, lat_max))
        & (df["dropoff_latitude"].between(lat_min, lat_max))
    )
    if "fare_amount" in df.columns:
        fare_mask = df["fare_amount"].between(0.01, 500)
        return df[coord_mask & fare_mask]
    else:
        return df[coord_mask]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day of week and month from pickup_datetime."""
    dt = pd.to_datetime(df["pickup_datetime"])
    df["pickup_hour"] = dt.dt.hour.astype(np.int8)
    df["pickup_dayofweek"] = dt.dt.dayofweek.astype(np.int8)
    df["pickup_month"] = dt.dt.month.astype(np.int8)
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create sine / cosine transforms for cyclic time features."""
    df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24).astype(np.float32)
    df["dow_sin"] = np.sin(2 * np.pi * df["pickup_dayofweek"] / 7).astype(np.float32)
    df["dow_cos"] = np.cos(2 * np.pi * df["pickup_dayofweek"] / 7).astype(np.float32)
    return df


def haversine(lon1, lat1, lon2, lat2):
    """Vectorised haversine distance in kilometres."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add haversine distance between pickup and dropoff points."""
    df["distance_km"] = haversine(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple interaction: distance per passenger."""
    eps = 1e-3
    df["distance_per_passenger"] = (
        df["distance_km"] / (df["passenger_count"] + eps)
    ).astype(np.float32)
    return df


trainKaggle = clean(trainKaggle)
trainKaggle = add_time_features(trainKaggle)
trainKaggle = add_distance_feature(trainKaggle)
trainKaggle = add_cyclical_features(trainKaggle)
trainKaggle = add_interaction_features(trainKaggle)

testKaggle = clean(testKaggle)
testKaggle = add_time_features(testKaggle)
testKaggle = add_distance_feature(testKaggle)
testKaggle = add_cyclical_features(testKaggle)
testKaggle = add_interaction_features(testKaggle)




## === cell 3
train_df, validation_df = train_test_split(trainKaggle, test_size=0.10, random_state=1)

train_labels = np.log1p(train_df["fare_amount"].values)
validation_labels = np.log1p(validation_df["fare_amount"].values)

drop_cols = ["fare_amount", "pickup_datetime", "key"]
train_df = train_df.drop(columns=drop_cols)
validation_df = validation_df.drop(columns=drop_cols)




## === cell 4
train_df_scaled = train_df.to_numpy(dtype=np.float32, copy=False)
validation_df_scaled = validation_df.to_numpy(dtype=np.float32, copy=False)




## === cell 5
gbr = GradientBoostingRegressor(
    n_estimators=500,  # fewer trees → much faster fit
    learning_rate=0.02,
    max_depth=6,  # shallower trees reduce computation per split
    subsample=0.9,
    random_state=42,
)
gbr.fit(train_df_scaled, train_labels)




## === cell 6
val_pred_log = gbr.predict(validation_df_scaled)
val_pred = np.expm1(val_pred_log)  # back‑to‑original scale
val_rmse = np.sqrt(mean_squared_error(np.expm1(validation_labels), val_pred))
print(f"Validation RMSE: {val_rmse:.5f}")




## === cell 7
test_features = testKaggle.drop(columns=["pickup_datetime", "key"])
test_scaled = test_features.to_numpy(dtype=np.float32, copy=False)




## === cell 8
test_pred_log = gbr.predict(test_scaled)
test_predictions = np.expm1(test_pred_log)  # convert back to fare amount




## === cell 9
def output_submission(
    df_keys: pd.DataFrame,
    preds: np.ndarray,
    key_col: str,
    target_col: str,
    filename: str,
):
    """Write a Kaggle submission file with the required columns."""
    sub = pd.DataFrame({key_col: df_keys[key_col], target_col: preds})
    if not filename.lower().endswith(".csv"):
        filename = f"{filename}.csv"
    sub.to_csv(filename, index=False)
    print(f"Submission written to {filename}")




## === cell 10
output_submission(
    testKaggle,
    test_predictions,
    key_col="key",
    target_col="fare_amount",
    filename=SUBMISSION_NAME,
)




## === cell 11
print("Sample prediction vs true label (validation)")
print(val_pred[0], np.expm1(validation_labels)[0])
