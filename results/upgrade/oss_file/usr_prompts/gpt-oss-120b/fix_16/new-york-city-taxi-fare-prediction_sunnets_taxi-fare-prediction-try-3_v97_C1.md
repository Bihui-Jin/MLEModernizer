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

4.1558

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.19714) has done: 'I add the missing utility functions (clean, add_time_features, add_distances_features, plotting, and output_submission) and replace the TensorFlow imports with the standalone `keras` package that is available, fixing the import‑related error. These changes resolve all NameError issues, ensure the model can be trained, and guarantee that a correctly‑named CSV file is written for submission.'
- What this solution (achieved 10.0171) has done: 'I replace the custom rmse metric with Keras’s built‑in RootMeanSquaredError to avoid the missing K.sqrt error, adjust the model compilation accordingly, and compute the validation RMSE manually after training. This fixes the runtime crashes and ensures a valid .csv submission is produced, while keeping the original model architecture and preprocessing unchanged.'
- What this solution (achieved 4.65502) has done: 'I replace the failing TensorFlow/Keras imports with a scikit‑learn regressor, fix the variable typo when computing validation RMSE, and remove the unnecessary loss‑plotting step. These changes eliminate the protobuf error, ensure a valid CSV submission is written, and, with a GradientBoostingRegressor, should bring the RMSE much closer to the target score while keeping the original preprocessing and feature engineering intact.'
- What this solution (achieved 4.8005) has done: 'I added cyclic hour and day‑of‑week features (sin/cos) in the time‑feature function to give the model a smoother representation of temporal patterns, and I slightly tuned the GradientBoostingRegressor (more trees, a bit lower learning rate and a deeper max depth with higher subsample) which should reduce the validation RMSE and move the score closer to the target without changing the overall pipeline.'
- What this solution (achieved 4.82289) has done: 'I increase the training sample size, add a log‑transformed distance feature (which often helps linear models like GradientBoosting), and slightly tune the GradientBoostingRegressor (more trees, lower learning rate, slightly shallower depth). These small, targeted changes keep the original pipeline intact while aiming to lower the RMSE toward the target value.'
- What this solution (achieved 4.87203) has done: 'I keep the overall pipeline unchanged but improve the model’s performance by (1) training the GradientBoostingRegressor on a log‑transformed fare amount (which often stabilises variance), then exponentiating the predictions back to the original scale, and (2) slightly strengthening the ensemble with more trees and a lower learning rate. These minimal tweaks preserve the core logic while expectedly lowering the validation RMSE toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

ROOT = "/kaggle/input"  # generic placeholder; replace if needed
TRAIN_PATH = os.path.join(ROOT, "train.csv")
TEST_PATH = os.path.join(ROOT, "test.csv")
SUBMISSION_PATH = "submission.csv"

DATASET_SIZE = 300_000  # rows to sample from training set (None → all)


def haversine(lon1, lat1, lon2, lat2):
    """Vectorised haversine distance (km) using float32 arithmetic."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def add_features(df):
    """Create distance and cyclic time features in‑place."""
    df["distance_km"] = haversine(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    )
    dt = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = dt.dt.hour.astype(np.float32)
    df["dow"] = dt.dt.dayofweek.astype(np.float32)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    return df


def clean_data(df, is_train=True):
    """Basic cleaning – remove obvious outliers and NaNs."""
    df = df.dropna()
    lon_min, lon_max = -74.5, -73.5
    lat_min, lat_max = 40.5, 41.0
    cond = (
        (df["pickup_longitude"].between(lon_min, lon_max))
        & (df["dropoff_longitude"].between(lon_min, lon_max))
        & (df["pickup_latitude"].between(lat_min, lat_max))
        & (df["dropoff_latitude"].between(lat_min, lat_max))
    )
    df = df[cond]
    if is_train:
        df = df[
            (df["fare_amount"] > 0)
            & (df["passenger_count"] > 0)
            & (df["passenger_count"] <= 6)
        ]
    return df




## === cell 1
train_usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
train_df = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    usecols=train_usecols,
    dtype={
        "fare_amount": "float32",
        "pickup_datetime": "str",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
)

test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_df = pd.read_csv(
    TEST_PATH,
    usecols=test_usecols,
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

train_df = clean_data(train_df, is_train=True)
test_df = clean_data(test_df, is_train=False)

train_df = add_features(train_df)
test_df = add_features(test_df)

train_labels = np.log1p(
    train_df["fare_amount"].values.astype(np.float32)
)  # log‑transform
feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance_km",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]
X = train_df[feature_cols].values.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(
    X, train_labels, test_size=0.2, random_state=42
)




## === cell 2
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)
X_test_scaled = scaler.transform(
    test_df[feature_cols].values.astype(np.float32)
).astype(np.float32)

gbr = GradientBoostingRegressor(
    n_estimators=800,  # was 500
    learning_rate=0.01,
    max_depth=6,
    subsample=0.9,
    random_state=42,
)

gbr.fit(X_train_scaled, y_train)

val_pred_log = gbr.predict(X_val_scaled)
val_pred = np.expm1(val_pred_log)
val_true = np.expm1(y_val)
val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
print(f"Validation RMSE: {val_rmse:.5f}")




## === cell 3
test_pred_log = gbr.predict(X_test_scaled)
test_pred = np.expm1(test_pred_log)

test_pred = np.maximum(test_pred, 0)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")
