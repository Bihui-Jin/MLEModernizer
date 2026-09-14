# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.12

# 3. Installed packages

geopandas==0.14.4
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

# 5. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
from sklearn.linear_model import Ridge  # regularized linear regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def distance_on_the_sphere(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometres."""
    earth_radius = 6371.0
    lat1 = np.asarray(lat1, dtype=np.float32)
    lon1 = np.asarray(lon1, dtype=np.float32)
    lat2 = np.asarray(lat2, dtype=np.float32)
    lon2 = np.asarray(lon2, dtype=np.float32)

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return earth_radius * c


features = [
    "hour",
    "year",
    "distance",
    "log_distance",
    "passenger_count",
    "is_rush_hour",
    "day_of_week",
    "distance_to_downtown",
]
target = "fare_amount"

possible_roots = [
    "/kaggle/input/new-york-city-taxi-fare-prediction",
    "./data/new-york-city-taxi-fare-prediction",
    "./data",
]
data_root = next((p for p in possible_roots if os.path.isdir(p)), ".")

train_path = os.path.join(data_root, "train.csv")

dtype = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

new_york_box = (-74.763379, -72.856164, 40.502009, 41.915509)

chunks = pd.read_csv(
    train_path,
    usecols=usecols,
    dtype=dtype,
    memory_map=True,
    chunksize=500_000,  # smaller chunk for safety
)

SAMPLE_FRAC = 0.20
rng = np.random.RandomState(42)

X_parts = []
y_log_parts = []

nyc_down_town = (-74.0063889, 40.7141667)  # (lon, lat)

for chunk in chunks:
    chunk = chunk.drop(columns=["key"])
    chunk = chunk[chunk["fare_amount"] >= 0.1]
    chunk = chunk.dropna()

    mask = (
        (chunk["pickup_longitude"] >= new_york_box[0])
        & (chunk["pickup_longitude"] <= new_york_box[1])
        & (chunk["pickup_latitude"] >= new_york_box[2])
        & (chunk["pickup_latitude"] <= new_york_box[3])
        & (chunk["dropoff_longitude"] >= new_york_box[0])
        & (chunk["dropoff_longitude"] <= new_york_box[1])
        & (chunk["dropoff_latitude"] >= new_york_box[2])
        & (chunk["dropoff_latitude"] <= new_york_box[3])
    )
    chunk = chunk[mask]

    chunk["distance_to_downtown"] = distance_on_the_sphere(
        nyc_down_town[1],
        nyc_down_town[0],
        chunk["pickup_latitude"],
        chunk["pickup_longitude"],
    ).astype(np.float32)

    row_mask = (chunk["passenger_count"] != 0) & (chunk["distance_to_downtown"] < 15)
    chunk = chunk[row_mask]

    chunk["distance"] = distance_on_the_sphere(
        chunk["pickup_latitude"],
        chunk["pickup_longitude"],
        chunk["dropoff_latitude"],
        chunk["dropoff_longitude"],
    ).astype(np.float32)

    chunk["log_distance"] = np.log1p(chunk["distance"]).astype(np.float32)

    chunk["pickup_datetime"] = pd.to_datetime(chunk["pickup_datetime"])
    chunk["hour"] = chunk["pickup_datetime"].dt.hour.astype(np.uint8)
    chunk["year"] = chunk["pickup_datetime"].dt.year.astype(np.int16)
    chunk["day_of_week"] = chunk["pickup_datetime"].dt.dayofweek.astype(np.uint8)
    chunk["is_rush_hour"] = (
        ((chunk["hour"] >= 7) & (chunk["hour"] <= 10))
        | ((chunk["hour"] >= 16) & (chunk["hour"] <= 19))
    ).astype(np.uint8)

    X_chunk = chunk.loc[:, features].values.astype(np.float32, copy=False)
    y_chunk = np.log1p(chunk.loc[:, target].values.astype(np.float32, copy=False))

    if SAMPLE_FRAC < 1.0:
        keep_mask = rng.rand(X_chunk.shape[0]) < SAMPLE_FRAC
        X_chunk = X_chunk[keep_mask]
        y_chunk = y_chunk[keep_mask]

    X_parts.append(X_chunk)
    y_log_parts.append(y_chunk)

X = np.ascontiguousarray(np.concatenate(X_parts, axis=0), dtype=np.float32)
y_log = np.ascontiguousarray(np.concatenate(y_log_parts, axis=0), dtype=np.float32)

print(f"Total training samples after filtering & sampling: {X.shape[0]}")




## === cell 1
X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.25, random_state=42
)

linear_model = make_pipeline(
    StandardScaler(), Ridge(alpha=0.5, random_state=42, solver="cholesky")
)
linear_model.fit(X_train, y_train_log)

y_val_pred_log = linear_model.predict(X_val)
y_val_pred = np.expm1(y_val_pred_log)
y_val = np.expm1(y_val_log)
val_rmse = np.sqrt(((y_val - y_val_pred) ** 2).mean())
print(f"Validation RMSE (raw fare): {val_rmse:.5f}")
print(f"Target RMSE: 5.54066 – difference: {val_rmse - 5.54066:.5f}")




## === cell 2
test_path = os.path.join(data_root, "test.csv")
test_dtype = {
    "key": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

test_data_set = pd.read_csv(
    test_path,
    usecols=test_usecols,
    dtype=test_dtype,
    memory_map=True,
)

test_data_set["distance"] = distance_on_the_sphere(
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
    test_data_set["dropoff_latitude"],
    test_data_set["dropoff_longitude"],
).astype(np.float32)

test_data_set["log_distance"] = np.log1p(test_data_set["distance"]).astype(np.float32)

test_data_set["distance_to_downtown"] = distance_on_the_sphere(
    nyc_down_town[1],
    nyc_down_town[0],
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
).astype(np.float32)

test_data_set["pickup_datetime"] = pd.to_datetime(test_data_set["pickup_datetime"])
test_data_set["hour"] = test_data_set["pickup_datetime"].dt.hour.astype(np.uint8)
test_data_set["year"] = test_data_set["pickup_datetime"].dt.year.astype(np.int16)
test_data_set["day_of_week"] = test_data_set["pickup_datetime"].dt.dayofweek.astype(
    np.uint8
)
test_data_set["is_rush_hour"] = (
    ((test_data_set["hour"] >= 7) & (test_data_set["hour"] <= 10))
    | ((test_data_set["hour"] >= 16) & (test_data_set["hour"] <= 19))
).astype(np.uint8)




## === cell 3
X_test = test_data_set[features].values.astype(np.float32, copy=False)
y_pred_log = linear_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_pred = np.maximum(y_pred, 0.0)  # enforce non‑negative fares

submission = pd.DataFrame(
    {"key": test_data_set["key"], "fare_amount": y_pred},
    columns=["key", "fare_amount"],
)

os.makedirs("./output", exist_ok=True)
submission_path = "./output/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
