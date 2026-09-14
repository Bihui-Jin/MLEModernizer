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

No external packages required in the script and installed.

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

3.6375985209102257

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.02934) has done: 'I replace the broken TensorFlow‑based model with a simple baseline that predicts the average fare from the training set. This removes the import errors, eliminates the faulty date‑feature code, and guarantees a valid `submission.csv` containing the required columns.'
- What this solution (achieved 10.02184) has done: 'I replace the constant‑mean baseline with a tiny linear model that uses three cheap features – haversine distance between pickup and drop‑off, passenger count, and pickup hour – and fit its coefficients by accumulating XᵀX and Xᵀy over the training CSV in chunks. This keeps the overall structure (reading CSVs in chunks, writing a `submission.csv`) while adding only lightweight feature engineering, which is expected to lower the RMSE from ~10 toward the target ~3.6.'
- What this solution (achieved 7.09199) has done: 'I extend the linear model with a few inexpensive, informative features (log distance, squared distance, latitude/longitude deltas, passenger count squared, and cyclical hour sin/cos) and adjust the matrix size accordingly. Adding a tiny ridge term ensures numerical stability, and clipping negative predictions to zero respects the fare domain. These minimal, targeted changes are expected to lower the RMSE toward the target without altering the overall workflow.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

print("Input directory contents:", os.listdir("../input"))




## === cell 1
df_test = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/test.csv",
    dtype={
        "pickup_longitude": np.float32,
        "pickup_latitude": np.float32,
        "dropoff_longitude": np.float32,
        "dropoff_latitude": np.float32,
        "passenger_count": np.int8,
    },
)
print("Test data shape:", df_test.shape)




## === cell 2
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km


p = 21
XTX = np.zeros((p, p))
XTy = np.zeros(p)

train_path = "../input/new-york-city-taxi-fare-prediction/train.csv"
chunk_iter = pd.read_csv(
    train_path,
    usecols=[
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype={
        "fare_amount": np.float32,
        "pickup_longitude": np.float32,
        "pickup_latitude": np.float32,
        "dropoff_longitude": np.float32,
        "dropoff_latitude": np.float32,
        "passenger_count": np.int8,
    },
    chunksize=1_000_000,
)

for chunk in chunk_iter:
    chunk = chunk.dropna()
    chunk = chunk[chunk["fare_amount"] < 200]

    dist = haversine(
        chunk["pickup_longitude"].values,
        chunk["pickup_latitude"].values,
        chunk["dropoff_longitude"].values,
        chunk["dropoff_latitude"].values,
    )
    delta_lat = chunk["dropoff_latitude"].values - chunk["pickup_latitude"].values
    delta_lon = chunk["dropoff_longitude"].values - chunk["pickup_longitude"].values
    log_dist = np.log1p(dist)
    dist_sq = dist**2

    passenger = chunk["passenger_count"].values.astype(float)
    passenger_sq = passenger**2
    log_passenger = np.log1p(passenger)

    hour = chunk["pickup_datetime"].str.slice(11, 13).astype(float)
    hour_rad = 2 * np.pi * hour / 24.0
    hour_sin = np.sin(hour_rad)
    hour_cos = np.cos(hour_rad)

    hour_pass = hour * passenger
    hour_cos_pass = hour_cos * passenger

    dist_pass = dist * passenger
    logdist_pass = log_dist * passenger
    dist_hour = dist * hour_sin  # example interaction with time of day

    X_chunk = np.column_stack(
        (
            np.ones(len(chunk)),  # intercept
            dist,
            log_dist,
            dist_sq,
            delta_lat,
            delta_lon,
            passenger,
            passenger_sq,
            hour,
            hour_sin,
            hour_cos,
            log_passenger,
            dist_pass,
            logdist_pass,
            dist_hour,
            hour_pass,  # new feature
            hour_cos_pass,  # new feature
            chunk["pickup_latitude"].values,
            chunk["pickup_longitude"].values,
            chunk["dropoff_latitude"].values,
            chunk["dropoff_longitude"].values,
        )
    )

    y_chunk = chunk["fare_amount"].values.astype(float)

    XTX += X_chunk.T @ X_chunk
    XTy += X_chunk.T @ y_chunk

ridge = 1e-4 * np.eye(p)
coeff = np.linalg.solve(XTX + ridge, XTy)
print("Fitted coefficients (raw‑fare model):", coeff)




## === cell 3
dist_test = haversine(
    df_test["pickup_longitude"].values,
    df_test["pickup_latitude"].values,
    df_test["dropoff_longitude"].values,
    df_test["dropoff_latitude"].values,
)
delta_lat_test = df_test["dropoff_latitude"].values - df_test["pickup_latitude"].values
delta_lon_test = (
    df_test["dropoff_longitude"].values - df_test["pickup_longitude"].values
)
log_dist_test = np.log1p(dist_test)
dist_sq_test = dist_test**2

passenger_test = df_test["passenger_count"].values.astype(float)
passenger_sq_test = passenger_test**2
log_passenger_test = np.log1p(passenger_test)

hour_test = df_test["pickup_datetime"].str.slice(11, 13).astype(float)
hour_rad_test = 2 * np.pi * hour_test / 24.0
hour_sin_test = np.sin(hour_rad_test)
hour_cos_test = np.cos(hour_rad_test)

hour_pass_test = hour_test * passenger_test
hour_cos_pass_test = hour_cos_test * passenger_test

dist_pass_test = dist_test * passenger_test
logdist_pass_test = log_dist_test * passenger_test
dist_hour_test = dist_test * hour_sin_test

X_test = np.column_stack(
    (
        np.ones(len(df_test)),
        dist_test,
        log_dist_test,
        dist_sq_test,
        delta_lat_test,
        delta_lon_test,
        passenger_test,
        passenger_sq_test,
        hour_test,
        hour_sin_test,
        hour_cos_test,
        log_passenger_test,
        dist_pass_test,
        logdist_pass_test,
        dist_hour_test,
        hour_pass_test,  # new feature
        hour_cos_pass_test,  # new feature
        df_test["pickup_latitude"].values,
        df_test["pickup_longitude"].values,
        df_test["dropoff_latitude"].values,
        df_test["dropoff_longitude"].values,
    )
)

y_pred = X_test @ coeff

y_pred = np.clip(y_pred, 0, 200)

submission = pd.DataFrame({"key": df_test["key"], "fare_amount": y_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}, shape: {submission.shape}")
