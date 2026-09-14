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

3.8575708313744257

# 6. Current score

4.8556

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.02965) has done: 'I fixed the import errors, corrected the data paths, added proper feature engineering (including haversine distance and datetime features), standardized the inputs, and rewrote the model training using tf.keras without the deprecated KerasClassifier wrapper. The script now trains a small dense network on a subset of the data, makes predictions for the test set, and writes a correctly‑formatted submission_file.csv so the pipeline runs end‑to‑end and yields a realistic RMSE that moves toward the target score.'
- What this solution (achieved 4.43811) has done: 'The fix adds a median imputer to replace NaNs before scaling, preventing the GradientBoostingRegressor error, and imports the required class. This minimal change restores the training pipeline, allowing the model to fit and produce a valid `submission_file.csv` while keeping the original logic intact.'
- What this solution (achieved 4.43637) has done: 'The fix limits the training size and reduces the number of boosting estimators, which cuts the dominant GBDT training time from well over ten minutes to under ten seconds while keeping all feature engineering, preprocessing, and model‑type unchanged.  The data‑loading and preprocessing steps are already fully vectorized, so no further speed gains are needed there.  All other logic (feature creation, imputation, scaling, prediction, and submission) remains identical, preserving the original result semantics.'
- What this solution (achieved 4.41688) has done: 'The changes reduce the training time of the GradientBoostingRegressor by lowering the number of trees and using stochastic boosting (subsample and max_features) while keeping the same model type and feature set. These hyper‑parameter adjustments preserve the overall logic and keep predictions comparable, and the rest of the pipeline (loading, feature engineering, imputation) remains unchanged.'
- What this solution (achieved 4.94479) has done: 'The changes limit the data size and model complexity to stay well under the 600‑second limit while keeping the exact same preprocessing, feature engineering, and model type. Reducing `nrows` to 200 000 still provides a representative sample, and lowering `n_estimators` to 300 cuts the training work roughly by half without altering the GradientBoostingRegressor logic or evaluation steps.'
- What this solution (achieved 4.8556) has done: 'I increase the GradientBoostingRegressor capacity modestly—raising `n_estimators` to 500, lowering `learning_rate` to 0.02, and expanding `max_depth` to 8. These small hyper‑parameter tweaks usually reduce RMSE for this dataset while keeping runtime well within the 600 s limit, moving the score closer to the target (lower is better). No other logic is altered.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

print("Environment ready")




## === cell 1
def find_file(possible_paths):
    """Return the first existing path from a list."""
    for p in possible_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the files exist: {possible_paths}")


def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance (km) using float32 arrays."""
    R = np.float32(6371.0)
    lat1 = np.radians(lat1.astype(np.float32, copy=False))
    lon1 = np.radians(lon1.astype(np.float32, copy=False))
    lat2 = np.radians(lat2.astype(np.float32, copy=False))
    lon2 = np.radians(lon2.astype(np.float32, copy=False))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_and_engineer(train_path, test_path, nrows=None):
    train_cols = [
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    test_cols = [
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]

    dtype_spec = {
        "pickup_longitude": np.float32,
        "pickup_latitude": np.float32,
        "dropoff_longitude": np.float32,
        "dropoff_latitude": np.float32,
        "passenger_count": np.int8,
        "fare_amount": np.float32,
        "key": str,
    }

    df_train = pd.read_csv(
        train_path,
        usecols=train_cols,
        parse_dates=["pickup_datetime"],
        nrows=nrows,
        dtype=dtype_spec,
    )
    df_test = pd.read_csv(
        test_path,
        usecols=test_cols,
        parse_dates=["pickup_datetime"],
        nrows=nrows,
        dtype=dtype_spec,
    )

    for df in (df_train, df_test):
        dt = df["pickup_datetime"]
        df["hour"] = dt.dt.hour.astype(np.int8)
        df["weekday"] = dt.dt.weekday.astype(np.int8)
        df["month"] = dt.dt.month.astype(np.int8)

    df_train["distance"] = haversine_distance(
        df_train["pickup_latitude"].values,
        df_train["pickup_longitude"].values,
        df_train["dropoff_latitude"].values,
        df_train["dropoff_longitude"].values,
    )
    df_test["distance"] = haversine_distance(
        df_test["pickup_latitude"].values,
        df_test["pickup_longitude"].values,
        df_test["dropoff_latitude"].values,
        df_test["dropoff_longitude"].values,
    )

    feature_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
        "hour",
        "weekday",
        "month",
    ]

    X_train = df_train[feature_cols].values.astype(np.float32, copy=False)
    y_train = df_train["fare_amount"].values.astype(np.float32, copy=False)
    X_test = df_test[feature_cols].values.astype(np.float32, copy=False)
    keys_test = df_test["key"].values

    valid_mask = ~np.isnan(y_train)
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]

    del df_train, df_test
    return X_train, y_train, X_test, keys_test




## === cell 2
train_file = find_file(
    [
        "../input/my-taxi-fare-data/train.csv",
        "../input/train.csv",
        "./train.csv",
        "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    ]
)
test_file = find_file(
    [
        "../input/my-taxi-fare-data/test.csv",
        "../input/test.csv",
        "./test.csv",
        "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    ]
)

X, y, X_test, test_keys = load_and_engineer(train_file, test_file, nrows=200000)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

median_log = np.nanmedian(y_train_log)
y_train_log = np.where(np.isnan(y_train_log), median_log, y_train_log)
y_val_log = np.where(np.isnan(y_val_log), median_log, y_val_log)




## === cell 3
def build_model():
    return GradientBoostingRegressor(
        n_estimators=500,  # increased from 300
        learning_rate=0.02,  # decreased to keep training stable
        max_depth=8,  # deeper trees for richer interactions
        subsample=0.8,
        max_features=0.8,
        random_state=42,
    )


model = build_model()
model.fit(X_train, y_train_log)

val_pred_log = model.predict(X_val)
val_pred = np.expm1(val_pred_log)
val_rmse = mean_squared_error(y_val, val_pred, squared=False)
print("Validation RMSE:", val_rmse)



## === cell 4
test_pred_log = model.predict(X_test)
test_pred = np.expm1(test_pred_log)
test_pred = np.maximum(test_pred, 0)  # fare cannot be negative

df_submission = pd.DataFrame({"key": test_keys, "fare_amount": test_pred})
output_path = "submission_file.csv"
df_submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
