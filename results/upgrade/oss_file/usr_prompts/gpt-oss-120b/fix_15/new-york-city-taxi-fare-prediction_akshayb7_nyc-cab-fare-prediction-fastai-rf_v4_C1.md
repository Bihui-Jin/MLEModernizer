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

3.7

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def _find_data_path():
    """Return the first directory (including sub‑dirs) that contains train.csv and test.csv."""
    candidates = [
        "./data",
        "./kaggle/data",
        "./kaggle/input",
        "./input",
        "./",
    ]
    for base in candidates:
        if not os.path.isdir(base):
            continue
        train_path = os.path.join(base, "train.csv")
        test_path = os.path.join(base, "test.csv")
        if os.path.isfile(train_path) and os.path.isfile(test_path):
            return base
        for root, _, files in os.walk(base):
            if "train.csv" in files and "test.csv" in files:
                return root
    raise FileNotFoundError(
        "Could not locate train.csv and test.csv in any known location."
    )


PATH = _find_data_path()

test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

test_dtype = {
    "key": str,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}




## === cell 1
def haversine_np(lon1, lat1, lon2, lat2):
    """Vectorized Haversine distance in kilometres."""
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(
        np.radians,
        [
            lon1.astype(float),
            lat1.astype(float),
            lon2.astype(float),
            lat2.astype(float),
        ],
    )
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c




## === cell 2
train_usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

df_raw = pd.read_csv(
    os.path.join(PATH, "train.csv"),
    usecols=train_usecols,
    parse_dates=["pickup_datetime"],
    dtype={
        "key": str,
        "fare_amount": np.float32,
        "pickup_longitude": np.float32,
        "pickup_latitude": np.float32,
        "dropoff_longitude": np.float32,
        "dropoff_latitude": np.float32,
        "passenger_count": np.int8,
    },
    nrows=800_000,  # read only 800k rows instead of the whole file
)

df_raw["distance"] = haversine_np(
    df_raw["pickup_longitude"],
    df_raw["pickup_latitude"],
    df_raw["dropoff_longitude"],
    df_raw["dropoff_latitude"],
)

df_raw["manhattan"] = np.abs(
    df_raw["pickup_longitude"] - df_raw["dropoff_longitude"]
) + np.abs(df_raw["pickup_latitude"] - df_raw["dropoff_latitude"])
df_raw["log_distance"] = np.log1p(df_raw["distance"])
df_raw["hour"] = df_raw["pickup_datetime"].dt.hour.astype(np.int8)
df_raw["weekday"] = df_raw["pickup_datetime"].dt.weekday.astype(np.int8)

feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
    "manhattan",
    "log_distance",
    "hour",
    "weekday",
]

X = df_raw[feature_cols].values
y = df_raw["fare_amount"].values

outlier_mask = (
    (df_raw["distance"] <= 100)
    & (df_raw["fare_amount"] >= 0)
    & (df_raw["fare_amount"] <= 200)
)
valid_mask = outlier_mask & ~np.isnan(X).any(axis=1)
X = X[valid_mask]
y = y[valid_mask]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




## === cell 3
rf = RandomForestRegressor(
    n_estimators=1200,
    max_features="sqrt",
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)

val_pred = rf.predict(X_val)
val_rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f"Validation RMSE: {val_rmse:.5f}")




## === cell 4
df_test = pd.read_csv(
    os.path.join(PATH, "test.csv"),
    usecols=test_usecols,
    dtype=test_dtype,
    parse_dates=["pickup_datetime"],
)

df_test["distance"] = haversine_np(
    df_test["pickup_longitude"],
    df_test["pickup_latitude"],
    df_test["dropoff_longitude"],
    df_test["dropoff_latitude"],
)

df_test["manhattan"] = np.abs(
    df_test["pickup_longitude"] - df_test["dropoff_longitude"]
) + np.abs(df_test["pickup_latitude"] - df_test["dropoff_latitude"])
df_test["log_distance"] = np.log1p(df_test["distance"])
df_test["hour"] = df_test["pickup_datetime"].dt.hour.astype(np.int8)
df_test["weekday"] = df_test["pickup_datetime"].dt.weekday.astype(np.int8)

X_test = df_test[feature_cols].values
if np.isnan(X_test).any():
    X_test = np.nan_to_num(X_test, nan=0.0)




## === cell 5
test_pred = rf.predict(X_test)
test_pred = np.clip(test_pred, 0, None)  # enforce non‑negative fares

submission = pd.DataFrame(
    {
        "key": df_test["key"],
        "fare_amount": test_pred,
    }
)

submission_path = "./submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
