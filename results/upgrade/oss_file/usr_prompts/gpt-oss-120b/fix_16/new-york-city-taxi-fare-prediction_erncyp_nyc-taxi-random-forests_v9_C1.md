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
seaborn==0.12.2
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

# 5. Target score

3.41764

# 6. Current score

5.18505

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.72263) has done: 'I add two informative features—`passenger_count` and `minute`—to the model input and increase the number of trees in the RandomForest from 20 to 100 to improve predictive power while keeping the same overall pipeline. These changes should lower the RMSE toward the target score.'
- What this solution (achieved 5.36121) has done: 'I fix the NaN‑related crashes by removing rows with non‑positive fares (which make `log1p` produce NaNs) and by filling missing “busyness” values before model training. These minimal fixes let the pipeline run end‑to‑end and generate a proper `submission.csv` without altering the core modeling logic.'
- What this solution (achieved 5.76185) has done: 'The main slowdown is the RandomForest training on 1 M rows. By loading the Intel‑optimized scikit‑learn extensions (`sklearn‑ex`) before importing `RandomForestRegressor`, the same model run with optimized C++ kernels, keeping the exact algorithm unchanged but drastically reducing runtime.'
- What this solution (achieved 5.18505) has done: 'The fixes address the KeyError when loading the test set by only applying dtypes to columns that have a defined type, and ensure `test_df` is created correctly. Small feature enhancements (cyclical hour encoding) are added to improve the model slightly without changing its core logic. All cells now run end‑to‑end and produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def find_data_path(filename: str) -> str:
    """
    Locate a data file inside the Kaggle folder hierarchy.
    Searches common directories and falls back to a recursive search.
    Returns the first existing path; otherwise returns the original filename.
    """
    possible_dirs = [
        "./kaggle/data",
        "./kaggle/data/new-york-city-taxi-fare-prediction",
        "./data",
        "./data/new-york-city-taxi-fare-prediction",
    ]
    for d in possible_dirs:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p

    for root, _, files in os.walk("."):
        if filename in files:
            return os.path.join(root, filename)

    return filename




## === cell 1
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
dtypes = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train_path = find_data_path("train.csv")
train_df = pd.read_csv(
    train_path,
    nrows=2_000_000,
    usecols=usecols,
    dtype=dtypes,
    low_memory=False,
)
train_df.dropna(inplace=True)
train_df = train_df[train_df["fare_amount"] > 0]

test_path = find_data_path("test.csv")
test_dtype = {k: dtypes[k] for k in usecols if k != "fare_amount" and k in dtypes}
test_df = pd.read_csv(
    test_path,
    usecols=[c for c in usecols if c != "fare_amount"],
    dtype=test_dtype,
    low_memory=False,
)




## === cell 2
def haversine_distance(lat1, lon1, lat2, lon2):
    """Return distance in kilometres between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km


def add_features(df):
    """Create simple engineered features used by the model."""
    dt = pd.to_datetime(df["pickup_datetime"])
    df["year"] = dt.dt.year.astype(np.int16)
    df["month"] = dt.dt.month.astype(np.int8)
    df["day"] = dt.dt.day.astype(np.int8)
    df["hour"] = dt.dt.hour.astype(np.int8)
    df["minute"] = dt.dt.minute.astype(np.int8)
    df["day_of_week"] = dt.dt.dayofweek.astype(np.int8)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["distance"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    ).astype(np.float32)
    feature_cols = [
        "distance",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "day_of_week",
        "hour_sin",
        "hour_cos",
        "passenger_count",
    ]
    return df[feature_cols]


X = add_features(train_df).values.astype(np.float32)
Y = np.log1p(train_df["fare_amount"].values.astype(np.float32))

X_test = add_features(test_df).values.astype(np.float32)




## === cell 3
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(
    n_estimators=400,  # slightly more trees for better accuracy
    max_features="sqrt",
    min_samples_leaf=1,
    min_samples_split=2,
    bootstrap=True,
    max_depth=None,
    max_samples=0.7,  # use a larger subset of data for slightly higher quality
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)

val_pred = np.expm1(rf.predict(X_val))
val_rmse = np.sqrt(mean_squared_error(np.expm1(y_val), val_pred))
print(f"Validation RMSE (raw target): {val_rmse:.5f}")




## === cell 4
test_pred = np.expm1(rf.predict(X_test))

submission = pd.DataFrame(
    {"key": test_df["key"], "fare_amount": test_pred},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
