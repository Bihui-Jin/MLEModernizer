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
scipy==1.15.3
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

from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

print("input folder contents:", os.listdir("../input"))

alpha_ang = 0.506


def distance_travel(df):
    """Add crude distance‑related columns (in‑place) and return df."""
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs() * 50
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs() * 69
    df["displacement_vector"] = np.sqrt(
        df.abs_diff_latitude**2 + df.abs_diff_longitude**2
    )
    df["actual_long"] = (
        df.displacement_vector
        * np.sin(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude) - alpha_ang)
    ).abs()
    df["actual_lat"] = (
        df.displacement_vector
        * np.cos(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude) - alpha_ang)
    ).abs()
    df["distance_travel"] = df.actual_long + df.actual_lat
    return df


def haversine_distance(df):
    """Compute great‑circle distance (miles) between pickup and dropoff points."""
    lat1 = np.radians(df.pickup_latitude)
    lon1 = np.radians(df.pickup_longitude)
    lat2 = np.radians(df.dropoff_latitude)
    lon2 = np.radians(df.dropoff_longitude)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_miles = 3958.8
    df["haversine"] = earth_radius_miles * c
    return df


def add_time_features(df):
    """Extract hour, weekday and month from pickup_datetime."""
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["month"] = df["pickup_datetime"].dt.month
    return df


def data_clean(df):
    """Basic cleaning, feature creation and outlier removal."""
    df = df[df.passenger_count > 0].copy()
    if "fare_amount" in df.columns:
        df["fare_amount"] = df["fare_amount"].astype(np.float64)
        df = df[df["fare_amount"] > 0]

    df = df.dropna(
        subset=[
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "pickup_datetime",
        ]
    )

    df = distance_travel(df)
    df = haversine_distance(df)
    df = add_time_features(df)

    df = df[df["distance_travel"] > 0]
    return df


def remove_outliers(df):
    """Trim extreme values that would hurt model training."""
    df = df[df["distance_travel"] < 30]
    if "fare_amount" in df.columns:
        df = df[df["fare_amount"] < 100]
    return df




## === cell 1
dtype_map = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train_path = "../input/train.csv"

train_df = pd.read_csv(
    train_path,
    nrows=2_000_000,
    dtype=dtype_map,
    low_memory=False,
    parse_dates=["pickup_datetime"],  # <-- fast datetime parsing
    usecols=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],  # <-- drop unused columns early
)

train_df = data_clean(train_df)
train_df = remove_outliers(train_df)

X_train = np.column_stack(
    (
        train_df["haversine"].values,
        train_df["passenger_count"].values,
        train_df["hour"].values,
        train_df["weekday"].values,
        train_df["month"].values,
        train_df["pickup_longitude"].values,
        train_df["pickup_latitude"].values,
        train_df["dropoff_longitude"].values,
        train_df["dropoff_latitude"].values,
    )
)
y_train = np.log1p(train_df["fare_amount"].values)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_train = X_train.astype(np.float32)




## === cell 2
gbr = GradientBoostingRegressor(
    n_estimators=500,  # increased from 200
    learning_rate=0.03,
    max_depth=4,  # increased from 3
    subsample=0.8,
    random_state=42,
)
gbr.fit(X_train, y_train)




## === cell 3
test_path = "../input/test.csv"
test_df = pd.read_csv(
    test_path,
    dtype={k: v for k, v in dtype_map.items() if k != "fare_amount"},
    low_memory=False,
    parse_dates=["pickup_datetime"],  # <-- fast datetime parsing
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],  # <-- keep only needed columns
)

test_df = data_clean(test_df)  # reuse cleaning + feature creation (no fare column)

X_test = np.column_stack(
    (
        test_df["haversine"].values,
        test_df["passenger_count"].values,
        test_df["hour"].values,
        test_df["weekday"].values,
        test_df["month"].values,
        test_df["pickup_longitude"].values,
        test_df["pickup_latitude"].values,
        test_df["dropoff_longitude"].values,
        test_df["dropoff_latitude"].values,
    )
)
X_test = imputer.transform(X_test)
X_test = X_test.astype(np.float32)

pred_log = gbr.predict(X_test)
predictions = np.expm1(pred_log)
predictions = np.clip(predictions, 0, None)




## === cell 4
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": predictions})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")
