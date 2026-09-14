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

# 5. Code solution

## === cell 0
import os, numpy as np, pandas as pd

print("Input folder contents:", os.listdir("../input"))




## === cell 1
usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtypes = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
train_path = "../input/train.csv"
train_df = pd.read_csv(
    train_path,
    usecols=usecols,
    dtype=dtypes,
    parse_dates=["pickup_datetime"],
    nrows=5_000_000,  # keep original sample size
)




## === cell 2
train_df["abs_diff_longitude"] = (
    train_df["dropoff_longitude"] - train_df["pickup_longitude"]
).abs()
train_df["abs_diff_latitude"] = (
    train_df["dropoff_latitude"] - train_df["pickup_latitude"]
).abs()


def haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in miles."""
    p = np.pi / 180.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))  # Earth radius ≈3958.8 miles


train_df["distance_miles"] = haversine_vec(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"],
)

train_df["hour"] = train_df["pickup_datetime"].dt.hour




## === cell 3
used_cols = [
    "fare_amount",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "distance_miles",
    "passenger_count",
    "hour",
]
train_df = train_df.dropna(subset=used_cols)

train_df = train_df[
    (train_df["abs_diff_longitude"] < 3.0)
    & (train_df["abs_diff_latitude"] < 3.0)
    & (train_df["fare_amount"] > 0)
    & (train_df["fare_amount"] <= 200)  # upper fare bound
    & (train_df["passenger_count"] <= 9)
    & (train_df["distance_miles"] > 0)
    & (train_df["distance_miles"] <= 100)  # reasonable distance bound
]

print(f"Rows after cleaning: {len(train_df)}")




## === cell 4
from sklearn.model_selection import train_test_split

y = train_df["fare_amount"].values.astype(np.float32)
X = train_df[
    [
        "abs_diff_longitude",
        "abs_diff_latitude",
        "distance_miles",
        "passenger_count",
        "hour",
    ]
].astype(np.float32)

train_X_df, val_X_df, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)




## === cell 5
def build_matrix(df):
    """Add a constant column for intercept (kept for compatibility)."""
    return np.column_stack(
        (
            df["abs_diff_longitude"].values,
            df["abs_diff_latitude"].values,
            df["distance_miles"].values,
            df["passenger_count"].values,
            df["hour"].values,
            np.ones(len(df)),
        )
    )


train_X = train_X_df.values  # shape (n_samples, 5), float32 already
val_X = val_X_df.values

print("Design matrix shapes:", train_X.shape, val_X.shape)




## === cell 6
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    random_state=42,
)
gbr.fit(train_X, train_y)

print("GradientBoostingRegressor trained.")




## === cell 7
from sklearn.metrics import mean_squared_error

val_pred = gbr.predict(val_X)
rmse = np.sqrt(mean_squared_error(val_y, val_pred))
print(f"Validation RMSE: {rmse:.4f}")




## === cell 8
test_path = "../input/test.csv"
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_dtypes = {
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
test_df = pd.read_csv(
    test_path,
    usecols=test_usecols,
    dtype=test_dtypes,
    parse_dates=["pickup_datetime"],
)

test_df["abs_diff_longitude"] = (
    test_df["dropoff_longitude"] - test_df["pickup_longitude"]
).abs()
test_df["abs_diff_latitude"] = (
    test_df["dropoff_latitude"] - test_df["pickup_latitude"]
).abs()
test_df["distance_miles"] = haversine_vec(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
)
test_df["hour"] = test_df["pickup_datetime"].dt.hour

test_features = test_df[
    [
        "abs_diff_longitude",
        "abs_diff_latitude",
        "distance_miles",
        "passenger_count",
        "hour",
    ]
].astype(np.float32)

test_predictions = gbr.predict(test_features.values)
test_predictions = np.round(test_predictions, 2)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_predictions})
submission.to_csv("submission.csv", index=False)
print("Submission written. Files in cwd:", os.listdir("."))
