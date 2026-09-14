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

4.051660609378363

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 11.91892) has done: 'The script is rewritten to load the correct CSV files, clean and engineer features without external image resources, use `sklearn` models (avoiding the failing Keras imports), and finally write a proper `submissiontry_water.csv` containing the required `key` and `fare_amount` columns. All original logic is preserved where possible, and the changes only fix runtime errors and enable a valid submission.'

# 9. Code solution

## === cell 0
def clean(df, is_test=False):
    """Basic cleaning: drop NaNs, remove impossible coordinates and outliers.
    For test data we skip fare_amount related filters because the column is absent."""
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any")
    print(" After dropna: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        | (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    print(" After removing identical coords: %d" % len(df))

    df = df[
        (df["pickup_longitude"] != 0)
        & (df["dropoff_longitude"] != 0)
        & (df["pickup_latitude"] != 0)
        & (df["dropoff_latitude"] != 0)
    ]
    print(" After removing zero coords: %d" % len(df))

    min_lon, max_lon, min_lat, max_lat = -74.5, -72.8, 40.5, 41.8
    df = df[
        (df["pickup_longitude"] >= min_lon)
        & (df["pickup_longitude"] <= max_lon)
        & (df["dropoff_longitude"] >= min_lon)
        & (df["dropoff_longitude"] <= max_lon)
        & (df["pickup_latitude"] >= min_lat)
        & (df["pickup_latitude"] <= max_lat)
        & (df["dropoff_latitude"] >= min_lat)
        & (df["dropoff_latitude"] <= max_lat)
    ]
    print(" After NYC bbox filter: %d" % len(df))

    if not is_test:
        df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 200)]
        df = df[df["passenger_count"] > 0]
        print(" After fare & passenger filter: %d" % len(df))
    return df


def add_time_features(df):
    """Extract useful temporal features from pickup_datetime."""
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_datetime"])
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    return df


def haversine(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in kilometers."""
    p = np.pi / 180.0
    a = (
        np.sin((lat2 - lat1) * p / 2.0) ** 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2.0) ** 2
    )
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def add_distance_features(df):
    """Add haversine distance and simple Manhattan approximation."""
    df["haversine"] = haversine(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["manhattan"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs() + (
        df["pickup_longitude"] - df["dropoff_longitude"]
    ).abs()
    return df




## === cell 1
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import os

TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

DATASET_SIZE = 200000  # sample size to keep runtime reasonable
RANDOM_STATE = 42
TEST_SIZE = 0.10  # validation split




## === cell 2
dtype_dict = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
train_df = pd.read_csv(TRAIN_PATH, dtype=dtype_dict, nrows=DATASET_SIZE)
test_df = pd.read_csv(TEST_PATH, dtype=dtype_dict)

print(f"train shape: {train_df.shape}, test shape: {test_df.shape}")




## === cell 3
print("Cleaning training data")
train_df = clean(train_df, is_test=False)

print("Cleaning test data")
test_df = clean(test_df, is_test=True)




## === cell 4
print("Adding temporal features")
train_df = add_time_features(train_df)
test_df = add_time_features(test_df)




## === cell 5
print("Adding distance features")
train_df = add_distance_features(train_df)
test_df = add_distance_features(test_df)




## === cell 6
drop_cols = ["key", "pickup_datetime"]
X = train_df.drop(columns=drop_cols + ["fare_amount"])
y = np.log1p(train_df["fare_amount"])

X_test = test_df.drop(columns=drop_cols)

print(f"Feature matrix shape: {X.shape}")




## === cell 7
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}")




## === cell 8
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)




## === cell 9
gbr = GradientBoostingRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE
)

gbr.fit(X_train, y_train)

val_pred_log = gbr.predict(X_val)
val_pred = np.expm1(val_pred_log)

val_rmse = np.sqrt(mean_squared_error(np.expm1(y_val), val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")




## === cell 10
test_pred_log = gbr.predict(X_test_scaled)
test_pred = np.expm1(test_pred_log)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission_path = os.path.join(".", SUBMISSION_NAME)
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
