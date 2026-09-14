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

4.031294871457426

# 6. Current score

6.19265

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.92572) has done: 'I fixed the script by removing the TensorFlow dependency (which was causing import errors), making the `clean` function safe for the test set, skipping cleaning on the test data to keep all rows, and replacing the neural network with a scikit‑learn RandomForest regressor. These changes let the pipeline run end‑to‑end, generate a valid CSV submission, and keep the core modelling approach while staying within the original feature‑engineering logic.'
- What this solution (achieved 5.97186) has done: 'I add a haversine distance feature (a more realistic measure of trip length) and increase the forest size to give the model a bit more capacity. Both changes keep the existing pipeline intact while targeting a lower RMSE, moving the score toward the target.'
- What this solution (achieved 6.19265) has done: 'The script now reads a smaller, still representative subset of the data, uses explicit float32 types for numeric arrays, and frees memory after heavy steps. These changes cut the training time of the GradientBoostingRegressor well below the 600 s limit while keeping the same preprocessing, feature engineering, and model‑definition logic, thus preserving result accuracy.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor  # switched from RandomForest
import gc  # for explicit memory cleanup


def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    mask = (
        (
            (df["dropoff_longitude"] != df["pickup_longitude"])
            | (df["dropoff_latitude"] != df["pickup_latitude"])
        )
        & (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    )
    MinMax = (-74.5, -72.8, 40.5, 41.8)
    mask &= (MinMax[0] <= df["pickup_longitude"]) & (
        df["pickup_longitude"] <= MinMax[1]
    )
    mask &= (MinMax[0] <= df["dropoff_longitude"]) & (
        df["dropoff_longitude"] <= MinMax[1]
    )
    mask &= (MinMax[2] <= df["pickup_latitude"]) & (df["pickup_latitude"] <= MinMax[3])
    mask &= (MinMax[2] <= df["dropoff_latitude"]) & (
        df["dropoff_latitude"] <= MinMax[3]
    )
    if "fare_amount" in df.columns:
        mask &= (0 < df["fare_amount"]) & (df["fare_amount"] <= 50)
    mask &= df["passenger_count"] > 0

    df = df[mask]
    print(" New size after all filters: %d" % len(df))
    return df


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["minute"] = df["pickup_datetime"].dt.minute
    df["second"] = df["pickup_datetime"].dt.second
    return df


def add_coordinate_features(df):
    df["latdiff"] = df["pickup_latitude"] - df["dropoff_latitude"]
    df["londiff"] = df["pickup_longitude"] - df["dropoff_longitude"]
    return df


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_distances_features(df):
    df["manhattan"] = manhattan(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    return df


def add_haversine_features(df):
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    df["haversine"] = earth_radius_km * c
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {id_column: raw_test[id_column], prediction_column: prediction.squeeze()}
    )
    df.to_csv(file_name, index=False)
    print(f"Output written to {file_name}")




## === cell 1
TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

DATASET_SIZE = 200000  # rows (reduced from 500k for speed)

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

dtype_dict = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

train_df = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    usecols=usecols,
    dtype=dtype_dict,
    parse_dates=False,
)

testKaggle = pd.read_csv(
    TEST_PATH,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype={k: v for k, v in dtype_dict.items() if k != "fare_amount"},
    parse_dates=False,
)

print(f"Train rows loaded: {len(train_df)}")
print(f"Test rows loaded: {len(testKaggle)}")




## === cell 2
train_df = clean(train_df)

train_df = add_time_features(train_df)
testKaggle = add_time_features(testKaggle)

train_df = add_coordinate_features(train_df)
testKaggle = add_coordinate_features(testKaggle)

train_df = add_distances_features(train_df)
testKaggle = add_distances_features(testKaggle)

train_df = add_haversine_features(train_df)
testKaggle = add_haversine_features(testKaggle)

train_df = train_df.drop(["pickup_datetime"], axis=1)
testKaggle = testKaggle.drop(["pickup_datetime"], axis=1)

print("Feature engineering completed.")
gc.collect()  # free intermediate objects




## === cell 3
scaler = preprocessing.MinMaxScaler()
features = [c for c in train_df.columns if c not in ["key", "fare_amount"]]

train_df[features] = scaler.fit_transform(train_df[features]).astype(np.float32)
testKaggle[features] = scaler.transform(testKaggle[features]).astype(np.float32)

train_labels = train_df["fare_amount"].values.astype(np.float32)
train_ids = train_df["key"].values
train_df = train_df.drop(["key", "fare_amount"], axis=1)

test_ids = testKaggle["key"].values
testKaggle = testKaggle.drop(["key"], axis=1)

X = train_df.values.astype(np.float32)
X_test = testKaggle.values.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(
    X, train_labels, test_size=0.1, random_state=42
)

print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
gc.collect()




## === cell 4
model = GradientBoostingRegressor(
    n_estimators=600,  # more trees for better fit
    learning_rate=0.05,  # moderate learning rate
    max_depth=5,  # depth suitable for this problem
    subsample=0.8,  # add some bagging effect
    random_state=42,
)

model.fit(X_train, y_train)

val_pred = model.predict(X_val)
val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
print(f"Validation RMSE: {val_rmse:.4f}")




## === cell 5
test_predictions = model.predict(X_test)
output_submission(
    pd.DataFrame({"key": test_ids}),
    test_predictions,
    "key",
    "fare_amount",
    SUBMISSION_NAME,
)
