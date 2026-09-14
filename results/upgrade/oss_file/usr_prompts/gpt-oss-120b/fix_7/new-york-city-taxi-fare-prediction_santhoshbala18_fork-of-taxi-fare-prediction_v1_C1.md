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
sklearn-pandas==2.2.0
xgboost==2.0.3

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

3.55848

# 6. Current score

10.04272

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.22767) has done: 'I expand the file‑search to include the usual Kaggle input directory, fix the airport‑coordinate check (the longitudes are negative in this dataset), and tidy the flow so every variable is defined before it’s used. These minimal changes resolve the “file not found” and subsequent NameError issues, and the corrected features should bring the RMSE toward the target while preserving the original model logic.'
- What this solution (achieved 10.04272) has done: 'I increase the amount of training data (from 500 k to 2 M rows) and keep a larger proportion for training (validation size 5 %). I also cast the computed distance feature to float32 to stay memory‑efficient. These small adjustments keep the original model unchanged while giving it more data and a slightly stronger training‑set, which should lower the RMSE toward the target.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


def find_file(*possible_paths):
    """Return the first existing path among the candidates."""
    for p in possible_paths:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"None of the expected files exist: {possible_paths}")


DATA_ROOTS = [
    "./data",
    "/kaggle/input",
    "/kaggle/input/new-york-city-taxi-fare-prediction",
    "/kaggle/input/data",
]

TRAIN_PATH = find_file(*[os.path.join(root, "train.csv") for root in DATA_ROOTS])
TEST_PATH = find_file(*[os.path.join(root, "test.csv") for root in DATA_ROOTS])

SUBMISSION_PATH = "submission.csv"




## === cell 1
dtype_map = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

df = pd.read_csv(
    TRAIN_PATH,
    dtype=dtype_map,
    usecols=list(dtype_map.keys()) + ["key", "pickup_datetime"],
    nrows=2_000_000,
)

test_set = pd.read_csv(
    TEST_PATH,
    dtype={k: v for k, v in dtype_map.items() if k != "fare_amount"},
    usecols=[
        c
        for c in ["key", "pickup_datetime"] + list(dtype_map.keys())
        if c != "fare_amount"
    ],
)




## === cell 2
def distance(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometers."""
    p = 0.017453292519943295  # pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(a))


df["distance_km"] = distance(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"],
).astype(
    "float32"
)  # keep dtype consistent and memory‑friendly

test_set["distance_km"] = distance(
    test_set["pickup_latitude"],
    test_set["pickup_longitude"],
    test_set["dropoff_latitude"],
    test_set["dropoff_longitude"],
).astype("float32")




## === cell 3
BB = (-75, -73, 40, 41.5)  # longitude, latitude bounds for NYC area


def select_within_boundingbox(df, BB):
    """Keep rows whose pickup and drop‑off points lie inside BB."""
    return (
        (df["pickup_longitude"] >= BB[0])
        & (df["pickup_longitude"] <= BB[1])
        & (df["pickup_latitude"] >= BB[2])
        & (df["pickup_latitude"] <= BB[3])
        & (df["dropoff_longitude"] >= BB[0])
        & (df["dropoff_longitude"] <= BB[1])
        & (df["dropoff_latitude"] >= BB[2])
        & (df["dropoff_latitude"] <= BB[3])
    )


print(f"Original training size: {len(df)}")
df = df[select_within_boundingbox(df, BB)]
df = df[(df["passenger_count"] > 0) & (df["passenger_count"] < 10)]
print(f"Filtered training size: {len(df)}")




## === cell 4
def add_datetime_info(dataset):
    """Extract hour, day, month and weekday from the pickup timestamp."""
    dataset["pickup_datetime"] = pd.to_datetime(dataset["pickup_datetime"])
    dataset["hour"] = dataset["pickup_datetime"].dt.hour.astype("int8")
    dataset["day"] = dataset["pickup_datetime"].dt.day.astype("int8")
    dataset["month"] = dataset["pickup_datetime"].dt.month.astype("int8")
    dataset["weekday"] = dataset["pickup_datetime"].dt.weekday.astype("int8")
    return dataset


df = add_datetime_info(df)
test_set = add_datetime_info(test_set)




## === cell 5
def add_binary_features(df):
    """Create simple binary flags useful for fare modelling."""
    df["is_night"] = np.where(
        ((df["hour"] >= 20) & (df["hour"] <= 23))
        | ((df["hour"] >= 0) & (df["hour"] < 6)),
        1,
        0,
    ).astype("int8")
    df["is_airport"] = np.where(
        ((df["dropoff_longitude"] >= -73.78) & (df["dropoff_longitude"] <= -73.77))
        | ((df["dropoff_latitude"] >= 40.63) & (df["dropoff_latitude"] <= 40.64)),
        1,
        0,
    ).astype("int8")
    df["is_surge"] = np.where(
        ((df["hour"] >= 16) & (df["hour"] < 20))
        & (df["weekday"] != 5)
        & (df["weekday"] != 6),
        1,
        0,
    ).astype("int8")
    return df


df = add_binary_features(df)
test_set = add_binary_features(test_set)




## === cell 6
df = df.drop(df[df["fare_amount"] <= 0].index, axis=0)




## === cell 7
X = df.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = df["fare_amount"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)

regressor = XGBRegressor(
    max_depth=10,
    learning_rate=0.1,
    n_estimators=500,
    objective="reg:squarederror",
    n_jobs=4,
    random_state=42,
    tree_method="hist",
    eval_metric="rmse",
)

regressor.fit(X_train, y_train)

val_pred = regressor.predict(X_val)
rmse = math.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {rmse:.5f}")




## === cell 8
test_features = test_set.drop(["key", "pickup_datetime"], axis=1)
test_keys = test_set["key"]
test_pred = regressor.predict(test_features)

submission = pd.DataFrame(
    {"key": test_keys, "fare_amount": test_pred},
    columns=["key", "fare_amount"],
)
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")
print("Submission shape:", submission.shape)




## === cell 9
submission.head()
