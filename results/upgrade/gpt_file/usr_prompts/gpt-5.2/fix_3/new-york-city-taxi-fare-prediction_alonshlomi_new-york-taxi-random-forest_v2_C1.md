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

3.12

# 3. Installed packages

geopandas==0.14.4
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

# 5. Target score

4.069922676663

# 6. Current score

7.1173

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 7.1173) has done: 'Your pipeline already runs and writes `submission.csv`, but it unintentionally *drops rows from the test set* during cleaning, which usually create an invalid submission (row count mismatch) and prevent getting a Kaggle score. I change the cleaning so that we only filter out bad rows in training (where we can), while keeping all test rows and instead clipping out-of-bounds coordinates to the NYC bounding box so every test key gets a prediction. I also align the training-time coordinate filtering and remove the fragile `for df in (train, test)` reassignment pattern to avoid silent mistakes. These are minimal changes that preserve your model/features and should yield a valid submission and a score that can move toward the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import os

BASE_PATH = "/kaggle/input"
TRAIN_PATH = f"{BASE_PATH}/train.csv"
TEST_PATH = f"{BASE_PATH}/test.csv"

print("Listing /kaggle/input:")
print(os.listdir(BASE_PATH))
print("Train exists:", os.path.exists(TRAIN_PATH))
print("Test exists:", os.path.exists(TEST_PATH))



## === cell 1
NROWS = 2_000_000  # adjust only if needed for memory/time; should be OK on Kaggle CPU

dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
    "key": "string",
}

train = pd.read_csv(
    TRAIN_PATH,
    nrows=NROWS,
    dtype=dtypes,
    parse_dates=["pickup_datetime"],
)
test = pd.read_csv(
    TEST_PATH,
    dtype={k: v for k, v in dtypes.items() if k != "fare_amount"},
    parse_dates=["pickup_datetime"],
)

print(train.shape, test.shape)
train.head()



## === cell 2
train.dropna(inplace=True)

train = train[(train["fare_amount"] > 0) & (train["fare_amount"] < 300)]
train = train[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 6)]

LON_MIN, LON_MAX = -75.0, -72.0
LAT_MIN, LAT_MAX = 40.0, 42.0

coord_mask_train = (
    train["pickup_longitude"].between(LON_MIN, LON_MAX)
    & train["dropoff_longitude"].between(LON_MIN, LON_MAX)
    & train["pickup_latitude"].between(LAT_MIN, LAT_MAX)
    & train["dropoff_latitude"].between(LAT_MIN, LAT_MAX)
)
train = train.loc[coord_mask_train].copy()

test = test.copy()
for col, lo, hi in [
    ("pickup_longitude", LON_MIN, LON_MAX),
    ("dropoff_longitude", LON_MIN, LON_MAX),
    ("pickup_latitude", LAT_MIN, LAT_MAX),
    ("dropoff_latitude", LAT_MIN, LAT_MAX),
]:
    test[col] = test[col].clip(lo, hi)

test["passenger_count"] = test["passenger_count"].clip(1, 6)

print("After cleaning:", train.shape, test.shape)




## === cell 3
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Great-circle distance (km) between two points given in degrees.
    Vectorized for pandas Series / numpy arrays.
    """
    R = 6371.0
    lat1 = np.radians(lat1.astype("float64"))
    lon1 = np.radians(lon1.astype("float64"))
    lat2 = np.radians(lat2.astype("float64"))
    lon2 = np.radians(lon2.astype("float64"))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c




## === cell 4
train["distance"] = haversine_distance(
    train["pickup_latitude"],
    train["pickup_longitude"],
    train["dropoff_latitude"],
    train["dropoff_longitude"],
).astype("float32")

train["day_of_week"] = train["pickup_datetime"].dt.dayofweek.astype("int8")
train["month"] = train["pickup_datetime"].dt.month.astype("int8")
train["hour"] = train["pickup_datetime"].dt.hour.astype("int8")

test["distance"] = haversine_distance(
    test["pickup_latitude"],
    test["pickup_longitude"],
    test["dropoff_latitude"],
    test["dropoff_longitude"],
).astype("float32")

test["day_of_week"] = test["pickup_datetime"].dt.dayofweek.astype("int8")
test["month"] = test["pickup_datetime"].dt.month.astype("int8")
test["hour"] = test["pickup_datetime"].dt.hour.astype("int8")

train.head()



## === cell 5
FEATURES = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
    "day_of_week",
    "month",
    "hour",
]

X = train[FEATURES]
y = train["fare_amount"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

random_forest = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f"Validation RMSE: {rmse:.6f}")



## === cell 6
random_forest.fit(X, y)

X_test = test[FEATURES]
y_pred_test = random_forest.predict(X_test).astype("float32")

y_pred_test = np.clip(y_pred_test, 0, None)

submission = pd.DataFrame({"key": test["key"].astype(str), "fare_amount": y_pred_test})

sample_sub_path = f"{BASE_PATH}/sample_submission.csv"
if os.path.exists(sample_sub_path):
    sample_sub = pd.read_csv(sample_sub_path, usecols=["key"])
    submission = sample_sub.merge(submission, on="key", how="left")
    fill_value = float(np.median(y.values))
    submission["fare_amount"] = (
        submission["fare_amount"].fillna(fill_value).astype("float32")
    )

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("File exists:", os.path.exists("submission.csv"))
print("Any NaNs in fare_amount:", submission["fare_amount"].isna().any())
