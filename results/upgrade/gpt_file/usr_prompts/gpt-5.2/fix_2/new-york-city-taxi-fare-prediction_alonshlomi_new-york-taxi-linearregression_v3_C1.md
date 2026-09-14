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

9.411138644112697

# 6. Current score

5.9689

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5.9689) has done: 'I fix the broken file paths so the notebook loads the provided `train.csv`/`test.csv` from `/kaggle/input/`, and I restore the missing variables (`train`, `model`, etc.) by making the pipeline run in-order end-to-end. To keep the core logic the same (LinearRegression on engineered time features + Haversine distance), I won’t change the model or feature set, only make data loading robust and add minimal filtering to remove clearly invalid rows that otherwise cause unstable training and worse RMSE. Finally, I ensure a valid `submission.csv` with exactly `key,fare_amount` is written.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

BASE_DIR = "/kaggle/input"
TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = os.path.join(
        BASE_DIR, "new-york-city-taxi-fare-prediction", "train.csv"
    )
if not os.path.exists(TEST_PATH):
    TEST_PATH = os.path.join(BASE_DIR, "new-york-city-taxi-fare-prediction", "test.csv")

print("Using train:", TRAIN_PATH)
print("Using test :", TEST_PATH)
print("Input root listing:", os.listdir(BASE_DIR)[:20])



## === cell 1
usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train = pd.read_csv(TRAIN_PATH, usecols=usecols_train, nrows=2_000_000)
test = pd.read_csv(TEST_PATH, usecols=usecols_test)

train.head(), test.head()



## === cell 2
train.dropna(inplace=True)
test.dropna(inplace=True)

train = train[
    (train["fare_amount"] > 0)
    & (train["fare_amount"] < 500)
    & (train["passenger_count"] >= 1)
    & (train["passenger_count"] <= 6)
    & (train["pickup_longitude"].between(-75, -72))
    & (train["dropoff_longitude"].between(-75, -72))
    & (train["pickup_latitude"].between(40, 42))
    & (train["dropoff_latitude"].between(40, 42))
].copy()

test = test[
    (test["passenger_count"] >= 0)
    & (test["passenger_count"] <= 6)
    & (test["pickup_longitude"].between(-75, -72))
    & (test["dropoff_longitude"].between(-75, -72))
    & (test["pickup_latitude"].between(40, 42))
    & (test["dropoff_latitude"].between(40, 42))
].copy()

print("Train shape after cleaning:", train.shape)
print("Test shape after cleaning :", test.shape)




## === cell 3
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth.
    Inputs in degrees; output in kilometers.
    """
    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance




## === cell 4
train["distance"] = haversine_distance(
    train["pickup_latitude"],
    train["pickup_longitude"],
    train["dropoff_latitude"],
    train["dropoff_longitude"],
)

train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], errors="coerce", utc=True
)
train.dropna(subset=["pickup_datetime"], inplace=True)

train["day_of_week"] = train["pickup_datetime"].dt.dayofweek
train["month"] = train["pickup_datetime"].dt.month
train["hour"] = train["pickup_datetime"].dt.hour

train[["key", "fare_amount", "distance", "day_of_week", "month", "hour"]].head(5)



## === cell 5
test["distance"] = haversine_distance(
    test["pickup_latitude"],
    test["pickup_longitude"],
    test["dropoff_latitude"],
    test["dropoff_longitude"],
)

test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], errors="coerce", utc=True
)
test.dropna(subset=["pickup_datetime"], inplace=True)

test["day_of_week"] = test["pickup_datetime"].dt.dayofweek
test["month"] = test["pickup_datetime"].dt.month
test["hour"] = test["pickup_datetime"].dt.hour

test[["key", "distance", "day_of_week", "month", "hour"]].head(5)



## === cell 6
X = train[["distance", "day_of_week", "month", "hour"]]
y = train["fare_amount"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_val = model.predict(X_val)
rmse_val = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
print("Validation RMSE:", rmse_val)



## === cell 7
model.fit(X, y)



## === cell 8
X_test = test[["distance", "day_of_week", "month", "hour"]]
y_pred_test = model.predict(X_test)

y_pred_test = np.maximum(y_pred_test, 0)

submission = pd.DataFrame(
    {"key": test["key"].astype(str), "fare_amount": y_pred_test.astype(float)}
)

sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = os.path.join(
        BASE_DIR, "new-york-city-taxi-fare-prediction", "sample_submission.csv"
    )
sample = pd.read_csv(sample_path)

submission = sample[["key"]].merge(submission, on="key", how="left")
submission["fare_amount"] = submission["fare_amount"].fillna(
    float(np.mean(y_pred_test))
)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
