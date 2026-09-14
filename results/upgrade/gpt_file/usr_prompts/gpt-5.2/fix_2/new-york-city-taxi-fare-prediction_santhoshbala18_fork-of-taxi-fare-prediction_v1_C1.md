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

8.05365

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 8.05365) has done: 'Your current score is far from the target (9.83 vs 3.56 RMSE; lower is better), and the biggest issue is that the “is_airport” feature is computed with the wrong longitude sign (NYC longitudes are negative), making it essentially noise and hurting performance. I make the same airport/night/surge feature logic consistent and correct for both train and test (minimal change), add a light but standard fare/outlier filter to reduce label noise (still same model/loop), and ensure the train/test feature columns are perfectly aligned before prediction. These changes typically reduce RMSE substantially for this competition without altering the core approach (Haversine + datetime features + XGBRegressor).'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

print("Listing ../input (if exists):")
try:
    print(os.listdir("../input"))
except Exception as e:
    print("Could not list ../input:", repr(e))

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SAMPLE_SUB_PATH = "../input/sample_submission.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/train.csv"
    TEST_PATH = "/kaggle/input/test.csv"
    SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"



## === cell 1
df = pd.read_csv(TRAIN_PATH, nrows=10**6)
test_set = pd.read_csv(TEST_PATH)




## === cell 2
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


df["distance_km"] = distance(
    df.pickup_latitude, df.pickup_longitude, df.dropoff_latitude, df.dropoff_longitude
)
test_set["distance_km"] = distance(
    test_set.pickup_latitude,
    test_set.pickup_longitude,
    test_set.dropoff_latitude,
    test_set.dropoff_longitude,
)



## === cell 3
BB = (-75, -73, 40, 41.5)


def select_within_boundingbox(d, BB):
    return (
        (d.pickup_longitude >= BB[0])
        & (d.pickup_longitude <= BB[1])
        & (d.pickup_latitude >= BB[2])
        & (d.pickup_latitude <= BB[3])
        & (d.dropoff_longitude >= BB[0])
        & (d.dropoff_longitude <= BB[1])
        & (d.dropoff_latitude >= BB[2])
        & (d.dropoff_latitude <= BB[3])
    )


print("Old size: %d" % len(df))
df = df[select_within_boundingbox(df, BB)]
df = df[(df.passenger_count > 0) & (df.passenger_count < 10)]
print("New size: %d" % len(df))




## === cell 4
def add_datetime_info(dataset):
    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], utc=True, errors="coerce"
    )
    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["weekday"] = dataset.pickup_datetime.dt.weekday
    return dataset


df = add_datetime_info(df)
test_set = add_datetime_info(test_set)




## === cell 5
def add_rule_features(d):
    d["is_night"] = np.where(
        (
            ((d["hour"] >= 20) & (d["hour"] <= 23))
            | ((d["hour"] >= 0) & (d["hour"] < 6))
        ),
        1,
        0,
    )
    d["is_airport"] = np.where(
        (
            ((d["dropoff_longitude"] >= -73.79) & (d["dropoff_longitude"] <= -73.76))
            & ((d["dropoff_latitude"] >= 40.62) & (d["dropoff_latitude"] <= 40.66))
        ),
        1,
        0,
    )
    d["is_surge"] = np.where(
        (
            ((d["hour"] >= 16) & (d["hour"] < 20))
            & ((d["weekday"] != 5) & (d["weekday"] != 6))
        ),
        1,
        0,
    )
    return d


df = add_rule_features(df)
test_set = add_rule_features(test_set)



## === cell 6
df = df.drop(df[df["fare_amount"] < 0].index, axis=0)
df = df[(df["fare_amount"] >= 2.5) & (df["fare_amount"] <= 200)].copy()



## === cell 7
try:
    plt.scatter(df["is_night"], df["fare_amount"], c="r", s=1, alpha=0.3)
    plt.show()
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 8
from sklearn.model_selection import train_test_split

X = df.drop(
    ["key", "fare_amount", "pickup_datetime", "hour", "day", "month", "weekday"], axis=1
)
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)



## === cell 9
from xgboost import XGBRegressor

regressor = XGBRegressor(
    max_depth=10, learning_rate=0.1, n_estimators=200, n_jobs=-1, random_state=42
)
regressor.fit(X_train, y_train)



## === cell 10
predictions = regressor.predict(X_test)



## === cell 11
import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(y_test, predictions))
print("Holdout RMSE:", rmse)



## === cell 12
test_set_features = test_set.drop(
    ["key", "pickup_datetime", "hour", "day", "month", "weekday"], axis=1
)

test_set_features = test_set_features.reindex(columns=X.columns)

test_set_key = test_set["key"]
y_pred_final = regressor.predict(test_set_features)

submission = pd.DataFrame(
    {"key": test_set_key, "fare_amount": y_pred_final}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")



## === cell 13
print("Submission shape:", submission.shape)
print(submission.head())
