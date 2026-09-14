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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

import os

print(os.listdir("../input"))




## === cell 1
df = pd.read_csv("../input/train.csv")
test_set = pd.read_csv("../input/test.csv")




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


def select_within_boundingbox(df, BB):
    return (
        (df.pickup_longitude >= BB[0])
        & (df.pickup_longitude <= BB[1])
        & (df.pickup_latitude >= BB[2])
        & (df.pickup_latitude <= BB[3])
        & (df.dropoff_longitude >= BB[0])
        & (df.dropoff_longitude <= BB[1])
        & (df.dropoff_latitude >= BB[2])
        & (df.dropoff_latitude <= BB[3])
    )


print("Old size: %d" % len(df))
df = df[select_within_boundingbox(df, BB)]
df = df[(df.passenger_count > 0) & (df.passenger_count < 10)]
print("New size: %d" % len(df))




## === cell 4
def add_datetime_info(dataset):
    dataset["pickup_datetime"] = pd.to_datetime(dataset["pickup_datetime"])

    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["weekday"] = dataset.pickup_datetime.dt.weekday

    return dataset


df = add_datetime_info(df)
test_set = add_datetime_info(test_set)




## === cell 5
df["is_night"] = np.where(
    (
        ((df["hour"] >= 20) & (df["hour"] <= 23))
        | ((df["hour"] >= 0) & (df["hour"] < 6))
    ),
    1,
    0,
)
df["is_airport"] = np.where(
    (
        ((df["dropoff_longitude"] >= 73.77) & (df["dropoff_longitude"] <= 73.78))
        | ((df["dropoff_latitude"] >= 40.63) & (df["dropoff_latitude"] <= 40.64))
    ),
    1,
    0,
)
df["is_surge"] = np.where(
    (
        ((df["hour"] >= 16) & (df["hour"] < 20))
        & ((df["weekday"] != 5) & (df["weekday"] != 6))
    ),
    1,
    0,
)




## === cell 6
df = df.drop(df[df["fare_amount"] < 0].index, axis=0)




## === cell 7
plt.scatter(df["is_night"], df["fare_amount"], c="r")
plt.show()




## === cell 8
from sklearn.model_selection import train_test_split

train = df.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
test = df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.1)




## === cell 9
from xgboost import XGBRegressor

regressor = XGBRegressor(
    max_depth=10,
    learning_rate=0.1,
    n_estimators=500,
    objective="reg:squarederror",
    n_jobs=4,
    random_state=42,
)
regressor.fit(X_train, y_train)




## === cell 10
predictions = regressor.predict(X_test)




## === cell 11
import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(y_test, predictions))

print(rmse)




## === cell 12
test_set["is_night"] = np.where(
    (
        ((test_set["hour"] >= 20) & (test_set["hour"] <= 23))
        | ((test_set["hour"] >= 0) & (test_set["hour"] < 6))
    ),
    1,
    0,
)
test_set["is_airport"] = np.where(
    (
        (
            (test_set["dropoff_longitude"] >= 73.77)
            & (test_set["dropoff_longitude"] <= 73.78)
        )
        | (
            (test_set["dropoff_latitude"] >= 40.63)
            & (test_set["dropoff_latitude"] <= 40.64)
        )
    ),
    1,
    0,
)
test_set["is_surge"] = np.where(
    (
        ((test_set["hour"] >= 16) & (test_set["hour"] < 20))
        & ((test_set["weekday"] != 5) & (test_set["weekday"] != 6))
    ),
    1,
    0,
)

test_set_features = test_set.drop(["key", "pickup_datetime"], axis=1)

test_set_key = test_set["key"]

y_pred_final = regressor.predict(test_set_features)

submission = pd.DataFrame(
    {"key": test_set_key, "fare_amount": y_pred_final}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)
print("Submitted")




## === cell 13
submission.shape
