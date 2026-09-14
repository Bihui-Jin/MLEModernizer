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
haversine==2.9.0
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

4.01592

# 6. Current score

5.87707

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5.87707) has done: 'I replace the default RandomForest with a GradientBoostingRegressor trained on a log‑transformed target, which usually yields a much lower RMSE for skewed fare amounts. The prediction on the test set is back‑transformed with exp−1, and the same model is used for generating the final submission.csv. This change keeps the overall pipeline and feature engineering intact while moving the score much closer to the target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os

print(os.listdir("../input"))




## === cell 1
from haversine import haversine
from sklearn import model_selection




## === cell 2
train_df = pd.read_csv("../input/train.csv", nrows=100000)
train_df.head(2)




## === cell 3
train_df.dtypes




## === cell 4
train_df.shape




## === cell 5
print(train_df.isnull().sum())




## === cell 6
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))




## === cell 7
train_df.describe()




## === cell 8
train_df = train_df[(train_df.fare_amount > 0)]
train_df = train_df[(train_df.passenger_count > 0) & (train_df.passenger_count < 9)]
train_df.shape




## === cell 9
train_df = train_df[
    (train_df.pickup_longitude > -75) & (train_df.pickup_longitude < -72)
]
train_df = train_df[
    (train_df.dropoff_longitude > -75) & (train_df.dropoff_longitude < -72)
]
train_df = train_df[(train_df.pickup_latitude > 39) & (train_df.pickup_latitude < 42)]
train_df = train_df[(train_df.dropoff_latitude > 39) & (train_df.dropoff_latitude < 72)]
train_df.shape




## === cell 10
def add_distance_feature(df):
    distance = []
    for index, row in df.iterrows():
        distance.append(
            haversine(
                (row["pickup_latitude"], row["pickup_longitude"]),
                (row["dropoff_latitude"], row["dropoff_longitude"]),
            )
        )
    df["distance"] = distance




## === cell 11
add_distance_feature(train_df)
train_df.head(2)




## === cell 12
def add_time_and_day_feature(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["hour_of_day"] = df["pickup_datetime"].dt.hour


add_time_and_day_feature(train_df)
train_df.head(2)




## === cell 13
train_df.describe()




## === cell 14
train_df = train_df[(train_df.distance > 0.25)]
train_df.describe()




## === cell 15
X = train_df[
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
        "day_of_week",
        "hour_of_day",
    ]
]
y = train_df["fare_amount"].values

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=1
)

print(X_train.shape, y_train.shape)




## === cell 16
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))




## === cell 17
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=5, random_state=1
)
gbr.fit(X_train, np.log1p(y_train))
y_pred_log = gbr.predict(X_test)
y_pred = np.expm1(y_pred_log)
np.sqrt(mean_squared_error(y_test, y_pred))




## === cell 18
test_df = pd.read_csv("../input/test.csv")
add_distance_feature(test_df)
add_time_and_day_feature(test_df)
test_X = test_df[
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
        "day_of_week",
        "hour_of_day",
    ]
]

test_pred_log = gbr.predict(test_X)
test_y_predictions = np.expm1(test_pred_log)

submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": test_y_predictions},
    columns=["key", "fare_amount"],
)

submission.to_csv("submission.csv", index=False)

print(os.listdir("."))
