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

# 5. Target score

3.47816

# 6. Current score

800.26126

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 800.26126) has done: 'The main timeout drivers are pandas doing repeated expensive filtering/dropping, multiple full DataFrame scans for “inspection” cells, and (most importantly) fitting a default `RandomForestRegressor` on 1,000,000 rows which is far too slow. The refactor below preserves the same features, cleaning rules, and model logic, but makes them run faster by (1) collapsing many drop operations into boolean masks, (2) replacing slow `apply(axis=1)` with equivalent vectorized math, (3) avoiding redundant heavy computations/prints, and (4) using scikit-learn’s Intel acceleration (`sklearnex`) and reusing the same loaded `sample_submission` DataFrame.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearnex import patch_sklearn

patch_sklearn()

import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

print(os.listdir("../input"))

np.random.seed(42)



## === cell 1
train_cols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_cols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtypes_train = {
    "fare_amount": "float64",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int64",
}
dtypes_test = {
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int64",
}

train = pd.read_csv(
    "../input/train.csv", nrows=1000000, usecols=train_cols, dtype=dtypes_train
)
test = pd.read_csv("../input/test.csv", usecols=test_cols, dtype=dtypes_test)



## === cell 2
train.shape



## === cell 3
test.shape



## === cell 4
train.head(10)



## === cell 5
train.describe()



## === cell 6
train.isnull().sum().sort_values(ascending=False)



## === cell 7
mask = np.ones(len(train), dtype=bool)

mask &= ~train.isnull().any(axis=1)

mask &= train["fare_amount"] >= 0

mask &= train["passenger_count"] != 208

mask &= train["pickup_latitude"].between(-90, 90)
mask &= train["dropoff_latitude"].between(-90, 90)
mask &= train["pickup_longitude"].between(-180, 180)

train = train.loc[mask].copy()

train.shape



## === cell 8
train["fare_amount"].describe()



## === cell 9
from collections import Counter

Counter(train["fare_amount"] < 0)



## === cell 10
train = train.drop(train[train["fare_amount"] < 0].index, axis=0)
train.shape



## === cell 11
train["fare_amount"].describe()



## === cell 12
train["fare_amount"].nlargest(20)



## === cell 13
train["passenger_count"].describe()



## === cell 14
train[train["passenger_count"] > 8]



## === cell 15
train = train.drop(train[train["passenger_count"] == 208].index, axis=0)



## === cell 16
train["passenger_count"].describe()



## === cell 17
train["pickup_latitude"].describe()



## === cell 18
train[train["pickup_latitude"] < -90]



## === cell 19
train[train["pickup_latitude"] > 90]



## === cell 20
train = train.drop(
    (
        (train[train["pickup_latitude"] < -90]) | (train[train["pickup_latitude"] > 90])
    ).index,
    axis=0,
)



## === cell 21
train.shape



## === cell 22
train["pickup_longitude"].describe()



## === cell 23
train[train["pickup_longitude"] < -180]



## === cell 24
train[train["pickup_longitude"] > 180]



## === cell 25
train = train.drop(
    (
        (train[train["pickup_longitude"] < -180])
        | (train[train["pickup_longitude"] > 180])
    ).index,
    axis=0,
)



## === cell 26
train[train["dropoff_latitude"] < -90]



## === cell 27
train[train["dropoff_latitude"] > 90]



## === cell 28
train = train.drop(
    (
        (train[train["dropoff_latitude"] < -90])
        | (train[train["dropoff_latitude"] > 90])
    ).index,
    axis=0,
)



## === cell 29
cond = (train["dropoff_latitude"] < -180) | (train["dropoff_latitude"] > 180)
cond.sum()



## === cell 30
train.dtypes



## === cell 31
train["key"] = pd.to_datetime(train["key"], errors="coerce", utc=False)
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], errors="coerce", utc=False
)



## === cell 32
test["key"] = pd.to_datetime(test["key"], errors="coerce", utc=False)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], errors="coerce", utc=False
)



## === cell 33
train.dtypes




## === cell 34
def _add_haversine(df, lat1, long1, lat2, long2):
    r = 6371.0
    phi1 = np.radians(df[lat1].to_numpy())
    phi2 = np.radians(df[lat2].to_numpy())
    delta_phi = np.radians((df[lat2] - df[lat1]).to_numpy())
    delta_lambda = np.radians((df[long2] - df[long1]).to_numpy())
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (
        np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    df["H_Distance"] = r * c


def haversine_distance(lat1, long1, lat2, long2):
    _add_haversine(train, lat1, long1, lat2, long2)
    _add_haversine(test, lat1, long1, lat2, long2)
    return train["H_Distance"]




## === cell 35
haversine_distance(
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)



## === cell 36
train["H_Distance"].head(10)



## === cell 37
for i in (train, test):
    dt = i["pickup_datetime"].dt
    i["Year"] = dt.year
    i["Month"] = dt.month
    i["Date"] = dt.day
    i["Day of Week"] = dt.dayofweek
    i["Hour"] = dt.hour



## === cell 38
train.loc[
    ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
    & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
    & (train["fare_amount"] == 0)
]



## === cell 39
cond = (
    ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
    & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
    & (train["fare_amount"] == 0)
)
train = train.loc[~cond].copy()



## === cell 40
train.shape



## === cell 41
train.loc[
    ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
    & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
    & (train["fare_amount"] == 0)
]



## === cell 42
cond = (
    ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
    & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
    & (train["fare_amount"] == 0)
)
train = train.loc[~cond].copy()



## === cell 43
high_distance = train.loc[(train["H_Distance"] > 200) & (train["fare_amount"] != 0)]



## === cell 44
high_distance.head()



## === cell 45
high_distance = high_distance.copy()
high_distance["H_Distance"] = (high_distance["fare_amount"] - 2.50) / 1.56



## === cell 46
train.update(high_distance)



## === cell 47
train[train["H_Distance"] == 0]



## === cell 48
train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)]



## === cell 49
cond = (train["H_Distance"] == 0) & (train["fare_amount"] == 0)
train = train.loc[~cond].copy()



## === cell 50
train[(train["H_Distance"] == 0)].shape



## === cell 51
rush_hour = train.loc[
    (
        ((train["Hour"] >= 6) & (train["Hour"] <= 20))
        & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
        & (train["H_Distance"] == 0)
        & (train["fare_amount"] < 2.5)
    )
]
rush_hour



## === cell 52
train = train.drop(rush_hour.index, axis=0)



## === cell 53
non_rush_hour = train.loc[
    (
        ((train["Hour"] < 6) | (train["Hour"] > 20))
        & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
        & (train["H_Distance"] == 0)
        & (train["fare_amount"] < 3.0)
    )
]
non_rush_hour



## === cell 54
weekends = train.loc[
    ((train["Day of Week"] == 0) | (train["Day of Week"] == 6))
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 3.0)
]
weekends



## === cell 55
train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]



## === cell 56
scenario_3 = train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]



## === cell 57
len(scenario_3)



## === cell 58
scenario_3.sort_values("H_Distance", ascending=False).head(20)



## === cell 59
scenario_3 = scenario_3.copy()
scenario_3["fare_amount"] = (scenario_3["H_Distance"] * 1.56) + 2.50



## === cell 60
scenario_3["fare_amount"].head()



## === cell 61
train.update(scenario_3)



## === cell 62
train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]



## === cell 63
scenario_4 = train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]



## === cell 64
len(scenario_4)



## === cell 65
scenario_4.loc[(scenario_4["fare_amount"] <= 3.0) & (scenario_4["H_Distance"] == 0)]



## === cell 66
scenario_4.loc[(scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)]



## === cell 67
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
]



## === cell 68
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
]



## === cell 69
len(scenario_4_sub)



## === cell 70
scenario_4_sub = scenario_4_sub.copy()
scenario_4_sub["H_Distance"] = (scenario_4_sub["fare_amount"] - 2.50) / 1.56



## === cell 71
train.update(scenario_4_sub)



## === cell 72
train.columns



## === cell 73
test.columns



## === cell 74
train = train.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 75
x_train = train.iloc[:, train.columns != "fare_amount"]
y_train = train["fare_amount"].values
x_test = test



## === cell 76
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_jobs=-1, random_state=42)
rf.fit(x_train, y_train)
rf_predict = rf.predict(x_test)



## === cell 77
submission = pd.read_csv("../input/sample_submission.csv")
submission["fare_amount"] = rf_predict
submission.to_csv("submission_1.csv", index=False)
submission.head(20)



## === cell 78
from sklearn import linear_model

lr = linear_model.LinearRegression(n_jobs=-1)
lr.fit(x_train, y_train)
lr_predict = lr.predict(x_test)



## === cell 79
submission2 = submission.copy()
submission2["fare_amount"] = lr_predict
submission2.to_csv("submission_2.csv", index=False)
submission2.head(20)
