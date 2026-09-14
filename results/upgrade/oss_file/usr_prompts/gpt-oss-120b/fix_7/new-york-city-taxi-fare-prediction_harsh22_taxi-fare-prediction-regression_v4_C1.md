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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import os

try:
    from sklearnex import patch

    patch()
except Exception:
    pass  # fallback to regular scikit‑learn

print(os.listdir("../input"))




## === cell 1
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")




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
train = train.dropna()




## === cell 8
train.shape




## === cell 9
train["fare_amount"].describe()




## === cell 10
from collections import Counter

Counter(train["fare_amount"] < 0)




## === cell 11
train = train.drop(train[train["fare_amount"] < 0].index, axis=0)
train.shape




## === cell 12
train["fare_amount"].describe()




## === cell 13
train["fare_amount"].sort_values(ascending=False)




## === cell 14
train["passenger_count"].describe()




## === cell 15
train[train["passenger_count"] > 8]




## === cell 16
train = train.drop(train[train["passenger_count"] == 208].index, axis=0)




## === cell 17
train["passenger_count"].describe()




## === cell 18
train["pickup_latitude"].describe()




## === cell 19
train[train["pickup_latitude"] < -90]




## === cell 20
train[train["pickup_latitude"] > 90]




## === cell 21
cond_lat = (train["pickup_latitude"] < -90) | (train["pickup_latitude"] > 90)
train = train.drop(train[cond_lat].index, axis=0)




## === cell 22
train.shape




## === cell 23
train["pickup_longitude"].describe()




## === cell 24
train[train["pickup_longitude"] < -180]




## === cell 25
train[train["pickup_longitude"] > 180]




## === cell 26
cond_long = (train["pickup_longitude"] < -180) | (train["pickup_longitude"] > 180)
train = train.drop(train[cond_long].index, axis=0)




## === cell 27
train[train["dropoff_latitude"] < -90]




## === cell 28
train[train["dropoff_latitude"] > 90]




## === cell 29
cond_doff_lat = (train["dropoff_latitude"] < -90) | (train["dropoff_latitude"] > 90)
train = train.drop(train[cond_doff_lat].index, axis=0)




## === cell 30
pass




## === cell 31
train.dtypes




## === cell 32
train["key"] = pd.to_datetime(train["key"], infer_datetime_format=True)
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], infer_datetime_format=True
)




## === cell 33
test["key"] = pd.to_datetime(test["key"], infer_datetime_format=True)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], infer_datetime_format=True
)




## === cell 34
train.dtypes




## === cell 35
def haversine_distance(lat1, long1, lat2, long2):
    data = [train, test]
    for i in data:
        r = 6371
        phi1 = np.radians(i[lat1])
        phi2 = np.radians(i[lat2])
        delta_phi = np.radians(i[lat2] - i[lat1])
        delta_lambda = np.radians(i[long2] - i[long1])
        a = (
            np.sin(delta_phi / 2.0) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        d = r * c  # in kilometers
        i["H_Distance"] = d
    return d




## === cell 36
haversine_distance(
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)




## === cell 37
train["H_Distance"].head(10)




## === cell 38
data = [train, test]
for i in data:
    i["Year"] = i["pickup_datetime"].dt.year
    i["Month"] = i["pickup_datetime"].dt.month
    i["Date"] = i["pickup_datetime"].dt.day
    i["Day of Week"] = i["pickup_datetime"].dt.dayofweek
    i["Hour"] = i["pickup_datetime"].dt.hour




## === cell 39
train.loc[
    ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
    & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
    & (train["fare_amount"] == 0)
]




## === cell 40
train = train.drop(
    train.loc[
        ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
        & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
        & (train["fare_amount"] == 0)
    ].index,
    axis=0,
)




## === cell 41
train.shape




## === cell 42
train.loc[
    ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
    & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
    & (train["fare_amount"] == 0)
]




## === cell 43
train = train.drop(
    train.loc[
        ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
        & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
        & (train["fare_amount"] == 0)
    ].index,
    axis=0,
)




## === cell 44
high_distance = train.loc[(train["H_Distance"] > 200) & (train["fare_amount"] != 0)]




## === cell 45
high_distance




## === cell 46
high_distance["H_Distance"] = high_distance.apply(
    lambda row: (row["fare_amount"] - 2.50) / 1.56, axis=1
)




## === cell 47
train.update(high_distance)




## === cell 48
train[train["H_Distance"] == 0]




## === cell 49
train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)]




## === cell 50
train = train.drop(
    train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)].index, axis=0
)




## === cell 51
train[train["H_Distance"] == 0].shape




## === cell 52
rush_hour = train.loc[
    (
        ((train["Hour"] >= 6) & (train["Hour"] <= 20))
        & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
        & (train["H_Distance"] == 0)
        & (train["fare_amount"] < 2.5)
    )
]
rush_hour




## === cell 53
train = train.drop(rush_hour.index, axis=0)




## === cell 54
non_rush_hour = train.loc[
    (
        ((train["Hour"] < 6) | (train["Hour"] > 20))
        & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
        & (train["H_Distance"] == 0)
        & (train["fare_amount"] < 3.0)
    )
]




## === cell 55
weekends = train.loc[
    ((train["Day of Week"] == 0) | (train["Day of Week"] == 6))
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 3.0)
]




## === cell 56
train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]




## === cell 57
scenario_3 = train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]




## === cell 58
len(scenario_3)




## === cell 59
scenario_3.sort_values("H_Distance", ascending=False)




## === cell 60
scenario_3["fare_amount"] = scenario_3.apply(
    lambda row: ((row["H_Distance"] * 1.56) + 2.50), axis=1
)




## === cell 61
scenario_3["fare_amount"]




## === cell 62
train.update(scenario_3)




## === cell 63
train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]




## === cell 64
scenario_4 = train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]




## === cell 65
len(scenario_4)




## === cell 66
scenario_4.loc[(scenario_4["fare_amount"] <= 3.0) & (scenario_4["H_Distance"] == 0)]




## === cell 67
scenario_4.loc[(scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)]




## === cell 68
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
]




## === cell 69
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
]




## === cell 70
len(scenario_4_sub)




## === cell 71
scenario_4_sub["H_Distance"] = scenario_4_sub.apply(
    lambda row: ((row["fare_amount"] - 2.50) / 1.56), axis=1
)




## === cell 72
train.update(scenario_4_sub)




## === cell 73
train.columns




## === cell 74
test.columns




## === cell 75
train = train.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)




## === cell 76
x_train = train.iloc[:, train.columns != "fare_amount"]
y_train = train["fare_amount"].values
x_test = test




## === cell 77
x_train = x_train.fillna(x_train.median())
x_test = x_test.fillna(x_train.median())

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(x_train, y_train)
rf_predict = np.clip(rf.predict(x_test), 0, None)




## === cell 78
submission = pd.read_csv("../input/sample_submission.csv")
submission["fare_amount"] = rf_predict
submission.to_csv("submission_1.csv", index=False)
submission.head(20)




## === cell 79
from sklearn import linear_model

lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)
lr_predict = np.clip(lr.predict(x_test), 0, None)




## === cell 80
submission = pd.read_csv("../input/sample_submission.csv")
submission["fare_amount"] = lr_predict
submission.to_csv("submission_2.csv", index=False)
submission.head(20)




## === cell 81
final_submission = pd.read_csv("../input/sample_submission.csv")
final_submission["fare_amount"] = rf_predict
final_submission.to_csv("submission.csv", index=False)
final_submission.head(20)
