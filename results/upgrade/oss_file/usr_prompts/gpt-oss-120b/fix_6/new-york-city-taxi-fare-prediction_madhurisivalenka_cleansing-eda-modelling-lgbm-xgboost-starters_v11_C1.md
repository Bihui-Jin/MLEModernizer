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
lightgbm==4.6.0
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
import seaborn as sns
import matplotlib.pyplot as plt
import os

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
test.isnull().sum().sort_values(ascending=False)




## === cell 8
train = train.dropna(axis=0, how="any")




## === cell 9
train.shape




## === cell 10
train["fare_amount"].describe()




## === cell 11
from collections import Counter

Counter(train["fare_amount"] < 0)




## === cell 12
train = train[train["fare_amount"] >= 0]
train.shape




## === cell 13
train["fare_amount"].describe()




## === cell 14
train["fare_amount"].sort_values(ascending=False).head()




## === cell 15
train["passenger_count"].describe()




## === cell 16
train[train["passenger_count"] > 6].head()




## === cell 17
train = train[train["passenger_count"] != 208]




## === cell 18
train["passenger_count"].describe()




## === cell 19
train["pickup_latitude"].describe()




## === cell 20
train[train["pickup_latitude"] < -90].shape




## === cell 21
train[train["pickup_latitude"] > 90].shape




## === cell 22
lat_mask = (train["pickup_latitude"] < -90) | (train["pickup_latitude"] > 90)
train = train[~lat_mask]




## === cell 23
train.shape




## === cell 24
train["pickup_longitude"].describe()




## === cell 25
train[train["pickup_longitude"] < -180].shape




## === cell 26
train[train["pickup_longitude"] > 180].shape




## === cell 27
lon_mask = (train["pickup_longitude"] < -180) | (train["pickup_longitude"] > 180)
train = train[~lon_mask]




## === cell 28
train.shape




## === cell 29
train[train["dropoff_latitude"] < -90].shape




## === cell 30
train[train["dropoff_latitude"] > 90].shape




## === cell 31
drop_lat_mask = (train["dropoff_latitude"] < -90) | (train["dropoff_latitude"] > 90)
train = train[~drop_lat_mask]




## === cell 32
train.shape




## === cell 33
(train["pickup_latitude"] < -90).any(), (train["pickup_latitude"] > 90).any()




## === cell 34
train.dtypes




## === cell 35
train["key"] = pd.to_datetime(train["key"], errors="coerce")
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"], errors="coerce")




## === cell 36
test["key"] = pd.to_datetime(test["key"], errors="coerce")
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"], errors="coerce")




## === cell 37
train.dtypes




## === cell 38
test.dtypes




## === cell 39
train.head()




## === cell 40
test.head()




## === cell 41
def haversine_distance(df):
    """
    Vectorised haversine distance (km) between pickup and drop‑off points.
    """
    R = 6371.0  # Earth radius in km
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    dlat = np.radians(df["dropoff_latitude"] - df["pickup_latitude"])
    dlon = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c




## === cell 42
train["H_Distance"] = haversine_distance(train)
test["H_Distance"] = haversine_distance(test)




## === cell 43
train["H_Distance"].head(10)




## === cell 44
test["H_Distance"].head(10)




## === cell 45
train.head(10)




## === cell 46
test.head(10)




## === cell 47
for df in (train, test):
    df["Year"] = df["pickup_datetime"].dt.year
    df["Month"] = df["pickup_datetime"].dt.month
    df["Date"] = df["pickup_datetime"].dt.day
    df["DayOfWeek"] = df["pickup_datetime"].dt.dayofweek
    df["Hour"] = df["pickup_datetime"].dt.hour




## === cell 48
train.head()




## === cell 49
test.head()




## === cell 50
plt.figure(figsize=(15, 7))
plt.hist(train["passenger_count"], bins=15)
plt.xlabel("No. of Passengers")
plt.ylabel("Frequency")




## === cell 51
plt.figure(figsize=(15, 7))
plt.scatter(x=train["passenger_count"], y=train["fare_amount"], s=1.5)
plt.xlabel("No. of Passengers")
plt.ylabel("Fare")




## === cell 52
plt.figure(figsize=(15, 7))
plt.scatter(x=train["Date"], y=train["fare_amount"], s=1.5)
plt.xlabel("Date")
plt.ylabel("Fare")




## === cell 53
plt.figure(figsize=(15, 7))
plt.hist(train["Hour"], bins=100)
plt.xlabel("Hour")
plt.ylabel("Frequency")




## === cell 54
plt.figure(figsize=(15, 7))
plt.scatter(x=train["Hour"], y=train["fare_amount"], s=1.5)
plt.xlabel("Hour")
plt.ylabel("Fare")




## === cell 55
plt.figure(figsize=(15, 7))
plt.hist(train["DayOfWeek"], bins=7)
plt.xlabel("Day of Week")
plt.ylabel("Frequency")




## === cell 56
plt.figure(figsize=(15, 7))
plt.scatter(x=train["DayOfWeek"], y=train["fare_amount"], s=1.5)
plt.xlabel("Day of Week")
plt.ylabel("Fare")




## === cell 57
train.sort_values(["H_Distance", "fare_amount"], ascending=False).head()




## === cell 58
len(train)




## === cell 59
bins_0 = train.loc[train["H_Distance"] == 0, "H_Distance"]
bins_1 = train.loc[
    (train["H_Distance"] > 0) & (train["H_Distance"] <= 10), "H_Distance"
]
bins_2 = train.loc[
    (train["H_Distance"] > 10) & (train["H_Distance"] <= 50), "H_Distance"
]
bins_3 = train.loc[
    (train["H_Distance"] > 50) & (train["H_Distance"] <= 100), "H_Distance"
]
bins_4 = train.loc[
    (train["H_Distance"] > 100) & (train["H_Distance"] <= 200), "H_Distance"
]
bins_5 = train.loc[
    (train["H_Distance"] > 200) & (train["H_Distance"] <= 300), "H_Distance"
]
bins_6 = train.loc[train["H_Distance"] > 300, "H_Distance"]
dist_bins = pd.concat([bins_0, bins_1, bins_2, bins_3, bins_4, bins_5, bins_6])
dist_bins = dist_bins.to_frame(name="H_Distance")
dist_bins["bins"] = pd.cut(
    dist_bins["H_Distance"],
    bins=[-1, 0, 10, 50, 100, 200, 300, np.inf],
    labels=["0", "0-10", "11-50", "51-100", "100-200", "201-300", ">300"],
)




## === cell 60
plt.figure(figsize=(15, 7))
dist_bins["bins"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("Distance Bins")
plt.ylabel("Frequency")




## === cell 61
from collections import Counter

Counter(dist_bins["bins"])




## === cell 62
mask = (
    (train["pickup_latitude"] == 0)
    & (train["pickup_longitude"] == 0)
    & (train["dropoff_latitude"] != 0)
    & (train["dropoff_longitude"] != 0)
    & (train["fare_amount"] == 0)
)
train = train[~mask]




## === cell 63
train.shape




## === cell 64
mask = (
    (train["dropoff_latitude"] == 0)
    & (train["dropoff_longitude"] == 0)
    & (train["pickup_latitude"] != 0)
    & (train["pickup_longitude"] != 0)
    & (train["fare_amount"] == 0)
)
train = train[~mask]




## === cell 65
train.shape




## === cell 66
train = train.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)




## === cell 67
train.columns




## === cell 68
test.columns




## === cell 69
X_train = train.drop("fare_amount", axis=1)
y_train = train["fare_amount"]
X_test = test.copy()




## === cell 70
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())




## === cell 71
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

import lightgbm as lgbm

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42,
}

train_set = lgbm.Dataset(X_tr, label=y_tr)
valid_set = lgbm.Dataset(X_val, label=y_val, reference=train_set)

callbacks = [lgbm.early_stopping(stopping_rounds=50, verbose=False)]

model = lgbm.train(
    params,
    train_set,
    num_boost_round=500,
    valid_sets=[valid_set],
    callbacks=callbacks,
)




## === cell 72
pred_test = model.predict(X_test)




## === cell 73
submission = pd.read_csv("../input/sample_submission.csv")
submission["fare_amount"] = pred_test
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
submission.head(10)
