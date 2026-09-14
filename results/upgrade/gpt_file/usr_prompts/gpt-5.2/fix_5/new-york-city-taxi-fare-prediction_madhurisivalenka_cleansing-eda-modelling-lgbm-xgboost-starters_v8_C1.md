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

print(os.listdir("../input"))



## === cell 1
train = pd.read_csv("../input/train.csv", nrows=1000000)
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
train = train.drop(train[train.isnull().any(axis=1)].index, axis=0)



## === cell 9
train.shape



## === cell 10
train["fare_amount"].describe()



## === cell 11
from collections import Counter

Counter(train["fare_amount"] < 0)



## === cell 12
train = train.drop(train[train["fare_amount"] < 0].index, axis=0)
train.shape



## === cell 13
train["fare_amount"].describe()



## === cell 14
train["fare_amount"].sort_values(ascending=False)



## === cell 15
train["passenger_count"].describe()



## === cell 16
train[train["passenger_count"] > 6]



## === cell 17
train = train.drop(train[train["passenger_count"] == 208].index, axis=0)



## === cell 18
train["passenger_count"].describe()



## === cell 19
train["pickup_latitude"].describe()



## === cell 20
train[train["pickup_latitude"] < -90]



## === cell 21
train[train["pickup_latitude"] > 90]



## === cell 22
mask = (train["pickup_latitude"] < -90) | (train["pickup_latitude"] > 90)
train = train.drop(train[mask].index, axis=0)



## === cell 23
train.shape



## === cell 24
train["pickup_longitude"].describe()



## === cell 25
train[train["pickup_longitude"] < -180]



## === cell 26
train[train["pickup_longitude"] > 180]



## === cell 27
mask = (train["pickup_longitude"] < -180) | (train["pickup_longitude"] > 180)
train = train.drop(train[mask].index, axis=0)



## === cell 28
train.shape



## === cell 29
train[train["dropoff_latitude"] < -90]



## === cell 30
train[train["dropoff_latitude"] > 90]



## === cell 31
mask = (train["dropoff_latitude"] < -90) | (train["dropoff_latitude"] > 90)
train = train.drop(train[mask].index, axis=0)



## === cell 32
train.shape



## === cell 33
train[(train["dropoff_longitude"] < -180) | (train["dropoff_longitude"] > 180)].head()



## === cell 34
mask = (train["dropoff_longitude"] < -180) | (train["dropoff_longitude"] > 180)
train = train.drop(train[mask].index, axis=0)



## === cell 35
nyc_bbox = {
    "lon_min": -74.3,
    "lon_max": -73.7,
    "lat_min": 40.5,
    "lat_max": 41.0,
}
train = train.loc[
    (train["pickup_longitude"].between(nyc_bbox["lon_min"], nyc_bbox["lon_max"]))
    & (train["dropoff_longitude"].between(nyc_bbox["lon_min"], nyc_bbox["lon_max"]))
    & (train["pickup_latitude"].between(nyc_bbox["lat_min"], nyc_bbox["lat_max"]))
    & (train["dropoff_latitude"].between(nyc_bbox["lat_min"], nyc_bbox["lat_max"]))
].copy()

train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] <= 250)].copy()

train.shape



## === cell 36
train.dtypes



## === cell 37
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"], errors="coerce")
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"], errors="coerce")



## === cell 38
train.dtypes



## === cell 39
test.dtypes



## === cell 40
train.head()



## === cell 41
test.head()




## === cell 42
def haversine_distance(lat1, long1, lat2, long2):
    data = [train, test]
    for i in data:
        R = 6371  # radius of earth in kilometers
        phi1 = np.radians(i[lat1])
        phi2 = np.radians(i[lat2])

        delta_phi = np.radians(i[lat2] - i[lat1])
        delta_lambda = np.radians(i[long2] - i[long1])

        a = (
            np.sin(delta_phi / 2.0) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
        )

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        d = R * c  # in kilometers
        i["H_Distance"] = d
    return d




## === cell 43
haversine_distance(
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)



## === cell 44
train["H_Distance"].head(10)



## === cell 45
test["H_Distance"].head(10)



## === cell 46
train.head(10)



## === cell 47
test.head(10)



## === cell 48
data = [train, test]
for i in data:
    i["Year"] = i["pickup_datetime"].dt.year
    i["Month"] = i["pickup_datetime"].dt.month
    i["Date"] = i["pickup_datetime"].dt.day
    i["Day of Week"] = i["pickup_datetime"].dt.dayofweek
    i["Hour"] = i["pickup_datetime"].dt.hour



## === cell 49
for i in [train, test]:
    i["abs_lon_diff"] = (i["pickup_longitude"] - i["dropoff_longitude"]).abs()
    i["abs_lat_diff"] = (i["pickup_latitude"] - i["dropoff_latitude"]).abs()
    i["manhattan_dist"] = i["abs_lon_diff"] + i["abs_lat_diff"]



## === cell 50
for i in [train, test]:
    dlon = (i["dropoff_longitude"] - i["pickup_longitude"]).astype(float)
    dlat = (i["dropoff_latitude"] - i["pickup_latitude"]).astype(float)
    i["euclid_dist"] = np.sqrt(dlon * dlon + dlat * dlat)
    i["dir_ratio"] = dlat / (dlon.abs() + 1e-6)



## === cell 51
train.head()



## === cell 52
test.head()



## === cell 53
plt.figure(figsize=(15, 7))
plt.hist(train["passenger_count"], bins=15)
plt.xlabel("No. of Passengers")
plt.ylabel("Frequency")



## === cell 54
plt.figure(figsize=(15, 7))
plt.scatter(x=train["passenger_count"], y=train["fare_amount"], s=1.5)
plt.xlabel("No. of Passengers")
plt.ylabel("Fare")



## === cell 55
plt.figure(figsize=(15, 7))
plt.scatter(x=train["Date"], y=train["fare_amount"], s=1.5)
plt.xlabel("Date")
plt.ylabel("Fare")



## === cell 56
plt.figure(figsize=(15, 7))
plt.hist(train["Hour"], bins=100)
plt.xlabel("Hour")
plt.ylabel("Frequency")



## === cell 57
plt.figure(figsize=(15, 7))
plt.scatter(x=train["Hour"], y=train["fare_amount"], s=1.5)
plt.xlabel("Hour")
plt.ylabel("Fare")



## === cell 58
plt.figure(figsize=(15, 7))
plt.hist(train["Day of Week"], bins=100)
plt.xlabel("Day of Week")
plt.ylabel("Frequency")



## === cell 59
plt.figure(figsize=(15, 7))
plt.scatter(x=train["Day of Week"], y=train["fare_amount"], s=1.5)
plt.xlabel("Day of Week")
plt.ylabel("Fare")



## === cell 60
train.sort_values(["H_Distance", "fare_amount"], ascending=False)



## === cell 61
len(train)



## === cell 62
bins_0 = train.loc[(train["H_Distance"] == 0), ["H_Distance"]]
bins_1 = train.loc[
    (train["H_Distance"] > 0) & (train["H_Distance"] <= 10), ["H_Distance"]
]
bins_2 = train.loc[
    (train["H_Distance"] > 10) & (train["H_Distance"] <= 50), ["H_Distance"]
]
bins_3 = train.loc[
    (train["H_Distance"] > 50) & (train["H_Distance"] <= 100), ["H_Distance"]
]
bins_4 = train.loc[
    (train["H_Distance"] > 100) & (train["H_Distance"] <= 200), ["H_Distance"]
]
bins_5 = train.loc[
    (train["H_Distance"] > 200) & (train["H_Distance"] <= 300), ["H_Distance"]
]
bins_6 = train.loc[(train["H_Distance"] > 300), ["H_Distance"]]
bins_0["bins"] = "0"
bins_1["bins"] = "0-10"
bins_2["bins"] = "11-50"
bins_3["bins"] = "51-100"
bins_4["bins"] = "100-200"
bins_5["bins"] = "201-300"
bins_6["bins"] = ">300"
dist_bins = pd.concat([bins_0, bins_1, bins_2, bins_3, bins_4, bins_5, bins_6])
dist_bins.columns



## === cell 63
plt.figure(figsize=(15, 7))
plt.hist(dist_bins["bins"], bins=75)
plt.xlabel("Bins")
plt.ylabel("Frequency")



## === cell 64
Counter(dist_bins["bins"])



## === cell 65
train.loc[
    ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
    & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
    & (train["fare_amount"] == 0)
]



## === cell 66
train = train.drop(
    train.loc[
        ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
        & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
        & (train["fare_amount"] == 0)
    ].index,
    axis=0,
)



## === cell 67
train.shape



## === cell 68
test.loc[
    ((test["pickup_latitude"] == 0) & (test["pickup_longitude"] == 0))
    & ((test["dropoff_latitude"] != 0) & (test["dropoff_longitude"] != 0))
]



## === cell 69
train.loc[
    ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
    & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
    & (train["fare_amount"] == 0)
]



## === cell 70
train = train.drop(
    train.loc[
        ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
        & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
        & (train["fare_amount"] == 0)
    ].index,
    axis=0,
)



## === cell 71
train.shape



## === cell 72
test.loc[
    ((test["pickup_latitude"] != 0) & (test["pickup_longitude"] != 0))
    & ((test["dropoff_latitude"] == 0) & (test["dropoff_longitude"] == 0))
]



## === cell 73
high_distance = train.loc[(train["H_Distance"] > 200) & (train["fare_amount"] != 0)]



## === cell 74
high_distance



## === cell 75
high_distance.shape



## === cell 76
hd_idx = high_distance.index
train.loc[hd_idx, "H_Distance"] = (train.loc[hd_idx, "fare_amount"] - 2.50) / 1.56



## === cell 77
high_distance



## === cell 78
train.update(high_distance)



## === cell 79
train.shape



## === cell 80
train[train["H_Distance"] == 0]



## === cell 81
train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)]



## === cell 82
train = train.drop(
    train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)].index, axis=0
)



## === cell 83
train[(train["H_Distance"] == 0)].shape



## === cell 84
rush_hour = train.loc[
    (
        ((train["Hour"] >= 6) & (train["Hour"] <= 20))
        & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
        & (train["H_Distance"] == 0)
        & (train["fare_amount"] < 2.5)
    )
]
rush_hour



## === cell 85
train = train.drop(rush_hour.index, axis=0)



## === cell 86
train.shape



## === cell 87
non_rush_hour = train.loc[
    (
        ((train["Hour"] < 6) | (train["Hour"] > 20))
        & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
        & (train["H_Distance"] == 0)
        & (train["fare_amount"] < 3.0)
    )
]
non_rush_hour



## === cell 88
weekends = train.loc[
    ((train["Day of Week"] == 0) | (train["Day of Week"] == 6))
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 3.0)
]
weekends



## === cell 89
train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]



## === cell 90
scenario_3 = train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]



## === cell 91
len(scenario_3)



## === cell 92
scenario_3.sort_values("H_Distance", ascending=False)



## === cell 93
s3_idx = scenario_3.index
train.loc[s3_idx, "fare_amount"] = (train.loc[s3_idx, "H_Distance"] * 1.56) + 2.50



## === cell 94
scenario_3["fare_amount"]



## === cell 95
train.update(scenario_3)



## === cell 96
train.shape



## === cell 97
train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]



## === cell 98
scenario_4 = train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]



## === cell 99
len(scenario_4)



## === cell 100
scenario_4.loc[(scenario_4["fare_amount"] <= 3.0) & (scenario_4["H_Distance"] == 0)]



## === cell 101
scenario_4.loc[(scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)]



## === cell 102
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
]



## === cell 103
len(scenario_4_sub)



## === cell 104
s4_idx = scenario_4_sub.index
train.loc[s4_idx, "H_Distance"] = (train.loc[s4_idx, "fare_amount"] - 2.50) / 1.56



## === cell 105
train.update(scenario_4_sub)



## === cell 106
train.shape



## === cell 107
train.columns



## === cell 108
test.columns



## === cell 109
test_key = test["key"].copy()

train = train.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 110
train.columns



## === cell 111
test.columns



## === cell 112
x_train = train.iloc[:, train.columns != "fare_amount"]
y_train = train.iloc[:, train.columns == "fare_amount"]
x_test = test



## === cell 113
x_train.shape



## === cell 114
x_train.columns



## === cell 115
y_train.shape



## === cell 116
x_test.shape



## === cell 117
x_test.columns



## === cell 118
train_mask = ~x_train.isnull().any(axis=1) & ~y_train.isnull().any(axis=1)
x_train = x_train.loc[train_mask].copy()
y_train = y_train.loc[train_mask, "fare_amount"].copy()

train_medians = x_train.median(numeric_only=True)
x_train = x_train.fillna(train_medians)
x_test = x_test.fillna(train_medians)

x_train.isnull().sum().sum(), x_test.isnull().sum().sum()



## === cell 119
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=1,
    min_samples_split=2,
    max_features="sqrt",
)
rf.fit(x_train, y_train)
rf_predict = rf.predict(x_test)



## === cell 120
rf_predict = np.asarray(rf_predict, dtype=float)
rf_predict = np.where(np.isfinite(rf_predict), rf_predict, np.nan)
fallback = float(y_train.mean())
rf_predict = np.where(np.isnan(rf_predict), fallback, rf_predict)
rf_predict = np.maximum(rf_predict, 0.0)



## === cell 121
submission = pd.read_csv("../input/sample_submission.csv")
submission["key"] = test_key.astype(str).values
submission["fare_amount"] = rf_predict
submission.to_csv("submission_1.csv", index=False)
submission.head(20)
