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

3.39809

# 6. Current score

4.21873

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 16.9774) has done: 'The main runtime issues come from pandas API changes: `any(1)` must be `any(axis=1)`, and your outlier filters accidentally used `|` between DataFrames instead of boolean masks. Fixing those keeps your cleaning logic the same but makes it execute correctly. The RandomForest crash is because NaNs still appear after feature engineering; we explicitly drop any remaining NaNs from train and impute test NaNs with train medians (score-neutral stability fix that preserves the model). Finally, we ensure the submission uses the test `key` values and writes a valid `.csv` with columns `key,fare_amount`.'
- What this solution (achieved 7.24468) has done: 'Your RMSE is far above the target, so we should make a small, legitimate improvement that preserves your overall pipeline. The biggest score hit comes from converting `key` to datetime and then back to string, which changes IDs and can misalign predictions vs required keys; we keep `key` as the original string throughout and only parse `pickup_datetime` for time features. Next, without changing the model type, we tune the existing RandomForest with slightly more trees and sensible constraints (still the same training approach) to reduce variance and improve generalization on this task. Finally, we keep your existing NaN-handling and ensure the submission uses the exact test keys (unchanged) and writes a valid `key,fare_amount` CSV.'
- What this solution (achieved 4.21873) has done: 'Your current score (7.24468 RMSE) is much worse than the target (3.39809), so we should make small, legitimate improvements that preserve your pipeline but reduce error. The biggest “low-risk, high-impact” change here is to add a couple of standard NYC-taxi baseline features (absolute lat/lon deltas and a simple Manhattan-distance proxy) while keeping your RandomForest approach unchanged; these features usually cut RMSE substantially without changing overall logic. We also add the missing outlier filter for invalid dropoff_longitude (you checked it but didn’t drop it), and we make the few `.apply(...)` distance/fare adjustments vectorized to avoid slowdowns and pandas chained-assignment issues while preserving the same semantics. Finally, we keep your exact submission schema and ensure column alignment is identical for train/test.'

# 9. Code solution

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
train.dtypes



## === cell 36
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"], errors="coerce")
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




## === cell 42
haversine_distance(
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)



## === cell 43
train["H_Distance"].head(10)



## === cell 44
test["H_Distance"].head(10)



## === cell 45
train.head(10)



## === cell 46
test.head(10)



## === cell 47
data = [train, test]
for i in data:
    i["Year"] = i["pickup_datetime"].dt.year
    i["Month"] = i["pickup_datetime"].dt.month
    i["Date"] = i["pickup_datetime"].dt.day
    i["Day of Week"] = i["pickup_datetime"].dt.dayofweek
    i["Hour"] = i["pickup_datetime"].dt.hour



## === cell 48
for i in [train, test]:
    i["abs_lon_diff"] = (i["pickup_longitude"] - i["dropoff_longitude"]).abs()
    i["abs_lat_diff"] = (i["pickup_latitude"] - i["dropoff_latitude"]).abs()
    i["manhattan_dist"] = i["abs_lon_diff"] + i["abs_lat_diff"]



## === cell 49
train.head()



## === cell 50
test.head()



## === cell 51
plt.figure(figsize=(15, 7))
plt.hist(train["passenger_count"], bins=15)
plt.xlabel("No. of Passengers")
plt.ylabel("Frequency")



## === cell 52
plt.figure(figsize=(15, 7))
plt.scatter(x=train["passenger_count"], y=train["fare_amount"], s=1.5)
plt.xlabel("No. of Passengers")
plt.ylabel("Fare")



## === cell 53
plt.figure(figsize=(15, 7))
plt.scatter(x=train["Date"], y=train["fare_amount"], s=1.5)
plt.xlabel("Date")
plt.ylabel("Fare")



## === cell 54
plt.figure(figsize=(15, 7))
plt.hist(train["Hour"], bins=100)
plt.xlabel("Hour")
plt.ylabel("Frequency")



## === cell 55
plt.figure(figsize=(15, 7))
plt.scatter(x=train["Hour"], y=train["fare_amount"], s=1.5)
plt.xlabel("Hour")
plt.ylabel("Fare")



## === cell 56
plt.figure(figsize=(15, 7))
plt.hist(train["Day of Week"], bins=100)
plt.xlabel("Day of Week")
plt.ylabel("Frequency")



## === cell 57
plt.figure(figsize=(15, 7))
plt.scatter(x=train["Day of Week"], y=train["fare_amount"], s=1.5)
plt.xlabel("Day of Week")
plt.ylabel("Fare")



## === cell 58
train.sort_values(["H_Distance", "fare_amount"], ascending=False)



## === cell 59
len(train)



## === cell 60
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



## === cell 61
plt.figure(figsize=(15, 7))
plt.hist(dist_bins["bins"], bins=75)
plt.xlabel("Bins")
plt.ylabel("Frequency")



## === cell 62
Counter(dist_bins["bins"])



## === cell 63
train.loc[
    ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
    & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
    & (train["fare_amount"] == 0)
]



## === cell 64
train = train.drop(
    train.loc[
        ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
        & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
        & (train["fare_amount"] == 0)
    ].index,
    axis=0,
)



## === cell 65
train.shape



## === cell 66
test.loc[
    ((test["pickup_latitude"] == 0) & (test["pickup_longitude"] == 0))
    & ((test["dropoff_latitude"] != 0) & (test["dropoff_longitude"] != 0))
]



## === cell 67
train.loc[
    ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
    & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
    & (train["fare_amount"] == 0)
]



## === cell 68
train = train.drop(
    train.loc[
        ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
        & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
        & (train["fare_amount"] == 0)
    ].index,
    axis=0,
)



## === cell 69
train.shape



## === cell 70
test.loc[
    ((test["pickup_latitude"] != 0) & (test["pickup_longitude"] != 0))
    & ((test["dropoff_latitude"] == 0) & (test["dropoff_longitude"] == 0))
]



## === cell 71
high_distance = train.loc[(train["H_Distance"] > 200) & (train["fare_amount"] != 0)]



## === cell 72
high_distance



## === cell 73
high_distance.shape



## === cell 74
hd_idx = high_distance.index
train.loc[hd_idx, "H_Distance"] = (train.loc[hd_idx, "fare_amount"] - 2.50) / 1.56



## === cell 75
high_distance



## === cell 76
train.update(high_distance)



## === cell 77
train.shape



## === cell 78
train[train["H_Distance"] == 0]



## === cell 79
train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)]



## === cell 80
train = train.drop(
    train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)].index, axis=0
)



## === cell 81
train[(train["H_Distance"] == 0)].shape



## === cell 82
rush_hour = train.loc[
    (
        ((train["Hour"] >= 6) & (train["Hour"] <= 20))
        & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
        & (train["H_Distance"] == 0)
        & (train["fare_amount"] < 2.5)
    )
]
rush_hour



## === cell 83
train = train.drop(rush_hour.index, axis=0)



## === cell 84
train.shape



## === cell 85
non_rush_hour = train.loc[
    (
        ((train["Hour"] < 6) | (train["Hour"] > 20))
        & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
        & (train["H_Distance"] == 0)
        & (train["fare_amount"] < 3.0)
    )
]
non_rush_hour



## === cell 86
weekends = train.loc[
    ((train["Day of Week"] == 0) | (train["Day of Week"] == 6))
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 3.0)
]
weekends



## === cell 87
train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]



## === cell 88
scenario_3 = train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]



## === cell 89
len(scenario_3)



## === cell 90
scenario_3.sort_values("H_Distance", ascending=False)



## === cell 91
s3_idx = scenario_3.index
train.loc[s3_idx, "fare_amount"] = (train.loc[s3_idx, "H_Distance"] * 1.56) + 2.50



## === cell 92
scenario_3["fare_amount"]



## === cell 93
train.update(scenario_3)



## === cell 94
train.shape



## === cell 95
train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]



## === cell 96
scenario_4 = train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]



## === cell 97
len(scenario_4)



## === cell 98
scenario_4.loc[(scenario_4["fare_amount"] <= 3.0) & (scenario_4["H_Distance"] == 0)]



## === cell 99
scenario_4.loc[(scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)]



## === cell 100
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
]



## === cell 101
len(scenario_4_sub)



## === cell 102
s4_idx = scenario_4_sub.index
train.loc[s4_idx, "H_Distance"] = (train.loc[s4_idx, "fare_amount"] - 2.50) / 1.56



## === cell 103
train.update(scenario_4_sub)



## === cell 104
train.shape



## === cell 105
train.columns



## === cell 106
test.columns



## === cell 107
test_key = test["key"].copy()

train = train.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 108
train.columns



## === cell 109
test.columns



## === cell 110
x_train = train.iloc[:, train.columns != "fare_amount"]
y_train = train.iloc[:, train.columns == "fare_amount"]
x_test = test



## === cell 111
x_train.shape



## === cell 112
x_train.columns



## === cell 113
y_train.shape



## === cell 114
x_test.shape



## === cell 115
x_test.columns



## === cell 116
train_mask = ~x_train.isnull().any(axis=1) & ~y_train.isnull().any(axis=1)
x_train = x_train.loc[train_mask].copy()
y_train = y_train.loc[train_mask, "fare_amount"].copy()

train_medians = x_train.median(numeric_only=True)
x_train = x_train.fillna(train_medians)
x_test = x_test.fillna(train_medians)

x_train.isnull().sum().sum(), x_test.isnull().sum().sum()



## === cell 117
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



## === cell 118
rf_predict = np.asarray(rf_predict, dtype=float)
rf_predict = np.where(np.isfinite(rf_predict), rf_predict, np.nan)
fallback = float(y_train.mean())
rf_predict = np.where(np.isnan(rf_predict), fallback, rf_predict)
rf_predict = np.maximum(rf_predict, 0.0)



## === cell 119
submission = pd.read_csv("../input/sample_submission.csv")
submission["key"] = test_key.astype(str).values
submission["fare_amount"] = rf_predict
submission.to_csv("submission_1.csv", index=False)
submission.head(20)
