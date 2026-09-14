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

16.77318

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 16.9774) has done: 'The timeout is driven by expensive Pandas operations repeated many times on a 1M-row frame (especially `drop(...index...)`, multiple boolean-filter materializations, and several `.apply(axis=1)` row-wise calls), plus heavy plotting/sorting that is irrelevant to producing the submission. I keep the exact same cleaning rules, feature engineering, and RandomForest training/prediction, but make them faster by (1) consolidating row drops into boolean masks (single pass), (2) replacing `.apply(axis=1)` with vectorized arithmetic on the same subsets, (3) avoiding `train.update(...)` in favor of direct `.loc[...]` assignment, and (4) disabling all plotting/expensive exploratory outputs so runtime is spent only on the pipeline needed for the final predictions.'
- What this solution (achieved 16.77318) has done: 'You’re currently crippling model performance by coercing the `key` (a string id) into `datetime`, which produces many `NaT` values and can silently misalign/contaminate later steps; keeping `key` as the original string also ensures the submission rows match the test set exactly. The other major issue is that you never filter invalid longitude ranges for `dropoff_longitude` (cell 32 mistakenly checks `dropoff_latitude` against [-180, 180]); fixing that restores a large amount of correct training signal without changing the model/feature set. Finally, to move RMSE closer to your target without changing the model, we clip negative predictions to 0 (fares can’t be negative), which typically improves RMSE on this competition. All changes are minimal and keep your RandomForest training and feature engineering intact.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

print(os.listdir("../input"))

np.random.seed(42)



## === cell 1
train = pd.read_csv(
    "../input/train.csv",
    nrows=1000000,
    parse_dates=["pickup_datetime"],
    dtype={
        "fare_amount": "float64",
        "pickup_longitude": "float64",
        "pickup_latitude": "float64",
        "dropoff_longitude": "float64",
        "dropoff_latitude": "float64",
        "passenger_count": "int64",
    },
)
test = pd.read_csv(
    "../input/test.csv",
    parse_dates=["pickup_datetime"],
    dtype={
        "pickup_longitude": "float64",
        "pickup_latitude": "float64",
        "dropoff_longitude": "float64",
        "dropoff_latitude": "float64",
        "passenger_count": "int64",
    },
)



## === cell 2
train.shape



## === cell 3
test.shape



## === cell 4
train.head(10)



## === cell 5
_ = train.isnull().sum()



## === cell 6
_ = test.isnull().sum()



## === cell 7
train = train.dropna(axis=0, how="any")



## === cell 8
train.shape



## === cell 9
train["fare_amount"].describe()



## === cell 10
from collections import Counter

Counter(train["fare_amount"] < 0)



## === cell 11
train = train.loc[train["fare_amount"] >= 0].copy()
train.shape



## === cell 12
train["fare_amount"].describe()



## === cell 13
train["fare_amount"].nlargest(10)



## === cell 14
train["passenger_count"].describe()



## === cell 15
train[train["passenger_count"] > 6].head()



## === cell 16
train = train.loc[train["passenger_count"] != 208].copy()



## === cell 17
train["passenger_count"].describe()



## === cell 18
train["pickup_latitude"].describe()



## === cell 19
train[train["pickup_latitude"] < -90].head()



## === cell 20
train[train["pickup_latitude"] > 90].head()



## === cell 21
train = train.loc[
    (train["pickup_latitude"] >= -90) & (train["pickup_latitude"] <= 90)
].copy()



## === cell 22
train.shape



## === cell 23
train["pickup_longitude"].describe()



## === cell 24
train[train["pickup_longitude"] < -180].head()



## === cell 25
train[train["pickup_longitude"] > 180].head()



## === cell 26
train = train.loc[
    (train["pickup_longitude"] >= -180) & (train["pickup_longitude"] <= 180)
].copy()



## === cell 27
train.shape



## === cell 28
train[train["dropoff_latitude"] < -90].head()



## === cell 29
train[train["dropoff_latitude"] > 90].head()



## === cell 30
train = train.loc[
    (train["dropoff_latitude"] >= -90) & (train["dropoff_latitude"] <= 90)
].copy()



## === cell 31
train.shape



## === cell 32
train[(train["dropoff_longitude"] < -180) | (train["dropoff_longitude"] > 180)].head()



## === cell 33
train = train.loc[
    (train["dropoff_longitude"] >= -180) & (train["dropoff_longitude"] <= 180)
].copy()



## === cell 34
train.dtypes



## === cell 35
train.dtypes



## === cell 36
test.dtypes



## === cell 37
train.dtypes



## === cell 38
test.dtypes



## === cell 39
train.head()



## === cell 40
test.head()




## === cell 41
def _add_haversine(df, lat1, lon1, lat2, lon2, out_col="H_Distance"):
    R = 6371.0
    phi1 = np.radians(df[lat1].to_numpy())
    phi2 = np.radians(df[lat2].to_numpy())
    dphi = np.radians((df[lat2] - df[lat1]).to_numpy())
    dlmb = np.radians((df[lon2] - df[lon1]).to_numpy())
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    df[out_col] = R * c


def haversine_distance(lat1, long1, lat2, long2):
    _add_haversine(train, lat1, long1, lat2, long2, out_col="H_Distance")
    _add_haversine(test, lat1, long1, lat2, long2, out_col="H_Distance")
    return train["H_Distance"]




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
for df in (train, test):
    dt = df["pickup_datetime"].dt
    df["Year"] = dt.year
    df["Month"] = dt.month
    df["Date"] = dt.day
    df["Day of Week"] = dt.dayofweek
    df["Hour"] = dt.hour



## === cell 48
train.head()



## === cell 49
test.head()



## === cell 50
pass



## === cell 51
pass



## === cell 52
pass



## === cell 53
pass



## === cell 54
pass



## === cell 55
pass



## === cell 56
pass



## === cell 57
train[["H_Distance", "fare_amount"]].nlargest(5, "H_Distance")



## === cell 58
len(train)



## === cell 59
pass



## === cell 60
pass



## === cell 61
pass



## === cell 62
train.loc[
    ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
    & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
    & (train["fare_amount"] == 0)
].head()



## === cell 63
mask_bad1 = (
    (train["pickup_latitude"].eq(0) & train["pickup_longitude"].eq(0))
    & (~train["dropoff_latitude"].eq(0) & ~train["dropoff_longitude"].eq(0))
    & train["fare_amount"].eq(0)
)
train = train.loc[~mask_bad1].copy()



## === cell 64
train.shape



## === cell 65
test.loc[
    ((test["pickup_latitude"] == 0) & (test["pickup_longitude"] == 0))
    & ((test["dropoff_latitude"] != 0) & (test["dropoff_longitude"] != 0))
].head()



## === cell 66
train.loc[
    ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
    & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
    & (train["fare_amount"] == 0)
].head()



## === cell 67
mask_bad2 = (
    (~train["pickup_latitude"].eq(0) & ~train["pickup_longitude"].eq(0))
    & (train["dropoff_latitude"].eq(0) & train["dropoff_longitude"].eq(0))
    & train["fare_amount"].eq(0)
)
train = train.loc[~mask_bad2].copy()



## === cell 68
train.shape



## === cell 69
test.loc[
    ((test["pickup_latitude"] != 0) & (test["pickup_longitude"] != 0))
    & ((test["dropoff_latitude"] == 0) & (test["dropoff_longitude"] == 0))
].head()



## === cell 70
high_distance = train.loc[
    (train["H_Distance"] > 200) & (train["fare_amount"] != 0)
].copy()



## === cell 71
high_distance.head()



## === cell 72
high_distance.shape



## === cell 73
high_distance.loc[:, "H_Distance"] = (high_distance["fare_amount"] - 2.50) / 1.56



## === cell 74
high_distance.head()



## === cell 75
train.loc[high_distance.index, "H_Distance"] = high_distance["H_Distance"]



## === cell 76
train.shape



## === cell 77
train[train["H_Distance"] == 0].head()



## === cell 78
train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)].head()



## === cell 79
train = train.loc[~((train["H_Distance"] == 0) & (train["fare_amount"] == 0))].copy()



## === cell 80
train[(train["H_Distance"] == 0)].shape



## === cell 81
rush_hour = train.loc[
    ((train["Hour"] >= 6) & (train["Hour"] <= 20))
    & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 2.5)
]
rush_hour.head()



## === cell 82
train = train.drop(rush_hour.index, axis=0)



## === cell 83
train.shape



## === cell 84
non_rush_hour = train.loc[
    (((train["Hour"] < 6) | (train["Hour"] > 20)))
    & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 3.0)
]
non_rush_hour.head()



## === cell 85
weekends = train.loc[
    ((train["Day of Week"] == 0) | (train["Day of Week"] == 6))
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 3.0)
]
weekends.head()



## === cell 86
train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)].head()



## === cell 87
scenario_3 = train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)].copy()



## === cell 88
len(scenario_3)



## === cell 89
scenario_3.nlargest(5, "H_Distance")



## === cell 90
scenario_3.loc[:, "fare_amount"] = (scenario_3["H_Distance"] * 1.56) + 2.50



## === cell 91
scenario_3["fare_amount"].head()



## === cell 92
train.loc[scenario_3.index, "fare_amount"] = scenario_3["fare_amount"]



## === cell 93
train.shape



## === cell 94
train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)].head()



## === cell 95
scenario_4 = train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)].copy()



## === cell 96
len(scenario_4)



## === cell 97
scenario_4.loc[
    (scenario_4["fare_amount"] <= 3.0) & (scenario_4["H_Distance"] == 0)
].head()



## === cell 98
scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
].head()



## === cell 99
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
].copy()



## === cell 100
len(scenario_4_sub)



## === cell 101
scenario_4_sub.loc[:, "H_Distance"] = (scenario_4_sub["fare_amount"] - 2.50) / 1.56



## === cell 102
train.loc[scenario_4_sub.index, "H_Distance"] = scenario_4_sub["H_Distance"]



## === cell 103
train.shape



## === cell 104
train.columns



## === cell 105
test.columns



## === cell 106
test_key = test["key"].copy()



## === cell 107
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
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_jobs=-1, random_state=42)
rf.fit(x_train, y_train.values.ravel())
rf_predict = rf.predict(x_test)



## === cell 117
rf_predict = np.clip(rf_predict, 0.0, None)



## === cell 118
submission = pd.DataFrame({"key": test_key.values, "fare_amount": rf_predict})
submission.to_csv("submission_1.csv", index=False)
submission.head(20)
