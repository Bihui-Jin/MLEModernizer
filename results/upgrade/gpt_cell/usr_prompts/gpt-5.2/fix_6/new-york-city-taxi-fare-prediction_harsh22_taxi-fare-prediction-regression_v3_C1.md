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

3.37198

# 6. Current score

16.77929

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 16.9774) has done: 'The timeout is dominated by (1) slow `apply(axis=1)` row-wise operations, (2) repeated `drop(...index...)` patterns that copy DataFrames many times, and (3) training a default `RandomForestRegressor` without parallelism. I keep the exact same feature engineering and cleaning logic, but replace row-wise `apply` with fully vectorized equivalent column operations and consolidate the filtering into boolean masks to avoid repeated full-frame copies. I also keep the same RandomForest model but enable deterministic parallel training/prediction via `n_jobs=-1` (same algorithm/semantics; just uses all CPU cores). These changes are provably equivalent and should bring runtime comfortably under 600 seconds.'
- What this solution (achieved 16.77318) has done: 'Your score is far above the target (RMSE 16.98 vs 3.37), so we should improve predictive accuracy without changing the overall approach (same features, same RandomForest training/prediction flow). The biggest likely issue is a cleaning bug: the code never removes invalid dropoff_longitude values (it mistakenly checks dropoff_latitude against [-180, 180]), which injects extreme distances and hurts RMSE. I fix that longitude filter (and keep all other cleaning/feature logic identical), and also add a minimal, metric-safe post-processing clip to keep predictions non-negative (fares can’t be < 0), which typically reduces RMSE a bit. The script still run end-to-end and write `submission_1.csv` with the required columns.'
- What this solution (achieved 16.77929) has done: 'Diagnosis: The crash happens in cell 81 when merging `submission` with `pred_df` on `"key"` because their dtypes don’t match: earlier cells convert `test["key"]` to `datetime64[ns]`, so `test_key` (used in `pred_df`) is datetime, while `sample_submission.csv` keeps `"key"` as an object/string. Pandas 2.x raises a `ValueError` for merges on incompatible key dtypes. The minimal fix is to coerce the `"key"` column in `submission` to the same dtype as `test_key` before the merge.

Patch summary: In cell 81, convert `submission["key"]` using `pd.to_datetime(..., errors="coerce")` so it matches `test_key`’s dtype, then perform the same merge and fill logic unchanged.

Updated cells: Only cell 81 is modified.

Compatibility notes for cell k+1: Cell 81 is the last provided cell; the output `submission` DataFrame and `submission_1.csv` format remain identical (same columns and values), just with a successful merge.

Assumptions: `sample_submission.csv` contains keys parseable by `pd.to_datetime` in the same format as `test["key"]` (as created in cell 34); any unparsable keys become `NaT` and be filled with the existing default `11.35` via the current `fillna` logic.'

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
train = pd.read_csv(
    "../input/train.csv",
    nrows=1000000,
    usecols=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
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
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
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
train.describe()



## === cell 6
train.isnull().sum().sort_values(ascending=False)



## === cell 7
train = train.loc[~train.isnull().any(axis=1)].copy()



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
train["fare_amount"].sort_values(ascending=False)



## === cell 14
train["passenger_count"].describe()



## === cell 15
train[train["passenger_count"] > 8]



## === cell 16
train = train.loc[train["passenger_count"] != 208].copy()



## === cell 17
train["passenger_count"].describe()



## === cell 18
train["pickup_latitude"].describe()



## === cell 19
train[train["pickup_latitude"] < -90]



## === cell 20
train[train["pickup_latitude"] > 90]



## === cell 21
train = train.loc[
    (train["pickup_latitude"] >= -90) & (train["pickup_latitude"] <= 90)
].copy()



## === cell 22
train.shape



## === cell 23
train["pickup_longitude"].describe()



## === cell 24
train[train["pickup_longitude"] < -180]



## === cell 25
train[train["pickup_longitude"] > 180]



## === cell 26
train = train.loc[
    (train["pickup_longitude"] >= -180) & (train["pickup_longitude"] <= 180)
].copy()



## === cell 27
train[train["dropoff_latitude"] < -90]



## === cell 28
train[train["dropoff_latitude"] > 90]



## === cell 29
train = train.loc[
    (train["dropoff_latitude"] >= -90) & (train["dropoff_latitude"] <= 90)
].copy()



## === cell 30
train = train.loc[
    (train["dropoff_longitude"] >= -180) & (train["dropoff_longitude"] <= 180)
].copy()



## === cell 31
test = test.loc[~test.isnull().any(axis=1)].copy()
test = test.loc[(test["passenger_count"] > 0) & (test["passenger_count"] <= 8)].copy()
test = test.loc[
    (test["pickup_latitude"] >= -90) & (test["pickup_latitude"] <= 90)
].copy()
test = test.loc[
    (test["dropoff_latitude"] >= -90) & (test["dropoff_latitude"] <= 90)
].copy()
test = test.loc[
    (test["pickup_longitude"] >= -180) & (test["pickup_longitude"] <= 180)
].copy()
test = test.loc[
    (test["dropoff_longitude"] >= -180) & (test["dropoff_longitude"] <= 180)
].copy()



## === cell 32
train.dtypes



## === cell 33
train["key"] = pd.to_datetime(train["key"], infer_datetime_format=True, cache=True)
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], infer_datetime_format=True, cache=True
)



## === cell 34
test["key"] = pd.to_datetime(test["key"], infer_datetime_format=True, cache=True)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], infer_datetime_format=True, cache=True
)



## === cell 35
train.dtypes




## === cell 36
def haversine_distance(lat1, long1, lat2, long2):
    data = [train, test]
    d_last = None
    r = 6371.0
    for df in data:
        phi1 = np.radians(df[lat1].to_numpy())
        phi2 = np.radians(df[lat2].to_numpy())
        delta_phi = np.radians((df[lat2] - df[lat1]).to_numpy())
        delta_lambda = np.radians((df[long2] - df[long1]).to_numpy())

        a = (
            np.sin(delta_phi / 2.0) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        d = r * c  # kilometers

        df["H_Distance"] = d
        d_last = d
    return d_last




## === cell 37
haversine_distance(
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)



## === cell 38
train["H_Distance"].head(10)



## === cell 39
data = [train, test]
for i in data:
    dt = i["pickup_datetime"].dt
    i["Year"] = dt.year
    i["Month"] = dt.month
    i["Date"] = dt.day
    i["Day of Week"] = dt.dayofweek
    i["Hour"] = dt.hour



## === cell 40
train.loc[
    ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
    & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
    & (train["fare_amount"] == 0)
]



## === cell 41
cond = (
    (train["pickup_latitude"].eq(0) & train["pickup_longitude"].eq(0))
    & (~train["dropoff_latitude"].eq(0) & ~train["dropoff_longitude"].eq(0))
    & train["fare_amount"].eq(0)
)
train = train.loc[~cond].copy()



## === cell 42
train.shape



## === cell 43
train.loc[
    ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
    & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
    & (train["fare_amount"] == 0)
]



## === cell 44
cond = (
    (~train["pickup_latitude"].eq(0) & ~train["pickup_longitude"].eq(0))
    & (train["dropoff_latitude"].eq(0) & train["dropoff_longitude"].eq(0))
    & train["fare_amount"].eq(0)
)
train = train.loc[~cond].copy()



## === cell 45
high_distance = train.loc[(train["H_Distance"] > 200) & (train["fare_amount"] != 0)]



## === cell 46
high_distance



## === cell 47
high_distance = high_distance.copy()
high_distance["H_Distance"] = (high_distance["fare_amount"] - 2.50) / 1.56



## === cell 48
train.update(high_distance)



## === cell 49
train[train["H_Distance"] == 0]



## === cell 50
train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)]



## === cell 51
train = train.loc[~((train["H_Distance"] == 0) & (train["fare_amount"] == 0))].copy()



## === cell 52
train[(train["H_Distance"] == 0)].shape



## === cell 53
rush_hour = train.loc[
    ((train["Hour"] >= 6) & (train["Hour"] <= 20))
    & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 2.5)
]
rush_hour



## === cell 54
train = train.drop(rush_hour.index, axis=0)



## === cell 55
non_rush_hour = train.loc[
    (((train["Hour"] < 6) | (train["Hour"] > 20)))
    & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 3.0)
]
non_rush_hour



## === cell 56
weekends = train.loc[
    ((train["Day of Week"] == 0) | (train["Day of Week"] == 6))
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 3.0)
]
weekends



## === cell 57
train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]



## === cell 58
scenario_3 = train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]



## === cell 59
len(scenario_3)



## === cell 60
scenario_3.sort_values("H_Distance", ascending=False)



## === cell 61
scenario_3 = scenario_3.copy()
scenario_3["fare_amount"] = (scenario_3["H_Distance"] * 1.56) + 2.50



## === cell 62
scenario_3["fare_amount"]



## === cell 63
train.update(scenario_3)



## === cell 64
train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]



## === cell 65
scenario_4 = train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]



## === cell 66
len(scenario_4)



## === cell 67
scenario_4.loc[(scenario_4["fare_amount"] <= 3.0) & (scenario_4["H_Distance"] == 0)]



## === cell 68
scenario_4.loc[(scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)]



## === cell 69
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
]



## === cell 70
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
]



## === cell 71
len(scenario_4_sub)



## === cell 72
scenario_4_sub = scenario_4_sub.copy()
scenario_4_sub["H_Distance"] = (scenario_4_sub["fare_amount"] - 2.50) / 1.56



## === cell 73
train.update(scenario_4_sub)



## === cell 74
train.columns



## === cell 75
test.columns



## === cell 76
test_key = test["key"].copy()



## === cell 77
train = train.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 78
x_train = train.iloc[:, train.columns != "fare_amount"]
y_train = train["fare_amount"].values
x_test = test



## === cell 79
x_test = x_test.reindex(columns=x_train.columns)



## === cell 80
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_jobs=-1, random_state=42)
rf.fit(x_train, y_train)
rf_predict = rf.predict(x_test)



## === cell 81
rf_predict = np.clip(rf_predict, 0.0, None)

submission = pd.read_csv("../input/sample_submission.csv")

if np.issubdtype(test_key.dtype, np.datetime64):
    submission["key"] = pd.to_datetime(submission["key"], errors="coerce", cache=True)

pred_df = pd.DataFrame({"key": test_key, "fare_amount": rf_predict})
submission = submission.drop(columns=["fare_amount"]).merge(
    pred_df, on="key", how="left"
)
submission["fare_amount"] = submission["fare_amount"].fillna(11.35)

submission.to_csv("submission_1.csv", index=False)
submission.head(20)
