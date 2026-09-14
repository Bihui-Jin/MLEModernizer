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

# 5. Target score

3.61434

# 6. Current score

15.25184

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.25184) has done: 'Your notebook likely didn’t yield a Kaggle score because it can fail before producing a usable submission: it converts `key` to datetime (but `key` has a trailing integer, so many rows become `NaT`), and it has a bugged longitude check that references `dropoff_latitude` instead of `dropoff_longitude`. I make minimal fixes to ensure the pipeline runs end-to-end and always writes a valid `submission.csv` with the required `key,fare_amount` columns aligned to `test.csv`. To move RMSE toward your target (lower is better) without changing the core modeling approach, I keep your LightGBM training but correct parameter typos (`reg_aplha`), remove deprecated/invalid flags (`silent`), and keep prediction using the trained model (no metric/loop changes). I also add a small safety clamp to predictions (non-negative) to reduce extreme-error cases without altering the overall approach.'
- What this solution (achieved 15.25184) has done: 'The crash happens because `x_train` ends up with 0 rows after the earlier cleaning steps, so `RandomForestRegressor.fit()` rightfully raises “Found array with 0 sample(s)”. The smallest safe fix is to guard against an empty training set in cell 115 and fall back to a deterministic constant prediction (using the median fare from the originally loaded `train` dataframe before feature dropping). This keeps the same model approach when data exists, and only changes behavior when training data is empty, allowing the notebook to proceed and produce a valid submission file. The patch also ensures `rf_predict` is always defined for cell 116.'
- What this solution (achieved 15.25184) has done: 'The crash happens because LightGBM receives an empty or non-2D training matrix (`x_train`) at the moment `lgbm.train()` constructs the Dataset, which triggers `ValueError: Input data must be 2 dimensional and non empty.`. This can occur if earlier cleaning leaves zero rows, or if `x_train` ends up as a 1D object. The minimal fix is to guard `lgbm.train()` in cell 121 similarly to the RandomForest guard in cell 115: if `x_train` is empty or not 2D, create a deterministic constant-prediction “model” with a `.predict()` method so cell 122 (and later inference code) can continue unchanged. This preserves the core training logic when data is valid, and only activates the fallback when training is impossible.'
- What this solution (achieved 15.25184) has done: 'Diagnosis: The crash happens in cell 123 because when `x_train` is empty/invalid, cell 121 replaces the LightGBM model with a fallback `_ConstantModel` that does not define `best_iteration`. Cell 123 always calls `model.predict(..., num_iteration=model.best_iteration)`, which exists on LightGBM boosters but not on `_ConstantModel`, causing an `AttributeError`.  
Patch summary: In cell 123, make the prediction call compatible with both model types by using `getattr(model, "best_iteration", None)` and only passing `num_iteration` when it exists and is not `None`. This preserves identical behavior for real LightGBM models while preventing failure for the constant fallback.  
Updated cells: Only cell 123 is modified.  
Compatibility notes for cell k+1: `pred_test_y` remains a NumPy array of length `x_test.shape[0]`, so cell 124 (`print(pred_test_y[:10])`) continues to work unchanged.  
Assumptions: The only non-LightGBM model path is the `_ConstantModel` defined in cell 121, and its `predict(X)` signature does not accept `num_iteration`.'
- What this solution (achieved 15.25184) has done: 'Your current RMSE (15.25) is far worse than the target (3.61), so we should improve performance with minimal, metric-aligned fixes that don’t change the overall modeling approach. The biggest issue is that you’re training on 1M rows but doing almost no NYC-specific geographic filtering, so the model learns from many invalid/outlier coordinates and huge fares, which inflates RMSE; I add the standard NYC bounding-box + fare cap + passenger_count sanity filter before feature engineering (same features/models, just cleaner data). I also fix the LightGBM training bug where `train_set = lgbm.Dataset(...)` is constructed even when `x_train` is empty (your guard happens too late), so the notebook always runs reliably without falling back unnecessarily. Finally, I keep your existing RF/LGB/XGB structure and submission writing, only ensuring the training matrices stay aligned and non-empty to move RMSE down toward the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
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
train["fare_amount"].sort_values(ascending=False).head(10)



## === cell 15
train["passenger_count"].describe()



## === cell 16
train[train["passenger_count"] > 6].head()



## === cell 17
train = train.drop(train[train["passenger_count"] == 208].index, axis=0)



## === cell 18
train["passenger_count"].describe()



## === cell 19
train["pickup_latitude"].describe()



## === cell 20
train[train["pickup_latitude"] < -90].head()



## === cell 21
train[train["pickup_latitude"] > 90].head()



## === cell 22
train = train.drop(
    ((train["pickup_latitude"] < -90) | (train["pickup_latitude"] > 90)).index, axis=0
)



## === cell 23
train.shape



## === cell 24
train["pickup_longitude"].describe()



## === cell 25
train[train["pickup_longitude"] < -180].head()



## === cell 26
train[train["pickup_longitude"] > 180].head()



## === cell 27
train = train.drop(
    ((train["pickup_longitude"] < -180) | (train["pickup_longitude"] > 180)).index,
    axis=0,
)



## === cell 28
train.shape



## === cell 29
train[train["dropoff_latitude"] < -90].head()



## === cell 30
train[train["dropoff_latitude"] > 90].head()



## === cell 31
train = train.drop(
    ((train["dropoff_latitude"] < -90) | (train["dropoff_latitude"] > 90)).index, axis=0
)



## === cell 32
train.shape



## === cell 33
train = train.drop(
    ((train["dropoff_longitude"] < -180) | (train["dropoff_longitude"] > 180)).index,
    axis=0,
)



## === cell 34
train.dtypes



## === cell 35
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"], errors="coerce")



## === cell 36
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
before_rows = len(train)
train = train.dropna(subset=["pickup_datetime"]).copy()

train = train[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 6)].copy()

nyc_lon_min, nyc_lon_max = -74.5, -72.8
nyc_lat_min, nyc_lat_max = 40.5, 41.8

train = train[
    (train["pickup_longitude"].between(nyc_lon_min, nyc_lon_max))
    & (train["dropoff_longitude"].between(nyc_lon_min, nyc_lon_max))
    & (train["pickup_latitude"].between(nyc_lat_min, nyc_lat_max))
    & (train["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max))
].copy()

train = train[(train["fare_amount"] >= 2.5) & (train["fare_amount"] <= 250.0)].copy()

print("NYC/sanity filter rows:", before_rows, "->", len(train))




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
train.sort_values(["H_Distance", "fare_amount"], ascending=False).head()



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
].head()



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
].head()



## === cell 67
train.loc[
    ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
    & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
    & (train["fare_amount"] == 0)
].head()



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
].head()



## === cell 71
high_distance = train.loc[(train["H_Distance"] > 200) & (train["fare_amount"] != 0)]



## === cell 72
high_distance.head()



## === cell 73
high_distance.shape



## === cell 74
high_distance["H_Distance"] = high_distance.apply(
    lambda row: (row["fare_amount"] - 2.50) / 1.56, axis=1
)



## === cell 75
high_distance.head()



## === cell 76
train.update(high_distance)



## === cell 77
train.shape



## === cell 78
train[train["H_Distance"] == 0].head()



## === cell 79
train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)].head()



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
    )
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 2.5)
]
rush_hour.head()



## === cell 83
train = train.drop(rush_hour.index, axis=0)



## === cell 84
train.shape



## === cell 85
non_rush_hour = train.loc[
    (
        ((train["Hour"] < 6) | (train["Hour"] > 20))
        & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
    )
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 3.0)
]
non_rush_hour.head()



## === cell 86
weekends = train.loc[
    ((train["Day of Week"] == 0) | (train["Day of Week"] == 6))
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 3.0)
]
weekends.head()



## === cell 87
train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)].head()



## === cell 88
scenario_3 = train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]



## === cell 89
len(scenario_3)



## === cell 90
scenario_3.sort_values("H_Distance", ascending=False).head()



## === cell 91
scenario_3["fare_amount"] = scenario_3.apply(
    lambda row: ((row["H_Distance"] * 1.56) + 2.50), axis=1
)



## === cell 92
scenario_3["fare_amount"].head()



## === cell 93
train.update(scenario_3)



## === cell 94
train.shape



## === cell 95
train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)].head()



## === cell 96
scenario_4 = train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]



## === cell 97
len(scenario_4)



## === cell 98
scenario_4.loc[
    (scenario_4["fare_amount"] <= 3.0) & (scenario_4["H_Distance"] == 0)
].head()



## === cell 99
scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
].head()



## === cell 100
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
]



## === cell 101
len(scenario_4_sub)



## === cell 102
scenario_4_sub["H_Distance"] = scenario_4_sub.apply(
    lambda row: ((row["fare_amount"] - 2.50) / 1.56), axis=1
)



## === cell 103
train.update(scenario_4_sub)



## === cell 104
train.shape



## === cell 105
train.columns



## === cell 106
test.columns



## === cell 107
train_features = train.drop(["key", "pickup_datetime"], axis=1)
test_features = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 108
train_features.columns



## === cell 109
test_features.columns



## === cell 110
x_train = train_features.iloc[:, train_features.columns != "fare_amount"]
y_train = train_features["fare_amount"].values
x_test = test_features



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

if x_train.shape[0] == 0:
    fallback_fare = float(pd.to_numeric(train["fare_amount"], errors="coerce").median())
    if not np.isfinite(fallback_fare):
        fallback_fare = 0.0
    rf_predict = np.full(
        shape=(x_test.shape[0],), fill_value=fallback_fare, dtype=float
    )
else:
    rf = RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1)
    rf.fit(x_train, y_train)
    rf_predict = rf.predict(x_test)



## === cell 117
submission = pd.DataFrame({"key": test["key"].values, "fare_amount": rf_predict})
submission["fare_amount"] = submission["fare_amount"].clip(lower=0)
submission.to_csv("submission_1.csv", index=False)
submission.head(20)



## === cell 118
import lightgbm as lgbm



## === cell 119
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "nthread": -1,
    "verbose": -1,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "max_depth": -1,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_alpha": 1.0,
    "reg_lambda": 0.001,
    "metric": "rmse",
    "min_split_gain": 0.5,
    "min_child_weight": 1,
    "min_child_samples": 10,
}



## === cell 120
pred_test_y = np.zeros(x_test.shape[0])
pred_test_y.shape



## === cell 121
if (
    (not hasattr(x_train, "shape"))
    or (len(x_train.shape) != 2)
    or (x_train.shape[0] == 0)
):
    train_set = None
else:
    train_set = lgbm.Dataset(x_train, y_train, free_raw_data=False)

train_set



## === cell 122
if train_set is None:
    fallback_fare = float(pd.to_numeric(train["fare_amount"], errors="coerce").median())
    if not np.isfinite(fallback_fare):
        fallback_fare = 0.0

    class _ConstantModel:
        def __init__(self, value):
            self.value = float(value)

        def predict(self, X):
            n = X.shape[0] if hasattr(X, "shape") else len(X)
            return np.full(shape=(n,), fill_value=self.value, dtype=float)

        def __repr__(self):
            return f"_ConstantModel(value={self.value})"

    model = _ConstantModel(fallback_fare)
else:
    model = lgbm.train(params, train_set=train_set, num_boost_round=300)



## === cell 123
print(model)



## === cell 124
best_iter = getattr(model, "best_iteration", None)
if best_iter is None:
    pred_test_y = model.predict(x_test)
else:
    pred_test_y = model.predict(x_test, num_iteration=best_iter)



## === cell 125
print(pred_test_y[:10])



## === cell 126
submission_lgb = pd.DataFrame({"key": test["key"].values, "fare_amount": pred_test_y})
submission_lgb["fare_amount"] = submission_lgb["fare_amount"].clip(lower=0)
submission_lgb.to_csv("submission_LGB.csv", index=False)
submission_lgb.head(20)



## === cell 127
import xgboost as xgb



## === cell 128
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)



## === cell 129
dtrain



## === cell 130
params_xgb = {
    "max_depth": 7,
    "eta": 1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
}
num_rounds = 50



## === cell 131
xb = xgb.train(params_xgb, dtrain, num_rounds)



## === cell 132
y_pred_xgb = xb.predict(dtest)
print(y_pred_xgb[:10])



## === cell 133
submission_xgb = pd.DataFrame({"key": test["key"].values, "fare_amount": y_pred_xgb})
submission_xgb["fare_amount"] = submission_xgb["fare_amount"].clip(lower=0)
submission_xgb.to_csv("submission_XGB.csv", index=False)
submission_xgb.head(20)



## === cell 134
submission_lgb.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_lgb.shape)
print(submission_lgb.head())
