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

5.5494

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 16.9774) has done: 'The main runtime issues come from pandas API changes: `any(1)` must be `any(axis=1)`, and your outlier filters accidentally used `|` between DataFrames instead of boolean masks. Fixing those keeps your cleaning logic the same but makes it execute correctly. The RandomForest crash is because NaNs still appear after feature engineering; we explicitly drop any remaining NaNs from train and impute test NaNs with train medians (score-neutral stability fix that preserves the model). Finally, we ensure the submission uses the test `key` values and writes a valid `.csv` with columns `key,fare_amount`.'
- What this solution (achieved 7.24468) has done: 'Your RMSE is far above the target, so we should make a small, legitimate improvement that preserves your overall pipeline. The biggest score hit comes from converting `key` to datetime and then back to string, which changes IDs and can misalign predictions vs required keys; we keep `key` as the original string throughout and only parse `pickup_datetime` for time features. Next, without changing the model type, we tune the existing RandomForest with slightly more trees and sensible constraints (still the same training approach) to reduce variance and improve generalization on this task. Finally, we keep your existing NaN-handling and ensure the submission uses the exact test keys (unchanged) and writes a valid `key,fare_amount` CSV.'
- What this solution (achieved 4.21873) has done: 'Your current score (7.24468 RMSE) is much worse than the target (3.39809), so we should make small, legitimate improvements that preserve your pipeline but reduce error. The biggest “low-risk, high-impact” change here is to add a couple of standard NYC-taxi baseline features (absolute lat/lon deltas and a simple Manhattan-distance proxy) while keeping your RandomForest approach unchanged; these features usually cut RMSE substantially without changing overall logic. We also add the missing outlier filter for invalid dropoff_longitude (you checked it but didn’t drop it), and we make the few `.apply(...)` distance/fare adjustments vectorized to avoid slowdowns and pandas chained-assignment issues while preserving the same semantics. Finally, we keep your exact submission schema and ensure column alignment is identical for train/test.'
- What this solution (achieved 5.5494) has done: 'The timeout is dominated by (1) fitting a 300-tree RandomForest on 1,000,000 rows and (2) a lot of heavy EDA/printing/plotting and repeated DataFrame slicing/dropping that adds overhead. I keep the exact same feature engineering and RandomForest configuration, but speed things up by reading only needed columns with explicit dtypes, performing all row filtering in a single boolean mask (instead of many `.drop(...)` passes), and rewriting the haversine computation to be vectorized per-DataFrame (no Python loop over a shared global). I also remove/skip all plotting and large sorting/printing cells that don’t affect the final model output. These changes preserve the algorithm, features, and training semantics while reducing CPU time and memory pressure enough to fit within 600 seconds.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd


np.random.seed(42)

print(os.listdir("../input"))



## === cell 1
usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_train = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
    "key": "string",
}
dtype_test = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
    "key": "string",
}

train = pd.read_csv(
    "../input/train.csv",
    nrows=1_000_000,
    usecols=usecols_train,
    dtype=dtype_train,
)
test = pd.read_csv(
    "../input/test.csv",
    usecols=usecols_test,
    dtype=dtype_test,
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
test.isnull().sum().sort_values(ascending=False)



## === cell 8
nyc_bbox = {
    "lon_min": -74.3,
    "lon_max": -73.7,
    "lat_min": 40.5,
    "lat_max": 41.0,
}

mask = np.ones(len(train), dtype=bool)

mask &= ~train.isnull().any(axis=1)

mask &= train["fare_amount"].notna()
mask &= train["fare_amount"].ge(0)  # later also restrict >0
mask &= train["passenger_count"].ne(208)

mask &= train["pickup_latitude"].between(-90, 90)
mask &= train["dropoff_latitude"].between(-90, 90)
mask &= train["pickup_longitude"].between(-180, 180)
mask &= train["dropoff_longitude"].between(-180, 180)

mask &= train["pickup_longitude"].between(nyc_bbox["lon_min"], nyc_bbox["lon_max"])
mask &= train["dropoff_longitude"].between(nyc_bbox["lon_min"], nyc_bbox["lon_max"])
mask &= train["pickup_latitude"].between(nyc_bbox["lat_min"], nyc_bbox["lat_max"])
mask &= train["dropoff_latitude"].between(nyc_bbox["lat_min"], nyc_bbox["lat_max"])

mask &= train["fare_amount"].gt(0) & train["fare_amount"].le(250)

train = train.loc[mask].copy()

train.shape



## === cell 9
train["fare_amount"].describe()



## === cell 10
from collections import Counter

Counter(train["fare_amount"] < 0)



## === cell 11
train.shape



## === cell 12
train["fare_amount"].describe()



## === cell 13
train["fare_amount"].head(10)



## === cell 14
train["passenger_count"].describe()



## === cell 15
train[train["passenger_count"] > 6].head()



## === cell 16
train["passenger_count"].describe()



## === cell 17
train["passenger_count"].describe()



## === cell 18
train["pickup_latitude"].describe()



## === cell 19
train[train["pickup_latitude"] < -90].head()



## === cell 20
train[train["pickup_latitude"] > 90].head()



## === cell 21
train.shape



## === cell 22
train.shape



## === cell 23
train["pickup_longitude"].describe()



## === cell 24
train[train["pickup_longitude"] < -180].head()



## === cell 25
train[train["pickup_longitude"] > 180].head()



## === cell 26
train.shape



## === cell 27
train.shape



## === cell 28
train[train["dropoff_latitude"] < -90].head()



## === cell 29
train[train["dropoff_latitude"] > 90].head()



## === cell 30
train.shape



## === cell 31
train.shape



## === cell 32
train[(train["dropoff_longitude"] < -180) | (train["dropoff_longitude"] > 180)].head()



## === cell 33
train.shape



## === cell 34
train.shape



## === cell 35
train.dtypes



## === cell 36
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], errors="coerce", cache=True
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], errors="coerce", cache=True
)



## === cell 37
train.dtypes



## === cell 38
test.dtypes



## === cell 39
train.head()



## === cell 40
test.head()




## === cell 41
def haversine_distance_df(df, lat1, long1, lat2, long2, out_col="H_Distance"):
    R = 6371.0  # km
    phi1 = np.radians(df[lat1].to_numpy(dtype=np.float64, copy=False))
    phi2 = np.radians(df[lat2].to_numpy(dtype=np.float64, copy=False))
    dphi = np.radians((df[lat2] - df[lat1]).to_numpy(dtype=np.float64, copy=False))
    dlambda = np.radians((df[long2] - df[long1]).to_numpy(dtype=np.float64, copy=False))

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (
        np.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    df[out_col] = (R * c).astype(np.float32)
    return df[out_col]




## === cell 42
haversine_distance_df(
    train,
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
)
haversine_distance_df(
    test, "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
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
    df["Year"] = dt.year.astype("int16")
    df["Month"] = dt.month.astype("int8")
    df["Date"] = dt.day.astype("int8")
    df["Day of Week"] = dt.dayofweek.astype("int8")
    df["Hour"] = dt.hour.astype("int8")



## === cell 48
for df in (train, test):
    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
    df["manhattan_dist"] = df["abs_lon_diff"] + df["abs_lat_diff"]



## === cell 49
for df in (train, test):
    dlon = (df["dropoff_longitude"] - df["pickup_longitude"]).to_numpy(
        dtype=np.float64, copy=False
    )
    dlat = (df["dropoff_latitude"] - df["pickup_latitude"]).to_numpy(
        dtype=np.float64, copy=False
    )
    euclid = np.sqrt(dlon * dlon + dlat * dlat)
    df["euclid_dist"] = euclid.astype(np.float32)
    df["dir_ratio"] = (dlat / (np.abs(dlon) + 1e-6)).astype(np.float32)



## === cell 50
train.head()



## === cell 51
test.head()



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
pass



## === cell 58
pass



## === cell 59
train[["H_Distance", "fare_amount"]].head()



## === cell 60
len(train)



## === cell 61
dist_bins = pd.DataFrame(
    {
        "bins": pd.cut(
            train["H_Distance"],
            bins=[-1e-12, 0, 10, 50, 100, 200, 300, np.inf],
            labels=["0", "0-10", "11-50", "51-100", "100-200", "201-300", ">300"],
        )
    }
)
dist_bins.columns



## === cell 62
pass



## === cell 63
Counter(dist_bins["bins"].astype(str))



## === cell 64
train.loc[
    ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
    & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
    & (train["fare_amount"] == 0)
].head()



## === cell 65
m = (
    ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
    & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
    & (train["fare_amount"] == 0)
)
train = train.loc[~m].copy()



## === cell 66
train.shape



## === cell 67
test.loc[
    ((test["pickup_latitude"] == 0) & (test["pickup_longitude"] == 0))
    & ((test["dropoff_latitude"] != 0) & (test["dropoff_longitude"] != 0))
].head()



## === cell 68
train.loc[
    ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
    & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
    & (train["fare_amount"] == 0)
].head()



## === cell 69
m = (
    ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
    & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
    & (train["fare_amount"] == 0)
)
train = train.loc[~m].copy()



## === cell 70
train.shape



## === cell 71
test.loc[
    ((test["pickup_latitude"] != 0) & (test["pickup_longitude"] != 0))
    & ((test["dropoff_latitude"] == 0) & (test["dropoff_longitude"] == 0))
].head()



## === cell 72
high_distance = train.loc[(train["H_Distance"] > 200) & (train["fare_amount"] != 0)]



## === cell 73
high_distance.head()



## === cell 74
high_distance.shape



## === cell 75
hd_idx = high_distance.index
train.loc[hd_idx, "H_Distance"] = (train.loc[hd_idx, "fare_amount"] - 2.50) / 1.56



## === cell 76
high_distance.head()



## === cell 77
train.shape



## === cell 78
train.shape



## === cell 79
train[train["H_Distance"] == 0].head()



## === cell 80
train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)].head()



## === cell 81
train = train.loc[~((train["H_Distance"] == 0) & (train["fare_amount"] == 0))].copy()



## === cell 82
train[(train["H_Distance"] == 0)].shape



## === cell 83
rush_hour = train.loc[
    (
        ((train["Hour"] >= 6) & (train["Hour"] <= 20))
        & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
        & (train["H_Distance"] == 0)
        & (train["fare_amount"] < 2.5)
    )
]
rush_hour.head()



## === cell 84
train = train.drop(rush_hour.index, axis=0)



## === cell 85
train.shape



## === cell 86
non_rush_hour = train.loc[
    (
        ((train["Hour"] < 6) | (train["Hour"] > 20))
        & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
        & (train["H_Distance"] == 0)
        & (train["fare_amount"] < 3.0)
    )
]
non_rush_hour.head()



## === cell 87
weekends = train.loc[
    ((train["Day of Week"] == 0) | (train["Day of Week"] == 6))
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 3.0)
]
weekends.head()



## === cell 88
train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)].head()



## === cell 89
scenario_3 = train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]



## === cell 90
len(scenario_3)



## === cell 91
scenario_3[["H_Distance", "fare_amount"]].head()



## === cell 92
s3_idx = scenario_3.index
train.loc[s3_idx, "fare_amount"] = (train.loc[s3_idx, "H_Distance"] * 1.56) + 2.50



## === cell 93
scenario_3["fare_amount"].head()



## === cell 94
train.shape



## === cell 95
train.shape



## === cell 96
train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)].head()



## === cell 97
scenario_4 = train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]



## === cell 98
len(scenario_4)



## === cell 99
scenario_4.loc[
    (scenario_4["fare_amount"] <= 3.0) & (scenario_4["H_Distance"] == 0)
].head()



## === cell 100
scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
].head()



## === cell 101
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
]



## === cell 102
len(scenario_4_sub)



## === cell 103
s4_idx = scenario_4_sub.index
train.loc[s4_idx, "H_Distance"] = (train.loc[s4_idx, "fare_amount"] - 2.50) / 1.56



## === cell 104
train.shape



## === cell 105
train.shape



## === cell 106
train.columns



## === cell 107
test.columns



## === cell 108
test_key = test["key"].copy()

train = train.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 109
train.columns



## === cell 110
test.columns



## === cell 111
x_train = train.iloc[:, train.columns != "fare_amount"]
y_train = train.iloc[:, train.columns == "fare_amount"]
x_test = test



## === cell 112
x_train.shape



## === cell 113
x_train.columns



## === cell 114
y_train.shape



## === cell 115
x_test.shape



## === cell 116
x_test.columns



## === cell 117
train_mask = ~x_train.isnull().any(axis=1) & ~y_train.isnull().any(axis=1)
x_train = x_train.loc[train_mask].copy()
y_train = y_train.loc[train_mask, "fare_amount"].copy()

train_medians = x_train.median(numeric_only=True)
x_train = x_train.fillna(train_medians)
x_test = x_test.fillna(train_medians)

x_train.isnull().sum().sum(), x_test.isnull().sum().sum()



## === cell 118
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

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



## === cell 119
rf_predict = np.asarray(rf_predict, dtype=float)
rf_predict = np.where(np.isfinite(rf_predict), rf_predict, np.nan)
fallback = float(y_train.mean())
rf_predict = np.where(np.isnan(rf_predict), fallback, rf_predict)
rf_predict = np.maximum(rf_predict, 0.0)



## === cell 120
submission = pd.read_csv("../input/sample_submission.csv")
submission["key"] = test_key.astype(str).values
submission["fare_amount"] = rf_predict
submission.to_csv("submission_1.csv", index=False)
submission.head(20)
