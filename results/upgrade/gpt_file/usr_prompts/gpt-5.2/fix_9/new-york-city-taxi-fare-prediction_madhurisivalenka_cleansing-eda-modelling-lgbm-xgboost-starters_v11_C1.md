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

5.66769

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.2038) has done: 'The timeout is dominated by repeated full-column scans, expensive `DataFrame.drop(...index...)` patterns that copy data many times, two `.apply(axis=1)` row-wise computations, and especially training a 200-tree RandomForest on 1,000,000 rows (which is far slower than LightGBM/XGBoost here). I preserve the exact feature engineering and model logic, but make it faster by: (1) consolidating all row filtering into boolean masks (single copy instead of many), (2) vectorizing the two `.apply(axis=1)` updates, (3) removing/guarding exploratory plots/sorts that don’t affect outputs, and (4) skipping the RandomForest fit (it produces `submission_1.csv` only and doesn’t affect the final `submission.csv`, which is LightGBM-based). LightGBM/XGBoost training remains the same approach and parameters, but with faster data handling and less pandas overhead.'
- What this solution (achieved 5.19137) has done: 'Your current score (5.2038 RMSE) is worse than the target (3.61434), so we should make small, legitimate improvements that reduce RMSE without changing the overall modeling approach. The biggest score win with minimal core-logic impact is to add standard, competition-proven data cleaning filters (NYC bounding box, reasonable fare caps, passenger_count range) while keeping your same haversine + datetime feature engineering and the same LightGBM training flow. I also align the submission `key` ordering to the test file (instead of relying on `sample_submission.csv` order), and clip negative predictions to zero to avoid obvious RMSE penalties. These are minimal changes that typically move this classic baseline closer to ~3.5–4.0 RMSE on this competition with 1M training rows.'
- What this solution (achieved 5.48168) has done: 'Your RMSE (5.19137) is worse than the target (3.61434), so we should make small, legitimate improvements that typically reduce error without changing your overall approach (same haversine+datetime features and same LightGBM training flow). The biggest likely driver of the high RMSE here is that you convert `key` to datetime and then submit that converted value, which breaks the required ID matching; we preserve the original `key` string for submission while keeping your datetime parsing for `pickup_datetime`. Next, we add one minimal, competition-standard cleanup that’s strongly correlated with RMSE improvements: remove rides with near-zero distance but non-trivial fare (often GPS glitches), which otherwise teach the model bad patterns. Finally, we keep the same models and parameters, but ensure LightGBM uses the same feature columns for train/test and that the submission is aligned to `test.csv` order (already is) with the preserved `key`.'
- What this solution (achieved 5.66107) has done: 'Your current RMSE (5.48168) is still far above the target (3.61434), so we should improve model quality with minimal, standard changes while keeping your same feature set (haversine + datetime parts) and the same LightGBM/XGBoost training flow. The biggest low-risk gain is to add a few well-known cleaning filters that remove GPS/label noise (stricter NYC bounding box, minimum distance for non-trivial fares, and removing obvious outliers like extreme fares-per-km), which usually improves this competition substantially without changing core logic. I also fix a small but important issue: your median-imputation uses training feature medians but then fills all test NaNs with that Series (misaligned index can silently produce many NaNs); we reindex medians to the test columns before fill to ensure proper alignment. Everything still runs end-to-end and writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.66418) has done: 'Your RMSE (5.66107) is worse than the target (3.61434), so we should make small, legitimate changes that usually improve this specific competition without changing your overall modeling approach (same haversine + datetime-part features and same LightGBM/XGBoost training flow). The biggest low-risk issue in your current pipeline is that `Year/Month/Date/Day of Week/Hour` become NaN whenever `pickup_datetime` parsing fails (common with `errors="coerce"`), and those NaNs then get median-imputed (often to unrealistic values), hurting generalization; we fix this by dropping rows with invalid `pickup_datetime` in train and filling invalid datetimes in test with the most common train datetime (mode) before extracting the same features. Next, we add one standard NYC Taxi Fare cleanup that materially reduces RMSE but keeps core logic intact: remove extreme `H_Distance` outliers (e.g., > 100 km) which are mostly GPS glitches and badly skew tree splits. Finally, we make sure all feature columns are float32/int consistently for LightGBM/XGBoost to reduce subtle dtype issues while keeping the same models and prediction logic, and we still write a valid `submission.csv`.'
- What this solution (achieved 5.70401) has done: 'Your current RMSE (5.66418) is still far above the target (3.61434), so we should make minimal, competition-standard cleaning changes that reduce label/GPS noise without changing your core feature set (haversine + datetime parts) or the LightGBM/XGBoost training flow. The biggest low-risk gain is to remove airport/outer-borough trips that your current NYC bounding box still includes poorly and that your simple feature set struggles with; we do this by tightening the lat/long bounds to the commonly used NYC core box. Next, we add a standard “minimum fare floor” rule (NYC taxi fares shouldn’t be below ~2.5 except for noise) and remove extreme passenger_count anomalies already handled, both of which usually reduce RMSE substantially. Finally, we keep the exact submission semantics but ensure we always write `submission.csv` from the LightGBM predictions (as you already do).'
- What this solution (achieved 5.66769) has done: 'We should move your RMSE down toward 3.61 (lower is better), so the smallest legitimate improvement is to strengthen data cleaning in a way that doesn’t change your feature set or model training flow. Your current hard “NYC core box” filter likely removes many legitimate trips (especially airports), which can hurt generalization and keep RMSE high; we replace it with the standard wider NYC bounding box used for this competition. Then we add two classic, low-risk quality filters that reduce label/GPS noise for this task: remove very small-distance rides with non-trivial fares, and remove extreme fare-per-km outliers with a slightly tighter band. Everything else (haversine + datetime features, LightGBM/XGBoost training, submission writing) stays the same and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

import os
import warnings

warnings.filterwarnings("ignore")

print(os.listdir("../input"))



## === cell 1
train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
test_dtypes = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

train = pd.read_csv("../input/train.csv", nrows=1000000, dtype=train_dtypes)
test = pd.read_csv("../input/test.csv", dtype=test_dtypes)



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
mask = ~train.isnull().any(axis=1)
mask &= train["fare_amount"].ge(0) & train["fare_amount"].le(250)
mask &= train["fare_amount"].ge(2.5)
mask &= train["passenger_count"].between(1, 6)

mask &= train["pickup_latitude"].between(-90, 90)
mask &= train["pickup_longitude"].between(-180, 180)
mask &= train["dropoff_latitude"].between(-90, 90)
mask &= train["dropoff_longitude"].between(-180, 180)

mask &= train["pickup_latitude"].between(40.5, 41.0)
mask &= train["dropoff_latitude"].between(40.5, 41.0)
mask &= train["pickup_longitude"].between(-74.5, -73.0)
mask &= train["dropoff_longitude"].between(-74.5, -73.0)

train = train.loc[mask].copy()



## === cell 9
train.shape



## === cell 10
train["fare_amount"].describe()



## === cell 11
from collections import Counter

Counter(train["fare_amount"] < 0)



## === cell 12
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
train.shape



## === cell 18
train["passenger_count"].describe()



## === cell 19
train["pickup_latitude"].describe()



## === cell 20
train[train["pickup_latitude"] < -90]



## === cell 21
train[train["pickup_latitude"] > 90]



## === cell 22
train.shape



## === cell 23
train.shape



## === cell 24
train["pickup_longitude"].describe()



## === cell 25
train[train["pickup_longitude"] < -180]



## === cell 26
train[train["pickup_longitude"] > 180]



## === cell 27
train.shape



## === cell 28
train.shape



## === cell 29
train[train["dropoff_latitude"] < -90]



## === cell 30
train[train["dropoff_latitude"] > 90]



## === cell 31
train.shape



## === cell 32
train.shape



## === cell 33
train.shape



## === cell 34
train.dtypes



## === cell 35
train_key_str = train["key"].astype(str).copy()
test_key_str = test["key"].astype(str).copy()

train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"], errors="coerce")
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"], errors="coerce")

train = train.loc[train["pickup_datetime"].notna()].copy()
if test["pickup_datetime"].isna().any():
    fill_dt = train["pickup_datetime"].mode(dropna=True)
    fill_dt = fill_dt.iloc[0] if len(fill_dt) else pd.Timestamp("2010-01-01")
    test["pickup_datetime"] = test["pickup_datetime"].fillna(fill_dt)



## === cell 36
train.dtypes



## === cell 37
test.dtypes



## === cell 38
train.head()



## === cell 39
test.head()




## === cell 40
def haversine_distance(lat1, long1, lat2, long2):
    data = [train, test]
    d_last = None
    R = 6371.0  # radius of earth in kilometers
    for df in data:
        phi1 = np.radians(df[lat1].to_numpy())
        phi2 = np.radians(df[lat2].to_numpy())
        delta_phi = np.radians((df[lat2] - df[lat1]).to_numpy())
        delta_lambda = np.radians((df[long2] - df[long1]).to_numpy())

        a = (np.sin(delta_phi / 2.0) ** 2) + (
            np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2)
        )
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        d = R * c  # km

        df["H_Distance"] = d.astype(np.float32, copy=False)
        d_last = d
    return d_last




## === cell 41
haversine_distance(
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)



## === cell 42
train["H_Distance"].head(10)



## === cell 43
test["H_Distance"].head(10)



## === cell 44
train.head(10)



## === cell 45
test.head(10)



## === cell 46
data = [train, test]
for i in data:
    dt = i["pickup_datetime"].dt
    i["Year"] = dt.year
    i["Month"] = dt.month
    i["Date"] = dt.day
    i["Day of Week"] = dt.dayofweek
    i["Hour"] = dt.hour



## === cell 47
train.head()



## === cell 48
test.head()



## === cell 49
if False:
    plt.figure(figsize=(15, 7))
    plt.hist(train["passenger_count"], bins=15)
    plt.xlabel("No. of Passengers")
    plt.ylabel("Frequency")



## === cell 50
if False:
    plt.figure(figsize=(15, 7))
    plt.scatter(x=train["passenger_count"], y=train["fare_amount"], s=1.5)
    plt.xlabel("No. of Passengers")
    plt.ylabel("Fare")



## === cell 51
if False:
    plt.figure(figsize=(15, 7))
    plt.scatter(x=train["Date"], y=train["fare_amount"], s=1.5)
    plt.xlabel("Date")
    plt.ylabel("Fare")



## === cell 52
if False:
    plt.figure(figsize=(15, 7))
    plt.hist(train["Hour"], bins=100)
    plt.xlabel("Hour")
    plt.ylabel("Frequency")



## === cell 53
if False:
    plt.figure(figsize=(15, 7))
    plt.scatter(x=train["Hour"], y=train["fare_amount"], s=1.5)
    plt.xlabel("Hour")
    plt.ylabel("Fare")



## === cell 54
if False:
    plt.figure(figsize=(15, 7))
    plt.hist(train["Day of Week"], bins=100)
    plt.xlabel("Day of Week")
    plt.ylabel("Frequency")



## === cell 55
if False:
    plt.figure(figsize=(15, 7))
    plt.scatter(x=train["Day of Week"], y=train["fare_amount"], s=1.5)
    plt.xlabel("Day of Week")
    plt.ylabel("Fare")



## === cell 56
if False:
    train.sort_values(["H_Distance", "fare_amount"], ascending=False)



## === cell 57
len(train)



## === cell 58
if False:
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
else:
    dist_bins = None



## === cell 59
if False and dist_bins is not None:
    plt.figure(figsize=(15, 7))
    plt.hist(dist_bins["bins"], bins=75)
    plt.xlabel("Bins")
    plt.ylabel("Frequency")



## === cell 60
if False and dist_bins is not None:
    Counter(dist_bins["bins"])



## === cell 61
train.loc[
    ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
    & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
    & (train["fare_amount"] == 0)
]



## === cell 62
cond = (
    ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
    & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
    & (train["fare_amount"] == 0)
)
train = train.loc[~cond].copy()



## === cell 63
train.shape



## === cell 64
test.loc[
    ((test["pickup_latitude"] == 0) & (test["pickup_longitude"] == 0))
    & ((test["dropoff_latitude"] != 0) & (test["dropoff_longitude"] != 0))
]



## === cell 65
train.loc[
    ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
    & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
    & (train["fare_amount"] == 0)
]



## === cell 66
cond = (
    ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
    & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
    & (train["fare_amount"] == 0)
)
train = train.loc[~cond].copy()



## === cell 67
train.shape



## === cell 68
test.loc[
    ((test["pickup_latitude"] != 0) & (test["pickup_longitude"] != 0))
    & ((test["dropoff_latitude"] == 0) & (test["dropoff_longitude"] == 0))
]



## === cell 69
high_distance = train.loc[(train["H_Distance"] > 200) & (train["fare_amount"] != 0)]



## === cell 70
high_distance



## === cell 71
high_distance.shape



## === cell 72
high_distance = high_distance.copy()
high_distance.loc[:, "H_Distance"] = (high_distance["fare_amount"] - 2.50) / 1.56



## === cell 73
high_distance



## === cell 74
train.update(high_distance)



## === cell 75
train.shape



## === cell 76
train[train["H_Distance"] == 0]



## === cell 77
train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)]



## === cell 78
train = train.loc[~((train["H_Distance"] == 0) & (train["fare_amount"] == 0))].copy()



## === cell 79
train[(train["H_Distance"] == 0)].shape



## === cell 80
rush_hour = train.loc[
    (
        ((train["Hour"] >= 6) & (train["Hour"] <= 20))
        & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
        & (train["H_Distance"] == 0)
        & (train["fare_amount"] < 2.5)
    )
]
rush_hour



## === cell 81
train = train.drop(rush_hour.index, axis=0)



## === cell 82
train.shape



## === cell 83
non_rush_hour = train.loc[
    (
        ((train["Hour"] < 6) | (train["Hour"] > 20))
        & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
        & (train["H_Distance"] == 0)
        & (train["fare_amount"] < 3.0)
    )
]
non_rush_hour



## === cell 84
weekends = train.loc[
    ((train["Day of Week"] == 0) | (train["Day of Week"] == 6))
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 3.0)
]
weekends



## === cell 85
train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]



## === cell 86
scenario_3 = train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]



## === cell 87
len(scenario_3)



## === cell 88
scenario_3.sort_values("H_Distance", ascending=False)



## === cell 89
scenario_3 = scenario_3.copy()
scenario_3.loc[:, "fare_amount"] = (scenario_3["H_Distance"] * 1.56) + 2.50



## === cell 90
scenario_3["fare_amount"]



## === cell 91
train.update(scenario_3)



## === cell 92
train.shape



## === cell 93
train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]



## === cell 94
scenario_4 = train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]



## === cell 95
len(scenario_4)



## === cell 96
scenario_4.loc[(scenario_4["fare_amount"] <= 3.0) & (scenario_4["H_Distance"] == 0)]



## === cell 97
scenario_4.loc[(scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)]



## === cell 98
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
]



## === cell 99
len(scenario_4_sub)



## === cell 100
scenario_4_sub = scenario_4_sub.copy()
scenario_4_sub.loc[:, "H_Distance"] = (scenario_4_sub["fare_amount"] - 2.50) / 1.56



## === cell 101
train.update(scenario_4_sub)



## === cell 102
train.shape



## === cell 103
train = train.loc[
    ~((train["H_Distance"] < 0.05) & (train["fare_amount"] > 10.0))
].copy()



## === cell 104
dist = train["H_Distance"].astype("float32")
fare = train["fare_amount"].astype("float32")
fare_per_km = fare / (dist + 1e-3)

mask_fpk = fare_per_km.between(0.7, 40.0)
mask_dist_fare = ~((dist > 80) & (fare < 20))  # long distance, too cheap
mask_dist_fare &= ~((dist < 0.1) & (fare > 30))  # almost no movement, very expensive
train = train.loc[mask_fpk & mask_dist_fare].copy()

train = train.loc[train["H_Distance"].between(0.0, 100.0)].copy()



## === cell 105
train.columns



## === cell 106
test.columns



## === cell 107
train_features = train.drop(["pickup_datetime"], axis=1)
test_features = test.drop(["pickup_datetime"], axis=1)



## === cell 108
train_features.columns



## === cell 109
test_features.columns



## === cell 110
x_train = train_features.drop(["fare_amount", "key"], axis=1)
y_train = train_features["fare_amount"].values
x_test = test_features.drop(["key"], axis=1)



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
x_train = x_train.apply(pd.to_numeric, errors="coerce")
x_test = x_test.apply(pd.to_numeric, errors="coerce")

train_mask = ~x_train.isna().any(axis=1) & ~pd.isna(y_train)
x_train = x_train.loc[train_mask].copy()
y_train = y_train[train_mask.values]

medians = x_train.median(numeric_only=True)
x_test = x_test.fillna(medians.reindex(x_test.columns))

x_test = x_test.reindex(columns=x_train.columns)

x_train = x_train.astype(np.float32, copy=False)
x_test = x_test.astype(np.float32, copy=False)
y_train = y_train.astype(np.float32, copy=False)



## === cell 117
if False:
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1)
    rf.fit(x_train, y_train)
    rf_predict = rf.predict(x_test)



## === cell 118
if False:
    submission = pd.read_csv("../input/sample_submission.csv")
    submission["fare_amount"] = rf_predict
    submission.to_csv("submission_1.csv", index=False)
    submission.head(20)



## === cell 119
import lightgbm as lgbm



## === cell 120
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
    "reg_alpha": 1,
    "reg_lambda": 0.001,
    "metric": "rmse",
    "min_split_gain": 0.5,
    "min_child_weight": 1,
    "min_child_samples": 10,
    "scale_pos_weight": 1,
}



## === cell 121
pred_test_y = np.zeros(x_test.shape[0])
pred_test_y.shape



## === cell 122
train_set = lgbm.Dataset(x_train, y_train)



## === cell 123
model = lgbm.train(params, train_set=train_set, num_boost_round=300)



## === cell 124
print(model)



## === cell 125
num_iter = (
    model.best_iteration
    if model.best_iteration is not None
    else model.current_iteration()
)
pred_test_y = model.predict(x_test, num_iteration=num_iter)



## === cell 126
pred_test_y = np.clip(pred_test_y, 0, None).astype(np.float32, copy=False)
print(pred_test_y)



## === cell 127
submission_lgb = pd.DataFrame({"key": test_key_str.values, "fare_amount": pred_test_y})
submission_lgb.to_csv("submission_LGB.csv", index=False)
submission_lgb.head(20)



## === cell 128
import xgboost as xgb



## === cell 129
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)



## === cell 130
dtrain



## === cell 131
params = {
    "max_depth": 7,
    "eta": 1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "verbosity": 0,
}
num_rounds = 50



## === cell 132
xb = xgb.train(params, dtrain, num_rounds)



## === cell 133
y_pred_xgb = xb.predict(dtest)
y_pred_xgb = np.clip(y_pred_xgb, 0, None).astype(np.float32, copy=False)
print(y_pred_xgb)



## === cell 134
submission_xgb = pd.DataFrame({"key": test_key_str.values, "fare_amount": y_pred_xgb})
submission_xgb.to_csv("submission_XGB.csv", index=False)
submission_xgb.head(20)



## === cell 135
submission_lgb.to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with columns:",
    list(submission_lgb.columns),
    "and shape:",
    submission_lgb.shape,
)
