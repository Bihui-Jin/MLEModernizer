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

3.40466

# 6. Current score

8.30971

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.50219) has done: 'Most of the timeout comes from doing expensive full-DataFrame sorts, repeated DataFrame filtering/dropping, and especially `DataFrame.apply(axis=1)` (row-wise Python loops) on up to 1M rows. I keep the exact same feature engineering, cleaning rules, and RandomForest training/prediction logic, but replace slow row-wise `apply` with vectorized NumPy math, combine multiple row drops into boolean masks (same semantics), and remove/avoid operations that compute huge intermediate tables but are not used for the model (plots/sorts). I also speed up CSV parsing with explicit dtypes and fast datetime parsing while preserving the same values, and ensure the RandomForest uses all CPU cores (`n_jobs=-1`) without changing the model type or training approach.'
- What this solution (achieved 17.70984) has done: 'Your current score (10.50 RMSE) is far from the target (3.40), so we should improve predictive signal with minimal, metric-aligned fixes while keeping the same overall pipeline (feature engineering + RandomForestRegressor). The biggest issue is that the model is using raw coordinates without strong geographic priors, and it also trains on many noisy/outlier trips (outside NYC area) which inflates RMSE. I add two lightweight, standard taxi-fare features (absolute lat/lon deltas and Manhattan distance) and apply a conservative NYC bounding-box filter on the training set only; this preserves the same model type and training loop but meaningfully reduces label noise. I also clip negative predictions to 0 to avoid obvious RMSE penalties from impossible fares, while keeping the same submission format and paths.'
- What this solution (achieved 8.30971) has done: 'Your current RMSE (17.71) is far worse than the target (3.40), so we should improve predictive signal with minimal, metric-aligned fixes while keeping the same RandomForest approach and existing feature set. The biggest remaining issue is that `S_Distance` is being overwritten for “high distance” rows using a fare-derived formula, which injects label information into a feature during training and then cannot be reproduced at test time—this train/test feature mismatch typically hurts generalization badly. I keep your `S_Distance` computation and all downstream features, but remove the label-derived overwrite and instead only filter extreme/unrealistic distances/fare outliers in training (a standard, conservative cleanup for this competition). This preserves your core logic (same model type, same training call, same feature engineering) while making train/test feature distributions consistent, which should move RMSE substantially toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestRegressor

print(os.listdir("../input"))

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
test_dtypes = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

train_df = pd.read_csv(
    "../input/train.csv",
    nrows=1000000,
    dtype=train_dtypes,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)
test_df = pd.read_csv(
    "../input/test.csv",
    dtype=test_dtypes,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)



## === cell 2
train_df.shape



## === cell 3
train_df.columns



## === cell 4
train_df.head()



## === cell 5
train_df.info()



## === cell 6
train_df.describe()



## === cell 7
test_df.info()



## === cell 8
test_df.describe()



## === cell 9
train_df.isnull().sum()



## === cell 10
train_df = train_df.dropna(axis=0, how="any")



## === cell 11
train_df.info()



## === cell 12
Counter(train_df["fare_amount"] < 0)



## === cell 13
train_df = train_df.loc[train_df["fare_amount"] >= 0].copy()
train_df.shape



## === cell 14
train_df.describe()



## === cell 15
Counter(train_df["passenger_count"] > 6)



## === cell 16
train_df = train_df.loc[train_df["passenger_count"] <= 6].copy()
train_df.shape



## === cell 17
Counter(train_df["pickup_latitude"] < -90)



## === cell 18
Counter(train_df["pickup_latitude"] > 90)



## === cell 19
train_df = train_df.loc[
    (train_df["pickup_latitude"] >= -90) & (train_df["pickup_latitude"] <= 90)
].copy()



## === cell 20
train_df.shape



## === cell 21
Counter(train_df["pickup_longitude"] < -180)



## === cell 22
Counter(train_df["pickup_longitude"] > 180)



## === cell 23
train_df = train_df.loc[train_df["pickup_longitude"] >= -180].copy()



## === cell 24
train_df.shape



## === cell 25
train_df.dtypes



## === cell 26
train_df.head(3)



## === cell 27
pass



## === cell 28
train_df.dtypes



## === cell 29
test_df.dtypes



## === cell 30
train_df.head()



## === cell 31
pass



## === cell 32
test_df.dtypes



## === cell 33
test_df.head()



## === cell 34
train_df.head()



## === cell 35
data = [train_df, test_df]
for i in data:
    dt = i["pickup_datetime"].dt
    i["date"] = dt.day.astype("uint8")
    i["month"] = dt.month.astype("uint8")
    i["day_of_week"] = dt.dayofweek.astype("uint8")
    i["hour"] = dt.hour.astype("uint8")
    i["year"] = dt.year.astype("uint16")



## === cell 36
train_df.head()



## === cell 37
train_df.describe()




## === cell 38
def sphere_distance(lat1, long1, lat2, long2):
    R = 6367.0
    for i in (train_df, test_df):
        phi1 = np.radians(i[lat1].to_numpy())
        phi2 = np.radians(i[lat2].to_numpy())
        delta_phi = np.radians((i[lat2] - i[lat1]).to_numpy())
        delta_lambda = np.radians((i[long2] - i[long1]).to_numpy())
        a = (
            np.sin(delta_phi / 2.0) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
        )
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        d = (R * c).astype("float32")
        i["S_Distance"] = d
    return d  # in Kilometer




## === cell 39
sphere_distance(
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)



## === cell 40
train_df.head()



## === cell 41
pass



## === cell 42
pass



## === cell 43
pass



## === cell 44
pass



## === cell 45
pass



## === cell 46
pass



## === cell 47
len(train_df)



## === cell 48
pass



## === cell 49
pass



## === cell 50
pass



## === cell 51
pass



## === cell 52
mask_bad = (
    (train_df["pickup_latitude"] == 0)
    & (train_df["pickup_longitude"] == 0)
    & (train_df["dropoff_latitude"] != 0)
    & (train_df["dropoff_longitude"] != 0)
    & (train_df["fare_amount"] == 0)
)
train_df = train_df.loc[~mask_bad].copy()



## === cell 53
train_df.shape



## === cell 54
pass



## === cell 55
train_df.shape



## === cell 56
high_distance = train_df.loc[
    (train_df["S_Distance"] > 200) & (train_df["fare_amount"] != 0)
].copy()



## === cell 57
high_distance



## === cell 58
high_distance.shape



## === cell 59
pass



## === cell 60
pass



## === cell 61
train_df



## === cell 62
train_df[train_df["S_Distance"] == 0]



## === cell 63
train_df[(train_df["S_Distance"] == 0) & (train_df["fare_amount"] == 0)]



## === cell 64
train_df = train_df.loc[
    ~((train_df["S_Distance"] == 0) & (train_df["fare_amount"] == 0))
].copy()



## === cell 65
rush_hour = train_df.loc[
    (
        (train_df["hour"] >= 6)
        & (train_df["hour"] <= 20)
        & (train_df["day_of_week"] >= 1)
        & (train_df["day_of_week"] <= 5)
        & (train_df["S_Distance"] == 0)
        & (train_df["fare_amount"] < 2.5)
    )
]
rush_hour



## === cell 66
train_df = train_df.drop(rush_hour.index, axis=0)



## === cell 67
train_df.shape



## === cell 68
non_rush_hour = train_df.loc[
    (
        ((train_df["hour"] < 6) | (train_df["hour"] > 20))
        & (train_df["day_of_week"] >= 1)
        & (train_df["day_of_week"] <= 5)
        & (train_df["S_Distance"] == 0)
        & (train_df["fare_amount"] < 3.0)
    )
]



## === cell 69
non_rush_hour



## === cell 70
non_rush_hour



## === cell 71
train_df.loc[(train_df["S_Distance"] != 0) & (train_df["fare_amount"] == 0)]



## === cell 72
scenario_3 = train_df.loc[
    (train_df["S_Distance"] != 0) & (train_df["fare_amount"] == 0)
].copy()
scenario_3



## === cell 73
scenario_3 = scenario_3



## === cell 74
scenario_3.loc[:, "fare_amount"] = ((scenario_3["S_Distance"] * 1.56) + 2.50).astype(
    "float32"
)



## === cell 75
scenario_3["fare_amount"]



## === cell 76
train_df.loc[(train_df["S_Distance"] == 0) & (train_df["fare_amount"] != 0)]



## === cell 77
scenario_4 = train_df.loc[
    (train_df["S_Distance"] == 0) & (train_df["fare_amount"] != 0)
].copy()



## === cell 78
scenario_4



## === cell 79
len(scenario_3)



## === cell 80
len(scenario_4)



## === cell 81
scenario_4.loc[(scenario_4["fare_amount"] <= 3.0) & (scenario_4["S_Distance"] == 0)]



## === cell 82
scenario_4.loc[(scenario_4["fare_amount"] > 3.0) & (scenario_4["S_Distance"] == 0)]



## === cell 83
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["S_Distance"] == 0)
].copy()



## === cell 84
len(scenario_4_sub)



## === cell 85
pass



## === cell 86
pass



## === cell 87
len(train_df)



## === cell 88
train_df.columns



## === cell 89
test_df.columns



## === cell 90
nyc_mask = (
    train_df["pickup_longitude"].between(-74.3, -73.6)
    & train_df["dropoff_longitude"].between(-74.3, -73.6)
    & train_df["pickup_latitude"].between(40.4, 41.0)
    & train_df["dropoff_latitude"].between(40.4, 41.0)
)
train_df = train_df.loc[nyc_mask].copy()



## === cell 91
train_df = train_df.loc[
    (train_df["fare_amount"] <= 250.0)
    & (train_df["S_Distance"] >= 0.0)
    & (train_df["S_Distance"] <= 100.0)
].copy()



## === cell 92
for df in (train_df, test_df):
    dlon = (df["pickup_longitude"] - df["dropoff_longitude"]).astype("float32")
    dlat = (df["pickup_latitude"] - df["dropoff_latitude"]).astype("float32")
    df["abs_dlon"] = np.abs(dlon).astype("float32")
    df["abs_dlat"] = np.abs(dlat).astype("float32")
    df["manhattan_dist"] = (df["abs_dlon"] + df["abs_dlat"]).astype("float32")



## === cell 93
train_df = train_df.drop(["key", "pickup_datetime"], axis=1)
test_df = test_df.drop(["key", "pickup_datetime"], axis=1)



## === cell 94
train_df.columns



## === cell 95
test_df.columns



## === cell 96
x_train = train_df.iloc[:, train_df.columns != "fare_amount"]
y_train = train_df["fare_amount"].values
x_test = test_df



## === cell 97
x_train.shape



## === cell 98
y_train.shape



## === cell 99
rg = RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE)
rg.fit(x_train, y_train)
y_predict = rg.predict(x_test)

y_predict = np.clip(y_predict, 0.0, None).astype("float32")
y_predict



## === cell 100
submission = pd.read_csv("../input/sample_submission.csv")
submission["fare_amount"] = y_predict
submission.to_csv("submission_1.csv", index=False)
submission.head(10)
