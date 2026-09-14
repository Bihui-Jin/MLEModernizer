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
import numpy as np
import pandas as pd
import os

print(os.listdir("../input"))



## === cell 1
train = pd.read_csv(
    "../input/train.csv",
    nrows=1000000,
    parse_dates=["key", "pickup_datetime"],
)
test = pd.read_csv(
    "../input/test.csv",
    parse_dates=["key", "pickup_datetime"],
)



## === cell 2
_ = train.shape



## === cell 3
_ = test.shape



## === cell 4
_ = None  # train.head(10) omitted for runtime



## === cell 5
_ = None  # train.describe() omitted for runtime



## === cell 6
_ = None  # train.isnull().sum() omitted for runtime



## === cell 7
train = train.loc[~train.isnull().any(axis=1)].copy()



## === cell 8
_ = train.shape



## === cell 9
_ = None  # train['fare_amount'].describe() omitted



## === cell 10
train = train.loc[train["fare_amount"] >= 0].copy()



## === cell 11
_ = train.shape



## === cell 12
_ = None  # train['fare_amount'].describe() omitted



## === cell 13
_ = None  # sort_values omitted



## === cell 14
_ = None  # passenger_count describe omitted



## === cell 15
_ = None  # train[train['passenger_count']>8] omitted



## === cell 16
train = train.loc[train["passenger_count"] != 208].copy()



## === cell 17
_ = None  # describe omitted



## === cell 18
_ = None  # describe omitted



## === cell 19
_ = None  # outlier view omitted



## === cell 20
_ = None  # outlier view omitted



## === cell 21
train = train.loc[
    (train["pickup_latitude"] >= -90) & (train["pickup_latitude"] <= 90)
].copy()



## === cell 22
_ = train.shape



## === cell 23
_ = None  # describe omitted



## === cell 24
_ = None  # view omitted



## === cell 25
_ = None  # view omitted



## === cell 26
train = train.loc[
    (train["pickup_longitude"] >= -180) & (train["pickup_longitude"] <= 180)
].copy()



## === cell 27
_ = None  # view omitted



## === cell 28
_ = None  # view omitted



## === cell 29
train = train.loc[
    (train["dropoff_latitude"] >= -90) & (train["dropoff_latitude"] <= 90)
].copy()



## === cell 30
_ = None  # incorrect original check omitted (it referenced dropoff_latitude for lon bounds)



## === cell 31
_ = train.dtypes  # already parsed dates at read-time



## === cell 32
_ = None  # to_datetime already done at read-time



## === cell 33
_ = train.dtypes




## === cell 34
def haversine_distance(lat1, long1, lat2, long2):
    for df in (train, test):
        r = 6371.0
        phi1 = np.radians(df[lat1].to_numpy())
        phi2 = np.radians(df[lat2].to_numpy())
        delta_phi = np.radians((df[lat2] - df[lat1]).to_numpy())
        delta_lambda = np.radians((df[long2] - df[long1]).to_numpy())

        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (
            np.sin(delta_lambda / 2.0) ** 2
        )
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        df["H_Distance"] = r * c
    return train["H_Distance"]




## === cell 35
haversine_distance(
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)



## === cell 36
_ = None  # head omitted



## === cell 37
for df in (train, test):
    dt = df["pickup_datetime"]
    df["Year"] = dt.dt.year
    df["Month"] = dt.dt.month
    df["Date"] = dt.dt.day
    df["Day of Week"] = dt.dt.dayofweek
    df["Hour"] = dt.dt.hour



## === cell 38
_ = None  # diagnostic view omitted



## === cell 39
m = (
    (train["pickup_latitude"] == 0)
    & (train["pickup_longitude"] == 0)
    & (train["dropoff_latitude"] != 0)
    & (train["dropoff_longitude"] != 0)
    & (train["fare_amount"] == 0)
)
train = train.loc[~m].copy()



## === cell 40
_ = train.shape



## === cell 41
_ = None  # diagnostic view omitted



## === cell 42
m = (
    (train["pickup_latitude"] != 0)
    & (train["pickup_longitude"] != 0)
    & (train["dropoff_latitude"] == 0)
    & (train["dropoff_longitude"] == 0)
    & (train["fare_amount"] == 0)
)
train = train.loc[~m].copy()



## === cell 43
mask_high_distance = (train["H_Distance"] > 200) & (train["fare_amount"] != 0)



## === cell 44
_ = None  # printing high_distance omitted



## === cell 45
train.loc[mask_high_distance, "H_Distance"] = (
    train.loc[mask_high_distance, "fare_amount"] - 2.50
) / 1.56



## === cell 46
_ = None  # train.update(high_distance) no longer needed due to in-place assignment



## === cell 47
_ = None  # view omitted



## === cell 48
_ = None  # view omitted



## === cell 49
train = train.loc[~((train["H_Distance"] == 0) & (train["fare_amount"] == 0))].copy()



## === cell 50
_ = None  # shape check omitted



## === cell 51
rush_hour_mask = (
    (train["Hour"] >= 6)
    & (train["Hour"] <= 20)
    & (train["Day of Week"] >= 1)
    & (train["Day of Week"] <= 5)
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 2.5)
)



## === cell 52
train = train.loc[~rush_hour_mask].copy()



## === cell 53
_ = None  # non_rush_hour diagnostic omitted



## === cell 54
_ = None  # weekends diagnostic omitted



## === cell 55
_ = None  # diagnostic omitted



## === cell 56
scenario_3_mask = (train["H_Distance"] != 0) & (train["fare_amount"] == 0)



## === cell 57
_ = int(scenario_3_mask.sum())



## === cell 58
_ = None  # sort diagnostic omitted



## === cell 59
train.loc[scenario_3_mask, "fare_amount"] = (
    train.loc[scenario_3_mask, "H_Distance"] * 1.56
) + 2.50



## === cell 60
_ = None  # display omitted



## === cell 61
_ = None  # update no longer needed (in-place assignment)



## === cell 62
_ = None  # diagnostic omitted



## === cell 63
scenario_4_mask = (train["H_Distance"] == 0) & (train["fare_amount"] != 0)



## === cell 64
_ = int(scenario_4_mask.sum())



## === cell 65
_ = None  # diagnostic omitted



## === cell 66
_ = None  # diagnostic omitted



## === cell 67
scenario_4_sub_mask = (train["fare_amount"] > 3.0) & (train["H_Distance"] == 0)



## === cell 68
scenario_4_sub_mask = (train["fare_amount"] > 3.0) & (train["H_Distance"] == 0)



## === cell 69
_ = int(scenario_4_sub_mask.sum())



## === cell 70
train.loc[scenario_4_sub_mask, "H_Distance"] = (
    train.loc[scenario_4_sub_mask, "fare_amount"] - 2.50
) / 1.56



## === cell 71
_ = None  # update no longer needed (in-place assignment)



## === cell 72
_ = train.columns



## === cell 73
_ = test.columns



## === cell 74
train = train.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 75
x_train = train.loc[:, train.columns != "fare_amount"]
y_train = train["fare_amount"].to_numpy()
x_test = test



## === cell 76
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(x_train, y_train)
rf_predict = rf.predict(x_test)



## === cell 77
submission = pd.read_csv("../input/sample_submission.csv")
submission["fare_amount"] = rf_predict
submission.to_csv("submission_1.csv", index=False)
_ = submission.head(20)



## === cell 78
from sklearn import linear_model

lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)
lr_predict = lr.predict(x_test)



## === cell 79
submission = pd.read_csv("../input/sample_submission.csv")
submission["fare_amount"] = lr_predict
submission.to_csv("submission_2.csv", index=False)
_ = submission.head(20)
