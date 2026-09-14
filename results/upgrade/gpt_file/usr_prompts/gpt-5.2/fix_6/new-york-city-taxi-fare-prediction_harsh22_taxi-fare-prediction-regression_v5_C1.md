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

if os.path.exists("../input"):
    print(os.listdir("../input"))
else:
    print("../input not found; listing /kaggle/input instead if available.")
    if os.path.exists("/kaggle/input"):
        print(os.listdir("/kaggle/input"))



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
train = train.drop(train[train.isnull().any(axis=1)].index, axis=0)



## === cell 8
train.shape



## === cell 9
train["fare_amount"].describe()



## === cell 10
from collections import Counter

Counter(train["fare_amount"] < 0)



## === cell 11
train = train.drop(train[train["fare_amount"] < 0].index, axis=0)
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
train = train.drop(train[train["passenger_count"] == 208].index, axis=0)



## === cell 17
train["passenger_count"].describe()



## === cell 18
train["pickup_latitude"].describe()



## === cell 19
train[train["pickup_latitude"] < -90]



## === cell 20
train[train["pickup_latitude"] > 90]



## === cell 21
mask_bad_pickup_lat = (train["pickup_latitude"] < -90) | (train["pickup_latitude"] > 90)
train = train.drop(train[mask_bad_pickup_lat].index, axis=0)



## === cell 22
train.shape



## === cell 23
train["pickup_longitude"].describe()



## === cell 24
train[train["pickup_longitude"] < -180]



## === cell 25
train[train["pickup_longitude"] > 180]



## === cell 26
mask_bad_pickup_lon = (train["pickup_longitude"] < -180) | (
    train["pickup_longitude"] > 180
)
train = train.drop(train[mask_bad_pickup_lon].index, axis=0)



## === cell 27
train[train["dropoff_latitude"] < -90]



## === cell 28
train[train["dropoff_latitude"] > 90]



## === cell 29
mask_bad_dropoff_lat = (train["dropoff_latitude"] < -90) | (
    train["dropoff_latitude"] > 90
)
train = train.drop(train[mask_bad_dropoff_lat].index, axis=0)



## === cell 30
train[(train["dropoff_longitude"] < -180) | (train["dropoff_longitude"] > 180)].head()



## === cell 31
mask_bad_dropoff_lon = (train["dropoff_longitude"] < -180) | (
    train["dropoff_longitude"] > 180
)
train = train.drop(train[mask_bad_dropoff_lon].index, axis=0)



## === cell 32
train.dtypes



## === cell 33
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], errors="coerce", infer_datetime_format=True
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], errors="coerce", infer_datetime_format=True
)

train = train.dropna(subset=["pickup_datetime"]).reset_index(drop=True)



## === cell 34
train.dtypes




## === cell 35
def add_haversine(df, lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1 = np.radians(df[lat1].astype(float))
    phi2 = np.radians(df[lat2].astype(float))
    dphi = np.radians((df[lat2] - df[lat1]).astype(float))
    dlambda = np.radians((df[lon2] - df[lon1]).astype(float))

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["H_Distance"] = r * c
    return df


train = add_haversine(
    train,
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
)
test = add_haversine(
    test, "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)



## === cell 36
train["H_Distance"].head(10)



## === cell 37
data = [train, test]
for i in data:
    i["Year"] = i["pickup_datetime"].dt.year
    i["Month"] = i["pickup_datetime"].dt.month
    i["Date"] = i["pickup_datetime"].dt.day
    i["Day of Week"] = i["pickup_datetime"].dt.dayofweek
    i["Hour"] = i["pickup_datetime"].dt.hour



## === cell 38
train.loc[
    ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
    & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
    & (train["fare_amount"] == 0)
]



## === cell 39
train = train.drop(
    train.loc[
        ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
        & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
        & (train["fare_amount"] == 0)
    ].index,
    axis=0,
)



## === cell 40
train.shape



## === cell 41
train.loc[
    ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
    & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
    & (train["fare_amount"] == 0)
]



## === cell 42
train = train.drop(
    train.loc[
        ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
        & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
        & (train["fare_amount"] == 0)
    ].index,
    axis=0,
)



## === cell 43
high_distance = train.loc[(train["H_Distance"] > 200) & (train["fare_amount"] != 0)]



## === cell 44
high_distance



## === cell 45
high_distance = high_distance.copy()



## === cell 46
pass



## === cell 47
train[train["H_Distance"] == 0]



## === cell 48
train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)]



## === cell 49
train = train.drop(
    train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)].index, axis=0
)



## === cell 50
train[(train["H_Distance"] == 0)].shape



## === cell 51
rush_hour = train.loc[
    (
        ((train["Hour"] >= 6) & (train["Hour"] <= 20))
        & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
        & (train["H_Distance"] == 0)
        & (train["fare_amount"] < 2.5)
    )
]
rush_hour



## === cell 52
train = train.drop(rush_hour.index, axis=0)



## === cell 53
non_rush_hour = train.loc[
    (
        ((train["Hour"] < 6) | (train["Hour"] > 20))
        & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
        & (train["H_Distance"] == 0)
        & (train["fare_amount"] < 3.0)
    )
]
non_rush_hour



## === cell 54
weekends = train.loc[
    ((train["Day of Week"] == 0) | (train["Day of Week"] == 6))
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 3.0)
]
weekends



## === cell 55
train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]



## === cell 56
scenario_3 = train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)]



## === cell 57
len(scenario_3)



## === cell 58
scenario_3.sort_values("H_Distance", ascending=False)



## === cell 59
scenario_3 = scenario_3.copy()



## === cell 60
scenario_3["fare_amount"] if "fare_amount" in scenario_3.columns else scenario_3



## === cell 61
pass



## === cell 62
train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]



## === cell 63
scenario_4 = train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)]



## === cell 64
len(scenario_4)



## === cell 65
scenario_4.loc[(scenario_4["fare_amount"] <= 3.0) & (scenario_4["H_Distance"] == 0)]



## === cell 66
scenario_4.loc[(scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)]



## === cell 67
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
]



## === cell 68
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
]



## === cell 69
len(scenario_4_sub)



## === cell 70
scenario_4_sub = scenario_4_sub.copy()



## === cell 71
pass



## === cell 72
train.columns



## === cell 73
test.columns



## === cell 74
nyc_bounds = {
    "min_lon": -74.3,
    "max_lon": -73.6,
    "min_lat": 40.5,
    "max_lat": 41.0,
}
mask_nyc_train = (
    train["pickup_longitude"].between(nyc_bounds["min_lon"], nyc_bounds["max_lon"])
    & train["dropoff_longitude"].between(nyc_bounds["min_lon"], nyc_bounds["max_lon"])
    & train["pickup_latitude"].between(nyc_bounds["min_lat"], nyc_bounds["max_lat"])
    & train["dropoff_latitude"].between(nyc_bounds["min_lat"], nyc_bounds["max_lat"])
)
train = train.loc[mask_nyc_train].reset_index(drop=True)

mask_zero_coords_train = (
    (train["pickup_longitude"] == 0)
    | (train["pickup_latitude"] == 0)
    | (train["dropoff_longitude"] == 0)
    | (train["dropoff_latitude"] == 0)
)
train = train.loc[~mask_zero_coords_train].reset_index(drop=True)

train = train.loc[
    (train["fare_amount"] >= 2.5) & (train["fare_amount"] <= 100.0)
].reset_index(drop=True)
train = train.loc[train["passenger_count"].between(1, 6)].reset_index(drop=True)

train = train.loc[
    ~((train["H_Distance"] < 0.01) & (train["fare_amount"] > 15.0))
].reset_index(drop=True)

train = train.loc[train["H_Distance"] > 0].reset_index(drop=True)



## === cell 75
test_key = test["key"].copy()

train = train.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 76
x_train = train.iloc[:, train.columns != "fare_amount"]
y_train = train["fare_amount"].values
x_test = test



## === cell 77
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
x_train_imp = imputer.fit_transform(x_train)
x_test_imp = imputer.transform(x_test)



## === cell 78
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(x_train_imp, y_train)
rf_predict = rf.predict(x_test_imp)

rf_predict = np.clip(rf_predict, 0, None)



## === cell 79
submission = pd.DataFrame({"key": test_key, "fare_amount": rf_predict})
submission.to_csv("submission.csv", index=False)
submission.head(20)
