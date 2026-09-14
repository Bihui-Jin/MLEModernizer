# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

geopandas==0.14.4
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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import math

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

SEED = 42
np.random.seed(SEED)


def resolve_path(fname: str) -> str:
    candidates = [
        os.path.join("../input", fname),
        os.path.join("/kaggle/input", fname),
        os.path.join("/kaggle/data", fname),
        os.path.join("/kaggle/data/new-york-city-taxi-fare-prediction", fname),
        os.path.join("/kaggle/input/new-york-city-taxi-fare-prediction", fname),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


print("Resolved train:", resolve_path("train.csv"))
print("Resolved test :", resolve_path("test.csv"))
print("Resolved sample:", resolve_path("sample_submission.csv"))



## === cell 1
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

cols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]



## === cell 2
train_path = resolve_path("train.csv")
test_path = resolve_path("test.csv")
sample_path = resolve_path("sample_submission.csv")

train = pd.read_csv(train_path, nrows=1000000, usecols=cols, dtype=types)
test = pd.read_csv(
    test_path,
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
)
samp = pd.read_csv(sample_path)



## === cell 3
train.dropna(how="any", axis="rows", inplace=True)
train = train[train.fare_amount > 0]
train = train[train.fare_amount < 250]  # common robust cap for this dataset
train = train[train["passenger_count"] <= 6]
train = train[train["passenger_count"] > 0]



## === cell 4
latitude_mask_pickup = (train.pickup_latitude > -90) & (train.pickup_latitude < 90)
train = train[latitude_mask_pickup]

latitude_mask_dropoff = (train.dropoff_latitude > -90) & (train.dropoff_latitude < 90)
train = train[latitude_mask_dropoff]



## === cell 5
longitude_mask_pickup = (train.pickup_longitude > -180) & (train.pickup_longitude < 180)
train = train[longitude_mask_pickup]

longitude_mask_dropoff = (train.dropoff_longitude > -180) & (
    train.dropoff_longitude < 180
)
train = train[longitude_mask_dropoff]



## === cell 6
nyc_lon_min, nyc_lon_max = -74.3, -73.7
nyc_lat_min, nyc_lat_max = 40.5, 41.0

bbox = (
    (train.pickup_longitude.between(nyc_lon_min, nyc_lon_max))
    & (train.dropoff_longitude.between(nyc_lon_min, nyc_lon_max))
    & (train.pickup_latitude.between(nyc_lat_min, nyc_lat_max))
    & (train.dropoff_latitude.between(nyc_lat_min, nyc_lat_max))
)
train = train[bbox]



## === cell 7
y = train.fare_amount.values
test_id = test.key

train_X = train.drop(["fare_amount"], axis=1)
test_X = test.copy()




## === cell 8
def week_num_vec(day_series: pd.Series) -> pd.Series:
    bins = [0, 7, 14, 21, 28, 31]
    labels = ["first", "second", "third", "fourth", "fifth"]
    return pd.cut(day_series, bins=bins, labels=labels, include_lowest=True, right=True)




## === cell 9
def add_geo_features(data):
    data["abs_diff_longitude"] = (data.dropoff_longitude - data.pickup_longitude).abs()
    data["abs_diff_latitude"] = (data.dropoff_latitude - data.pickup_latitude).abs()

    data["manhattan_distance"] = data["abs_diff_longitude"] + data["abs_diff_latitude"]

    data["squared_long"] = np.power(data["abs_diff_longitude"], 2)
    data["squared_lat"] = np.power(data["abs_diff_latitude"], 2)

    data["euclid_disance"] = np.sqrt(data["squared_long"] + data["squared_lat"])

    return data




## === cell 10
def add_time_features(data):
    dt = pd.to_datetime(data.pickup_datetime, errors="coerce", utc=True)

    data["hour"] = dt.dt.hour.astype("int16")
    data["day_of_week"] = dt.dt.day_name()
    dom = dt.dt.day.astype("int16")
    data["week_of_month"] = week_num_vec(dom)
    data["month"] = dt.dt.month.astype("int16")
    data["year"] = dt.dt.year.astype("int16")

    data["hour"] = data["hour"].astype("string")
    data["month"] = data["month"].astype("string")
    data["year"] = data["year"].astype("string")

    data["day_of_week"] = data["day_of_week"].astype("category")
    data["week_of_month"] = data["week_of_month"].astype("category")

    return data




## === cell 11
train_X = add_geo_features(train_X)
test_X = add_geo_features(test_X)



## === cell 12
train_X = add_time_features(train_X)
test_X = add_time_features(test_X)

train_X = train_X.dropna(axis=0, how="any")
y = y[train_X.index.values]

features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "euclid_disance",
    "manhattan_distance",
]

train_X = train_X[features]
test_X = test_X[features]

train_X = pd.get_dummies(train_X)
test_X = pd.get_dummies(test_X)

train_X, test_X = train_X.align(test_X, join="outer", axis=1, fill_value=0)



## --- ERROR in cell 12, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2842475371.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      5[0m [0;31m# This is a small data hygiene step that improves stability/score without changing model logic.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0mtrain_X[0m [0;34m=[0m [0mtrain_X[0m[0;34m.[0m[0mdropna[0m[0;34m([0m[0maxis[0m[0;34m=[0m[0;36m0[0m[0;34m,[0m [0mhow[0m[0;34m=[0m[0;34m"any"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 7[0;31m [0my[0m [0;34m=[0m [0my[0m[0;34m[[0m[0mtrain_X[0m[0;34m.[0m[0mindex[0m[0;34m.[0m[0mvalues[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m [0;34m[0m[0m
[1;32m      9[0m features = [

[0;31mIndexError[0m: index 974566 is out of bounds for axis 0 with size 974566

## === cell 13
x = train_X
x_test = test_X
