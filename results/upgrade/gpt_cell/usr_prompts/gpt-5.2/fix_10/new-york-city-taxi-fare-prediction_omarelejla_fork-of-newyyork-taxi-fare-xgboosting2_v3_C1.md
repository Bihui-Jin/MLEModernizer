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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
xgboost==2.0.3

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
import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf>=3.20.3,<5"]
)

import tensorflow as tf

print(tf.__version__)



## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
)  # unused but kept to preserve your original imports

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

NROWS_TRAIN = 600_000
NROWS_TEST = 10_000  # test has ~9914 rows; keep as-is

dataset_train = pd.read_csv(train_iop_path, nrows=NROWS_TRAIN, index_col="key")
dataset_test = pd.read_csv(test_iop_path, nrows=NROWS_TEST, index_col="key")



## === cell 2
print("dataset_train old size", len(dataset_train))

dataset_train = dataset_train[dataset_train.dropoff_longitude != 0]

coord_ok = (
    dataset_train["pickup_longitude"].between(-75, -72)
    & dataset_train["dropoff_longitude"].between(-75, -72)
    & dataset_train["pickup_latitude"].between(40, 42)
    & dataset_train["dropoff_latitude"].between(40, 42)
)
nyc_ok = (
    dataset_train["pickup_longitude"].between(-74.3, -73.7)
    & dataset_train["dropoff_longitude"].between(-74.3, -73.7)
    & dataset_train["pickup_latitude"].between(40.5, 40.95)
    & dataset_train["dropoff_latitude"].between(40.5, 40.95)
)

fare_ok = dataset_train["fare_amount"].between(2.5, 250.0)
pax_ok = dataset_train["passenger_count"].between(1, 6)

same_loc = (dataset_train["pickup_longitude"] == dataset_train["dropoff_longitude"]) & (
    dataset_train["pickup_latitude"] == dataset_train["dropoff_latitude"]
)

dataset_train = dataset_train[coord_ok & nyc_ok & fare_ok & pax_ok & (~same_loc)].copy()

dx = dataset_train["dropoff_longitude"] - dataset_train["pickup_longitude"]
dy = dataset_train["dropoff_latitude"] - dataset_train["pickup_latitude"]
deg_dist = np.sqrt(dx * dx + dy * dy)
dataset_train = dataset_train[deg_dist.between(0.0005, 1.0)].copy()

swap_like = (
    dataset_train["pickup_latitude"].between(-75, -72)  # lat looks like a longitude
    | dataset_train["dropoff_latitude"].between(-75, -72)
    | dataset_train["pickup_longitude"].between(40, 42)  # lon looks like a latitude
    | dataset_train["dropoff_longitude"].between(40, 42)
)
dataset_train = dataset_train[~swap_like].copy()

dx = dataset_train["dropoff_longitude"] - dataset_train["pickup_longitude"]
dy = dataset_train["dropoff_latitude"] - dataset_train["pickup_latitude"]
deg_dist = np.sqrt(dx * dx + dy * dy)
fare_per_deg = dataset_train["fare_amount"] / np.maximum(deg_dist, 1e-6)
dataset_train = dataset_train[fare_per_deg.between(5.0, 500.0)].copy()


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0088
    lon1 = np.radians(lon1.astype("float64"))
    lat1 = np.radians(lat1.astype("float64"))
    lon2 = np.radians(lon2.astype("float64"))
    lat2 = np.radians(lat2.astype("float64"))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


trip_km = haversine_km(
    dataset_train["pickup_longitude"],
    dataset_train["pickup_latitude"],
    dataset_train["dropoff_longitude"],
    dataset_train["dropoff_latitude"],
)

dataset_train = dataset_train[trip_km.between(0.05, 80.0)].copy()
trip_km = trip_km.loc[dataset_train.index]

fare_per_km = dataset_train["fare_amount"] / np.maximum(trip_km, 0.05)
dataset_train = dataset_train[fare_per_km.between(0.5, 50.0)].copy()

print("new size", len(dataset_train))
dataset_train.head(5)



## === cell 3
print("dataset_test old size", len(dataset_test))
print("new size (unchanged to preserve required row count)", len(dataset_test))
dataset_test.head(5)




## === cell 4
def get_year(pickup_date):
    return pickup_date.year


def get_month(pickup_date):
    return pickup_date.month


def get_day(pickup_date):
    return pickup_date.day


def get_hour(pickup_date):
    return pickup_date.hour




## === cell 5
from datetime import datetime as dt
import warnings


def preparedataset2(datasetname):
    warnings.filterwarnings("ignore")

    datasetname = datasetname.copy()

    datasetname["pickup_datetime"] = pd.to_datetime(
        datasetname["pickup_datetime"], errors="coerce", utc=False
    )

    datasetname["pickup_year"] = datasetname["pickup_datetime"].dt.year
    datasetname["pickup_month"] = datasetname["pickup_datetime"].dt.month
    datasetname["pickup_day"] = datasetname["pickup_datetime"].dt.day
    datasetname["pickup_hour"] = datasetname["pickup_datetime"].dt.hour

    datasetname["x_dis"] = (
        datasetname["dropoff_longitude"] - datasetname["pickup_longitude"]
    )
    datasetname["y_dis"] = (
        datasetname["dropoff_latitude"] - datasetname["pickup_latitude"]
    )
    datasetname["dis"] = (datasetname["x_dis"] ** 2 + datasetname["y_dis"] ** 2) ** 0.5

    plon = datasetname["pickup_longitude"].astype("float64")
    plat = datasetname["pickup_latitude"].astype("float64")
    dlon = datasetname["dropoff_longitude"].astype("float64")
    dlat = datasetname["dropoff_latitude"].astype("float64")

    datasetname["haversine_km"] = haversine_km(plon, plat, dlon, dlat)

    JFK = (-73.7781, 40.6413)
    LGA = (-73.8740, 40.7769)
    EWR = (-74.1745, 40.6895)
    MANHATTAN = (-73.9855, 40.7580)

    datasetname["pickup_to_jfk_km"] = haversine_km(plon, plat, JFK[0], JFK[1])
    datasetname["dropoff_to_jfk_km"] = haversine_km(dlon, dlat, JFK[0], JFK[1])

    datasetname["pickup_to_lga_km"] = haversine_km(plon, plat, LGA[0], LGA[1])
    datasetname["dropoff_to_lga_km"] = haversine_km(dlon, dlat, LGA[0], LGA[1])

    datasetname["pickup_to_ewr_km"] = haversine_km(plon, plat, EWR[0], EWR[1])
    datasetname["dropoff_to_ewr_km"] = haversine_km(dlon, dlat, EWR[0], EWR[1])

    datasetname["pickup_to_manhattan_km"] = haversine_km(
        plon, plat, MANHATTAN[0], MANHATTAN[1]
    )
    datasetname["dropoff_to_manhattan_km"] = haversine_km(
        dlon, dlat, MANHATTAN[0], MANHATTAN[1]
    )

    datasetname = datasetname.drop(["pickup_datetime"], axis=1)
    datasetname = datasetname.drop(["pickup_longitude"], axis=1)
    datasetname = datasetname.drop(["dropoff_latitude"], axis=1)
    datasetname = datasetname.drop(["dropoff_longitude"], axis=1)
    datasetname = datasetname.drop(["pickup_latitude"], axis=1)

    return datasetname




## === cell 6
from datetime import datetime as dt
import warnings

warnings.filterwarnings("ignore")


def preparedataset(datasetname):
    datasetname["pickup_year"] = 0
    datasetname["pickup_month"] = 0
    datasetname["pickup_day"] = 0
    datasetname["pickup_hour"] = 0
    datasetname["dis"] = 0
    datasetname["x_dis"] = 0
    datasetname["y_dis"] = 0

    datasetname.head()

    for k in range(len(datasetname.index)):
        datetime = dt.strptime(
            datasetname["pickup_datetime"][k].replace("UTC", ""), "%Y-%m-%d %H:%M:%S "
        )
        datasetname["pickup_year"][k] = datetime.year
        datasetname["pickup_month"][k] = datetime.month
        datasetname["pickup_day"][k] = datetime.day
        datasetname["pickup_hour"][k] = datetime.hour

    datasetname["x_dis"] = (
        datasetname["dropoff_longitude"] - datasetname["pickup_longitude"]
    )
    datasetname["y_dis"] = (
        datasetname["dropoff_latitude"] - datasetname["pickup_latitude"]
    )
    datasetname["dis"] = (
        (datasetname["dropoff_longitude"] - datasetname["pickup_longitude"]) ** 2
        + (datasetname["dropoff_latitude"] - datasetname["pickup_latitude"]) ** 2
    ) ** 0.5

    datasetname = datasetname.drop(["pickup_datetime"], axis=1)
    datasetname = datasetname.drop(["pickup_longitude"], axis=1)
    datasetname = datasetname.drop(["dropoff_latitude"], axis=1)
    datasetname = datasetname.drop(["dropoff_longitude"], axis=1)
    datasetname = datasetname.drop(["pickup_latitude"], axis=1)

    return datasetname




## === cell 7
df = preparedataset2(dataset_train)

before = len(df)
df = df.dropna(
    subset=["pickup_year", "pickup_month", "pickup_day", "pickup_hour"]
).copy()
after = len(df)
print("Dropped rows with invalid datetime-derived features:", before - after)

df.head(5)



## --- ERROR in cell 7, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3235048341.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mdf[0m [0;34m=[0m [0mpreparedataset2[0m[0;34m([0m[0mdataset_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m [0;31m# Change (score-relevant): drop rows with invalid/NaT datetime (prevents noisy NaNs in time features)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;31m# This is training-only noise reduction and preserves evaluation semantics.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mbefore[0m [0;34m=[0m [0mlen[0m[0;34m([0m[0mdf[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3005937642.py[0m in [0;36mpreparedataset2[0;34m(datasetname)[0m
[1;32m     43[0m     [0mMANHATTAN[0m [0;34m=[0m [0;34m([0m[0;34m-[0m[0;36m73.9855[0m[0;34m,[0m [0;36m40.7580[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     44[0m [0;34m[0m[0m
[0;32m---> 45[0;31m     [0mdatasetname[0m[0;34m[[0m[0;34m"pickup_to_jfk_km"[0m[0;34m][0m [0;34m=[0m [0mhaversine_km[0m[0;34m([0m[0mplon[0m[0;34m,[0m [0mplat[0m[0;34m,[0m [0mJFK[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m,[0m [0mJFK[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     46[0m     [0mdatasetname[0m[0;34m[[0m[0;34m"dropoff_to_jfk_km"[0m[0;34m][0m [0;34m=[0m [0mhaversine_km[0m[0;34m([0m[0mdlon[0m[0;34m,[0m [0mdlat[0m[0;34m,[0m [0mJFK[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m,[0m [0mJFK[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     47[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/790915095.py[0m in [0;36mhaversine_km[0;34m(lon1, lat1, lon2, lat2)[0m
[1;32m     49[0m     [0mlon1[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlon1[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0;34m"float64"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     50[0m     [0mlat1[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlat1[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0;34m"float64"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 51[0;31m     [0mlon2[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlon2[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0;34m"float64"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     52[0m     [0mlat2[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlat2[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0;34m"float64"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     53[0m     [0mdlon[0m [0;34m=[0m [0mlon2[0m [0;34m-[0m [0mlon1[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'float' object has no attribute 'astype'

## === cell 8
test_df = preparedataset2(dataset_test)

for c in ["pickup_year", "pickup_month", "pickup_day", "pickup_hour"]:
    if c in test_df.columns:
        test_df[c] = test_df[c].fillna(0)

test_df.head(5)
