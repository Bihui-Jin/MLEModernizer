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
    lon1 = np.radians(np.asarray(lon1, dtype="float64"))
    lat1 = np.radians(np.asarray(lat1, dtype="float64"))
    lon2 = np.radians(np.asarray(lon2, dtype="float64"))
    lat2 = np.radians(np.asarray(lat2, dtype="float64"))
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
trip_km = pd.Series(trip_km, index=dataset_train.index).loc[dataset_train.index]

fare_per_km = dataset_train["fare_amount"] / np.maximum(trip_km, 0.05)
dataset_train = dataset_train[fare_per_km.between(0.5, 50.0)].copy()

print("new size", len(dataset_train))
dataset_train.head(5)



## --- ERROR in cell 2, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2213132862.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     64[0m )
[1;32m     65[0m [0;34m[0m[0m
[0;32m---> 66[0;31m [0mdataset_train[0m [0;34m=[0m [0mdataset_train[0m[0;34m[[0m[0mtrip_km[0m[0;34m.[0m[0mbetween[0m[0;34m([0m[0;36m0.05[0m[0;34m,[0m [0;36m80.0[0m[0;34m)[0m[0;34m][0m[0;34m.[0m[0mcopy[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     67[0m [0mtrip_km[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mSeries[0m[0;34m([0m[0mtrip_km[0m[0;34m,[0m [0mindex[0m[0;34m=[0m[0mdataset_train[0m[0;34m.[0m[0mindex[0m[0;34m)[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0mdataset_train[0m[0;34m.[0m[0mindex[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m     68[0m [0;34m[0m[0m

[0;31mAttributeError[0m: 'numpy.ndarray' object has no attribute 'between'

## === cell 3
print("dataset_test old size", len(dataset_test))
print("new size (unchanged to preserve required row count)", len(dataset_test))
dataset_test.head(5)
