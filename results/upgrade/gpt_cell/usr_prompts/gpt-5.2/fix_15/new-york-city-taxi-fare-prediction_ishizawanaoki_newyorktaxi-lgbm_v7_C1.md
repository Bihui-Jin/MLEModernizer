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

3.10

# 2. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
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
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1_000_000
)
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
sample_submission = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)



## === cell 2
train.isnull().sum()



## === cell 3
train.dropna(inplace=True)



## === cell 4
train.describe()



## === cell 5
train.quantile(0.99, numeric_only=True)



## === cell 6
train.quantile(0.01, numeric_only=True)



## === cell 7
train.describe()



## === cell 8
train = train[
    (train["fare_amount"] > 0)
    & (train["fare_amount"] <= 250)
    & (train["passenger_count"] >= 1)
    & (train["passenger_count"] <= 6)
    & (train["pickup_longitude"].between(-74.5, -72.8))
    & (train["dropoff_longitude"].between(-74.5, -72.8))
    & (train["pickup_latitude"].between(40.5, 41.8))
    & (train["dropoff_latitude"].between(40.5, 41.8))
].copy()

train.reset_index(drop=True, inplace=True)
train



## === cell 9
test_keys = test["key"].astype(str).copy()

y_train_raw = train["fare_amount"].astype(np.float64).copy()

train_features_only = train.drop(columns=["fare_amount"])
data = pd.concat([train_features_only, test], sort=False, ignore_index=True)



## === cell 10
data.head()



## === cell 11
keys_all = data["key"].astype(str).copy()




## === cell 12
def haversine_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(np.float64))
    lat1 = np.radians(lat1.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km


pickup_dt = pd.to_datetime(data["pickup_datetime"], errors="coerce", utc=True)

data["pickup_hour"] = pickup_dt.dt.hour.fillna(-1).astype(np.int16)
data["pickup_dayofweek"] = pickup_dt.dt.dayofweek.fillna(-1).astype(np.int16)
data["pickup_month"] = pickup_dt.dt.month.fillna(-1).astype(np.int16)
data["pickup_year"] = pickup_dt.dt.year.fillna(-1).astype(np.int16)
data["is_weekend"] = (data["pickup_dayofweek"] >= 5).fillna(False).astype(np.int8)

n_train_tmp = len(train_features_only)
for col, lo, hi in [
    ("pickup_longitude", -74.5, -72.8),
    ("dropoff_longitude", -74.5, -72.8),
    ("pickup_latitude", 40.5, 41.8),
    ("dropoff_latitude", 40.5, 41.8),
]:
    data.loc[n_train_tmp:, col] = data.loc[n_train_tmp:, col].clip(lo, hi)

data["abs_lon_diff"] = (
    (data["pickup_longitude"] - data["dropoff_longitude"]).abs().astype(np.float64)
)
data["abs_lat_diff"] = (
    (data["pickup_latitude"] - data["dropoff_latitude"]).abs().astype(np.float64)
)
data["manhattan_km_approx"] = (
    (data["abs_lon_diff"] * 84.0) + (data["abs_lat_diff"] * 111.0)
).astype(np.float64)

data["haversine_km"] = haversine_np(
    data["pickup_longitude"].values,
    data["pickup_latitude"].values,
    data["dropoff_longitude"].values,
    data["dropoff_latitude"].values,
)

JFK_LON, JFK_LAT = -73.7781, 40.6413
LGA_LON, LGA_LAT = -73.8740, 40.7769
EWR_LON, EWR_LAT = -74.1745, 40.6895
MAN_LON, MAN_LAT = -73.9851, 40.7589  # Times Sq / Midtown proxy

data["pickup_to_jfk_km"] = haversine_np(
    data["pickup_longitude"].values, data["pickup_latitude"].values, JFK_LON, JFK_LAT
).astype(np.float64)
data["dropoff_to_jfk_km"] = haversine_np(
    data["dropoff_longitude"].values, data["dropoff_latitude"].values, JFK_LON, JFK_LAT
).astype(np.float64)

data["pickup_to_lga_km"] = haversine_np(
    data["pickup_longitude"].values, data["pickup_latitude"].values, LGA_LON, LGA_LAT
).astype(np.float64)
data["dropoff_to_lga_km"] = haversine_np(
    data["dropoff_longitude"].values, data["dropoff_latitude"].values, LGA_LON, LGA_LAT
).astype(np.float64)

data["pickup_to_ewr_km"] = haversine_np(
    data["pickup_longitude"].values, data["pickup_latitude"].values, EWR_LON, EWR_LAT
).astype(np.float64)
data["dropoff_to_ewr_km"] = haversine_np(
    data["dropoff_longitude"].values, data["dropoff_latitude"].values, EWR_LON, EWR_LAT
).astype(np.float64)

data["pickup_to_manhattan_km"] = haversine_np(
    data["pickup_longitude"].values, data["pickup_latitude"].values, MAN_LON, MAN_LAT
).astype(np.float64)
data["dropoff_to_manhattan_km"] = haversine_np(
    data["dropoff_longitude"].values, data["dropoff_latitude"].values, MAN_LON, MAN_LAT
).astype(np.float64)

data = data.drop(["key", "pickup_datetime"], axis=1)

data.head()



## --- ERROR in cell 12, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1750503647.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     56[0m [0mMAN_LON[0m[0;34m,[0m [0mMAN_LAT[0m [0;34m=[0m [0;34m-[0m[0;36m73.9851[0m[0;34m,[0m [0;36m40.7589[0m  [0;31m# Times Sq / Midtown proxy[0m[0;34m[0m[0;34m[0m[0m
[1;32m     57[0m [0;34m[0m[0m
[0;32m---> 58[0;31m data["pickup_to_jfk_km"] = haversine_np(
[0m[1;32m     59[0m     [0mdata[0m[0;34m[[0m[0;34m"pickup_longitude"[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m,[0m [0mdata[0m[0;34m[[0m[0;34m"pickup_latitude"[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m,[0m [0mJFK_LON[0m[0;34m,[0m [0mJFK_LAT[0m[0;34m[0m[0;34m[0m[0m
[1;32m     60[0m ).astype(np.float64)

[0;32m/tmp/ipykernel_11/1750503647.py[0m in [0;36mhaversine_np[0;34m(lon1, lat1, lon2, lat2)[0m
[1;32m      2[0m     [0mlon1[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlon1[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mfloat64[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m     [0mlat1[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlat1[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mfloat64[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m     [0mlon2[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlon2[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mfloat64[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m     [0mlat2[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlat2[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mfloat64[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m

[0;31mAttributeError[0m: 'float' object has no attribute 'astype'

## === cell 13
n_train = len(train)  # original (cleaned) train length
train_fe = data.iloc[:n_train].copy()
test_fe = data.iloc[n_train:].copy()

distance_mask = (train_fe["haversine_km"] >= 0.01) & (train_fe["haversine_km"] <= 80.0)
train_fe = train_fe.loc[distance_mask].copy()
y_train_raw = y_train_raw.loc[distance_mask].copy()

dist = train_fe["haversine_km"].astype(np.float64)
fare = y_train_raw.astype(np.float64)
fare_per_km = fare / (dist + 1e-3)

rate_mask = (fare_per_km >= 0.5) & (fare_per_km <= 50.0)

tiny_dist_high_fare_mask = ~((dist < 0.1) & (fare > 30.0))

mask2 = rate_mask & tiny_dist_high_fare_mask
train_fe = train_fe.loc[mask2].copy()
y_train_raw = y_train_raw.loc[mask2].copy()

train_fe.reset_index(drop=True, inplace=True)
y_train_raw.reset_index(drop=True, inplace=True)

train_fe.shape, test_fe.shape
