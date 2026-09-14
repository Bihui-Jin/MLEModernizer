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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=1_000_000,
    parse_dates=["pickup_datetime"],
)
test = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)
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
train.query("passenger_count > 6")



## === cell 6
train.query("passenger_count < 1")



## === cell 7
train.query("fare_amount < 0")



## === cell 8
train = train.query("1 <= passenger_count <= 6 and 0 <= fare_amount").copy()
train = train.query("fare_amount <= 250").copy()

train = train[
    (train["pickup_longitude"].between(-75, -72))
    & (train["dropoff_longitude"].between(-75, -72))
    & (train["pickup_latitude"].between(40, 42))
    & (train["dropoff_latitude"].between(40, 42))
].copy()

train.describe()



## === cell 9
train.reset_index(drop=True, inplace=True)
train



## === cell 10
test_keys = test["key"].astype(str).copy()

train["_is_train"] = 1
test["_is_train"] = 0
test["fare_amount"] = np.nan

data = pd.concat([train, test], sort=False, ignore_index=True)



## === cell 11
data.head()




## === cell 12
def haversine_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km


def bearing_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.arctan2(y, x)  # [-pi, pi]
    return brng


data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"], errors="coerce")

data["pickup_hour"] = data["pickup_datetime"].dt.hour.astype("float32")
data["pickup_weekday"] = data["pickup_datetime"].dt.weekday.astype("float32")
data["pickup_month"] = data["pickup_datetime"].dt.month.astype("float32")

data["distance_km"] = haversine_np(
    data["pickup_longitude"],
    data["pickup_latitude"],
    data["dropoff_longitude"],
    data["dropoff_latitude"],
).astype("float32")

data["delta_lon"] = (
    data["dropoff_longitude"].astype(float) - data["pickup_longitude"].astype(float)
).astype("float32")
data["delta_lat"] = (
    data["dropoff_latitude"].astype(float) - data["pickup_latitude"].astype(float)
).astype("float32")

lat_rad = np.radians(data["pickup_latitude"].astype(float))
data["manhattan_km"] = (
    111.0 * np.abs(data["delta_lat"].astype(float))
    + 111.0 * np.cos(lat_rad) * np.abs(data["delta_lon"].astype(float))
).astype("float32")

data["log_distance_km"] = np.log1p(data["distance_km"].astype(float)).astype("float32")

data["distance_km2"] = (data["distance_km"].astype(float) ** 2).astype("float32")
data["center_lon"] = (
    (data["pickup_longitude"].astype(float) + data["dropoff_longitude"].astype(float))
    / 2.0
).astype("float32")
data["center_lat"] = (
    (data["pickup_latitude"].astype(float) + data["dropoff_latitude"].astype(float))
    / 2.0
).astype("float32")
data["bearing"] = bearing_np(
    data["pickup_longitude"],
    data["pickup_latitude"],
    data["dropoff_longitude"],
    data["dropoff_latitude"],
).astype("float32")

nyc_lon, nyc_lat = -73.985428, 40.748817  # Midtown Manhattan-ish (ESB)
data["pickup_to_nyc_km"] = haversine_np(
    data["pickup_longitude"], data["pickup_latitude"], nyc_lon, nyc_lat
).astype("float32")
data["dropoff_to_nyc_km"] = haversine_np(
    data["dropoff_longitude"], data["dropoff_latitude"], nyc_lon, nyc_lat
).astype("float32")


def _in_box(lon, lat, lon_min, lon_max, lat_min, lat_max):
    return (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)


p_lon = data["pickup_longitude"].astype(float)
p_lat = data["pickup_latitude"].astype(float)
d_lon = data["dropoff_longitude"].astype(float)
d_lat = data["dropoff_latitude"].astype(float)

jfk_p = _in_box(p_lon, p_lat, -73.90, -73.75, 40.62, 40.68)
jfk_d = _in_box(d_lon, d_lat, -73.90, -73.75, 40.62, 40.68)
lga_p = _in_box(p_lon, p_lat, -73.90, -73.84, 40.75, 40.79)
lga_d = _in_box(d_lon, d_lat, -73.90, -73.84, 40.75, 40.79)
ewr_p = _in_box(p_lon, p_lat, -74.20, -74.13, 40.67, 40.71)
ewr_d = _in_box(d_lon, d_lat, -74.20, -74.13, 40.67, 40.71)

data["airport_trip"] = (
    (jfk_p | lga_p | ewr_p | jfk_d | lga_d | ewr_d).astype("int8")
).astype("float32")

data = data.drop("pickup_datetime", axis=1)
data["key"] = data["key"].astype(str)

data = data.reset_index(drop=True)
is_train_mask = data["_is_train"] == 1
data = data.loc[(~is_train_mask) | (data["distance_km"].between(0.0, 200.0))].copy()

train_mask = data["_is_train"] == 1
fare = data.loc[train_mask, "fare_amount"].astype(float)
dist = data.loc[train_mask, "distance_km"].astype(float)
fare_per_km = fare / np.maximum(dist, 0.1)
keep_train = fare_per_km.between(0.5, 50.0)  # broad bounds to only drop egregious cases
data = pd.concat(
    [data.loc[~train_mask], data.loc[train_mask].loc[keep_train]],
    axis=0,
    ignore_index=True,
)

train_na_mask = (data["_is_train"] == 1) & data.isna().any(axis=1)
if train_na_mask.any():
    data = data.loc[~train_na_mask].copy()

data = data.reset_index(drop=True)
data.head()



## --- ERROR in cell 12, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2655481370.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     73[0m [0;31m# NYC center proximity (helps separate dense-Manhattan vs outer borough/airport trips)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     74[0m [0mnyc_lon[0m[0;34m,[0m [0mnyc_lat[0m [0;34m=[0m [0;34m-[0m[0;36m73.985428[0m[0;34m,[0m [0;36m40.748817[0m  [0;31m# Midtown Manhattan-ish (ESB)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 75[0;31m data["pickup_to_nyc_km"] = haversine_np(
[0m[1;32m     76[0m     [0mdata[0m[0;34m[[0m[0;34m"pickup_longitude"[0m[0;34m][0m[0;34m,[0m [0mdata[0m[0;34m[[0m[0;34m"pickup_latitude"[0m[0;34m][0m[0;34m,[0m [0mnyc_lon[0m[0;34m,[0m [0mnyc_lat[0m[0;34m[0m[0;34m[0m[0m
[1;32m     77[0m ).astype("float32")

[0;32m/tmp/ipykernel_11/2655481370.py[0m in [0;36mhaversine_np[0;34m(lon1, lat1, lon2, lat2)[0m
[1;32m      2[0m     [0mlon1[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlon1[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mfloat[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m     [0mlat1[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlat1[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mfloat[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m     [0mlon2[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlon2[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mfloat[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m     [0mlat2[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlat2[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mfloat[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m     [0mdlon[0m [0;34m=[0m [0mlon2[0m [0;34m-[0m [0mlon1[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'float' object has no attribute 'astype'

## === cell 13
train = data.loc[data["_is_train"] == 1].copy()
test = data.loc[data["_is_train"] == 0].copy()

y_train = np.log1p(train["fare_amount"].astype(float).clip(lower=0.0))

X_train = train.drop(["fare_amount", "_is_train", "key"], axis=1)
X_test = test.drop(["fare_amount", "_is_train", "key"], axis=1)

X_train.head()
