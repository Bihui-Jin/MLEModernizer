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
sklearn-pandas==2.2.0
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
import numpy as np
import pandas as pd
import os

print(os.listdir("../input"))



## === cell 1
training_data = pd.read_csv("../input/train.csv", nrows=2500000)
test_data = pd.read_csv("../input/test.csv")



## === cell 2
training_data



## === cell 3
X_train = training_data.copy()
X_test = test_data.copy()
Y_train = training_data.copy()




## === cell 4
def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(
        subset=[
            "fare_amount",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
            "pickup_datetime",
        ]
    ).copy()

    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    )
    df = df.dropna(subset=["pickup_datetime"])
    df["pickup_datetime"] = df["pickup_datetime"].dt.tz_convert(None)

    df = df[
        (df["pickup_longitude"].between(-75.0, -72.0))
        & (df["dropoff_longitude"].between(-75.0, -72.0))
        & (df["pickup_latitude"].between(40.0, 42.0))
        & (df["dropoff_latitude"].between(40.0, 42.0))
    ]

    df = df[
        (df["pickup_longitude"].between(-74.5, -72.8))
        & (df["dropoff_longitude"].between(-74.5, -72.8))
        & (df["pickup_latitude"].between(40.5, 41.9))
        & (df["dropoff_latitude"].between(40.5, 41.9))
    ]

    df["passenger_count"] = df["passenger_count"].clip(1, 6)

    df = df[df["fare_amount"].between(2.5, 250.0)]

    same_loc = (df["pickup_longitude"] == df["dropoff_longitude"]) & (
        df["pickup_latitude"] == df["dropoff_latitude"]
    )
    df = df[~same_loc]

    return df


training_data = _basic_clean(training_data)

X_train = training_data.copy()
Y_train = training_data.copy()




## === cell 5
def haversine_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(np.float64))
    lat1 = np.radians(lat1.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0088 * c
    return km


def _manhattan_km_from_latlon(lat_dist_deg, lon_dist_deg, ref_lat_deg):
    lat_km = 111.32 * lat_dist_deg.astype(np.float64)
    lon_km = (
        111.32 * np.cos(np.radians(ref_lat_deg.astype(np.float64)))
    ) * lon_dist_deg.astype(np.float64)
    return lat_km + lon_km


JFK_LON, JFK_LAT = -73.7781, 40.6413
LGA_LON, LGA_LAT = -73.8740, 40.7769
EWR_LON, EWR_LAT = -74.1745, 40.6895
MH_LON, MH_LAT = -73.985428, 40.748817  # midtown/Empire State vicinity


X_train["hour"] = X_train["pickup_datetime"].dt.hour.astype(np.int16)
X_train["dayofweek"] = X_train["pickup_datetime"].dt.dayofweek.astype(np.int16)
X_train["month"] = X_train["pickup_datetime"].dt.month.astype(np.int16)
X_train["year"] = X_train["pickup_datetime"].dt.year.astype(np.int16)
X_train["minute_of_day"] = (
    X_train["pickup_datetime"].dt.hour * 60 + X_train["pickup_datetime"].dt.minute
).astype(np.int16)
X_train["dayofyear"] = X_train["pickup_datetime"].dt.dayofyear.astype(np.int16)

X_train["latitude_distance"] = (
    (X_train["dropoff_latitude"] - X_train["pickup_latitude"]).abs().astype(np.float32)
)
X_train["longitude_distance"] = (
    (X_train["dropoff_longitude"] - X_train["pickup_longitude"])
    .abs()
    .astype(np.float32)
)

X_train["haversine_km"] = haversine_np(
    X_train["pickup_longitude"],
    X_train["pickup_latitude"],
    X_train["dropoff_longitude"],
    X_train["dropoff_latitude"],
).astype(np.float32)

mean_lat = ((X_train["pickup_latitude"] + X_train["dropoff_latitude"]) / 2.0).astype(
    np.float32
)
X_train["manhattan_km"] = _manhattan_km_from_latlon(
    X_train["latitude_distance"], X_train["longitude_distance"], mean_lat
).astype(np.float32)

X_train["pickup_to_jfk_km"] = haversine_np(
    X_train["pickup_longitude"], X_train["pickup_latitude"], JFK_LON, JFK_LAT
).astype(np.float32)
X_train["dropoff_to_jfk_km"] = haversine_np(
    X_train["dropoff_longitude"], X_train["dropoff_latitude"], JFK_LON, JFK_LAT
).astype(np.float32)
X_train["pickup_to_lga_km"] = haversine_np(
    X_train["pickup_longitude"], X_train["pickup_latitude"], LGA_LON, LGA_LAT
).astype(np.float32)
X_train["dropoff_to_lga_km"] = haversine_np(
    X_train["dropoff_longitude"], X_train["dropoff_latitude"], LGA_LON, LGA_LAT
).astype(np.float32)
X_train["pickup_to_ewr_km"] = haversine_np(
    X_train["pickup_longitude"], X_train["pickup_latitude"], EWR_LON, EWR_LAT
).astype(np.float32)
X_train["dropoff_to_ewr_km"] = haversine_np(
    X_train["dropoff_longitude"], X_train["dropoff_latitude"], EWR_LON, EWR_LAT
).astype(np.float32)
X_train["pickup_to_mh_km"] = haversine_np(
    X_train["pickup_longitude"], X_train["pickup_latitude"], MH_LON, MH_LAT
).astype(np.float32)
X_train["dropoff_to_mh_km"] = haversine_np(
    X_train["dropoff_longitude"], X_train["dropoff_latitude"], MH_LON, MH_LAT
).astype(np.float32)

dist_mask = X_train["haversine_km"].between(0.1, 100.0) & X_train[
    "manhattan_km"
].between(0.1, 200.0)
X_train = X_train.loc[dist_mask].copy()
training_data = training_data.loc[X_train.index].copy()

X_train = X_train.drop(
    columns=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_longitude",
        "pickup_latitude",
    ]
)



## --- ERROR in cell 5, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3401727530.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     61[0m [0;34m[0m[0m
[1;32m     62[0m [0;31m# Landmark distances (pickup and dropoff)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 63[0;31m X_train["pickup_to_jfk_km"] = haversine_np(
[0m[1;32m     64[0m     [0mX_train[0m[0;34m[[0m[0;34m"pickup_longitude"[0m[0;34m][0m[0;34m,[0m [0mX_train[0m[0;34m[[0m[0;34m"pickup_latitude"[0m[0;34m][0m[0;34m,[0m [0mJFK_LON[0m[0;34m,[0m [0mJFK_LAT[0m[0;34m[0m[0;34m[0m[0m
[1;32m     65[0m ).astype(np.float32)

[0;32m/tmp/ipykernel_11/3401727530.py[0m in [0;36mhaversine_np[0;34m(lon1, lat1, lon2, lat2)[0m
[1;32m      2[0m     [0mlon1[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlon1[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mfloat64[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m     [0mlat1[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlat1[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mfloat64[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m     [0mlon2[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlon2[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mfloat64[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m     [0mlat2[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlat2[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mfloat64[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m     [0mdlon[0m [0;34m=[0m [0mlon2[0m [0;34m-[0m [0mlon1[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'float' object has no attribute 'astype'

## === cell 6
X_test["passenger_count"] = X_test["passenger_count"].clip(1, 6)

coord_mask = (
    (X_test["pickup_longitude"].between(-74.5, -72.8))
    & (X_test["dropoff_longitude"].between(-74.5, -72.8))
    & (X_test["pickup_latitude"].between(40.5, 41.9))
    & (X_test["dropoff_latitude"].between(40.5, 41.9))
)
for c in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]:
    X_test.loc[~coord_mask, c] = np.nan

X_test["pickup_datetime"] = pd.to_datetime(
    X_test["pickup_datetime"], errors="coerce", utc=True
)
median_dt = training_data["pickup_datetime"].median()
if pd.isna(median_dt):
    median_dt = pd.Timestamp("2010-01-01")
X_test["pickup_datetime"] = X_test["pickup_datetime"].fillna(
    pd.Timestamp(median_dt).tz_localize("UTC")
)
X_test["pickup_datetime"] = X_test["pickup_datetime"].dt.tz_convert(None)

train_medians = training_data[
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
].median(numeric_only=True)
for c in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]:
    X_test[c] = X_test[c].fillna(train_medians[c])

X_test["hour"] = X_test["pickup_datetime"].dt.hour.astype(np.int16)
X_test["dayofweek"] = X_test["pickup_datetime"].dt.dayofweek.astype(np.int16)
X_test["month"] = X_test["pickup_datetime"].dt.month.astype(np.int16)
X_test["year"] = X_test["pickup_datetime"].dt.year.astype(np.int16)
X_test["minute_of_day"] = (
    X_test["pickup_datetime"].dt.hour * 60 + X_test["pickup_datetime"].dt.minute
).astype(np.int16)
X_test["dayofyear"] = X_test["pickup_datetime"].dt.dayofyear.astype(np.int16)

X_test["latitude_distance"] = (
    (X_test["dropoff_latitude"] - X_test["pickup_latitude"]).abs().astype(np.float32)
)
X_test["longitude_distance"] = (
    (X_test["dropoff_longitude"] - X_test["pickup_longitude"]).abs().astype(np.float32)
)

X_test["haversine_km"] = haversine_np(
    X_test["pickup_longitude"],
    X_test["pickup_latitude"],
    X_test["dropoff_longitude"],
    X_test["dropoff_latitude"],
).astype(np.float32)

mean_lat_t = ((X_test["pickup_latitude"] + X_test["dropoff_latitude"]) / 2.0).astype(
    np.float32
)
X_test["manhattan_km"] = _manhattan_km_from_latlon(
    X_test["latitude_distance"], X_test["longitude_distance"], mean_lat_t
).astype(np.float32)

X_test["pickup_to_jfk_km"] = haversine_np(
    X_test["pickup_longitude"], X_test["pickup_latitude"], JFK_LON, JFK_LAT
).astype(np.float32)
X_test["dropoff_to_jfk_km"] = haversine_np(
    X_test["dropoff_longitude"], X_test["dropoff_latitude"], JFK_LON, JFK_LAT
).astype(np.float32)
X_test["pickup_to_lga_km"] = haversine_np(
    X_test["pickup_longitude"], X_test["pickup_latitude"], LGA_LON, LGA_LAT
).astype(np.float32)
X_test["dropoff_to_lga_km"] = haversine_np(
    X_test["dropoff_longitude"], X_test["dropoff_latitude"], LGA_LON, LGA_LAT
).astype(np.float32)
X_test["pickup_to_ewr_km"] = haversine_np(
    X_test["pickup_longitude"], X_test["pickup_latitude"], EWR_LON, EWR_LAT
).astype(np.float32)
X_test["dropoff_to_ewr_km"] = haversine_np(
    X_test["dropoff_longitude"], X_test["dropoff_latitude"], EWR_LON, EWR_LAT
).astype(np.float32)
X_test["pickup_to_mh_km"] = haversine_np(
    X_test["pickup_longitude"], X_test["pickup_latitude"], MH_LON, MH_LAT
).astype(np.float32)
X_test["dropoff_to_mh_km"] = haversine_np(
    X_test["dropoff_longitude"], X_test["dropoff_latitude"], MH_LON, MH_LAT
).astype(np.float32)

X_test["haversine_km"] = X_test["haversine_km"].clip(0.1, 100.0)
X_test["manhattan_km"] = X_test["manhattan_km"].clip(0.1, 200.0)

X_test = X_test.drop(
    columns=[
        "key",
        "pickup_datetime",
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_longitude",
        "pickup_latitude",
    ]
)
