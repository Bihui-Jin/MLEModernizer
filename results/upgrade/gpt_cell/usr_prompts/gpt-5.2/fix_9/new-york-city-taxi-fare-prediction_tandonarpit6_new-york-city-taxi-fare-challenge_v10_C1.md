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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))


## === cell 1
training_data = pd.read_csv("../input/train.csv", nrows=2000000)
test_data = pd.read_csv("../input/test.csv")


## === cell 2
training_data


## === cell 3
X_train = training_data.copy()
X_test = test_data.copy()
Y_train = training_data.copy()




## === cell 4
def _clean_train(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(
        subset=[
            "fare_amount",
            "pickup_datetime",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
    ).copy()

    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 250)]
    df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)]

    df = df[
        (df["pickup_longitude"].between(-74.5, -72.8))
        & (df["dropoff_longitude"].between(-74.5, -72.8))
        & (df["pickup_latitude"].between(40.5, 41.8))
        & (df["dropoff_latitude"].between(40.5, 41.8))
    ]

    same_coord = (df["pickup_longitude"] == df["dropoff_longitude"]) & (
        df["pickup_latitude"] == df["dropoff_latitude"]
    )
    df = df[~same_coord]

    return df


training_data = _clean_train(training_data)

X_train = training_data.copy()
Y_train = training_data.copy()




## === cell 5
def _haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def _bearing(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


NYC_LON, NYC_LAT = (
    -73.985428,
    40.748817,
)  # Manhattan (Empire State Building) as a robust center proxy


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

    df["hour"] = df["pickup_datetime"].dt.hour
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month
    df["year"] = df["pickup_datetime"].dt.year
    df["dayofyear"] = df["pickup_datetime"].dt.dayofyear

    df["latitude_distance"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
    df["longitude_distance"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["manhattan_approx"] = df["latitude_distance"] + df["longitude_distance"]

    df["delta_lat"] = (df["dropoff_latitude"] - df["pickup_latitude"]).astype(float)
    df["delta_lon"] = (df["dropoff_longitude"] - df["pickup_longitude"]).astype(float)
    df["delta_lat_x_delta_lon"] = df["delta_lat"] * df["delta_lon"]

    df["haversine_km"] = _haversine_km(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )
    df["bearing"] = _bearing(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )

    df["pickup_to_center_km"] = _haversine_km(
        df["pickup_longitude"], df["pickup_latitude"], NYC_LON, NYC_LAT
    )
    df["dropoff_to_center_km"] = _haversine_km(
        df["dropoff_longitude"], df["dropoff_latitude"], NYC_LON, NYC_LAT
    )
    df["center_dist_diff_km"] = (
        df["dropoff_to_center_km"] - df["pickup_to_center_km"]
    ).astype(float)

    return df


X_train = _add_features(X_train)

X_train["_fare_amount_tmp"] = training_data["fare_amount"].values
X_train = X_train[(X_train["haversine_km"] > 0.05) & (X_train["haversine_km"] < 100.0)]

fare_per_km = X_train["_fare_amount_tmp"] / (X_train["haversine_km"] + 1e-6)
X_train = X_train[(fare_per_km > 0.5) & (fare_per_km < 100.0)]

training_data = training_data.loc[X_train.index].copy()
Y_train = training_data.copy()

X_train = X_train.drop(
    columns=[
        "key",
        "pickup_datetime",
        "_fare_amount_tmp",
    ]
)


## --- ERROR in cell 5, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2988331174.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     77[0m [0;34m[0m[0m
[1;32m     78[0m [0;34m[0m[0m
[0;32m---> 79[0;31m [0mX_train[0m [0;34m=[0m [0m_add_features[0m[0;34m([0m[0mX_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     80[0m [0;34m[0m[0m
[1;32m     81[0m [0mX_train[0m[0;34m[[0m[0;34m"_fare_amount_tmp"[0m[0;34m][0m [0;34m=[0m [0mtraining_data[0m[0;34m[[0m[0;34m"fare_amount"[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2988331174.py[0m in [0;36m_add_features[0;34m(df)[0m
[1;32m     64[0m [0;34m[0m[0m
[1;32m     65[0m     [0;31m# Change (score-relevant): distance from pickup/dropoff to NYC center improves airport/outer-borough pricing patterns.[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 66[0;31m     df["pickup_to_center_km"] = _haversine_km(
[0m[1;32m     67[0m         [0mdf[0m[0;34m[[0m[0;34m"pickup_longitude"[0m[0;34m][0m[0;34m,[0m [0mdf[0m[0;34m[[0m[0;34m"pickup_latitude"[0m[0;34m][0m[0;34m,[0m [0mNYC_LON[0m[0;34m,[0m [0mNYC_LAT[0m[0;34m[0m[0;34m[0m[0m
[1;32m     68[0m     )

[0;32m/tmp/ipykernel_11/2988331174.py[0m in [0;36m_haversine_km[0;34m(lon1, lat1, lon2, lat2)[0m
[1;32m      2[0m     [0mlon1[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlon1[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mfloat[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m     [0mlat1[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlat1[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mfloat[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m     [0mlon2[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlon2[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mfloat[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m     [0mlat2[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlat2[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mfloat[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m     [0mdlon[0m [0;34m=[0m [0mlon2[0m [0;34m-[0m [0mlon1[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'float' object has no attribute 'astype'

## === cell 6
X_test = _add_features(X_test)

X_test = X_test.drop(
    columns=[
        "key",
        "pickup_datetime",
    ]
)
