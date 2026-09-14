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
import os
import numpy as np
import pandas as pd



## === cell 1
TRAIN_PATH_CANDIDATES = [
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/data/train.csv",
]
TEST_PATH_CANDIDATES = [
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/data/test.csv",
]
SAMPLE_SUB_PATH_CANDIDATES = [
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/sample_submission.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {paths}")


TRAIN_PATH = _first_existing(TRAIN_PATH_CANDIDATES)
TEST_PATH = _first_existing(TEST_PATH_CANDIDATES)
SAMPLE_SUB_PATH = _first_existing(SAMPLE_SUB_PATH_CANDIDATES)

print("Using:")
print("TRAIN_PATH:", TRAIN_PATH)
print("TEST_PATH:", TEST_PATH)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)



## === cell 2
TRAIN_NROWS = int(os.environ.get("TRAIN_NROWS", "5000000"))  # can override if needed
train = pd.read_csv(TRAIN_PATH, nrows=TRAIN_NROWS)
test = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_SUB_PATH)

print("Loaded train rows:", len(train), "test rows:", len(test))



## === cell 3
train.isnull().sum()



## === cell 4
train.dropna(inplace=True)



## === cell 5
train.describe()



## === cell 6
train.query("passenger_count > 6")



## === cell 7
train.query("passenger_count < 1")



## === cell 8
train.query("fare_amount < 0")



## === cell 9
train.query("pickup_longitude < -180 or pickup_longitude > 180")



## === cell 10
train.query("dropoff_longitude < -180 or dropoff_longitude > 180")



## === cell 11
train.query("pickup_latitude < -90 or pickup_latitude > 90")



## === cell 12
train.query("dropoff_latitude < -90 or dropoff_latitude > 90")



## === cell 13
train = train.query(
    "1 <= passenger_count <= 6 and "
    "0 <= fare_amount <= 250 and "
    "-180 <= pickup_longitude <= 180 and "
    "-180 <= dropoff_longitude <= 180 and "
    "-90 <= pickup_latitude <= 90 and "
    "-90 <= dropoff_latitude <= 90 and "
    "-74.3 <= pickup_longitude <= -73.7 and "
    "-74.3 <= dropoff_longitude <= -73.7 and "
    "40.5 <= pickup_latitude <= 41.0 and "
    "40.5 <= dropoff_latitude <= 41.0"
)

zero_dist = (train["pickup_longitude"] == train["dropoff_longitude"]) & (
    train["pickup_latitude"] == train["dropoff_latitude"]
)
train = train.loc[~(zero_dist & (train["fare_amount"] > 5.0))].copy()

train.describe()



## === cell 14
train.reset_index(drop=True, inplace=True)
train



## === cell 15
n_train = len(train)
data = pd.concat([train, test], sort=False, ignore_index=True)



## === cell 16
data.head()




## === cell 17
def _haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(np.float64))
    lat1 = np.radians(lat1.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def _bearing(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(np.float64))
    lat1 = np.radians(lat1.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


data["pickup_datetime"] = pd.to_datetime(
    data["pickup_datetime"], errors="coerce", utc=True
)
data["pickup_hour"] = data["pickup_datetime"].dt.hour.astype("Int16")
data["pickup_dayofweek"] = data["pickup_datetime"].dt.dayofweek.astype("Int16")
data["pickup_month"] = data["pickup_datetime"].dt.month.astype("Int16")

for c in ["pickup_hour", "pickup_dayofweek", "pickup_month"]:
    data[c] = data[c].astype("float32").fillna(-1.0)

data["abs_lon_diff"] = (data["dropoff_longitude"] - data["pickup_longitude"]).abs()
data["abs_lat_diff"] = (data["dropoff_latitude"] - data["pickup_latitude"]).abs()

data["haversine_km"] = _haversine_km(
    data["pickup_longitude"].values,
    data["pickup_latitude"].values,
    data["dropoff_longitude"].values,
    data["dropoff_latitude"].values,
)
data["manhattan_km"] = _haversine_km(
    data["pickup_longitude"].values,
    data["pickup_latitude"].values,
    data["dropoff_longitude"].values,
    data["pickup_latitude"].values,
) + _haversine_km(
    data["pickup_longitude"].values,
    data["pickup_latitude"].values,
    data["pickup_longitude"].values,
    data["dropoff_latitude"].values,
)

data["bearing"] = _bearing(
    data["pickup_longitude"].values,
    data["pickup_latitude"].values,
    data["dropoff_longitude"].values,
    data["dropoff_latitude"].values,
)

JFK_LON, JFK_LAT = -73.7781, 40.6413
EWR_LON, EWR_LAT = -74.1745, 40.6895
LGA_LON, LGA_LAT = -73.8740, 40.7769
NYC_LON, NYC_LAT = -73.9855, 40.7580  # Times Sq / Midtown reference

data["pickup_dist_jfk_km"] = _haversine_km(
    data["pickup_longitude"].values, data["pickup_latitude"].values, JFK_LON, JFK_LAT
).astype(np.float32)
data["dropoff_dist_jfk_km"] = _haversine_km(
    data["dropoff_longitude"].values, data["dropoff_latitude"].values, JFK_LON, JFK_LAT
).astype(np.float32)
data["pickup_dist_ewr_km"] = _haversine_km(
    data["pickup_longitude"].values, data["pickup_latitude"].values, EWR_LON, EWR_LAT
).astype(np.float32)
data["dropoff_dist_ewr_km"] = _haversine_km(
    data["dropoff_longitude"].values, data["dropoff_latitude"].values, EWR_LON, EWR_LAT
).astype(np.float32)
data["pickup_dist_lga_km"] = _haversine_km(
    data["pickup_longitude"].values, data["pickup_latitude"].values, LGA_LON, LGA_LAT
).astype(np.float32)
data["dropoff_dist_lga_km"] = _haversine_km(
    data["dropoff_longitude"].values, data["dropoff_latitude"].values, LGA_LON, LGA_LAT
).astype(np.float32)
data["pickup_dist_midtown_km"] = _haversine_km(
    data["pickup_longitude"].values, data["pickup_latitude"].values, NYC_LON, NYC_LAT
).astype(np.float32)
data["dropoff_dist_midtown_km"] = _haversine_km(
    data["dropoff_longitude"].values, data["dropoff_latitude"].values, NYC_LON, NYC_LAT
).astype(np.float32)

data["key"] = data["key"].astype(str)
data = data.drop("pickup_datetime", axis=1)
data.head()



## --- ERROR in cell 17, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/158633704.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     67[0m [0mNYC_LON[0m[0;34m,[0m [0mNYC_LAT[0m [0;34m=[0m [0;34m-[0m[0;36m73.9855[0m[0;34m,[0m [0;36m40.7580[0m  [0;31m# Times Sq / Midtown reference[0m[0;34m[0m[0;34m[0m[0m
[1;32m     68[0m [0;34m[0m[0m
[0;32m---> 69[0;31m data["pickup_dist_jfk_km"] = _haversine_km(
[0m[1;32m     70[0m     [0mdata[0m[0;34m[[0m[0;34m"pickup_longitude"[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m,[0m [0mdata[0m[0;34m[[0m[0;34m"pickup_latitude"[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m,[0m [0mJFK_LON[0m[0;34m,[0m [0mJFK_LAT[0m[0;34m[0m[0;34m[0m[0m
[1;32m     71[0m ).astype(np.float32)

[0;32m/tmp/ipykernel_11/158633704.py[0m in [0;36m_haversine_km[0;34m(lon1, lat1, lon2, lat2)[0m
[1;32m      2[0m     [0mlon1[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlon1[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mfloat64[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m     [0mlat1[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlat1[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mfloat64[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m     [0mlon2[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlon2[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mfloat64[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m     [0mlat2[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlat2[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mfloat64[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m     [0mdlon[0m [0;34m=[0m [0mlon2[0m [0;34m-[0m [0mlon1[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'float' object has no attribute 'astype'

## === cell 18
train = data.iloc[:n_train].copy()
test = data.iloc[n_train:].copy()

eps = 1e-3
fare_per_km = train["fare_amount"] / (train["haversine_km"] + eps)
train = train.loc[(fare_per_km <= 100.0) & (fare_per_km >= 0.0)].copy()

y_train = train["fare_amount"]
X_train = train.drop("fare_amount", axis=1)
X_test = test.drop("fare_amount", axis=1)

if "key" in X_train.columns:
    X_train = X_train.drop(columns=["key"])
if "key" in X_test.columns:
    X_test = X_test.drop(columns=["key"])

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

X_train.head()
