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

3.8

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
)
train_data.head()



## === cell 2
train_data.shape



## === cell 3
train_data.info()



## === cell 4
test_data = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
test_data.head()



## === cell 5
test_data.info()



## === cell 6
train_data.isna().sum()



## === cell 7
train_data["Difference_longitude"] = np.abs(
    np.asarray(train_data["pickup_longitude"] - train_data["dropoff_longitude"])
)
train_data["Difference_latitude"] = np.abs(
    np.asarray(train_data["pickup_latitude"] - train_data["dropoff_latitude"])
)

test_data["Difference_longitude"] = np.abs(
    np.asarray(test_data["pickup_longitude"] - test_data["dropoff_longitude"])
)
test_data["Difference_latitude"] = np.abs(
    np.asarray(test_data["pickup_latitude"] - test_data["dropoff_latitude"])
)



## === cell 8
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")



## === cell 9
plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## === cell 10
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## === cell 11
train_data = train_data[
    (train_data["fare_amount"] > 0) & (train_data["fare_amount"] <= 250)
]
train_data = train_data[
    (train_data["passenger_count"] >= 1) & (train_data["passenger_count"] <= 6)
]

train_data = train_data[
    (train_data["pickup_longitude"].between(-74.3, -72.9))
    & (train_data["dropoff_longitude"].between(-74.3, -72.9))
    & (train_data["pickup_latitude"].between(40.5, 41.8))
    & (train_data["dropoff_latitude"].between(40.5, 41.8))
].copy()



## === cell 12
train_data = train_data[
    ~(
        (train_data["pickup_longitude"] == train_data["dropoff_longitude"])
        & (train_data["pickup_latitude"] == train_data["dropoff_latitude"])
    )
].copy()



## === cell 13
test_data["passenger_count"] = pd.to_numeric(
    test_data["passenger_count"], errors="coerce"
)
test_data["passenger_count"] = (
    test_data["passenger_count"].clip(lower=1, upper=6).fillna(1)
)



## === cell 14
train_dt = pd.to_datetime(train_data["pickup_datetime"], errors="coerce", utc=True)
test_dt = pd.to_datetime(test_data["pickup_datetime"], errors="coerce", utc=True)

train_dt = train_dt.fillna(pd.Timestamp("2009-01-01", tz="UTC"))
test_dt = test_dt.fillna(pd.Timestamp("2009-01-01", tz="UTC"))

train_data["Weekday"] = train_dt.dt.weekday.astype(np.int8)
test_data["Weekday"] = test_dt.dt.weekday.astype(np.int8)

train_data["pickuptime"] = (train_dt.dt.hour * 100 + train_dt.dt.minute).astype(
    np.int16
)
test_data["pickuptime"] = (test_dt.dt.hour * 100 + test_dt.dt.minute).astype(np.int16)



## === cell 15
train_data.head()



## === cell 16
test_data.head()



## === cell 17
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 18
train_data["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)
test_data["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)



## === cell 19
all_days = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
train_data["Weekday"] = pd.Categorical(train_data["Weekday"], categories=all_days)
test_data["Weekday"] = pd.Categorical(test_data["Weekday"], categories=all_days)

train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])

train_one_hot, test_one_hot = train_one_hot.align(
    test_one_hot, join="outer", axis=1, fill_value=0
)

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 20
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 21
train_data["pickuptime"] = (
    pd.to_numeric(train_data["pickuptime"], errors="coerce").fillna(0).astype(int)
)
test_data["pickuptime"] = (
    pd.to_numeric(test_data["pickuptime"], errors="coerce").fillna(0).astype(int)
)



## === cell 22
train_data.head()



## === cell 23
R = 6373.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c

train_data["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1

a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data["Distance"] = np.asarray(distance) * 0.621



## === cell 24
R = 6373.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

lat3 = np.zeros(len(train_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_data)) + np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff = lon3 - lon2
d_lat_dropoff = lat3 - lat2
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
train_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

train_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

lat3 = np.zeros(len(test_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_data)) + np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff = lon3 - lon2
d_lat_dropoff = lat3 - lat2
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
test_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

test_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 25
train_data["Distance"] = np.round(train_data["Distance"], 2)
train_data["Pickup_Distance_airport"] = np.round(
    train_data["Pickup_Distance_airport"], 2
)
train_data["Dropoff_Distance_airport"] = np.round(
    train_data["Dropoff_Distance_airport"], 2
)
test_data["Distance"] = np.round(test_data["Distance"], 2)
test_data["Pickup_Distance_airport"] = np.round(test_data["Pickup_Distance_airport"], 2)
test_data["Dropoff_Distance_airport"] = np.round(
    test_data["Dropoff_Distance_airport"], 2
)



## === cell 26
train_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 27
train_data = train_data[
    (train_data["Distance"] > 0.05) & (train_data["Distance"] < 60)
].copy()
fare_per_mile = train_data["fare_amount"] / train_data["Distance"]
train_data = train_data[fare_per_mile.between(0.5, 50)].copy()



## === cell 28
try:
    pt_bin = (train_data["pickuptime"].astype(np.int32) // 50).clip(lower=0, upper=47)
    med_dist_by_bin = train_data.groupby(pt_bin)["Distance"].median()
    typical = pt_bin.map(med_dist_by_bin).astype(float)

    ratio = train_data["Distance"] / (typical.replace(0, np.nan))
    mask = ratio.isna() | ratio.between(0.05, 25.0)
    train_data = train_data.loc[mask].copy()
except Exception:
    pass



## === cell 29
try:
    drop_dt = pd.to_datetime(
        train_data["key"].astype(str).str.slice(0, 26),
        errors="coerce",
        utc=True,
    )
    pick_dt = train_dt.loc[train_data.index]

    dur_sec = (drop_dt - pick_dt).dt.total_seconds()
    dur_hr = dur_sec / 3600.0

    mph = train_data["Distance"] / dur_hr

    mask = (
        dur_sec.notna()
        & (dur_sec > 0)
        & (dur_sec <= 4 * 3600)
        & mph.between(0.1, 100.0)
    )
    train_data = train_data.loc[mask].copy()

    train_dt = pick_dt.loc[train_data.index]
except Exception:
    pass



## === cell 30
_eps = 1e-12
dl_mean = np.mean(train_data["Difference_longitude"])
dl_std = np.std(train_data["Difference_longitude"])
dl_std = dl_std if dl_std > _eps else 1.0

train_data["Difference_longitude"] = train_data["Difference_longitude"] - dl_mean
train_data["Difference_longitude"] = train_data["Difference_longitude"] / dl_std



## === cell 31
dlat_mean = np.mean(train_data["Difference_latitude"])
dlat_std = np.std(train_data["Difference_latitude"])
dlat_std = dlat_std if dlat_std > _eps else 1.0

train_data["Difference_latitude"] = train_data["Difference_latitude"] - dlat_mean
train_data["Difference_latitude"] = train_data["Difference_latitude"] / dlat_std



## === cell 32
test_data["Difference_longitude"] = test_data["Difference_longitude"] - dl_mean
test_data["Difference_longitude"] = test_data["Difference_longitude"] / dl_std

test_data["Difference_latitude"] = test_data["Difference_latitude"] - dlat_mean
test_data["Difference_latitude"] = test_data["Difference_latitude"] / dlat_std



## === cell 33
scale_cols = ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]
scale_stats = {}
for col in scale_cols:
    m = float(np.mean(train_data[col]))
    s = float(np.std(train_data[col]))
    s = s if s > _eps else 1.0
    scale_stats[col] = (m, s)
    train_data[col] = (train_data[col] - m) / s
    test_data[col] = (test_data[col] - m) / s



## === cell 34
train_data.shape



## === cell 35
test_data.shape



## === cell 36
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

X = (
    X.apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

train_dt_series = train_dt.loc[train_data.index]
order = np.argsort(train_dt_series.values.astype("datetime64[ns]"), kind="mergesort")
X_sorted = X.iloc[order]
y_sorted = y.iloc[order]

split_idx = int(len(X_sorted) * 0.99)
X_train = X_sorted.iloc[:split_idx]
y_train = y_sorted.iloc[:split_idx]
X_test = X_sorted.iloc[split_idx:]
y_test = y_sorted.iloc[split_idx:]



## === cell 37
test_keys = test_data["key"].copy()

test_data_features = test_data.drop(columns=["key"], errors="ignore")
test_data_features = (
    test_data_features.apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

test_X = test_data_features.reindex(columns=list(X.columns), fill_value=0.0)

assert len(test_X.columns) == X.shape[1]
assert test_keys.shape[0] == test_X.shape[0]



## === cell 38
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## --- ERROR in cell 38, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2448030707.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m [0mlr[0m [0;34m=[0m [0mLinearRegression[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m [0mlr[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX_train[0m[0;34m,[0m [0my_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0mprint[0m[0;34m([0m[0mlr[0m[0;34m.[0m[0mscore[0m[0;34m([0m[0mX_test[0m[0;34m,[0m [0my_test[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight)[0m
[1;32m    646[0m         [0maccept_sparse[0m [0;34m=[0m [0;32mFalse[0m [0;32mif[0m [0mself[0m[0;34m.[0m[0mpositive[0m [0;32melse[0m [0;34m[[0m[0;34m"csr"[0m[0;34m,[0m [0;34m"csc"[0m[0;34m,[0m [0;34m"coo"[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    647[0m [0;34m[0m[0m
[0;32m--> 648[0;31m         X, y = self._validate_data(
[0m[1;32m    649[0m             [0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0maccept_sparse[0m[0;34m=[0m[0maccept_sparse[0m[0;34m,[0m [0my_numeric[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0mmulti_output[0m[0;34m=[0m[0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m    650[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/base.py[0m in [0;36m_validate_data[0;34m(self, X, y, reset, validate_separately, **check_params)[0m
[1;32m    582[0m                 [0my[0m [0;34m=[0m [0mcheck_array[0m[0;34m([0m[0my[0m[0;34m,[0m [0minput_name[0m[0;34m=[0m[0;34m"y"[0m[0;34m,[0m [0;34m**[0m[0mcheck_y_params[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    583[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 584[0;31m                 [0mX[0m[0;34m,[0m [0my[0m [0;34m=[0m [0mcheck_X_y[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0;34m**[0m[0mcheck_params[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    585[0m             [0mout[0m [0;34m=[0m [0mX[0m[0;34m,[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    586[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py[0m in [0;36mcheck_X_y[0;34m(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)[0m
[1;32m   1104[0m         )
[1;32m   1105[0m [0;34m[0m[0m
[0;32m-> 1106[0;31m     X = check_array(
[0m[1;32m   1107[0m         [0mX[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1108[0m         [0maccept_sparse[0m[0;34m=[0m[0maccept_sparse[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py[0m in [0;36mcheck_array[0;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)[0m
[1;32m    929[0m         [0mn_samples[0m [0;34m=[0m [0m_num_samples[0m[0;34m([0m[0marray[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    930[0m         [0;32mif[0m [0mn_samples[0m [0;34m<[0m [0mensure_min_samples[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 931[0;31m             raise ValueError(
[0m[1;32m    932[0m                 [0;34m"Found array with %d sample(s) (shape=%s) while a"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    933[0m                 [0;34m" minimum of %d is required%s."[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Found array with 0 sample(s) (shape=(0, 14)) while a minimum of 1 is required by LinearRegression.

## === cell 39
pred = lr.predict(test_X)

pred = np.clip(pred, 0, 250)
pred = np.round(pred, 2)
