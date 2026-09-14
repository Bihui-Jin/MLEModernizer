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

3.9

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
import numpy as np
import pandas as pd
import os

print(os.listdir("../input")[:20])


## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=10_000_000)
train_df.dtypes


## === cell 2
train_df.head()


## === cell 3
test_df = pd.read_csv("../input/test.csv")
test_df.dtypes


## === cell 4
test_df.head()




## === cell 5
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)


## === cell 6
print(train_df.isnull().sum())


## === cell 7
print(test_df.isnull().sum())


## === cell 8
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))


## === cell 9
print("Old size (pre-clean): %d" % len(train_df))

train_df = train_df[(train_df.fare_amount > 0) & (train_df.fare_amount <= 200)]

train_df = train_df[(train_df.passenger_count >= 1) & (train_df.passenger_count <= 6)]

train_df = train_df[
    (train_df.pickup_longitude.between(-74.5, -72.8))
    & (train_df.dropoff_longitude.between(-74.5, -72.8))
    & (train_df.pickup_latitude.between(40.5, 41.8))
    & (train_df.dropoff_latitude.between(40.5, 41.8))
]

train_df = train_df[
    (train_df.abs_diff_longitude > 0) | (train_df.abs_diff_latitude > 0)
]
train_df = train_df[
    (train_df.abs_diff_longitude < 2.0) & (train_df.abs_diff_latitude < 2.0)
]

print("New size (post-clean): %d" % len(train_df))


## === cell 10
try:
    plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")
except Exception as e:
    print("Plot skipped:", repr(e))


## === cell 11
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))


## === cell 12
print("Old size: %d" % len(test_df))
print("New size: %d (unchanged to preserve all keys)" % len(test_df))


## === cell 13
train_dt = pd.to_datetime(train_df["pickup_datetime"], errors="coerce")
test_dt = pd.to_datetime(test_df["pickup_datetime"], errors="coerce")

train_df["pickup_time"] = train_dt.dt.hour * 100 + train_dt.dt.minute
test_df["pickup_time"] = test_dt.dt.hour * 100 + test_dt.dt.minute

train_df["Weekday"] = train_dt.dt.dayofweek
test_df["Weekday"] = test_dt.dt.dayofweek


## === cell 14
train_df.head()


## === cell 15
test_df.head()


## === cell 16
train_df.head()


## === cell 17
test_df.head()


## === cell 18
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)


## === cell 19
train_df["Weekday"].replace(
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
test_df["Weekday"].replace(
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


## === cell 20
train_df.head()


## === cell 21
test_df.head()


## === cell 22
train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"])

train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)

missing_in_test = set(train_one_hot.columns) - set(test_one_hot.columns)
for col in missing_in_test:
    test_df[col] = 0
missing_in_train = set(test_one_hot.columns) - set(train_one_hot.columns)
for col in missing_in_train:
    train_df[col] = 0

weekday_cols = sorted(list(set(train_one_hot.columns) | set(test_one_hot.columns)))

train_df = train_df.drop(columns=["Weekday"])
test_df = test_df.drop(columns=["Weekday"])

train_df = train_df.reindex(
    columns=[c for c in train_df.columns if c not in weekday_cols] + weekday_cols
)
test_df = test_df.reindex(
    columns=[c for c in test_df.columns if c not in weekday_cols] + weekday_cols
)


## === cell 23
train_df.head()


## === cell 24
test_df.head()


## === cell 25
pass


## === cell 26
train_df["pickup_time"] = (
    pd.to_numeric(train_df["pickup_time"], errors="coerce").fillna(0).astype(int)
)
test_df["pickup_time"] = (
    pd.to_numeric(test_df["pickup_time"], errors="coerce").fillna(0).astype(int)
)


## === cell 27
train_df.head()


## === cell 28
test_df.head()


## === cell 29
R = 6373.0
lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_df["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(test_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_df["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_df["Distance"] = np.asarray(distance) * 0.621


## === cell 30
R = 6373.0
lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

lat3 = np.zeros(len(train_df["pickup_latitude"])) + np.radians(40.6413)
lon3 = np.zeros(len(train_df["pickup_longitude"])) + np.radians(-73.7781)
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
train_df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_df["dropoff_longitude"]))

lat3 = np.zeros(len(test_df["pickup_latitude"])) + np.radians(40.6413)
lon3 = np.zeros(len(test_df["pickup_latitude"])) + np.radians(-73.7781)
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
test_df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621


## === cell 31
print("Old size (pre-distance-clean): %d" % len(train_df))
train_df = train_df[(train_df["Distance"] >= 0) & (train_df["Distance"] <= 100)]
train_df = train_df[
    (train_df["Pickup_Distance_airport"] >= 0)
    & (train_df["Pickup_Distance_airport"] <= 150)
    & (train_df["Dropoff_Distance_airport"] >= 0)
    & (train_df["Dropoff_Distance_airport"] <= 150)
]
print("New size (post-distance-clean): %d" % len(train_df))


## === cell 32
before = len(train_df)
train_df = train_df[
    (train_df["abs_diff_longitude"] <= 1.5) & (train_df["abs_diff_latitude"] <= 1.5)
]
print(
    "Post leverage-control clean size: %d (removed %d rows)"
    % (len(train_df), before - len(train_df))
)


## === cell 33
before = len(train_df)
dist = train_df["Distance"].astype(float)

min_dist = 0.05  # miles
ppm = (train_df["fare_amount"].astype(float) / np.maximum(dist, min_dist)).astype(float)

train_df = train_df[(dist >= 0) & (dist <= 100)]
train_df = train_df[
    ((dist >= min_dist) & (ppm >= 1.0) & (ppm <= 50.0))
    | ((dist < min_dist) & (train_df["fare_amount"].astype(float) <= 30.0))
]
print(
    "Post fare-per-mile clean size: %d (removed %d rows)"
    % (len(train_df), before - len(train_df))
)


## === cell 34
log_fare = np.log1p(train_df["fare_amount"].astype(float))
q1, q3 = np.percentile(log_fare, [25, 75])
iqr = q3 - q1
low, high = q1 - 3.0 * iqr, q3 + 3.0 * iqr  # conservative fence to avoid over-pruning
before = len(train_df)
train_df = train_df[(log_fare >= low) & (log_fare <= high)]
print(
    "Post log-fare IQR-clean size: %d (removed %d rows)"
    % (len(train_df), before - len(train_df))
)


## === cell 35
train_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)


## === cell 36
abs_lon_mean = float(np.mean(train_df["abs_diff_longitude"]))
abs_lon_var = (
    float(np.var(train_df["abs_diff_longitude"]))
    if float(np.var(train_df["abs_diff_longitude"])) != 0.0
    else 1.0
)

train_df["abs_diff_longitude"] = np.abs(train_df["abs_diff_longitude"]) - abs_lon_mean
train_df["abs_diff_longitude"] = train_df["abs_diff_longitude"] / abs_lon_var


## === cell 37
abs_lat_mean = float(np.mean(train_df["abs_diff_latitude"]))
abs_lat_var = (
    float(np.var(train_df["abs_diff_latitude"]))
    if float(np.var(train_df["abs_diff_latitude"])) != 0.0
    else 1.0
)

train_df["abs_diff_latitude"] = np.abs(train_df["abs_diff_latitude"]) - abs_lat_mean
train_df["abs_diff_latitude"] = train_df["abs_diff_latitude"] / abs_lat_var


## === cell 38
test_df["abs_diff_longitude"] = np.abs(test_df["abs_diff_longitude"]) - abs_lon_mean
test_df["abs_diff_longitude"] = test_df["abs_diff_longitude"] / abs_lon_var


## === cell 39
test_df["abs_diff_latitude"] = np.abs(test_df["abs_diff_latitude"]) - abs_lat_mean
test_df["abs_diff_latitude"] = test_df["abs_diff_latitude"] / abs_lat_var


## === cell 40
_cont_cols = [
    "Distance",
    "Pickup_Distance_airport",
    "Dropoff_Distance_airport",
    "pickup_time",
]
_cont_means = {}
_cont_stds = {}
for c in _cont_cols:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
    test_df[c] = pd.to_numeric(test_df[c], errors="coerce")
    m = float(train_df[c].mean())
    s = float(train_df[c].std(ddof=0))
    if not np.isfinite(s) or s == 0.0:
        s = 1.0
    _cont_means[c] = m
    _cont_stds[c] = s
    train_df[c] = (train_df[c] - m) / s
    test_df[c] = (test_df[c] - m) / s


## === cell 41
train_df = train_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.replace([np.inf, -np.inf], np.nan)

train_df = train_df.dropna(axis=0, how="any")

test_df = test_df.fillna(0)


## === cell 42
train_df.shape


## === cell 43
test_df.shape


## === cell 44
train_df.head()


## === cell 45
from sklearn.model_selection import train_test_split

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]

test_X = test_df.drop(["key"], axis=1)

test_X = test_X.reindex(columns=X.columns, fill_value=0)
X = X.reindex(columns=test_X.columns, fill_value=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)


## === cell 46
from sklearn.linear_model import HuberRegressor

y_train_log = np.log1p(y_train.astype(float))

from sklearn.linear_model import LinearRegression

_lr = LinearRegression()
_lr.fit(X_train, y_train_log)
_resid = y_train_log.values - _lr.predict(X_train)
_scale = float(np.median(np.abs(_resid)) / 0.6745)  # robust MAD->sigma
if not np.isfinite(_scale) or _scale <= 1e-6:
    _scale = float(np.std(_resid)) if float(np.std(_resid)) > 1e-6 else 1.0

hr = HuberRegressor(max_iter=1000, epsilon=1.35, alpha=1e-4, scale=_scale)
hr.fit(X_train, y_train_log)

val_pred = np.expm1(hr.predict(X_test))
val_pred = np.clip(val_pred, 0, float(train_df["fare_amount"].max()))
rmse = float(np.sqrt(np.mean((val_pred - y_test.values) ** 2)))
print("Validation RMSE (original space):", rmse)
print("Using Huber scale (log-space):", _scale)


## --- ERROR in cell 46, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3578770414.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     16[0m     [0m_scale[0m [0;34m=[0m [0mfloat[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mstd[0m[0;34m([0m[0m_resid[0m[0;34m)[0m[0;34m)[0m [0;32mif[0m [0mfloat[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mstd[0m[0;34m([0m[0m_resid[0m[0;34m)[0m[0;34m)[0m [0;34m>[0m [0;36m1e-6[0m [0;32melse[0m [0;36m1.0[0m[0;34m[0m[0;34m[0m[0m
[1;32m     17[0m [0;34m[0m[0m
[0;32m---> 18[0;31m [0mhr[0m [0;34m=[0m [0mHuberRegressor[0m[0;34m([0m[0mmax_iter[0m[0;34m=[0m[0;36m1000[0m[0;34m,[0m [0mepsilon[0m[0;34m=[0m[0;36m1.35[0m[0;34m,[0m [0malpha[0m[0;34m=[0m[0;36m1e-4[0m[0;34m,[0m [0mscale[0m[0;34m=[0m[0m_scale[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     19[0m [0mhr[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX_train[0m[0;34m,[0m [0my_train_log[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m [0;34m[0m[0m

[0;31mTypeError[0m: HuberRegressor.__init__() got an unexpected keyword argument 'scale'

## === cell 47
pred = np.expm1(hr.predict(test_X))
pred = np.clip(pred, 0, float(train_df["fare_amount"].max()))
pred = pred.astype(float)
