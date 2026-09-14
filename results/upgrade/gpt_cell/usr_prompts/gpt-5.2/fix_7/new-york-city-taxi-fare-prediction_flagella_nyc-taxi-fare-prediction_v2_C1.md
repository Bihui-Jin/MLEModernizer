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

3.11

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

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE



## === cell 2
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1000000
)
test_raw = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
train.shape, test_raw.shape



## === cell 3
train.head()



## === cell 4
train.isnull().sum()



## === cell 5
train = train.dropna(axis="rows")
test_raw.isnull().sum()



## === cell 6
train.head()



## === cell 7
train["fare_amount"].describe()



## === cell 8
train = train[(train["fare_amount"] >= 2.5) & (train["fare_amount"] <= 250)].copy()

train.drop(train[train["pickup_longitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["pickup_latitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["dropoff_longitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["dropoff_latitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["passenger_count"] > 5].index, axis=0, inplace=True)
train.drop(train[train["passenger_count"] == 0].index, axis=0, inplace=True)




## === cell 9
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def add_features(df):
    df = df.copy()
    df["haversine_km"] = haversine_km(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    )
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["pickup_hour"] = dt.dt.hour.astype("float32")
    df["pickup_dayofweek"] = dt.dt.dayofweek.astype("float32")
    return df


train = add_features(train)



## === cell 10
train.drop(["key"], axis=1, inplace=True)



## === cell 11
train.drop(["pickup_datetime"], axis=1, inplace=True)



## === cell 12
train.dropna(inplace=True)

train["pickup_longitude"] = train["pickup_longitude"].clip(-75, -72)
train["dropoff_longitude"] = train["dropoff_longitude"].clip(-75, -72)
train["pickup_latitude"] = train["pickup_latitude"].clip(40, 42)
train["dropoff_latitude"] = train["dropoff_latitude"].clip(40, 42)

train["haversine_km"] = haversine_km(
    train["pickup_longitude"].values,
    train["pickup_latitude"].values,
    train["dropoff_longitude"].values,
    train["dropoff_latitude"].values,
)

train = train[(train["haversine_km"] >= 0.05) & (train["haversine_km"] <= 100)].copy()



## === cell 13
X, y = train.drop("fare_amount", axis=1), train["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12
)

scale_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "haversine_km",
    "pickup_hour",
    "pickup_dayofweek",
]
pas_col = ["passenger_count"]

scaler = StandardScaler()
X_train_scaled_cont = scaler.fit_transform(X_train[scale_cols])
X_test_scaled_cont = scaler.transform(X_test[scale_cols])

X_train_scaled = np.hstack([X_train_scaled_cont, X_train[pas_col].to_numpy()])
X_test_scaled = np.hstack([X_test_scaled_cont, X_test[pas_col].to_numpy()])



## === cell 14
xgb_r = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    gamma=0.0,
    n_jobs=-1,
    random_state=123,
)
xgb_r.fit(X_train_scaled, y_train)



## === cell 15
y_pred = xgb_r.predict(X_test_scaled)
rmse = np.sqrt(MSE(y_test, y_pred))
print("RMSE : % f" % (rmse))



## === cell 16
test = test_raw.copy()
test = add_features(test)

feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "haversine_km",
    "pickup_hour",
    "pickup_dayofweek",
]

train_feature_medians = X[feature_cols].median(numeric_only=True)
test[feature_cols] = test[feature_cols].fillna(train_feature_medians)

test["pickup_longitude"] = test["pickup_longitude"].clip(-75, -72)
test["dropoff_longitude"] = test["dropoff_longitude"].clip(-75, -72)
test["pickup_latitude"] = test["pickup_latitude"].clip(40, 42)
test["dropoff_latitude"] = test["dropoff_latitude"].clip(40, 42)
test["passenger_count"] = test["passenger_count"].clip(lower=1, upper=5)

test["haversine_km"] = haversine_km(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
).clip(lower=0.0, upper=100.0)

test.shape



## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2707880348.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     30[0m     [0mtest[0m[0;34m[[0m[0;34m"dropoff_longitude"[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     31[0m     [0mtest[0m[0;34m[[0m[0;34m"dropoff_latitude"[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 32[0;31m ).clip(lower=0.0, upper=100.0)
[0m[1;32m     33[0m [0;34m[0m[0m
[1;32m     34[0m [0mtest[0m[0;34m.[0m[0mshape[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/core/_methods.py[0m in [0;36m_clip[0;34m(a, min, max, out, **kwargs)[0m
[1;32m     90[0m [0;32mdef[0m [0m_clip[0m[0;34m([0m[0ma[0m[0;34m,[0m [0mmin[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mmax[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mout[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     91[0m     [0;32mif[0m [0mmin[0m [0;32mis[0m [0;32mNone[0m [0;32mand[0m [0mmax[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 92[0;31m         [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"One of max or min must be given"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     93[0m [0;34m[0m[0m
[1;32m     94[0m     [0;32mif[0m [0mmin[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: One of max or min must be given

## === cell 17
test_keys = test["key"].copy()
X_sub = test.drop(["key", "pickup_datetime"], axis=1)

X_sub_scaled_cont = scaler.transform(X_sub[scale_cols])
X_sub_scaled = np.hstack([X_sub_scaled_cont, X_sub[pas_col].to_numpy()])
