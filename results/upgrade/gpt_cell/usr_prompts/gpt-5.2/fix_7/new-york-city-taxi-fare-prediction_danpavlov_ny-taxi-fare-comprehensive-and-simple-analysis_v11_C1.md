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
geopy==2.4.1
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
from geopy.distance import great_circle
from sklearn import metrics, ensemble, linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 1
print(os.listdir("../input"))



## === cell 2
test = pd.read_csv("../input/test.csv")



## === cell 3
test.dtypes



## === cell 4
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}



## === cell 5
train = pd.read_csv("../input/train.csv", nrows=2000000, dtype=types)



## === cell 6
train.head()



## === cell 7
train.describe()



## === cell 8
sns.distplot(train["fare_amount"])



## === cell 9
sns.distplot(train["passenger_count"])



## === cell 10
train.isnull().sum()



## === cell 11
train.dropna(inplace=True)



## === cell 12
train = train[(train["fare_amount"] > 0) & (train["fare_amount"] < 250)]

train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]



## === cell 13
train.describe()




## === cell 14
def dist_calc(df):
    R = 6371.0088  # mean Earth radius in km
    lat1 = np.radians(df["pickup_latitude"].astype("float64").values)
    lon1 = np.radians(df["pickup_longitude"].astype("float64").values)
    lat2 = np.radians(df["dropoff_latitude"].astype("float64").values)
    lon2 = np.radians(df["dropoff_longitude"].astype("float64").values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    df["distance"] = (R * c).astype("float32")




## === cell 15
dist_calc(train)
dist_calc(test)




## === cell 16
def add_geo_features(df):
    df["abs_lon_diff"] = (
        (df["pickup_longitude"] - df["dropoff_longitude"]).abs().astype("float32")
    )
    df["abs_lat_diff"] = (
        (df["pickup_latitude"] - df["dropoff_latitude"]).abs().astype("float32")
    )
    df["manhattan_deg"] = (df["abs_lon_diff"] + df["abs_lat_diff"]).astype("float32")
    return df


train = add_geo_features(train)
test = add_geo_features(test)



## === cell 17
train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "")
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)



## === cell 18
test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "")
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)



## === cell 19
for df in (train, test):
    df["year"] = df.pickup_datetime.dt.year.astype("int16")
    df["month"] = df.pickup_datetime.dt.month.astype("int8")
    df["dayofweek"] = df.pickup_datetime.dt.dayofweek.astype("int8")
    df["hour"] = df.pickup_datetime.dt.hour.astype("int8")




## === cell 20
def add_airport_features(df):
    JFK = (-73.7781, 40.6413)
    LGA = (-73.8740, 40.7769)
    EWR = (-74.1745, 40.6895)
    MAN = (-73.9857, 40.7484)

    def haversine_km(lon1, lat1, lon2, lat2):
        R = 6371.0088
        lon1 = np.radians(lon1.astype("float64"))
        lat1 = np.radians(lat1.astype("float64"))
        lon2 = np.radians(lon2.astype("float64"))
        lat2 = np.radians(lat2.astype("float64"))
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        return (R * c).astype("float32")

    df["pickup_jfk_km"] = haversine_km(
        df["pickup_longitude"], df["pickup_latitude"], JFK[0], JFK[1]
    )
    df["dropoff_jfk_km"] = haversine_km(
        df["dropoff_longitude"], df["dropoff_latitude"], JFK[0], JFK[1]
    )
    df["pickup_lga_km"] = haversine_km(
        df["pickup_longitude"], df["pickup_latitude"], LGA[0], LGA[1]
    )
    df["dropoff_lga_km"] = haversine_km(
        df["dropoff_longitude"], df["dropoff_latitude"], LGA[0], LGA[1]
    )
    df["pickup_ewr_km"] = haversine_km(
        df["pickup_longitude"], df["pickup_latitude"], EWR[0], EWR[1]
    )
    df["dropoff_ewr_km"] = haversine_km(
        df["dropoff_longitude"], df["dropoff_latitude"], EWR[0], EWR[1]
    )
    df["pickup_man_km"] = haversine_km(
        df["pickup_longitude"], df["pickup_latitude"], MAN[0], MAN[1]
    )
    df["dropoff_man_km"] = haversine_km(
        df["dropoff_longitude"], df["dropoff_latitude"], MAN[0], MAN[1]
    )

    df["is_airport_trip"] = (
        (
            (df["pickup_jfk_km"] < 2.0)
            | (df["dropoff_jfk_km"] < 2.0)
            | (df["pickup_lga_km"] < 2.0)
            | (df["dropoff_lga_km"] < 2.0)
            | (df["pickup_ewr_km"] < 2.0)
            | (df["dropoff_ewr_km"] < 2.0)
        )
    ).astype("uint8")
    return df


train = add_airport_features(train)
test = add_airport_features(test)



## --- ERROR in cell 20, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/649082939.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     64[0m [0;34m[0m[0m
[1;32m     65[0m [0;34m[0m[0m
[0;32m---> 66[0;31m [0mtrain[0m [0;34m=[0m [0madd_airport_features[0m[0;34m([0m[0mtrain[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     67[0m [0mtest[0m [0;34m=[0m [0madd_airport_features[0m[0;34m([0m[0mtest[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     68[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/649082939.py[0m in [0;36madd_airport_features[0;34m(df)[0m
[1;32m     25[0m [0;34m[0m[0m
[1;32m     26[0m     [0;31m# Pickup/dropoff distance to airports and Manhattan center[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 27[0;31m     df["pickup_jfk_km"] = haversine_km(
[0m[1;32m     28[0m         [0mdf[0m[0;34m[[0m[0;34m"pickup_longitude"[0m[0;34m][0m[0;34m,[0m [0mdf[0m[0;34m[[0m[0;34m"pickup_latitude"[0m[0;34m][0m[0;34m,[0m [0mJFK[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m,[0m [0mJFK[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m     29[0m     )

[0;32m/tmp/ipykernel_11/649082939.py[0m in [0;36mhaversine_km[0;34m(lon1, lat1, lon2, lat2)[0m
[1;32m     13[0m         [0mlon1[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlon1[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0;34m"float64"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     14[0m         [0mlat1[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlat1[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0;34m"float64"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 15[0;31m         [0mlon2[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlon2[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0;34m"float64"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     16[0m         [0mlat2[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mradians[0m[0;34m([0m[0mlat2[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0;34m"float64"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     17[0m         [0mdlon[0m [0;34m=[0m [0mlon2[0m [0;34m-[0m [0mlon1[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'float' object has no attribute 'astype'

## === cell 21
test.head()
