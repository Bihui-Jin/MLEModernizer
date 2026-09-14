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
lightgbm==4.6.0
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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
plt.style.use('fivethirtyeight')

import geopy.distance


import os
print(os.listdir("../input"))
import gc



## === cell 1
def load_Data():
    train = pd.read_csv("../input/train.csv", nrows=10_00_000, low_memory=True)
    test = pd.read_csv("../input/test.csv", nrows=10_00_000, low_memory=True)
    return train,test


def prepare_distance_features(df):
    df['longitude_distance'] = abs(df['pickup_longitude'] - df['dropoff_longitude'])
    df['latitude_distance'] = abs(df['pickup_latitude'] - df['dropoff_latitude'])

    df['distance_travelled'] = (df['longitude_distance'] ** 2 + df['latitude_distance'] ** 2) ** .5
    df['distance_travelled_sin'] = np.sin((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5)
    df['distance_travelled_cos'] = np.cos((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5)
    df['distance_travelled_sin_sqrd'] = np.sin((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5) ** 2
    df['distance_travelled_cos_sqrd'] = np.cos((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5) ** 2

    R = 6371e3 # Metres
    phi1 = np.radians(df['pickup_latitude'])
    phi2 = np.radians(df['dropoff_latitude'])
    phi_chg = np.radians(df['pickup_latitude'] - df['dropoff_latitude'])
    delta_chg = np.radians(df['pickup_longitude'] - df['dropoff_longitude'])
    a = np.sin(phi_chg / 2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_chg / 2)
    c = 2 * np.arctan2(a ** .5, (1-a) ** .5)
    d = R * c
    df['haversine'] = d

    y = np.sin(delta_chg * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    df['bearing'] = np.arctan2(y, x)

    return df

def prepare_time_features(df):
    df['pickup_datetime'] = df['pickup_datetime'].str.replace(" UTC", "")
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
    df['hour_of_day'] = df.pickup_datetime.dt.hour
    df['week'] = df.pickup_datetime.dt.week
    df['month'] = df.pickup_datetime.dt.month
    df["year"] = df.pickup_datetime.dt.year
    df['day_of_year'] = df.pickup_datetime.dt.dayofyear
    df['week_of_year'] = df.pickup_datetime.dt.weekofyear
    df["weekday"] = df.pickup_datetime.dt.weekday
    df["quarter"] = df.pickup_datetime.dt.quarter
    df["day_of_month"] = df.pickup_datetime.dt.day
    
    return df


## === cell 2
train, test = load_Data()


## === cell 3
def prepare_time_features(df):
    df["pickup_datetime"] = df["pickup_datetime"].str.replace(" UTC", "")
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
    )
    df["hour_of_day"] = df.pickup_datetime.dt.hour
    df["week"] = df.pickup_datetime.dt.isocalendar().week.astype(int)
    df["month"] = df.pickup_datetime.dt.month
    df["year"] = df.pickup_datetime.dt.year
    df["day_of_year"] = df.pickup_datetime.dt.dayofyear
    df["week_of_year"] = df.pickup_datetime.dt.isocalendar().week.astype(int)
    df["weekday"] = df.pickup_datetime.dt.weekday
    df["quarter"] = df.pickup_datetime.dt.quarter
    df["day_of_month"] = df.pickup_datetime.dt.day

    return df


train = prepare_time_features(train)
train = prepare_distance_features(train)

test = prepare_time_features(test)
test = prepare_distance_features(test)


## === cell 4
train.describe()


## === cell 5
train.info()


## === cell 6
train.isnull().sum()


## === cell 7
train = train.fillna(0)


## === cell 8
train['key2'] = pd.to_datetime(train['key'], errors='coerce')
train['key2'].head()
train.info()


## === cell 9
test['key2'] = pd.to_datetime(test['key'], errors='coerce')


## === cell 10
train['fare_amount'].plot(kind='box')


## === cell 11
gc.collect()
train.describe()


## === cell 12
print("% of fares above 25$ - {:0.2f}".format(train[train['fare_amount'] > 25]['key'].count()*100/train['key'].count()))
print("% of fares above 50$ - {:0.2f}".format(train[train['fare_amount'] > 50]['key'].count()*100/train['key'].count()))
print("% of fares above 100$ - {:0.2f}".format(train[train['fare_amount'] > 100]['key'].count()*100/train['key'].count()))
print("% of fares below 0$ - {:0.2f}".format(train[train['fare_amount'] < 0]['key'].count()*100/train['key'].count()))


## === cell 13

fig, axarr = plt.subplots(2, 2, figsize=(20, 10))

train[~(train['fare_amount'] > 25)]['fare_amount'].plot(kind="box",ax=axarr[0][0])
train[~(train['fare_amount'] > 50)]['fare_amount'].plot(kind="box",ax=axarr[0][1])
train[~(train['fare_amount'] > 100)]['fare_amount'].plot(kind="box",ax=axarr[1][0])
train[~(train['fare_amount'] < 0 )]['fare_amount'].plot(kind='box',ax = axarr[1][1])


## === cell 14
train['passenger_count'].plot(kind='box')


## === cell 15
print("Count of invalid pickup latitude", train[(train['pickup_latitude'] > 90) | (train['pickup_latitude'] < -90) ]['pickup_latitude'].count())
print("Count of invalid dropoff latitude", train[(train['dropoff_latitude'] > 90) | (train['dropoff_latitude'] < -90) ]['dropoff_latitude'].count())
print("Count of invalid pickup longitude", train[(train['pickup_longitude'] > 180) | (train['pickup_longitude'] < -180) ]['pickup_longitude'].count())
print("Count of invalid dropoff longitude", train[(train['dropoff_longitude'] > 180) | (train['dropoff_longitude'] < -180) ]['dropoff_longitude'].count())


## === cell 16
print("Count of invalid pickup latitude", test[(test['pickup_latitude'] > 90) | (test['pickup_latitude'] < -90) ]['pickup_latitude'].count())
print("Count of invalid dropoff latitude", test[(test['dropoff_latitude'] > 90) | (test['dropoff_latitude'] < -90) ]['dropoff_latitude'].count())
print("Count of invalid pickup longitude", test[(test['pickup_longitude'] > 180) | (test['pickup_longitude'] < -180) ]['pickup_longitude'].count())
print("Count of invalid dropoff longitude", test[(test['dropoff_longitude'] > 180) | (test['dropoff_longitude'] < -180) ]['dropoff_longitude'].count())


## === cell 17
train = train[~((train['pickup_latitude'] > 90) | (train['pickup_latitude'] < -90))]
train = train[~((train['dropoff_latitude'] > 90) | (train['dropoff_latitude'] < -90))]
train = train[~((train['pickup_longitude'] > 180) | (train['pickup_longitude'] < -180))]
train = train[~((train['dropoff_longitude'] > 180) | (train['dropoff_longitude'] < -180))]


## === cell 18
train["distance"] = train[
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
].apply(
    lambda x: geopy.distance.geodesic(
        (x["pickup_latitude"], x["pickup_longitude"]),
        (x["dropoff_latitude"], x["dropoff_longitude"]),
    ).km,
    axis=1,
)


## === cell 19
train.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'],axis=1, inplace=True)


## === cell 20
test['distance']=test[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
                                                                                                geopy.distance.VincentyDistance((x['pickup_latitude'],x['pickup_longitude']),
                                                                                                                               (x['dropoff_latitude'],x['dropoff_longitude'])).km,axis=1)


## --- ERROR in cell 20, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3547695418.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m test['distance']=test[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
[0m[1;32m      2[0m                                                                                                 geopy.distance.VincentyDistance((x['pickup_latitude'],x['pickup_longitude']),
[1;32m      3[0m                                                                                                                                (x['dropoff_latitude'],x['dropoff_longitude'])).km,axis=1)

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36mapply[0;34m(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)[0m
[1;32m  10372[0m             [0mkwargs[0m[0;34m=[0m[0mkwargs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m  10373[0m         )
[0;32m> 10374[0;31m         [0;32mreturn[0m [0mop[0m[0;34m.[0m[0mapply[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__finalize__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mmethod[0m[0;34m=[0m[0;34m"apply"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m  10375[0m [0;34m[0m[0m
[1;32m  10376[0m     def map(

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py[0m in [0;36mapply[0;34m(self)[0m
[1;32m    914[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mapply_raw[0m[0;34m([0m[0mengine[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mengine[0m[0;34m,[0m [0mengine_kwargs[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mengine_kwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    915[0m [0;34m[0m[0m
[0;32m--> 916[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mapply_standard[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    917[0m [0;34m[0m[0m
[1;32m    918[0m     [0;32mdef[0m [0magg[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py[0m in [0;36mapply_standard[0;34m(self)[0m
[1;32m   1061[0m     [0;32mdef[0m [0mapply_standard[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1062[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mengine[0m [0;34m==[0m [0;34m"python"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1063[0;31m             [0mresults[0m[0;34m,[0m [0mres_index[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mapply_series_generator[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1064[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1065[0m             [0mresults[0m[0;34m,[0m [0mres_index[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mapply_series_numba[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py[0m in [0;36mapply_series_generator[0;34m(self)[0m
[1;32m   1079[0m             [0;32mfor[0m [0mi[0m[0;34m,[0m [0mv[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0mseries_gen[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1080[0m                 [0;31m# ignore SettingWithCopy here in case the user mutates[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1081[0;31m                 [0mresults[0m[0;34m[[0m[0mi[0m[0;34m][0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mfunc[0m[0;34m([0m[0mv[0m[0;34m,[0m [0;34m*[0m[0mself[0m[0;34m.[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mself[0m[0;34m.[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1082[0m                 [0;32mif[0m [0misinstance[0m[0;34m([0m[0mresults[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m,[0m [0mABCSeries[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1083[0m                     [0;31m# If we have a view on v, we need to make a copy because[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3547695418.py[0m in [0;36m<lambda>[0;34m(x)[0m
[1;32m      1[0m test['distance']=test[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
[0;32m----> 2[0;31m                                                                                                 geopy.distance.VincentyDistance((x['pickup_latitude'],x['pickup_longitude']),
[0m[1;32m      3[0m                                                                                                                                (x['dropoff_latitude'],x['dropoff_longitude'])).km,axis=1)

[0;31mAttributeError[0m: module 'geopy.distance' has no attribute 'VincentyDistance'

## === cell 21
test.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'],axis=1, inplace=True)
