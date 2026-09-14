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
import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

Debug = False

NROWS = 10000000
if Debug is True: NROWS = 5000
train = pd.read_csv('../input/train.csv', nrows = NROWS, index_col = "key")
train = train.dropna()
test_df = pd.read_csv('../input/test.csv', index_col = "key")
testdex = test_df.index


## === cell 1
print("Percent of Training Set with Zero and Below Fair: ", round(((train.loc[train["fare_amount"] <= 0, "fare_amount"].shape[0]/train.shape[0]) * 100),5))
print("Percent of Training Set 200 and Above Fair: ", round((train.loc[train["fare_amount"] >= 200, "fare_amount"].shape[0]/train.shape[0]) * 100,5))
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] <= 200),:]
print("\nPercent of Training Set with Zero and Below Passenger Count: ", round((train.loc[train["passenger_count"] <= 0, "passenger_count"].shape[0]/train.shape[0]) * 100,5))
print("Percent of Training Set with Nine and Above Passenger Count: ", round((train.loc[train["passenger_count"] >= 9, "passenger_count"].shape[0]/train.shape[0]) * 100,5))
train = train.loc[(train["passenger_count"] > 0) & (train["passenger_count"] <= 9),:]


## === cell 2
def clean_df(df):
    return df[(df.fare_amount > 0) & 
            (df.pickup_longitude > -80) & (df.pickup_longitude < -70) &
            (df.pickup_latitude > 35) & (df.pickup_latitude < 45) &
            (df.dropoff_longitude > -80) & (df.dropoff_longitude < -70) &
            (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45)]


## === cell 3
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
    df['day_of_year'] = df.pickup_datetime.dt.dayofyear
    df['week_of_year'] = df.pickup_datetime.dt.weekofyear
    df["Weekday"] = df.pickup_datetime.dt.weekday
    df["Quarter"] = df.pickup_datetime.dt.quarter
    df["Day of Month"] = df.pickup_datetime.dt.day
    
    return df


## === cell 4
def prepare_distance_features(df):
    df["longitude_distance"] = abs(df["pickup_longitude"] - df["dropoff_longitude"])
    df["latitude_distance"] = abs(df["pickup_latitude"] - df["dropoff_latitude"])

    df["distance_travelled"] = (
        df["longitude_distance"] ** 2 + df["latitude_distance"] ** 2
    ) ** 0.5
    df["distance_travelled_sin"] = np.sin(
        (df["longitude_distance"] ** 2 * df["latitude_distance"] ** 2) ** 0.5
    )
    df["distance_travelled_cos"] = np.cos(
        (df["longitude_distance"] ** 2 * df["latitude_distance"] ** 2) ** 0.5
    )
    df["distance_travelled_sin_sqrd"] = (
        np.sin((df["longitude_distance"] ** 2 * df["latitude_distance"] ** 2) ** 0.5)
        ** 2
    )
    df["distance_travelled_cos_sqrd"] = (
        np.cos((df["longitude_distance"] ** 2 * df["latitude_distance"] ** 2) ** 0.5)
        ** 2
    )

    R = 6371e3  # Metres
    phi1 = np.radians(df["pickup_latitude"])
    phi2 = np.radians(df["dropoff_latitude"])
    phi_chg = np.radians(df["pickup_latitude"] - df["dropoff_latitude"])
    delta_chg = np.radians(df["pickup_longitude"] - df["dropoff_longitude"])
    a = np.sin(phi_chg / 2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_chg / 2)
    c = 2 * np.arctan2(a**0.5, (1 - a) ** 0.5)
    d = R * c
    df["haversine"] = d

    y = np.sin(delta_chg * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    df["bearing"] = np.arctan2(y, x)

    return df


def prepare_time_features(df):
    df["pickup_datetime"] = df["pickup_datetime"].str.replace(" UTC", "")
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
    )
    df["hour_of_day"] = df.pickup_datetime.dt.hour

    iso_week = df.pickup_datetime.dt.isocalendar().week.astype(np.int16)
    df["week"] = iso_week
    df["month"] = df.pickup_datetime.dt.month
    df["day_of_year"] = df.pickup_datetime.dt.dayofyear
    df["week_of_year"] = iso_week
    df["Weekday"] = df.pickup_datetime.dt.weekday
    df["Quarter"] = df.pickup_datetime.dt.quarter
    df["Day of Month"] = df.pickup_datetime.dt.day

    return df


## === cell 5
f, ax = plt.subplots(1,2,figsize = [10,5])
sns.countplot(train["passenger_count"], ax=ax[0])
sns.countplot(test_df["passenger_count"], ax=ax[1])
ax[0].set_title("Train Set - Passenger Count")
ax[1].set_title("Test Set - Passenger Count")
plt.show()


## === cell 6
f, ax = plt.subplots(figsize=[6,5])
sns.kdeplot(train["fare_amount"], ax=ax)
ax.set_title("Fare Distribution")
plt.show()


## === cell 7
def time_slicer(df, timeframes, value, color="purple"):
    """
    Function to count observation occurrence through different lenses of time.
    """
    f, ax = plt.subplots(len(timeframes), figsize = [12,10])
    for i,x in enumerate(timeframes):
        df.loc[:,[x,value]].groupby([x]).mean().plot(ax=ax[i],color=color)
        ax[i].set_ylabel(value.replace("_", " ").title())
        ax[i].set_title("{} by {}".format(value.replace("_", " ").title(), x.replace("_", " ").title()))
        ax[i].set_xlabel("")
    ax[len(timeframes)-1].set_xlabel("Time Frame")
    plt.tight_layout(pad=0)


## === cell 8
time_slicer(df=train, timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"], value = "fare_amount", color="blue")


## --- ERROR in cell 8, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1570644381.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mtime_slicer[0m[0;34m([0m[0mdf[0m[0;34m=[0m[0mtrain[0m[0;34m,[0m [0mtimeframes[0m[0;34m=[0m[0;34m[[0m[0;34m"day_of_year"[0m[0;34m,[0m [0;34m"month"[0m[0;34m,[0m [0;34m"Day of Month"[0m[0;34m,[0m [0;34m"week"[0m[0;34m,[0m [0;34m"hour_of_day"[0m[0;34m][0m[0;34m,[0m [0mvalue[0m [0;34m=[0m [0;34m"fare_amount"[0m[0;34m,[0m [0mcolor[0m[0;34m=[0m[0;34m"blue"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/2575660034.py[0m in [0;36mtime_slicer[0;34m(df, timeframes, value, color)[0m
[1;32m      5[0m     [0mf[0m[0;34m,[0m [0max[0m [0;34m=[0m [0mplt[0m[0;34m.[0m[0msubplots[0m[0;34m([0m[0mlen[0m[0;34m([0m[0mtimeframes[0m[0;34m)[0m[0;34m,[0m [0mfigsize[0m [0;34m=[0m [0;34m[[0m[0;36m12[0m[0;34m,[0m[0;36m10[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m     [0;32mfor[0m [0mi[0m[0;34m,[0m[0mx[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0mtimeframes[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 7[0;31m         [0mdf[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m[0;34m[[0m[0mx[0m[0;34m,[0m[0mvalue[0m[0;34m][0m[0;34m][0m[0;34m.[0m[0mgroupby[0m[0;34m([0m[0;34m[[0m[0mx[0m[0;34m][0m[0;34m)[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mplot[0m[0;34m([0m[0max[0m[0;34m=[0m[0max[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m,[0m[0mcolor[0m[0;34m=[0m[0mcolor[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m         [0max[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m.[0m[0mset_ylabel[0m[0;34m([0m[0mvalue[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m"_"[0m[0;34m,[0m [0;34m" "[0m[0;34m)[0m[0;34m.[0m[0mtitle[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m         [0max[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m.[0m[0mset_title[0m[0;34m([0m[0;34m"{} by {}"[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mvalue[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m"_"[0m[0;34m,[0m [0;34m" "[0m[0;34m)[0m[0;34m.[0m[0mtitle[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0mx[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m"_"[0m[0;34m,[0m [0;34m" "[0m[0;34m)[0m[0;34m.[0m[0mtitle[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m__getitem__[0;34m(self, key)[0m
[1;32m   1182[0m             [0;32mif[0m [0mself[0m[0;34m.[0m[0m_is_scalar_access[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1183[0m                 [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mobj[0m[0;34m.[0m[0m_get_value[0m[0;34m([0m[0;34m*[0m[0mkey[0m[0;34m,[0m [0mtakeable[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0m_takeable[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1184[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_getitem_tuple[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1185[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1186[0m             [0;31m# we by definition only have the 0th axis[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_getitem_tuple[0;34m(self, tup)[0m
[1;32m   1375[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_multi_take[0m[0;34m([0m[0mtup[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1376[0m [0;34m[0m[0m
[0;32m-> 1377[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_getitem_tuple_same_dim[0m[0;34m([0m[0mtup[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1378[0m [0;34m[0m[0m
[1;32m   1379[0m     [0;32mdef[0m [0m_get_label[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mlabel[0m[0;34m,[0m [0maxis[0m[0;34m:[0m [0mAxisInt[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_getitem_tuple_same_dim[0;34m(self, tup)[0m
[1;32m   1018[0m                 [0;32mcontinue[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1019[0m [0;34m[0m[0m
[0;32m-> 1020[0;31m             [0mretval[0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0mretval[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mname[0m[0;34m)[0m[0;34m.[0m[0m_getitem_axis[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0mi[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1021[0m             [0;31m# We should never have retval.ndim < self.ndim, as that should[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1022[0m             [0;31m#  be handled by the _getitem_lowerdim call above.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_getitem_axis[0;34m(self, key, axis)[0m
[1;32m   1418[0m                     [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"Cannot index with multidimensional key"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1419[0m [0;34m[0m[0m
[0;32m-> 1420[0;31m                 [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_getitem_iterable[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1421[0m [0;34m[0m[0m
[1;32m   1422[0m             [0;31m# nested tuple slicing[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_getitem_iterable[0;34m(self, key, axis)[0m
[1;32m   1358[0m [0;34m[0m[0m
[1;32m   1359[0m         [0;31m# A collection of keys[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1360[0;31m         [0mkeyarr[0m[0;34m,[0m [0mindexer[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_get_listlike_indexer[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1361[0m         return self.obj._reindex_with_indexers(
[1;32m   1362[0m             [0;34m{[0m[0maxis[0m[0;34m:[0m [0;34m[[0m[0mkeyarr[0m[0;34m,[0m [0mindexer[0m[0;34m][0m[0;34m}[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0mallow_dups[0m[0;34m=[0m[0;32mTrue[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_get_listlike_indexer[0;34m(self, key, axis)[0m
[1;32m   1556[0m         [0maxis_name[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mobj[0m[0;34m.[0m[0m_get_axis_name[0m[0;34m([0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1557[0m [0;34m[0m[0m
[0;32m-> 1558[0;31m         [0mkeyarr[0m[0;34m,[0m [0mindexer[0m [0;34m=[0m [0max[0m[0;34m.[0m[0m_get_indexer_strict[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1559[0m [0;34m[0m[0m
[1;32m   1560[0m         [0;32mreturn[0m [0mkeyarr[0m[0;34m,[0m [0mindexer[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36m_get_indexer_strict[0;34m(self, key, axis_name)[0m
[1;32m   6198[0m             [0mkeyarr[0m[0;34m,[0m [0mindexer[0m[0;34m,[0m [0mnew_indexer[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_reindex_non_unique[0m[0;34m([0m[0mkeyarr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   6199[0m [0;34m[0m[0m
[0;32m-> 6200[0;31m         [0mself[0m[0;34m.[0m[0m_raise_if_missing[0m[0;34m([0m[0mkeyarr[0m[0;34m,[0m [0mindexer[0m[0;34m,[0m [0maxis_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6201[0m [0;34m[0m[0m
[1;32m   6202[0m         [0mkeyarr[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mtake[0m[0;34m([0m[0mindexer[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36m_raise_if_missing[0;34m(self, key, indexer, axis_name)[0m
[1;32m   6250[0m [0;34m[0m[0m
[1;32m   6251[0m             [0mnot_found[0m [0;34m=[0m [0mlist[0m[0;34m([0m[0mensure_index[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[[0m[0mmissing_mask[0m[0;34m.[0m[0mnonzero[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m][0m[0;34m.[0m[0munique[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 6252[0;31m             [0;32mraise[0m [0mKeyError[0m[0;34m([0m[0;34mf"{not_found} not in index"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6253[0m [0;34m[0m[0m
[1;32m   6254[0m     [0;34m@[0m[0moverload[0m[0;34m[0m[0;34m[0m[0m

[0;31mKeyError[0m: "['day_of_year'] not in index"

## === cell 9
time_slicer(df=train, timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"], value = "distance_travelled", color = "green")
