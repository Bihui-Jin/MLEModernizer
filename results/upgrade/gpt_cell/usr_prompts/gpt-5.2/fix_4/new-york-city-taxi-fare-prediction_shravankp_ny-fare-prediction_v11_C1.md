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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("/kaggle/input"))

import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import math


## === cell 1
train = pd.read_csv('../input/train.csv', nrows=1000000)
test = pd.read_csv('../input/test.csv')
train.head()


## === cell 2
len(train['fare_amount'].unique())


## === cell 3
train.isnull().sum()


## === cell 4
train = train.dropna(how='any',axis=0)


## === cell 5
train['abs_diff_longitude'] = np.abs(train['dropoff_longitude'] - train['pickup_longitude'])
train['abs_diff_latitude'] = np.abs(train['dropoff_latitude'] - train['pickup_latitude'])
test['abs_diff_longitude'] = np.abs(test['dropoff_longitude'] - test['pickup_longitude'])
test['abs_diff_latitude'] = np.abs(test['dropoff_latitude'] - test['pickup_latitude'])


## === cell 7
train = train.loc[train['fare_amount']>0,:]
train = train.loc[(train["passenger_count"]<=6) & (train["passenger_count"]>0),:]
train = train.loc[(train["abs_diff_latitude"]<2) & (train["abs_diff_longitude"]<2),:]
train = train.loc[(train["abs_diff_latitude"]>0) & (train["abs_diff_longitude"]>0),:]


## === cell 9
train.loc[:,'timestamp_with_key'] = train.loc[:,'key'] 
test.loc[:,'timestamp_with_key'] = test.loc[:,'key']
train.key = pd.DataFrame({'key':train['key'].str.split('.').str[1].astype('int')})
test.key = pd.DataFrame({'key':test['key'].str.split('.').str[1].astype('int')})


## === cell 10
from math import floor


def chooseSlot(x):
    hr = x.hour
    return int(hr / 3 + 1)


train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], utc=True, errors="raise"
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], utc=True, errors="raise"
)

train["time_slot"] = pd.DataFrame(
    list(map(lambda x: chooseSlot(x), train["pickup_datetime"][:])), index=train.index
)
test["time_slot"] = pd.DataFrame(
    list(map(lambda x: chooseSlot(x), test["pickup_datetime"][:])), index=test.index
)


## === cell 11
train["weekday_no"] = ((train["pickup_datetime"].dt.weekday + 1) % 7).astype(int)
test["weekday_no"] = ((test["pickup_datetime"].dt.weekday + 1) % 7).astype(int)


## === cell 14
def dist_haversine(x):
    R = 6371 #for metres 6371e3
    picklat = math.radians(x[1])
    droplat = math.radians(x[3])
    latdiff = abs(droplat-picklat)
    picklon = math.radians(x[0])
    droplon = math.radians(x[2])
    londiff = abs(droplon-picklon)

    a = math.sin(latdiff/2) * math.sin(latdiff/2) +\
            math.cos(picklat) * math.cos(droplat) *\
            math.sin(londiff/2) * math.sin(londiff/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return (R * c)

train['dist_haversine_km'] = pd.DataFrame(list(
    map(lambda x: dist_haversine(x), train[["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]].values)),
                                      index=train.index)
test['dist_haversine_km'] = pd.DataFrame(list(
    map(lambda x: dist_haversine(x), test[["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]].values)),
                                      index=test.index)


## === cell 15
train['fare_per_km'] = train['fare_amount']/(train["dist_haversine_km"])
train['fare_per_km_passenger'] = train['fare_amount']/(train['dist_haversine_km']*train['passenger_count'])


## === cell 16
train.groupby('key').agg({'fare_per_km_passenger':'mean','key':'count','passenger_count':'mean','fare_amount':'mean'})


## === cell 17
train.loc[train['fare_per_km_passenger']>20,['fare_per_km_passenger','fare_amount','dist_haversine_km']]


## === cell 18
grouped_df = train.groupby('key')
count = 0
for key, item in grouped_df:
    count += 1
    if count == 2: ## to view key = 2
        filtered = grouped_df.get_group(key)["dist_haversine_km"]>1 #ignoring drives within 1km
        df = pd.DataFrame(grouped_df.get_group(key).loc[filtered,:].sort_values(by='pickup_datetime'))
        break


## === cell 19
indexes = ['key',df['pickup_datetime'].dt.strftime('%a'),'time_slot']
grouped = df[:][:].groupby(indexes).agg({'fare_per_km_passenger':'mean','time_slot':'count'})
grouped.rename(columns={'time_slot':'count'},inplace=True)
grouped


## === cell 20
reindexed = grouped.reset_index().drop('key',axis=1)
get_max_count = reindexed.groupby(['pickup_datetime']).agg({'count':'max'})
get_max_count = get_max_count.reindex(reindexed['pickup_datetime'], method='ffill')
reindexed = reindexed .set_index('pickup_datetime')
reindexed.loc[get_max_count['count'] == reindexed['count'],:]


## === cell 21
train = train.loc[ ~ ((train['fare_per_km']<0.2) & (train['dist_haversine_km']>1))]
train = train.loc[~ ((train['dist_haversine_km']<0.01) & (train['fare_per_km']>50))]


## === cell 22
print(train.shape[0] - train.loc[train['pickup_latitude'].between(39,42) | train['dropoff_latitude'].between(39,42) |
          train['pickup_longitude'].between(-74.4,-72.8) |  train['dropoff_longitude'].between(-74.4,-72.8)].shape[0])
train = train.loc[train['pickup_latitude'].between(39,42) & train['dropoff_latitude'].between(39,42) &
                  train['pickup_longitude'].between(-74.4,-72.8) &  train['dropoff_longitude'].between(-74.4,-72.8)]


## === cell 23
train.loc[:,'pickuplat_no'], pick_lat_bin = pd.cut(train['pickup_latitude'],100, labels=False, retbins=True)
train.loc[:,'pickuplong_no'], pick_long_bin = pd.cut(train['pickup_longitude'],100, labels=False, retbins=True)
train.loc[:,'dropofflat_no'], drop_lat_bin = pd.cut(train['dropoff_latitude'],100, labels=False, retbins=True)
train.loc[:,'dropofflong_no'], drop_long_bin = pd.cut(train['dropoff_longitude'],100, labels=False, retbins=True)
test.loc[:,'pickuplat_no'] = pd.cut(test['pickup_latitude'], pick_lat_bin, labels=False)
test.loc[:,'pickuplong_no'] = pd.cut(test['pickup_longitude'], pick_long_bin, labels=False)
test.loc[:,'dropofflat_no'] = pd.cut(test['dropoff_latitude'], drop_lat_bin, labels=False)
test.loc[:,'dropofflong_no'] = pd.cut(test['dropoff_longitude'], drop_long_bin, labels=False)


## === cell 24
train.loc[:, "pickdrop_lat_diff"] = abs(
    train["pickuplat_no"].astype(int) - train["dropofflat_no"].astype(int)
)  # .astype('category')
train.loc[:, "pickdrop_long_diff"] = abs(
    train["pickuplong_no"].astype(int) - train["dropofflong_no"].astype(int)
)  # .astype('category')
train.loc[:, "final_dist_factor"] = train["pickdrop_lat_diff"].astype(int) + train[
    "pickdrop_long_diff"
].astype(
    int
)  # .astype('category')

test.loc[:, "pickdrop_lat_diff"] = abs(
    test["pickuplat_no"].astype("Int64") - test["dropofflat_no"].astype("Int64")
)  # .astype('category')
test.loc[:, "pickdrop_long_diff"] = abs(
    test["pickuplong_no"].astype("Int64") - test["dropofflong_no"].astype("Int64")
)  # .astype('category')
test.loc[:, "final_dist_factor"] = test["pickdrop_lat_diff"].astype("Int64") + test[
    "pickdrop_long_diff"
].astype(
    "Int64"
)  # .astype('category')


## === cell 25
print(train.shape,test.shape)


## === cell 26
train = train.drop(['fare_per_km_passenger','fare_per_km'],axis=1)


## === cell 27
def calc_cwd_factor(df, col):
    new_df = df.groupby(col)['key'].count().sort_values(ascending=False).reset_index()
    new_df['cwd_factor'] = 1
    count = 1
    for i in range(1,new_df.shape[0]):
        count += 1
        if new_df.loc[i-1,'key'] == new_df.loc[i,'key']:
            count -= 1
        new_df.loc[i,'cwd_factor'] = count
    new_df.index = new_df[col]
    return new_df

fact_df = calc_cwd_factor(train, 'pickuplat_no')
train.loc[:,'pickuplat_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], train['pickuplat_no']))
fact_df = calc_cwd_factor(train, 'pickuplong_no')
train.loc[:,'pickuplong_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], train['pickuplong_no']))
fact_df = calc_cwd_factor(train, 'dropofflat_no')
train.loc[:,'dropofflat_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], train['dropofflat_no']))
fact_df = calc_cwd_factor(train, 'dropofflong_no')
train.loc[:,'dropofflong_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], train['dropofflong_no']))
fact_df = calc_cwd_factor(test, 'pickuplat_no')
test.loc[:,'pickuplat_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], test['pickuplat_no']))
fact_df = calc_cwd_factor(test, 'pickuplong_no')
test.loc[:,'pickuplong_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], test['pickuplong_no']))
fact_df = calc_cwd_factor(test, 'dropofflat_no')
test.loc[:,'dropofflat_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], test['dropofflat_no']))
fact_df = calc_cwd_factor(test, 'dropofflong_no')
test.loc[:,'dropofflong_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], test['dropofflong_no']))


## --- ERROR in cell 27, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36mget_loc[0;34m(self, key)[0m
[1;32m   3804[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3805[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_engine[0m[0;34m.[0m[0mget_loc[0m[0;34m([0m[0mcasted_key[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3806[0m         [0;32mexcept[0m [0mKeyError[0m [0;32mas[0m [0merr[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32mindex.pyx[0m in [0;36mpandas._libs.index.IndexEngine.get_loc[0;34m()[0m

[0;32mindex.pyx[0m in [0;36mpandas._libs.index.IndexEngine.get_loc[0;34m()[0m

[0;32mpandas/_libs/hashtable_class_helper.pxi[0m in [0;36mpandas._libs.hashtable.Float64HashTable.get_item[0;34m()[0m

[0;32mpandas/_libs/hashtable_class_helper.pxi[0m in [0;36mpandas._libs.hashtable.Float64HashTable.get_item[0;34m()[0m

[0;31mKeyError[0m: nan

The above exception was the direct cause of the following exception:

[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1212928172.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     21[0m [0;31m#do the same for test set:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     22[0m [0mfact_df[0m [0;34m=[0m [0mcalc_cwd_factor[0m[0;34m([0m[0mtest[0m[0;34m,[0m [0;34m'pickuplat_no'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 23[0;31m [0mtest[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m[0;34m'pickuplat_cwd_factor'[0m[0;34m][0m [0;34m=[0m [0mlist[0m[0;34m([0m[0mmap[0m[0;34m([0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0mfact_df[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0mx[0m[0;34m,[0m[0;34m'cwd_factor'[0m[0;34m][0m[0;34m,[0m [0mtest[0m[0;34m[[0m[0;34m'pickuplat_no'[0m[0;34m][0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     24[0m [0mfact_df[0m [0;34m=[0m [0mcalc_cwd_factor[0m[0;34m([0m[0mtest[0m[0;34m,[0m [0;34m'pickuplong_no'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     25[0m [0mtest[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m[0;34m'pickuplong_cwd_factor'[0m[0;34m][0m [0;34m=[0m [0mlist[0m[0;34m([0m[0mmap[0m[0;34m([0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0mfact_df[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0mx[0m[0;34m,[0m[0;34m'cwd_factor'[0m[0;34m][0m[0;34m,[0m [0mtest[0m[0;34m[[0m[0;34m'pickuplong_no'[0m[0;34m][0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1212928172.py[0m in [0;36m<lambda>[0;34m(x)[0m
[1;32m     21[0m [0;31m#do the same for test set:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     22[0m [0mfact_df[0m [0;34m=[0m [0mcalc_cwd_factor[0m[0;34m([0m[0mtest[0m[0;34m,[0m [0;34m'pickuplat_no'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 23[0;31m [0mtest[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m[0;34m'pickuplat_cwd_factor'[0m[0;34m][0m [0;34m=[0m [0mlist[0m[0;34m([0m[0mmap[0m[0;34m([0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0mfact_df[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0mx[0m[0;34m,[0m[0;34m'cwd_factor'[0m[0;34m][0m[0;34m,[0m [0mtest[0m[0;34m[[0m[0;34m'pickuplat_no'[0m[0;34m][0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     24[0m [0mfact_df[0m [0;34m=[0m [0mcalc_cwd_factor[0m[0;34m([0m[0mtest[0m[0;34m,[0m [0;34m'pickuplong_no'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     25[0m [0mtest[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m[0;34m'pickuplong_cwd_factor'[0m[0;34m][0m [0;34m=[0m [0mlist[0m[0;34m([0m[0mmap[0m[0;34m([0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0mfact_df[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0mx[0m[0;34m,[0m[0;34m'cwd_factor'[0m[0;34m][0m[0;34m,[0m [0mtest[0m[0;34m[[0m[0;34m'pickuplong_no'[0m[0;34m][0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m__getitem__[0;34m(self, key)[0m
[1;32m   1181[0m             [0mkey[0m [0;34m=[0m [0mtuple[0m[0;34m([0m[0mcom[0m[0;34m.[0m[0mapply_if_callable[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mobj[0m[0;34m)[0m [0;32mfor[0m [0mx[0m [0;32min[0m [0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1182[0m             [0;32mif[0m [0mself[0m[0;34m.[0m[0m_is_scalar_access[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1183[0;31m                 [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mobj[0m[0;34m.[0m[0m_get_value[0m[0;34m([0m[0;34m*[0m[0mkey[0m[0;34m,[0m [0mtakeable[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0m_takeable[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1184[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_getitem_tuple[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1185[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36m_get_value[0;34m(self, index, col, takeable)[0m
[1;32m   4219[0m             [0;31m#  results if our categories are integers that dont match our codes[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4220[0m             [0;31m# IntervalIndex: IntervalTree has no get_loc[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4221[0;31m             [0mrow[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mindex[0m[0;34m.[0m[0mget_loc[0m[0;34m([0m[0mindex[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   4222[0m             [0;32mreturn[0m [0mseries[0m[0;34m.[0m[0m_values[0m[0;34m[[0m[0mrow[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m   4223[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36mget_loc[0;34m(self, key)[0m
[1;32m   3810[0m             ):
[1;32m   3811[0m                 [0;32mraise[0m [0mInvalidIndexError[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3812[0;31m             [0;32mraise[0m [0mKeyError[0m[0;34m([0m[0mkey[0m[0;34m)[0m [0;32mfrom[0m [0merr[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3813[0m         [0;32mexcept[0m [0mTypeError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3814[0m             [0;31m# If we have a listlike key, _check_indexing_error will raise[0m[0;34m[0m[0;34m[0m[0m

[0;31mKeyError[0m: nan

## === cell 28
print(train.shape,test.shape)
