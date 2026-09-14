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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

import os
print(os.listdir("../input"))



## === cell 1
types = {'fare_amount': 'float32',
         'pickup_longitude': 'float32',
         'pickup_latitude': 'float32',
         'dropoff_longitude': 'float32',
         'dropoff_latitude': 'float32',
         'passenger_count': 'uint8'}


cols = ['fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 
        'dropoff_longitude', 'dropoff_latitude', 'passenger_count']


## === cell 2
train = pd.read_csv('../input/train.csv', nrows=1000000, usecols=cols, dtype=types)
test = pd.read_csv('../input/test.csv')
samp = pd.read_csv('../input/sample_submission.csv')


## === cell 3
train.dropna(how = 'any', axis = 'rows', inplace=True)
train = train[train.fare_amount > 0]
train = train[train['passenger_count'] <= 6]


## === cell 4
latitude_mask_pickup = (train.pickup_latitude > -90) &  (train.pickup_latitude < 90)

train = train[latitude_mask_pickup]

latitude_mask_dropoff = (train.dropoff_latitude > -90) &  (train.dropoff_latitude < 90)

train = train[latitude_mask_dropoff]


## === cell 5
longitude_mask_pickup = (train.pickup_longitude > -180) &  (train.pickup_longitude < 180)

train = train[longitude_mask_pickup]

longitude_mask_dropoff = (train.dropoff_longitude > -180) &  (train.dropoff_longitude < 180)

train = train[longitude_mask_dropoff]


## === cell 6
all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['fare_amount'] , axis=1, inplace=True)
y = train.fare_amount.values
n_train = len(train)
n_test = len(test)
test_id = test.key


## === cell 7
def week_num(day):
    '''
    given the day of the month, return the week number of the month
    '''
    if day <=7 : return 'first'
    if (day > 7) and (day <= 14): return 'second'
    if (day > 14) and (day <= 21): return 'third'
    if (day > 21) and (day <= 28): return 'fourth'
    return 'fifth'


## === cell 8
def add_time_features(data):
    data.pickup_datetime =  pd.to_datetime(data.pickup_datetime)

    data['hour'] = data.pickup_datetime.dt.hour
    data['day_of_week'] = data.pickup_datetime.dt.weekday_name
    data['day_of_month'] = data.pickup_datetime.dt.day
    data['week_of_month'] = data.day_of_month.map(week_num)
    data['month'] = data.pickup_datetime.dt.month
    data['year'] = data.pickup_datetime.dt.year

    data.hour = data.hour.astype(str)
    data.month = data.month.astype(str)
    data.year = data.year.astype(str)
    data.drop('day_of_month', axis=1, inplace=True)
    
    return data


## === cell 9
def add_geo_features(data):
    data['abs_diff_longitude'] = (data.dropoff_longitude - data.pickup_longitude).abs()
    data['abs_diff_latitude'] = (data.dropoff_latitude - data.pickup_latitude).abs()

    data['manhattan_distance'] = data['abs_diff_longitude'] + data['abs_diff_latitude']

    data['squared_long'] = np.power(data['abs_diff_longitude'],2)
    data['squared_lat'] = np.power(data['abs_diff_latitude'],2)

    data['euclid_disance'] = np.sqrt(data['squared_long'] + data['squared_lat'])
    
    return data


## === cell 10
def add_time_features(data):
    data.pickup_datetime = pd.to_datetime(data.pickup_datetime)

    data["hour"] = data.pickup_datetime.dt.hour
    data["day_of_week"] = data.pickup_datetime.dt.day_name()
    data["day_of_month"] = data.pickup_datetime.dt.day
    data["week_of_month"] = data.day_of_month.map(week_num)
    data["month"] = data.pickup_datetime.dt.month
    data["year"] = data.pickup_datetime.dt.year

    data.hour = data.hour.astype(str)
    data.month = data.month.astype(str)
    data.year = data.year.astype(str)
    data.drop("day_of_month", axis=1, inplace=True)

    return data


## === cell 11
all_data = add_geo_features(all_data)


## === cell 12
features = ['passenger_count', 'hour', 'day_of_week', 'week_of_month', 'month', 'year', 'abs_diff_longitude', 'abs_diff_latitude',
           'euclid_disance', 'manhattan_distance']

all_data = all_data[features]

all_data = pd.get_dummies(all_data)


## --- ERROR in cell 12, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2310379602.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m            'euclid_disance', 'manhattan_distance']
[1;32m      3[0m [0;34m[0m[0m
[0;32m----> 4[0;31m [0mall_data[0m [0;34m=[0m [0mall_data[0m[0;34m[[0m[0mfeatures[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0;34m[0m[0m
[1;32m      6[0m [0mall_data[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mget_dummies[0m[0;34m([0m[0mall_data[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36m__getitem__[0;34m(self, key)[0m
[1;32m   4106[0m             [0;32mif[0m [0mis_iterator[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4107[0m                 [0mkey[0m [0;34m=[0m [0mlist[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4108[0;31m             [0mindexer[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mcolumns[0m[0;34m.[0m[0m_get_indexer_strict[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0;34m"columns"[0m[0;34m)[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   4109[0m [0;34m[0m[0m
[1;32m   4110[0m         [0;31m# take() does not accept boolean indexers[0m[0;34m[0m[0;34m[0m[0m

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

[0;31mKeyError[0m: "['hour', 'day_of_week', 'week_of_month', 'month', 'year'] not in index"

## === cell 13
x = all_data[:n_train]
x_test = all_data[n_train:]
