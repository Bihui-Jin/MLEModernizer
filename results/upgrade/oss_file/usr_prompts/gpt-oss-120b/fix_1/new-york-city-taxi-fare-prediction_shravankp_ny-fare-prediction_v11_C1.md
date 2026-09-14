# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict the fare amount for a taxi ride given the pickup and dropoff locations.

## Metric
Root mean-squared error.

## Submission Format
For each `key` in the test set, you must predict a value for the `fare_amount` variable. The file should contain a header and have the following format:

```
key,fare_amount
2015-01-27 13:08:24.0000002,11.00
2015-02-27 13:08:24.0000002,12.05
2015-03-27 13:08:24.0000002,11.23
2015-04-27 13:08:24.0000002,14.17
2015-05-27 13:08:24.0000002,15.12
etc
```

## Dataset
- **train.csv** - Input features and target `fare_amount` values for the training set (about 55M rows).
- **test.csv** - Input features for the test set (about 10K rows). Your goal is to predict `fare_amount` for each row.
- **sample_submission.csv** - a sample submission file in the correct format (columns `key` and `fare_amount`). This file 'predicts' `fare_amount` to be $`11.35` for all rows, which is the mean `fare_amount` from the training set.

### Data fields
**ID**

- **key** - Unique `string` identifying each row in both the training and test sets. Comprised of **pickup_datetime** plus a unique integer, but this doesn't matter, it should just be used as a unique ID field.Required in your submission CSV. Not necessarily needed in the training set, but could be useful to simulate a 'submission file' while doing cross-validation within the training set.

**Features**

- **pickup_datetime** - `timestamp` value indicating when the taxi ride started.
- **pickup_longitude** - `float` for longitude coordinate of where the taxi ride started.
- **pickup_latitude** - `float` for latitude coordinate of where the taxi ride started.
- **dropoff_longitude** - `float` for longitude coordinate of where the taxi ride ended.
- **dropoff_latitude** - `float` for latitude coordinate of where the taxi ride ended.
- **passenger_count** - `integer` indicating the number of passengers in the taxi ride.

**Target**

- **fare_amount** - `float` dollar amount of the cost of the taxi ride. This value is only in the training set; this is what you are predicting in the test set and it is required in your submission CSV.

# 2. Python version

3.7

# 3. Installed packages

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

# 4. Data file paths

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

# 5. Target score

4.41786

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
    return int(hr/3 + 1)

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], infer_datetime_format=True).dt.tz_localize('UTC')
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], infer_datetime_format=True).dt.tz_localize('UTC')
train['time_slot'] = pd.DataFrame(list(map(lambda x : chooseSlot(x), train['pickup_datetime'][:])), index=train.index)
test['time_slot'] = pd.DataFrame(list(map(lambda x : chooseSlot(x), test['pickup_datetime'][:])), index=test.index)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1350252973.py in <cell line: 0>()
      7     return int(hr/3 + 1)
      8 
----> 9 train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], infer_datetime_format=True).dt.tz_localize('UTC')
     10 test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], infer_datetime_format=True).dt.tz_localize('UTC')
     11 train['time_slot'] = pd.DataFrame(list(map(lambda x : chooseSlot(x), train['pickup_datetime'][:])), index=train.index)

/usr/local/lib/python3.11/dist-packages/pandas/core/accessor.py in f(self, *args, **kwargs)
    110         def _create_delegator_method(name: str):
    111             def f(self, *args, **kwargs):
--> 112                 return self._delegate_method(name, *args, **kwargs)
    113 
    114             f.__name__ = name

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/accessors.py in _delegate_method(self, name, *args, **kwargs)
    130 
    131         method = getattr(values, name)
--> 132         result = method(*args, **kwargs)
    133 
    134         if not is_list_like(result):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/datetimes.py in tz_localize(self, tz, ambiguous, nonexistent)
    291         nonexistent: TimeNonexistent = "raise",
    292     ) -> Self:
--> 293         arr = self._data.tz_localize(tz, ambiguous, nonexistent)
    294         return type(self)._simple_new(arr, name=self.name)
    295 

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/_mixins.py in method(self, *args, **kwargs)
     79     def method(self, *args, **kwargs):
     80         if self.ndim == 1:
---> 81             return meth(self, *args, **kwargs)
     82 
     83         flags = self._ndarray.flags

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimes.py in tz_localize(self, tz, ambiguous, nonexistent)
   1081                 new_dates = tz_convert_from_utc(self.asi8, self.tz, reso=self._creso)
   1082             else:
-> 1083                 raise TypeError("Already tz-aware, use tz_convert to convert.")
   1084         else:
   1085             tz = timezones.maybe_get_tz(tz)

TypeError: Already tz-aware, use tz_convert to convert.

## === cell 11
train['weekday_no'] = pd.DataFrame(list(map(lambda x : x.strftime('%w'), train['pickup_datetime'][:])), dtype=int, index=train.index)
test['weekday_no'] = pd.DataFrame(list(map(lambda x : x.strftime('%w'), test['pickup_datetime'][:])), dtype=int, index=test.index)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3858976809.py in <cell line: 0>()
----> 1 train['weekday_no'] = pd.DataFrame(list(map(lambda x : x.strftime('%w'), train['pickup_datetime'][:])), dtype=int, index=train.index)
      2 test['weekday_no'] = pd.DataFrame(list(map(lambda x : x.strftime('%w'), test['pickup_datetime'][:])), dtype=int, index=test.index)

/tmp/ipykernel_11/3858976809.py in <lambda>(x)
----> 1 train['weekday_no'] = pd.DataFrame(list(map(lambda x : x.strftime('%w'), train['pickup_datetime'][:])), dtype=int, index=train.index)
      2 test['weekday_no'] = pd.DataFrame(list(map(lambda x : x.strftime('%w'), test['pickup_datetime'][:])), dtype=int, index=test.index)

AttributeError: 'str' object has no attribute 'strftime'

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


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1506473754.py in <cell line: 0>()
----> 1 indexes = ['key',df['pickup_datetime'].dt.strftime('%a'),'time_slot']
      2 grouped = df[:][:].groupby(indexes).agg({'fare_per_km_passenger':'mean','time_slot':'count'})
      3 grouped.rename(columns={'time_slot':'count'},inplace=True)
      4 grouped

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/accessor.py in __get__(self, obj, cls)
    222             # we're accessing the attribute of the class, i.e., Dataset.geo
    223             return self._accessor
--> 224         accessor_obj = self._accessor(obj)
    225         # Replace the property with the accessor object. Inspired by:
    226         # https://www.pydanny.com/cached-property.html

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/accessors.py in __new__(cls, data)
    641             return PeriodProperties(data, orig)
    642 
--> 643         raise AttributeError("Can only use .dt accessor with datetimelike values")

AttributeError: Can only use .dt accessor with datetimelike values

## === cell 20
reindexed = grouped.reset_index().drop('key',axis=1)
get_max_count = reindexed.groupby(['pickup_datetime']).agg({'count':'max'})
get_max_count = get_max_count.reindex(reindexed['pickup_datetime'], method='ffill')
reindexed = reindexed .set_index('pickup_datetime')
reindexed.loc[get_max_count['count'] == reindexed['count'],:]


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1265395355.py in <cell line: 0>()
----> 1 reindexed = grouped.reset_index().drop('key',axis=1)
      2 get_max_count = reindexed.groupby(['pickup_datetime']).agg({'count':'max'})
      3 get_max_count = get_max_count.reindex(reindexed['pickup_datetime'], method='ffill')
      4 reindexed = reindexed .set_index('pickup_datetime')
      5 reindexed.loc[get_max_count['count'] == reindexed['count'],:]

NameError: name 'grouped' is not defined

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
train.loc[:,'pickdrop_lat_diff'] = abs(train['pickuplat_no'].astype(int) - train['dropofflat_no'].astype(int))#.astype('category')
train.loc[:,'pickdrop_long_diff'] = abs(train['pickuplong_no'].astype(int) - train['dropofflong_no'].astype(int))#.astype('category')
train.loc[:,'final_dist_factor'] = (train['pickdrop_lat_diff'].astype(int) + train['pickdrop_long_diff'].astype(int))#.astype('category')
test.loc[:,'pickdrop_lat_diff'] = abs(test['pickuplat_no'].astype(int) - test['dropofflat_no'].astype(int))#.astype('category')
test.loc[:,'pickdrop_long_diff'] = abs(test['pickuplong_no'].astype(int) - test['dropofflong_no'].astype(int))#.astype('category')
test.loc[:,'final_dist_factor'] = (test['pickdrop_lat_diff'].astype(int) + test['pickdrop_long_diff'].astype(int))#.astype('category')


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
IntCastingNaNError                        Traceback (most recent call last)
/tmp/ipykernel_11/1455156028.py in <cell line: 0>()
      2 train.loc[:,'pickdrop_long_diff'] = abs(train['pickuplong_no'].astype(int) - train['dropofflong_no'].astype(int))#.astype('category')
      3 train.loc[:,'final_dist_factor'] = (train['pickdrop_lat_diff'].astype(int) + train['pickdrop_long_diff'].astype(int))#.astype('category')
----> 4 test.loc[:,'pickdrop_lat_diff'] = abs(test['pickuplat_no'].astype(int) - test['dropofflat_no'].astype(int))#.astype('category')
      5 test.loc[:,'pickdrop_long_diff'] = abs(test['pickuplong_no'].astype(int) - test['dropofflong_no'].astype(int))#.astype('category')
      6 test.loc[:,'final_dist_factor'] = (test['pickdrop_lat_diff'].astype(int) + test['pickdrop_long_diff'].astype(int))#.astype('category')

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
     99 
    100     elif np.issubdtype(arr.dtype, np.floating) and dtype.kind in "iu":
--> 101         return _astype_float_to_int_nansafe(arr, dtype, copy)
    102 
    103     elif arr.dtype == object:

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_float_to_int_nansafe(values, dtype, copy)
    143     """
    144     if not np.isfinite(values).all():
--> 145         raise IntCastingNaNError(
    146             "Cannot convert non-finite values (NA or inf) to integer"
    147         )

IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer

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
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.Float64HashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.Float64HashTable.get_item()

KeyError: nan

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1212928172.py in <cell line: 0>()
     21 #do the same for test set:
     22 fact_df = calc_cwd_factor(test, 'pickuplat_no')
---> 23 test.loc[:,'pickuplat_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], test['pickuplat_no']))
     24 fact_df = calc_cwd_factor(test, 'pickuplong_no')
     25 test.loc[:,'pickuplong_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], test['pickuplong_no']))

/tmp/ipykernel_11/1212928172.py in <lambda>(x)
     21 #do the same for test set:
     22 fact_df = calc_cwd_factor(test, 'pickuplat_no')
---> 23 test.loc[:,'pickuplat_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], test['pickuplat_no']))
     24 fact_df = calc_cwd_factor(test, 'pickuplong_no')
     25 test.loc[:,'pickuplong_cwd_factor'] = list(map(lambda x: fact_df.loc[x,'cwd_factor'], test['pickuplong_no']))

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1181             key = tuple(com.apply_if_callable(x, self.obj) for x in key)
   1182             if self._is_scalar_access(key):
-> 1183                 return self.obj._get_value(*key, takeable=self._takeable)
   1184             return self._getitem_tuple(key)
   1185         else:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _get_value(self, index, col, takeable)
   4219             #  results if our categories are integers that dont match our codes
   4220             # IntervalIndex: IntervalTree has no get_loc
-> 4221             row = self.index.get_loc(index)
   4222             return series._values[row]
   4223 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: nan

## === cell 28
print(train.shape,test.shape)


## === cell 29
import sklearn
from sklearn import *
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


## === cell 30
orig_train = train.copy()
orig_test = test.copy()


## === cell 31
train = orig_train.copy()
test = orig_test.copy()


## === cell 32

train = shuffle(train.iloc[:,:]).reset_index(drop=True)
val = train.iloc[int(0.9*train.shape[0]):,:]
train = train.iloc[:int(0.9*train.shape[0]),:]

train_cols = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude',
              'abs_diff_longitude', 'abs_diff_latitude', 'dist_haversine_km']
test_cols = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude',
             'abs_diff_longitude', 'abs_diff_latitude','dist_haversine_km']

transformer = Normalizer().fit(train.loc[:,train_cols])
train.loc[:,train_cols] = transformer.transform(train.loc[:,train_cols])
val.loc[:,train_cols] = transformer.transform(val.loc[:,train_cols])         
test.loc[:,test_cols] = transformer.transform(test.loc[:,test_cols])


## === cell 33
train_y = train['fare_amount'][:]
val_y = val['fare_amount'][:]
cols = [i for i in train.columns if i not in ['fare_amount','key','pickup_datetime','timestamp_with_key']]        
train_x = train.loc[:, cols]
val_x = val.loc[:, cols]
test_x = test.loc[:,cols]


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/4110408202.py in <cell line: 0>()
      4 train_x = train.loc[:, cols]
      5 val_x = val.loc[:, cols]
----> 6 test_x = test.loc[:,cols]
      7 # print(train.columns, train.shape)
      8 # print(val.columns, val.shape)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1182             if self._is_scalar_access(key):
   1183                 return self.obj._get_value(*key, takeable=self._takeable)
-> 1184             return self._getitem_tuple(key)
   1185         else:
   1186             # we by definition only have the 0th axis

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
   1375             return self._multi_take(tup)
   1376 
-> 1377         return self._getitem_tuple_same_dim(tup)
   1378 
   1379     def _get_label(self, label, axis: AxisInt):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple_same_dim(self, tup)
   1018                 continue
   1019 
-> 1020             retval = getattr(retval, self.name)._getitem_axis(key, axis=i)
   1021             # We should never have retval.ndim < self.ndim, as that should
   1022             #  be handled by the _getitem_lowerdim call above.

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['pickdrop_lat_diff', 'pickdrop_long_diff', 'final_dist_factor', 'pickuplat_cwd_factor', 'pickuplong_cwd_factor', 'dropofflat_cwd_factor', 'dropofflong_cwd_factor'] not in index"

## === cell 35
import xgboost as xgb
from xgboost import XGBRegressor


## === cell 36
xgbr = XGBRegressor()
xgbr.fit(train_x, train_y)


## === cell 37
pred_train = xgbr.predict(train_x).round(decimals = 2)
pred_val = xgbr.predict(val_x).round(decimals = 2)
pred_test = xgbr.predict(test_x).round(decimals = 2)


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3404735244.py in <cell line: 0>()
      1 pred_train = xgbr.predict(train_x).round(decimals = 2)
      2 pred_val = xgbr.predict(val_x).round(decimals = 2)
----> 3 pred_test = xgbr.predict(test_x).round(decimals = 2)

NameError: name 'test_x' is not defined

## === cell 38
from sklearn.metrics import mean_squared_error
rmse_train = np.sqrt(mean_squared_error(train_y, pred_train))
rmse_val = np.sqrt(mean_squared_error(val_y, pred_val))
print(rmse_train, rmse_val)


## === cell 39
final = pd.DataFrame({'key':test.timestamp_with_key, 'fare_amount':pred_test}, columns = ['key', 'fare_amount'])
final.to_csv('submission.csv', index = False)


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/84813213.py in <cell line: 0>()
----> 1 final = pd.DataFrame({'key':test.timestamp_with_key, 'fare_amount':pred_test}, columns = ['key', 'fare_amount'])
      2 final.to_csv('submission.csv', index = False)

NameError: name 'pred_test' is not defined

## === cell 40
final.head()


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2472795012.py in <cell line: 0>()
----> 1 final.head()

NameError: name 'final' is not defined
