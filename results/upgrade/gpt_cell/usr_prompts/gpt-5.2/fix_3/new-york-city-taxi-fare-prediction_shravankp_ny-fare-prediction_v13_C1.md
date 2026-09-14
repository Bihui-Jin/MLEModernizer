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

eli5==0.13.0
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
train.isnull().sum()


## === cell 3
train = train.dropna(how='any',axis=0)


## === cell 4
train['abs_diff_longitude'] = np.abs(train['dropoff_longitude'] - train['pickup_longitude'])
train['abs_diff_latitude'] = np.abs(train['dropoff_latitude'] - train['pickup_latitude'])
test['abs_diff_longitude'] = np.abs(test['dropoff_longitude'] - test['pickup_longitude'])
test['abs_diff_latitude'] = np.abs(test['dropoff_latitude'] - test['pickup_latitude'])


## === cell 6
train = train.loc[train['fare_amount']>0,:]
train = train.loc[(train["passenger_count"]<=6) & (train["passenger_count"]>0),:]
train = train.loc[(train["abs_diff_latitude"]<2) & (train["abs_diff_longitude"]<2),:]
train = train.loc[(train["abs_diff_latitude"]>0) & (train["abs_diff_longitude"]>0),:]


## === cell 8
train.loc[:,'timestamp_with_key'] = train.loc[:,'key'] 
test.loc[:,'timestamp_with_key'] = test.loc[:,'key']
train.key = pd.DataFrame({'key':train['key'].str.split('.').str[1].astype('int')})
test.key = pd.DataFrame({'key':test['key'].str.split('.').str[1].astype('int')})


## === cell 10
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], infer_datetime_format=True, utc=True
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], infer_datetime_format=True, utc=True
)

train.loc[:, "hour_no"] = train["pickup_datetime"][:].dt.strftime("%-H").astype("int")
test.loc[:, "hour_no"] = test["pickup_datetime"][:].dt.strftime("%-H").astype("int")
train.loc[:, "weekday_no"] = train["pickup_datetime"][:].dt.strftime("%w").astype("int")
test.loc[:, "weekday_no"] = test["pickup_datetime"][:].dt.strftime("%w").astype("int")
train.loc[:, "day_no"] = train["pickup_datetime"][:].dt.strftime("%-d").astype("int")
test.loc[:, "day_no"] = test["pickup_datetime"][:].dt.strftime("%-d").astype("int")
train.loc[:, "year_no"] = train["pickup_datetime"][:].dt.strftime("%-y").astype("int")
test.loc[:, "year_no"] = test["pickup_datetime"][:].dt.strftime("%-y").astype("int")


## === cell 13
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


## === cell 14
train['fare_per_km'] = train['fare_amount']/(train["dist_haversine_km"])
train['fare_per_km_passenger'] = train['fare_amount']/(train['dist_haversine_km']*train['passenger_count'])


## === cell 15
train.groupby('key').agg({'fare_per_km_passenger':'mean','key':'count','passenger_count':'mean','fare_amount':'mean'})


## === cell 16
train.loc[train['fare_per_km_passenger']>20,['fare_per_km_passenger','fare_amount','dist_haversine_km']]


## === cell 17
grouped_df = train.groupby('key')
count = 0
for key, item in grouped_df:
    count += 1
    if count == 2: ## to view key = 2
        filtered = grouped_df.get_group(key)["dist_haversine_km"]>1 #ignoring drives within 1km
        df = pd.DataFrame(grouped_df.get_group(key).loc[filtered,:].sort_values(by='pickup_datetime'))
        break


## === cell 18
indexes = ['key',df['pickup_datetime'].dt.strftime('%a'),'hour_no']
grouped = df[:][:].groupby(indexes).agg({'fare_per_km_passenger':'mean','hour_no':'count'})
grouped.rename(columns={'hour_no':'count'},inplace=True)
grouped


## === cell 19
reindexed = grouped.reset_index().drop('key',axis=1)
get_max_count = reindexed.groupby(['pickup_datetime']).agg({'count':'max'})
get_max_count = get_max_count.reindex(reindexed['pickup_datetime'], method='ffill')
reindexed = reindexed .set_index('pickup_datetime')
reindexed.loc[get_max_count['count'] == reindexed['count'],:]


## === cell 20
train = train.loc[ ~ ((train['fare_per_km']<0.2) & (train['dist_haversine_km']>1))]
train = train.loc[~ ((train['dist_haversine_km']<0.01) & (train['fare_per_km']>50))]


## === cell 21
print(train.shape[0] - train.loc[train['pickup_latitude'].between(40.5,41) | train['dropoff_latitude'].between(40.5,41) |
          train['pickup_longitude'].between(-74,-73.9) |  train['dropoff_longitude'].between(-74,-73.9)].shape[0])
old_train = train.copy()
train = train.loc[train['pickup_latitude'].between(40.5,41) & train['dropoff_latitude'].between(40.5,41) &
                  train['pickup_longitude'].between(-74,-73.9) &  train['dropoff_longitude'].between(-74,-73.9)]


## === cell 22
train.loc[:,'pickuplat_no'], pick_lat_bin = pd.cut(train['pickup_latitude'],100, labels=False, retbins=True)
train.loc[:,'pickuplong_no'], pick_long_bin = pd.cut(train['pickup_longitude'],100, labels=False, retbins=True)
train.loc[:,'dropofflat_no'], drop_lat_bin = pd.cut(train['dropoff_latitude'],100, labels=False, retbins=True)
train.loc[:,'dropofflong_no'], drop_long_bin = pd.cut(train['dropoff_longitude'],100, labels=False, retbins=True)
test['pickuplat_no'] = pd.cut(test['pickup_latitude'], pick_lat_bin, labels=False).fillna(int(train.loc[:,'pickuplat_no'].mean()))
test['pickuplong_no'] = pd.cut(test['pickup_longitude'], pick_long_bin, labels=False).fillna(int(train.loc[:,'pickuplong_no'].mean()))


## === cell 23
test['dropofflat_no'] = pd.cut(test['dropoff_latitude'], drop_lat_bin, labels=False).fillna(int(train.loc[:,'dropofflat_no'].mean()))
test['dropofflong_no'] = pd.cut(test['dropoff_longitude'], drop_long_bin, labels=False).fillna(int(train.loc[:,'dropofflong_no'].mean()))


## === cell 24
train.loc[:,'pickdrop_lat_diff'] = abs(train['pickuplat_no'].astype(int) - train['dropofflat_no'].astype(int))#.astype('category')
train.loc[:,'pickdrop_long_diff'] = abs(train['pickuplong_no'].astype(int) - train['dropofflong_no'].astype(int))#.astype('category')
train.loc[:,'final_dist_factor'] = (train['pickdrop_lat_diff'].astype(int) + train['pickdrop_long_diff'].astype(int))#.astype('category')
test.loc[:,'pickdrop_lat_diff'] = abs(test['pickuplat_no'].astype(int) - test['dropofflat_no'].astype(int))#.astype('category')
test.loc[:,'pickdrop_long_diff'] = abs(test['pickuplong_no'].astype(int) - test['dropofflong_no'].astype(int))#.astype('category')
test.loc[:,'final_dist_factor'] = (test['pickdrop_lat_diff'].astype(int) + test['pickdrop_long_diff'].astype(int))#.astype('category')


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


## === cell 32

train = shuffle(train.iloc[:,:]).reset_index(drop=True)
val = train.iloc[int(0.9*train.shape[0]):,:]
train = train.iloc[:int(0.9*train.shape[0]),:]

cols_to_normalize = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude',
              'abs_diff_longitude', 'abs_diff_latitude', 'dist_haversine_km']

transformer = Normalizer().fit(train.loc[:,cols_to_normalize])
train.loc[:,cols_to_normalize] = transformer.transform(train.loc[:,cols_to_normalize])
val.loc[:,cols_to_normalize] = transformer.transform(val.loc[:,cols_to_normalize])         
test.loc[:,cols_to_normalize] = transformer.transform(test.loc[:,cols_to_normalize])


## === cell 34
import seaborn as sns
categorical_cols = [i for i in train.columns if i not in ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude',
'abs_diff_longitude', 'abs_diff_latitude', 'dist_haversine_km','key','pickup_datetime','timestamp_with_key','fare_amount']]
numerical_cols =['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude',
'abs_diff_longitude', 'abs_diff_latitude', 'dist_haversine_km','fare_amount']
object_cols = ['key','pickup_datetime','timestamp_with_key']
cor = train.loc[:,numerical_cols]
f, ax = plt.subplots(1,1,figsize=(18,7))
sns.heatmap(cor.corr(), annot=True)


## === cell 36
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

obj = SelectKBest(f_regression, k=10)
obj.fit(train[categorical_cols], train['fare_amount'])



## === cell 37
categories_selected = []
for i in range(len(categorical_cols)):
    if obj.get_support()[i]:
        categories_selected.append(categorical_cols[i])
categories_selected


## === cell 38
train_y = train['fare_amount'][:]
val_y = val['fare_amount'][:]
cols = [i for i in categories_selected+numerical_cols if i not in ['fare_amount'] + object_cols]
train_x = train.loc[:, cols]
val_x = val.loc[:, cols]
test_x = test.loc[:,cols]


## === cell 40
import xgboost as xgb
from xgboost import XGBRegressor


## === cell 41
xgbr = XGBRegressor()
xgb_train = xgbr.fit(train_x, train_y)


## === cell 42
pred_train = xgbr.predict(train_x).round(decimals = 2)
pred_val = xgbr.predict(val_x).round(decimals = 2)
pred_test = xgbr.predict(test_x).round(decimals = 2)


## === cell 43
from sklearn.metrics import mean_squared_error
rmse_train = np.sqrt(mean_squared_error(train_y, pred_train))
rmse_val = np.sqrt(mean_squared_error(val_y, pred_val))
print(rmse_train, rmse_val)


## === cell 44
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(xgb_train, random_state=1).fit(val_x, val_y)
eli5.show_weights(perm, feature_names=val_x.columns.tolist())


## --- ERROR in cell 44, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;31mAttributeError[0m: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 45
final = pd.DataFrame({'key':test.timestamp_with_key, 'fare_amount':pred_test}, columns = ['key', 'fare_amount'])
final.to_csv('submission.csv', index = False)
