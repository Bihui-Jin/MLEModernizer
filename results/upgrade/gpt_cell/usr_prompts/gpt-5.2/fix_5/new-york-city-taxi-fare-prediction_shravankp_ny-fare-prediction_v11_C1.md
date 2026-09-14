# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

21.8961

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 21.8961) has done: 'Diagnosis: The crash in cell 27 happens because some binned location columns in `test` (e.g., `pickuplat_no`) contain `NaN` values after `pd.cut` (values falling outside the train-derived bin edges). The current code then tries to look up `fact_df.loc[nan, 'cwd_factor']`, which raises `KeyError: nan` because `nan` is not a valid index label. This is specific to the test set and must be handled while preserving the same “cwd_factor” logic.

Patch summary: In cell 27 only, replace the per-row `.loc` lookup via `map(lambda x: fact_df.loc[x, ...])` with a safe `.map` against a Series indexed by the bin value, then fill missing mappings (including NaNs / unseen bins) with a deterministic default (`0`). The core `calc_cwd_factor` logic is unchanged; only the assignment mechanism is made robust to NaNs/unseen categories.

Updated cells: cell 27.

Compatibility notes for cell k+1: `train` and `test` keep the same columns as intended (`*_cwd_factor` are created), and their types remain numeric; subsequent cells that reference these columns continue to work.

Assumptions: Using `0` as the default factor for unseen/NaN bins is acceptable as a neutral deterministic fallback and only affects rows that previously crashed (no change for rows with valid bin indices).'

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
    new_df = df.groupby(col)["key"].count().sort_values(ascending=False).reset_index()
    new_df["cwd_factor"] = 1
    count = 1
    for i in range(1, new_df.shape[0]):
        count += 1
        if new_df.loc[i - 1, "key"] == new_df.loc[i, "key"]:
            count -= 1
        new_df.loc[i, "cwd_factor"] = count
    new_df.index = new_df[col]
    return new_df


fact_df = calc_cwd_factor(train, "pickuplat_no")
train.loc[:, "pickuplat_cwd_factor"] = (
    train["pickuplat_no"].map(fact_df["cwd_factor"]).fillna(0).astype(int)
)

fact_df = calc_cwd_factor(train, "pickuplong_no")
train.loc[:, "pickuplong_cwd_factor"] = (
    train["pickuplong_no"].map(fact_df["cwd_factor"]).fillna(0).astype(int)
)

fact_df = calc_cwd_factor(train, "dropofflat_no")
train.loc[:, "dropofflat_cwd_factor"] = (
    train["dropofflat_no"].map(fact_df["cwd_factor"]).fillna(0).astype(int)
)

fact_df = calc_cwd_factor(train, "dropofflong_no")
train.loc[:, "dropofflong_cwd_factor"] = (
    train["dropofflong_no"].map(fact_df["cwd_factor"]).fillna(0).astype(int)
)

fact_df = calc_cwd_factor(test, "pickuplat_no")
test.loc[:, "pickuplat_cwd_factor"] = (
    test["pickuplat_no"].map(fact_df["cwd_factor"]).fillna(0).astype(int)
)

fact_df = calc_cwd_factor(test, "pickuplong_no")
test.loc[:, "pickuplong_cwd_factor"] = (
    test["pickuplong_no"].map(fact_df["cwd_factor"]).fillna(0).astype(int)
)

fact_df = calc_cwd_factor(test, "dropofflat_no")
test.loc[:, "dropofflat_cwd_factor"] = (
    test["dropofflat_no"].map(fact_df["cwd_factor"]).fillna(0).astype(int)
)

fact_df = calc_cwd_factor(test, "dropofflong_no")
test.loc[:, "dropofflong_cwd_factor"] = (
    test["dropofflong_no"].map(fact_df["cwd_factor"]).fillna(0).astype(int)
)


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


## === cell 38
from sklearn.metrics import mean_squared_error
rmse_train = np.sqrt(mean_squared_error(train_y, pred_train))
rmse_val = np.sqrt(mean_squared_error(val_y, pred_val))
print(rmse_train, rmse_val)


## === cell 39
final = pd.DataFrame({'key':test.timestamp_with_key, 'fare_amount':pred_test}, columns = ['key', 'fare_amount'])
final.to_csv('submission.csv', index = False)


## === cell 40
final.head()
