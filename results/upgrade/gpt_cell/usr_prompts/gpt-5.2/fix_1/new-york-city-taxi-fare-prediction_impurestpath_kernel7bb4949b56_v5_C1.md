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

3.8

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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('seaborn-whitegrid')


## === cell 1
df_train = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1_000_000,parse_dates=["pickup_datetime"])
df_train.head()


## === cell 2
df_train.describe()


## === cell 3
df_test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv",parse_dates=["pickup_datetime"])
df_test.head()


## === cell 4
df_test.describe()


## === cell 5
print('Old size: %d' % len(df_train))
df_train = df_train[df_train.fare_amount>=0]
print('New size: %d' % len(df_train))


## === cell 6
print('Old size: %d' % len(df_train))
df_train = df_train.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(df_train))


## === cell 7
df_train[df_train.fare_amount < 80].fare_amount.hist(bins=100)
plt.xlabel('fare $USD')


## === cell 8
df_train['diff_long'] = (df_train.dropoff_longitude - df_train.pickup_longitude).abs()
df_train['diff_long'].describe()


## === cell 9
df_train['diff_lat'] = (df_train.dropoff_latitude - df_train.pickup_latitude).abs()
df_train['diff_lat'].describe()


## === cell 10
print('Old size: %d' % len(df_train))
df_train = df_train[(df_train.diff_long < 5.0) & (df_train.diff_lat < 5.0)]
print('New size: %d' % len(df_train))


## === cell 11
df_train['year'] = df_train.pickup_datetime.apply(lambda t: t.year)
df_train['weekday'] = df_train.pickup_datetime.apply(lambda t: t.weekday())
df_train['hour'] = df_train.pickup_datetime.apply(lambda t: t.hour)


## === cell 12
df_train.describe()


## === cell 13
df_train[['fare_amount', 'hour']].groupby(['hour'], as_index=False).mean().sort_values(by='fare_amount', ascending=False)


## === cell 14
df_train[['fare_amount', 'weekday']].groupby(['weekday'], as_index=False).mean().sort_values(by='fare_amount', ascending=False)


## === cell 15
df_train[['fare_amount', 'year']].groupby(['year'], as_index=False).mean().sort_values(by='fare_amount', ascending=False)


## === cell 16
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...


## === cell 17
df_train['distance'] = distance(df_train.pickup_latitude, df_train.pickup_longitude, \
                                      df_train.dropoff_latitude, df_train.dropoff_longitude)


## === cell 18
plot = df_train.plot.scatter('distance', 'fare_amount')


## === cell 19
plot = df_train[(df_train.distance < 50) & (df_train.fare_amount < 100)].plot.scatter('distance', 'fare_amount',alpha=0.1)


## === cell 20
print('Old size: %d' % len(df_train))
df_train = df_train[(df_train.distance >= 0.1)]
print('New size: %d' % len(df_train))


## === cell 21
plot = df_train[(df_train.distance < 50) & (df_train.fare_amount < 100)].plot.scatter('distance', 'fare_amount',alpha=0.1)


## === cell 22
print('Old size: %d' % len(df_train))
df_train = df_train[(df_train.distance <= 50)]
print('New size: %d' % len(df_train))


## === cell 23
plot = df_train.plot.scatter('distance', 'fare_amount',alpha=0.1)


## === cell 24
print('Old size: %d' % len(df_train))
df_train = df_train[(df_train.fare_amount <= 200)]
print('New size: %d' % len(df_train))


## === cell 25
plot = df_train.plot.scatter('distance', 'fare_amount',alpha=0.1)


## === cell 26
features = ['year', 'hour', 'distance']
X = df_train[features].values
y = df_train['fare_amount'].values


## === cell 28
df_test['year'] = df_test.pickup_datetime.apply(lambda t: t.year)
df_test['hour'] = df_test.pickup_datetime.apply(lambda t: t.hour)
df_test['distance'] = distance(df_test.pickup_latitude, df_test.pickup_longitude, \
                                      df_test.dropoff_latitude, df_test.dropoff_longitude)


## === cell 29
X_kaggle_test = df_test[features].values


## === cell 31
from sklearn.model_selection import train_test_split
import xgboost as xgb

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.3)

def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'},
                    dtrain=matrix_train,num_boost_round=100, 
                    early_stopping_rounds=100,evals=[(matrix_test,'test')])
    return model

model = XGBmodel(X_train,X_test,y_train,y_test)
prediction = model.predict(xgb.DMatrix(X_kaggle_test), ntree_limit = model.best_ntree_limit)


## --- ERROR in cell 31, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1823918905.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     13[0m [0;34m[0m[0m
[1;32m     14[0m [0mmodel[0m [0;34m=[0m [0mXGBmodel[0m[0;34m([0m[0mX_train[0m[0;34m,[0m[0mX_test[0m[0;34m,[0m[0my_train[0m[0;34m,[0m[0my_test[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 15[0;31m [0mprediction[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mxgb[0m[0;34m.[0m[0mDMatrix[0m[0;34m([0m[0mX_kaggle_test[0m[0;34m)[0m[0;34m,[0m [0mntree_limit[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mbest_ntree_limit[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mAttributeError[0m: 'Booster' object has no attribute 'best_ntree_limit'

## === cell 32
submission = pd.DataFrame({
        "key": df_test['key'],
        "fare_amount": prediction.round(2)
})

submission.to_csv('submission.csv',index=False)
submission
