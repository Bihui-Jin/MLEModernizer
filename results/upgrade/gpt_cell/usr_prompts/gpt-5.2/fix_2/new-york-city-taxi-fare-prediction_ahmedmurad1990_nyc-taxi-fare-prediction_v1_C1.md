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


import os
print(os.listdir("../input"))



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background')
sns.set_style("darkgrid")


## === cell 2
%%time
train_path  = '../input/train.csv'

traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

cols = list(traintypes.keys())
train_df = pd.read_csv(train_path, usecols=cols, dtype=traintypes, nrows=2_000_000)


## === cell 3
%%time
train_df.to_feather('nyc_taxi_data_raw.feather')


## === cell 4

df_train = pd.read_feather('nyc_taxi_data_raw.feather')


## === cell 5
df_train.dtypes


## === cell 6
df_train.describe()


## === cell 7
len(df_train[df_train.fare_amount > 0])


## === cell 8
df_train = df_train[df_train.fare_amount>=0]


## === cell 9
sns.distplot(df_train[df_train.fare_amount < 100].fare_amount, bins=50);


## === cell 10
df_train.isnull().sum()


## === cell 11
df_train = df_train.dropna(how = 'any', axis = 'rows')


## === cell 12
df_test =  pd.read_csv('../input/test.csv')
df_test.head(5)


## === cell 13
df_test.describe()


## === cell 14
df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")


## === cell 15
df_train['pickup_datetime']


## === cell 16
df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")


## === cell 17
def add_new_date_time_features(dataset):
    dataset['hour'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['year'] = dataset.pickup_datetime.dt.year
    dataset['day_of_week'] = dataset.pickup_datetime.dt.dayofweek
    
    return dataset


## === cell 18
df_train = add_new_date_time_features(df_train)
df_test = add_new_date_time_features(df_test)


## === cell 19
df_train.describe()


## === cell 20
df_train = df_train[df_train.pickup_longitude.between(df_test.pickup_longitude.min(), df_test.pickup_longitude.max())]
df_train = df_train[df_train.pickup_latitude.between(df_test.pickup_latitude.min(), df_test.pickup_latitude.max())]
df_train = df_train[df_train.dropoff_longitude.between(df_test.dropoff_longitude.min(), df_test.dropoff_longitude.max())]
df_train = df_train[df_train.dropoff_latitude.between(df_test.dropoff_latitude.min(), df_test.dropoff_latitude.max())]


## === cell 21
df_train.shape


## === cell 22
def calculate_abs_different(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
calculate_abs_different(df_train)
calculate_abs_different(df_test)


## === cell 23
def convert_different_miles(df):
    df['abs_diff_longitude'] = df.abs_diff_longitude*50
    df['abs_diff_latitude'] = df.abs_diff_latitude*69
convert_different_miles(df_train)
convert_different_miles(df_test)


## === cell 24
meas_ang = 0.506 # 29 degrees = 0.506 radians (https://en.wikipedia.org/wiki/Commissioners%27_Plan_of_1811)
import math


def add_distance(df):
    df['Euclidean'] = (df.abs_diff_latitude**2 + df.abs_diff_longitude**2)**0.5 ### as the crow flies  
    df['delta_manh_long'] = (df.Euclidean*np.sin(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude)-meas_ang)).abs()
    df['delta_manh_lat'] = (df.Euclidean*np.cos(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude)-meas_ang)).abs()
    df['distance'] = df.delta_manh_long + df.delta_manh_lat
    df.drop(['abs_diff_longitude', 'abs_diff_latitude','Euclidean', 'delta_manh_long', 'delta_manh_lat'], axis=1, inplace=True)
add_distance(df_train)
add_distance(df_test)


## === cell 25
df_train.head()


## === cell 26
def calculate_direction(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return distance along great radius between pickup and dropoff coordinates.
    """
    R_earth = 6371
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                             [pickup_lat, pickup_lon, 
                                                              dropoff_lat, dropoff_lon])
    dlat = dropoff_lat - pickup_lat
    dlon = pickup_lon - dropoff_lon
    
    a = np.arctan2(np.sin(dlon * np.cos(dropoff_lat)),np.cos(pickup_lat) * np.sin(dropoff_lat) - np.sin(pickup_lat) * np.cos(dropoff_lat) * np.cos(dlon))
    return a


## === cell 27
train_df['direction'] = calculate_direction(train_df['pickup_latitude'].values, train_df['pickup_longitude'].values, 
                                   train_df['dropoff_latitude'].values , train_df['dropoff_longitude'].values) 
df_test['direction'] = calculate_direction(df_test['pickup_latitude'].values, df_test['pickup_longitude'].values, 
                                   df_test['dropoff_latitude'].values , df_test['dropoff_longitude'].values) 


## === cell 28
train_df['pickup_latitude'].apply(lambda x: np.radians(x))
train_df['pickup_longitude'].apply(lambda x: np.radians(x))
train_df['dropoff_latitude'].apply(lambda x: np.radians(x))
train_df['dropoff_longitude'].apply(lambda x: np.radians(x))

df_test['pickup_latitude'].apply(lambda x: np.radians(x))
df_test['pickup_longitude'].apply(lambda x: np.radians(x))
df_test['dropoff_latitude'].apply(lambda x: np.radians(x))
df_test['dropoff_longitude'].apply(lambda x: np.radians(x))


## === cell 29
sns.jointplot(x='distance', y='fare_amount', data=df_train)


## === cell 30
df_train.drop(columns=['pickup_datetime'], inplace=True)

y = df_train['fare_amount']
df_train = df_train.drop(columns=['fare_amount'])


## === cell 31
df_train.head()


## === cell 32
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
x_train,x_test,y_train,y_test = train_test_split(df_train,y,random_state=123,test_size=0.1)


## === cell 33
params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'nthread': 4,
        'num_leaves': 31,
        'learning_rate': 0.1,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 10000 ,
        'bagging_freq': 10,
        'metric': 'rmse',  
        'zero_as_missing': True,
        'num_rounds':50000
    }


## === cell 34
train_set = lgbm.Dataset(
    x_train, y_train, categorical_feature=["year", "month", "day", "day_of_week"]
)
valid_set = lgbm.Dataset(
    x_test, y_test, categorical_feature=["year", "month", "day", "day_of_week"]
)

model = lgbm.train(
    params,
    train_set=train_set,
    num_boost_round=10000,
    valid_sets=[valid_set],
    callbacks=[
        lgbm.early_stopping(stopping_rounds=1000),
        lgbm.log_evaluation(period=500),
    ],
)


## === cell 35
df_train.describe()


## === cell 36
test_key = df_test['key']

df_test.drop(columns=["pickup_datetime",'key'], axis=1, inplace=True)


## === cell 37
prediction = model.predict(df_test, num_iteration = model.best_iteration)      


## --- ERROR in cell 37, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mLightGBMError[0m                             Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1730928441.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mprediction[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mdf_test[0m[0;34m,[0m [0mnum_iteration[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mbest_iteration[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36mpredict[0;34m(self, data, start_iteration, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, validate_features, **kwargs)[0m
[1;32m   4765[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4766[0m                 [0mnum_iteration[0m [0;34m=[0m [0;34m-[0m[0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4767[0;31m         return predictor.predict(
[0m[1;32m   4768[0m             [0mdata[0m[0;34m=[0m[0mdata[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4769[0m             [0mstart_iteration[0m[0;34m=[0m[0mstart_iteration[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36mpredict[0;34m(self, data, start_iteration, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, validate_features)[0m
[1;32m   1202[0m             )
[1;32m   1203[0m         [0;32melif[0m [0misinstance[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mndarray[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1204[0;31m             preds, nrow = self.__pred_for_np2d(
[0m[1;32m   1205[0m                 [0mmat[0m[0;34m=[0m[0mdata[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1206[0m                 [0mstart_iteration[0m[0;34m=[0m[0mstart_iteration[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m__pred_for_np2d[0;34m(self, mat, start_iteration, num_iteration, predict_type)[0m
[1;32m   1359[0m             [0;32mreturn[0m [0mpreds[0m[0;34m,[0m [0mnrow[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1360[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1361[0;31m             return self.__inner_predict_np2d(
[0m[1;32m   1362[0m                 [0mmat[0m[0;34m=[0m[0mmat[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1363[0m                 [0mstart_iteration[0m[0;34m=[0m[0mstart_iteration[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m__inner_predict_np2d[0;34m(self, mat, start_iteration, num_iteration, predict_type, preds)[0m
[1;32m   1305[0m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"Wrong length of pre-allocated predict array"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1306[0m         [0mout_num_preds[0m [0;34m=[0m [0mctypes[0m[0;34m.[0m[0mc_int64[0m[0;34m([0m[0;36m0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1307[0;31m         _safe_call(
[0m[1;32m   1308[0m             _LIB.LGBM_BoosterPredictForMat(
[1;32m   1309[0m                 [0mself[0m[0;34m.[0m[0m_handle[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m_safe_call[0;34m(ret)[0m
[1;32m    311[0m     """
[1;32m    312[0m     [0;32mif[0m [0mret[0m [0;34m!=[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 313[0;31m         [0;32mraise[0m [0mLightGBMError[0m[0;34m([0m[0m_LIB[0m[0;34m.[0m[0mLGBM_GetLastError[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mdecode[0m[0;34m([0m[0;34m"utf-8"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    314[0m [0;34m[0m[0m
[1;32m    315[0m [0;34m[0m[0m

[0;31mLightGBMError[0m: The number of features in data (12) is not the same as it was in training data (11).
You can set ``predict_disable_shape_check=true`` to discard this error, but please be aware what you are doing.

## === cell 38
submission = pd.DataFrame({
        "key": test_key,
        "fare_amount": prediction
})

submission.to_csv('taxi_fare_submission.csv',index=False)
