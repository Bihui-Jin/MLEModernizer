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
import numpy as np 
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os

print(os.listdir("../input"))


## === cell 1
train_df =  pd.read_csv('../input/train.csv', nrows = 1_000_000)
train_df.dtypes


## === cell 2
print(train_df.isnull().sum())


## === cell 3
train_df = train_df.dropna(how = 'any', axis = 'rows')


## === cell 4
train_df.head()


## === cell 5
train_df.iloc[:1000].plot.scatter('pickup_longitude', 'pickup_latitude')
train_df.iloc[:1000].plot.scatter('dropoff_longitude', 'dropoff_latitude')

train_df.describe()


## === cell 6
def clean_df(df):
    return df[(df.fare_amount > 0) & 
            (df.pickup_longitude > -80) & (df.pickup_longitude < -70) &
            (df.pickup_latitude > 35) & (df.pickup_latitude < 45) &
            (df.dropoff_longitude > -80) & (df.dropoff_longitude < -70) &
            (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45) &
            (df.passenger_count > 0) & (df.passenger_count < 10)]

train_df = clean_df(train_df)
print(len(train_df))


## === cell 7
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R_earth = 6371
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                             [pickup_lat, pickup_lon, 
                                                              dropoff_lat, dropoff_lon])
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2
    
    return 2 * R_earth * np.arcsin(np.sqrt(a))

def add_datetime_info(dataset):
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'])
    
    dataset['hour'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['weekday'] = dataset.pickup_datetime.dt.weekday
    
    return dataset

train_df['distance'] = sphere_dist(train_df['pickup_latitude'], train_df['pickup_longitude'], 
                                   train_df['dropoff_latitude'] , train_df['dropoff_longitude'])

train_df = add_datetime_info(train_df)

train_df.head()


## === cell 8
train_df.drop(columns=['key', 'pickup_datetime'], inplace=True)
train_df.head()


## === cell 9
train_df['pickup_long_15'] = train_df['pickup_longitude']*np.cos(15* np.pi / 180) - train_df['pickup_latitude']*np.sin(15* np.pi/180)
train_df['pickup_long_30'] = train_df['pickup_longitude']*np.cos(30* np.pi / 180) - train_df['pickup_latitude']*np.sin(30* np.pi/180)
train_df['pickup_long_45'] = train_df['pickup_longitude']*np.cos(45* np.pi / 180) - train_df['pickup_latitude']*np.sin(45* np.pi/180)
train_df['pickup_long_60'] = train_df['pickup_longitude']*np.cos(60* np.pi / 180) - train_df['pickup_latitude']*np.sin(60* np.pi/180)
train_df['pickup_long_75'] = train_df['pickup_longitude']*np.cos(75* np.pi / 180) - train_df['pickup_latitude']*np.sin(75* np.pi/180)

train_df['pickup_lat_15'] = train_df['pickup_longitude']*np.sin(15* np.pi / 180) + train_df['pickup_latitude']*np.cos(15* np.pi/180)
train_df['pickup_lat_30'] = train_df['pickup_longitude']*np.sin(30* np.pi / 180) + train_df['pickup_latitude']*np.cos(30* np.pi/180)
train_df['pickup_lat_45'] = train_df['pickup_longitude']*np.sin(45* np.pi / 180) + train_df['pickup_latitude']*np.cos(45* np.pi/180)
train_df['pickup_lat_60'] = train_df['pickup_longitude']*np.sin(60* np.pi / 180) + train_df['pickup_latitude']*np.cos(60* np.pi/180)
train_df['pickup_lat_75'] = train_df['pickup_longitude']*np.sin(75* np.pi / 180) + train_df['pickup_latitude']*np.cos(75* np.pi/180)

train_df['dropoff_long_15'] = train_df['dropoff_longitude']*np.cos(15* np.pi / 180) - train_df['dropoff_latitude']*np.sin(15* np.pi/180)
train_df['dropoff_long_30'] = train_df['dropoff_longitude']*np.cos(30* np.pi / 180) - train_df['dropoff_latitude']*np.sin(30* np.pi/180)
train_df['dropoff_long_45'] = train_df['dropoff_longitude']*np.cos(45* np.pi / 180) - train_df['dropoff_latitude']*np.sin(45* np.pi/180)
train_df['dropoff_long_60'] = train_df['dropoff_longitude']*np.cos(60* np.pi / 180) - train_df['dropoff_latitude']*np.sin(60* np.pi/180)
train_df['dropoff_long_75'] = train_df['dropoff_longitude']*np.cos(75* np.pi / 180) - train_df['dropoff_latitude']*np.sin(75* np.pi/180)

train_df['dropoff_lat_15'] = train_df['dropoff_longitude']*np.sin(15* np.pi / 180) + train_df['dropoff_latitude']*np.cos(15* np.pi/180)
train_df['dropoff_lat_30'] = train_df['dropoff_longitude']*np.sin(30* np.pi / 180) + train_df['dropoff_latitude']*np.cos(30* np.pi/180)
train_df['dropoff_lat_45'] = train_df['dropoff_longitude']*np.sin(45* np.pi / 180) + train_df['dropoff_latitude']*np.cos(45* np.pi/180)
train_df['dropoff_lat_60'] = train_df['dropoff_longitude']*np.sin(60* np.pi / 180) + train_df['dropoff_latitude']*np.cos(60* np.pi/180)
train_df['dropoff_lat_75'] = train_df['dropoff_longitude']*np.sin(75* np.pi / 180) + train_df['dropoff_latitude']*np.cos(75* np.pi/180)


## === cell 10
y = train_df['fare_amount']
train = train_df.drop(columns=['fare_amount'])

x_train,x_test,y_train,y_test = train_test_split(train,y,random_state=0,test_size=0.2)


## === cell 11
def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'},
                    dtrain=matrix_train,num_boost_round=100, 
                    early_stopping_rounds=100,evals=[(matrix_test,'test')])
    return model

model = XGBmodel(x_train,x_test,y_train,y_test)


## === cell 12
test_df =  pd.read_csv('../input/test.csv')
test_df['distance'] = sphere_dist(test_df['pickup_latitude'], test_df['pickup_longitude'], 
                                   test_df['dropoff_latitude'] , test_df['dropoff_longitude'])
test_df = add_datetime_info(test_df)
test_df['pickup_long_15'] = test_df['pickup_longitude']*np.cos(15* np.pi / 180) - test_df['pickup_latitude']*np.sin(15* np.pi/180)
test_df['pickup_long_30'] = test_df['pickup_longitude']*np.cos(30* np.pi / 180) - test_df['pickup_latitude']*np.sin(30* np.pi/180)
test_df['pickup_long_45'] = test_df['pickup_longitude']*np.cos(45* np.pi / 180) - test_df['pickup_latitude']*np.sin(45* np.pi/180)
test_df['pickup_long_60'] = test_df['pickup_longitude']*np.cos(60* np.pi / 180) - test_df['pickup_latitude']*np.sin(60* np.pi/180)
test_df['pickup_long_75'] = test_df['pickup_longitude']*np.cos(75* np.pi / 180) - test_df['pickup_latitude']*np.sin(75* np.pi/180)

test_df['pickup_lat_15'] = test_df['pickup_longitude']*np.sin(15* np.pi / 180) + test_df['pickup_latitude']*np.cos(15* np.pi/180)
test_df['pickup_lat_30'] = test_df['pickup_longitude']*np.sin(30* np.pi / 180) + test_df['pickup_latitude']*np.cos(30* np.pi/180)
test_df['pickup_lat_45'] = test_df['pickup_longitude']*np.sin(45* np.pi / 180) + test_df['pickup_latitude']*np.cos(45* np.pi/180)
test_df['pickup_lat_60'] = test_df['pickup_longitude']*np.sin(60* np.pi / 180) + test_df['pickup_latitude']*np.cos(60* np.pi/180)
test_df['pickup_lat_75'] = test_df['pickup_longitude']*np.sin(75* np.pi / 180) + test_df['pickup_latitude']*np.cos(75* np.pi/180)

test_df['dropoff_long_15'] = test_df['dropoff_longitude']*np.cos(15* np.pi / 180) - test_df['dropoff_latitude']*np.sin(15* np.pi/180)
test_df['dropoff_long_30'] = test_df['dropoff_longitude']*np.cos(30* np.pi / 180) - test_df['dropoff_latitude']*np.sin(30* np.pi/180)
test_df['dropoff_long_45'] = test_df['dropoff_longitude']*np.cos(45* np.pi / 180) - test_df['dropoff_latitude']*np.sin(45* np.pi/180)
test_df['dropoff_long_60'] = test_df['dropoff_longitude']*np.cos(60* np.pi / 180) - test_df['dropoff_latitude']*np.sin(60* np.pi/180)
test_df['dropoff_long_75'] = test_df['dropoff_longitude']*np.cos(75* np.pi / 180) - test_df['dropoff_latitude']*np.sin(75* np.pi/180)

test_df['dropoff_lat_15'] = test_df['dropoff_longitude']*np.sin(15* np.pi / 180) + test_df['dropoff_latitude']*np.cos(15* np.pi/180)
test_df['dropoff_lat_30'] = test_df['dropoff_longitude']*np.sin(30* np.pi / 180) + test_df['dropoff_latitude']*np.cos(30* np.pi/180)
test_df['dropoff_lat_45'] = test_df['dropoff_longitude']*np.sin(45* np.pi / 180) + test_df['dropoff_latitude']*np.cos(45* np.pi/180)
test_df['dropoff_lat_60'] = test_df['dropoff_longitude']*np.sin(60* np.pi / 180) + test_df['dropoff_latitude']*np.cos(60* np.pi/180)
test_df['dropoff_lat_75'] = test_df['dropoff_longitude']*np.sin(75* np.pi / 180) + test_df['dropoff_latitude']*np.cos(75* np.pi/180)

test_key = test_df['key']
x_pred = test_df.drop(columns=['key', 'pickup_datetime'])

prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)


## --- ERROR in cell 12, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/144985416.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     34[0m [0;34m[0m[0m
[1;32m     35[0m [0;31m#Predict from test set[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 36[0;31m [0mprediction[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mxgb[0m[0;34m.[0m[0mDMatrix[0m[0;34m([0m[0mx_pred[0m[0;34m)[0m[0;34m,[0m [0mntree_limit[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mbest_ntree_limit[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mAttributeError[0m: 'Booster' object has no attribute 'best_ntree_limit'

## === cell 13
submission = pd.DataFrame({
        "key": test_key,
        "fare_amount": prediction.round(2)
})

submission.to_csv('taxi_fare_submission.csv',index=False)
submission
