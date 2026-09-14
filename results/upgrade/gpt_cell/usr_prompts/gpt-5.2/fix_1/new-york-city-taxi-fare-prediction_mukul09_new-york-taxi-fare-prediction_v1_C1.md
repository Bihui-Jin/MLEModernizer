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

from math import radians, cos, sin, asin, sqrt

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GridSearchCV


import warnings
warnings.filterwarnings("ignore")


## === cell 1
original_train_data = pd.read_csv('../input/train.csv', nrows=6000000)
train_data = original_train_data.sample(n=100000)
train_data.info()


## === cell 2
test_data = pd.read_csv('../input/test.csv')
test_data.info()


## === cell 3
train_data.isnull().sum()


## === cell 4
train_data.dropna(axis=0,inplace=True)


## === cell 5
train_data.describe()


## === cell 6
train_data = train_data[train_data['fare_amount']>0]

train_data = train_data[(train_data['passenger_count']<=6)& (train_data['passenger_count']>0)]

train_data = train_data[(train_data['pickup_latitude']>-90)| (train_data['pickup_latitude']<=90)]
train_data = train_data[(train_data['dropoff_latitude']>-90)| (train_data['dropoff_latitude']<=90)]

train_data = train_data[(train_data['pickup_longitude']>=-180)| (train_data['pickup_longitude']<=180)]
train_data = train_data[(train_data['dropoff_longitude']>=-180)| (train_data['dropoff_longitude']<=180)]


## === cell 7
train_data.shape


## === cell 8
train_data.info()


## === cell 9
train_data.head(5)


## === cell 10
def distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return distance along great radius between pickup and dropoff coordinates.
    """
    R_earth = 6371
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                             [pickup_lat, pickup_lon, 
                                                              dropoff_lat, dropoff_lon])
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2
    
    return 2 * R_earth * np.arcsin(np.sqrt(a))
    
def date_time_info(data):
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'], format="%Y-%m-%d %H:%M:%S UTC")
    
    data['hour'] = data['pickup_datetime'].dt.hour
    data['day']  = data['pickup_datetime'].dt.day
    data['month'] = data['pickup_datetime'].dt.month
    data['weekday'] = data['pickup_datetime'].dt.weekday
    data['year']    = data['pickup_datetime'].dt.year
    
    return data


train_data = date_time_info(train_data)
train_data['distance'] = distance(train_data['pickup_latitude'], 
                                     train_data['pickup_longitude'],
                                     train_data['dropoff_latitude'] ,
                                     train_data['dropoff_longitude'])

train_data.head()


## === cell 11
train_data.drop(['key', 'pickup_datetime'],axis =1, inplace = True)
train_data.head()


## === cell 12
test_data.head()


## === cell 13
test_data = date_time_info(test_data)
test_data['distance'] = distance(test_data['pickup_latitude'], test_data['pickup_longitude'], 
                                   test_data['dropoff_latitude'] , test_data['dropoff_longitude'])

test_key = test_data['key']
x_pred = test_data.drop(columns=['key', 'pickup_datetime'])


## === cell 14
y = train_data['fare_amount']
X = train_data.drop(['fare_amount'],axis=1)

cv_split = KFold(n_splits=10,random_state=0)


## --- ERROR in cell 14, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4068645844.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0mX[0m [0;34m=[0m [0mtrain_data[0m[0;34m.[0m[0mdrop[0m[0;34m([0m[0;34m[[0m[0;34m'fare_amount'[0m[0;34m][0m[0;34m,[0m[0maxis[0m[0;34m=[0m[0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;34m[0m[0m
[0;32m----> 4[0;31m [0mcv_split[0m [0;34m=[0m [0mKFold[0m[0;34m([0m[0mn_splits[0m[0;34m=[0m[0;36m10[0m[0;34m,[0m[0mrandom_state[0m[0;34m=[0m[0;36m0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py[0m in [0;36m__init__[0;34m(self, n_splits, shuffle, random_state)[0m
[1;32m    449[0m [0;34m[0m[0m
[1;32m    450[0m     [0;32mdef[0m [0m__init__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mn_splits[0m[0;34m=[0m[0;36m5[0m[0;34m,[0m [0;34m*[0m[0;34m,[0m [0mshuffle[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m [0mrandom_state[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 451[0;31m         [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__init__[0m[0;34m([0m[0mn_splits[0m[0;34m=[0m[0mn_splits[0m[0;34m,[0m [0mshuffle[0m[0;34m=[0m[0mshuffle[0m[0;34m,[0m [0mrandom_state[0m[0;34m=[0m[0mrandom_state[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    452[0m [0;34m[0m[0m
[1;32m    453[0m     [0;32mdef[0m [0m_iter_test_indices[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mX[0m[0;34m,[0m [0my[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mgroups[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py[0m in [0;36m__init__[0;34m(self, n_splits, shuffle, random_state)[0m
[1;32m    306[0m [0;34m[0m[0m
[1;32m    307[0m         [0;32mif[0m [0;32mnot[0m [0mshuffle[0m [0;32mand[0m [0mrandom_state[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m  [0;31m# None is the default[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 308[0;31m             raise ValueError(
[0m[1;32m    309[0m                 [0;34m"Setting a random_state has no effect since shuffle is "[0m[0;34m[0m[0;34m[0m[0m
[1;32m    310[0m                 [0;34m"False. You should leave "[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.

## === cell 15
xgb = XGBRegressor(random_state=0)
base_results = cross_validate(xgb, X,y, cv = cv_split)
xgb.fit(X,y)
