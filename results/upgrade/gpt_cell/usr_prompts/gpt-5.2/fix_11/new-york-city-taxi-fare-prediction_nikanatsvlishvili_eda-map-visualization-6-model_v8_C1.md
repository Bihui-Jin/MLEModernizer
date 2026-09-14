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

3.10

# 2. Installed packages

folium==0.20.0
geopandas==0.14.4
haversine==2.9.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")
from mpl_toolkits import mplot3d
import seaborn as sns

import math
from math import sqrt

from numpy import absolute
from numpy import mean
from numpy import std

from sklearn import metrics
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn import neighbors

from sklearn.preprocessing import StandardScaler


## === cell 2
train = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv", nrows = 100000, parse_dates=["pickup_datetime"]
                   )
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")


## === cell 3
print(train.shape)
print(test.shape)


## === cell 4
train.head()


## === cell 5
train.dtypes


## === cell 6
train.describe()


## === cell 7
print(train.isnull().sum())


## === cell 8
train = train.dropna(how = 'any', axis = 'rows')


## === cell 9
print('Old size: %d' % len(train))


## === cell 10
train = train.drop(train[train.fare_amount<2.5].index, axis = 0)
train = train.drop(train[train.fare_amount>300].index, axis = 0)


## === cell 11
train = train.drop(train[train['passenger_count']>6].index, axis = 0)
train = train.drop(train[train['passenger_count']<0].index, axis = 0)


## === cell 12
train = train.drop(train[train['pickup_latitude']<-90].index, axis = 0)
train = train.drop(train[train['pickup_latitude']>90].index, axis = 0)


## === cell 13
train = train.drop(train[train['pickup_longitude']<-180].index, axis = 0)
train = train.drop(train[train['pickup_longitude']>180].index, axis = 0)


## === cell 14
train = train.drop(train[train['dropoff_latitude']<-90].index, axis = 0)
train = train.drop(train[train['dropoff_latitude']>90].index, axis = 0)

train = train.drop(train[train['dropoff_longitude']<-180].index, axis = 0)
train = train.drop(train[train['dropoff_longitude']>180].index, axis = 0)


## === cell 15
def select_outside_boundingbox(df, BB):
    filter_df = df.loc[(df['pickup_longitude'] < BB[0]) | (df['pickup_longitude'] > BB[1]) | \
           (df['pickup_latitude'] < BB[2]) | (df['pickup_latitude'] > BB[3]) | \
           (df['dropoff_longitude'] < BB[0]) | (df['dropoff_longitude'] > BB[1]) | \
           (df['dropoff_latitude'] < BB[2]) | (df['dropoff_latitude'] > BB[3])]
    
    return filter_df

NYC_BB = (-74.5, -72.8, 40.5, 41.8)


## === cell 16
outliers = select_outside_boundingbox(train, NYC_BB)
outliers


## === cell 17
train = train.drop(outliers.index, axis = 0)


## === cell 18
print('New size: %d' % len(train))


## === cell 20
test.dtypes


## === cell 21
train["loc1"] = train[["pickup_latitude","pickup_longitude"]].apply(tuple, axis=1)
train["loc2"] = train[["dropoff_latitude","dropoff_longitude"]].apply(tuple, axis=1)


## === cell 22
import haversine as hs

        
train['H_Distance'] = train.apply(lambda row: hs.haversine(row.loc1,row.loc2), axis=1)


## === cell 23
def chebyshev(pickup_long, dropoff_long, pickup_lat, dropoff_lat):
    return np.maximum(np.absolute(pickup_long - dropoff_long), np.absolute(pickup_lat - dropoff_lat))

train['Chebyshev'] = chebyshev(train['pickup_longitude'], train['dropoff_longitude'], train['pickup_latitude'], train['dropoff_latitude'])


## === cell 24
train.head()


## === cell 25
train["hour"] = train.pickup_datetime.dt.hour
train["day_of_week"] = train.pickup_datetime.dt.weekday
train["day_of_month"] = train.pickup_datetime.dt.day
train["week"] = train.pickup_datetime.dt.isocalendar().week.astype(int)
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year - 2000

train["minute"] = train["pickup_datetime"].dt.minute
train["second"] = train["pickup_datetime"].dt.second
train["dayofyear"] = train["pickup_datetime"].dt.dayofyear


## === cell 26
train.head()


## === cell 27
train["fare_to_dist_ratio"] = train["fare_amount"] / ( train["H_Distance"]+0.0001)


## === cell 28
train = train.drop(train[train['loc1']==train['loc2']].index, axis = 0)


## === cell 29
def add_distances_from_airport(dataset):
    jfk_coords = (40.639722, -73.778889)
    ewr_coords = (40.6925, -74.168611)
    lga_coords = (40.77725, -73.872611)

    dataset['pickup_jfk_distance'] = dataset.apply(lambda row: hs.haversine(jfk_coords,row.loc1), axis=1)
    dataset['dropof_jfk_distance'] = dataset.apply(lambda row: hs.haversine(jfk_coords,row.loc2), axis=1)
    
    
    dataset['pickup_ewr_distance'] = dataset.apply(lambda row: hs.haversine(ewr_coords,row.loc1), axis=1)
    dataset['dropof_ewr_distance'] = dataset.apply(lambda row: hs.haversine(ewr_coords,row.loc2), axis=1)
    
    
    dataset['pickup_lga_distance'] = dataset.apply(lambda row: hs.haversine(lga_coords,row.loc1), axis=1)
    dataset['dropof_lga_distance'] = dataset.apply(lambda row: hs.haversine(lga_coords,row.loc2), axis=1)

    return dataset


train = add_distances_from_airport(train)


## === cell 30
test['pickup_datetime']  = pd.to_datetime(test['pickup_datetime'])

test["loc1"] = test[["pickup_latitude","pickup_longitude"]].apply(tuple, axis=1)
test["loc2"] = test[["dropoff_latitude","dropoff_longitude"]].apply(tuple, axis=1)

test['H_Distance'] = test.apply(lambda row: hs.haversine(row.loc1,row.loc2), axis=1)

test['Chebyshev'] = chebyshev(test['pickup_longitude'], test['dropoff_longitude'], test['pickup_latitude'], test['dropoff_latitude'])


test["hour"] = test.pickup_datetime.dt.hour
test["day_of_week"] = test.pickup_datetime.dt.weekday
test["day_of_month"] = test.pickup_datetime.dt.day
test["week"] = test.pickup_datetime.dt.week
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year - 2000

test['minute'] =test['pickup_datetime'].dt.minute
test['second'] = test['pickup_datetime'].dt.second
test['dayofyear'] = test['pickup_datetime'].dt.dayofyear

test = add_distances_from_airport(test)


## --- ERROR in cell 30, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1534197004.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     12[0m [0mtest[0m[0;34m[[0m[0;34m"day_of_week"[0m[0;34m][0m [0;34m=[0m [0mtest[0m[0;34m.[0m[0mpickup_datetime[0m[0;34m.[0m[0mdt[0m[0;34m.[0m[0mweekday[0m[0;34m[0m[0;34m[0m[0m
[1;32m     13[0m [0mtest[0m[0;34m[[0m[0;34m"day_of_month"[0m[0;34m][0m [0;34m=[0m [0mtest[0m[0;34m.[0m[0mpickup_datetime[0m[0;34m.[0m[0mdt[0m[0;34m.[0m[0mday[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 14[0;31m [0mtest[0m[0;34m[[0m[0;34m"week"[0m[0;34m][0m [0;34m=[0m [0mtest[0m[0;34m.[0m[0mpickup_datetime[0m[0;34m.[0m[0mdt[0m[0;34m.[0m[0mweek[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     15[0m [0mtest[0m[0;34m[[0m[0;34m"month"[0m[0;34m][0m [0;34m=[0m [0mtest[0m[0;34m.[0m[0mpickup_datetime[0m[0;34m.[0m[0mdt[0m[0;34m.[0m[0mmonth[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m [0mtest[0m[0;34m[[0m[0;34m"year"[0m[0;34m][0m [0;34m=[0m [0mtest[0m[0;34m.[0m[0mpickup_datetime[0m[0;34m.[0m[0mdt[0m[0;34m.[0m[0myear[0m [0;34m-[0m [0;36m2000[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'DatetimeProperties' object has no attribute 'week'

## === cell 31
def downcast(df):

    df_int = df.select_dtypes(include=['int64', 'int32', 'int16', 'int8', 'int'])
    df[df_int.columns] = df_int.apply(pd.to_numeric,downcast='unsigned')
    
    df_float = df.select_dtypes(include=['float64', 'float32', 'float16', 'float'])
    df[df_float.columns] = df_float.apply(pd.to_numeric,downcast='float')
        
    return df
downcast(train)
downcast(test)
train.dtypes
