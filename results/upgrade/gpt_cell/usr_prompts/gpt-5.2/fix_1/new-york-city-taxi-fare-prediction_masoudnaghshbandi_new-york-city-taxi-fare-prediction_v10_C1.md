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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as MSE


## === cell 2
train = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv", nrows = 1000000)
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")


## === cell 3
train.shape, test.shape


## === cell 4
train.head()


## === cell 5
train.isnull().sum()


## === cell 6
train = train.dropna(how = 'any', axis = 'rows')


## === cell 7
test.isnull().sum()


## === cell 8
train.head()


## === cell 9
train['fare_amount'].describe()


## === cell 10
train.drop(train[train['pickup_longitude'] == 0].index, axis=0, inplace = True)
train.drop(train[train['pickup_latitude'] == 0].index, axis=0, inplace = True)
train.drop(train[train['dropoff_longitude'] == 0].index, axis=0, inplace = True)
train.drop(train[train['dropoff_latitude'] == 0].index, axis=0, inplace = True)
train.drop(train[train['passenger_count'] == 208].index, axis=0, inplace = True)
train.drop(train[train['passenger_count'] > 5].index, axis=0, inplace = True)
train.drop(train[train['passenger_count'] == 0].index, axis=0, inplace = True)


## === cell 11
train.head()


## === cell 12
train.drop(['key'], axis=1,inplace=True)


## === cell 13
train.drop(['pickup_datetime'], axis=1,inplace=True)


## === cell 14
train.dropna(inplace=True)

train.drop(train.index[(train.pickup_longitude < -75) | 
           (train.pickup_longitude > -72) | 
           (train.pickup_latitude < 40) | 
           (train.pickup_latitude > 42)],inplace=True)
train.drop(train.index[(train.dropoff_longitude < -75) | 
           (train.dropoff_longitude > -72) | 
           (train.dropoff_latitude < 40) | 
           (train.dropoff_latitude > 42)],inplace=True)


## === cell 15
train.head()


## === cell 16
X, y = train.drop('fare_amount', axis = 1), train['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)


## === cell 17
scaler = StandardScaler()


## === cell 18
scaler.fit_transform(X_train,X_test)


## === cell 19
xgb_r = xgb.XGBRegressor(objective ='reg:linear',
                  n_estimators = 200, seed = 123)


## === cell 20
xgb_r.fit(X_train,y_train)


## === cell 21
y_pred = xgb_r.predict(X_test)


## === cell 22
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE : % f" %(rmse))


## --- ERROR in cell 22, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/790994069.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mrmse[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0msqrt[0m[0;34m([0m[0mMSE[0m[0;34m([0m[0my_test[0m[0;34m,[0m [0mpred[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mprint[0m[0;34m([0m[0;34m"RMSE : % f"[0m [0;34m%[0m[0;34m([0m[0mrmse[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: name 'pred' is not defined

## === cell 23
test.drop(['key'], axis=1,inplace=True)
