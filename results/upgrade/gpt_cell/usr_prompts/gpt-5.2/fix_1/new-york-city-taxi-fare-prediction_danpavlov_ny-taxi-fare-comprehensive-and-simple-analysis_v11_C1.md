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
geopy==2.4.1
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
from geopy.distance import great_circle
from sklearn import metrics, ensemble, linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


## === cell 1
print(os.listdir("../input"))


## === cell 2
test = pd.read_csv('../input/test.csv')


## === cell 3
test.dtypes


## === cell 4
types = {'fare_amount': 'float32',
         'pickup_longitude': 'float32',
         'pickup_latitude': 'float32',
         'dropoff_longitude': 'float32',
         'dropoff_latitude': 'float32',
         'passenger_count': 'uint8'}


## === cell 5
train = pd.read_csv('../input/train.csv',nrows=500000,dtype=types)


## === cell 6
train.head()


## === cell 7
train.describe()


## === cell 8
sns.distplot(train['fare_amount'])


## === cell 9
sns.distplot(train['passenger_count'])


## === cell 10
train.isnull().sum()


## === cell 11
train.dropna(inplace=True)


## === cell 12
train = train[train['fare_amount'] > 0]
train = train[train['pickup_longitude'] < -72]
train = train[(train['pickup_latitude'] > 40) & (train['pickup_latitude'] < 44)]
train = train[train['dropoff_longitude'] < -72]
train = train[(train['dropoff_latitude'] > 40) & (train['dropoff_latitude'] < 44)]
train = train[(train['passenger_count'] > 0) & (train['passenger_count'] < 10)]


## === cell 14
train.describe()


## === cell 15
def dist_calc(df):
    for i,row in df.iterrows():
        df.at[i,'distance'] = great_circle((row['pickup_latitude'],row['pickup_longitude']),(row['dropoff_latitude'],row['dropoff_longitude'])).km


## === cell 16
dist_calc(train)
dist_calc(test)


## === cell 17
train['pickup_datetime'] = train['pickup_datetime'].str.replace(" UTC", "")
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')


## === cell 18
test['pickup_datetime'] = test['pickup_datetime'].str.replace(" UTC", "")
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')


## === cell 19
train["year"] = train.pickup_datetime.dt.year
train["hour"] = train.pickup_datetime.dt.hour

test["year"] = test.pickup_datetime.dt.year
test["hour"] = test.pickup_datetime.dt.hour


## === cell 20
test.head()


## === cell 21
plt.figure(figsize=(15,8))
sns.heatmap(train.drop(['key','pickup_datetime'],axis=1).corr(),annot=True,fmt='.4f')


## === cell 22
X = train.drop(['key','fare_amount','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1)
y = train['fare_amount']


## === cell 23
X.head()


## === cell 24
y.head()


## === cell 25
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


## === cell 26
test_pred = test.drop(['key','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1)


## === cell 28
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.score(X_train,y_train))
print(lm.score(X_test,y_test))


## === cell 29
y_pred = lm.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
lrmse


## === cell 30
LinearPredictions = lm.predict(test_pred)
LinearPredictions = np.round(LinearPredictions, decimals=2)
LinearPredictions


## === cell 31
LinearPredictions.size


## === cell 32
linear_submission = pd.DataFrame({"key": test['key'],"fare_amount": LinearPredictions},columns = ['key','fare_amount'])


## === cell 33
linear_submission.head()


## === cell 34
def XGBoost(X_train,X_test,y_train,y_test):
    dtrain = xgb.DMatrix(X_train,label=y_train)
    dtest = xgb.DMatrix(X_test,label=y_test)

    return xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'}
                    ,dtrain=dtrain,num_boost_round=300, 
                    early_stopping_rounds=20,evals=[(dtest,'test')],)


## === cell 35
xgbm = XGBoost(X_train,X_test,y_train,y_test)
XGBPredictions = xgbm.predict(xgb.DMatrix(test_pred), ntree_limit = xgbm.best_ntree_limit)


## --- ERROR in cell 35, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1472417481.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m#Non-scaled values[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0mxgbm[0m [0;34m=[0m [0mXGBoost[0m[0;34m([0m[0mX_train[0m[0;34m,[0m[0mX_test[0m[0;34m,[0m[0my_train[0m[0;34m,[0m[0my_test[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m [0mXGBPredictions[0m [0;34m=[0m [0mxgbm[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mxgb[0m[0;34m.[0m[0mDMatrix[0m[0;34m([0m[0mtest_pred[0m[0;34m)[0m[0;34m,[0m [0mntree_limit[0m [0;34m=[0m [0mxgbm[0m[0;34m.[0m[0mbest_ntree_limit[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mAttributeError[0m: 'Booster' object has no attribute 'best_ntree_limit'

## === cell 36
XGBPredictions
