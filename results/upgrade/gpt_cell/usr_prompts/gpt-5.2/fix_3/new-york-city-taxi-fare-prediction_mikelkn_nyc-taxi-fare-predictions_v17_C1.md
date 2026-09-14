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
train = pd.read_csv('../input/train.csv', nrows = 1000000)

test = pd.read_csv('../input/test.csv')

train.head()


## === cell 2
test.head()


## === cell 3
train.shape


## === cell 4
test.shape


## === cell 5
train.dtypes.value_counts()


## === cell 6
test.dtypes.value_counts()


## === cell 7
train.isnull().sum()


## === cell 8
train = train.dropna()
train.isnull().sum()


## === cell 9
(train ==0).astype(int).sum()


## === cell 10
train = train.loc[~(train==0).any(axis =1)]


## === cell 11
(train ==0).astype(int).sum()


## === cell 12
train.shape


## === cell 13
train.describe()


## === cell 16
train.describe()


## === cell 17
train.dtypes.value_counts()


## === cell 18
object_data = train.dtypes == object
categoricals = train.columns[object_data]
categoricals


## === cell 19
train.drop('key', axis = 1, inplace = True)
train.head()


## === cell 20
import datetime as dt

def date_extraction(data):
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    data['year'] = data['pickup_datetime'].dt.year
    data['month'] = data['pickup_datetime'].dt.month
    data['weekday'] = data['pickup_datetime'].dt.day
    data['hour'] = data['pickup_datetime'].dt.hour
    data = data.drop('pickup_datetime', axis = 1, inplace = True)
    
    return data
    
date_extraction(train)
    


## === cell 21
train.head()


## === cell 22
date_extraction(test)
test.head()


## === cell 24
def long_lat_distance (x):
    x['Longitude_distance'] = np.radians(x['pickup_longitude'] - x['dropoff_longitude'])
    x['Latitude_distance'] = np.radians(x['pickup_latitude'] - x['dropoff_latitude']) 
    x['distance_travelled/10e3'] = ((x['Longitude_distance']**2 + x['Latitude_distance']**2)**0.5) *1000
    return x   


## === cell 25
for x in [train, test]:
    long_lat_distance(x)
    
train.head()


## === cell 26
def harvesine(x):
    r = 6371000 
    d = x['distance_travelled/10e3']
    theta_1 = np.radians(x['dropoff_latitude'])
    theta_2 = np.radians(x['pickup_latitude'])
    lambda_1 = np.radians(x['dropoff_longitude'])
    lambda_2 = np.radians(x['dropoff_longitude'])
    theta_diff = x['Longitude_distance']
    lambda_diff = x['Latitude_distance']
    
    a = np.sin(theta_diff/2)**2 + np.cos(theta_1)*np.cos(theta_2)*np.sin(lambda_diff/2)**2
    c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
    x['harvesine/km'] = (r * c)/1000


## === cell 27
for x in [train, test]:
    harvesine(x)
    
train.head()


## === cell 28
train.dtypes.value_counts()


## === cell 30
train.head()


## === cell 31
test.head()


## === cell 32
train.describe()


## === cell 33
print('Are there any nulls\nan in the train data: ')
print(train.isnull().sum())

print('\nAre there any nulls\nans in the test data: ')
print(test.isnull().sum())


## === cell 34
train['harvesine/km'] = train['harvesine/km'].fillna(train['harvesine/km'].median())


## === cell 35
from sklearn.ensemble import RandomForestRegressor

feature_cols = [x for x in train.columns if x!= 'fare_amount']
X = train[feature_cols]
y = train['fare_amount']


## === cell 36
correlations = X.corrwith(y)
correlations = abs(correlations*100)
correlations.sort_values(ascending = False, inplace= True)

correlations


## === cell 37
ax = correlations.plot(kind='bar')
ax.set(ylim=[-1, 1], ylabel='pearson correlation');


## === cell 38
train.head()


## === cell 40

train_1 = train.drop(['pickup_longitude', 'dropoff_longitude','pickup_latitude','dropoff_latitude',
                    'Longitude_distance', 'Latitude_distance'], axis =1)

train_1.head()


## === cell 41
train_1['harvesine/km'] =train_1['harvesine/km'].round(2) 
train_1['distance_travelled/10e3'] =train_1['distance_travelled/10e3'].round(2) 

train_1.head()


## === cell 43
train_1.describe()


## === cell 45
test_1 = test.drop(['pickup_longitude', 'dropoff_longitude','pickup_latitude','dropoff_latitude',
                    'Longitude_distance', 'Latitude_distance'], axis =1)

test_1.head()


## === cell 46
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV, RidgeCV
import matplotlib.pyplot as plt
import seaborn as sns


## === cell 47

def rmse(ytrue, ypredicted):
    return np.sqrt(mean_squared_error(ytrue, ypredicted))


## === cell 48
feat_cols = [x for x in train_1.columns if x!= 'fare_amount']
X_1 = train_1[feat_cols]
y_1 = train_1['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size = 0.25, random_state = 42)


## === cell 49

lr = LinearRegression().fit(X_train, y_train)
lr_rmse = rmse(y_test, lr.predict(X_test))
print(lr_rmse)


## === cell 50
%matplotlib inline
f = plt.figure(figsize=(6,6))
ax = plt.axes()

ax.plot(y_test, lr.predict(X_test), marker = 'o', ls = '')
lim = (0, y_test.max())
ax.set(xlabel =' actual fare amount',
      ylabel = 'predicted_amount',
      xlim = lim,
      ylim = (0, 100), 
      title = 'Linear regression results');


## === cell 51
alphas = [0.05,0.03, 0.01, 0.5,0.3, 0.1, 1,3, 5]
rr = RidgeCV(alphas=alphas , cv = 4).fit(X_train, y_train)
rr_rmse = rmse(y_test, rr.predict(X_test))

print(rr.alpha_, rr_rmse)


## === cell 52
alphas = np.array([0.05, 0.03, 0.01, 0.5, 0.3, 0.1, 1, 3, 5])

la = LassoCV(alphas=alphas, max_iter=int(5e4), cv=4).fit(X_train, y_train)
la_rmse = rmse(y_test, la.predict(X_test))
print(la.alpha_, la_rmse)


## === cell 53

l1_ratios = np.linspace(0.1, 0.5, 5)
alphas = np.array([0.05,0.03, 0.01, 0.5,0.3, 0.1, 1,3, 5])

en = ElasticNetCV(alphas=alphas, 
                            l1_ratio=l1_ratios,
                            max_iter=1e4).fit(X_train, y_train)
en_rmse = rmse(y_test, en.predict(X_test))

print(en.alpha_, en.l1_ratio_, en_rmse)


## --- ERROR in cell 53, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mInvalidParameterError[0m                     Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2606884394.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      6[0m en = ElasticNetCV(alphas=alphas, 
[1;32m      7[0m                             [0ml1_ratio[0m[0;34m=[0m[0ml1_ratios[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 8[0;31m                             max_iter=1e4).fit(X_train, y_train)
[0m[1;32m      9[0m [0men_rmse[0m [0;34m=[0m [0mrmse[0m[0;34m([0m[0my_test[0m[0;34m,[0m [0men[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mX_test[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_coordinate_descent.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight)[0m
[1;32m   1505[0m         """
[1;32m   1506[0m [0;34m[0m[0m
[0;32m-> 1507[0;31m         [0mself[0m[0;34m.[0m[0m_validate_params[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1508[0m [0;34m[0m[0m
[1;32m   1509[0m         [0;31m# This makes sure that there is no duplication in memory.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/base.py[0m in [0;36m_validate_params[0;34m(self)[0m
[1;32m    598[0m         [0maccepted[0m [0mconstraints[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    599[0m         """
[0;32m--> 600[0;31m         validate_parameter_constraints(
[0m[1;32m    601[0m             [0mself[0m[0;34m.[0m[0m_parameter_constraints[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    602[0m             [0mself[0m[0;34m.[0m[0mget_params[0m[0;34m([0m[0mdeep[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/_param_validation.py[0m in [0;36mvalidate_parameter_constraints[0;34m(parameter_constraints, params, caller_name)[0m
[1;32m     95[0m                 )
[1;32m     96[0m [0;34m[0m[0m
[0;32m---> 97[0;31m             raise InvalidParameterError(
[0m[1;32m     98[0m                 [0;34mf"The {param_name!r} parameter of {caller_name} must be"[0m[0;34m[0m[0;34m[0m[0m
[1;32m     99[0m                 [0;34mf" {constraints_str}. Got {param_val!r} instead."[0m[0;34m[0m[0;34m[0m[0m

[0;31mInvalidParameterError[0m: The 'max_iter' parameter of ElasticNetCV must be an int in the range [1, inf). Got 10000.0 instead.

## === cell 54
rf = RandomForestRegressor(n_estimators = 100, max_features = 5)
rf = rf.fit(X_train, y_train)
