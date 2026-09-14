# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.8987

# 6. Current score

5.19425

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
object_data = train.dtypes == np.object
categoricals = train.columns[object_data]
categoricals


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/983781536.py in <cell line: 0>()
      1 #lets take care of the object data first
----> 2 object_data = train.dtypes == np.object
      3 categoricals = train.columns[object_data]
      4 categoricals

/usr/local/lib/python3.11/dist-packages/numpy/__init__.py in __getattr__(attr)
    322 
    323         if attr in __former_attrs__:
--> 324             raise AttributeError(__former_attrs__[attr])
    325 
    326         if attr == 'testing':

AttributeError: module 'numpy' has no attribute 'object'.
`np.object` was a deprecated alias for the builtin `object`. To avoid this error in existing code, use `object` by itself. Doing this will not modify any behavior and is safe. 
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations

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
alphas = np.array([0.05,0.03, 0.01, 0.5,0.3, 0.1, 1,3, 5])
la = LassoCV(alphas = alphas ,max_iter = 5e4, cv = 4).fit(X_train, y_train)
la_rmse = rmse(y_test, la.predict(X_test))
print(la.alpha_, la_rmse)


## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
InvalidParameterError                     Traceback (most recent call last)
/tmp/ipykernel_11/3472443304.py in <cell line: 0>()
      1 #Lasso CV
      2 alphas = np.array([0.05,0.03, 0.01, 0.5,0.3, 0.1, 1,3, 5])
----> 3 la = LassoCV(alphas = alphas ,max_iter = 5e4, cv = 4).fit(X_train, y_train)
      4 la_rmse = rmse(y_test, la.predict(X_test))
      5 print(la.alpha_, la_rmse)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_coordinate_descent.py in fit(self, X, y, sample_weight)
   1505         """
   1506 
-> 1507         self._validate_params()
   1508 
   1509         # This makes sure that there is no duplication in memory.

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_params(self)
    598         accepted constraints.
    599         """
--> 600         validate_parameter_constraints(
    601             self._parameter_constraints,
    602             self.get_params(deep=False),

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_param_validation.py in validate_parameter_constraints(parameter_constraints, params, caller_name)
     95                 )
     96 
---> 97             raise InvalidParameterError(
     98                 f"The {param_name!r} parameter of {caller_name} must be"
     99                 f" {constraints_str}. Got {param_val!r} instead."

InvalidParameterError: The 'max_iter' parameter of LassoCV must be an int in the range [1, inf). Got 50000.0 instead.

## === cell 53

l1_ratios = np.linspace(0.1, 0.5, 5)
alphas = np.array([0.05,0.03, 0.01, 0.5,0.3, 0.1, 1,3, 5])

en = ElasticNetCV(alphas=alphas, 
                            l1_ratio=l1_ratios,
                            max_iter=1e4).fit(X_train, y_train)
en_rmse = rmse(y_test, en.predict(X_test))

print(en.alpha_, en.l1_ratio_, en_rmse)


## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
InvalidParameterError                     Traceback (most recent call last)
/tmp/ipykernel_11/2606884394.py in <cell line: 0>()
      6 en = ElasticNetCV(alphas=alphas, 
      7                             l1_ratio=l1_ratios,
----> 8                             max_iter=1e4).fit(X_train, y_train)
      9 en_rmse = rmse(y_test, en.predict(X_test))
     10 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_coordinate_descent.py in fit(self, X, y, sample_weight)
   1505         """
   1506 
-> 1507         self._validate_params()
   1508 
   1509         # This makes sure that there is no duplication in memory.

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_params(self)
    598         accepted constraints.
    599         """
--> 600         validate_parameter_constraints(
    601             self._parameter_constraints,
    602             self.get_params(deep=False),

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_param_validation.py in validate_parameter_constraints(parameter_constraints, params, caller_name)
     95                 )
     96 
---> 97             raise InvalidParameterError(
     98                 f"The {param_name!r} parameter of {caller_name} must be"
     99                 f" {constraints_str}. Got {param_val!r} instead."

InvalidParameterError: The 'max_iter' parameter of ElasticNetCV must be an int in the range [1, inf). Got 10000.0 instead.

## === cell 54
rf = RandomForestRegressor(n_estimators = 100, max_features = 5)
rf = rf.fit(X_train, y_train)


## === cell 55

labels = ['Linear','lasso', 'Ridge', 'Elastic-Net']
models_rmse = [ lr_rmse, la_rmse, rr_rmse, en_rmse]
rmse_df = pd.Series(models_rmse, index = labels).to_frame()
rmse_df.rename(columns = {0: 'Errors'}, inplace = True)
rmse_df


## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1529119540.py in <cell line: 0>()
      2 
      3 labels = ['Linear','lasso', 'Ridge', 'Elastic-Net']
----> 4 models_rmse = [ lr_rmse, la_rmse, rr_rmse, en_rmse]
      5 rmse_df = pd.Series(models_rmse, index = labels).to_frame()
      6 rmse_df.rename(columns = {0: 'Errors'}, inplace = True)

NameError: name 'la_rmse' is not defined

## === cell 57
test.head()


## === cell 58
test_1.drop('key', axis = 1, inplace = True)
test_1.head()


## === cell 59
final_prediction = rf.predict(test_1)

NYCtaxiFare_submission = pd.DataFrame({'key': test.key, 'fare_amount': final_prediction})
NYCtaxiFare_submission.to_csv('NYCtaxiFare_prediction.csv', index = False)


## === cell 60
NYCtaxiFare_submission.head()
