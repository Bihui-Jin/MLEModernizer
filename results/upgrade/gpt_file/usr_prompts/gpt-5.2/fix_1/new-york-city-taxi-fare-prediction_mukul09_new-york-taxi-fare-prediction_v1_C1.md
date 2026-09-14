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

3.27133

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4068645844.py in <cell line: 0>()
      2 X = train_data.drop(['fare_amount'],axis=1)
      3 
----> 4 cv_split = KFold(n_splits=10,random_state=0)

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in __init__(self, n_splits, shuffle, random_state)
    449 
    450     def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
--> 451         super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    452 
    453     def _iter_test_indices(self, X, y=None, groups=None):

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in __init__(self, n_splits, shuffle, random_state)
    306 
    307         if not shuffle and random_state is not None:  # None is the default
--> 308             raise ValueError(
    309                 "Setting a random_state has no effect since shuffle is "
    310                 "False. You should leave "

ValueError: Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.

## === cell 15
xgb = XGBRegressor(random_state=0)
base_results = cross_validate(xgb, X,y, cv = cv_split)
xgb.fit(X,y)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1612077715.py in <cell line: 0>()
      1 xgb = XGBRegressor(random_state=0)
----> 2 base_results = cross_validate(xgb, X,y, cv = cv_split)
      3 xgb.fit(X,y)

NameError: name 'cv_split' is not defined

## === cell 16
print('Best XGB parameters: ', xgb.get_params())
print('Before XGB Training score mean: {:.2f}'.format(base_results['train_score'].mean()*100))
print('Before XGB Training score mean: {:.2f}'.format(base_results['test_score'].mean()*100))
print('#'*20)


print('After XGB Parameters: ', tune_model.best_params_)
print('After XGB Training score mean: {:.2f}'.format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print('After XGB Testing score mean: {:.2f}'.format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_].mean()*100))
print('#'*20)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/766581707.py in <cell line: 0>()
      1 print('Best XGB parameters: ', xgb.get_params())
----> 2 print('Before XGB Training score mean: {:.2f}'.format(base_results['train_score'].mean()*100))
      3 print('Before XGB Training score mean: {:.2f}'.format(base_results['test_score'].mean()*100))
      4 print('#'*20)
      5 

NameError: name 'base_results' is not defined

## === cell 17
params = {
    'max_depth': [8], #Result of tuning with CV
    'eta':[.03], #Result of tuning with CV
    'subsample': [1], #Result of tuning with CV
    'colsample_bytree': [0.8], #Result of tuning with CV
    'objective':['reg:linear'],
    'eval_metric':['rmse'],
    'silent': [1]
}

submit_xgb = GridSearchCV(XGBRegressor(), param_grid=params,
                                         scoring = 'neg_mean_squared_error',
                          cv = cv_split)
submit_xgb.fit(X,y)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2705247093.py in <cell line: 0>()
     14 submit_xgb = GridSearchCV(XGBRegressor(), param_grid=params,
     15                                          scoring = 'neg_mean_squared_error',
---> 16                           cv = cv_split)
     17 submit_xgb.fit(X,y)

NameError: name 'cv_split' is not defined

## === cell 18
prediction = submit_xgb.predict(x_pred)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3542946799.py in <cell line: 0>()
----> 1 prediction = submit_xgb.predict(x_pred)

NameError: name 'submit_xgb' is not defined

## === cell 19
submission = pd.DataFrame({
        "key": test_key,
        "fare_amount": prediction.round(2)
})

submission.to_csv('taxi_fare_submission.csv',index=False)
submission.head()


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2495037895.py in <cell line: 0>()
      1 submission = pd.DataFrame({
      2         "key": test_key,
----> 3         "fare_amount": prediction.round(2)
      4 })
      5 

NameError: name 'prediction' is not defined
