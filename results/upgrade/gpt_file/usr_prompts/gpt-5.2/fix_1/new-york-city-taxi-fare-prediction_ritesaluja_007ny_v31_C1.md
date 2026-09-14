# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

5.75918

# 6. Current score

6.78541

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))

import seaborn as sns
import matplotlib.pyplot as plt

import gc

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import lightgbm as lgb

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler


## === cell 1
test = pd.read_csv("../input/test.csv")


## === cell 2
gc.collect()
R = 6371e3
a1 = np.radians(test['pickup_latitude'])
a2 = np.radians(test['dropoff_latitude'])
da = np.radians(test['dropoff_latitude']-test['pickup_latitude'])
dl = np.radians(test['dropoff_longitude']-test['pickup_longitude'])

test = test.drop(columns = ['pickup_latitude','dropoff_latitude','pickup_longitude','dropoff_longitude'])

a = np.sin(da/2) * np.sin(da/2) + np.cos(a1) * np.cos(a2) * np.sin(dl/2) * np.sin(dl/2)
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
d = R * c;

del R,c,a,a1,a2,da,dl

test['Distance']= pd.Series(d)
del d
gc.collect()


## === cell 3
train_path  = '../input/train.csv'

traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

cols = list(traintypes.keys())

dftemp = pd.read_csv(train_path, usecols=cols, dtype=traintypes, nrows = 500000)
gc.collect()


## === cell 4
dftemp.to_feather('nyc_taxi_data_raw.feather')
del dftemp,traintypes,cols
gc.collect()


## === cell 5
df = pd.read_feather('nyc_taxi_data_raw.feather')



## === cell 6
gc.collect()
R = 6371e3
a1 = np.radians(df['pickup_latitude'])
a2 = np.radians(df['dropoff_latitude'])
da = np.radians(df['dropoff_latitude']-df['pickup_latitude'])
dl = np.radians(df['dropoff_longitude']-df['pickup_longitude'])

df = df.drop(columns = ['pickup_latitude','dropoff_latitude','pickup_longitude','dropoff_longitude'])

a = np.sin(da/2) * np.sin(da/2) + np.cos(a1) * np.cos(a2) * np.sin(dl/2) * np.sin(dl/2)
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
d = R * c;

del R,c,a,a1,a2,da,dl

df['Distance']= pd.Series(d)
del d
gc.collect()


## === cell 7
sns.lmplot(x='Distance',y='fare_amount', data = df)


## === cell 8
df = df.dropna()
sns.distplot(df['passenger_count'])
gc.collect()


## === cell 9
df = df[~(df[['passenger_count']] == 0).any(axis=1)]
scaler = MinMaxScaler()
gc.collect()


## === cell 10
sns.set(style="whitegrid")
sns.boxplot(x="passenger_count", y="Distance", data=df)
plt.grid(True)
plt.show()
gc.collect()


## === cell 11

y = df['fare_amount']
x = df.drop(columns=['pickup_datetime','fare_amount'])
gc.collect()


## === cell 12
scaler.fit_transform(x) 
gc.collect()


## === cell 13

ada = RandomForestRegressor()
ada.fit(x,y)
y_pred = ada.predict(x)
print('The RMSE of prediction on training is:',mean_squared_error(y, y_pred) ** 0.5)
model = ada
"""
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.20)
del x,y
gc.collect()
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
rfc = RandomForestRegressor() 
# Use a grid over parameters of interest
param_grid = { 
           "n_estimators" : [ 36, 45, 54, 63,100],
           "max_depth" : [ 5, 10, 15, 20,40],
            "min_samples_leaf" : [1,5,10,20,40]
           }
model = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 10)
model.fit(X_train, y_train)
print(model.best_params_)
y_pred = model.predict(X_val)
print('The rmse of prediction is:', mean_squared_error(y_val, y_pred) ** 0.5)
gc.collect()


X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.20)
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)


# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 500,
    'feature_fraction': 0.9,
    'learning_rate': 0.005,
    'verbose': 0,
    'bagging_fraction': 0.8,
    'bagging_freq': 10,
    'early_stopping_round':100
}
#'bagging_fraction': 0.8,
#    'bagging_freq': 10,


print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,
                
                )
"""
gc.collect()


## === cell 14
x_test = test.drop(columns=['key','pickup_datetime'])
gc.collect()


## === cell 15
scaler.transform(x_test)
gc.collect()


## === cell 16
y_test = model.predict(x_test)
y_test =  y_test
df = pd.DataFrame()
df["key"] = test["key"]
df["fare_amount"] = y_test
df.to_csv("sample_submission.csv", index = False)
