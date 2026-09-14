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

3.10

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

4.37007

# 6. Current score

6.63328

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd 

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import seaborn as sns
import matplotlib.pyplot as plt


## === cell 1
train = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows = 10000, parse_dates=["pickup_datetime"])
print(f'{train.shape} shape')
train.head()


## === cell 2
print(train.info())
train.describe()


## === cell 3
test = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv", parse_dates=["pickup_datetime"])
print(f'{test.shape} shape')
test.head() 


## === cell 4
test.info()


## === cell 5
test.describe()


## === cell 6
train.describe()


## === cell 7
train = train.drop(columns=['key'])
test = test.drop(columns=['key'])


## === cell 8
train.head()


## === cell 9
train[train['passenger_count']>6]


## === cell 10
train = train.drop(train[train['passenger_count']==208].index)


## === cell 11
train = train.drop(train[train['fare_amount']<0].index)


## === cell 12

train['year'] = train['pickup_datetime'].dt.year
train['month'] = train['pickup_datetime'].dt.month
train['day'] = train['pickup_datetime'].dt.day
train['hour'] = train['pickup_datetime'].dt.hour
train['minute'] = train['pickup_datetime'].dt.minute

test['year'] = test['pickup_datetime'].dt.year
test['month'] = test['pickup_datetime'].dt.month
test['day'] = test['pickup_datetime'].dt.day
test['hour'] = test['pickup_datetime'].dt.hour
test['minute'] = test['pickup_datetime'].dt.minute


## === cell 13

train = train.drop(columns=['pickup_datetime'])
test = test.drop(columns=['pickup_datetime'])


## === cell 14
train.describe()


## === cell 15
train.shape


## === cell 16
train.describe()


## === cell 17
train = train[(train['pickup_latitude']>40)&(train['pickup_latitude']<45)]
train = train[(train['dropoff_latitude']>40)&(train['dropoff_latitude']<45)]

train = train[(train['pickup_longitude']<-71)&(train['pickup_longitude']>-79)]
train = train[(train['dropoff_longitude']<-71)&(train['dropoff_longitude']>-79)]


## === cell 18
train.describe()


## === cell 19
train.shape


## === cell 20
train.head()


## === cell 21
test.head()


## === cell 22
train.corr()


## === cell 23
variables = ['fare_amount', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'year','month', 'day', 'hour','minute']

for var in variables:
    plt.figure() # Creating a rectangle (figure) for each plot
    sns.regplot(x=var, y='fare_amount', data=train).set();


## === cell 24
x = train.loc[:, train.columns != 'fare_amount']
x_test1 = test
y = train['fare_amount'].values


## === cell 25
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


## === cell 26
linreg = LinearRegression()


linreg.fit(x_train,y_train)

y_pred = linreg.predict(x_test)


## === cell 27
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(rmse)


## === cell 28
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)


## === cell 29
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(x_train, y_train)
lasso_pred = lasso.predict(x_test)


## === cell 30
mse = mean_squared_error(y_test, lasso_pred)
rmse = np.sqrt(mse)
print(rmse)


## === cell 31
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(x_train, y_train)
ridge_pred = ridge.predict(x_test)


## === cell 32
mse = mean_squared_error(y_test, ridge_pred)
rmse = np.sqrt(mse)
print(rmse)


## === cell 33
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)


## === cell 34
mse = mean_squared_error(y_test, rf_pred)
rmse = np.sqrt(mse)
print(rmse)


## === cell 35
results = pd.DataFrame({'Actual': y_test, 'Predicted': rf_pred})
print(results)


## === cell 36
pred = rf.predict(test)


## === cell 37
submission = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv')
submission['fare_amount'] = pred
submission.to_csv('submission.csv', index=False)
submission.head()
