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

5.92108

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from pandas import read_csv, DataFrame, to_datetime
from numpy import radians, sin, cos, arcsin, sqrt, mean
from sklearn import ensemble
import time
from datetime import datetime
import xgboost as xgb


## === cell 1
train_data = read_csv("../input/train.csv")
test = read_csv("../input/test.csv")
sample_sub = read_csv("../input/sample_submission.csv")


## === cell 2
train = train_data[:1000000]


## === cell 3

pickup_longitude_min = test.pickup_longitude.min()
pickup_longitude_max = test.pickup_latitude.max()
pickup_latitude_min = test.pickup_latitude.min()
pickup_latitude_max = test.pickup_latitude.max()
dropoff_longitude_min = test.dropoff_longitude.min()
dropoff_longitude_max = test.dropoff_longitude.max()
dropoff_latitude_min = test.dropoff_latitude.min()
dropoff_latitude_max = test.dropoff_latitude.max()

train = train.loc[(train['fare_amount'] > 0) & (train['fare_amount'] < 300)]
train = train.loc[(train['pickup_longitude'] > pickup_longitude_min) & (train['pickup_longitude'] < pickup_longitude_max)]
train = train.loc[(train['pickup_latitude'] > pickup_latitude_min) & (train['pickup_latitude'] < pickup_latitude_max)]
train = train.loc[(train['dropoff_longitude'] > dropoff_longitude_min) & (train['dropoff_longitude'] < dropoff_longitude_max)]
train = train.loc[(train['dropoff_latitude'] > dropoff_latitude_min) & (train['dropoff_latitude'] < dropoff_latitude_max)]


## === cell 4

def rasst(value1, value2, value3, value4):

    longitude_1, latitude_1, longitude_2, latitude_2 = value1, value2, value3, value4
    longitude_1, latitude_1, longitude_2, latitude_2 = map(radians, [longitude_1, latitude_1, longitude_2, latitude_2])

    dlongitude = longitude_2 - longitude_1
    dlatitude = latitude_2 - latitude_1

    value = sin(dlatitude/2.0)**2 + cos(latitude_1) * cos(latitude_2) * sin(dlongitude/2.0)**2

    c = 2 * arcsin(sqrt(value))
    km = c * 6367
    return km


## === cell 5

train['pickup_datetime'] = to_datetime(train['pickup_datetime'])
train['hour_of_day'] = train.pickup_datetime.dt.hour.astype(float)
train['day'] = train.pickup_datetime.dt.day.astype(float)
train['week'] = train.pickup_datetime.dt.week.astype(float)
train['month'] = train.pickup_datetime.dt.month.astype(float)
train['day_of_year'] = train.pickup_datetime.dt.dayofyear.astype(float)
train['week_of_year'] = train.pickup_datetime.dt.weekofyear.astype(float)
train['passenger_count'] = train['passenger_count'].astype(float)
train['rasst'] = rasst(train['pickup_longitude'], train['pickup_latitude'], train['dropoff_longitude'], train['dropoff_latitude'])

test['pickup_datetime'] = to_datetime(test['pickup_datetime'])
test['hour_of_day'] = test.pickup_datetime.dt.hour.astype(float)
test['day'] = test.pickup_datetime.dt.day.astype(float)
test['week'] = test.pickup_datetime.dt.week.astype(float)
test['month'] = test.pickup_datetime.dt.month.astype(float)
test['day_of_year'] = test.pickup_datetime.dt.dayofyear.astype(float)
test['week_of_year'] = test.pickup_datetime.dt.weekofyear.astype(float)
test['passenger_count'] = test['passenger_count'].astype(float)
test['rasst'] = rasst(test['pickup_longitude'], test['pickup_latitude'], test['dropoff_longitude'], test['dropoff_latitude'])


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1445400232.py in <cell line: 0>()
      4 train['hour_of_day'] = train.pickup_datetime.dt.hour.astype(float)
      5 train['day'] = train.pickup_datetime.dt.day.astype(float)
----> 6 train['week'] = train.pickup_datetime.dt.week.astype(float)
      7 train['month'] = train.pickup_datetime.dt.month.astype(float)
      8 train['day_of_year'] = train.pickup_datetime.dt.dayofyear.astype(float)

AttributeError: 'DatetimeProperties' object has no attribute 'week'

## === cell 6
test.head()


## === cell 7
train.head()


## === cell 8
train_y = train["fare_amount"]
test_key = test['key']
train_x = train.drop(["fare_amount", "key"], axis = 1)
train_x = train_x.drop(['pickup_datetime'], axis = 1)
test = test.drop(['pickup_datetime', 'key'], axis = 1)


## === cell 9
train_xgb = xgb.DMatrix(train_x, train_y)
test_xgb = xgb.DMatrix(test)


## === cell 10

num_round = 5
param = {'max_depth':12, 'eta':0.2,'min_child-weight':2, 'gamma':2, 'booster':'dart', 'three-method':'approx', 'normalize_type':'forest', 'rate_drop':0.3, 'eval_metric':'rmse'}
train = xgb.train(param, train_xgb, num_round)
predict = train.predict(test_xgb, ntree_limit = num_round)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4092120907.py in <cell line: 0>()
      4 param = {'max_depth':12, 'eta':0.2,'min_child-weight':2, 'gamma':2, 'booster':'dart', 'three-method':'approx', 'normalize_type':'forest', 'rate_drop':0.3, 'eval_metric':'rmse'}
      5 train = xgb.train(param, train_xgb, num_round)
----> 6 predict = train.predict(test_xgb, ntree_limit = num_round)

TypeError: Booster.predict() got an unexpected keyword argument 'ntree_limit'

## === cell 11
res = DataFrame(test_key)


## === cell 12
res.insert(1,'fare_amount', predict)
res.to_csv('submission.csv', index=False)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1215406896.py in <cell line: 0>()
----> 1 res.insert(1,'fare_amount', predict)
      2 res.to_csv('submission.csv', index=False)

NameError: name 'predict' is not defined
