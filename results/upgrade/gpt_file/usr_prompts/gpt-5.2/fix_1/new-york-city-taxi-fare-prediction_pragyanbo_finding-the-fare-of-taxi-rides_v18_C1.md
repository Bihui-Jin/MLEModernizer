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

4.05769

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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline


## === cell 1
train_df = pd.read_csv('../input/train.csv', nrows = 1000000)


## === cell 2
train_df.shape


## === cell 3
test_df = pd.read_csv('../input/test.csv')


## === cell 4
test_df.shape


## === cell 5
train_df.head(5)


## === cell 6
train_df.isnull().sum()


## === cell 7
train_df.dropna(inplace=True)


## === cell 8
train_df.describe()


## === cell 9
train_df = train_df[train_df['fare_amount']>0]


## === cell 10
train_df.shape


## === cell 11
def distance(lat1, lon1, lat2, lon2):
    a = 0.5 - np.cos((lat2 - lat1) *  0.017453292519943295)/2 + np.cos(lat1 * 0.017453292519943295) * np.cos(lat2 * 0.017453292519943295) * (1 - np.cos((lon2 - lon1) *  0.017453292519943295)) / 2
    res = 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
    return res


## === cell 12
train_df['distance'] = distance(train_df.pickup_latitude, train_df.pickup_longitude, \
                                      train_df.dropoff_latitude,train_df.dropoff_longitude)


## === cell 13
test_df['distance'] = distance(test_df.pickup_latitude, test_df.pickup_longitude, \
                                      test_df.dropoff_latitude,test_df.dropoff_longitude)


## === cell 14
train_df = train_df[train_df['distance']<15]


## === cell 15
train_df.describe()


## === cell 16
train_df = train_df[(train_df['passenger_count']!=0) & (train_df['passenger_count']<10)]


## === cell 17
train_df['hour'] = train_df.pickup_datetime.apply(lambda x: pd.to_datetime(x).hour)
train_df['year'] = train_df.pickup_datetime.apply(lambda x: pd.to_datetime(x).year)

test_df['hour'] = test_df.pickup_datetime.apply(lambda x: pd.to_datetime(x).hour)
test_df['year'] = test_df.pickup_datetime.apply(lambda x: pd.to_datetime(x).year)


## === cell 18
feat_cols_s = ['distance','passenger_count','hour','year']

X = train_df[feat_cols_s]
y = train_df['fare_amount']


## === cell 19
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


## === cell 24
import xgboost as xgb


## === cell 25
def XGBoost(X_train,X_test,y_train,y_test,num_rounds=300):
    dtrain = xgb.DMatrix(X_train,label=y_train)
    dtest = xgb.DMatrix(X_test,label=y_test)

    return xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'}
                    ,dtrain=dtrain,num_boost_round=num_rounds, 
                    early_stopping_rounds=20,evals=[(dtest,'test')],)


## === cell 26
xgbm = XGBoost(X_train,X_test,y_train,y_test)
xgbm_pred = xgbm.predict(xgb.DMatrix(test_df[feat_cols_s]), ntree_limit = xgbm.best_ntree_limit)


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3245024008.py in <cell line: 0>()
      1 xgbm = XGBoost(X_train,X_test,y_train,y_test)
----> 2 xgbm_pred = xgbm.predict(xgb.DMatrix(test_df[feat_cols_s]), ntree_limit = xgbm.best_ntree_limit)

AttributeError: 'Booster' object has no attribute 'best_ntree_limit'

## === cell 27
submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount':xgbm_pred},
    columns = ['key', 'fare_amount'])
submission.to_csv('XGboost regression.csv', index = False)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/248633694.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {'key': test_df.key, 'fare_amount':xgbm_pred},
      3     columns = ['key', 'fare_amount'])
      4 submission.to_csv('XGboost regression.csv', index = False)

NameError: name 'xgbm_pred' is not defined

## === cell 28
importance=xgbm.get_score()
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'score'])
plt.figure()

df.plot(kind='barh', x='feature', y='score', legend=False, figsize=(10, 25))
plt.title("Feature Importance")


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3372917548.py in <cell line: 0>()
      1 importance=xgbm.get_score()
----> 2 importance = sorted(importance.items(), key=operator.itemgetter(1))
      3 df = pd.DataFrame(importance, columns=['feature', 'score'])
      4 plt.figure()
      5 

NameError: name 'operator' is not defined
