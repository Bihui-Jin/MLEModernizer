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
geopy==2.4.1
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

4.04952

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
plt.style.use('fivethirtyeight')

import geopy.distance


import os
print(os.listdir("../input"))
import gc


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn import svm
from sklearn.linear_model import SGDRegressor
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


## === cell 1
def load_Data():
    train = pd.read_csv("../input/train.csv", nrows=10_00_000, low_memory=True)
    test = pd.read_csv("../input/test.csv", nrows=10_00_000, low_memory=True)
    return train,test







    


## === cell 2
train, test = load_Data()


## === cell 3
train.head(5)


## === cell 4
train.describe()


## === cell 5
train.info()


## === cell 6
train.isnull().sum()


## === cell 7
train = train.fillna(0)


## === cell 8
train['key2'] = pd.to_datetime(train['key'], errors='coerce')
train['key2'].head()
train.info()


## === cell 9
test['key2'] = pd.to_datetime(test['key'], errors='coerce')


## === cell 10
train['fare_amount'].plot(kind='box')


## === cell 11
gc.collect()
train.describe()


## === cell 12
print("% of fares above 25$ - {:0.2f}".format(train[train['fare_amount'] > 25]['key'].count()*100/train['key'].count()))
print("% of fares above 50$ - {:0.2f}".format(train[train['fare_amount'] > 50]['key'].count()*100/train['key'].count()))
print("% of fares above 100$ - {:0.2f}".format(train[train['fare_amount'] > 100]['key'].count()*100/train['key'].count()))
print("% of fares below 0$ - {:0.2f}".format(train[train['fare_amount'] < 0]['key'].count()*100/train['key'].count()))


## === cell 13

fig, axarr = plt.subplots(2, 2, figsize=(20, 10))

train[~(train['fare_amount'] > 25)]['fare_amount'].plot(kind="box",ax=axarr[0][0])
train[~(train['fare_amount'] > 50)]['fare_amount'].plot(kind="box",ax=axarr[0][1])
train[~(train['fare_amount'] > 100)]['fare_amount'].plot(kind="box",ax=axarr[1][0])
train[~(train['fare_amount'] < 0 )]['fare_amount'].plot(kind='box',ax = axarr[1][1])


## === cell 14
train['passenger_count'].plot(kind='box')


## === cell 15
print("Count of invalid pickup latitude", train[(train['pickup_latitude'] > 90) | (train['pickup_latitude'] < -90) ]['pickup_latitude'].count())
print("Count of invalid dropoff latitude", train[(train['dropoff_latitude'] > 90) | (train['dropoff_latitude'] < -90) ]['dropoff_latitude'].count())
print("Count of invalid pickup longitude", train[(train['pickup_longitude'] > 180) | (train['pickup_longitude'] < -180) ]['pickup_longitude'].count())
print("Count of invalid dropoff longitude", train[(train['dropoff_longitude'] > 180) | (train['dropoff_longitude'] < -180) ]['dropoff_longitude'].count())


## === cell 16
print("Count of invalid pickup latitude", test[(test['pickup_latitude'] > 90) | (test['pickup_latitude'] < -90) ]['pickup_latitude'].count())
print("Count of invalid dropoff latitude", test[(test['dropoff_latitude'] > 90) | (test['dropoff_latitude'] < -90) ]['dropoff_latitude'].count())
print("Count of invalid pickup longitude", test[(test['pickup_longitude'] > 180) | (test['pickup_longitude'] < -180) ]['pickup_longitude'].count())
print("Count of invalid dropoff longitude", test[(test['dropoff_longitude'] > 180) | (test['dropoff_longitude'] < -180) ]['dropoff_longitude'].count())


## === cell 17
train = train[~((train['pickup_latitude'] > 90) | (train['pickup_latitude'] < -90))]
train = train[~((train['dropoff_latitude'] > 90) | (train['dropoff_latitude'] < -90))]
train = train[~((train['pickup_longitude'] > 180) | (train['pickup_longitude'] < -180))]
train = train[~((train['dropoff_longitude'] > 180) | (train['dropoff_longitude'] < -180))]


## === cell 18
train['distance']=train[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
                                                                                                geopy.distance.VincentyDistance((x['pickup_latitude'],x['pickup_longitude']),
                                                                                                                               (x['dropoff_latitude'],x['dropoff_longitude'])).km,axis=1)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/59650148.py in <cell line: 0>()
----> 1 train['distance']=train[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
      2                                                                                                 geopy.distance.VincentyDistance((x['pickup_latitude'],x['pickup_longitude']),
      3                                                                                                                                (x['dropoff_latitude'],x['dropoff_longitude'])).km,axis=1)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/tmp/ipykernel_11/59650148.py in <lambda>(x)
      1 train['distance']=train[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
----> 2                                                                                                 geopy.distance.VincentyDistance((x['pickup_latitude'],x['pickup_longitude']),
      3                                                                                                                                (x['dropoff_latitude'],x['dropoff_longitude'])).km,axis=1)

AttributeError: module 'geopy.distance' has no attribute 'VincentyDistance'

## === cell 19
train.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'],axis=1, inplace=True)


## === cell 20
test['distance']=test[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
                                                                                                geopy.distance.VincentyDistance((x['pickup_latitude'],x['pickup_longitude']),
                                                                                                                               (x['dropoff_latitude'],x['dropoff_longitude'])).km,axis=1)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3547695418.py in <cell line: 0>()
----> 1 test['distance']=test[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
      2                                                                                                 geopy.distance.VincentyDistance((x['pickup_latitude'],x['pickup_longitude']),
      3                                                                                                                                (x['dropoff_latitude'],x['dropoff_longitude'])).km,axis=1)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/tmp/ipykernel_11/3547695418.py in <lambda>(x)
      1 test['distance']=test[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
----> 2                                                                                                 geopy.distance.VincentyDistance((x['pickup_latitude'],x['pickup_longitude']),
      3                                                                                                                                (x['dropoff_latitude'],x['dropoff_longitude'])).km,axis=1)

AttributeError: module 'geopy.distance' has no attribute 'VincentyDistance'

## === cell 21
test.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'],axis=1, inplace=True)


## === cell 22
print("% of trips above 25 KM - {:0.2f}".format(train[train['distance'] > 25]['key'].count()*100/train.count()['key']))


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'distance'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3462529605.py in <cell line: 0>()
----> 1 print("% of trips above 25 KM - {:0.2f}".format(train[train['distance'] > 25]['key'].count()*100/train.count()['key']))

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'distance'

## === cell 23
train['year'] = train['key2'].apply(lambda x : x.year)
train['month'] = train['key2'].apply(lambda x : x.month)
train['day'] = train['key2'].apply(lambda x : x.day)
train['day of week'] = train['key2'].apply(lambda x : x.weekday())
train['hour'] = train['key2'].apply(lambda x : x.hour)
train["week"] = train['key2'].apply(lambda x: x.week)
train['day_of_year'] = train['key2'].apply(lambda x:x.dayofyear)
train['week_of_year'] = train['key2'].apply(lambda x:x.weekofyear)
train['quarter'] = train['key2'].apply(lambda x:x.quarter)


## === cell 24
train.columns


## === cell 25
test['year'] = test['key2'].apply(lambda x : x.year)
test['month'] = test['key2'].apply(lambda x : x.month)
test['day'] = test['key2'].apply(lambda x : x.day)
test['day of week'] = test['key2'].apply(lambda x : x.weekday())
test['hour'] = test['key2'].apply(lambda x : x.hour)
test["week"] = test['key2'].apply(lambda x: x.week)
test['day_of_year'] = test['key2'].apply(lambda x:x.dayofyear)
test['week_of_year'] = test['key2'].apply(lambda x:x.weekofyear)
test['quarter'] = test['key2'].apply(lambda x:x.quarter)


## === cell 26
column_list = ['passenger_count', 'distance', 'year', 'month', 'day', 'day of week', 'hour','week','day_of_year', 'week_of_year', 'quarter']
y_train_ = train['fare_amount']
X_train_ = train.drop(['fare_amount'], axis=1)


## === cell 27
X_train_ = train[column_list] 
X_test = test[column_list]


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2999224643.py in <cell line: 0>()
----> 1 X_train_ = train[column_list]
      2 X_test = test[column_list]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['distance'] not in index"

## === cell 28
X_train_.shape,y_train_.shape,X_test.shape


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2368879003.py in <cell line: 0>()
----> 1 X_train_.shape,y_train_.shape,X_test.shape

NameError: name 'X_test' is not defined

## === cell 29
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train_, y_train_, test_size=0.1)


## === cell 30
X_train.shape,X_val.shape,y_train.shape,y_val.shape


## === cell 31
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error


## === cell 32
lgb  = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,
       learning_rate=0.1, max_depth=8, min_child_samples=55,
       min_child_weight=0.001, min_split_gain=0.1, n_estimators=500,
       n_jobs=-1, num_leaves=45, objective=None, random_state=None,
       reg_alpha=5.0, reg_lambda=3.0, silent=True, subsample=1.0,
       subsample_for_bin=200000, subsample_freq=1)




## === cell 33
%%time
lgb.fit(X_train,y_train)


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<timed eval> in <module>

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in fit(self, X, y, sample_weight, init_score, eval_set, eval_names, eval_sample_weight, eval_init_score, eval_metric, feature_name, categorical_feature, callbacks, init_model)
   1396     ) -> "LGBMRegressor":
   1397         """Docstring is inherited from the LGBMModel."""
-> 1398         super().fit(
   1399             X,
   1400             y,

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in fit(self, X, y, sample_weight, init_score, group, eval_set, eval_names, eval_sample_weight, eval_class_weight, eval_init_score, eval_group, eval_metric, feature_name, categorical_feature, callbacks, init_model)
   1047         callbacks.append(record_evaluation(evals_result))
   1048 
-> 1049         self._Booster = train(
   1050             params=params,
   1051             train_set=train_set,

/usr/local/lib/python3.11/dist-packages/lightgbm/engine.py in train(params, train_set, num_boost_round, valid_sets, valid_names, feval, init_model, keep_training_booster, callbacks)
    295     # construct booster
    296     try:
--> 297         booster = Booster(params=params, train_set=train_set)
    298         if is_valid_contain_train:
    299             booster.set_train_data_name(train_data_name)

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in __init__(self, params, train_set, model_file, model_str)
   3654                 )
   3655             # construct booster object
-> 3656             train_set.construct()
   3657             # copy the parameters from train_set
   3658             params.update(train_set.get_params())

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in construct(self)
   2588             else:
   2589                 # create train
-> 2590                 self._lazy_init(
   2591                     data=self.data,
   2592                     label=self.label,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _lazy_init(self, data, label, reference, weight, group, init_score, predictor, feature_name, categorical_feature, params, position)
   2121             categorical_feature = reference.categorical_feature
   2122         if isinstance(data, pd_DataFrame):
-> 2123             data, feature_name, categorical_feature, self.pandas_categorical = _data_from_pandas(
   2124                 data=data,
   2125                 feature_name=feature_name,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _data_from_pandas(data, feature_name, categorical_feature, pandas_categorical)
    866 
    867     return (
--> 868         _pandas_to_numpy(data, target_dtype=target_dtype),
    869         feature_name,
    870         categorical_feature,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _pandas_to_numpy(data, target_dtype)
    812     target_dtype: "np.typing.DTypeLike",
    813 ) -> np.ndarray:
--> 814     _check_for_bad_pandas_dtypes(data.dtypes)
    815     try:
    816         # most common case (no nullable dtypes)

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _check_for_bad_pandas_dtypes(pandas_dtypes_series)
    803     ]
    804     if bad_pandas_dtypes:
--> 805         raise ValueError(
    806             f"pandas dtypes must be int, float or bool.\nFields with bad pandas dtypes: {', '.join(bad_pandas_dtypes)}"
    807         )

ValueError: pandas dtypes must be int, float or bool.
Fields with bad pandas dtypes: key: object, pickup_datetime: object, key2: datetime64[ns]

## === cell 34
%%time
pred = lgb.predict(X_val)


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
<timed exec> in <module>

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in predict(self, X, raw_score, start_iteration, num_iteration, pred_leaf, pred_contrib, validate_features, **kwargs)
   1104         """Docstring is set after definition, using a template."""
   1105         if not self.__sklearn_is_fitted__():
-> 1106             raise LGBMNotFittedError("Estimator not fitted, call fit before exploiting the model.")
   1107         if not isinstance(X, (pd_DataFrame, dt_DataTable)):
   1108             X = _LGBMValidateData(

NotFittedError: Estimator not fitted, call fit before exploiting the model.

## === cell 35
%%time
from sklearn.metrics import r2_score
print(r2_score(y_val, pred))
print(np.sqrt(mean_squared_error(y_val, pred)))


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'pred' is not defined

## === cell 36
lgb.fit(X_train_,y_train_)


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3584850368.py in <cell line: 0>()
----> 1 lgb.fit(X_train_,y_train_)

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in fit(self, X, y, sample_weight, init_score, eval_set, eval_names, eval_sample_weight, eval_init_score, eval_metric, feature_name, categorical_feature, callbacks, init_model)
   1396     ) -> "LGBMRegressor":
   1397         """Docstring is inherited from the LGBMModel."""
-> 1398         super().fit(
   1399             X,
   1400             y,

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in fit(self, X, y, sample_weight, init_score, group, eval_set, eval_names, eval_sample_weight, eval_class_weight, eval_init_score, eval_group, eval_metric, feature_name, categorical_feature, callbacks, init_model)
   1047         callbacks.append(record_evaluation(evals_result))
   1048 
-> 1049         self._Booster = train(
   1050             params=params,
   1051             train_set=train_set,

/usr/local/lib/python3.11/dist-packages/lightgbm/engine.py in train(params, train_set, num_boost_round, valid_sets, valid_names, feval, init_model, keep_training_booster, callbacks)
    295     # construct booster
    296     try:
--> 297         booster = Booster(params=params, train_set=train_set)
    298         if is_valid_contain_train:
    299             booster.set_train_data_name(train_data_name)

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in __init__(self, params, train_set, model_file, model_str)
   3654                 )
   3655             # construct booster object
-> 3656             train_set.construct()
   3657             # copy the parameters from train_set
   3658             params.update(train_set.get_params())

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in construct(self)
   2588             else:
   2589                 # create train
-> 2590                 self._lazy_init(
   2591                     data=self.data,
   2592                     label=self.label,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _lazy_init(self, data, label, reference, weight, group, init_score, predictor, feature_name, categorical_feature, params, position)
   2121             categorical_feature = reference.categorical_feature
   2122         if isinstance(data, pd_DataFrame):
-> 2123             data, feature_name, categorical_feature, self.pandas_categorical = _data_from_pandas(
   2124                 data=data,
   2125                 feature_name=feature_name,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _data_from_pandas(data, feature_name, categorical_feature, pandas_categorical)
    866 
    867     return (
--> 868         _pandas_to_numpy(data, target_dtype=target_dtype),
    869         feature_name,
    870         categorical_feature,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _pandas_to_numpy(data, target_dtype)
    812     target_dtype: "np.typing.DTypeLike",
    813 ) -> np.ndarray:
--> 814     _check_for_bad_pandas_dtypes(data.dtypes)
    815     try:
    816         # most common case (no nullable dtypes)

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _check_for_bad_pandas_dtypes(pandas_dtypes_series)
    803     ]
    804     if bad_pandas_dtypes:
--> 805         raise ValueError(
    806             f"pandas dtypes must be int, float or bool.\nFields with bad pandas dtypes: {', '.join(bad_pandas_dtypes)}"
    807         )

ValueError: pandas dtypes must be int, float or bool.
Fields with bad pandas dtypes: key: object, pickup_datetime: object, key2: datetime64[ns]

## === cell 37
def display_importances(feature_importance_df_, doWorst=False, n_feat=50):
    if not doWorst:
        cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[:n_feat].index        
    else:
        cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[-n_feat:].index
    
    mean_imp = feature_importance_df_[["feature", "importance"]].groupby("feature").mean()
    df_2_neglect = mean_imp[mean_imp['importance'] < 1e-3]
    print('The list of features with 0 importance: ')
    print(df_2_neglect.index.values.tolist())
    del mean_imp, df_2_neglect
    
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    
    plt.figure(figsize=(8,10))
    sns.barplot(x="importance", y="feature", 
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')


importance_df = pd.DataFrame()
importance_df["feature"] = column_list
importance_df["importance"] = lgb.feature_importances_
display_importances(feature_importance_df_=importance_df, n_feat=20)


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/3253657048.py in <cell line: 0>()
     26 importance_df = pd.DataFrame()
     27 importance_df["feature"] = column_list
---> 28 importance_df["importance"] = lgb.feature_importances_
     29 display_importances(feature_importance_df_=importance_df, n_feat=20)

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in feature_importances_(self)
   1264         """
   1265         if not self.__sklearn_is_fitted__():
-> 1266             raise LGBMNotFittedError("No feature_importances found. Need to call fit beforehand.")
   1267         return self._Booster.feature_importance(importance_type=self.importance_type)  # type: ignore[union-attr]
   1268 

NotFittedError: No feature_importances found. Need to call fit beforehand.

## === cell 38
lgb.fit(X_train_,y_train_)


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3584850368.py in <cell line: 0>()
----> 1 lgb.fit(X_train_,y_train_)

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in fit(self, X, y, sample_weight, init_score, eval_set, eval_names, eval_sample_weight, eval_init_score, eval_metric, feature_name, categorical_feature, callbacks, init_model)
   1396     ) -> "LGBMRegressor":
   1397         """Docstring is inherited from the LGBMModel."""
-> 1398         super().fit(
   1399             X,
   1400             y,

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in fit(self, X, y, sample_weight, init_score, group, eval_set, eval_names, eval_sample_weight, eval_class_weight, eval_init_score, eval_group, eval_metric, feature_name, categorical_feature, callbacks, init_model)
   1047         callbacks.append(record_evaluation(evals_result))
   1048 
-> 1049         self._Booster = train(
   1050             params=params,
   1051             train_set=train_set,

/usr/local/lib/python3.11/dist-packages/lightgbm/engine.py in train(params, train_set, num_boost_round, valid_sets, valid_names, feval, init_model, keep_training_booster, callbacks)
    295     # construct booster
    296     try:
--> 297         booster = Booster(params=params, train_set=train_set)
    298         if is_valid_contain_train:
    299             booster.set_train_data_name(train_data_name)

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in __init__(self, params, train_set, model_file, model_str)
   3654                 )
   3655             # construct booster object
-> 3656             train_set.construct()
   3657             # copy the parameters from train_set
   3658             params.update(train_set.get_params())

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in construct(self)
   2588             else:
   2589                 # create train
-> 2590                 self._lazy_init(
   2591                     data=self.data,
   2592                     label=self.label,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _lazy_init(self, data, label, reference, weight, group, init_score, predictor, feature_name, categorical_feature, params, position)
   2121             categorical_feature = reference.categorical_feature
   2122         if isinstance(data, pd_DataFrame):
-> 2123             data, feature_name, categorical_feature, self.pandas_categorical = _data_from_pandas(
   2124                 data=data,
   2125                 feature_name=feature_name,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _data_from_pandas(data, feature_name, categorical_feature, pandas_categorical)
    866 
    867     return (
--> 868         _pandas_to_numpy(data, target_dtype=target_dtype),
    869         feature_name,
    870         categorical_feature,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _pandas_to_numpy(data, target_dtype)
    812     target_dtype: "np.typing.DTypeLike",
    813 ) -> np.ndarray:
--> 814     _check_for_bad_pandas_dtypes(data.dtypes)
    815     try:
    816         # most common case (no nullable dtypes)

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _check_for_bad_pandas_dtypes(pandas_dtypes_series)
    803     ]
    804     if bad_pandas_dtypes:
--> 805         raise ValueError(
    806             f"pandas dtypes must be int, float or bool.\nFields with bad pandas dtypes: {', '.join(bad_pandas_dtypes)}"
    807         )

ValueError: pandas dtypes must be int, float or bool.
Fields with bad pandas dtypes: key: object, pickup_datetime: object, key2: datetime64[ns]

## === cell 39
y_pred = lgb.predict(X_test)


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1348825562.py in <cell line: 0>()
----> 1 y_pred = lgb.predict(X_test)

NameError: name 'X_test' is not defined

## === cell 40
submission = pd.DataFrame({"key": test['key'], "fare_amount": y_pred})
submission.to_csv('submission.csv',index=False)


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2893137182.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"key": test['key'], "fare_amount": y_pred})
      2 submission.to_csv('submission.csv',index=False)

NameError: name 'y_pred' is not defined
