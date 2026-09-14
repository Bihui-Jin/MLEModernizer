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

7.78808

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


import os
print(os.listdir("../input"))

import geopy.distance



## === cell 1
df_train= pd.read_csv("../input/train.csv", nrows=500000,low_memory=True)


## === cell 2
df_test = pd.read_csv("../input/test.csv")


## === cell 3
df_train.info()


## === cell 4
df_train.head(5)


## === cell 5
df_train['key2'] = pd.to_datetime(df_train['key'], errors='coerce')


## === cell 6
df_train.info()


## === cell 7
df_test['key2'] = pd.to_datetime(df_test['key'], errors='coerce')


## === cell 8
df_train.drop('pickup_datetime', axis=1, inplace=True)


## === cell 9
df_test.drop('pickup_datetime', axis=1, inplace=True)


## === cell 10
df_train.isnull().sum()


## === cell 11
df_train.dropna(axis=0, inplace=True)


## === cell 12
df_train['fare_amount'].plot(kind='box')


## === cell 13
df_train.describe()


## === cell 14
print("% of fares above 25$ - {:0.2f}".format(df_train[df_train['fare_amount'] > 25]['key'].count()*100/df_train['key'].count()))


## === cell 15
df_train = df_train[~(df_train['fare_amount'] > 25)]


## === cell 16
df_train['fare_amount'].plot(kind='box')


## === cell 17
df_train = df_train[~(df_train['fare_amount'] < 0)]


## === cell 18
df_train['fare_amount'].plot(kind='box')


## === cell 19
df_train.info()


## === cell 20
df_train['passenger_count'].plot(kind='box')


## === cell 21
print("% of passengers above 6 - {:0.2f}".format(df_train[df_train['passenger_count'] > 6]['key'].count()*100/df_train['key'].count()))


## === cell 22
df_train = df_train[~(df_train['passenger_count'] > 6)]


## === cell 23
df_train['passenger_count'].plot(kind='box')


## === cell 24
print("Count of invalid pickup latitude", df_train[(df_train['pickup_latitude'] > 90) | (df_train['pickup_latitude'] < -90) ]['pickup_latitude'].count())
print("Count of invalid dropoff latitude", df_train[(df_train['dropoff_latitude'] > 90) | (df_train['dropoff_latitude'] < -90) ]['dropoff_latitude'].count())
print("Count of invalid pickup longitude", df_train[(df_train['pickup_longitude'] > 180) | (df_train['pickup_longitude'] < -180) ]['pickup_longitude'].count())
print("Count of invalid dropoff longitude", df_train[(df_train['dropoff_longitude'] > 180) | (df_train['dropoff_longitude'] < -180) ]['dropoff_longitude'].count())


## === cell 25
print("Count of invalid pickup latitude", df_test[(df_test['pickup_latitude'] > 90) | (df_test['pickup_latitude'] < -90) ]['pickup_latitude'].count())
print("Count of invalid dropoff latitude", df_test[(df_test['dropoff_latitude'] > 90) | (df_test['dropoff_latitude'] < -90) ]['dropoff_latitude'].count())
print("Count of invalid pickup longitude", df_test[(df_test['pickup_longitude'] > 180) | (df_test['pickup_longitude'] < -180) ]['pickup_longitude'].count())
print("Count of invalid dropoff longitude", df_test[(df_test['dropoff_longitude'] > 180) | (df_test['dropoff_longitude'] < -180) ]['dropoff_longitude'].count())


## === cell 26
df_train = df_train[~((df_train['pickup_latitude'] > 90) | (df_train['pickup_latitude'] < -90))]
df_train = df_train[~((df_train['dropoff_latitude'] > 90) | (df_train['dropoff_latitude'] < -90))]
df_train = df_train[~((df_train['pickup_longitude'] > 180) | (df_train['pickup_longitude'] < -180))]
df_train = df_train[~((df_train['dropoff_longitude'] > 180) | (df_train['dropoff_longitude'] < -180))]


## === cell 27
df_train['distance']=df_train[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
                                                                                                geopy.distance.VincentyDistance((x['pickup_latitude'],x['pickup_longitude']),
                                                                                                                               (x['dropoff_latitude'],x['dropoff_longitude'])).km,axis=1)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3075914770.py in <cell line: 0>()
----> 1 df_train['distance']=df_train[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
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

/tmp/ipykernel_11/3075914770.py in <lambda>(x)
      1 df_train['distance']=df_train[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
----> 2                                                                                                 geopy.distance.VincentyDistance((x['pickup_latitude'],x['pickup_longitude']),
      3                                                                                                                                (x['dropoff_latitude'],x['dropoff_longitude'])).km,axis=1)

AttributeError: module 'geopy.distance' has no attribute 'VincentyDistance'

## === cell 28
df_train.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'],axis=1, inplace=True)


## === cell 29
df_test['distance']=df_test[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
                                                                                                geopy.distance.VincentyDistance((x['pickup_latitude'],x['pickup_longitude']),
                                                                                                                               (x['dropoff_latitude'],x['dropoff_longitude'])).km,axis=1)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1983770964.py in <cell line: 0>()
----> 1 df_test['distance']=df_test[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
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

/tmp/ipykernel_11/1983770964.py in <lambda>(x)
      1 df_test['distance']=df_test[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
----> 2                                                                                                 geopy.distance.VincentyDistance((x['pickup_latitude'],x['pickup_longitude']),
      3                                                                                                                                (x['dropoff_latitude'],x['dropoff_longitude'])).km,axis=1)

AttributeError: module 'geopy.distance' has no attribute 'VincentyDistance'

## === cell 30
df_test.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'],axis=1, inplace=True)


## === cell 31
print("% of trips above 25 KM - {:0.2f}".format(df_train[df_train['distance'] > 25]['key'].count()*100/df_train.count()['key']))


## --- ERROR in cell 31, traceback:
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
/tmp/ipykernel_11/3535569852.py in <cell line: 0>()
----> 1 print("% of trips above 25 KM - {:0.2f}".format(df_train[df_train['distance'] > 25]['key'].count()*100/df_train.count()['key']))

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

## === cell 32
df_train = df_train[~(df_train['distance'] > 25)]


## --- ERROR in cell 32, traceback:
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
/tmp/ipykernel_11/3691610880.py in <cell line: 0>()
----> 1 df_train = df_train[~(df_train['distance'] > 25)]

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

## === cell 33
df_train['distance'].plot(kind='box')


## --- ERROR in cell 33, traceback:
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
/tmp/ipykernel_11/3578842799.py in <cell line: 0>()
----> 1 df_train['distance'].plot(kind='box')

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

## === cell 34
df_train['year'] = df_train['key2'].apply(lambda x : x.year)
df_train['month'] = df_train['key2'].apply(lambda x : x.month)
df_train['day'] = df_train['key2'].apply(lambda x : x.day)
df_train['day of week'] = df_train['key2'].apply(lambda x : x.weekday())
df_train['hour'] = df_train['key2'].apply(lambda x : x.hour)


## === cell 35
df_train.head(10)


## === cell 36
df_test['year'] = df_test['key2'].apply(lambda x : x.year)
df_test['month'] = df_test['key2'].apply(lambda x : x.month)
df_test['day'] = df_test['key2'].apply(lambda x : x.day)
df_test['day of week'] = df_test['key2'].apply(lambda x : x.weekday())
df_test['hour'] = df_test['key2'].apply(lambda x : x.hour)


## === cell 37
Y = df_train['fare_amount']
df_train.drop('fare_amount',axis=1, inplace=True)


## === cell 38
df_train.head(5)


## === cell 39
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train[['passenger_count', 'distance', 'year', 'month', 'day', 'day of week', 'hour']], Y, test_size=0.1)


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/419948505.py in <cell line: 0>()
      1 from sklearn.model_selection import train_test_split
----> 2 X_train, X_test, y_train, y_test = train_test_split(df_train[['passenger_count', 'distance', 'year', 'month', 'day', 'day of week', 'hour']], Y, test_size=0.1)

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

## === cell 40
import xgboost as xg


## === cell 41
xgbr = xg.XGBRegressor(learning_rate=.35,max_depth=4,n_estimators=150, booster='dart')


## === cell 42
xgbr.fit(X_train, y_train)


## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1534063561.py in <cell line: 0>()
----> 1 xgbr.fit(X_train, y_train)

NameError: name 'X_train' is not defined

## === cell 43
pred = xgbr.predict(X_test)


## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3142217647.py in <cell line: 0>()
----> 1 pred = xgbr.predict(X_test)

NameError: name 'X_test' is not defined

## === cell 44
from sklearn.metrics import r2_score
r2_score(y_test, pred)


## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3784418990.py in <cell line: 0>()
      1 from sklearn.metrics import r2_score
----> 2 r2_score(y_test, pred)

NameError: name 'y_test' is not defined

## === cell 45
xgbr.feature_importances_


## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/2228383682.py in <cell line: 0>()
----> 1 xgbr.feature_importances_

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in feature_importances_(self)
   1312 
   1313         """
-> 1314         b: Booster = self.get_booster()
   1315 
   1316         def dft() -> str:

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in get_booster(self)
    723             from sklearn.exceptions import NotFittedError
    724 
--> 725             raise NotFittedError("need to call fit or load_model beforehand")
    726         return self._Booster
    727 

NotFittedError: need to call fit or load_model beforehand

## === cell 46
pred = xgbr.predict(df_test[['passenger_count', 'distance', 'year', 'month', 'day', 'day of week', 'hour']])


## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2080544195.py in <cell line: 0>()
----> 1 pred = xgbr.predict(df_test[['passenger_count', 'distance', 'year', 'month', 'day', 'day of week', 'hour']])

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

## === cell 47
submission = pd.DataFrame({"key": df_test['key'], "fare_amount": pred})
submission.to_csv('submission.csv',index=False)


## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/535797992.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"key": df_test['key'], "fare_amount": pred})
      2 submission.to_csv('submission.csv',index=False)

NameError: name 'pred' is not defined
