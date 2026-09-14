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

9.06494

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
import datetime as dt

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
import xgboost as xgb

train = pd.read_csv('../input/train.csv', nrows=10000)
test = pd.read_csv('../input/test.csv')

train.dtypes


## === cell 1
print('Sum of NaN values for each column')
print(train.isnull().sum())

train = train.dropna()
print('Sum of NaN values for each column after dropping NaN')
print(train.isnull().sum())


## === cell 2
train.describe()


## === cell 3

train = train.loc[(train['fare_amount'] > 0) & (train['fare_amount'] < 200)]
train = train.loc[(train['pickup_longitude'] > -300) & (train['pickup_longitude'] < 300)]
train = train.loc[(train['pickup_latitude'] > -300) & (train['pickup_latitude'] < 300)]
train = train.loc[(train['dropoff_longitude'] > -300) & (train['dropoff_longitude'] < 300)]
train = train.loc[(train['dropoff_latitude'] > -300) & (train['dropoff_latitude'] < 300)]
train = train.loc[train['passenger_count'] <= 8]
train.describe()


## === cell 4
print('Sum of NaN values for each column')
print(test.isnull().sum())


## === cell 5
combine = [test, train]
for dataset in combine:
    dataset['longitude_distance'] = dataset['pickup_longitude'] - dataset['dropoff_longitude']
    dataset['latitude_distance'] = dataset['pickup_latitude'] - dataset['dropoff_latitude']
    
    dataset['distance_travelled'] = (dataset['longitude_distance'] ** 2 + dataset['latitude_distance'] ** 2) ** .5
    dataset['distance_travelled_sin'] = np.sin((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5)
    dataset['distance_travelled_cos'] = np.cos((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5)
    dataset['distance_travelled_sin_sqrd'] = np.sin((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5) ** 2
    dataset['distance_travelled_cos_sqrd'] = np.cos((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5) ** 2
    
    R = 6371e3 # Metres
    phi1 = np.radians(dataset['pickup_latitude'])
    phi2 = np.radians(dataset['dropoff_latitude'])
    phi_chg = np.radians(dataset['pickup_latitude'] - dataset['dropoff_latitude'])
    delta_chg = np.radians(dataset['pickup_longitude'] - dataset['dropoff_longitude'])
    a = np.sin(phi_chg / 2) ** .5 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_chg / 2) ** .5
    c = 2 * np.arctan2(a ** .5, (1-a) ** .5)
    d = R * c
    dataset['haversine'] = d
    
    y = np.sin(delta_chg * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    dataset['bearing'] = np.arctan2(y, x)
    
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'])
    dataset['hour_of_day'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['week'] = dataset.pickup_datetime.dt.week
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['day_of_year'] = dataset.pickup_datetime.dt.dayofyear
    dataset['week_of_year'] = dataset.pickup_datetime.dt.weekofyear
    

train = train.loc[train['haversine'] != 0]
train = train.dropna()

    
train.head()


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4147312766.py in <cell line: 0>()
     35     dataset['hour_of_day'] = dataset.pickup_datetime.dt.hour
     36     dataset['day'] = dataset.pickup_datetime.dt.day
---> 37     dataset['week'] = dataset.pickup_datetime.dt.week
     38     dataset['month'] = dataset.pickup_datetime.dt.month
     39     dataset['day_of_year'] = dataset.pickup_datetime.dt.dayofyear

AttributeError: 'DatetimeProperties' object has no attribute 'week'

## === cell 6
print('Train data: Sum of NaN values for each column')
print(train.isnull().sum())
print('Test data: Sum of NaN values for each column')
print(test.isnull().sum())


## === cell 7
median = test['haversine'].median()
test['haversine'] = test['haversine'].fillna(median)


## === cell 8
colormap = plt.cm.RdBu
plt.figure(figsize=(20,20))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3114158282.py in <cell line: 0>()
      3 plt.figure(figsize=(20,20))
      4 plt.title('Pearson Correlation of Features', y=1.05, size=15)
----> 5 sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, 
      6             square=True, cmap=colormap, linecolor='white', annot=True)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in corr(self, method, min_periods, numeric_only)
  11047         cols = data.columns
  11048         idx = cols.copy()
> 11049         mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
  11050 
  11051         if method == "pearson":

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in to_numpy(self, dtype, copy, na_value)
   1991         if dtype is not None:
   1992             dtype = np.dtype(dtype)
-> 1993         result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
   1994         if result.dtype is not dtype:
   1995             result = np.asarray(result, dtype=dtype)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in as_array(self, dtype, copy, na_value)
   1692                 arr.flags.writeable = False
   1693         else:
-> 1694             arr = self._interleave(dtype=dtype, na_value=na_value)
   1695             # The underlying data was copied within _interleave, so no need
   1696             # to further copy if copy=True or setting na_value

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in _interleave(self, dtype, na_value)
   1751             else:
   1752                 arr = blk.get_values(dtype)
-> 1753             result[rl.indexer] = arr
   1754             itemmask[rl.indexer] = 1
   1755 

ValueError: could not convert string to float: '2009-06-15 17:26:21.0000001'

## === cell 9
train_features_to_keep = ['haversine', 'fare_amount']
train.drop(train.columns.difference(train_features_to_keep), 1, inplace=True)

test_features_to_keep = ['haversine', 'key']
test.drop(test.columns.difference(test_features_to_keep), 1, inplace=True)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/115699782.py in <cell line: 0>()
      1 # Let's drop all the irrelevant features
      2 train_features_to_keep = ['haversine', 'fare_amount']
----> 3 train.drop(train.columns.difference(train_features_to_keep), 1, inplace=True)
      4 
      5 test_features_to_keep = ['haversine', 'key']

TypeError: DataFrame.drop() takes from 1 to 2 positional arguments but 3 positional arguments (and 1 keyword-only argument) were given

## === cell 11
x_train = train.drop('fare_amount', axis=1)
y_train = train['fare_amount']
x_test = test.drop('key', axis=1)

from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(x_train, y_train)
regr_pred = regr.predict(x_test)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
rfr_pred = rfr.predict(x_test)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4180467674.py in <cell line: 0>()
      9 from sklearn.linear_model import LinearRegression
     10 regr = LinearRegression()
---> 11 regr.fit(x_train, y_train)
     12 regr_pred = regr.predict(x_test)
     13 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in fit(self, X, y, sample_weight)
    646         accept_sparse = False if self.positive else ["csr", "csc", "coo"]
    647 
--> 648         X, y = self._validate_data(
    649             X, y, accept_sparse=accept_sparse, y_numeric=True, multi_output=True
    650         )

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1104         )
   1105 
-> 1106     X = check_array(
   1107         X,
   1108         accept_sparse=accept_sparse,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    877                     array = xp.astype(array, dtype, copy=False)
    878                 else:
--> 879                     array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
    880             except ComplexWarning as complex_warning:
    881                 raise ValueError(

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_array_api.py in _asarray_with_order(array, dtype, order, copy, xp)
    183     if xp.__name__ in {"numpy", "numpy.array_api"}:
    184         # Use NumPy API to support order
--> 185         array = numpy.asarray(array, order=order, dtype=dtype)
    186         return xp.asarray(array, copy=copy)
    187     else:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __array__(self, dtype, copy)
   2151     ) -> np.ndarray:
   2152         values = self._values
-> 2153         arr = np.asarray(values, dtype=dtype)
   2154         if (
   2155             astype_is_view(values.dtype, arr.dtype)

ValueError: could not convert string to float: '2009-06-15 17:26:21.0000001'

## === cell 12
x_pred = test.drop('key', axis=1)

x_train,x_test,y_train,y_test = train_test_split(train.drop('fare_amount',axis=1),train.pop('fare_amount'),random_state=123,test_size=0.2)

def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'}
                    ,dtrain=matrix_train,num_boost_round=200, 
                    early_stopping_rounds=20,evals=[(matrix_test,'test')],)
    return model

model=XGBmodel(x_train,x_test,y_train,y_test)
xgb_pred = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3087475804.py in <cell line: 0>()
     13     return model
     14 
---> 15 model=XGBmodel(x_train,x_test,y_train,y_test)
     16 xgb_pred = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)

/tmp/ipykernel_11/3087475804.py in XGBmodel(x_train, x_test, y_train, y_test)
      6 
      7 def XGBmodel(x_train,x_test,y_train,y_test):
----> 8     matrix_train = xgb.DMatrix(x_train,label=y_train)
      9     matrix_test = xgb.DMatrix(x_test,label=y_test)
     10     model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'}

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in __init__(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)
    855             return
    856 
--> 857         handle, feature_names, feature_types = dispatch_data_backend(
    858             data,
    859             missing=self.missing,

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in dispatch_data_backend(data, missing, threads, feature_names, feature_types, enable_categorical, data_split_mode)
   1087         data = pd.DataFrame(data)
   1088     if _is_pandas_df(data):
-> 1089         return _from_pandas_df(
   1090             data, enable_categorical, missing, threads, feature_names, feature_types
   1091         )

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _from_pandas_df(data, enable_categorical, missing, nthread, feature_names, feature_types)
    520     feature_types: Optional[FeatureTypes],
    521 ) -> DispatchedDataBackendReturnType:
--> 522     data, feature_names, feature_types = _transform_pandas_df(
    523         data, enable_categorical, feature_names, feature_types
    524     )

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _transform_pandas_df(data, enable_categorical, feature_names, feature_types, meta, meta_type)
    488             or is_pa_ext_dtype(dtype)
    489         ):
--> 490             _invalid_dataframe_dtype(data)
    491         if is_pa_ext_dtype(dtype):
    492             pyarrow_extension = True

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _invalid_dataframe_dtype(data)
    306     type_err = "DataFrame.dtypes for data must be int, float, bool or category."
    307     msg = f"""{type_err} {_ENABLE_CAT_ERR} {err}"""
--> 308     raise ValueError(msg)
    309 
    310 

ValueError: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, The experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:key: object, pickup_datetime: object

## === cell 13
regr_pred, rfr_pred, xgb_pred


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1633247124.py in <cell line: 0>()
----> 1 regr_pred, rfr_pred, xgb_pred

NameError: name 'regr_pred' is not defined

## === cell 14
regr_weight = 1
rfr_weight = 1
xgb_weight = 3
prediction = (regr_pred * regr_weight + rfr_pred * rfr_weight + xgb_pred * xgb_weight) / (regr_weight + rfr_weight + xgb_weight)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3455956942.py in <cell line: 0>()
      3 rfr_weight = 1
      4 xgb_weight = 3
----> 5 prediction = (regr_pred * regr_weight + rfr_pred * rfr_weight + xgb_pred * xgb_weight) / (regr_weight + rfr_weight + xgb_weight)

NameError: name 'regr_pred' is not defined

## === cell 15
prediction


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/695847562.py in <cell line: 0>()
----> 1 prediction

NameError: name 'prediction' is not defined

## === cell 16
submission = pd.DataFrame({
        "key": test['key'],
        "fare_amount": prediction.round(2)
})

submission.to_csv('sub_fare.csv',index=False)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2856729682.py in <cell line: 0>()
      2 submission = pd.DataFrame({
      3         "key": test['key'],
----> 4         "fare_amount": prediction.round(2)
      5 })
      6 

NameError: name 'prediction' is not defined

## === cell 17
submission


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/351652756.py in <cell line: 0>()
----> 1 submission

NameError: name 'submission' is not defined
