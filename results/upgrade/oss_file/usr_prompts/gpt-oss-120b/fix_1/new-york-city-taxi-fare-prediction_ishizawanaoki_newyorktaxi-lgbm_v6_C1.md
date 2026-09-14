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

3.10

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
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

3.7695

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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv", nrows = 1_000_000)
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
sample_submission = pd.read_csv("../input/new-york-city-taxi-fare-prediction/sample_submission.csv")


## === cell 2
train.isnull().sum()


## === cell 3
train.dropna(inplace=True)


## === cell 4
train.describe()


## === cell 5
train.quantile(0.99)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/201084695.py in <cell line: 0>()
      1 ## 分位数99%
----> 2 train.quantile(0.99)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in quantile(self, q, axis, numeric_only, interpolation, method)
  12144             # error: List item 0 has incompatible type "float | ExtensionArray |
  12145             # ndarray[Any, Any] | Index | Series | Sequence[float]"; expected "float"
> 12146             res_df = self.quantile(
  12147                 [q],  # type: ignore[list-item]
  12148                 axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in quantile(self, q, axis, numeric_only, interpolation, method)
  12189             )
  12190         if method == "single":
> 12191             res = data._mgr.quantile(qs=q, interpolation=interpolation)
  12192         elif method == "table":
  12193             valid_interpolation = {"nearest", "lower", "higher"}

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in quantile(self, qs, interpolation)
   1546         new_axes[1] = Index(qs, dtype=np.float64)
   1547 
-> 1548         blocks = [
   1549             blk.quantile(qs=qs, interpolation=interpolation) for blk in self.blocks
   1550         ]

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in <listcomp>(.0)
   1547 
   1548         blocks = [
-> 1549             blk.quantile(qs=qs, interpolation=interpolation) for blk in self.blocks
   1550         ]
   1551 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in quantile(self, qs, interpolation)
   1889         assert is_list_like(qs)  # caller is responsible for this
   1890 
-> 1891         result = quantile_compat(self.values, np.asarray(qs._values), interpolation)
   1892         # ensure_block_shape needed for cases where we start with EA and result
   1893         #  is ndarray, e.g. IntegerArray, SparseArray

/usr/local/lib/python3.11/dist-packages/pandas/core/array_algos/quantile.py in quantile_compat(values, qs, interpolation)
     37         fill_value = na_value_for_dtype(values.dtype, compat=False)
     38         mask = isna(values)
---> 39         return quantile_with_mask(values, mask, fill_value, qs, interpolation)
     40     else:
     41         return values._quantile(qs, interpolation)

/usr/local/lib/python3.11/dist-packages/pandas/core/array_algos/quantile.py in quantile_with_mask(values, mask, fill_value, qs, interpolation)
     95         result = np.repeat(flat, len(values)).reshape(len(values), len(qs))
     96     else:
---> 97         result = _nanpercentile(
     98             values,
     99             qs * 100.0,

/usr/local/lib/python3.11/dist-packages/pandas/core/array_algos/quantile.py in _nanpercentile(values, qs, na_value, mask, interpolation)
    216         return result
    217     else:
--> 218         return np.percentile(
    219             values,
    220             qs,

/usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py in percentile(a, q, axis, out, overwrite_input, method, keepdims, interpolation)
   4281     if not _quantile_is_valid(q):
   4282         raise ValueError("Percentiles must be in the range [0, 100]")
-> 4283     return _quantile_unchecked(
   4284         a, q, axis, out, overwrite_input, method, keepdims)
   4285 

/usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py in _quantile_unchecked(a, q, axis, out, overwrite_input, method, keepdims)
   4553                         keepdims=False):
   4554     """Assumes that q is in [0, 1], and is an ndarray"""
-> 4555     return _ureduce(a,
   4556                     func=_quantile_ureduce_func,
   4557                     q=q,

/usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py in _ureduce(a, func, keepdims, **kwargs)
   3821                 kwargs['out'] = out[(Ellipsis, ) + index_out]
   3822 
-> 3823     r = func(a, **kwargs)
   3824 
   3825     if out is not None:

/usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py in _quantile_ureduce_func(a, q, axis, out, overwrite_input, method)
   4720         else:
   4721             arr = a.copy()
-> 4722     result = _quantile(arr,
   4723                        quantiles=q,
   4724                        axis=axis,

/usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py in _quantile(arr, quantiles, axis, method, out)
   4839         result_shape = virtual_indexes.shape + (1,) * (arr.ndim - 1)
   4840         gamma = gamma.reshape(result_shape)
-> 4841         result = _lerp(previous,
   4842                        next,
   4843                        gamma,

/usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py in _lerp(a, b, t, out)
   4653         Output array.
   4654     """
-> 4655     diff_b_a = subtract(b, a)
   4656     # asanyarray is a stop-gap until gh-13105
   4657     lerp_interpolation = asanyarray(add(a, diff_b_a * t, out=out))

TypeError: unsupported operand type(s) for -: 'str' and 'str'

## === cell 6
train.quantile(0.01)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/540126338.py in <cell line: 0>()
      1 ## 分位数1%
----> 2 train.quantile(0.01)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in quantile(self, q, axis, numeric_only, interpolation, method)
  12144             # error: List item 0 has incompatible type "float | ExtensionArray |
  12145             # ndarray[Any, Any] | Index | Series | Sequence[float]"; expected "float"
> 12146             res_df = self.quantile(
  12147                 [q],  # type: ignore[list-item]
  12148                 axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in quantile(self, q, axis, numeric_only, interpolation, method)
  12189             )
  12190         if method == "single":
> 12191             res = data._mgr.quantile(qs=q, interpolation=interpolation)
  12192         elif method == "table":
  12193             valid_interpolation = {"nearest", "lower", "higher"}

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in quantile(self, qs, interpolation)
   1546         new_axes[1] = Index(qs, dtype=np.float64)
   1547 
-> 1548         blocks = [
   1549             blk.quantile(qs=qs, interpolation=interpolation) for blk in self.blocks
   1550         ]

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in <listcomp>(.0)
   1547 
   1548         blocks = [
-> 1549             blk.quantile(qs=qs, interpolation=interpolation) for blk in self.blocks
   1550         ]
   1551 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in quantile(self, qs, interpolation)
   1889         assert is_list_like(qs)  # caller is responsible for this
   1890 
-> 1891         result = quantile_compat(self.values, np.asarray(qs._values), interpolation)
   1892         # ensure_block_shape needed for cases where we start with EA and result
   1893         #  is ndarray, e.g. IntegerArray, SparseArray

/usr/local/lib/python3.11/dist-packages/pandas/core/array_algos/quantile.py in quantile_compat(values, qs, interpolation)
     37         fill_value = na_value_for_dtype(values.dtype, compat=False)
     38         mask = isna(values)
---> 39         return quantile_with_mask(values, mask, fill_value, qs, interpolation)
     40     else:
     41         return values._quantile(qs, interpolation)

/usr/local/lib/python3.11/dist-packages/pandas/core/array_algos/quantile.py in quantile_with_mask(values, mask, fill_value, qs, interpolation)
     95         result = np.repeat(flat, len(values)).reshape(len(values), len(qs))
     96     else:
---> 97         result = _nanpercentile(
     98             values,
     99             qs * 100.0,

/usr/local/lib/python3.11/dist-packages/pandas/core/array_algos/quantile.py in _nanpercentile(values, qs, na_value, mask, interpolation)
    216         return result
    217     else:
--> 218         return np.percentile(
    219             values,
    220             qs,

/usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py in percentile(a, q, axis, out, overwrite_input, method, keepdims, interpolation)
   4281     if not _quantile_is_valid(q):
   4282         raise ValueError("Percentiles must be in the range [0, 100]")
-> 4283     return _quantile_unchecked(
   4284         a, q, axis, out, overwrite_input, method, keepdims)
   4285 

/usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py in _quantile_unchecked(a, q, axis, out, overwrite_input, method, keepdims)
   4553                         keepdims=False):
   4554     """Assumes that q is in [0, 1], and is an ndarray"""
-> 4555     return _ureduce(a,
   4556                     func=_quantile_ureduce_func,
   4557                     q=q,

/usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py in _ureduce(a, func, keepdims, **kwargs)
   3821                 kwargs['out'] = out[(Ellipsis, ) + index_out]
   3822 
-> 3823     r = func(a, **kwargs)
   3824 
   3825     if out is not None:

/usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py in _quantile_ureduce_func(a, q, axis, out, overwrite_input, method)
   4720         else:
   4721             arr = a.copy()
-> 4722     result = _quantile(arr,
   4723                        quantiles=q,
   4724                        axis=axis,

/usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py in _quantile(arr, quantiles, axis, method, out)
   4839         result_shape = virtual_indexes.shape + (1,) * (arr.ndim - 1)
   4840         gamma = gamma.reshape(result_shape)
-> 4841         result = _lerp(previous,
   4842                        next,
   4843                        gamma,

/usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py in _lerp(a, b, t, out)
   4653         Output array.
   4654     """
-> 4655     diff_b_a = subtract(b, a)
   4656     # asanyarray is a stop-gap until gh-13105
   4657     lerp_interpolation = asanyarray(add(a, diff_b_a * t, out=out))

TypeError: unsupported operand type(s) for -: 'str' and 'str'

## === cell 7
train = train.query('1 <= passenger_count <= 6 and 3.3 <= fare_amount <= 52.33')
train.describe()


## === cell 8
train.reset_index(drop=True, inplace=True)
train


## === cell 9
data = pd.concat([train, test], sort=False)


## === cell 10
data.head()


## === cell 11
data = data.drop('pickup_datetime', axis=1)
data['key'] = data['key'].str.replace('[- :]', '')
data['key'] = data['key'].astype(float)

data.head()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/767791090.py in <cell line: 0>()
      3 data = data.drop('pickup_datetime', axis=1)
      4 data['key'] = data['key'].str.replace('[- :]', '')
----> 5 data['key'] = data['key'].astype(float)
      6 
      7 data.head()

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
    131     if copy or arr.dtype == object or dtype == object:
    132         # Explicit copy, or required since NumPy can't view from / to object.
--> 133         return arr.astype(dtype, copy=True)
    134 
    135     return arr.astype(dtype, copy=copy)

ValueError: could not convert string to float: '2009-06-15 17:26:21.0000001'

## === cell 12
train = data[:len(train)]
test = data[len(train):]

y_train = train['fare_amount']
X_train = train.drop('fare_amount', axis=1)
X_test = test.drop('fare_amount', axis=1)

X_train.head()


## === cell 13
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train),))
cv = KFold(n_splits=5, shuffle=True, random_state=0)

categorical_features = []


## === cell 14
import lightgbm as lgb

params = {
    'objective': 'regression',
    'max_bin': 300,
    'learning_rate': 0.05,
    'num_leaves': 40,
}

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train.loc[train_index, :]
    X_val = X_train.loc[valid_index, :]
    y_tr = y_train[train_index]
    y_val = y_train[valid_index]

    lgb_train = lgb.Dataset(X_tr, y_tr,
                            categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(X_val, y_val,
                           reference=lgb_train,
                           categorical_feature=categorical_features)

    model = lgb.train(params, lgb_train,
                      valid_sets=[lgb_train, lgb_eval],
                      verbose_eval=10,
                      num_boost_round=1000,
                      early_stopping_rounds=10)

    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    y_preds.append(y_pred)
    models.append(model)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/953766480.py in <cell line: 0>()
     20                            categorical_feature=categorical_features)
     21 
---> 22     model = lgb.train(params, lgb_train,
     23                       valid_sets=[lgb_train, lgb_eval],
     24                       verbose_eval=10,

TypeError: train() got an unexpected keyword argument 'verbose_eval'

## === cell 15
pd.DataFrame(oof_train).to_csv('oof_train_kfold.csv', index=False)

scores = [
    m.best_score['valid_1']['l2'] for m in models
]
score = sum(scores) / len(scores)
print('===CV scores===')
print(scores)
print(score)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
/tmp/ipykernel_11/1628822012.py in <cell line: 0>()
      4     m.best_score['valid_1']['l2'] for m in models
      5 ]
----> 6 score = sum(scores) / len(scores)
      7 print('===CV scores===')
      8 print(scores)

ZeroDivisionError: division by zero

## === cell 16
from sklearn.metrics import mean_squared_error

y_pred_oof = oof_train
np.sqrt(mean_squared_error(y_train, y_pred_oof))


## === cell 17
len(y_preds)


## === cell 18
y_preds[0][:10]


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3255462404.py in <cell line: 0>()
----> 1 y_preds[0][:10]

IndexError: list index out of range

## === cell 19
y_sub = sum(y_preds) / len(y_preds)
y_sub[:10]


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
/tmp/ipykernel_11/3911375596.py in <cell line: 0>()
----> 1 y_sub = sum(y_preds) / len(y_preds)
      2 y_sub[:10]

ZeroDivisionError: division by zero

## === cell 20
sub_lgb = sample_submission

sub_lgb['fare_amount'] = y_sub
sub_lgb.to_csv("submission_lightgbm.csv", index=False)

sub_lgb.head()


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3178377420.py in <cell line: 0>()
      1 sub_lgb = sample_submission
      2 
----> 3 sub_lgb['fare_amount'] = y_sub
      4 sub_lgb.to_csv("submission_lightgbm.csv", index=False)
      5 

NameError: name 'y_sub' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Submission should have a fare_amount column
