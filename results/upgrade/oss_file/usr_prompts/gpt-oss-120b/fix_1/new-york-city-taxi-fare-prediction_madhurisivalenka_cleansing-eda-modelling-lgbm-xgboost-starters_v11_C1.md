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

3.61434

# 6. Current score

15.25184

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


## === cell 1
train = pd.read_csv("../input/train.csv", nrows = 1000000)
test = pd.read_csv("../input/test.csv")


## === cell 2
train.shape


## === cell 3
test.shape


## === cell 4
train.head(10)


## === cell 5
train.describe()


## === cell 6
train.isnull().sum().sort_values(ascending=False)


## === cell 7
test.isnull().sum().sort_values(ascending=False)


## === cell 8
train = train.drop(train[train.isnull().any(1)].index, axis = 0)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2334086834.py in <cell line: 0>()
      1 #drop the missing values
----> 2 train = train.drop(train[train.isnull().any(1)].index, axis = 0)

TypeError: DataFrame.any() takes 1 positional argument but 2 were given

## === cell 9
train.shape


## === cell 10
train['fare_amount'].describe()


## === cell 11
from collections import Counter
Counter(train['fare_amount']<0)


## === cell 12
train = train.drop(train[train['fare_amount']<0].index, axis=0)
train.shape


## === cell 13
train['fare_amount'].describe()


## === cell 14
train['fare_amount'].sort_values(ascending=False)


## === cell 15
train['passenger_count'].describe()


## === cell 16
train[train['passenger_count']>6]


## === cell 17
train = train.drop(train[train['passenger_count']==208].index, axis = 0)


## === cell 18
train['passenger_count'].describe()


## === cell 19
train['pickup_latitude'].describe()


## === cell 20
train[train['pickup_latitude']<-90]


## === cell 21
train[train['pickup_latitude']>90]


## === cell 22
train = train.drop(((train[train['pickup_latitude']<-90])|(train[train['pickup_latitude']>90])).index, axis=0)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in na_logical_op(x, y, op)
    361         #  (xint or xbool) and (yint or bool)
--> 362         result = op(x, y)
    363     except TypeError:

TypeError: unsupported operand type(s) for |: 'float' and 'float'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1850217903.py in <cell line: 0>()
      1 #We need to drop these outliers
----> 2 train = train.drop(((train[train['pickup_latitude']<-90])|(train[train['pickup_latitude']>90])).index, axis=0)

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/common.py in new_method(self, other)
     74         other = item_from_zerodim(other)
     75 
---> 76         return method(self, other)
     77 
     78     return new_method

/usr/local/lib/python3.11/dist-packages/pandas/core/arraylike.py in __or__(self, other)
     76     @unpack_zerodim_and_defer("__or__")
     77     def __or__(self, other):
---> 78         return self._logical_method(other, operator.or_)
     79 
     80     @unpack_zerodim_and_defer("__ror__")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _arith_method(self, other, op)
   7911 
   7912         with np.errstate(all="ignore"):
-> 7913             new_data = self._dispatch_frame_op(other, op, axis=axis)
   7914         return self._construct_result(new_data)
   7915 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _dispatch_frame_op(self, right, func, axis)
   7954 
   7955             # TODO operate_blockwise expects a manager of the same type
-> 7956             bm = self._mgr.operate_blockwise(
   7957                 # error: Argument 1 to "operate_blockwise" of "ArrayManager" has
   7958                 # incompatible type "Union[ArrayManager, BlockManager]"; expected

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in operate_blockwise(self, other, array_op)
   1509         Apply array_op blockwise with another (aligned) BlockManager.
   1510         """
-> 1511         return operate_blockwise(self, other, array_op)
   1512 
   1513     def _equal_values(self: BlockManager, other: BlockManager) -> bool:

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/ops.py in operate_blockwise(left, right, array_op)
     63     res_blks: list[Block] = []
     64     for lvals, rvals, locs, left_ea, right_ea, rblk in _iter_block_pairs(left, right):
---> 65         res_values = array_op(lvals, rvals)
     66         if (
     67             left_ea

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in logical_op(left, right, op)
    452             is_other_int_dtype = lib.is_integer(rvalues)
    453 
--> 454         res_values = na_logical_op(lvalues, rvalues, op)
    455 
    456         # For int vs int `^`, `|`, `&` are bitwise operators and return

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in na_logical_op(x, y, op)
    367             x = ensure_object(x)
    368             y = ensure_object(y)
--> 369             result = libops.vec_binop(x.ravel(), y.ravel(), op)
    370         else:
    371             # let null fall thru

ops.pyx in pandas._libs.ops.vec_binop()

ops.pyx in pandas._libs.ops.vec_binop()

TypeError: unsupported operand type(s) for |: 'float' and 'bool'

## === cell 23
train.shape


## === cell 24
train['pickup_longitude'].describe()


## === cell 25
train[train['pickup_longitude']<-180]


## === cell 26
train[train['pickup_longitude']>180]


## === cell 27
train = train.drop(((train[train['pickup_longitude']<-180])|(train[train['pickup_longitude']>180])).index, axis=0)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in na_logical_op(x, y, op)
    361         #  (xint or xbool) and (yint or bool)
--> 362         result = op(x, y)
    363     except TypeError:

TypeError: unsupported operand type(s) for |: 'float' and 'bool'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2808801605.py in <cell line: 0>()
----> 1 train = train.drop(((train[train['pickup_longitude']<-180])|(train[train['pickup_longitude']>180])).index, axis=0)

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/common.py in new_method(self, other)
     74         other = item_from_zerodim(other)
     75 
---> 76         return method(self, other)
     77 
     78     return new_method

/usr/local/lib/python3.11/dist-packages/pandas/core/arraylike.py in __or__(self, other)
     76     @unpack_zerodim_and_defer("__or__")
     77     def __or__(self, other):
---> 78         return self._logical_method(other, operator.or_)
     79 
     80     @unpack_zerodim_and_defer("__ror__")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _arith_method(self, other, op)
   7911 
   7912         with np.errstate(all="ignore"):
-> 7913             new_data = self._dispatch_frame_op(other, op, axis=axis)
   7914         return self._construct_result(new_data)
   7915 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _dispatch_frame_op(self, right, func, axis)
   7954 
   7955             # TODO operate_blockwise expects a manager of the same type
-> 7956             bm = self._mgr.operate_blockwise(
   7957                 # error: Argument 1 to "operate_blockwise" of "ArrayManager" has
   7958                 # incompatible type "Union[ArrayManager, BlockManager]"; expected

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in operate_blockwise(self, other, array_op)
   1509         Apply array_op blockwise with another (aligned) BlockManager.
   1510         """
-> 1511         return operate_blockwise(self, other, array_op)
   1512 
   1513     def _equal_values(self: BlockManager, other: BlockManager) -> bool:

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/ops.py in operate_blockwise(left, right, array_op)
     63     res_blks: list[Block] = []
     64     for lvals, rvals, locs, left_ea, right_ea, rblk in _iter_block_pairs(left, right):
---> 65         res_values = array_op(lvals, rvals)
     66         if (
     67             left_ea

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in logical_op(left, right, op)
    452             is_other_int_dtype = lib.is_integer(rvalues)
    453 
--> 454         res_values = na_logical_op(lvalues, rvalues, op)
    455 
    456         # For int vs int `^`, `|`, `&` are bitwise operators and return

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in na_logical_op(x, y, op)
    367             x = ensure_object(x)
    368             y = ensure_object(y)
--> 369             result = libops.vec_binop(x.ravel(), y.ravel(), op)
    370         else:
    371             # let null fall thru

ops.pyx in pandas._libs.ops.vec_binop()

ops.pyx in pandas._libs.ops.vec_binop()

TypeError: unsupported operand type(s) for |: 'float' and 'bool'

## === cell 28
train.shape


## === cell 29
train[train['dropoff_latitude']<-90]


## === cell 30
train[train['dropoff_latitude']>90]


## === cell 31
train = train.drop(((train[train['dropoff_latitude']<-90])|(train[train['dropoff_latitude']>90])).index, axis=0)


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in na_logical_op(x, y, op)
    361         #  (xint or xbool) and (yint or bool)
--> 362         result = op(x, y)
    363     except TypeError:

TypeError: unsupported operand type(s) for |: 'float' and 'float'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3820880749.py in <cell line: 0>()
----> 1 train = train.drop(((train[train['dropoff_latitude']<-90])|(train[train['dropoff_latitude']>90])).index, axis=0)

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/common.py in new_method(self, other)
     74         other = item_from_zerodim(other)
     75 
---> 76         return method(self, other)
     77 
     78     return new_method

/usr/local/lib/python3.11/dist-packages/pandas/core/arraylike.py in __or__(self, other)
     76     @unpack_zerodim_and_defer("__or__")
     77     def __or__(self, other):
---> 78         return self._logical_method(other, operator.or_)
     79 
     80     @unpack_zerodim_and_defer("__ror__")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _arith_method(self, other, op)
   7911 
   7912         with np.errstate(all="ignore"):
-> 7913             new_data = self._dispatch_frame_op(other, op, axis=axis)
   7914         return self._construct_result(new_data)
   7915 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _dispatch_frame_op(self, right, func, axis)
   7954 
   7955             # TODO operate_blockwise expects a manager of the same type
-> 7956             bm = self._mgr.operate_blockwise(
   7957                 # error: Argument 1 to "operate_blockwise" of "ArrayManager" has
   7958                 # incompatible type "Union[ArrayManager, BlockManager]"; expected

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in operate_blockwise(self, other, array_op)
   1509         Apply array_op blockwise with another (aligned) BlockManager.
   1510         """
-> 1511         return operate_blockwise(self, other, array_op)
   1512 
   1513     def _equal_values(self: BlockManager, other: BlockManager) -> bool:

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/ops.py in operate_blockwise(left, right, array_op)
     63     res_blks: list[Block] = []
     64     for lvals, rvals, locs, left_ea, right_ea, rblk in _iter_block_pairs(left, right):
---> 65         res_values = array_op(lvals, rvals)
     66         if (
     67             left_ea

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in logical_op(left, right, op)
    452             is_other_int_dtype = lib.is_integer(rvalues)
    453 
--> 454         res_values = na_logical_op(lvalues, rvalues, op)
    455 
    456         # For int vs int `^`, `|`, `&` are bitwise operators and return

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in na_logical_op(x, y, op)
    367             x = ensure_object(x)
    368             y = ensure_object(y)
--> 369             result = libops.vec_binop(x.ravel(), y.ravel(), op)
    370         else:
    371             # let null fall thru

ops.pyx in pandas._libs.ops.vec_binop()

ops.pyx in pandas._libs.ops.vec_binop()

TypeError: unsupported operand type(s) for |: 'float' and 'bool'

## === cell 32
train.shape


## === cell 33
train[train['dropoff_latitude']<-180]|train[train['dropoff_latitude']>180]


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in na_logical_op(x, y, op)
    361         #  (xint or xbool) and (yint or bool)
--> 362         result = op(x, y)
    363     except TypeError:

TypeError: unsupported operand type(s) for |: 'float' and 'float'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4028530560.py in <cell line: 0>()
----> 1 train[train['dropoff_latitude']<-180]|train[train['dropoff_latitude']>180]

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/common.py in new_method(self, other)
     74         other = item_from_zerodim(other)
     75 
---> 76         return method(self, other)
     77 
     78     return new_method

/usr/local/lib/python3.11/dist-packages/pandas/core/arraylike.py in __or__(self, other)
     76     @unpack_zerodim_and_defer("__or__")
     77     def __or__(self, other):
---> 78         return self._logical_method(other, operator.or_)
     79 
     80     @unpack_zerodim_and_defer("__ror__")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _arith_method(self, other, op)
   7911 
   7912         with np.errstate(all="ignore"):
-> 7913             new_data = self._dispatch_frame_op(other, op, axis=axis)
   7914         return self._construct_result(new_data)
   7915 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _dispatch_frame_op(self, right, func, axis)
   7954 
   7955             # TODO operate_blockwise expects a manager of the same type
-> 7956             bm = self._mgr.operate_blockwise(
   7957                 # error: Argument 1 to "operate_blockwise" of "ArrayManager" has
   7958                 # incompatible type "Union[ArrayManager, BlockManager]"; expected

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in operate_blockwise(self, other, array_op)
   1509         Apply array_op blockwise with another (aligned) BlockManager.
   1510         """
-> 1511         return operate_blockwise(self, other, array_op)
   1512 
   1513     def _equal_values(self: BlockManager, other: BlockManager) -> bool:

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/ops.py in operate_blockwise(left, right, array_op)
     63     res_blks: list[Block] = []
     64     for lvals, rvals, locs, left_ea, right_ea, rblk in _iter_block_pairs(left, right):
---> 65         res_values = array_op(lvals, rvals)
     66         if (
     67             left_ea

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in logical_op(left, right, op)
    452             is_other_int_dtype = lib.is_integer(rvalues)
    453 
--> 454         res_values = na_logical_op(lvalues, rvalues, op)
    455 
    456         # For int vs int `^`, `|`, `&` are bitwise operators and return

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in na_logical_op(x, y, op)
    367             x = ensure_object(x)
    368             y = ensure_object(y)
--> 369             result = libops.vec_binop(x.ravel(), y.ravel(), op)
    370         else:
    371             # let null fall thru

ops.pyx in pandas._libs.ops.vec_binop()

ops.pyx in pandas._libs.ops.vec_binop()

TypeError: unsupported operand type(s) for |: 'float' and 'bool'

## === cell 34
train.dtypes


## === cell 35
train['key'] = pd.to_datetime(train['key'])
train['pickup_datetime']  = pd.to_datetime(train['pickup_datetime'])


## === cell 36
test['key'] = pd.to_datetime(test['key'])
test['pickup_datetime']  = pd.to_datetime(test['pickup_datetime'])


## === cell 37
train.dtypes


## === cell 38
test.dtypes


## === cell 39
train.head()


## === cell 40
test.head()


## === cell 41
def haversine_distance(lat1, long1, lat2, long2):
    data = [train, test]
    for i in data:
        R = 6371  #radius of earth in kilometers
        phi1 = np.radians(i[lat1])
        phi2 = np.radians(i[lat2])
    
        delta_phi = np.radians(i[lat2]-i[lat1])
        delta_lambda = np.radians(i[long2]-i[long1])
    
        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
        d = (R * c) #in kilometers
        i['H_Distance'] = d
    return d


## === cell 42
haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')


## === cell 43
train['H_Distance'].head(10)


## === cell 44
test['H_Distance'].head(10)


## === cell 45
train.head(10)


## === cell 46
test.head(10)


## === cell 47
data = [train,test]
for i in data:
    i['Year'] = i['pickup_datetime'].dt.year
    i['Month'] = i['pickup_datetime'].dt.month
    i['Date'] = i['pickup_datetime'].dt.day
    i['Day of Week'] = i['pickup_datetime'].dt.dayofweek
    i['Hour'] = i['pickup_datetime'].dt.hour


## === cell 48
train.head()


## === cell 49
test.head()


## === cell 50
plt.figure(figsize=(15,7))
plt.hist(train['passenger_count'], bins=15)
plt.xlabel('No. of Passengers')
plt.ylabel('Frequency')


## === cell 51
plt.figure(figsize=(15,7))
plt.scatter(x=train['passenger_count'], y=train['fare_amount'], s=1.5)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')


## === cell 52
plt.figure(figsize=(15,7))
plt.scatter(x=train['Date'], y=train['fare_amount'], s=1.5)
plt.xlabel('Date')
plt.ylabel('Fare')


## === cell 53
plt.figure(figsize=(15,7))
plt.hist(train['Hour'], bins=100)
plt.xlabel('Hour')
plt.ylabel('Frequency')


## === cell 54
plt.figure(figsize=(15,7))
plt.scatter(x=train['Hour'], y=train['fare_amount'], s=1.5)
plt.xlabel('Hour')
plt.ylabel('Fare')


## === cell 55
plt.figure(figsize=(15,7))
plt.hist(train['Day of Week'], bins=100)
plt.xlabel('Day of Week')
plt.ylabel('Frequency')


## === cell 56
plt.figure(figsize=(15,7))
plt.scatter(x=train['Day of Week'], y=train['fare_amount'], s=1.5)
plt.xlabel('Day of Week')
plt.ylabel('Fare')


## === cell 57
train.sort_values(['H_Distance','fare_amount'], ascending=False)


## === cell 58
len(train)


## === cell 59
bins_0 = train.loc[(train['H_Distance'] == 0), ['H_Distance']]
bins_1 = train.loc[(train['H_Distance'] > 0) & (train['H_Distance'] <= 10),['H_Distance']]
bins_2 = train.loc[(train['H_Distance'] > 10) & (train['H_Distance'] <= 50),['H_Distance']]
bins_3 = train.loc[(train['H_Distance'] > 50) & (train['H_Distance'] <= 100),['H_Distance']]
bins_4 = train.loc[(train['H_Distance'] > 100) & (train['H_Distance'] <= 200),['H_Distance']]
bins_5 = train.loc[(train['H_Distance'] > 200) & (train['H_Distance'] <= 300),['H_Distance']]
bins_6 = train.loc[(train['H_Distance'] > 300),['H_Distance']]
bins_0['bins'] = '0'
bins_1['bins'] = '0-10'
bins_2['bins'] = '11-50'
bins_3['bins'] = '51-100'
bins_4['bins'] = '100-200'
bins_5['bins'] = '201-300'
bins_6['bins'] = '>300'
dist_bins =pd.concat([bins_0,bins_1,bins_2,bins_3,bins_4,bins_5,bins_6])
dist_bins.columns


## === cell 60
plt.figure(figsize=(15,7))
plt.hist(dist_bins['bins'], bins=75)
plt.xlabel('Bins')
plt.ylabel('Frequency')


## === cell 61
Counter(dist_bins['bins'])


## === cell 62
train.loc[((train['pickup_latitude']==0) & (train['pickup_longitude']==0))&((train['dropoff_latitude']!=0) & (train['dropoff_longitude']!=0)) & (train['fare_amount']==0)]


## === cell 63
train = train.drop(train.loc[((train['pickup_latitude']==0) & (train['pickup_longitude']==0))&((train['dropoff_latitude']!=0) & (train['dropoff_longitude']!=0)) & (train['fare_amount']==0)].index, axis=0)


## === cell 64
train.shape


## === cell 65
test.loc[((test['pickup_latitude']==0) & (test['pickup_longitude']==0))&((test['dropoff_latitude']!=0) & (test['dropoff_longitude']!=0))]


## === cell 66
train.loc[((train['pickup_latitude']!=0) & (train['pickup_longitude']!=0))&((train['dropoff_latitude']==0) & (train['dropoff_longitude']==0)) & (train['fare_amount']==0)]


## === cell 67
train = train.drop(train.loc[((train['pickup_latitude']!=0) & (train['pickup_longitude']!=0))&((train['dropoff_latitude']==0) & (train['dropoff_longitude']==0)) & (train['fare_amount']==0)].index, axis=0)


## === cell 68
train.shape


## === cell 69
test.loc[((test['pickup_latitude']!=0) & (test['pickup_longitude']!=0))&((test['dropoff_latitude']==0) & (test['dropoff_longitude']==0))]


## === cell 70
high_distance = train.loc[(train['H_Distance']>200)&(train['fare_amount']!=0)]


## === cell 71
high_distance


## === cell 72
high_distance.shape


## === cell 73
high_distance['H_Distance'] = high_distance.apply(
    lambda row: (row['fare_amount'] - 2.50)/1.56,
    axis=1
)


## === cell 74
high_distance


## === cell 75
train.update(high_distance)


## === cell 76
train.shape


## === cell 77
train[train['H_Distance']==0]


## === cell 78
train[(train['H_Distance']==0)&(train['fare_amount']==0)]


## === cell 79
train = train.drop(train[(train['H_Distance']==0)&(train['fare_amount']==0)].index, axis = 0)


## === cell 80
train[(train['H_Distance']==0)].shape


## === cell 81
rush_hour = train.loc[(((train['Hour']>=6)&(train['Hour']<=20)) & ((train['Day of Week']>=1) & (train['Day of Week']<=5)) & (train['H_Distance']==0) & (train['fare_amount'] < 2.5))]
rush_hour


## === cell 82
train=train.drop(rush_hour.index, axis=0)


## === cell 83
train.shape


## === cell 84
non_rush_hour = train.loc[(((train['Hour']<6)|(train['Hour']>20)) & ((train['Day of Week']>=1)&(train['Day of Week']<=5)) & (train['H_Distance']==0) & (train['fare_amount'] < 3.0))]
non_rush_hour


## === cell 85
weekends = train.loc[((train['Day of Week']==0) | (train['Day of Week']==6)) & (train['H_Distance']==0) & (train['fare_amount'] < 3.0)]
weekends


## === cell 86
train.loc[(train['H_Distance']!=0) & (train['fare_amount']==0)]


## === cell 87
scenario_3 = train.loc[(train['H_Distance']!=0) & (train['fare_amount']==0)]


## === cell 88
len(scenario_3)


## === cell 89
scenario_3.sort_values('H_Distance', ascending=False)


## === cell 90
scenario_3['fare_amount'] = scenario_3.apply(
    lambda row: ((row['H_Distance'] * 1.56) + 2.50), axis=1
)


## === cell 91
scenario_3['fare_amount']


## === cell 92
train.update(scenario_3)


## === cell 93
train.shape


## === cell 94
train.loc[(train['H_Distance']==0) & (train['fare_amount']!=0)]


## === cell 95
scenario_4 = train.loc[(train['H_Distance']==0) & (train['fare_amount']!=0)]


## === cell 96
len(scenario_4)


## === cell 97
scenario_4.loc[(scenario_4['fare_amount']<=3.0)&(scenario_4['H_Distance']==0)]


## === cell 98
scenario_4.loc[(scenario_4['fare_amount']>3.0)&(scenario_4['H_Distance']==0)]


## === cell 99
scenario_4_sub = scenario_4.loc[(scenario_4['fare_amount']>3.0)&(scenario_4['H_Distance']==0)]


## === cell 100
len(scenario_4_sub)


## === cell 101
scenario_4_sub['H_Distance'] = scenario_4_sub.apply(
lambda row: ((row['fare_amount']-2.50)/1.56), axis=1
)


## === cell 102
train.update(scenario_4_sub)


## === cell 103
train.shape


## === cell 104
train.columns


## === cell 105
test.columns


## === cell 106
train = train.drop(['key','pickup_datetime'], axis = 1)
test = test.drop(['key','pickup_datetime'], axis = 1)


## === cell 107
train.columns


## === cell 108
test.columns


## === cell 109
x_train = train.iloc[:,train.columns!='fare_amount']
y_train = train['fare_amount'].values
x_test = test


## === cell 110
x_train.shape


## === cell 111
x_train.columns


## === cell 112
y_train.shape


## === cell 113
x_test.shape


## === cell 114
x_test.columns


## === cell 115
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
rf_predict = rf.predict(x_test)


## --- ERROR in cell 115, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2439309602.py in <cell line: 0>()
      1 from sklearn.ensemble import RandomForestRegressor
      2 rf = RandomForestRegressor()
----> 3 rf.fit(x_train, y_train)
      4 rf_predict = rf.predict(x_test)
      5 #print(rf_predict)

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in fit(self, X, y, sample_weight)
    343         if issparse(y):
    344             raise ValueError("sparse multilabel-indicator for y is not supported.")
--> 345         X, y = self._validate_data(
    346             X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
    347         )

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
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Input X contains NaN.
RandomForestRegressor does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values

## === cell 116
submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = rf_predict
submission.to_csv('submission_1.csv', index=False)
submission.head(20)


## --- ERROR in cell 116, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3262662821.py in <cell line: 0>()
      1 submission = pd.read_csv('../input/sample_submission.csv')
----> 2 submission['fare_amount'] = rf_predict
      3 submission.to_csv('submission_1.csv', index=False)
      4 submission.head(20)

NameError: name 'rf_predict' is not defined

## === cell 117
import lightgbm as lgbm


## === cell 118
params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'nthread': -1,
        'verbose': 0,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': -1,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.6,
        'reg_aplha': 1,
        'reg_lambda': 0.001,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1     
    }


## === cell 119
pred_test_y = np.zeros(x_test.shape[0])
pred_test_y.shape


## === cell 120
train_set = lgbm.Dataset(x_train, y_train, silent=True)
train_set


## --- ERROR in cell 120, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4203970104.py in <cell line: 0>()
----> 1 train_set = lgbm.Dataset(x_train, y_train, silent=True)
      2 train_set

TypeError: Dataset.__init__() got an unexpected keyword argument 'silent'

## === cell 121
model = lgbm.train(params, train_set = train_set, num_boost_round=300)


## --- ERROR in cell 121, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1324757783.py in <cell line: 0>()
----> 1 model = lgbm.train(params, train_set = train_set, num_boost_round=300)

NameError: name 'train_set' is not defined

## === cell 122
print(model)


## --- ERROR in cell 122, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2971902120.py in <cell line: 0>()
----> 1 print(model)

NameError: name 'model' is not defined

## === cell 123
pred_test_y = model.predict(x_test, num_iteration = model.best_iteration)


## --- ERROR in cell 123, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1409266571.py in <cell line: 0>()
----> 1 pred_test_y = model.predict(x_test, num_iteration = model.best_iteration)

NameError: name 'model' is not defined

## === cell 124
print(pred_test_y)


## === cell 125
submission['fare_amount'] = pred_test_y
submission.to_csv('submission_LGB.csv', index=False)
submission.head(20)


## === cell 126
import xgboost as xgb 


## === cell 127
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)


## === cell 128
dtrain


## === cell 129
params = {'max_depth':7,
          'eta':1,
          'silent':1,
          'objective':'reg:linear',
          'eval_metric':'rmse',
          'learning_rate':0.05
         }
num_rounds = 50


## === cell 130
xb = xgb.train(params, dtrain, num_rounds)


## === cell 131
y_pred_xgb = xb.predict(dtest)
print(y_pred_xgb)


## === cell 132
submission['fare_amount'] = y_pred_xgb
submission.to_csv('submission_XGB.csv', index=False)
submission.head(20)
