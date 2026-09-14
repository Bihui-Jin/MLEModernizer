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

3.40466

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
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
import os
print(os.listdir("../input"))



## === cell 1
train_df=pd.read_csv("../input/train.csv",nrows=1000000)
test_df=pd.read_csv("../input/test.csv")


## === cell 2
train_df.shape


## === cell 3
train_df.columns


## === cell 4
train_df.head()


## === cell 5
train_df.info()


## === cell 6
train_df.describe()


## === cell 7
test_df.info()


## === cell 8
test_df.describe()


## === cell 9
train_df.isnull().sum()


## === cell 10
train_df= train_df.drop(train_df[train_df.isnull().any(1)].index, axis = 0)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2597430988.py in <cell line: 0>()
      1 #Drop the missing values record
----> 2 train_df= train_df.drop(train_df[train_df.isnull().any(1)].index, axis = 0)

TypeError: DataFrame.any() takes 1 positional argument but 2 were given

## === cell 11
train_df.info()


## === cell 12
Counter(train_df['fare_amount']<0)


## === cell 13
train_df= train_df.drop(train_df[train_df['fare_amount']<0].index, axis = 0)
train_df.shape


## === cell 14
train_df.describe()


## === cell 15
Counter(train_df['passenger_count']>6)


## === cell 16
train_df= train_df.drop(train_df[train_df['passenger_count']>6].index, axis = 0)
train_df.shape


## === cell 17
Counter(train_df['pickup_latitude']<-90)


## === cell 18
Counter(train_df['pickup_latitude']>90)


## === cell 19
train_df = train_df.drop(((train_df[train_df['pickup_latitude']<-90])|(train_df[train_df['pickup_latitude']>90])).index, axis=0)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in na_logical_op(x, y, op)
    361         #  (xint or xbool) and (yint or bool)
--> 362         result = op(x, y)
    363     except TypeError:

TypeError: unsupported operand type(s) for |: 'float' and 'float'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1193954877.py in <cell line: 0>()
      1 #We need to drop these outliers
----> 2 train_df = train_df.drop(((train_df[train_df['pickup_latitude']<-90])|(train_df[train_df['pickup_latitude']>90])).index, axis=0)

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

## === cell 20
train_df.shape


## === cell 21
Counter(train_df['pickup_longitude']<-180)


## === cell 22
Counter(train_df['pickup_longitude']>180)


## === cell 23
train_df = train_df.drop((train_df[train_df['pickup_longitude']<-180]).index, axis=0)


## === cell 24
train_df.shape


## === cell 25
train_df.dtypes


## === cell 26
train_df.head(3)


## === cell 27
train_df['key']=pd.to_datetime(train_df['key'])
train_df['pickup_datetime']=pd.to_datetime(train_df['pickup_datetime'])


## === cell 28
train_df.dtypes


## === cell 29
test_df.dtypes


## === cell 30
train_df.head()


## === cell 31
test_df['key']=pd.to_datetime(test_df['key'])
test_df['pickup_datetime']=pd.to_datetime(test_df['pickup_datetime'])


## === cell 32
test_df.dtypes


## === cell 33
test_df.head()


## === cell 34
train_df.head()


## === cell 35
data=[train_df,test_df]
for i in data:
    i['date']=i['pickup_datetime'].dt.day
    i['month']=i['pickup_datetime'].dt.month
    i['day_of_week']=i['pickup_datetime'].dt.dayofweek
    i['hour']=i['pickup_datetime'].dt.hour
    i['year']=i['pickup_datetime'].dt.year
    


## === cell 36
train_df.head()


## === cell 37
train_df.describe()


## === cell 38
def sphere_distance(lat1,long1,lat2,long2):
    data=[train_df,test_df]
    for i in data:
        R=6367
        phi1 = np.radians(i[lat1])
        phi2 = np.radians(i[lat2])
        delta_phi = np.radians(i[lat2]-i[lat1])
        delta_lambda = np.radians(i[long2]-i[long1])
        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d = (R * c)
        i['S_Distance'] = d
    return d #in Kilometer


## === cell 39
sphere_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')


## === cell 40
train_df.head()


## === cell 41
plt.hist(train_df['passenger_count'], bins=15)
plt.xlabel('No. of Passengers')
plt.ylabel('Frequency')


## === cell 42
plt.scatter(x=train_df['passenger_count'], y=train_df['fare_amount'],s=2.0)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')


## === cell 43
plt.scatter(x=train_df['date'], y=train_df['fare_amount'])
plt.xlabel('Date')
plt.ylabel('Fare')


## === cell 44
plt.hist(train_df['hour'],bins=50)
plt.xlabel('Date')
plt.ylabel('Fare')


## === cell 45
plt.hist(train_df['day_of_week'],bins=20)
plt.xlabel('Date')
plt.ylabel('Fare')


## === cell 46
plt.scatter(x=train_df['day_of_week'], y=train_df['fare_amount'])
plt.xlabel('Date of week')
plt.ylabel('Fare')


## === cell 47
len(train_df)


## === cell 48
train_df.sort_values(['S_Distance','fare_amount'], ascending=False)


## === cell 49
dis_0 = train_df.loc[(train_df['S_Distance'] == 0), ['S_Distance']]
dis_1 = train_df.loc[(train_df['S_Distance'] > 0) & (train_df['S_Distance'] <= 10), ['S_Distance']]
dis_2 = train_df.loc[(train_df['S_Distance'] > 10) & (train_df['S_Distance'] <= 50), ['S_Distance']]
dis_3 = train_df.loc[(train_df['S_Distance'] > 50) & (train_df['S_Distance'] <= 100), ['S_Distance']]
dis_4 = train_df.loc[(train_df['S_Distance'] > 100) & (train_df['S_Distance'] <= 200), ['S_Distance']]
dis_5 = train_df.loc[(train_df['S_Distance'] > 200) & (train_df['S_Distance'] <= 300), ['S_Distance']]
dis_6 = train_df.loc[(train_df['S_Distance'] > 300) & (train_df['S_Distance'] <= 500), ['S_Distance']]
dis_7 = train_df.loc[(train_df['S_Distance'] > 500), ['S_Distance']]
dis_0['bins']='0'
dis_1['bins']='0-10'
dis_2['bins']='11-50'
dis_3['bins']='51-100'
dis_4['bins']='101-200'
dis_5['bins']='201-300'
dis_6['bins']='301-500'
dis_7['bins']='>500'
dis_bin=pd.concat([dis_0,dis_1,dis_2,dis_3,dis_4,dis_5,dis_6,dis_7])
dis_bin


## === cell 50
x=Counter(dis_bin['bins'])
x


## === cell 51
train_df.loc[((train_df['pickup_latitude']==0) & (train_df['pickup_longitude']==0))&((train_df['dropoff_latitude']!=0) & (train_df['dropoff_longitude']!=0)) & (train_df['fare_amount']==0)]


## === cell 52
train_df.loc[((train_df['pickup_latitude']==0) & (train_df['pickup_longitude']==0))&((train_df['dropoff_latitude']!=0) & (train_df['dropoff_longitude']!=0)) & (train_df['fare_amount']==0)]


## === cell 53
train_df = train_df.drop(train_df.loc[((train_df['pickup_latitude']==0) & (train_df['pickup_longitude']==0))&((train_df['dropoff_latitude']!=0) & (train_df['dropoff_longitude']!=0)) & (train_df['fare_amount']==0)].index, axis=0)


## === cell 54
train_df.shape


## === cell 55
train_df = train_df.drop(train_df.loc[((train_df['pickup_latitude']==0) & (train_df['pickup_longitude']==0))&((train_df['dropoff_latitude']!=0) & (train_df['dropoff_longitude']!=0)) & (train_df['fare_amount']==0)].index, axis=0)


## === cell 56
train_df.shape


## === cell 57
high_distance = train_df.loc[(train_df['S_Distance']>200)&(train_df['fare_amount']!=0)]


## === cell 58
high_distance


## === cell 59
high_distance.shape


## === cell 60
high_distance['S_Distance'] = high_distance.apply(
    lambda row: (row['fare_amount'] - 2.50)/1.56,
    axis=1
)


## === cell 61
high_distance


## === cell 62
train_df.update(high_distance)


## === cell 63
train_df


## === cell 64
train_df[train_df['S_Distance']==0]


## === cell 65
train_df[(train_df['S_Distance']==0)&(train_df['fare_amount']==0)]


## === cell 66
train_df = train_df.drop(train_df[(train_df['S_Distance']==0)&(train_df['fare_amount']==0)].index, axis = 0)


## === cell 67
rush_hour = train_df.loc[(((train_df['hour']>=6)&(train_df['hour']<=20)) & ((train_df['day_of_week']>=1) & (train_df['day_of_week']<=5)) & (train_df['S_Distance']==0) & (train_df['fare_amount'] < 2.5))]
rush_hour


## === cell 68
train_df=train_df.drop(rush_hour.index,axis=0)


## === cell 69
train_df.shape


## === cell 70
non_rush_hour = train_df.loc[(((train_df['hour']<6)|(train_df['hour']>20)) & ((train_df['day_of_week']>=1)&(train_df['day_of_week']<=5)) & (train_df['S_Distance']==0) & (train_df['fare_amount'] < 3.0))]


## === cell 71
non_rush_hour


## === cell 72
non_rush_hour = train_df.loc[(((train_df['hour']<6)|(train_df['hour']>20)) & ((train_df['day_of_week']>=1)&(train_df['day_of_week']<=5)) & (train_df['S_Distance']==0) & (train_df['fare_amount'] < 3.0))]
non_rush_hour


## === cell 73
train_df.loc[(train_df['S_Distance']!=0) & (train_df['fare_amount']==0)]


## === cell 74
scenario_3 = train_df.loc[(train_df['S_Distance']!=0) & (train_df['fare_amount']==0)]
scenario_3


## === cell 75
scenario_3 = train_df.loc[(train_df['S_Distance']!=0) & (train_df['fare_amount']==0)]


## === cell 76
scenario_3['fare_amount'] = scenario_3.apply(
    lambda row: ((row['S_Distance'] * 1.56) + 2.50), axis=1
)


## === cell 77
scenario_3['fare_amount']


## === cell 78
train_df.loc[(train_df['S_Distance']==0) & (train_df['fare_amount']!=0)]


## === cell 79
scenario_4 = train_df.loc[(train_df['S_Distance']==0) & (train_df['fare_amount']!=0)]


## === cell 80
scenario_4


## === cell 81
len(scenario_3)


## === cell 82
len(scenario_4)


## === cell 83
scenario_4.loc[(scenario_4['fare_amount']<=3.0)&(scenario_4['S_Distance']==0)]


## === cell 84
scenario_4.loc[(scenario_4['fare_amount']>3.0)&(scenario_4['S_Distance']==0)]


## === cell 85
scenario_4_sub = scenario_4.loc[(scenario_4['fare_amount']>3.0)&(scenario_4['S_Distance']==0)]


## === cell 86
len(scenario_4_sub)


## === cell 87
scenario_4_sub['S_Distance'] = scenario_4_sub.apply(
lambda row: ((row['fare_amount']-2.50)/1.56), axis=1
)


## === cell 88
train_df.update(scenario_4_sub)


## === cell 89
len(train_df)


## === cell 90
train_df.columns


## === cell 91
test_df.columns


## === cell 92
train_df = train_df.drop(['key','pickup_datetime'], axis = 1)
test_df = test_df.drop(['key','pickup_datetime'], axis = 1)


## === cell 93
train_df.columns


## === cell 94
test_df.columns


## === cell 95
x_train = train_df.iloc[:,train_df.columns!='fare_amount']
y_train = train_df['fare_amount'].values
x_test = test_df


## === cell 96
x_train.shape


## === cell 97
y_train.shape


## === cell 98
rg=RandomForestRegressor()
rg.fit(x_train,y_train)
y_predict=rg.predict(x_test)
y_predict


## --- ERROR in cell 98, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1906517522.py in <cell line: 0>()
      1 rg=RandomForestRegressor()
----> 2 rg.fit(x_train,y_train)
      3 y_predict=rg.predict(x_test)
      4 y_predict

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

## === cell 99
submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = y_predict
submission.to_csv('submission_1.csv', index=False)
submission.head(10)


## --- ERROR in cell 99, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/403163980.py in <cell line: 0>()
      1 submission = pd.read_csv('../input/sample_submission.csv')
----> 2 submission['fare_amount'] = y_predict
      3 submission.to_csv('submission_1.csv', index=False)
      4 submission.head(10)

NameError: name 'y_predict' is not defined
