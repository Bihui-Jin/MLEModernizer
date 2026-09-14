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

5.68866

# 6. Current score

816.49706

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



## === cell 1
data =  pd.read_csv('../input/train.csv', nrows = 15_000_000)


## === cell 2
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(data)


## === cell 3
from datetime import datetime
data['datetime_object'] = [datetime.strptime(date,'%Y-%m-%d %H:%M:%S %Z') for date in data['pickup_datetime']]


## === cell 4
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...

data['distance_miles'] = distance(data.pickup_latitude, data.pickup_longitude, \
                                      data.dropoff_latitude, data.dropoff_longitude)

data.distance_miles.describe()
data.groupby('passenger_count')['distance_miles', 'fare_amount'].mean()
print("Average $USD/Mile : {:0.2f}".format(data.fare_amount.sum()/data.distance_miles.sum()))
data['fare_per_mile'] = data.fare_amount / data.distance_miles
data['inv_distance_miles'] = 1/data.distance_miles
data['hour'] = [date.hour for date in data['datetime_object']]
data['year'] = [date.year for date in data['datetime_object']]


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2308811224.py in <cell line: 0>()
     12 #plt.title('Histogram ride distances in miles')
     13 data.distance_miles.describe()
---> 14 data.groupby('passenger_count')['distance_miles', 'fare_amount'].mean()
     15 print("Average $USD/Mile : {:0.2f}".format(data.fare_amount.sum()/data.distance_miles.sum()))
     16 data['fare_per_mile'] = data.fare_amount / data.distance_miles

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in __getitem__(self, key)
   1945             # if len == 1, then it becomes a SeriesGroupBy and this is actually
   1946             # valid syntax, so don't raise
-> 1947             raise ValueError(
   1948                 "Cannot subset columns with a tuple with more than one element. "
   1949                 "Use a list instead."

ValueError: Cannot subset columns with a tuple with more than one element. Use a list instead.

## === cell 5
import seaborn as sns
import matplotlib.pyplot as plt
corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))

k = 14 #number of variables for heatmap
cols = corrmat.nlargest(k, 'fare_amount')['fare_amount'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3460213759.py in <cell line: 0>()
      1 import seaborn as sns
      2 import matplotlib.pyplot as plt
----> 3 corrmat = data.corr()
      4 f, ax = plt.subplots(figsize=(12, 9))
      5 

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

## === cell 6
print('Old size: %d' % len(data))
data = data.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(data))


## === cell 7
print('Old size: %d' % len(data))
data = data[(data.abs_diff_longitude < 3.0) & (data.abs_diff_latitude < 3.0)]
data = data[data.fare_amount>0]
data = data[data.passenger_count <=9]
data = data[(data.distance_miles>0.0)]
print('New size: %d' % len(data))


## === cell 8
print(data['datetime_object'][0])

plot = data.iloc[:1000].plot.scatter('hour', 'fare_amount')


## --- ERROR in cell 8, traceback:
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

KeyError: 'hour'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1472715781.py in <cell line: 0>()
      1 print(data['datetime_object'][0])
      2 
----> 3 plot = data.iloc[:1000].plot.scatter('hour', 'fare_amount')

/usr/local/lib/python3.11/dist-packages/pandas/plotting/_core.py in scatter(self, x, y, s, c, **kwargs)
   1746             ...                       colormap='viridis')
   1747         """
-> 1748         return self(kind="scatter", x=x, y=y, s=s, c=c, **kwargs)
   1749 
   1750     def hexbin(

/usr/local/lib/python3.11/dist-packages/pandas/plotting/_core.py in __call__(self, *args, **kwargs)
    973         if kind in self._dataframe_kinds:
    974             if isinstance(data, ABCDataFrame):
--> 975                 return plot_backend.plot(data, x=x, y=y, kind=kind, **kwargs)
    976             else:
    977                 raise ValueError(f"plot kind {kind} can only be used for data frames")

/usr/local/lib/python3.11/dist-packages/pandas/plotting/_matplotlib/__init__.py in plot(data, kind, **kwargs)
     69             kwargs["ax"] = getattr(ax, "left_ax", ax)
     70     plot_obj = PLOT_CLASSES[kind](data, **kwargs)
---> 71     plot_obj.generate()
     72     plot_obj.draw()
     73     return plot_obj.result

/usr/local/lib/python3.11/dist-packages/pandas/plotting/_matplotlib/core.py in generate(self)
    499         self._compute_plot_data()
    500         fig = self.fig
--> 501         self._make_plot(fig)
    502         self._add_table()
    503         self._make_legend()

/usr/local/lib/python3.11/dist-packages/pandas/plotting/_matplotlib/core.py in _make_plot(self, fig)
   1350             label = None
   1351         scatter = ax.scatter(
-> 1352             data[x].values,
   1353             data[y].values,
   1354             c=c_values,

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

KeyError: 'hour'

## === cell 9
from sklearn.model_selection import train_test_split
y = data.fare_amount
X = data.drop('fare_amount', axis=1)
train_df, val_df, train_y, val_y = train_test_split(X, y,test_size=0.2)
train_df.dtypes


## === cell 10
def get_input_matrix(df):
    return np.column_stack((df.abs_diff_longitude, df.abs_diff_latitude, np.ones(len(df))))

train_X = get_input_matrix(train_df)

print(train_X.shape)
print(train_y.shape)


## === cell 11
(w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond = None)
print(w)


## === cell 12
w_OLS = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_X.T, train_X)), train_X.T), train_y)
print(w_OLS)


## === cell 13
test_df = pd.read_csv('../input/test.csv')

test_df.dtypes


## === cell 14
add_travel_vector_features(test_df)
test_X = get_input_matrix(test_df)
add_travel_vector_features(val_df)
val_X = get_input_matrix(val_df)
test_y_predictions = np.matmul(test_X, w).round(decimals = 2)
val_y_predictions = np.matmul(val_X, w).round(decimals = 2)

from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(val_y, val_y_predictions)))
submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount': test_y_predictions},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)

print(os.listdir('.'))
