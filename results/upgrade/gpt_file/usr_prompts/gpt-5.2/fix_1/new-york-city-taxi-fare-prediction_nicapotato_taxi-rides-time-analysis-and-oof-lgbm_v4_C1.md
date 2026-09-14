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

3.43884

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

Debug = False

NROWS = 10000000
if Debug is True: NROWS = 5000
train = pd.read_csv('../input/train.csv', nrows = NROWS, index_col = "key")
train = train.dropna()
test_df = pd.read_csv('../input/test.csv', index_col = "key")
testdex = test_df.index


## === cell 1
print("Percent of Training Set with Zero and Below Fair: ", round(((train.loc[train["fare_amount"] <= 0, "fare_amount"].shape[0]/train.shape[0]) * 100),5))
print("Percent of Training Set 200 and Above Fair: ", round((train.loc[train["fare_amount"] >= 200, "fare_amount"].shape[0]/train.shape[0]) * 100,5))
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] <= 200),:]
print("\nPercent of Training Set with Zero and Below Passenger Count: ", round((train.loc[train["passenger_count"] <= 0, "passenger_count"].shape[0]/train.shape[0]) * 100,5))
print("Percent of Training Set with Nine and Above Passenger Count: ", round((train.loc[train["passenger_count"] >= 9, "passenger_count"].shape[0]/train.shape[0]) * 100,5))
train = train.loc[(train["passenger_count"] > 0) & (train["passenger_count"] <= 9),:]


## === cell 2
def clean_df(df):
    return df[(df.fare_amount > 0) & 
            (df.pickup_longitude > -80) & (df.pickup_longitude < -70) &
            (df.pickup_latitude > 35) & (df.pickup_latitude < 45) &
            (df.dropoff_longitude > -80) & (df.dropoff_longitude < -70) &
            (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45)]


## === cell 3
def prepare_distance_features(df):
    df['longitude_distance'] = abs(df['pickup_longitude'] - df['dropoff_longitude'])
    df['latitude_distance'] = abs(df['pickup_latitude'] - df['dropoff_latitude'])

    df['distance_travelled'] = (df['longitude_distance'] ** 2 + df['latitude_distance'] ** 2) ** .5
    df['distance_travelled_sin'] = np.sin((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5)
    df['distance_travelled_cos'] = np.cos((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5)
    df['distance_travelled_sin_sqrd'] = np.sin((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5) ** 2
    df['distance_travelled_cos_sqrd'] = np.cos((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5) ** 2

    R = 6371e3 # Metres
    phi1 = np.radians(df['pickup_latitude'])
    phi2 = np.radians(df['dropoff_latitude'])
    phi_chg = np.radians(df['pickup_latitude'] - df['dropoff_latitude'])
    delta_chg = np.radians(df['pickup_longitude'] - df['dropoff_longitude'])
    a = np.sin(phi_chg / 2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_chg / 2)
    c = 2 * np.arctan2(a ** .5, (1-a) ** .5)
    d = R * c
    df['haversine'] = d

    y = np.sin(delta_chg * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    df['bearing'] = np.arctan2(y, x)

    return df

def prepare_time_features(df):
    df['pickup_datetime'] = df['pickup_datetime'].str.replace(" UTC", "")
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
    df['hour_of_day'] = df.pickup_datetime.dt.hour
    df['week'] = df.pickup_datetime.dt.week
    df['month'] = df.pickup_datetime.dt.month
    df['day_of_year'] = df.pickup_datetime.dt.dayofyear
    df['week_of_year'] = df.pickup_datetime.dt.weekofyear
    df["Weekday"] = df.pickup_datetime.dt.weekday
    df["Quarter"] = df.pickup_datetime.dt.quarter
    df["Day of Month"] = df.pickup_datetime.dt.day
    
    return df


## === cell 4
%%time
train = clean_df(train)

train = prepare_distance_features(train)
test_df = prepare_distance_features(test_df)

train = prepare_time_features(train)
test_df = prepare_time_features(test_df)

train["fare_to_dist_ratio"] = train["fare_amount"] / ( train["distance_travelled"]+0.0001)
train["fare_npassenger_to_dist_ratio"] = (train["fare_amount"] / train["passenger_count"]) /( train["distance_travelled"]+0.0001)


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<timed exec> in <module>

/tmp/ipykernel_11/2710755726.py in prepare_time_features(df)
     35     df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
     36     df['hour_of_day'] = df.pickup_datetime.dt.hour
---> 37     df['week'] = df.pickup_datetime.dt.week
     38     df['month'] = df.pickup_datetime.dt.month
     39     df['day_of_year'] = df.pickup_datetime.dt.dayofyear

AttributeError: 'DatetimeProperties' object has no attribute 'week'

## === cell 5
f, ax = plt.subplots(1,2,figsize = [10,5])
sns.countplot(train["passenger_count"], ax=ax[0])
sns.countplot(test_df["passenger_count"], ax=ax[1])
ax[0].set_title("Train Set - Passenger Count")
ax[1].set_title("Test Set - Passenger Count")
plt.show()


## === cell 6
f, ax = plt.subplots(figsize=[6,5])
sns.kdeplot(train["fare_amount"], ax=ax)
ax.set_title("Fare Distribution")
plt.show()


## === cell 7
def time_slicer(df, timeframes, value, color="purple"):
    """
    Function to count observation occurrence through different lenses of time.
    """
    f, ax = plt.subplots(len(timeframes), figsize = [12,10])
    for i,x in enumerate(timeframes):
        df.loc[:,[x,value]].groupby([x]).mean().plot(ax=ax[i],color=color)
        ax[i].set_ylabel(value.replace("_", " ").title())
        ax[i].set_title("{} by {}".format(value.replace("_", " ").title(), x.replace("_", " ").title()))
        ax[i].set_xlabel("")
    ax[len(timeframes)-1].set_xlabel("Time Frame")
    plt.tight_layout(pad=0)


## === cell 8
time_slicer(df=train, timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"], value = "fare_amount", color="blue")


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1570644381.py in <cell line: 0>()
----> 1 time_slicer(df=train, timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"], value = "fare_amount", color="blue")

/tmp/ipykernel_11/2575660034.py in time_slicer(df, timeframes, value, color)
      5     f, ax = plt.subplots(len(timeframes), figsize = [12,10])
      6     for i,x in enumerate(timeframes):
----> 7         df.loc[:,[x,value]].groupby([x]).mean().plot(ax=ax[i],color=color)
      8         ax[i].set_ylabel(value.replace("_", " ").title())
      9         ax[i].set_title("{} by {}".format(value.replace("_", " ").title(), x.replace("_", " ").title()))

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1182             if self._is_scalar_access(key):
   1183                 return self.obj._get_value(*key, takeable=self._takeable)
-> 1184             return self._getitem_tuple(key)
   1185         else:
   1186             # we by definition only have the 0th axis

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
   1375             return self._multi_take(tup)
   1376 
-> 1377         return self._getitem_tuple_same_dim(tup)
   1378 
   1379     def _get_label(self, label, axis: AxisInt):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple_same_dim(self, tup)
   1018                 continue
   1019 
-> 1020             retval = getattr(retval, self.name)._getitem_axis(key, axis=i)
   1021             # We should never have retval.ndim < self.ndim, as that should
   1022             #  be handled by the _getitem_lowerdim call above.

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

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

KeyError: "['day_of_year'] not in index"

## === cell 9
time_slicer(df=train, timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"], value = "distance_travelled", color = "green")


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1031008983.py in <cell line: 0>()
----> 1 time_slicer(df=train, timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"], value = "distance_travelled", color = "green")

/tmp/ipykernel_11/2575660034.py in time_slicer(df, timeframes, value, color)
      5     f, ax = plt.subplots(len(timeframes), figsize = [12,10])
      6     for i,x in enumerate(timeframes):
----> 7         df.loc[:,[x,value]].groupby([x]).mean().plot(ax=ax[i],color=color)
      8         ax[i].set_ylabel(value.replace("_", " ").title())
      9         ax[i].set_title("{} by {}".format(value.replace("_", " ").title(), x.replace("_", " ").title()))

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1182             if self._is_scalar_access(key):
   1183                 return self.obj._get_value(*key, takeable=self._takeable)
-> 1184             return self._getitem_tuple(key)
   1185         else:
   1186             # we by definition only have the 0th axis

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
   1375             return self._multi_take(tup)
   1376 
-> 1377         return self._getitem_tuple_same_dim(tup)
   1378 
   1379     def _get_label(self, label, axis: AxisInt):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple_same_dim(self, tup)
   1018                 continue
   1019 
-> 1020             retval = getattr(retval, self.name)._getitem_axis(key, axis=i)
   1021             # We should never have retval.ndim < self.ndim, as that should
   1022             #  be handled by the _getitem_lowerdim call above.

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

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

KeyError: "['day_of_year'] not in index"

## === cell 10
time_slicer(df=train, timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"], value = "fare_to_dist_ratio", color = "red")


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2868667556.py in <cell line: 0>()
----> 1 time_slicer(df=train, timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"], value = "fare_to_dist_ratio", color = "red")

/tmp/ipykernel_11/2575660034.py in time_slicer(df, timeframes, value, color)
      5     f, ax = plt.subplots(len(timeframes), figsize = [12,10])
      6     for i,x in enumerate(timeframes):
----> 7         df.loc[:,[x,value]].groupby([x]).mean().plot(ax=ax[i],color=color)
      8         ax[i].set_ylabel(value.replace("_", " ").title())
      9         ax[i].set_title("{} by {}".format(value.replace("_", " ").title(), x.replace("_", " ").title()))

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1182             if self._is_scalar_access(key):
   1183                 return self.obj._get_value(*key, takeable=self._takeable)
-> 1184             return self._getitem_tuple(key)
   1185         else:
   1186             # we by definition only have the 0th axis

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
   1375             return self._multi_take(tup)
   1376 
-> 1377         return self._getitem_tuple_same_dim(tup)
   1378 
   1379     def _get_label(self, label, axis: AxisInt):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple_same_dim(self, tup)
   1018                 continue
   1019 
-> 1020             retval = getattr(retval, self.name)._getitem_axis(key, axis=i)
   1021             # We should never have retval.ndim < self.ndim, as that should
   1022             #  be handled by the _getitem_lowerdim call above.

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['day_of_year', 'fare_to_dist_ratio'], dtype='object')] are in the [columns]"

## === cell 11
time_slicer(df=train, timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"], value = "fare_npassenger_to_dist_ratio", color = "orange")


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2215984125.py in <cell line: 0>()
----> 1 time_slicer(df=train, timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"], value = "fare_npassenger_to_dist_ratio", color = "orange")

/tmp/ipykernel_11/2575660034.py in time_slicer(df, timeframes, value, color)
      5     f, ax = plt.subplots(len(timeframes), figsize = [12,10])
      6     for i,x in enumerate(timeframes):
----> 7         df.loc[:,[x,value]].groupby([x]).mean().plot(ax=ax[i],color=color)
      8         ax[i].set_ylabel(value.replace("_", " ").title())
      9         ax[i].set_title("{} by {}".format(value.replace("_", " ").title(), x.replace("_", " ").title()))

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1182             if self._is_scalar_access(key):
   1183                 return self.obj._get_value(*key, takeable=self._takeable)
-> 1184             return self._getitem_tuple(key)
   1185         else:
   1186             # we by definition only have the 0th axis

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
   1375             return self._multi_take(tup)
   1376 
-> 1377         return self._getitem_tuple_same_dim(tup)
   1378 
   1379     def _get_label(self, label, axis: AxisInt):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple_same_dim(self, tup)
   1018                 continue
   1019 
-> 1020             retval = getattr(retval, self.name)._getitem_axis(key, axis=i)
   1021             # We should never have retval.ndim < self.ndim, as that should
   1022             #  be handled by the _getitem_lowerdim call above.

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['day_of_year', 'fare_npassenger_to_dist_ratio'], dtype='object')] are in the [columns]"

## === cell 12
y = train.fare_amount.copy()
test_df.drop("pickup_datetime", axis = 1, inplace=True)
train = train[test_df.columns]
print("Does Train feature equal test feature?: ", all(train.columns == test_df.columns))

X_train, X_val, y_train, y_val = train_test_split(train, y, test_size = .1, random_state = 23)

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_val, label=y_val)


## === cell 13
print("Light Gradient Boosting Regressor: ")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
                }


## === cell 14
modelstart = time.time()
lgb_reg = lgb.train(
    lgbm_params,
    dtrain,
    num_boost_round=500,
    valid_sets=[dtrain, dvalid],
    valid_names=['train','valid'],
    early_stopping_rounds=50,
    verbose_eval=100
)
print("Model Evaluation Stage")
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_val, lgb_reg.predict(X_val))))


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/562867445.py in <cell line: 0>()
      1 # Go Go Go
      2 modelstart = time.time()
----> 3 lgb_reg = lgb.train(
      4     lgbm_params,
      5     dtrain,

TypeError: train() got an unexpected keyword argument 'early_stopping_rounds'

## === cell 15
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(lgb_reg, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")
plt.savefig('feature_import.png')


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/564116850.py in <cell line: 0>()
      1 # Feature Importance Plot
      2 f, ax = plt.subplots(figsize=[7,10])
----> 3 lgb.plot_importance(lgb_reg, max_num_features=50, ax=ax)
      4 plt.title("Light GBM Feature Importance")
      5 plt.savefig('feature_import.png')

NameError: name 'lgb_reg' is not defined

## === cell 16
lgpred = lgb_reg.predict(test_df)
lgsub = pd.DataFrame(lgpred,columns=["fare_amount"],index=testdex)
lgsub.to_csv("lgsub.csv",index=True,header=True)
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))
lgsub.head()


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1637515440.py in <cell line: 0>()
----> 1 lgpred = lgb_reg.predict(test_df)
      2 lgsub = pd.DataFrame(lgpred,columns=["fare_amount"],index=testdex)
      3 lgsub.to_csv("lgsub.csv",index=True,header=True)
      4 print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
      5 print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

NameError: name 'lgb_reg' is not defined
