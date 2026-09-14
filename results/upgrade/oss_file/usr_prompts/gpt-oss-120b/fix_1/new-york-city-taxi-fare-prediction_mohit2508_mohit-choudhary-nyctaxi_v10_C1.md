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

No external packages required in the script and installed.

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

5.83413681510617

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
import math
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import feather as fe
import os
plt.style.use('seaborn-whitegrid')

df = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv',nrows=5000000,low_memory=True)
df.to_feather('nycTaxi.feather')

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2363300586.py in <cell line: 0>()
      5 import matplotlib.pyplot as plt
      6 import seaborn as sns
----> 7 import feather as fe
      8 import os
      9 plt.style.use('seaborn-whitegrid')

ModuleNotFoundError: No module named 'feather'

## === cell 1
df = pd.read_feather('nycTaxi.feather')

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/256336392.py in <cell line: 0>()
----> 1 df = pd.read_feather('nycTaxi.feather')

/usr/local/lib/python3.11/dist-packages/pandas/io/feather_format.py in read_feather(path, columns, use_threads, storage_options, dtype_backend)
    118     check_dtype_backend(dtype_backend)
    119 
--> 120     with get_handle(
    121         path, "rb", storage_options=storage_options, is_text=False
    122     ) as handles:

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    880         else:
    881             # Binary mode
--> 882             handle = open(handle, ioargs.mode)
    883         handles.append(handle)
    884 

FileNotFoundError: [Errno 2] No such file or directory: 'nycTaxi.feather'

## === cell 3
df=df[df.passenger_count>0]

df=df[df.dropoff_latitude!=0 ]
df=df[df.pickup_longitude!=0 ]
df=df[df.pickup_latitude!=0 ]
df=df[df.dropoff_longitude!=0]

df= df[df.fare_amount>2.5]
df= df[df.fare_amount<100]

missing_values = df.isnull().sum()

df=df.dropna()


df['year'], df['hour'] = df['pickup_datetime'].str.split(' ', 1).str
df['hour'] = df.hour.str[0:2]+df.hour.str[3:5]
df['year'] = df.year.str[:4]
df=df.dropna()


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1504502438.py in <cell line: 0>()
      1 #Passenger Count >0
----> 2 df=df[df.passenger_count>0]
      3 
      4 # dropping rows with any geo co-ordinate = 0
      5 df=df[df.dropoff_latitude!=0 ]

NameError: name 'df' is not defined

## === cell 4
df[['year','hour']] = df[['year','hour']].apply(pd.to_numeric)
print(df.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2043909740.py in <cell line: 0>()
----> 1 df[['year','hour']] = df[['year','hour']].apply(pd.to_numeric)
      2 print(df.head())

NameError: name 'df' is not defined

## === cell 6
def select_within_newYork(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
           (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
           (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
           (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])
NYC = (-74.5, -72.8, 40.5, 41.8)

df = df[select_within_newYork(df, NYC)]

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3749176466.py in <cell line: 0>()
      6 NYC = (-74.5, -72.8, 40.5, 41.8)
      7 
----> 8 df = df[select_within_newYork(df, NYC)]

NameError: name 'df' is not defined

## === cell 9

def haversine_np(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    miles = 6367 * c *0.62137
    return miles

df['distance'] = \
    haversine_np(df.pickup_longitude, df.pickup_latitude,df.dropoff_longitude,
                 df.dropoff_latitude)

print(df.head())



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1392330420.py in <cell line: 0>()
     18 
     19 df['distance'] = \
---> 20     haversine_np(df.pickup_longitude, df.pickup_latitude,df.dropoff_longitude,
     21                  df.dropoff_latitude)
     22 

NameError: name 'df' is not defined

## === cell 11
    print('Co-relation b/w Fare and Distance')
    print(st.pearsonr(df.distance, df.fare_amount))
    print(df['distance'].corr(df['fare_amount'], method='pearson')) 
    df=df[df.distance<=30]

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3592890246.py in <cell line: 0>()
      1 print('Co-relation b/w Fare and Distance')
----> 2 print(st.pearsonr(df.distance, df.fare_amount))
      3 print(df['distance'].corr(df['fare_amount'], method='pearson'))
      4 df=df[df.distance<=30]

NameError: name 'df' is not defined

## === cell 12
    fig, axs = plt.subplots(1, 2, figsize=(16,6))
    con = (df.distance < 30)  & (df.distance>0.5) & (df.fare_amount>0) & (df.fare_amount <200) 
    axs[0].scatter(df[con].distance, df[con].fare_amount, alpha=0.3)
    axs[0].set_xlabel('Distance')
    axs[0].set_ylabel('Fare')
    axs[0].set_title('Distance vs Fare')

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1261388843.py in <cell line: 0>()
      1 # Visualisation
      2 fig, axs = plt.subplots(1, 2, figsize=(16,6))
----> 3 con = (df.distance < 30)  & (df.distance>0.5) & (df.fare_amount>0) & (df.fare_amount <200)
      4 axs[0].scatter(df[con].distance, df[con].fare_amount, alpha=0.3)
      5 axs[0].set_xlabel('Distance')

NameError: name 'df' is not defined

## === cell 13
df['fare-bin'] = pd.cut(df['fare_amount'], bins = list(range(0, 50, 5))).astype(str)
df.loc[df['fare-bin'] == 'nan', 'fare-bin'] = '[45+]'
df.loc[df['fare-bin'] == '(5, 10]', 'fare-bin'] = '(05, 10]'
df.groupby('fare-bin')['distance'].mean().sort_index().plot.bar(color = 'g');
plt.title('Average Distance by Fare Amount');
plt.ylabel('Mean Distance');

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2832546078.py in <cell line: 0>()
----> 1 df['fare-bin'] = pd.cut(df['fare_amount'], bins = list(range(0, 50, 5))).astype(str)
      2 df.loc[df['fare-bin'] == 'nan', 'fare-bin'] = '[45+]'
      3 df.loc[df['fare-bin'] == '(5, 10]', 'fare-bin'] = '(05, 10]'
      4 df.groupby('fare-bin')['distance'].mean().sort_index().plot.bar(color = 'g');
      5 plt.title('Average Distance by Fare Amount');

NameError: name 'df' is not defined

## === cell 15
    print('corelation b/w Distance and Time of Day')
    print(st.pearsonr(df.distance, df.hour))


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2742307998.py in <cell line: 0>()
      1 print('corelation b/w Distance and Time of Day')
----> 2 print(st.pearsonr(df.distance, df.hour))
      3 #     print(df['hour'].corr(df['distance'], method='pearson'))

NameError: name 'df' is not defined

## === cell 16
    fig, axs = plt.subplots(1, 2, figsize=(16,6))
    con = (df.distance<30) & (df.fare_amount>0) 
    axs[0].scatter(df[con].hour, df[con].distance, alpha=0.2)
    axs[0].set_xlabel('Time Of Day (hours)')
    axs[0].set_ylabel('Distance')
    axs[0].set_title('Time of Day vs Distance')

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/485673499.py in <cell line: 0>()
      1 fig, axs = plt.subplots(1, 2, figsize=(16,6))
----> 2 con = (df.distance<30) & (df.fare_amount>0)
      3 axs[0].scatter(df[con].hour, df[con].distance, alpha=0.2)
      4 axs[0].set_xlabel('Time Of Day (hours)')
      5 axs[0].set_ylabel('Distance')

NameError: name 'df' is not defined

## === cell 18
    print('corelation b/w Fare and Time of Day')
    print(st.pearsonr(df.fare_amount, df.hour))
    print(df['fare_amount'].corr(df['hour'], method='pearson'))

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/302670694.py in <cell line: 0>()
      1 print('corelation b/w Fare and Time of Day')
----> 2 print(st.pearsonr(df.fare_amount, df.hour))
      3 print(df['fare_amount'].corr(df['hour'], method='pearson'))

NameError: name 'df' is not defined

## === cell 19
    fig, axs = plt.subplots(1, 2, figsize=(20,8))
    con =  (df.fare_amount>1) & (df.fare_amount <200) 
    axs[0].scatter(df[con].hour, df[con].fare_amount, alpha=0.5)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Fare')
    axs[0].set_title('Time of Day vs Fare')

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2320799943.py in <cell line: 0>()
      1 fig, axs = plt.subplots(1, 2, figsize=(20,8))
----> 2 con =  (df.fare_amount>1) & (df.fare_amount <200)
      3 axs[0].scatter(df[con].hour, df[con].fare_amount, alpha=0.5)
      4 axs[0].set_xlabel('Time')
      5 axs[0].set_ylabel('Fare')

NameError: name 'df' is not defined

## === cell 21
times_sq = (-73.985130,40.758896)

def plot_location_fare(loc, name, range=1.0):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
   
    idx = (haversine_np(df.pickup_longitude, df.pickup_latitude, loc[0], loc[1]) < range)
    df[idx].hour.hist(bins=100, ax=axs[0])
    axs[0].set_xlabel('Time')
    axs[0].set_title('Histogram pickup location within {} mile of {}'.format(range, name))

    idx = (haversine_np(df.dropoff_longitude, df.dropoff_latitude, loc[0], loc[1]) < range)
    df[idx].hour.hist(bins=100, ax=axs[1])
    axs[1].set_xlabel('Time')
    axs[1].set_title('Histogram dropoff location within {} mile of {}'.format(range, name));
    
plot_location_fare(times_sq, 'Times Square')



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3251389244.py in <cell line: 0>()
     15     axs[1].set_title('Histogram dropoff location within {} mile of {}'.format(range, name));
     16 
---> 17 plot_location_fare(times_sq, 'Times Square')
     18 

/tmp/ipykernel_11/3251389244.py in plot_location_fare(loc, name, range)
      5     fig, axs = plt.subplots(1, 2, figsize=(14, 5))
      6 
----> 7     idx = (haversine_np(df.pickup_longitude, df.pickup_latitude, loc[0], loc[1]) < range)
      8     df[idx].hour.hist(bins=100, ax=axs[0])
      9     axs[0].set_xlabel('Time')

NameError: name 'df' is not defined

## === cell 23
df.hist(column='hour',bins=100)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1052498947.py in <cell line: 0>()
----> 1 df.hist(column='hour',bins=100)

NameError: name 'df' is not defined

## === cell 25
    df['diff_lon'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['diff_lat'] = (df.dropoff_latitude - df.pickup_latitude).abs()

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/12210236.py in <cell line: 0>()
----> 1 df['diff_lon'] = (df.dropoff_longitude - df.pickup_longitude).abs()
      2 df['diff_lat'] = (df.dropoff_latitude - df.pickup_latitude).abs()

NameError: name 'df' is not defined

## === cell 26
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

lr = LinearRegression()

lr.fit(df[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','distance','passenger_count']], df['fare_amount'])

print('Intercept', round(lr.intercept_, 4))
print('Lat diff coef: ', round(lr.coef_[0], 4), 
      '\tLong diff coef:', round(lr.coef_[1], 4),
      '\tDistance coef:', round(lr.coef_[2], 4))


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3568308970.py in <cell line: 0>()
      4 lr = LinearRegression()
      5 
----> 6 lr.fit(df[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','distance','passenger_count']], df['fare_amount'])
      7 
      8 print('Intercept', round(lr.intercept_, 4))

NameError: name 'df' is not defined

## === cell 28
test = pd.read_csv('../input/new-york-city-taxi-fare-prediction/test.csv',low_memory=True)
print(test.head())
test['diff_lat'] = (test.dropoff_latitude-test.pickup_latitude).abs()
test['diff_long'] = (test.dropoff_longitude-test.pickup_longitude).abs()
test['distance'] = \
    haversine_np(test.pickup_longitude, test.pickup_latitude,test.dropoff_longitude,
                 test.dropoff_latitude)
test['year'], test['hour'] = test['pickup_datetime'].str.split(' ', 1).str
test['hour'] = test.hour.str[0:2]+test.hour.str[3:5]
test['year'] = test.year.str[:4]
test[['year','hour']] = test[['year','hour']].apply(pd.to_numeric)

test_id = list(test.pop('key'))
test.describe()

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/638997497.py in <cell line: 0>()
      7     haversine_np(test.pickup_longitude, test.pickup_latitude,test.dropoff_longitude,
      8                  test.dropoff_latitude)
----> 9 test['year'], test['hour'] = test['pickup_datetime'].str.split(' ', 1).str
     10 test['hour'] = test.hour.str[0:2]+test.hour.str[3:5]
     11 test['year'] = test.year.str[:4]

/usr/local/lib/python3.11/dist-packages/pandas/core/strings/accessor.py in wrapper(self, *args, **kwargs)
    135                 )
    136                 raise TypeError(msg)
--> 137             return func(self, *args, **kwargs)
    138 
    139         wrapper.__name__ = func_name

TypeError: StringMethods.split() takes from 1 to 2 positional arguments but 3 were given

## === cell 29
preds = lr.predict(test[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','distance','passenger_count']])

sub = pd.DataFrame({'key': test_id, 'fare_amount': preds})
sub.to_csv('output.csv', index = False)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/2562702060.py in <cell line: 0>()
----> 1 preds = lr.predict(test[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','distance','passenger_count']])
      2 
      3 sub = pd.DataFrame({'key': test_id, 'fare_amount': preds})
      4 sub.to_csv('output.csv', index = False)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in predict(self, X)
    352             Returns predicted values.
    353         """
--> 354         return self._decision_function(X)
    355 
    356     def _set_intercept(self, X_offset, y_offset, X_scale):

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in _decision_function(self, X)
    333 
    334     def _decision_function(self, X):
--> 335         check_is_fitted(self)
    336 
    337         X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This LinearRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
