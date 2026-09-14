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
scipy==1.15.3
sklearn-pandas==2.2.0
tqdm==4.67.1

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

3.3504463754491907

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
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import os
from tqdm import tqdm
from scipy.sparse import csr_matrix, hstack
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

print(os.listdir("../input"))



## === cell 2
nyc_weather = pd.read_csv('../input/nyc-weather/nyc_weather.csv')
weather_cols = ['DATE','AWND','PRCP','SNOW','TMAX','TMIN']
nyc_weather = nyc_weather[weather_cols].copy()
nyc_weather['DATE'] = pd.to_datetime(nyc_weather['DATE'], utc=True, format='%m/%d/%Y') 
nyc_weather.head()


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2347640056.py in <cell line: 0>()
----> 1 nyc_weather = pd.read_csv('../input/nyc-weather/nyc_weather.csv')
      2 weather_cols = ['DATE','AWND','PRCP','SNOW','TMAX','TMIN']
      3 nyc_weather = nyc_weather[weather_cols].copy()
      4 nyc_weather['DATE'] = pd.to_datetime(nyc_weather['DATE'], utc=True, format='%m/%d/%Y')
      5 nyc_weather.head()

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../input/nyc-weather/nyc_weather.csv'

## === cell 4
plt.rc('figure', figsize=(15, 8))
plt.subplot(1,2,1)
plt.hist(nyc_weather.TMAX, bins =  30)
plt.xlabel('Temperature (C)')
plt.ylabel('Frequency Count')
plt.title('Max Daily Temperature')
plt.subplot(1,2,2)
plt.hist(nyc_weather.TMIN, bins =  30)
plt.xlabel('Temperature (C)')
plt.ylabel('Frequency Count')
plt.title('Min Daily Temperature')
plt.show()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/933130244.py in <cell line: 0>()
      1 plt.rc('figure', figsize=(15, 8))
      2 plt.subplot(1,2,1)
----> 3 plt.hist(nyc_weather.TMAX, bins =  30)
      4 plt.xlabel('Temperature (C)')
      5 plt.ylabel('Frequency Count')

NameError: name 'nyc_weather' is not defined

## === cell 5
nyc_weather.describe()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2564558254.py in <cell line: 0>()
----> 1 nyc_weather.describe()

NameError: name 'nyc_weather' is not defined

## === cell 7
%%time
train = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv',  nrows= 15_000_000) 
test = pd.read_csv('../input/new-york-city-taxi-fare-prediction/test.csv') 
test_id = test.key.values #set this value for final submission
print(train.info())

## === cell 8
%%time
train['pickup_datetime'] = train['pickup_datetime'].str.slice(0, 16)
test['pickup_datetime'] = test['pickup_datetime'].str.slice(0, 16)

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M') 
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], utc=True,format='%Y-%m-%d %H:%M') 

train_sample = train.dropna().copy()
del train

train_sample.drop(labels='key', axis=1, inplace=True)
test.drop(labels='key', axis=1, inplace=True)

train_sample.loc[:,'passenger_count'] = train_sample.passenger_count.astype(dtype = 'uint8')
train_sample['pickup_longitude'] = train_sample.pickup_longitude.astype(dtype = 'float32')
train_sample['pickup_latitude'] = train_sample.pickup_latitude.astype(dtype = 'float32')
train_sample['dropoff_longitude'] = train_sample.dropoff_longitude.astype(dtype = 'float32')
train_sample['dropoff_latitude'] = train_sample.dropoff_latitude.astype(dtype = 'float32')
train_sample['fare_amount'] = train_sample.fare_amount.astype(dtype = 'float32')

test['pickup_longitude'] = test.pickup_longitude.astype(dtype = 'float32')
test['pickup_latitude'] = test.pickup_latitude.astype(dtype = 'float32')
test['dropoff_longitude'] = test.dropoff_longitude.astype(dtype = 'float32')
test['dropoff_latitude'] = test.dropoff_latitude.astype(dtype = 'float32')

train_sample = train_sample.loc[train_sample.pickup_longitude.between(test.pickup_longitude.min(), test.pickup_longitude.max())]
train_sample = train_sample.loc[train_sample.pickup_latitude.between(test.pickup_latitude.min(), test.pickup_latitude.max())]
train_sample = train_sample.loc[train_sample.dropoff_longitude.between(test.dropoff_longitude.min(), test.dropoff_longitude.max())]
train_sample = train_sample.loc[train_sample.dropoff_latitude.between(test.dropoff_latitude.min(), test.dropoff_latitude.max())]

train_sample['hour'] = train_sample['pickup_datetime'].apply(lambda time: time.hour)
train_sample['month'] = train_sample['pickup_datetime'].apply(lambda time: time.month)
train_sample['day_of_week'] = train_sample['pickup_datetime'].apply(lambda time: time.dayofweek)
train_sample['year'] = train_sample['pickup_datetime'].apply(lambda t: t.year)


test['hour'] = test['pickup_datetime'].apply(lambda time: time.hour)
test['month'] = test['pickup_datetime'].apply(lambda time: time.month)
test['day_of_week'] = test['pickup_datetime'].apply(lambda time: time.dayofweek)
test['year'] = test['pickup_datetime'].apply(lambda t: t.year)

train_sample['hour'] = train_sample.hour.astype(dtype = 'uint8')
train_sample['month'] = train_sample.month.astype(dtype = 'uint8')
train_sample['day_of_week'] = train_sample.day_of_week.astype(dtype = 'uint8')
train_sample['year'] = train_sample.year.astype(dtype = 'uint16')


test['hour'] = test.hour.astype(dtype = 'uint8')
test['month'] = test.month.astype(dtype = 'uint8')
test['day_of_week'] = test.day_of_week.astype(dtype = 'uint8')
test['year'] = test.year.astype(dtype = 'uint16')

## === cell 9
%%time

train_sample['pickup_day'] = train_sample.pickup_datetime.dt.floor('d')
train_sample = train_sample.merge(nyc_weather, how = 'left', left_on ='pickup_day', right_on = 'DATE')
train_sample.drop(columns = ['pickup_day','DATE'], axis = 0, inplace = True)

test['pickup_day'] = test.pickup_datetime.dt.floor('d')
test = test.merge(nyc_weather, how = 'left', left_on ='pickup_day', right_on = 'DATE')
test.drop(columns = ['pickup_day','DATE'], axis = 0, inplace = True)

train_sample['AWND'] = train_sample.AWND.astype(dtype = 'float16')
train_sample['PRCP'] = train_sample.PRCP.astype(dtype = 'float16')
train_sample['SNOW'] = train_sample.day_of_week.astype(dtype = 'float16')
train_sample['TMAX'] = train_sample.TMAX.astype(dtype = 'float16')
train_sample['TMIN'] = train_sample.TMAX.astype(dtype = 'float16')

test['AWND'] = test.AWND.astype(dtype = 'float16')
test['PRCP'] = test.PRCP.astype(dtype = 'float16')
test['SNOW'] = test.day_of_week.astype(dtype = 'float16')
test['TMAX'] = test.TMAX.astype(dtype = 'float16')
test['TMIN'] = test.TMAX.astype(dtype = 'float16')

train_sample['hot_day'] = np.where(train_sample.TMAX >= 30,1,0)
train_sample['cold_day'] = np.where(train_sample.TMIN <= 0,1,0)

test['hot_day'] =  np.where(test.TMAX >= 35,1,0)
test['cold_day'] = np.where(test.TMIN <= -5,1,0)
train_sample['hot_day'] = train_sample.hot_day.astype(dtype = 'uint8')
train_sample['cold_day'] = train_sample.cold_day.astype(dtype = 'uint8')
test['hot_day'] = test.hot_day.astype(dtype = 'uint8')
test['cold_day'] = test.cold_day.astype(dtype = 'uint8')

def degree_to_radion(degree):
    return degree*(np.pi/180)

def calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
    
    from_lat = degree_to_radion(pickup_latitude)
    from_long = degree_to_radion(pickup_longitude)
    to_lat = degree_to_radion(dropoff_latitude)
    to_long = degree_to_radion(dropoff_longitude)
    
    radius = 6371.01
    
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = np.sin(lat_diff / 2)**2 + np.cos(degree_to_radion(from_lat)) * np.cos(degree_to_radion(to_lat)) * np.sin(long_diff / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return radius * c

train_sample['distance'] = calculate_distance(train_sample.pickup_latitude, train_sample.pickup_longitude, train_sample.dropoff_latitude, train_sample.dropoff_longitude)
test['distance'] = calculate_distance(test.pickup_latitude, test.pickup_longitude, test.dropoff_latitude, test.dropoff_longitude)

train_sample['distance'] = train_sample.distance.astype(dtype = 'float32')
test['distance'] = test.distance.astype(dtype = 'float32')


train_sample['day_hour'] = train_sample.day_of_week.astype(str) + "_" + train_sample.hour.astype(str)
train_sample['day_hour'] = train_sample['day_hour'].astype('category')

test['day_hour'] = test.day_of_week.astype(str) + test.hour.astype(str)
test['day_hour'] = test['day_hour'].astype('category')

train_sample = train_sample[train_sample.fare_amount > 0]
print(train_sample.info())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'nyc_weather' is not defined

## === cell 10
holidays = pd.read_csv('../input/us-bank-holidays-20092018/US Bank Holidays 2012-2018.csv')
holidays['Date'] = pd.to_datetime(holidays['Date'], utc=True, format='%m/%d/%y') 
train_sample['pickup_day'] = train_sample.pickup_datetime.dt.floor('d')
train_sample = train_sample.merge(holidays, left_on = 'pickup_day', right_on = 'Date', how = 'left')
train_sample['Holiday'] =train_sample.Holiday.fillna('None')

le = LabelEncoder()
train_sample['holiday'] = le.fit_transform(train_sample.Holiday.values)
train_sample.drop(['Holiday','Date','pickup_day'], axis = 1, inplace = True)

test['pickup_day'] = test.pickup_datetime.dt.floor('d')
test = test.merge(holidays, left_on = 'pickup_day', right_on = 'Date', how = 'left')
test['Holiday'] =test.Holiday.fillna('None')

test['holiday'] = le.fit_transform(test.Holiday.values)
test.drop(['Holiday','Date','pickup_day'], axis = 1, inplace = True)


train_sample['holiday'] = train_sample.holiday.astype(dtype = 'uint8')
test['holiday'] = test.holiday.astype(dtype = 'uint8')

train_sample.info()

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3465981524.py in <cell line: 0>()
      1 #holidays
----> 2 holidays = pd.read_csv('../input/us-bank-holidays-20092018/US Bank Holidays 2012-2018.csv')
      3 holidays['Date'] = pd.to_datetime(holidays['Date'], utc=True, format='%m/%d/%y')
      4 train_sample['pickup_day'] = train_sample.pickup_datetime.dt.floor('d')
      5 train_sample = train_sample.merge(holidays, left_on = 'pickup_day', right_on = 'Date', how = 'left')

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../input/us-bank-holidays-20092018/US Bank Holidays 2012-2018.csv'

## === cell 11
train_sample.head()

## === cell 13
%%time
full_pickups = pd.concat([train_sample[['pickup_longitude','pickup_latitude']],test[['pickup_longitude','pickup_latitude']]], axis = 0)
full_pickups.columns = ['x','y']
full_dropoffs = pd.concat([train_sample[['dropoff_longitude','dropoff_latitude']],test[['dropoff_longitude','dropoff_latitude']]], axis = 0)
full_dropoffs.columns = ['x','y']
full_locs = pd.concat([full_pickups,full_dropoffs], axis = 0)

full_locs['x'] = full_locs.x.round(4)
full_locs['y'] = full_locs.y.round(4)

full_locs = full_locs.groupby(['x','y']).count().reset_index()
full_locs.info()


## === cell 14
%%time
X_df = full_locs.copy() #.sample(100000) #
X_kmeans = full_locs.values

num_clusters = 200

with open('../input/taxi-weather-holidays-kmeans-neighborhoods/kmeans_200_round4.pkl', 'rb') as fid:
    kmeans = pickle.load(fid)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
<timed exec> in <module>

FileNotFoundError: [Errno 2] No such file or directory: '../input/taxi-weather-holidays-kmeans-neighborhoods/kmeans_200_round4.pkl'

## === cell 15
z = kmeans.predict(X_kmeans)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1645187891.py in <cell line: 0>()
      1 #create labels for graph below
----> 2 z = kmeans.predict(X_kmeans)

NameError: name 'kmeans' is not defined

## === cell 16
centers = kmeans.cluster_centers_

x_centers = [pair[0] for pair in centers]
y_centers = [pair[1] for pair in centers]
z_centers = np.arange(num_clusters)

plt.subplot(1,2,1)
plt.scatter(X_df['x'], X_df['y'], c=z)
plt.gray()
plt.xlabel('Pickup/Dropoff Longitude')
plt.ylabel('Pickup/Dropoff Latitude')
plt.title('Clusters of NYC locations')
plt.subplot(1,2,2)
plt.scatter(x_centers, y_centers, c=z_centers)
plt.gray()
plt.xlabel('Pickup/Dropoff Longitude')
plt.ylabel('Pickup/Dropoff Latitude')
plt.title('Cluster Centers of NYC locations')

plt.show()

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/809121302.py in <cell line: 0>()
----> 1 centers = kmeans.cluster_centers_
      2 
      3 x_centers = [pair[0] for pair in centers]
      4 y_centers = [pair[1] for pair in centers]
      5 z_centers = np.arange(num_clusters)

NameError: name 'kmeans' is not defined

## === cell 17
%%time
del X_kmeans, X_df, full_pickups, full_dropoffs 

train_sample['pickup_neighborhood'] = kmeans.predict(np.column_stack([train_sample.pickup_longitude.values,train_sample.pickup_latitude.values]))
train_sample['dropoff_neighborhood'] = kmeans.predict(np.column_stack([train_sample.dropoff_longitude.values,train_sample.dropoff_latitude.values]))

test['pickup_neighborhood'] =  kmeans.predict(np.column_stack([test.pickup_longitude.values, test.pickup_latitude.values]))
test['dropoff_neighborhood'] = kmeans.predict(np.column_stack([test.dropoff_longitude.values,test.dropoff_latitude.values]))

train_sample['pickup_neighborhood'] = train_sample.pickup_neighborhood.astype(dtype = 'uint8')
train_sample['dropoff_neighborhood'] = train_sample.dropoff_neighborhood.astype(dtype = 'uint8')

test['pickup_neighborhood'] = test.pickup_neighborhood.astype(dtype = 'uint8')
test['dropoff_neighborhood'] = test.dropoff_neighborhood.astype(dtype = 'uint8')

print(train_sample.info())

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'kmeans' is not defined

## === cell 18
with open('kmeans_200_round4_v2.pkl', 'wb') as fid:
    pickle.dump(kmeans, fid)    


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/537044278.py in <cell line: 0>()
      1 #save kmeans model for future use
      2 with open('kmeans_200_round4_v2.pkl', 'wb') as fid:
----> 3     pickle.dump(kmeans, fid)

NameError: name 'kmeans' is not defined

## === cell 20
%%time

categorical_cols = ['day_hour','month','year','pickup_neighborhood','dropoff_neighborhood','passenger_count','hot_day','cold_day','holiday'] #,'jfk_pickup','jfk_dropoff','lga_pickup','lga_dropoff','ewr_pickup','ewr_dropoff'     'hour','day_of_week', 'pickup_lat_round','pickup_long_round'
numerical_cols = ['distance','AWND','PRCP','SNOW'] # 'delta_lat','delta_long', , 'pickup_latitude','pickup_longitude','TMAX','TMIN'

X_cats = train_sample[categorical_cols].values
X_cats_test = test[categorical_cols].values
X_cats_full = np.append(X_cats, X_cats_test, axis = 0)

ohe = OneHotEncoder(categories = 'auto')
X_onehot = ohe.fit_transform(X_cats_full)
del X_cats,X_cats_test, X_cats_full

X_nums = train_sample[numerical_cols].values
X_nums_test = test[numerical_cols].values
X_nums_full = np.append(X_nums, X_nums_test, axis = 0)
X_nums_sparse = csr_matrix(X_nums_full)
del X_nums, X_nums_test, X_nums_full

X_full = hstack([X_onehot, X_nums_sparse]).tocsr()

si = SimpleImputer()
X_full_imputed = si.fit_transform(X_full)

X = X_full_imputed[:train_sample.shape[0],:]
X_public = X_full_imputed[train_sample.shape[0]:,:]

y = train_sample.fare_amount.values
del X_onehot, X_nums_sparse


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .1)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
<timed exec> in <module>

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

KeyError: "['day_hour', 'pickup_neighborhood', 'dropoff_neighborhood', 'hot_day', 'cold_day', 'holiday'] not in index"

## === cell 22
%%time

params = {'objective': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 50,
          'max_depth': 8,
          'learning_rate': 0.5,
          'bagging_fraction': 0.8,
          'feature_fraction': 0.8,
          'min_split_gain': 0.02,
          'min_child_samples': 10, 
          'min_child_weight': 0.02, 
          'lambda_l2': 0.0475,
          'verbosity': -1,
          'data_random_seed': 17,
          'early_stop': 100,
          'verbose_eval': 100,
          'num_rounds': 100} #500

d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)
watchlist = [d_train, d_test]
num_rounds = 100
verbose_eval = 100
early_stop = 100
model_lgb = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop)
    
pred_test_y_lgb = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)

print("LGB Loss = " + str(sqrt(mean_squared_error(y_test,pred_test_y_lgb))))


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'X_train' is not defined

## === cell 23
lgb_public= model_lgb.predict(X_public, num_iteration=model_lgb.best_iteration)

final_pred_public =lgb_public.flatten()

test_predictions_lgb = [float(np.asscalar(x)) for x in final_pred_public]
test_predictions_lgb = [x if x>0 else 0 for x in test_predictions_lgb]
sample = pd.DataFrame({'key': test_id,'fare_amount':test_predictions_lgb})
sample = sample.reindex(['key', 'fare_amount'], axis=1)
sample.to_csv('submission_lgb.csv', index=False)
sample.head()

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2502207253.py in <cell line: 0>()
----> 1 lgb_public= model_lgb.predict(X_public, num_iteration=model_lgb.best_iteration)
      2 
      3 final_pred_public =lgb_public.flatten()
      4 
      5 #clean and format final submission

NameError: name 'model_lgb' is not defined

## === cell 24

plt.rc('figure', figsize=(10, 10))
plt.hist(test_predictions_lgb, bins = 100)
plt.xlabel('Predicticted Price')
plt.ylabel('Frequency')
plt.title('Predictions from LGB')
plt.show()

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/601318626.py in <cell line: 0>()
      1 plt.rc('figure', figsize=(10, 10))
----> 2 plt.hist(test_predictions_lgb, bins = 100)
      3 plt.xlabel('Predicticted Price')
      4 plt.ylabel('Frequency')
      5 plt.title('Predictions from LGB')

NameError: name 'test_predictions_lgb' is not defined
