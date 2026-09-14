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

3.95459

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

KMS_PER_RADIAN = 6371.0088

JFK_GEO_LOCATION = (40.6413, -73.7781)
LGR_GEO_LOCATION = (40.7769, -73.8740)
EWR_GEO_LOCATION = (40.6895, -74.1745)


## === cell 1

MAX_TRAINING_SIZE = 1_000_00

EPS_IN_KM = 0.5           ## NOTE that lat/long are available till 5th decimal value & 0.1km = 1.xe-5, hence avoid using smaller DBSCAN's eps, i.e., radius threshold for clustering
MIN_SAMPLES_CLUSTER = 500

RADIUS_VICINITY_AIRPORTS = 1.0

THERSHOLD_TRIP_FARE_RATE = 50.0

THRESHOLD_TRIP_DISTANCE = 25.0


## === cell 2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
plt.style.use('seaborn-whitegrid')

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

import timeit
from sklearn import metrics
from haversine import haversine


## === cell 3
start_time = timeit.default_timer()

df_train =  pd.read_csv('../input/train.csv', nrows = MAX_TRAINING_SIZE, parse_dates=["pickup_datetime"])
df_holdout =  pd.read_csv('../input/test.csv', parse_dates=["pickup_datetime"])
test_key = df_holdout['key']
df_train.drop(columns = ['key'], inplace=True)
df_holdout.drop(columns = ['key'], inplace=True)

elapsed = timeit.default_timer() - start_time
elapsed


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3618388343.py in <cell line: 0>()
----> 1 start_time = timeit.default_timer()
      2 
      3 # read data in pandas dataframe
      4 df_train =  pd.read_csv('../input/train.csv', nrows = MAX_TRAINING_SIZE, parse_dates=["pickup_datetime"])
      5 df_holdout =  pd.read_csv('../input/test.csv', parse_dates=["pickup_datetime"])

NameError: name 'timeit' is not defined

## === cell 4
print('Old size: %d' % len(df_train))

df_train = df_train[df_train.fare_amount >=0]

df_train = df_train.dropna(how='any', axis='rows')

df_train = df_train.drop(index= df_train[df_train.passenger_count >= 7].index, axis='rows')
df_train = df_train.drop(index= df_train[df_train.passenger_count == 0].index, axis='rows')

print('New size: %d' % len(df_train))


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1379924533.py in <cell line: 0>()
----> 1 print('Old size: %d' % len(df_train))
      2 
      3 ### Ignore -ve fare
      4 df_train = df_train[df_train.fare_amount >=0]
      5 

NameError: name 'df_train' is not defined

## === cell 5


def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
           (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
           (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
           (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])
            

BB = (-74.5, -72.8, 40.5, 41.8)

print('Old size: %d' % len(df_train))
df_train = df_train[select_within_boundingbox(df_train, BB)]
print('New size: %d' % len(df_train))


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1535071740.py in <cell line: 0>()
     16 BB = (-74.5, -72.8, 40.5, 41.8)
     17 
---> 18 print('Old size: %d' % len(df_train))
     19 df_train = df_train[select_within_boundingbox(df_train, BB)]
     20 print('New size: %d' % len(df_train))

NameError: name 'df_train' is not defined

## === cell 6
def addPickDropDistanceFeature(df):

    df['trip_distance'] = df.apply(
        (lambda row: haversine(
            (row['pickup_latitude'], row['pickup_longitude']),
            (row['dropoff_latitude'], row['dropoff_longitude']))
        ),
        axis='columns'
    )
    return df

def addAirportDistanceFeatures(df):

    df['pickup_distance_to_jfk'] = df.apply(
        (lambda row: haversine(
            (row['pickup_latitude'], row['pickup_longitude']),
            (JFK_GEO_LOCATION[0], JFK_GEO_LOCATION[1]))
        ),
        axis='columns'
    )

    df['drop_distance_to_jfk'] = df.apply(
        (lambda row: haversine(
            (row['dropoff_latitude'], row['dropoff_longitude']),
            (JFK_GEO_LOCATION[0], JFK_GEO_LOCATION[1]))
        ),
        axis='columns'
    )

    df['pickup_distance_to_lgr'] = df.apply(
        (lambda row: haversine(
            (row['pickup_latitude'], row['pickup_longitude']),
            (LGR_GEO_LOCATION[0], LGR_GEO_LOCATION[1]))
        ),
        axis='columns'
    )

    df['drop_distance_to_lgr'] = df.apply(
        (lambda row: haversine(
            (row['dropoff_latitude'], row['dropoff_longitude']),
            (LGR_GEO_LOCATION[0], LGR_GEO_LOCATION[1]))
        ),
        axis='columns'
    )

    df['pickup_distance_to_ewr'] = df.apply(
        (lambda row: haversine(
            (row['pickup_latitude'], row['pickup_longitude']),
            (EWR_GEO_LOCATION[0], EWR_GEO_LOCATION[1]))
        ),
        axis='columns'
    )

    df['drop_distance_to_ewr'] = df.apply(
        (lambda row: haversine(
            (row['dropoff_latitude'], row['dropoff_longitude']),
            (EWR_GEO_LOCATION[0], EWR_GEO_LOCATION[1]))
        ),
        axis='columns'
    )
    
    return df

def getAirportTrips(df, airportVicinity):
    ids = (df.pickup_distance_to_jfk < airportVicinity) | (df.drop_distance_to_jfk < airportVicinity) | (df.pickup_distance_to_lgr < airportVicinity) | (df.drop_distance_to_lgr < airportVicinity) | (df.pickup_distance_to_ewr < airportVicinity) | (df.drop_distance_to_ewr < airportVicinity)
    
    return ids


## === cell 7
start_time = timeit.default_timer()

df_train = addPickDropDistanceFeature(df_train)
df_holdout = addPickDropDistanceFeature(df_holdout)

elapsed = timeit.default_timer() - start_time
elapsed


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1685045264.py in <cell line: 0>()
----> 1 start_time = timeit.default_timer()
      2 
      3 # Add pickup-dropoff distance feature
      4 df_train = addPickDropDistanceFeature(df_train)
      5 df_holdout = addPickDropDistanceFeature(df_holdout)

NameError: name 'timeit' is not defined

## === cell 8

bucketsCount = 100
feat = 'trip_distance'

df_train[feat].hist(bins=bucketsCount, figsize = (15,8))
df_holdout[feat].hist(bins=bucketsCount, figsize = (15,8))
plt.yscale('log')
plt.xlabel(feat)
plt.ylabel("Frequency Log")


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/758086700.py in <cell line: 0>()
      6 feat = 'trip_distance'
      7 
----> 8 df_train[feat].hist(bins=bucketsCount, figsize = (15,8))
      9 df_holdout[feat].hist(bins=bucketsCount, figsize = (15,8))
     10 plt.yscale('log')

NameError: name 'df_train' is not defined

## === cell 9

print('Old size: %d' % len(df_train))
df_train = df_train[df_train.trip_distance < THRESHOLD_TRIP_DISTANCE]
print('New size: %d' % len(df_train))


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/351931314.py in <cell line: 0>()
      1 #(len(df_train[df_train[feat] > THRESHOLD_TRIP_DISTANCE]), len(df_holdout[df_holdout[feat] > THRESHOLD_TRIP_DISTANCE]))
      2 
----> 3 print('Old size: %d' % len(df_train))
      4 df_train = df_train[df_train.trip_distance < THRESHOLD_TRIP_DISTANCE]
      5 print('New size: %d' % len(df_train))

NameError: name 'df_train' is not defined

## === cell 10
def add_datetime_features(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")
    
    df['hour'] = df.pickup_datetime.dt.hour
    df['day'] = df.pickup_datetime.dt.day
    df['month'] = df.pickup_datetime.dt.month
    df['weekday'] = df.pickup_datetime.dt.weekday
    df['year'] = df.pickup_datetime.dt.year
    
    return df

start_time = timeit.default_timer()

df_train = add_datetime_features(df_train)
df_holdout = add_datetime_features(df_holdout)

elapsed = timeit.default_timer() - start_time
elapsed


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1921256647.py in <cell line: 0>()
     11     return df
     12 
---> 13 start_time = timeit.default_timer()
     14 
     15 df_train = add_datetime_features(df_train)

NameError: name 'timeit' is not defined

## === cell 11
train_len = len(df_train)
df_nyc_taxi = pd.concat([df_train, df_holdout], axis=0, ignore_index=False, sort=False)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1444448405.py in <cell line: 0>()
      1 # Merging both 'train' & 'holdout' for common feature engineering afterwhich 'holdout' data will be extracted
----> 2 train_len = len(df_train)
      3 df_nyc_taxi = pd.concat([df_train, df_holdout], axis=0, ignore_index=False, sort=False)
      4 #df_nyc_taxi.info()

NameError: name 'df_train' is not defined

## === cell 12
from sklearn.cluster import DBSCAN

EPS_IN_RADIAN = EPS_IN_KM / KMS_PER_RADIAN


## === cell 13
start_time = timeit.default_timer()

dbscan_pick = DBSCAN(eps=EPS_IN_RADIAN, min_samples=MIN_SAMPLES_CLUSTER, algorithm='ball_tree', metric='haversine').fit(np.radians(df_nyc_taxi.loc[:,'pickup_longitude':'pickup_latitude']))
labels_pick = dbscan_pick.labels_

elapsed = timeit.default_timer() - start_time
elapsed


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3155480323.py in <cell line: 0>()
----> 1 start_time = timeit.default_timer()
      2 
      3 dbscan_pick = DBSCAN(eps=EPS_IN_RADIAN, min_samples=MIN_SAMPLES_CLUSTER, algorithm='ball_tree', metric='haversine').fit(np.radians(df_nyc_taxi.loc[:,'pickup_longitude':'pickup_latitude']))
      4 labels_pick = dbscan_pick.labels_
      5 

NameError: name 'timeit' is not defined

## === cell 14
start_time = timeit.default_timer()

dbscan_drop = DBSCAN(eps=EPS_IN_RADIAN, min_samples=MIN_SAMPLES_CLUSTER, algorithm='ball_tree', metric='haversine').fit(np.radians(df_nyc_taxi.loc[:,'dropoff_longitude':'dropoff_latitude']))
labels_drop = dbscan_drop.labels_

elapsed = timeit.default_timer() - start_time
elapsed


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4123096670.py in <cell line: 0>()
----> 1 start_time = timeit.default_timer()
      2 
      3 dbscan_drop = DBSCAN(eps=EPS_IN_RADIAN, min_samples=MIN_SAMPLES_CLUSTER, algorithm='ball_tree', metric='haversine').fit(np.radians(df_nyc_taxi.loc[:,'dropoff_longitude':'dropoff_latitude']))
      4 labels_drop = dbscan_drop.labels_
      5 

NameError: name 'timeit' is not defined

## === cell 15
df_nyc_taxi['dense_DBSCAN_trips'] = ((labels_pick != -1) & (labels_drop != -1))


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3765892631.py in <cell line: 0>()
----> 1 df_nyc_taxi['dense_DBSCAN_trips'] = ((labels_pick != -1) & (labels_drop != -1))

NameError: name 'labels_pick' is not defined

## === cell 16
df_tmp = df_nyc_taxi.loc[df_nyc_taxi.dense_DBSCAN_trips == 1]
plt.plot(df_tmp.pickup_longitude, df_tmp.pickup_latitude, 'o')


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1543722320.py in <cell line: 0>()
      1 # NOTE that our focus of DBSCAN is not to differentiate levels of clusters, i.e., set of connected clusters => hence we are plotting all clusters together
----> 2 df_tmp = df_nyc_taxi.loc[df_nyc_taxi.dense_DBSCAN_trips == 1]
      3 plt.plot(df_tmp.pickup_longitude, df_tmp.pickup_latitude, 'o')

NameError: name 'df_nyc_taxi' is not defined

## === cell 17
df_train = df_nyc_taxi.iloc[:train_len, :]
df_holdout = df_nyc_taxi.iloc[train_len:, :].iloc[:, df_nyc_taxi.columns != 'fare_amount']

(len(df_train), len(df_holdout))


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3111130415.py in <cell line: 0>()
----> 1 df_train = df_nyc_taxi.iloc[:train_len, :]
      2 df_holdout = df_nyc_taxi.iloc[train_len:, :].iloc[:, df_nyc_taxi.columns != 'fare_amount']
      3 
      4 (len(df_train), len(df_holdout))

NameError: name 'df_nyc_taxi' is not defined

## === cell 18
df_train.loc[df_train.trip_distance < 0.2, 'trip_distance'] = 0.2

(df_train.fare_amount / df_train.trip_distance).hist(bins=bucketsCount, figsize = (15,8))
plt.yscale('log')
plt.xlabel('trip_rate')
plt.ylabel("Log Frequency")


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4229886392.py in <cell line: 0>()
      1 # Ceiling near-zero fare values to 0.2 to check fare/dist behavior
----> 2 df_train.loc[df_train.trip_distance < 0.2, 'trip_distance'] = 0.2
      3 
      4 (df_train.fare_amount / df_train.trip_distance).hist(bins=bucketsCount, figsize = (15,8))
      5 plt.yscale('log')

NameError: name 'df_train' is not defined

## === cell 19
df_train['trip_rate'] = df_train.apply(
    (lambda row: (row.fare_amount / row.trip_distance)),
    axis='columns'
)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2474588891.py in <cell line: 0>()
----> 1 df_train['trip_rate'] = df_train.apply(
      2     (lambda row: (row.fare_amount / row.trip_distance)),
      3     axis='columns'
      4 )

NameError: name 'df_train' is not defined

## === cell 21
'''
ids = (df_train.trip_rate < THERSHOLD_TRIP_FARE_RATE)

print('Old size: %d' % len(df_train))
df_train = df_train[ids]
print('New size: %d' % len(df_train))
'''


## === cell 22
start_time = timeit.default_timer()

df_train = addAirportDistanceFeatures(df_train)
df_holdout = addAirportDistanceFeatures(df_holdout)

elapsed = timeit.default_timer() - start_time
elapsed


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1123184207.py in <cell line: 0>()
----> 1 start_time = timeit.default_timer()
      2 
      3 # Add airport trips distance features
      4 df_train = addAirportDistanceFeatures(df_train)
      5 df_holdout = addAirportDistanceFeatures(df_holdout)

NameError: name 'timeit' is not defined

## === cell 24

airportTripsIds = getAirportTrips(df_holdout, RADIUS_VICINITY_AIRPORTS)
df_holdout['airport_bound'] = airportTripsIds

airportTripsIds = getAirportTrips(df_train, RADIUS_VICINITY_AIRPORTS)
df_train['airport_bound'] = airportTripsIds
df_airport_trips = df_train.loc[airportTripsIds]
df_city_trips = df_train.loc[-airportTripsIds]


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/354914057.py in <cell line: 0>()
      1 # Split training data into Airport & City trips
      2 
----> 3 airportTripsIds = getAirportTrips(df_holdout, RADIUS_VICINITY_AIRPORTS)
      4 df_holdout['airport_bound'] = airportTripsIds
      5 

NameError: name 'df_holdout' is not defined

## === cell 25
pd.DataFrame(data={'Airport Trips' : df_airport_trips.trip_rate, 'City Trips' : df_city_trips.trip_rate}).describe()


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1980698976.py in <cell line: 0>()
      1 # Compare trip_rate for Airport & City trips
----> 2 pd.DataFrame(data={'Airport Trips' : df_airport_trips.trip_rate, 'City Trips' : df_city_trips.trip_rate}).describe()

NameError: name 'df_airport_trips' is not defined

## === cell 26
pd.DataFrame(data={'Good Density Trips' : df_train.loc[df_train.dense_DBSCAN_trips == 1].trip_rate, 'LOW Density Pickups' : df_train.loc[df_train.dense_DBSCAN_trips == 0].trip_rate}).describe()


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2438460651.py in <cell line: 0>()
      1 # Compare trip_rate for good -vs- poorly dense trips
----> 2 pd.DataFrame(data={'Good Density Trips' : df_train.loc[df_train.dense_DBSCAN_trips == 1].trip_rate, 'LOW Density Pickups' : df_train.loc[df_train.dense_DBSCAN_trips == 0].trip_rate}).describe()

NameError: name 'df_train' is not defined

## === cell 27
df_train = df_train.drop(columns = ['pickup_datetime', 'pickup_distance_to_jfk', 'drop_distance_to_jfk', 'pickup_distance_to_lgr', 'drop_distance_to_lgr', 'pickup_distance_to_ewr', 'drop_distance_to_ewr', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'trip_rate'])
df_train.info()


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3736982815.py in <cell line: 0>()
----> 1 df_train = df_train.drop(columns = ['pickup_datetime', 'pickup_distance_to_jfk', 'drop_distance_to_jfk', 'pickup_distance_to_lgr', 'drop_distance_to_lgr', 'pickup_distance_to_ewr', 'drop_distance_to_ewr', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'trip_rate'])
      2 df_train.info()

NameError: name 'df_train' is not defined

## === cell 28
y = df_train['fare_amount']
train = df_train.drop(columns=['fare_amount'])

x_train, x_test, y_train, y_test = train_test_split(train, y, random_state=0, test_size=0.01)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1825457831.py in <cell line: 0>()
----> 1 y = df_train['fare_amount']
      2 train = df_train.drop(columns=['fare_amount'])
      3 
      4 x_train, x_test, y_train, y_test = train_test_split(train, y, random_state=0, test_size=0.01)

NameError: name 'df_train' is not defined

## === cell 29
params = {
    'max_depth': 8, #Result of tuning with CV
    'eta':.03, #Result of tuning with CV
    'subsample': 1, #Result of tuning with CV
    'colsample_bytree': 0.8, #Result of tuning with CV
    'objective':'reg:linear',
    'eval_metric':'rmse',
    'silent': 1
}

CV=False
if CV:
    dtrain = xgb.DMatrix(train,label=y)
    gridsearch_params = [
        (eta)
        for eta in np.arange(.04, 0.12, .02)
    ]

    min_rmse = float("Inf")
    best_params = None
    for (eta) in gridsearch_params:
        print("CV with eta={} ".format(
                                 eta))

        params['eta'] = eta

        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            nfold=3,
            metrics={'rmse'},
            early_stopping_rounds=10
        )

        mean_rmse = cv_results['test-rmse-mean'].min()
        boost_rounds = cv_results['test-rmse-mean'].argmin()
        print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = (eta)

    print("Best params: {}, RMSE: {}".format(best_params, min_rmse))
else:
    params['silent'] = 0 #Turn on output
    print(params)


## === cell 30
def XGBmodel(x_train,x_test,y_train,y_test,params):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params=params,
                    dtrain=matrix_train,num_boost_round=5000, 
                    early_stopping_rounds=10,evals=[(matrix_test,'test')])
    return model

model = XGBmodel(x_train,x_test,y_train,y_test,params)


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2363490757.py in <cell line: 0>()
      7     return model
      8 
----> 9 model = XGBmodel(x_train,x_test,y_train,y_test,params)

NameError: name 'x_train' is not defined

## === cell 31
x_pred = df_holdout.drop(columns = ['pickup_datetime', 'pickup_distance_to_jfk', 'drop_distance_to_jfk', 'pickup_distance_to_lgr', 'drop_distance_to_lgr', 'pickup_distance_to_ewr', 'drop_distance_to_ewr', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'])

prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2216237716.py in <cell line: 0>()
      1 #Read and preprocess test set
----> 2 x_pred = df_holdout.drop(columns = ['pickup_datetime', 'pickup_distance_to_jfk', 'drop_distance_to_jfk', 'pickup_distance_to_lgr', 'drop_distance_to_lgr', 'pickup_distance_to_ewr', 'drop_distance_to_ewr', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'])
      3 
      4 #Predict from test set
      5 prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)

NameError: name 'df_holdout' is not defined

## === cell 32
len(test_key)


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3634691910.py in <cell line: 0>()
----> 1 len(test_key)

NameError: name 'test_key' is not defined

## === cell 33
submission = pd.DataFrame({
        "key": test_key,
        "fare_amount": prediction.round(2)
})

submission.to_csv('taxi_fare_submission.csv',index=False)
submission.head()


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1489528079.py in <cell line: 0>()
      1 #Create submission file
      2 submission = pd.DataFrame({
----> 3         "key": test_key,
      4         "fare_amount": prediction.round(2)
      5 })

NameError: name 'test_key' is not defined
