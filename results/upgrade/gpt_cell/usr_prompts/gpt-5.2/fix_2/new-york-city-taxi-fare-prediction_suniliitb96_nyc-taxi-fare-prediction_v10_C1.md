# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

No external packages required in the script and installed.

# 3. Data file paths

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

# 4. Code solution

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
import timeit

start_time = timeit.default_timer()

df_train = pd.read_csv(
    "../input/train.csv", nrows=MAX_TRAINING_SIZE, parse_dates=["pickup_datetime"]
)
df_test = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])
test_key = df_test["key"]
df_train.drop(columns=["key"], inplace=True)
df_test.drop(columns=["key"], inplace=True)

elapsed = timeit.default_timer() - start_time
elapsed


## === cell 4
df_train.head()


## === cell 5
print('Old size: %d' % len(df_train))

df_train = df_train[df_train.fare_amount >=0]

df_train = df_train.dropna(how='any', axis='rows')

df_train = df_train.drop(index= df_train[df_train.passenger_count >= 7].index, axis='rows')
df_train = df_train.drop(index= df_train[df_train.passenger_count == 0].index, axis='rows')

print('New size: %d' % len(df_train))


## === cell 6


def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
           (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
           (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
           (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])
            

BB = (-74.5, -72.8, 40.5, 41.8)

print('Old size: %d' % len(df_train))
df_train = df_train[select_within_boundingbox(df_train, BB)]
print('New size: %d' % len(df_train))


## === cell 7
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


## === cell 8
start_time = timeit.default_timer()

df_train = addPickDropDistanceFeature(df_train)
df_test = addPickDropDistanceFeature(df_test)

elapsed = timeit.default_timer() - start_time
elapsed


## --- ERROR in cell 8, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2866252144.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m [0;31m# Add pickup-dropoff distance feature[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m [0mdf_train[0m [0;34m=[0m [0maddPickDropDistanceFeature[0m[0;34m([0m[0mdf_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0mdf_test[0m [0;34m=[0m [0maddPickDropDistanceFeature[0m[0;34m([0m[0mdf_test[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2819859981.py[0m in [0;36maddPickDropDistanceFeature[0;34m(df)[0m
[1;32m      2[0m [0;32mdef[0m [0maddPickDropDistanceFeature[0m[0;34m([0m[0mdf[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;34m[0m[0m
[0;32m----> 4[0;31m     df['trip_distance'] = df.apply(
[0m[1;32m      5[0m         (lambda row: haversine(
[1;32m      6[0m             [0;34m([0m[0mrow[0m[0;34m[[0m[0;34m'pickup_latitude'[0m[0;34m][0m[0;34m,[0m [0mrow[0m[0;34m[[0m[0;34m'pickup_longitude'[0m[0;34m][0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36mapply[0;34m(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)[0m
[1;32m  10372[0m             [0mkwargs[0m[0;34m=[0m[0mkwargs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m  10373[0m         )
[0;32m> 10374[0;31m         [0;32mreturn[0m [0mop[0m[0;34m.[0m[0mapply[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__finalize__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mmethod[0m[0;34m=[0m[0;34m"apply"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m  10375[0m [0;34m[0m[0m
[1;32m  10376[0m     def map(

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py[0m in [0;36mapply[0;34m(self)[0m
[1;32m    914[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mapply_raw[0m[0;34m([0m[0mengine[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mengine[0m[0;34m,[0m [0mengine_kwargs[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mengine_kwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    915[0m [0;34m[0m[0m
[0;32m--> 916[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mapply_standard[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    917[0m [0;34m[0m[0m
[1;32m    918[0m     [0;32mdef[0m [0magg[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py[0m in [0;36mapply_standard[0;34m(self)[0m
[1;32m   1061[0m     [0;32mdef[0m [0mapply_standard[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1062[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mengine[0m [0;34m==[0m [0;34m"python"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1063[0;31m             [0mresults[0m[0;34m,[0m [0mres_index[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mapply_series_generator[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1064[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1065[0m             [0mresults[0m[0;34m,[0m [0mres_index[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mapply_series_numba[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py[0m in [0;36mapply_series_generator[0;34m(self)[0m
[1;32m   1079[0m             [0;32mfor[0m [0mi[0m[0;34m,[0m [0mv[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0mseries_gen[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1080[0m                 [0;31m# ignore SettingWithCopy here in case the user mutates[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1081[0;31m                 [0mresults[0m[0;34m[[0m[0mi[0m[0;34m][0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mfunc[0m[0;34m([0m[0mv[0m[0;34m,[0m [0;34m*[0m[0mself[0m[0;34m.[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mself[0m[0;34m.[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1082[0m                 [0;32mif[0m [0misinstance[0m[0;34m([0m[0mresults[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m,[0m [0mABCSeries[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1083[0m                     [0;31m# If we have a view on v, we need to make a copy because[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2819859981.py[0m in [0;36m<lambda>[0;34m(row)[0m
[1;32m      3[0m [0;34m[0m[0m
[1;32m      4[0m     df['trip_distance'] = df.apply(
[0;32m----> 5[0;31m         (lambda row: haversine(
[0m[1;32m      6[0m             [0;34m([0m[0mrow[0m[0;34m[[0m[0;34m'pickup_latitude'[0m[0;34m][0m[0;34m,[0m [0mrow[0m[0;34m[[0m[0;34m'pickup_longitude'[0m[0;34m][0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m             (row['dropoff_latitude'], row['dropoff_longitude']))

[0;31mNameError[0m: name 'haversine' is not defined

## === cell 9

bucketsCount = 100
feat = 'trip_distance'


df_train[feat].hist(bins=bucketsCount, figsize = (15,8))
df_test[feat].hist(bins=bucketsCount, figsize = (15,8))
plt.yscale('log')
plt.xlabel(feat)
plt.ylabel("Frequency Log")
'''
#newFeat = 'catFare'
#df_train[newFeat] = pd.cut(df_train[feat], custom_bucket_array)
#labels, levels = pd.factorize(df_train[newFeat])
#df_train[newFeat] = labels
'''
