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
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3618388343.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mstart_time[0m [0;34m=[0m [0mtimeit[0m[0;34m.[0m[0mdefault_timer[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m [0;31m# read data in pandas dataframe[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0mdf_train[0m [0;34m=[0m  [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34m'../input/train.csv'[0m[0;34m,[0m [0mnrows[0m [0;34m=[0m [0mMAX_TRAINING_SIZE[0m[0;34m,[0m [0mparse_dates[0m[0;34m=[0m[0;34m[[0m[0;34m"pickup_datetime"[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mdf_holdout[0m [0;34m=[0m  [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34m'../input/test.csv'[0m[0;34m,[0m [0mparse_dates[0m[0;34m=[0m[0;34m[[0m[0;34m"pickup_datetime"[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: name 'timeit' is not defined

## === cell 4
print('Old size: %d' % len(df_train))

df_train = df_train[df_train.fare_amount >=0]

df_train = df_train.dropna(how='any', axis='rows')

df_train = df_train.drop(index= df_train[df_train.passenger_count >= 7].index, axis='rows')
df_train = df_train.drop(index= df_train[df_train.passenger_count == 0].index, axis='rows')

print('New size: %d' % len(df_train))
