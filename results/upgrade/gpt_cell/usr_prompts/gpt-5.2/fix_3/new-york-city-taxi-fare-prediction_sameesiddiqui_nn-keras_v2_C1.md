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
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


## === cell 1
datatypes = {'key': 'str', 
              'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

train_df = pd.read_csv('../input/train.csv', nrows=1000000, dtype=datatypes)


## === cell 2
train_df.describe()


## === cell 3
test_df = pd.read_csv('../input/test.csv', dtype=datatypes)
test_df.describe()


## === cell 4
def distance_between_points(df):
    df['diff_lat'] = abs(df['dropoff_latitude'] - df['pickup_latitude'])
    df['diff_long'] = abs(df['dropoff_longitude'] - df['pickup_longitude'])
    df['manhattan_dist'] = df['diff_lat'] + df['diff_long']
    
distance_between_points(train_df)


## === cell 5
def extract_date_details(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC')
    df['year'] = df['pickup_datetime'].apply(lambda date: date.year)
    df['month'] = df['pickup_datetime'].apply(lambda date: date.month)
    df['day'] = df['pickup_datetime'].apply(lambda date: date.weekday())
    df['hour'] = df['pickup_datetime'].apply(lambda date: date.hour)
    
extract_date_details(train_df)
train_df


## === cell 6
def remove_outliers(df):
    df = df.dropna()
    
    df = df[(df['diff_lat'] < 5.0) & (df['diff_long'] < 5.0)]
    df = df[(df['diff_lat'] > .001) & (df['diff_long'] > .001)]
    
    df = df[(df['pickup_longitude'] < -72) & (df['pickup_longitude'] > -75)]
    df = df[(df['pickup_latitude'] < 42) & (df['pickup_latitude'] > 39)]
    df = df[(df['dropoff_longitude'] < -72) & (df['dropoff_longitude'] > -75)]
    df = df[(df['dropoff_latitude'] < 42) & (df['dropoff_latitude'] > 39)]

    df = df[(df['fare_amount'] > 2.50) & (df['fare_amount'] < 200) & (df['passenger_count'] <= 6) & (df['passenger_count'] > 0)] 
    return df
    
train_df = remove_outliers(train_df)
len(train_df)


## === cell 7
plt.scatter(train_df[:10000]['manhattan_dist'], train_df[:10000]['fare_amount'])
plt.xlabel('manhattan distance')
plt.ylabel('fare')
plt.show()


## === cell 8
train_df.describe()


## === cell 9
def convert_to_one_hot (column, num_buckets, df, starting_index = 0):
    df_size = df.shape[0]
    one_hots = np.zeros((df_size, num_buckets), dtype='byte')
    one_hots[np.arange(df_size), df[column].values - starting_index] = 1
    return one_hots


## === cell 10
year = convert_to_one_hot('year', 7, train_df, 2009)
hour = convert_to_one_hot('hour', 24, train_df, 0)


## === cell 11
train_df.shape


## === cell 12
def bucketize_feature(df,column):
    buckets = train_df[column].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]).values
    bins = np.array(df[column].values)
    
    lower_bound = -100000
    for i in range(buckets.shape[0]):
        upper_bound = buckets[i]
        bins[(bins >= lower_bound) & (bins < upper_bound)] = i
        lower_bound = upper_bound
    bins[(bins < 0) | (bins > 8)] = 9
    bins = np.array(bins, dtype='byte')

    return bins
p_long = bucketize_feature(train_df, 'pickup_longitude')
p_lat = bucketize_feature(train_df, 'pickup_latitude')
d_long = bucketize_feature(train_df, 'dropoff_longitude')
d_lat = bucketize_feature(train_df, 'dropoff_latitude')


## === cell 13
print(p_long)
print(p_lat)


## === cell 14
def feature_cross(a1, a2):
    rows = a1.shape[0]
    cols = 100
    cross = np.zeros((rows, cols), dtype='byte')
    cross[np.arange(rows), (a1 * 10) + a2] = 1
    return cross

p_lat_x_long = feature_cross(p_lat, p_long)
d_lat_x_long = feature_cross(d_lat, d_long)


## === cell 15
unique, counts = np.unique(p_long, return_counts=True)
print (np.asarray((unique, counts)).T)
unique, counts = np.unique(p_lat, return_counts=True)
print (np.asarray((unique, counts)).T)


## === cell 16
print (p_lat_x_long.shape)
print (d_lat_x_long.shape)
print (year.shape)
print (hour.shape)
print (train_df['manhattan_dist'].shape)


## === cell 17
manhattan = train_df['manhattan_dist'].values.reshape(len(train_df), 1)

train_X = np.concatenate((p_lat_x_long, d_lat_x_long, year, hour, manhattan), axis=1)
train_y = train_df['fare_amount'].values
print(train_X.shape)
print(train_y.shape)


## === cell 18
validate_df = pd.read_csv('../input/train.csv', skiprows=range(1,1000001), nrows=10000, dtype=datatypes)


## === cell 19
distance_between_points(validate_df)
validate_df = remove_outliers(validate_df)

def extract_features(df):
    extract_date_details(df)
    p_lo = bucketize_feature(df, 'pickup_longitude')
    p_la = bucketize_feature(df, 'pickup_latitude')
    d_lo = bucketize_feature(df, 'dropoff_longitude')
    d_la = bucketize_feature(df, 'dropoff_latitude')
    p_la_x_lo = feature_cross(p_la, p_lo)
    d_la_x_lo = feature_cross(d_la, d_lo)
    yr = convert_to_one_hot('year', 7, df, 2009)
    hr = convert_to_one_hot('hour', 24, df, 0)
    manhattan = df['manhattan_dist'].values.reshape(len(df), 1)

    print (p_la_x_lo.shape)
    print (d_la_x_lo.shape)
    print (yr.shape)
    print (hr.shape)
    print (manhattan.shape)

    X = np.concatenate((p_la_x_lo, d_la_x_lo, yr, hr, manhattan), axis=1)
    return X

X = extract_features(validate_df)
true_y = validate_df['fare_amount'].values


## === cell 20
import os
import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from packaging import version as _version
    import protobuf as _protobuf_pkg  # type: ignore

    _pb_ver = _version.parse(_protobuf_pkg.__version__)
    if _pb_ver.major >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        for _m in list(sys.modules.keys()):
            if _m.startswith("google.protobuf") or _m == "protobuf":
                sys.modules.pop(_m, None)
except Exception:
    pass

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(layers.Dense(128, activation="relu", input_dim=232))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(1))
model.compile(
    optimizer="adam", loss="mse", metrics=["mae"]  # mean squared error
)  # mean absolute error
model.fit(train_X, train_y, epochs=5, batch_size=256)


## --- ERROR in cell 20, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;31mAttributeError[0m: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 21
result = model.predict(X).flatten()
