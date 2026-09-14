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

3.10

# 2. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
tf_keras==2.18.0
xgboost==2.0.3

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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


## === cell 1
train_df = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv",nrows = 1000000)
test_df = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")


## === cell 2
train_df.isnull().sum()


## === cell 3
train_df.dropna(axis=0, subset=['dropoff_longitude', 'dropoff_latitude'], inplace=True)
train_df = train_df.reset_index(drop=True)


## === cell 4
pd.set_option('display.float_format', lambda x: '%.5f' % x)
train_df.describe()


## === cell 5
print('Number of observations out of valid range in coordinate columns:', end="\n")

print('pickup_longitude', end=': ')
print((train_df.pickup_longitude <-180).sum()+(train_df.pickup_longitude > 180).sum())

print('pickup_latitude', end=': ')
print((train_df.pickup_latitude <-90).sum()+(train_df.pickup_latitude > 90).sum())

print('dropoff_longitude', end=': ')
print((train_df.dropoff_longitude <-180).sum()+(train_df.dropoff_longitude > 180).sum())

print('dropoff_latitude', end=': ')
print((train_df.dropoff_latitude <-90).sum()+(train_df.dropoff_latitude > 90).sum())


## === cell 6
train_df = train_df.drop(train_df[(train_df.pickup_longitude < -180) | (train_df.pickup_longitude > 180)].index, axis=0)
train_df = train_df.drop(train_df[(train_df.pickup_latitude < -90) | (train_df.pickup_latitude > 90)].index, axis=0)
train_df = train_df.drop(train_df[(train_df.dropoff_longitude < -180) | (train_df.dropoff_longitude > 180)].index, axis=0)
train_df = train_df.drop(train_df[(train_df.dropoff_latitude < -90) | (train_df.dropoff_latitude > 90)].index, axis=0)


## === cell 7
train_df.describe()


## === cell 8
train_df[(train_df.pickup_longitude>=40)]


## === cell 9

indx = train_df[(train_df.pickup_longitude>=40)].index
train_df.loc[indx,['dropoff_longitude','dropoff_latitude']] = train_df.loc[indx,['dropoff_latitude','dropoff_longitude']].values
train_df.loc[indx,['pickup_longitude','pickup_latitude']] = train_df.loc[indx,['pickup_latitude','pickup_longitude']].values


## === cell 10


train_df = train_df.drop(train_df[(train_df.pickup_longitude<-75) | (train_df.pickup_longitude>-72)].index, axis=0)
train_df = train_df.drop(train_df[(train_df.dropoff_longitude<-75) | (train_df.dropoff_longitude>-72)].index, axis=0)
train_df = train_df.drop(train_df[(train_df.pickup_latitude<40) | (train_df.pickup_latitude>42)].index, axis=0)
train_df = train_df.drop(train_df[(train_df.dropoff_latitude<40) | (train_df.dropoff_latitude>42)].index, axis=0)


## === cell 11
train_df.describe()


## === cell 12
train_df.passenger_count.value_counts()


## === cell 13
train_df = train_df.drop(train_df[train_df.passenger_count == 0].index, axis=0)


## === cell 14
train_df.fare_amount.sort_values(ascending=False)


## === cell 15
train_df = train_df.drop(train_df[train_df.fare_amount <= 0].index, axis=0)
train_df['fare_amount'].sort_values(ascending=False)


## === cell 16
test_df.isna().sum()


## === cell 17
test_df.describe()


## === cell 18
train_df.dtypes


## === cell 19
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])

test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'])


## === cell 20
def date_splitter(df):
    df['Year'] = df['pickup_datetime'].dt.year
    df['Month'] = df['pickup_datetime'].dt.month
    df['Day'] = df['pickup_datetime'].dt.day
    df['Weekday'] = df['pickup_datetime'].dt.dayofweek
    df['Hour'] = df['pickup_datetime'].dt.hour
    

date_splitter(train_df)
date_splitter(test_df)

train_df.drop(['pickup_datetime'], axis=1, inplace=True)
test_df.drop(['pickup_datetime'], axis=1, inplace=True)


## === cell 21
import math

def haversine_distance(df):
    coord = ['pickup_latitude', 
             'pickup_longitude', 
             'dropoff_latitude', 
             'dropoff_longitude']
    
    phi1, lambda1, phi2, lambda2 = [df[i]*math.pi/180.0 for i in coord]
    
    R = 6371
    
    dPhi = (phi2 - phi1)
    dLambda = (lambda2 - lambda1)
    
    a = np.sin(dPhi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dLambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (R * c)
    
    df['Distance'] = d
    
    
haversine_distance(train_df)
haversine_distance(test_df)


## === cell 22
train_df.Distance.sort_values()


## === cell 23
train_df = train_df.drop(train_df[train_df.Distance<0.5].index, axis=0)


## === cell 24
sns.scatterplot(x='passenger_count', y='fare_amount', data=train_df)


## === cell 25
sns.scatterplot(x='Year', y='fare_amount', data=train_df)


## === cell 26
sns.scatterplot(x='Month', y='fare_amount', data=train_df)


## === cell 27
sns.scatterplot(x='Day', y='fare_amount', data=train_df)


## === cell 28
sns.scatterplot(x='Weekday', y='fare_amount', data=train_df)


## === cell 29
sns.scatterplot(x='Hour', y='fare_amount', data=train_df)


## === cell 30
sns.scatterplot(x='Distance', y='fare_amount', data=train_df)


## === cell 31
features = train_df.columns[2:]
y = train_df.columns[1]


## === cell 32
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


## === cell 33
X_train, X_test, y_train, y_test = train_test_split(train_df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'Year', 'Month', 'Day', 'Weekday', 'Hour', 'Distance']], train_df['fare_amount'], test_size=0.30, random_state=42)


## === cell 43
import os
import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver
except Exception:
    _pb_ver = None

if (
    _pb_ver is None
    or _pb_ver.split(".", 1)[0].isdigit()
    and int(_pb_ver.split(".", 1)[0]) >= 5
):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])
    import importlib

    importlib.invalidate_caches()

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


## === cell 44
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_scalin = scaler.fit_transform(X_train)
val_scalin = scaler.transform(X_test)
test_scalin = scaler.transform(test_df[features])


## === cell 45
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 


## === cell 46
def build_and_compile_model(dim):
    model = keras.Sequential([

      layers.Dense(128, activation='relu', input_dim=dim),
      layers.BatchNormalization(),

      layers.Dense(64, activation='relu'),
      layers.BatchNormalization(),

      layers.Dense(32, activation='relu'),
      layers.BatchNormalization(),

      layers.Dense(8, activation='relu'),
      layers.BatchNormalization(),

      layers.Dense(1)
    ])
    
    model.compile(loss=root_mean_squared_error,
                optimizer=tf.keras.optimizers.Adam(0.01), metrics=['mae'])
    return model


## === cell 47
dnn_model = build_and_compile_model(dim=features.shape[0])


## === cell 48
ep_no = 10
Batch = 128


## === cell 49
%%time
history = dnn_model.fit(
    train_scalin,
    y_train,
    validation_data=(val_scalin, y_test),
    validation_steps=len(val_scalin) // Batch,
    batch_size= Batch,
    epochs=ep_no, verbose=1)


## --- ERROR in cell 49, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m<timed exec>[0m in [0;36m<module>[0;34m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    120[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m             [0;31m# `keras.config.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/282239312.py[0m in [0;36mroot_mean_squared_error[0;34m(y_true, y_pred)[0m
[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m [0;32mdef[0m [0mroot_mean_squared_error[0m[0;34m([0m[0my_true[0m[0;34m,[0m [0my_pred[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m         [0;32mreturn[0m [0mK[0m[0;34m.[0m[0msqrt[0m[0;34m([0m[0mK[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0mK[0m[0;34m.[0m[0msquare[0m[0;34m([0m[0my_pred[0m [0;34m-[0m [0my_true[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mAttributeError[0m: module 'keras.api.backend' has no attribute 'sqrt'

## === cell 51
prediction = dnn_model.predict(test_scalin, batch_size=Batch, verbose=1)
