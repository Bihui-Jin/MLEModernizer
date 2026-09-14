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
ipywidgets==8.1.5
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
tqdm==4.67.1

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
plt.rcParams['figure.figsize'] = 16,9
from tqdm import tqdm
from ipywidgets import widgets


## === cell 1
def read_csv_sampled(path, frac, chunksize=10**5, random_state=None):
    samples = []

    for df_chunk in tqdm(pd.read_csv(path, chunksize=chunksize)):
        samples.append(df_chunk.sample(frac=frac, random_state=random_state))

    df = pd.concat(samples, ignore_index=True)
    df['pickup_datetime'] = df['pickup_datetime'].apply(pd.Timestamp)
    
    return df


## === cell 2
df_train_sample = read_csv_sampled('../input/train.csv', 0.01, random_state=1989)


## === cell 3
df_train_sample.info()


## === cell 4
df_test = pd.read_csv('../input/test.csv', parse_dates=['pickup_datetime'])
df_test.info()


## === cell 5
df_train_sample.dropna(inplace=True)


## === cell 6
is_weird = (df_train_sample['fare_amount'] < 0)
is_weird |= ~df_train_sample['pickup_latitude'].between(40, 42)
is_weird |= ~df_train_sample['pickup_longitude'].between(-75, -72)
is_weird |= ~df_train_sample['dropoff_latitude'].between(40, 42)
is_weird |= ~df_train_sample['dropoff_longitude'].between(-75, -72)
is_weird |= (df_train_sample['passenger_count'] == 0)
print(is_weird.sum())

df_train_sample = df_train_sample[~is_weird]


## === cell 7
df_train_sample.describe()


## === cell 8
df_test.describe()


## === cell 9
sns.distplot(df_train_sample['fare_amount'].apply(np.log1p).dropna())


## === cell 10
plt.scatter(df_train_sample['pickup_longitude'], df_train_sample['pickup_latitude'], c=df_train_sample['fare_amount'].apply(np.log1p), alpha=0.7, s=1, lw=0)
plt.xlim(*np.percentile(df_train_sample['pickup_longitude'], [1, 99]))
plt.ylim(*np.percentile(df_train_sample['pickup_latitude'], [1, 99]))
plt.colorbar()


## === cell 11
def prep_data(df, shuffle=False):
    X_cat = np.vstack([
        df['pickup_datetime'].dt.hour, # 0-23
        df['pickup_datetime'].dt.weekday + 24, # 24-30
        df['pickup_datetime'].dt.dayofyear + 30, # 31-396,
        df['pickup_datetime'].dt.weekofyear + 396, # 397-449,
        df['pickup_datetime'].dt.year - 2009 + 450, # 450-456
        df['passenger_count'] + 456 # 457-463
    ]).T
    
    X_deg = df[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']].values
    X_deg /= 180
    
    if 'fare_amount' in df.columns:
        y = df['fare_amount'].values
    
    if shuffle:
        rnd_ind = np.random.permutation(len(df))
        X_cat = X_cat[rnd_ind]
        X_deg = X_deg[rnd_ind]
        
        if 'fare_amount' in df.columns:
            y = y[rnd_ind]
        
    if 'fare_amount' in df.columns: 
        return (X_cat, X_deg), y

    return (X_cat, X_deg)


## === cell 12
def prep_data(df, shuffle=False):
    weekofyear = df["pickup_datetime"].dt.isocalendar().week.astype(int).to_numpy()

    X_cat = np.vstack(
        [
            df["pickup_datetime"].dt.hour,  # 0-23
            df["pickup_datetime"].dt.weekday + 24,  # 24-30
            df["pickup_datetime"].dt.dayofyear + 30,  # 31-396,
            weekofyear + 396,  # 397-449,
            df["pickup_datetime"].dt.year - 2009 + 450,  # 450-456
            df["passenger_count"] + 456,  # 457-463
        ]
    ).T

    X_deg = df[
        ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    ].values
    X_deg /= 180

    if "fare_amount" in df.columns:
        y = df["fare_amount"].values

    if shuffle:
        rnd_ind = np.random.permutation(len(df))
        X_cat = X_cat[rnd_ind]
        X_deg = X_deg[rnd_ind]

        if "fare_amount" in df.columns:
            y = y[rnd_ind]

    if "fare_amount" in df.columns:
        return (X_cat, X_deg), y

    return (X_cat, X_deg)


## === cell 13
import os
import sys
import importlib

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import keras.layers as lyr
import keras.activations as act
from keras.models import Model
import keras.backend as K
import tensorflow as tf


## --- ERROR in cell 13, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;31mAttributeError[0m: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 14
def K_haversine_bearing(x):
    R = 6371e3
    
    x_rad = x * np.pi
    
    lat1, lng1, lat2, lng2 = x_rad[:,0], x_rad[:,1], x_rad[:,2], x_rad[:,3]
    
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    
    a = K.sin(dlat/2) * K.sin(dlat/2) + K.cos(lat1) * K.cos(lat2) * K.sin(dlng/2) * K.sin(dlng/2);
    c = 2 * tf.atan2(K.sqrt(a), K.sqrt(1-a))
    
    d = K.log((R * c) + 1)
    
    x = K.sin(dlng) * K.cos(lat2)
    y = (K.cos(lat1) * K.sin(lat2)) - (K.sin(lat1) * K.cos(lat2) * K.cos(dlng))
    b = tf.atan2(x, y) / np.pi
    
    return K.concatenate([K.reshape(d, (-1, 1)), K.reshape(b, (-1, 1))])
