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
import subprocess
import gc


## === cell 1
TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'


## === cell 2
p = subprocess.Popen(['wc', '-l', TRAIN_PATH], stdout=subprocess.PIPE, 
                                               stderr=subprocess.PIPE)
result, err = p.communicate()
if p.returncode != 0:
    raise IOError(err)
n_rows = int(result.strip().split()[0])+1


## === cell 3
def compute_haversine_distance(df, lat1='pickup_latitude', long1='pickup_longitude', lat2='dropoff_latitude', long2='dropoff_longitude'):
    R = 3959 # radius of earth in miles
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])

    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    d = (R * c)
    df["distance"] = d.astype('float32')


## === cell 4
MIN_FARE = 2.50
MAX_FARE = 500

MIN_PASSENGER = 1
MAX_PASSENGER = 6

def clean_data(df, test=False):
    compute_haversine_distance(df)
    add_date_features(df, test)
    
    if not test:
        df.drop(df[df.isnull().any(1)].index, axis = 0, inplace=True)

        df.drop(((df[df.fare_amount>MAX_FARE]) | (df[df.fare_amount<MIN_FARE])).index, axis=0, inplace=True)
        df.drop(df[df.passenger_count > MAX_PASSENGER].index, axis = 0, inplace=True)
        df.drop(df[df.passenger_count < MIN_PASSENGER].index, axis = 0, inplace=True)
        df.drop(((df[df.pickup_latitude>90])    | (df[df.pickup_latitude<-90])    ).index, axis=0, inplace=True)
        df.drop(((df[df.pickup_longitude>180])  | (df[df.pickup_longitude<-180])  ).index, axis=0, inplace=True)
        df.drop(((df[df.dropoff_latitude>90])   | (df[df.dropoff_latitude<-90])   ).index, axis=0, inplace=True)
        df.drop(((df[df.dropoff_longitude>180]) | (df[df.dropoff_longitude<-180]) ).index, axis=0, inplace=True)
    
        df.drop(df[df.distance > 100].index, axis = 0, inplace=True)
        df.drop(df[df.distance <= 0].index, axis = 0, inplace=True)

    if not test:
        df.drop(columns=['pickup_datetime'], inplace=True) 

def add_date_features(df, test=False):
    df["pickup_datetime_clone"] = df["pickup_datetime"].values
    df.pickup_datetime_clone = df.pickup_datetime_clone.str.slice(0, 16)
    df.pickup_datetime_clone = pd.to_datetime(df.pickup_datetime_clone, utc=True, format='%Y-%m-%d %H:%M')
    df['year'] = df.pickup_datetime_clone.dt.year.astype('uint8')
    df['month'] = df.pickup_datetime_clone.dt.month.astype('uint8')
    df['day'] = df.pickup_datetime_clone.dt.day.astype('uint8')
    df['dayofweek'] = df.pickup_datetime_clone.dt.dayofweek.astype('uint8')
    df['hour'] = df.pickup_datetime_clone.dt.hour.astype('uint8')
    df['minute'] = df.pickup_datetime_clone.dt.minute.astype('uint8')
    df.drop(columns=['pickup_datetime_clone'], inplace=True) 


## === cell 5
traintypes = {
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
cols = list(traintypes.keys())
chunksize = 2**21  # 2,097,152
total_chunk = n_rows // chunksize + 1
df_list = []  # list to hold the batch dataframe
i = 0

_original_clean_data = clean_data


def clean_data(df, test=False):
    compute_haversine_distance(df)
    add_date_features(df, test)

    if not test:
        df.drop(df[df.isnull().any(axis=1)].index, axis=0, inplace=True)

        df.drop(
            df[(df.fare_amount > MAX_FARE) | (df.fare_amount < MIN_FARE)].index,
            axis=0,
            inplace=True,
        )
        df.drop(df[df.passenger_count > MAX_PASSENGER].index, axis=0, inplace=True)
        df.drop(df[df.passenger_count < MIN_PASSENGER].index, axis=0, inplace=True)
        df.drop(
            df[(df.pickup_latitude > 90) | (df.pickup_latitude < -90)].index,
            axis=0,
            inplace=True,
        )
        df.drop(
            df[(df.pickup_longitude > 180) | (df.pickup_longitude < -180)].index,
            axis=0,
            inplace=True,
        )
        df.drop(
            df[(df.dropoff_latitude > 90) | (df.dropoff_latitude < -90)].index,
            axis=0,
            inplace=True,
        )
        df.drop(
            df[(df.dropoff_longitude > 180) | (df.dropoff_longitude < -180)].index,
            axis=0,
            inplace=True,
        )

        df.drop(df[df.distance > 100].index, axis=0, inplace=True)
        df.drop(df[df.distance <= 0].index, axis=0, inplace=True)

    if not test:
        df.drop(columns=["pickup_datetime"], inplace=True)


for df_chunk in pd.read_csv(
    TRAIN_PATH, usecols=cols, dtype=traintypes, chunksize=chunksize
):
    i = i + 1
    print(f"DataFrame Chunk {i:02d}/{total_chunk}")
    clean_data(df_chunk)
    df_list.append(df_chunk)
    del df_chunk
    break
print("Complete")


## === cell 6
X = pd.concat(df_list)
del df_list


## === cell 7
minmax = pd.DataFrame()
norm = pd.DataFrame()
float32cols = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "distance"]
float16cols = ["passenger_count", "year", "month", "day", "dayofweek", "hour", "minute"]
for col in float32cols:
    col_min = X[col].min()
    col_max = X[col].max()
    minmax[col] = (col_min, col_max)
    norm[col] = ((X[col] - col_min) / (col_max-col_min)).astype('float32')
    
for col in float16cols:
    col_min = X[col].min()
    col_max = X[col].max()
    minmax[col] = (col_min, col_max)
    norm[col] = ((X[col] - col_min) / (col_max-col_min)).astype('float16')

norm["fare_amount"] = X.fare_amount
X = norm
del norm
print(X[0:5])
X.info()


## === cell 8
X = X.sample(frac=1).reset_index(drop=True)


## === cell 9
gc.collect()


## === cell 10
y = X.fare_amount
X.drop(columns="fare_amount", inplace=True)
X.info()


## === cell 11
validation_portion = 2.5/100
index = int(X.shape[0]*validation_portion)
print("training:\t%d\nvalidation:\t%d" % (n_rows-index, index))


## === cell 12
val_X = X[0:index]
X.drop(X.index[0:index], inplace=True)


## === cell 13
val_y = y[0:index]
y.drop(y.index[0:index], inplace=True)


## === cell 14
import sys
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']
sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


## === cell 15
X.drop(columns="fare_amount", inplace=True, errors="ignore")


## === cell 16
import os
import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==5.28.3"]
)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras import metrics
from keras import backend as K

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass


## === cell 17
model = Sequential()

model.add(Dense(64, input_dim=X.shape[1], activation="relu"))
model.add(Dropout(0.25))

for i in range(5):
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

model.add(Dense(1))

model.compile(loss="mean_squared_error", optimizer="nadam", metrics=["mae"])


## === cell 18
num_epochs = 1
batch_size = 2**10
history = model.fit(X.values, y.values, 
                    validation_data = (val_X, val_y),
                    shuffle=True, 
                    epochs=num_epochs, 
                    batch_size=batch_size)


## === cell 19
plt.figure()
plt.plot(history.history['loss'], color="blue")
plt.plot(history.history['val_loss'], color="red")
plt.legend(['Train', 'Validation'], loc='upper left')
plt.ylabel("loss")
plt.xlabel("epoch")

plt.figure()
plt.plot(history.history['mean_absolute_error'], color="blue")
plt.plot(history.history['val_mean_absolute_error'], color="red")
plt.legend(['Train', 'Validation'], loc='upper left')
plt.ylabel("Mean Abs. Error")
plt.xlabel("epoch")


## --- ERROR in cell 19, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/809960117.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      7[0m [0;34m[0m[0m
[1;32m      8[0m [0mplt[0m[0;34m.[0m[0mfigure[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 9[0;31m [0mplt[0m[0;34m.[0m[0mplot[0m[0;34m([0m[0mhistory[0m[0;34m.[0m[0mhistory[0m[0;34m[[0m[0;34m'mean_absolute_error'[0m[0;34m][0m[0;34m,[0m [0mcolor[0m[0;34m=[0m[0;34m"blue"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     10[0m [0mplt[0m[0;34m.[0m[0mplot[0m[0;34m([0m[0mhistory[0m[0;34m.[0m[0mhistory[0m[0;34m[[0m[0;34m'val_mean_absolute_error'[0m[0;34m][0m[0;34m,[0m [0mcolor[0m[0;34m=[0m[0;34m"red"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m [0mplt[0m[0;34m.[0m[0mlegend[0m[0;34m([0m[0;34m[[0m[0;34m'Train'[0m[0;34m,[0m [0;34m'Validation'[0m[0;34m][0m[0;34m,[0m [0mloc[0m[0;34m=[0m[0;34m'upper left'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mKeyError[0m: 'mean_absolute_error'

## === cell 20
val_pred = model.predict(val_X[0:5]).flatten()
print("actual: "+str(val_y[0:5].values))
print("pred:   "+str(val_pred))
