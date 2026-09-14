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

26.87094

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}
cols = list(traintypes.keys())
chunksize = 2**21 # 2,097,152
total_chunk = n_rows // chunksize + 1
df_list = [] # list to hold the batch dataframe
i=0

for df_chunk in pd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes, chunksize=chunksize):    
    i = i+1
    print(f'DataFrame Chunk {i:02d}/{total_chunk}')
    clean_data(df_chunk)
    df_list.append(df_chunk)
    del df_chunk
    break
print("Complete")


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2878170775.py in <cell line: 0>()
     16     # Each chunk is a corresponding dataframe
     17     print(f'DataFrame Chunk {i:02d}/{total_chunk}')
---> 18     clean_data(df_chunk)
     19     # Alternatively, append the chunk to list and merge all
     20     df_list.append(df_chunk)

/tmp/ipykernel_11/2837320428.py in clean_data(df, test)
     12         # 1.
     13         # we have so much data, we can afford to just remove the nulls
---> 14         df.drop(df[df.isnull().any(1)].index, axis = 0, inplace=True)
     15         # 2.
     16 

TypeError: DataFrame.any() takes 1 positional argument but 2 were given

## === cell 6
X = pd.concat(df_list)
del df_list


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2756518164.py in <cell line: 0>()
      1 # Merge all dataframes into one dataframe
----> 2 X = pd.concat(df_list)
      3 del df_list

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/concat.py in concat(objs, axis, join, ignore_index, keys, levels, names, verify_integrity, sort, copy)
    380         copy = False
    381 
--> 382     op = _Concatenator(
    383         objs,
    384         axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/concat.py in __init__(self, objs, axis, join, keys, levels, names, ignore_index, verify_integrity, copy, sort)
    443         self.copy = copy
    444 
--> 445         objs, keys = self._clean_keys_and_objs(objs, keys)
    446 
    447         # figure out what our result ndim is going to be

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/concat.py in _clean_keys_and_objs(self, objs, keys)
    505 
    506         if len(objs_list) == 0:
--> 507             raise ValueError("No objects to concatenate")
    508 
    509         if keys is None:

ValueError: No objects to concatenate

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


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/819426067.py in <cell line: 0>()
      6 float16cols = ["passenger_count", "year", "month", "day", "dayofweek", "hour", "minute"]
      7 for col in float32cols:
----> 8     col_min = X[col].min()
      9     col_max = X[col].max()
     10     minmax[col] = (col_min, col_max)

NameError: name 'X' is not defined

## === cell 8
X = X.sample(frac=1).reset_index(drop=True)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1504734925.py in <cell line: 0>()
      1 # 1. Shuffling
----> 2 X = X.sample(frac=1).reset_index(drop=True)

NameError: name 'X' is not defined

## === cell 9
gc.collect()


## === cell 10
y = X.fare_amount
X.drop(columns="fare_amount", inplace=True)
X.info()


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/24282127.py in <cell line: 0>()
      1 # 2. take out the answers
----> 2 y = X.fare_amount
      3 X.drop(columns="fare_amount", inplace=True)
      4 X.info()

NameError: name 'X' is not defined

## === cell 11
validation_portion = 2.5/100
index = int(X.shape[0]*validation_portion)
print("training:\t%d\nvalidation:\t%d" % (n_rows-index, index))


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/362932902.py in <cell line: 0>()
      1 # 3. Splitting off a validation set
      2 validation_portion = 2.5/100
----> 3 index = int(X.shape[0]*validation_portion)
      4 print("training:\t%d\nvalidation:\t%d" % (n_rows-index, index))

NameError: name 'X' is not defined

## === cell 12
val_X = X[0:index]
X.drop(X.index[0:index], inplace=True)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1000326277.py in <cell line: 0>()
----> 1 val_X = X[0:index]
      2 X.drop(X.index[0:index], inplace=True)

NameError: name 'X' is not defined

## === cell 13
val_y = y[0:index]
y.drop(y.index[0:index], inplace=True)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4168837963.py in <cell line: 0>()
----> 1 val_y = y[0:index]
      2 y.drop(y.index[0:index], inplace=True)

NameError: name 'y' is not defined

## === cell 14
import sys
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']
sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


## === cell 15
X.drop(columns="fare_amount", inplace=True)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1594576676.py in <cell line: 0>()
      1 # gotta do this again because sometimes it comes back when we drop inplace then garbage collect
----> 2 X.drop(columns="fare_amount", inplace=True)

NameError: name 'X' is not defined

## === cell 16
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras import metrics
from keras import backend as K
K.set_image_dim_ordering('tf')

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 17
model = Sequential()

model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model.add(Dropout(0.25))

for i in range(5):
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))


model.add(Dense(1))

model.compile(loss='mean_squared_error',
              optimizer='nadam', 
              metrics=[metrics.mae])


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2242636776.py in <cell line: 0>()
      1 model = Sequential()
      2 
----> 3 model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
      4 model.add(Dropout(0.25))
      5 

NameError: name 'X' is not defined

## === cell 18
num_epochs = 1
batch_size = 2**10
history = model.fit(X.values, y.values, 
                    validation_data = (val_X, val_y),
                    shuffle=True, 
                    epochs=num_epochs, 
                    batch_size=batch_size)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2753237750.py in <cell line: 0>()
      1 num_epochs = 1
      2 batch_size = 2**10
----> 3 history = model.fit(X.values, y.values, 
      4                     validation_data = (val_X, val_y),
      5                     shuffle=True,

NameError: name 'X' is not defined

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
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/809960117.py in <cell line: 0>()
      1 plt.figure()
----> 2 plt.plot(history.history['loss'], color="blue")
      3 plt.plot(history.history['val_loss'], color="red")
      4 plt.legend(['Train', 'Validation'], loc='upper left')
      5 plt.ylabel("loss")

NameError: name 'history' is not defined

## === cell 20
val_pred = model.predict(val_X[0:5]).flatten()
print("actual: "+str(val_y[0:5].values))
print("pred:   "+str(val_pred))


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/524543933.py in <cell line: 0>()
----> 1 val_pred = model.predict(val_X[0:5]).flatten()
      2 print("actual: "+str(val_y[0:5].values))
      3 print("pred:   "+str(val_pred))

NameError: name 'val_X' is not defined

## === cell 21
traintypes = {'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}
cols = list(traintypes.keys())
cols.append('key')

X_test = pd.read_csv(TEST_PATH, usecols=cols, dtype=traintypes)
clean_data(X_test, test=True)
X_test_key = X_test['key']
X_test.drop(columns=['pickup_datetime'], inplace=True)
X_test.drop(columns=['key'], inplace=True)

float32cols = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "distance"]
float16cols = ["passenger_count", "year", "month", "day", "dayofweek", "hour", "minute"]

for col in float16cols:
    col_min, col_max = minmax[col]
    X_test[col] = ((X_test[col] - col_min) / (col_max-col_min)).astype('float16')

for col in float32cols:
    col_min, col_max = minmax[col]
    X_test[col] = ((X_test[col] - col_min) / (col_max-col_min)).astype('float32')


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2646298234.py in <cell line: 0>()
     19 
     20 for col in float16cols:
---> 21     col_min, col_max = minmax[col]
     22     X_test[col] = ((X_test[col] - col_min) / (col_max-col_min)).astype('float16')
     23 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    415                 raise KeyError(key) from err
    416         if isinstance(key, Hashable):
--> 417             raise KeyError(key)
    418         self._check_indexing_error(key)
    419         raise KeyError(key)

KeyError: 'passenger_count'

## === cell 22
pred = model.predict(X_test).flatten()
pred = np.round(pred,2)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2143493999.py in <cell line: 0>()
----> 1 pred = model.predict(X_test).flatten()
      2 pred = np.round(pred,2)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in build(self, input_shape)
    162             return
    163         if not self._layers:
--> 164             raise ValueError(
    165                 f"Sequential model {self.name} cannot be built because it has "
    166                 "no layers. Call `model.add(layer)`."

ValueError: Sequential model sequential cannot be built because it has no layers. Call `model.add(layer)`.

## === cell 23
results = pd.DataFrame({'key': X_test_key, 'fare_amount': pred})
results.key = results.key.astype(str)
results.info()
results.to_csv('submission.csv', index=False)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1516778705.py in <cell line: 0>()
----> 1 results = pd.DataFrame({'key': X_test_key, 'fare_amount': pred})
      2 results.key = results.key.astype(str)
      3 results.info()
      4 # print(len(results)) # should be 9914
      5 # print(results[0:5])

NameError: name 'pred' is not defined
