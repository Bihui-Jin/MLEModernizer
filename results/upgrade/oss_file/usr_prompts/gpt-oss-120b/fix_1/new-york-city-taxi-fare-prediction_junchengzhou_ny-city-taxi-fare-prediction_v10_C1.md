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

4.33523

# 6. Current score

60.67916

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from math import radians, cos, sin, asin, sqrt
import warnings
warnings.filterwarnings('ignore')


## === cell 2
train = pd.read_csv('../input/train.csv',nrows=10_000_000)


## === cell 3
train = train.loc[(train['fare_amount'] > 0) & (train['fare_amount'] < 200)]
train = train.loc[(train['pickup_longitude'] > -150) & (train['pickup_longitude'] < 0)]
train = train.loc[(train['pickup_latitude'] > 0) & (train['pickup_latitude'] < 80)]
train = train.loc[(train['dropoff_longitude'] > -150) & (train['dropoff_longitude'] < 0)]
train = train.loc[(train['dropoff_latitude'] > 0) & (train['dropoff_longitude'] < 80)]
train = train.loc[train['passenger_count'] <= 8]


## === cell 4
train.head()


## === cell 5
test = pd.read_csv('../input/test.csv')


## === cell 6
test.head()


## === cell 7
train.dtypes


## === cell 8
def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1 
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # 地球平均半径，单位为公里
    return c * r


## === cell 9
def add_travel_distance_vector_features(df):
    df['distance'] = haversine(df.dropoff_longitude,df.dropoff_latitude,df.pickup_longitude,df.pickup_latitude)
add_travel_distance_vector_features(train)
add_travel_distance_vector_features(test)


## === cell 10
train.dtypes


## === cell 11
train.drop(['dropoff_longitude','dropoff_latitude','pickup_longitude','pickup_latitude','pickup_datetime'],axis=1,inplace=True)
test.drop(['dropoff_longitude','dropoff_latitude','pickup_longitude','pickup_latitude','pickup_datetime'],axis=1,inplace=True)


## === cell 13
train.head()


## === cell 14
train.isnull().sum()


## === cell 15
train.dropna(how = 'any', axis = 'rows', inplace=True)


## === cell 16
train.describe().astype('float16')


## === cell 17
sns.kdeplot(train.fare_amount, shade=True)


## === cell 18
train = train.loc[train['fare_amount'] <= 100]


## === cell 19
sns.kdeplot(train.fare_amount, shade=True)


## === cell 20
train.key = pd.to_datetime(train.key).values.astype(np.int64)
test.key = pd.to_datetime(test.key).values.astype(np.int64)


## === cell 21
train.head()


## === cell 22
train.distance.describe().astype('float16')


## === cell 23
y = train.pop('fare_amount')
X = train


## === cell 24
from sklearn.preprocessing import StandardScaler


## === cell 25
scaler = StandardScaler()


## === cell 26
X = scaler.fit_transform(X)
test = scaler.transform(test)


## === cell 27
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)


## === cell 28
y_train = y_train.round(0).astype('int8')


## === cell 29
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_train=np_utils.to_categorical(y_train, 101)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 30
from keras.layers import Dense, Input,Dropout,LeakyReLU,LSTM
from keras.models import Model
from keras import backend as Backend


## === cell 31
def rmse(y_true, y_pred):
    return Backend.sqrt(Backend.square(y_pred - y_true))


## === cell 32
def nn(n_feature,k=1200):
    model_in = Input(shape=(n_feature,))
    model = Dense(k)(model_in)
    model = LeakyReLU(0.15)(model)
    model = Dropout(0.2)(model)
    
    model = Dense(k)(model)
    model = LeakyReLU(0.15)(model)
    model = Dropout(0.2)(model)
    
    model = Dense(k)(model)
    model = LeakyReLU(0.15)(model)
    model = Dropout(0.2)(model)
    
    model = Dense(k)(model)
    model = LeakyReLU(0.15)(model)
    model = Dropout(0.2)(model)
    
    model = Dense(101,activation='softmax')(model)
    
    model = Model(inputs=model_in,outputs=model)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=[rmse])
    return model


## === cell 34
def lstm(n):
    model_in = Input(shape=(1,n))
    model = LSTM(10)(model_in)
    model = Dense(1,activation='linear')(model)
    model = Model(model_in,model)
    model.compile(loss='mse', optimizer='adam',metrics=[rmse])
    return model


## === cell 35
model = nn(X.shape[1])


## === cell 36
history = model.fit(X_train,y_train,batch_size=10000,epochs=2,verbose=1,validation_data=(X_train,y_train))


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4245281704.py in <cell line: 0>()
      1 # history = model.fit(X_train,y_train,batch_size=1000,epochs=10,verbose=1)
----> 2 history = model.fit(X_train,y_train,batch_size=10000,epochs=2,verbose=1,validation_data=(X_train,y_train))

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/nn.py in categorical_crossentropy(target, output, from_logits, axis)
    658     for e1, e2 in zip(target.shape, output.shape):
    659         if e1 is not None and e2 is not None and e1 != e2:
--> 660             raise ValueError(
    661                 "Arguments `target` and `output` must have the same shape. "
    662                 "Received: "

ValueError: Arguments `target` and `output` must have the same shape. Received: target.shape=(None, 1), output.shape=(None, 101)

## === cell 37
plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.title('model rmse')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1667325992.py in <cell line: 0>()
----> 1 plt.plot(history.history['rmse'])
      2 plt.plot(history.history['val_rmse'])
      3 plt.title('model rmse')
      4 plt.ylabel('rmse')
      5 plt.xlabel('epoch')

NameError: name 'history' is not defined

## === cell 38
pres = model.predict(test)


## === cell 39
pres = pres.argmax(axis=-1)


## === cell 40
test = pd.read_csv('../input/test.csv')


## === cell 41
submission = pd.DataFrame(
    {'key': test.key, 'fare_amount': pres.reshape(9914)},
    columns = ['key', 'fare_amount'])


## === cell 42
submission.to_csv('submission.csv', index = False)


## === cell 43
print(os.listdir('.'))


## === cell 44
print(submission.key+','+*(submission.fare_amount), sep='\n')


## --- ERROR in cell 44, traceback:
  File "/tmp/ipykernel_11/1128071884.py", line 1
    print(submission.key+','+*(submission.fare_amount), sep='\n')
                             ^
SyntaxError: invalid syntax
