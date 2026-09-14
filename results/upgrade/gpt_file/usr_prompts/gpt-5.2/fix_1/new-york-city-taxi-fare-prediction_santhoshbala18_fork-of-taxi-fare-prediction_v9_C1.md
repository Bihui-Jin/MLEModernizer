# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
sklearn-pandas==2.2.0
xgboost==2.0.3

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

3.35291

# 6. Current score

8.96469

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

import os
print(os.listdir("../input"))


import random
random.seed(113)


## === cell 1
'''df_list = [] # list to hold the batch dataframe
chunksize = 10_000_000 # 10 million rows at one go. 
i=0
for df_chunk in pd.read_csv('../input/train.csv', chunksize=chunksize):
    
    i = i+1
    print(f'DataFrame Chunk {i}')
    df_list.append(df_chunk) 
    if i == 4:
        break
df = pd.concat(df_list)
del df_list
    
# save both training and test data to feather format
os.makedirs('tmp', exist_ok=True)
df.to_feather('tmp/taxi-train.feather')
test_set = pd.read_csv("../input/test.csv")
test_set.to_feather('tmp/taxi-test.feather')
df = pd.read_feather('tmp/taxi-train.feather')
test_set = pd.read_feather('tmp/taxi-test.feather')
'''
df = pd.read_csv('../input/train.csv',nrows =10**6 )
test_set = pd.read_csv("../input/test.csv")


## === cell 2
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
    
    
df['distance_km'] = distance(df.pickup_latitude, df.pickup_longitude, \
                                   df.dropoff_latitude, df.dropoff_longitude)

test_set['distance_km'] = distance(test_set.pickup_latitude, test_set.pickup_longitude, \
                                   test_set.dropoff_latitude, test_set.dropoff_longitude)
add_travel_vector_features(df)
add_travel_vector_features(test_set)


## === cell 3
BB = (-75, -73, 40, 41.5)

def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
           (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
           (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
           (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3]) 

print('Old size: %d' % len(df))
df = df[select_within_boundingbox(df, BB)]
df = df[(df.passenger_count > 0) & (df.passenger_count <= 6)]
print('New size: %d' % len(df))


## === cell 4
def add_datetime_info(dataset):
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'])
    
    dataset['hour'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['weekday'] = dataset.pickup_datetime.dt.weekday
    dataset['year'] = dataset.pickup_datetime.dt.year
    
    return dataset

df = add_datetime_info(df)
test_set = add_datetime_info(test_set)


## === cell 5
df['is_night'] = np.where((((df['hour']>=20) & (df['hour']<=23)) | ((df['hour']>=0) & (df['hour']<6))),1,0) 
df['is_airport'] = np.where((((df['dropoff_longitude']>=73.77) & (df['dropoff_longitude']<=73.78)) | ((df['dropoff_latitude']>=40.63) & (df['dropoff_latitude']<=40.64))),1,0) 
df['is_surge']= np.where((((df['hour']>=16) & (df['hour']<20)) & ((df['weekday']!=5) & (df['weekday']!=6))),1,0)


## === cell 6
df = df.drop(df[df['fare_amount']<0].index,axis=0)


## === cell 7
from sklearn.model_selection import train_test_split
train = df.drop(['key','fare_amount','pickup_datetime','hour','day','month','weekday'],axis=1)
test = df['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size = 0.1)


## === cell 8
y_train.mean()


## === cell 9
from xgboost import XGBRegressor
regressor = XGBRegressor(max_depth =10, learning_rate=0.1, n_estimators=200)
regressor.fit(X_train, y_train)


## === cell 10
predictions = regressor.predict(X_test)


## === cell 11
import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(y_test,predictions))

print(rmse)


## === cell 12
X_train = X_train.drop(['abs_diff_longitude', 'abs_diff_latitude', 'year','is_airport'],axis=1)
X_test = X_test.drop(['abs_diff_longitude', 'abs_diff_latitude', 'year','is_airport'],axis=1)


## === cell 13
X_train.columns


## === cell 14
'''from keras import Sequential
from keras import layers
from keras.layers import Dense,Dropout
from keras.callbacks import ModelCheckpoint,  ReduceLROnPlateau
from keras import optimizers
from keras.regularizers import l2

neuralNetwork = Sequential()
neuralNetwork.add(Dense(units=1048,input_shape=(8,),activation='relu',kernel_regularizer = l2(1e-2)))
neuralNetwork.add(Dropout(0.5))
neuralNetwork.add(Dense(units=10,activation='relu',kernel_regularizer = l2(1e-2)))
neuralNetwork.add(Dropout(0.5))
neuralNetwork.add(Dense(units=1,activation='relu',kernel_regularizer = l2(1e-2)))

neuralNetwork.compile(optimizer='Adam', 
              loss='mean_squared_error')

filepath = './model_weights/weights-improvement-10M.hdf5'
best_callback = ModelCheckpoint(filepath, 
                                save_best_only=True)

#lr_sched = ReduceLROnPlateau(monitor='loss', factor = 0.2, patience = 10, verbose = 1)

history = neuralNetwork.fit(X_train, y_train, 
          epochs=20,
          verbose=0,
          batch_size=2048)

y_pred = neuralNetwork.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test,y_pred))

print(rmse)
'''


## === cell 15
test_set['is_night'] = np.where((((test_set['hour']>=20) & (test_set['hour']<=23)) | ((test_set['hour']>=0) & (test_set['hour']<6))),1,0) 
test_set['is_airport'] = np.where((((test_set['dropoff_longitude']>=73.77) & (test_set['dropoff_longitude']<=73.78)) | ((test_set['dropoff_latitude']>=40.63) & (test_set['dropoff_latitude']<=40.64))),1,0) 
test_set['is_surge']= np.where((((test_set['hour']>=16) & (test_set['hour']<20)) & ((test_set['weekday']!=5) & (test_set['weekday']!=6))),1,0)
test_set['is_BeforeRaise']= np.where(((test_set['year']>=2012) & (test_set['month']>=8)),1,0)

test_set_features = test_set.drop(['key','pickup_datetime','hour','day','month','weekday','is_BeforeRaise'],axis=1)

test_set_key = test_set['key']

y_pred_reg = regressor.predict(test_set_features)


## === cell 16
'''test_set_features = test_set_features.drop(['abs_diff_longitude', 'abs_diff_latitude', 'year','is_airport'],axis=1)
y_pred_neuralNet = neuralNetwork.predict(test_set_features)
y_pred_neuralNet = y_pred_neuralNet.reshape(9914,)
y_pred_final = (y_pred_reg+y_pred_neuralNet)/2'''


## === cell 17
submission = pd.DataFrame(
    {'key': test_set_key, 'fare_amount': y_pred_reg},
    columns = ['key', 'fare_amount'])
submission = submission.round({'fare_amount':2})
submission.to_csv('submission_combo.csv', index = False)
print(submission.head())
print("Submitted")


## === cell 18
submission.shape
