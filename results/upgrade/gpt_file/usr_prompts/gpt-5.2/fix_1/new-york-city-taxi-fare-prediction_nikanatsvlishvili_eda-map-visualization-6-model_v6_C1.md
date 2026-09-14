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

3.10

# 3. Installed packages

folium==0.20.0
geopandas==0.14.4
haversine==2.9.0
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

22.32348

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from mpl_toolkits import mplot3d
import seaborn as sns

import math
from math import sqrt

from numpy import absolute
from numpy import mean
from numpy import std

from sklearn import metrics
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold 
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn import neighbors


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError 
from keras.layers import BatchNormalization


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv", nrows = 100000, parse_dates=["pickup_datetime"]
                   )
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")


## === cell 3
print(train.shape)
print(test.shape)


## === cell 4
train.head()


## === cell 5
train.dtypes


## === cell 6
train.describe()


## === cell 7
print(train.isnull().sum())


## === cell 8
train = train.dropna(how = 'any', axis = 'rows')


## === cell 9
print('Old size: %d' % len(train))


## === cell 10
train = train.drop(train[train.fare_amount<2.5].index, axis = 0)
train = train.drop(train[train.fare_amount>300].index, axis = 0)


## === cell 11
train = train.drop(train[train['passenger_count']>6].index, axis = 0)
train = train.drop(train[train['passenger_count']<0].index, axis = 0)


## === cell 12
train = train.drop(train[train['pickup_latitude']<-90].index, axis = 0)
train = train.drop(train[train['pickup_latitude']>90].index, axis = 0)


## === cell 13
train = train.drop(train[train['pickup_longitude']<-180].index, axis = 0)
train = train.drop(train[train['pickup_longitude']>180].index, axis = 0)


## === cell 14
train = train.drop(train[train['dropoff_latitude']<-90].index, axis = 0)
train = train.drop(train[train['dropoff_latitude']>90].index, axis = 0)

train = train.drop(train[train['dropoff_longitude']<-180].index, axis = 0)
train = train.drop(train[train['dropoff_longitude']>180].index, axis = 0)


## === cell 15
def select_outside_boundingbox(df, BB):
    filter_df = df.loc[(df['pickup_longitude'] < BB[0]) | (df['pickup_longitude'] > BB[1]) | \
           (df['pickup_latitude'] < BB[2]) | (df['pickup_latitude'] > BB[3]) | \
           (df['dropoff_longitude'] < BB[0]) | (df['dropoff_longitude'] > BB[1]) | \
           (df['dropoff_latitude'] < BB[2]) | (df['dropoff_latitude'] > BB[3])]
    
    return filter_df

NYC_BB = (-74.5, -72.8, 40.5, 41.8)


## === cell 16
outliers = select_outside_boundingbox(train, NYC_BB)
outliers


## === cell 17
train = train.drop(outliers.index, axis = 0)


## === cell 18
print('New size: %d' % len(train))


## === cell 20
test.dtypes


## === cell 21
train["loc1"] = train[["pickup_latitude","pickup_longitude"]].apply(tuple, axis=1)
train["loc2"] = train[["dropoff_latitude","dropoff_longitude"]].apply(tuple, axis=1)


## === cell 22
import haversine as hs

        
train['H_Distance'] = train.apply(lambda row: hs.haversine(row.loc1,row.loc2), axis=1)


## === cell 23
def chebyshev(pickup_long, dropoff_long, pickup_lat, dropoff_lat):
    return np.maximum(np.absolute(pickup_long - dropoff_long), np.absolute(pickup_lat - dropoff_lat))

train['Chebyshev'] = chebyshev(train['pickup_longitude'], train['dropoff_longitude'], train['pickup_latitude'], train['dropoff_latitude'])


## === cell 24
train.head()


## === cell 25
train["hour"] = train.pickup_datetime.dt.hour
train["day_of_week"] = train.pickup_datetime.dt.weekday
train["day_of_month"] = train.pickup_datetime.dt.day
train["week"] = train.pickup_datetime.dt.week
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year - 2000

train['minute'] =train['pickup_datetime'].dt.minute
train['second'] = train['pickup_datetime'].dt.second
train['dayofyear'] = train['pickup_datetime'].dt.dayofyear


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1606786005.py in <cell line: 0>()
      2 train["day_of_week"] = train.pickup_datetime.dt.weekday
      3 train["day_of_month"] = train.pickup_datetime.dt.day
----> 4 train["week"] = train.pickup_datetime.dt.week
      5 train["month"] = train.pickup_datetime.dt.month
      6 train["year"] = train.pickup_datetime.dt.year - 2000

AttributeError: 'DatetimeProperties' object has no attribute 'week'

## === cell 26
train.head()


## === cell 27
train["fare_to_dist_ratio"] = train["fare_amount"] / ( train["H_Distance"]+0.0001)


## === cell 28
train = train.drop(train[train['loc1']==train['loc2']].index, axis = 0)


## === cell 29
def add_distances_from_airport(dataset):
    jfk_coords = (40.639722, -73.778889)
    ewr_coords = (40.6925, -74.168611)
    lga_coords = (40.77725, -73.872611)

    dataset['pickup_jfk_distance'] = dataset.apply(lambda row: hs.haversine(jfk_coords,row.loc1), axis=1)
    dataset['dropof_jfk_distance'] = dataset.apply(lambda row: hs.haversine(jfk_coords,row.loc2), axis=1)
    
    
    dataset['pickup_ewr_distance'] = dataset.apply(lambda row: hs.haversine(ewr_coords,row.loc1), axis=1)
    dataset['dropof_ewr_distance'] = dataset.apply(lambda row: hs.haversine(ewr_coords,row.loc2), axis=1)
    
    
    dataset['pickup_lga_distance'] = dataset.apply(lambda row: hs.haversine(lga_coords,row.loc1), axis=1)
    dataset['dropof_lga_distance'] = dataset.apply(lambda row: hs.haversine(lga_coords,row.loc2), axis=1)

    return dataset


train = add_distances_from_airport(train)


## === cell 30
test['pickup_datetime']  = pd.to_datetime(test['pickup_datetime'])

test["loc1"] = test[["pickup_latitude","pickup_longitude"]].apply(tuple, axis=1)
test["loc2"] = test[["dropoff_latitude","dropoff_longitude"]].apply(tuple, axis=1)

test['H_Distance'] = test.apply(lambda row: hs.haversine(row.loc1,row.loc2), axis=1)

test['Chebyshev'] = chebyshev(test['pickup_longitude'], test['dropoff_longitude'], test['pickup_latitude'], test['dropoff_latitude'])


test["hour"] = test.pickup_datetime.dt.hour
test["day_of_week"] = test.pickup_datetime.dt.weekday
test["day_of_month"] = test.pickup_datetime.dt.day
test["week"] = test.pickup_datetime.dt.week
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year - 2000

test['minute'] =test['pickup_datetime'].dt.minute
test['second'] = test['pickup_datetime'].dt.second
test['dayofyear'] = test['pickup_datetime'].dt.dayofyear

test = add_distances_from_airport(test)


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1534197004.py in <cell line: 0>()
     12 test["day_of_week"] = test.pickup_datetime.dt.weekday
     13 test["day_of_month"] = test.pickup_datetime.dt.day
---> 14 test["week"] = test.pickup_datetime.dt.week
     15 test["month"] = test.pickup_datetime.dt.month
     16 test["year"] = test.pickup_datetime.dt.year - 2000

AttributeError: 'DatetimeProperties' object has no attribute 'week'

## === cell 31
def downcast(df):

    df_int = df.select_dtypes(include=['int64', 'int32', 'int16', 'int8', 'int'])
    df[df_int.columns] = df_int.apply(pd.to_numeric,downcast='unsigned')
    
    df_float = df.select_dtypes(include=['float64', 'float32', 'float16', 'float'])
    df[df_float.columns] = df_float.apply(pd.to_numeric,downcast='float')
        
    return df
downcast(train)
downcast(test)
train.dtypes


## === cell 32
train.plot(y='pickup_latitude',x='pickup_longitude',kind="scatter",alpha=0.7,s=0.02)
city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
plt.title("Pickups Data")

plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()


## === cell 33
train.plot(y='dropoff_latitude',x='dropoff_longitude',kind="scatter",alpha=0.5,s=0.02)
city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
plt.title("Dropoff Data")

plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()


## === cell 34
import folium


long_trips=train[train['H_Distance']>=10]

drop_map = folium.Map(location = [40.730610,-73.935242],zoom_start = 12)


for index, row in long_trips.iterrows():
    
    folium.CircleMarker([row['dropoff_latitude'], row['dropoff_longitude']],
                        radius=3,
                        color="green", 
                        fill_opacity=0.9
                       ).add_to(drop_map)
for index, row in long_trips.iterrows():
    
    folium.CircleMarker([row['pickup_latitude'], row['pickup_longitude']],
                        radius=3,
                        color="blue", 
                        fill_opacity=0.9
                       ).add_to(drop_map)
drop_map
