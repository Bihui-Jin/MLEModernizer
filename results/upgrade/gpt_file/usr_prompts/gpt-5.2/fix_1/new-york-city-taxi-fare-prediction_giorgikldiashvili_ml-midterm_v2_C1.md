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

3.12

# 3. Installed packages

geopandas==0.14.4
geopy==2.4.1
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

3.61245

# 6. Current score

1086.87849

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
train_df =  pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv', nrows = 1_000_000)
test_df =  pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')


## === cell 2
print(train_df)
print(test_df)

missing_values = train_df.isnull()
ans = missing_values.sum()
print(ans)


## === cell 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


## === cell 4
print(train_df.isnull().sum())

train_df.dropna(inplace=True)

train_df = train_df[train_df['passenger_count'] < 8]
train_df = train_df[train_df['fare_amount'] > 0]

print(train_df.isnull().sum())


## === cell 5
min(test_df.pickup_longitude.min(), test_df.dropoff_longitude.min()), \
max(test_df.pickup_longitude.max(), test_df.dropoff_longitude.max())


## === cell 6
min(test_df.pickup_latitude.min(), test_df.dropoff_latitude.min()), \
max(test_df.pickup_latitude.max(), test_df.dropoff_latitude.max())


## === cell 7
RANGE = (-74.26, -72.99, 40.56, 41.71)

def select_within_boundingbox(df, RANGE):
    return (df.pickup_longitude >= RANGE[0]) & (df.pickup_longitude <= RANGE[1]) & \
           (df.pickup_latitude >= RANGE[2]) & (df.pickup_latitude <= RANGE[3]) & \
           (df.dropoff_longitude >= RANGE[0]) & (df.dropoff_longitude <= RANGE[1]) & \
           (df.dropoff_latitude >= RANGE[2]) & (df.dropoff_latitude <= RANGE[3])

print('Old size: %d' % len(train_df))
train_df = train_df[select_within_boundingbox(train_df, RANGE)]
print('New size: %d' % len(train_df))


## === cell 8
from geopy.distance import great_circle

def calculate_distance(df):
    pickup_coords = list(zip(df['pickup_latitude'], df['pickup_longitude']))
    dropoff_coords = list(zip(df['dropoff_latitude'], df['dropoff_longitude']))
    distances = [great_circle(pickup, dropoff).miles for pickup, dropoff in zip(pickup_coords, dropoff_coords)]
    return distances

def add_distance_to_df(df):
    df['distance'] = calculate_distance(df)

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year

add_distance_to_df(train_df)
add_distance_to_df(test_df)


## === cell 9
import matplotlib.pyplot as plt

def plt_distance_to_fare(df, sample_size=100_000):
    df = df.sample(sample_size)
    plt.scatter(df['distance'], df['fare_amount'], s=1)
    plt.title('Distance and Fare Amount')
    plt.xlabel('Distance')
    plt.ylabel('Fare Amount')
    plt.show()

plt_distance_to_fare(train_df)


## === cell 10
nn_df = train_df.copy()

train_df = train_df[train_df['distance'] < 25]
train_df = train_df[train_df['distance'] > 0.1]

plt_distance_to_fare(train_df)


## === cell 11

features = ['passenger_count', 'distance', 'hour', 'day', 'month', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
X = train_df[features]
y = train_df['fare_amount']


## === cell 12
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print("RMSE:", rmse)


## === cell 13
def plot_linear(X_valid, y_valid, y_pred, column='distance'):
    plt.scatter(X_valid[column], y_valid, color='blue', label='Data', s=2)

    plt.plot(X_valid[column], y_pred, color='red', linewidth=1, label='Linear Regression')

    plt.title('Linear Regression Model')
    plt.xlabel('Distance')
    plt.ylabel('Fare Amount')
    plt.legend()
    plt.show()
    
plot_linear(X_valid, y_valid, y_pred)


## === cell 14
X = test_df[features]

y_pred = model.predict(X)

print(y_pred)
submission_df = test_df[["key"]]
submission_df["fare_amount"] = y_pred
print(submission_df)

submission_df.to_csv('linear_submission.csv', index=False)


## === cell 15
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

X = train_df[features]
y = train_df['fare_amount']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(learning_rate=0.1)  # You can specify parameters here if needed
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print("RMSE:", rmse)


## === cell 16
X = test_df[features]

y_pred = model.predict(X)

print(y_pred)
test_df["fare_amount"] = y_pred
submission_df = test_df[["key", "fare_amount"]]
print(submission_df)
submission_df.to_csv('xgboost_submission.csv', index=False)


## === cell 17
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras import backend as K

model = Sequential([
    Dense(32, activation='relu', input_shape=(len(features),)),
    Dense(16, activation='relu'),
    Dense(1)
])
    
model.compile(optimizer='adam', loss="mean_squared_error")


X = nn_df[features]
y = nn_df['fare_amount']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=2, batch_size=32) # ბევრი ეპოქა არც სჭირდება, 1-ის მერეც უკვე მაქსიმუმ პერფორმანსს აღწევს


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 18
loss = model.evaluate(X_valid, y_valid)
print("Test Loss:", loss)

X = test_df[features]
y_pred = model.predict(X)


## === cell 19
test_df["fare_amount"] = y_pred
submission_df = test_df[["key", "fare_amount"]]
print(submission_df)
submission_df.to_csv('nn_submission.csv', index=False)
