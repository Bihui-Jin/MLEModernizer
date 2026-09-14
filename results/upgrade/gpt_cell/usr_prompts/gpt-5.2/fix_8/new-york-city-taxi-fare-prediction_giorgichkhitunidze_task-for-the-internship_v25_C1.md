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

3.10

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

3.41747

# 6. Current score

315.10391

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 586.20508) has done: 'Diagnosis: The crash happens during the `tensorflow` import in cell 0, before any data is loaded. With TensorFlow 2.18.0 and protobuf 6.33.0, an incompatibility can surface where TensorFlow (or one of its bundled proto-generated modules) expects `MessageFactory.GetPrototype`, which was removed/changed in newer protobuf versions, causing `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is an import-time failure, so the only viable fix is to patch the protobuf API for compatibility before importing TensorFlow.

Patch summary: In cell 0, add a small compatibility shim that defines `GetPrototype` as an alias to `GetMessageClass` on `google.protobuf.message_factory.MessageFactory` when missing. This keeps TensorFlow’s expected API available without changing any modeling/training logic, and it unblocks the imports so later cells run unchanged.

Updated cells: Only cell 0 is modified (the failing cell), and the rest of the imports remain the same.

Compatibility notes for cell k+1: Cell 1 depends on `pd` and `np` imports from cell 0; these remain unchanged. No variable names or interfaces used by cell 1 are altered.

Assumptions: `google.protobuf` is available (it is, via the installed `protobuf==6.33.0`), and TensorFlow’s import error is due specifically to the missing `MessageFactory.GetPrototype` attribute.'
- What this solution (achieved 392.9367) has done: 'Diagnosis: The crash happens during the protobuf compatibility shim in cell 0: it tries to access `_message_factory.MessageFactory.GetPrototype`, but in protobuf 6.x the `MessageFactory` API has changed and that attribute may not exist (or `MessageFactory` may not expose the expected class attributes). The current guard still triggers an AttributeError in this environment, preventing the rest of the imports (including TensorFlow) from running. We need to make the shim robust by catching AttributeError and only patching when both the class and replacement method exist. This keeps the intended compatibility behavior without changing any downstream logic.

Patch summary: Wrap the protobuf shim in a `try/except` and use `getattr` checks to safely detect and alias `GetPrototype` to `GetMessageClass` only when available. This prevents the AttributeError while preserving the original intent. No other imports or modeling logic are modified.

Updated cells: Only cell 0 is changed.

Compatibility notes for cell k+1: Cell 1 depends on successful imports in cell 0; after this fix, all imported names (`pd`, `np`, `tf`, etc.) remain available with the same interfaces, so cell 1 run unchanged.

Assumptions: The goal of the shim is only to avoid protobuf API incompatibility errors in TensorFlow-related imports; if neither method exists, leaving it unpatched is acceptable and should not worsen behavior compared to the current crash.'
- What this solution (achieved 377.74973) has done: 'The crash happens during the protobuf compatibility shim in cell 0: in protobuf 6.x `MessageFactory` no longer exposes `GetPrototype`, and trying to access it via an instance triggers an `AttributeError` in this environment. The fix is to make the shim robust to both class-level and instance-level APIs by patching `google.protobuf.message_factory.MessageFactory` only if it exists and only if `GetPrototype` is missing but `GetMessageClass` is present, while safely handling both methods. This keeps the intent (backward compatibility for TensorFlow/protobuf interaction) without changing any modeling or downstream logic. No changes are needed for later cells; all imports and symbols remain identical.'
- What this solution (achieved 409.52244) has done: 'Diagnosis: The crash happens during the protobuf monkey-patch in cell 0. In protobuf 6.x, `google.protobuf.message_factory.MessageFactory` is no longer the same API surface (and in this environment, the imported `message_factory` doesn’t expose a class where adding `GetPrototype` works), so the attempted `getattr(..., "MessageFactory")` path doesn’t succeed as intended and the patch still triggers an AttributeError about `GetPrototype`. Since TensorFlow 2.18 + protobuf 6.33 works without this legacy patch, the safest minimal fix is to remove/skip this monkey-patch entirely so the imports proceed normally. This keeps the notebook’s core modeling logic unchanged and unblocks execution.

Patch summary: Remove the incompatible protobuf `MessageFactory.GetPrototype` monkey-patch block in cell 0 so TensorFlow can import without raising `AttributeError`.

Updated cells: Only cell 0 is changed.

Compatibility notes for cell k+1: No variables from the removed patch are used later; all existing imports (`tf`, `keras`, `layers`, `callbacks`) remain defined exactly as before, so cell 1 and later cells remain compatible.

Assumptions: TensorFlow 2.18.0 in this environment is compatible with protobuf 6.33.0 without requiring the legacy `GetPrototype` shim.'
- What this solution (achieved 49.92373) has done: 'The crash happens immediately on importing TensorFlow because the installed `protobuf==6.33.0` is incompatible with `tensorflow==2.18.0`, triggering `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during import. The minimal, localized fix is to set the environment variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` **before** importing TensorFlow so it uses the pure-Python protobuf implementation and avoids the incompatible C++ API. This keeps the existing TensorFlow/Keras usage and downstream code unchanged. No other logic is modified.'
- What this solution (achieved 125.02586) has done: 'The crash happens during the TensorFlow import because `protobuf==6.33.0` is incompatible with TF 2.18 in this environment, triggering the `MessageFactory.GetPrototype` AttributeError. The smallest deterministic fix is to force TensorFlow to use the pure-Python protobuf runtime *before* TensorFlow is imported. In your cell, the environment variable is currently set **after** importing `protobuf` indirectly via other libs and after importing TensorFlow-related modules, so it has no effect. I move the `os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"` assignment to the very top of the cell, before any TensorFlow/protobuf-dependent imports, without changing any modeling logic.'
- What this solution (achieved 315.10391) has done: 'Diagnosis: The crash happens during the TensorFlow import in cell 0 due to an incompatibility between `protobuf==6.33.0` and the TensorFlow 2.18 stack in this environment, surfacing as `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. Setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` alone is not sufficient here, because the imported `google.protobuf` package version is still incompatible. The minimal deterministic fix is to pin protobuf to a TensorFlow-compatible major version (<5) at runtime (before importing TensorFlow) and restart the Python process import state in-cell by forcing a pip install early.

Patch summary: In cell 0 only, install a protobuf version `<5` (e.g., `4.25.3`) before importing TensorFlow/Keras. Keep the rest of the imports and core logic unchanged so downstream cells (including cell 1) see the same symbols and behavior, except that TensorFlow can import successfully.

Updated cells: cell 0 only.

Compatibility notes for cell k+1: Cell 1 is unchanged and still expects `pd` to be imported in cell 0; this remains true. The patch only ensures TensorFlow imports cleanly; no variables used by cell 1 are renamed or removed.

Assumptions: The environment allows installing wheels via pip at runtime (common in Kaggle-like notebooks). `protobuf==4.25.3` is available and compatible with TensorFlow 2.18 in this runtime.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (16, 8)
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks


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
f, axes = plt.subplots(1, 2)

sns.barplot(x='passenger_count', y='fare_amount', data=train_df, ax=axes[0])
sns.scatterplot(x='passenger_count', y='fare_amount', data=train_df, ax=axes[1])


## === cell 25
f, axes = plt.subplots(1, 2)

sns.barplot(x='Year', y='fare_amount', data=train_df, ax=axes[0])
sns.scatterplot(x='Year', y='fare_amount', data=train_df, ax=axes[1])


## === cell 26
f, axes = plt.subplots(1, 2)

sns.barplot(x='Month', y='fare_amount', data=train_df, ax=axes[0])
sns.scatterplot(x='Month', y='fare_amount', data=train_df, ax=axes[1])


## === cell 27
f, axes = plt.subplots(1, 2)

sns.barplot(x='Day', y='fare_amount', data=train_df, ax=axes[0])
sns.scatterplot(x='Day', y='fare_amount', data=train_df, ax=axes[1])


## === cell 28
f, axes = plt.subplots(1, 2)

sns.barplot(x='Weekday', y='fare_amount', data=train_df, ax=axes[0])
sns.scatterplot(x='Weekday', y='fare_amount', data=train_df, ax=axes[1])


## === cell 29
f, axes = plt.subplots(1, 2)

sns.barplot(x='Hour', y='fare_amount', data=train_df, ax=axes[0])
sns.scatterplot(x='Hour', y='fare_amount', data=train_df, ax=axes[1])


## === cell 30
train_df['DistanceGroups'] = pd.qcut(train_df['Distance'], 10)


## === cell 31
f, axes = plt.subplots(1, 2)
plt.setp( axes[0].xaxis.get_majorticklabels(), rotation=70 )
sns.barplot(x='DistanceGroups', y='fare_amount', data=train_df, ax=axes[0])
sns.scatterplot(x='Distance', y='fare_amount', data=train_df, ax=axes[1])


## === cell 32
train_df.columns


## === cell 33
features = train_df.columns[2:-1]
outcome = train_df.columns[1]


## === cell 34
X_train, X_test, y_train, y_test = train_test_split(train_df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'Year', 'Month', 'Day', 'Weekday', 'Hour', 'Distance']], train_df['fare_amount'], test_size=0.30, random_state=42)


## === cell 42
scaler = StandardScaler()

scaled_train = scaler.fit_transform(X_train)
scaled_valid = scaler.transform(X_test)
scaled_test = scaler.transform(test_df[features])


## === cell 43
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=5,
    restore_best_weights=True,
)


model = keras.Sequential([
    
    layers.Dense(128, activation='relu', input_dim=features.shape[0]),
    layers.BatchNormalization(),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),

    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),

    layers.Dense(8, activation='relu'),
    layers.BatchNormalization(),

    layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=[tf.keras.metrics.RootMeanSquaredError()]
)


## === cell 44
history = model.fit(
    scaled_train,
    y_train,
    validation_data=(scaled_valid, y_test),
    batch_size = 256,
    epochs = 50,
    callbacks=[early_stopping]
)


## === cell 45
prediction = model.predict(scaled_test, batch_size=256, verbose=1)
prediction = prediction.ravel()


submission = pd.DataFrame({
        "key": test_df['key'],
        "fare_amount": prediction
})

submission.to_csv('taxi_fare_submission.csv',index=False)
