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

3.49822

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 383.02398) has done: 'I fix the runtime crash caused by an incompatibility between `tensorflow.keras` and the standalone `keras==3.x` package by using only `tf.keras` throughout. I also fix the custom RMSE loss to use TensorFlow ops instead of `keras.backend` functions that don’t exist in Keras 3, which currently prevents training. To improve RMSE toward your target (without changing the model architecture or training loop), I correct a feature-selection bug where `features = train_df.columns[2:]` mistakenly includes `fare_amount` and excludes `pickup_longitude`, causing mis-scaling/misaligned inference. Finally, I ensure the submission is written with the required columns and `.csv` suffix.'
- What this solution (achieved 373.64553) has done: 'I fix the crash in the TensorFlow import by forcing TensorFlow to use the pure-Python protobuf implementation, which avoids the `MessageFactory.GetPrototype` incompatibility seen in this Kaggle image. Then I make sure model training uses a correct validation setup (remove `validation_steps`, which is only meant for generator/`tf.data` inputs and can silently mis-handle numpy arrays), which should significantly reduce the RMSE toward your target without changing the model architecture or loss. Finally, I ensure the submission file is written as a proper `.csv` with the required `key` and `fare_amount` columns.'
- What this solution (achieved 122.43807) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting the protobuf environment variables before any TensorFlow/protobuf-related imports and by importing `tf.keras` only. Then I ensure the training target is a clean NumPy float32 array (to avoid dtype/object issues) and clamp negative predictions to 0 at inference time (fares can’t be negative), which is a minimal post-processing step that typically improves RMSE substantially without changing the model/training core. Finally, I keep the submission format exactly as required and verify the `.csv` is written successfully.'
- What this solution (achieved 479.01113) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting the protobuf environment variables before any protobuf/TensorFlow import and by force-importing `google.protobuf` first, which stabilizes TF 2.18 in this Kaggle image. I also make sure the TensorFlow import happens in the first cell so later cells don’t partially execute with a broken TF state. To move RMSE strongly toward your target without changing the model architecture or training loop, I add a minimal, standard data-cleaning filter to remove extreme outlier fares and unrealistic trips (very long distances), which otherwise dominate RMSE and yield huge errors. Finally, I keep the same submission format/filename and ensure predictions remain non-negative.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3111604776.py in <cell line: 0>()
     13 import seaborn as sns
     14 
---> 15 import tensorflow as tf
     16 from tensorflow.keras import layers
     17 

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     16 
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message
     20 from google.protobuf import reflection as _reflection

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     36 import weakref
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod
     40 from google.protobuf import text_format

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     27   # TODO: Remove this import after fix api_implementation
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True
     31 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 1
train_df = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1000000
)
test_df = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 2
train_df.isnull().sum()



## === cell 3
train_df.dropna(axis=0, subset=["dropoff_longitude", "dropoff_latitude"], inplace=True)
train_df = train_df.reset_index(drop=True)



## === cell 4
pd.set_option("display.float_format", lambda x: "%.5f" % x)
train_df.describe()



## === cell 5
print("Number of observations out of valid range in coordinate columns:", end="\n")

print("pickup_longitude", end=": ")
print(
    (train_df.pickup_longitude < -180).sum() + (train_df.pickup_longitude > 180).sum()
)

print("pickup_latitude", end=": ")
print((train_df.pickup_latitude < -90).sum() + (train_df.pickup_latitude > 90).sum())

print("dropoff_longitude", end=": ")
print(
    (train_df.dropoff_longitude < -180).sum() + (train_df.dropoff_longitude > 180).sum()
)

print("dropoff_latitude", end=": ")
print((train_df.dropoff_latitude < -90).sum() + (train_df.dropoff_latitude > 90).sum())



## === cell 6
train_df = train_df.drop(
    train_df[
        (train_df.pickup_longitude < -180) | (train_df.pickup_longitude > 180)
    ].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[(train_df.pickup_latitude < -90) | (train_df.pickup_latitude > 90)].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[
        (train_df.dropoff_longitude < -180) | (train_df.dropoff_longitude > 180)
    ].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[
        (train_df.dropoff_latitude < -90) | (train_df.dropoff_latitude > 90)
    ].index,
    axis=0,
)



## === cell 7
train_df.describe()



## === cell 8
train_df[(train_df.pickup_longitude >= 40)]



## === cell 9
indx = train_df[(train_df.pickup_longitude >= 40)].index
train_df.loc[indx, ["dropoff_longitude", "dropoff_latitude"]] = train_df.loc[
    indx, ["dropoff_latitude", "dropoff_longitude"]
].values
train_df.loc[indx, ["pickup_longitude", "pickup_latitude"]] = train_df.loc[
    indx, ["pickup_latitude", "pickup_longitude"]
].values



## === cell 10
train_df = train_df.drop(
    train_df[
        (train_df.pickup_longitude < -75) | (train_df.pickup_longitude > -72)
    ].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[
        (train_df.dropoff_longitude < -75) | (train_df.dropoff_longitude > -72)
    ].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[(train_df.pickup_latitude < 40) | (train_df.pickup_latitude > 42)].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[(train_df.dropoff_latitude < 40) | (train_df.dropoff_latitude > 42)].index,
    axis=0,
)



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
train_df = train_df.drop(train_df[train_df.fare_amount > 200].index, axis=0)

train_df["fare_amount"].sort_values(ascending=False)



## === cell 16
test_df.isna().sum()



## === cell 17
test_df.describe()



## === cell 18
train_df.dtypes



## === cell 19
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])




## === cell 20
def date_splitter(df):
    df["Year"] = df["pickup_datetime"].dt.year
    df["Month"] = df["pickup_datetime"].dt.month
    df["Day"] = df["pickup_datetime"].dt.day
    df["Weekday"] = df["pickup_datetime"].dt.dayofweek
    df["Hour"] = df["pickup_datetime"].dt.hour


date_splitter(train_df)
date_splitter(test_df)

train_df.drop(["pickup_datetime"], axis=1, inplace=True)
test_df.drop(["pickup_datetime"], axis=1, inplace=True)



## === cell 21
import math


def haversine_distance(df):
    coord = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]

    phi1, lambda1, phi2, lambda2 = [df[i] * math.pi / 180.0 for i in coord]

    R = 6371
    dPhi = phi2 - phi1
    dLambda = lambda2 - lambda1

    a = (
        np.sin(dPhi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dLambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c

    df["Distance"] = d


haversine_distance(train_df)
haversine_distance(test_df)



## === cell 22
train_df.Distance.sort_values()



## === cell 23
train_df = train_df.drop(train_df[train_df.Distance < 0.5].index, axis=0)
train_df = train_df.drop(train_df[train_df.Distance > 100].index, axis=0)



## === cell 24
sns.scatterplot(x="passenger_count", y="fare_amount", data=train_df)



## === cell 25
sns.scatterplot(x="Year", y="fare_amount", data=train_df)



## === cell 26
sns.scatterplot(x="Month", y="fare_amount", data=train_df)



## === cell 27
sns.scatterplot(x="Day", y="fare_amount", data=train_df)



## === cell 28
sns.scatterplot(x="Weekday", y="fare_amount", data=train_df)



## === cell 29
sns.scatterplot(x="Hour", y="fare_amount", data=train_df)



## === cell 30
sns.scatterplot(x="Distance", y="fare_amount", data=train_df)



## === cell 31
FEATURES = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "Year",
    "Month",
    "Day",
    "Weekday",
    "Hour",
    "Distance",
]



## === cell 32
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



## === cell 33
X_train, X_test, y_train, y_test = train_test_split(
    train_df[FEATURES], train_df["fare_amount"], test_size=0.30, random_state=42
)



## === cell 34
scaler = StandardScaler()
train_scalin = scaler.fit_transform(X_train)
val_scalin = scaler.transform(X_test)
test_scalin = scaler.transform(test_df[FEATURES])

y_train = y_train.to_numpy(dtype=np.float32)
y_test = y_test.to_numpy(dtype=np.float32)




## === cell 35
def root_mean_squared_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))




## === cell 36
def build_and_compile_model(dim):
    model = tf.keras.Sequential(
        [
            layers.Dense(128, activation="relu", input_shape=(dim,)),
            layers.BatchNormalization(),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(32, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(8, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(1),
        ]
    )

    model.compile(
        loss=root_mean_squared_error,
        optimizer=tf.keras.optimizers.Adam(0.01),
        metrics=["mae"],
    )
    return model




## === cell 37
dnn_model = build_and_compile_model(dim=len(FEATURES))



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3539990001.py in <cell line: 0>()
----> 1 dnn_model = build_and_compile_model(dim=len(FEATURES))
      2 

/tmp/ipykernel_11/3953290504.py in build_and_compile_model(dim)
      1 def build_and_compile_model(dim):
----> 2     model = tf.keras.Sequential(
      3         [
      4             layers.Dense(128, activation="relu", input_shape=(dim,)),
      5             layers.BatchNormalization(),

NameError: name 'tf' is not defined

## === cell 38
ep_no = 10
Batch = 128



## === cell 39
history = dnn_model.fit(
    train_scalin,
    y_train,
    validation_data=(val_scalin, y_test),
    batch_size=Batch,
    epochs=ep_no,
    verbose=1,
)



## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2517992122.py in <cell line: 0>()
----> 1 history = dnn_model.fit(
      2     train_scalin,
      3     y_train,
      4     validation_data=(val_scalin, y_test),
      5     batch_size=Batch,

NameError: name 'dnn_model' is not defined

## === cell 40
prediction = dnn_model.predict(test_scalin, batch_size=Batch, verbose=1).ravel()

prediction = np.clip(prediction, 0.0, None)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": prediction})
submission.to_csv("taxi_fare_submission.csv", index=False)
print(submission.head())
print("Wrote taxi_fare_submission.csv with shape:", submission.shape)
print("fare_amount stats:", submission["fare_amount"].describe())

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2284433070.py in <cell line: 0>()
----> 1 prediction = dnn_model.predict(test_scalin, batch_size=Batch, verbose=1).ravel()
      2 
      3 prediction = np.clip(prediction, 0.0, None)
      4 
      5 submission = pd.DataFrame({"key": test_df["key"], "fare_amount": prediction})

NameError: name 'dnn_model' is not defined
