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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
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

4.3735

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.68165) has done: 'I fix the runtime errors caused by the Keras 3 API changes (optimizer name, removed `vis_utils`, and protobuf-related import crash) while keeping the same model architecture/training loop and feature logic. I also correct a couple of logic bugs in the time-feature flags (they currently always return 1 due to incorrect conditions), which should legitimately improve RMSE toward your target without changing the overall approach. I ensure file paths match your environment (`/kaggle/input/...`) and that a valid `submissiontry_water.csv` is always written with the required `key,fare_amount` columns. Finally, I make the plotting/model-visualization cells non-fatal so the notebook runs end-to-end.'
- What this solution (achieved 5.65597) has done: 'I fix the immediate runtime crash in the first cell (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by pinning protobuf to a compatible pure-Python implementation *before* importing `tf_keras`, which is a common Kaggle/TensorFlow-protobuf mismatch. Then I correct a key data-loading bug where `fare_amount` was accidentally excluded from the training read (`usecols` missing it), which severely hurts model learning and Kaggle RMSE; this change preserves your model/training loop but restores the intended target signal. Finally, I keep all paths and submission writing unchanged and add a small safety clamp on predictions (non-negative) to avoid impossible negative fares that can worsen RMSE, without altering the core approach.'

# 9. Code solution

## === cell 0
import os
import sys
import random
import warnings
import subprocess

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import message_factory  # noqa: F401

    if not hasattr(message_factory.MessageFactory(), "GetPrototype"):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "uninstall", "-y", "protobuf"]
        )
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, BatchNormalization
from tf_keras.callbacks import ModelCheckpoint
from tf_keras import optimizers, regularizers
from tf_keras import backend

warnings.filterwarnings("ignore")

SEED = 1
np.random.seed(SEED)
random.seed(SEED)

TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

print("Imports OK. Using TRAIN_PATH:", TRAIN_PATH, "TEST_PATH:", TEST_PATH)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3041123812.py in <cell line: 0>()
     33 from sklearn.model_selection import train_test_split
     34 
---> 35 import tf_keras as keras
     36 from tf_keras.models import Sequential
     37 from tf_keras.layers import Dense, BatchNormalization

/usr/local/lib/python3.11/dist-packages/tf_keras/__init__.py in <module>
      1 """AUTOGENERATED. DO NOT EDIT."""
      2 
----> 3 from tf_keras import __internal__
      4 from tf_keras import activations
      5 from tf_keras import applications

/usr/local/lib/python3.11/dist-packages/tf_keras/__internal__/__init__.py in <module>
      1 """AUTOGENERATED. DO NOT EDIT."""
      2 
----> 3 from tf_keras.__internal__ import backend
      4 from tf_keras.__internal__ import layers
      5 from tf_keras.__internal__ import losses

/usr/local/lib/python3.11/dist-packages/tf_keras/__internal__/backend/__init__.py in <module>
      1 """AUTOGENERATED. DO NOT EDIT."""
      2 
----> 3 from tf_keras.src.backend import _initialize_variables as initialize_variables
      4 from tf_keras.src.backend import track_variable

/usr/local/lib/python3.11/dist-packages/tf_keras/src/__init__.py in <module>
     19 """
     20 
---> 21 from tf_keras.src import applications
     22 from tf_keras.src import distribute
     23 from tf_keras.src import layers

/usr/local/lib/python3.11/dist-packages/tf_keras/src/applications/__init__.py in <module>
     16 
     17 
---> 18 from tf_keras.src.applications.convnext import ConvNeXtBase
     19 from tf_keras.src.applications.convnext import ConvNeXtLarge
     20 from tf_keras.src.applications.convnext import ConvNeXtSmall

/usr/local/lib/python3.11/dist-packages/tf_keras/src/applications/convnext.py in <module>
     24 
     25 import numpy as np
---> 26 import tensorflow.compat.v2 as tf
     27 
     28 from tf_keras.src import backend

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

ImportError: cannot import name 'builder' from 'google.protobuf.internal' (/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/__init__.py)

## === cell 1
def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    has_target = "fare_amount" in df.columns

    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        & (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    print(" New size after removing same long lat: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    ]
    print(" New size after removing 0 long lat: %d" % len(df))

    MinMax = (-74.5, -72.8, 40.5, 41.8)
    df = df[
        (MinMax[0] <= df["pickup_longitude"]) & (df["pickup_longitude"] <= MinMax[1])
    ]
    df = df[
        (MinMax[0] <= df["dropoff_longitude"]) & (df["dropoff_longitude"] <= MinMax[1])
    ]
    df = df[(MinMax[2] <= df["pickup_latitude"]) & (df["pickup_latitude"] <= MinMax[3])]
    df = df[
        (MinMax[2] <= df["dropoff_latitude"]) & (df["dropoff_latitude"] <= MinMax[3])
    ]
    print(" New size after NYC lang lot: %d" % len(df))

    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["pickup_longitude"] != 0)]
    df = df[(df["dropoff_longitude"] != 0)]
    df = df[(df["dropoff_latitude"] != 0)]
    print(" New size after lang lot > 0: %d" % len(df))

    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)]
    print(" New size after lang - lot > 0.001: %d" % len(df))

    print(" New size after only NYC: %d" % len(df))

    if has_target:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))
    else:
        print(" Skipping fare_amount outlier removal for test data.")

    df = df[(df["passenger_count"] > 0)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    return df


def remove_datapoints_from_water(df):
    raise RuntimeError(
        "remove_datapoints_from_water() requires internet to fetch a mask image; not supported in Kaggle runtime."
    )


def late_night(row):
    h = row["hour"]
    return 1 if (h <= 3) else 0


def night(row):
    h = row["hour"]
    wd = row["weekday"]
    return 1 if ((h >= 20 or h <= 6) and wd < 5) else 0


def rush_hour(row):
    h = row["hour"]
    wd = row["weekday"]
    return 1 if ((16 <= h <= 20) and wd < 5) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    if dt.isna().any():
        dt2 = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        dt = dt.fillna(dt2)

    df["year"] = dt.dt.year.astype("int16")
    df["month"] = dt.dt.month.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["weekday"] = dt.dt.weekday.astype("int8")

    df["pickup_datetime"] = dt.astype(str)

    df["night"] = df.apply(lambda x: night(x), axis=1).astype("int8")
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1).astype("int8")
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1).astype("int8")
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["latdiff"] = (lat1 - lat2).abs()
    df["londiff"] = (lon1 - lon2).abs()
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = np.sqrt(np.abs(lon1 - lon2) ** 2 + np.abs(lat1 - lat2) ** 2)
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction.reshape(-1), columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history.get("loss", []))
    plt.plot(history.history.get("val_loss", []))
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()

    plt.figure(figsize=(20, 10))
    if "rmse" in history.history and "val_rmse" in history.history:
        plt.plot(history.history["rmse"])
        plt.plot(history.history["val_rmse"])
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper right")
        plt.show()




## === cell 2
datatypes = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

trainKaggle = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    dtype=datatypes,
    usecols=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)

testKaggle = pd.read_csv(
    TEST_PATH,
    dtype={
        "key": "str",
        "pickup_datetime": "str",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
)

print("Loaded trainKaggle columns:", list(trainKaggle.columns))
print("Loaded testKaggle columns:", list(testKaggle.columns))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4172085155.py in <cell line: 0>()
     13 # but prevents accidental downstream misalignment and makes debugging easier).
     14 trainKaggle = pd.read_csv(
---> 15     TRAIN_PATH,
     16     nrows=DATASET_SIZE,
     17     dtype=datatypes,

NameError: name 'TRAIN_PATH' is not defined

## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/19325734.py in <cell line: 0>()
----> 1 train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
      2 test_df = test_df[:10000]
      3 

NameError: name 'trainKaggle' is not defined

## === cell 4
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/395627971.py in <cell line: 0>()
----> 1 print("testKaggle Size %d" % len(testKaggle))
      2 print("train_df Size %d" % len(train_df))
      3 print("test_df Size %d" % len(test_df))
      4 

NameError: name 'testKaggle' is not defined

## === cell 5
train_df.describe()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 6
test_df.describe()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294142452.py in <cell line: 0>()
----> 1 test_df.describe()
      2 

NameError: name 'test_df' is not defined

## === cell 7
print("train_df clean")
train_df = clean(train_df)
test_df = clean(test_df)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1475834959.py in <cell line: 0>()
      1 print("train_df clean")
----> 2 train_df = clean(train_df)
      3 test_df = clean(test_df)
      4 

NameError: name 'train_df' is not defined

## === cell 8
train_df



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3752401607.py in <cell line: 0>()
----> 1 train_df
      2 

NameError: name 'train_df' is not defined

## === cell 9
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1145226127.py in <cell line: 0>()
      1 print("train_df add_time_features")
----> 2 train_df = add_time_features(train_df)
      3 print("test_df add_time_features")
      4 test_df = add_time_features(test_df)
      5 print("testKaggle add_time_features")

NameError: name 'train_df' is not defined

## === cell 10
train_df.describe()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 11
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1245147733.py in <cell line: 0>()
      1 print("train_df add_coordinate_features")
----> 2 train_df = add_coordinate_features(train_df)
      3 print("test_df add_coordinate_features")
      4 test_df = add_coordinate_features(test_df)
      5 print("testKaggle add_coordinate_features")

NameError: name 'train_df' is not defined

## === cell 12
train_df.describe()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 13
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)
print("Done with Adding features")



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1336277678.py in <cell line: 0>()
      1 print("train_df add_distances_features")
----> 2 train_df = add_distances_features(train_df)
      3 print("test_df add_distances_features")
      4 test_df = add_distances_features(test_df)
      5 print("testKaggle add_distances_features")

NameError: name 'train_df' is not defined

## === cell 14
train_df.describe()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 15
try:
    plot = train_df.iloc[:2000].plot.scatter("latdiff", "londiff")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "weekday")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "night")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "late_night")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "rush_hour")
    plt.show()
except Exception as e:
    print("Plotting skipped:", repr(e))



## === cell 16
dropped_columns = [
    "pickup_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "key",
]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns, axis=1)

print("Done with dropped_columns")



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1358742248.py in <cell line: 0>()
      9 ]
     10 
---> 11 train_df = train_df.drop(dropped_columns, axis=1)
     12 test_df = test_df.drop(dropped_columns, axis=1)
     13 testKaggle_clean = testKaggle.drop(dropped_columns, axis=1)

NameError: name 'train_df' is not defined

## === cell 17
train_df.shape



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/186828624.py in <cell line: 0>()
----> 1 train_df.shape
      2 

NameError: name 'train_df' is not defined

## === cell 18
train_df.describe()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 19
test_df.describe()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294142452.py in <cell line: 0>()
----> 1 test_df.describe()
      2 

NameError: name 'test_df' is not defined

## === cell 20
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1154705564.py in <cell line: 0>()
----> 1 train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)
      2 

NameError: name 'train_df' is not defined

## === cell 21
train_df_main = train_df
validation_df_main = validation_df



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2901810251.py in <cell line: 0>()
----> 1 train_df_main = train_df
      2 validation_df_main = validation_df
      3 

NameError: name 'train_df' is not defined

## === cell 22
validation_df.describe()



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3277078349.py in <cell line: 0>()
----> 1 validation_df.describe()
      2 

NameError: name 'validation_df' is not defined

## === cell 23
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/786339670.py in <cell line: 0>()
----> 1 train_labels = train_df["fare_amount"].values
      2 validation_labels = validation_df["fare_amount"].values
      3 test_labels = test_df["fare_amount"].values
      4 
      5 train_df = train_df.drop(["fare_amount"], axis=1)

NameError: name 'train_df' is not defined

## === cell 24
test_labels



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/805156543.py in <cell line: 0>()
----> 1 test_labels
      2 

NameError: name 'test_labels' is not defined

## === cell 25
train_df.describe()



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 26
validation_df.describe()



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3277078349.py in <cell line: 0>()
----> 1 validation_df.describe()
      2 

NameError: name 'validation_df' is not defined

## === cell 27
test_df.describe()



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294142452.py in <cell line: 0>()
----> 1 test_df.describe()
      2 

NameError: name 'test_df' is not defined

## === cell 28
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/65882481.py in <cell line: 0>()
      1 scaler = preprocessing.MinMaxScaler()
----> 2 train_df_scaled = scaler.fit_transform(train_df)
      3 validation_df_scaled = scaler.transform(validation_df)
      4 test_scaled = scaler.transform(test_df)
      5 testKaggle_scaled = scaler.transform(testKaggle_clean)

NameError: name 'train_df' is not defined

## === cell 29
test_scaled




## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1156171887.py in <cell line: 0>()
----> 1 test_scaled
      2 
      3 

NameError: name 'test_scaled' is not defined

## === cell 30
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))




## === cell 31
checkpoint = ModelCheckpoint(filepath="my_model.h5", verbose=1, save_best_only=True)

model = Sequential()
model.add(
    Dense(
        256,
        activation="linear",
        input_dim=train_df_scaled.shape[1],
        activity_regularizer=regularizers.l1(0.01),
    )
)
model.add(BatchNormalization())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(32, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(8, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mean_squared_error", optimizer=adam, metrics=["mae", rmse, "mse"])

print("Dataset size: %s" % DATASET_SIZE)
print("Epochs: %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % list(train_df.columns))
model.summary()

history = model.fit(
    x=train_df_scaled,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[checkpoint],
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1925287230.py in <cell line: 0>()
----> 1 checkpoint = ModelCheckpoint(filepath="my_model.h5", verbose=1, save_best_only=True)
      2 
      3 model = Sequential()
      4 model.add(
      5     Dense(

NameError: name 'ModelCheckpoint' is not defined

## === cell 32
try:
    from IPython.display import SVG, display
    from tf_keras.utils import model_to_dot

    display(SVG(model_to_dot(model).create(prog="dot", format="svg")))
except Exception as e:
    print("Model visualization skipped:", repr(e))



## === cell 33
plot_loss_accuracy_rmse(history)



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2579130885.py in <cell line: 0>()
----> 1 plot_loss_accuracy_rmse(history)
      2 

NameError: name 'history' is not defined

## === cell 34
score = model.evaluate(train_df_scaled, train_labels, verbose=1)
print(score)
print("train mean_squared_error:", score[0])
print("train mae:", score[1])
print("train rmse:", score[2])
print("train mse:", score[3])



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4103044261.py in <cell line: 0>()
----> 1 score = model.evaluate(train_df_scaled, train_labels, verbose=1)
      2 print(score)
      3 print("train mean_squared_error:", score[0])
      4 print("train mae:", score[1])
      5 print("train rmse:", score[2])

NameError: name 'model' is not defined

## === cell 35
score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print(score)
print("Validation mean_squared_error:", score[0])
print("Validation mae:", score[1])
print("Validation rmse:", score[2])
print("Validation mse:", score[3])



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2701956346.py in <cell line: 0>()
----> 1 score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
      2 print(score)
      3 print("Validation mean_squared_error:", score[0])
      4 print("Validation mae:", score[1])
      5 print("Validation rmse:", score[2])

NameError: name 'model' is not defined

## === cell 36
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print("Test mean_squared_error:", score[0])
print("Test mae:", score[1])
print("Test rmse:", score[2])
print("Test mse:", score[3])



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1107548853.py in <cell line: 0>()
----> 1 score = model.evaluate(test_scaled, test_labels, verbose=1)
      2 print(score)
      3 print("Test mean_squared_error:", score[0])
      4 print("Test mae:", score[1])
      5 print("Test rmse:", score[2])

NameError: name 'model' is not defined

## === cell 37
validation_predictions = model.predict(validation_df_scaled).flatten()

plt.scatter(validation_labels, validation_predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot(
    [validation_predictions.min(), validation_predictions.max()],
    [validation_predictions.min(), validation_predictions.max()],
    "k--",
    lw=4,
)
plt.show()



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/867209637.py in <cell line: 0>()
----> 1 validation_predictions = model.predict(validation_df_scaled).flatten()
      2 
      3 plt.scatter(validation_labels, validation_predictions)
      4 plt.xlabel("True Values")
      5 plt.ylabel("Predictions")

NameError: name 'model' is not defined

## === cell 38
test_predictions = model.predict(test_scaled).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot(
    [test_predictions.min(), test_predictions.max()],
    [test_predictions.min(), test_predictions.max()],
    "k--",
    lw=4,
)
plt.show()



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2983374147.py in <cell line: 0>()
----> 1 test_predictions = model.predict(test_scaled).flatten()
      2 
      3 plt.scatter(test_labels, test_predictions)
      4 plt.xlabel("True Values")
      5 plt.ylabel("Predictions")

NameError: name 'model' is not defined

## === cell 39
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_labels[np.argmax(test_predictions)])
test_df.iloc[np.argmax(test_predictions)]



## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3782083718.py in <cell line: 0>()
----> 1 print(np.argmax(test_predictions))
      2 print(test_predictions[np.argmax(test_predictions)])
      3 print(test_labels[np.argmax(test_predictions)])
      4 test_df.iloc[np.argmax(test_predictions)]
      5 

NameError: name 'test_predictions' is not defined

## === cell 40
print(np.argmin(test_predictions))
print(test_predictions[np.argmin(test_predictions)])
print(test_labels[np.argmin(test_predictions)])
test_df.iloc[np.argmin(test_predictions)]



## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1775569351.py in <cell line: 0>()
----> 1 print(np.argmin(test_predictions))
      2 print(test_predictions[np.argmin(test_predictions)])
      3 print(test_labels[np.argmin(test_predictions)])
      4 test_df.iloc[np.argmin(test_predictions)]
      5 

NameError: name 'test_predictions' is not defined

## === cell 41
fig, ax = plt.subplots()
ax.scatter(test_labels, test_predictions)
ax.plot(
    [test_labels.min(), test_labels.max()],
    [test_labels.min(), test_labels.max()],
    "k--",
    lw=4,
)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()



## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3918075604.py in <cell line: 0>()
      1 fig, ax = plt.subplots()
----> 2 ax.scatter(test_labels, test_predictions)
      3 ax.plot(
      4     [test_labels.min(), test_labels.max()],
      5     [test_labels.min(), test_labels.max()],

NameError: name 'test_labels' is not defined

## === cell 42
plt.figure(figsize=(20, 10))
plt.plot(validation_labels[:100])
plt.plot(validation_predictions[:100])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()

validation_df_scaled[52]
validation_df.iloc[52]



## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3059730636.py in <cell line: 0>()
      1 plt.figure(figsize=(20, 10))
----> 2 plt.plot(validation_labels[:100])
      3 plt.plot(validation_predictions[:100])
      4 plt.title("Prediction vs Actual")
      5 plt.ylabel("Fare Amount")

NameError: name 'validation_labels' is not defined

## === cell 43
plt.figure(figsize=(20, 10))
plt.plot(test_labels[:100])
plt.plot(test_predictions[:100])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()



## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1260486629.py in <cell line: 0>()
      1 plt.figure(figsize=(20, 10))
----> 2 plt.plot(test_labels[:100])
      3 plt.plot(test_predictions[:100])
      4 plt.title("Prediction vs Actual")
      5 plt.ylabel("Fare Amount")

NameError: name 'test_labels' is not defined

## === cell 44
error = validation_predictions - validation_labels
plt.hist(error, bins=100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")
plt.show()



## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1681881283.py in <cell line: 0>()
----> 1 error = validation_predictions - validation_labels
      2 plt.hist(error, bins=100)
      3 plt.xlabel("Prediction Error")
      4 _ = plt.ylabel("Count")
      5 plt.show()

NameError: name 'validation_predictions' is not defined

## === cell 45
error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")
plt.show()



## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/386566829.py in <cell line: 0>()
----> 1 error = test_predictions - test_labels
      2 plt.hist(error, bins=50)
      3 plt.xlabel("Prediction Error")
      4 _ = plt.ylabel("Count")
      5 plt.show()

NameError: name 'test_predictions' is not defined

## === cell 46
errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
print(len(error))
print(len(errorGreaterZero))

plt.hist(errorGreaterZero, bins=100)
plt.xlabel("Prediction Error (|error|>=1)")
_ = plt.ylabel("Count")
plt.show()



## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1278946122.py in <cell line: 0>()
----> 1 errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
      2 print(len(error))
      3 print(len(errorGreaterZero))
      4 
      5 plt.hist(errorGreaterZero, bins=100)

NameError: name 'error' is not defined

## === cell 47
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)

predictionKaggle = np.maximum(predictionKaggle, 0.0)



## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2873405754.py in <cell line: 0>()
----> 1 predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)
      2 
      3 # Keep the existing safety clamp (score-neutral to mildly positive for RMSE).
      4 predictionKaggle = np.maximum(predictionKaggle, 0.0)
      5 

NameError: name 'model' is not defined

## === cell 48
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)

sub = pd.read_csv(SUBMISSION_NAME)
print(sub.head())
print("Submission shape:", sub.shape)
print("Columns:", list(sub.columns))
print("Saved submission to:", os.path.abspath(SUBMISSION_NAME))

## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2590011050.py in <cell line: 0>()
----> 1 output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
      2 
      3 sub = pd.read_csv(SUBMISSION_NAME)
      4 print(sub.head())
      5 print("Submission shape:", sub.shape)

NameError: name 'testKaggle' is not defined
