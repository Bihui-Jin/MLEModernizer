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

4.60368

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 30.2958) has done: 'I fix the three blockers preventing your notebook from running end-to-end: (1) the Keras/TF import stack causing the protobuf `MessageFactory.GetPrototype` crash, (2) the NYC “water mask” URL read failing due to `plt.imread()` not supporting URLs in this environment, and (3) the Keras 3 optimizer API change (`optimizers.adam` no longer exists). These fixes keep the same model architecture and training loop, but allow training to actually happen and produce a valid `submissiontry_water.csv`. I also make datetime parsing robust (your strict `%Z` format often fails on this dataset) and remove the accidental “accuracy” metric on a regression model (score-neutral but prevents misleading behavior/overhead). With training running and water-filtering restored locally (no internet), the score should move substantially down from ~15 toward your ~4.6 target.'
- What this solution (achieved 66.64292) has done: 'I fix the two execution blockers: the protobuf/Keras import crash and the MinMaxScaler failing because a string column (`key`) is still present in the feature matrix. The fixes are minimal and keep your model architecture/training loop intact: switch to the stable `tf_keras` import path (and stop forcing the pure-Python protobuf), and drop `key` from train/val/test feature frames before scaling (while keeping it in `testKaggle` for submission). I also make the train/test CSV paths robust to the dataset being located under either `/kaggle/input/` or `/kaggle/data/`, without changing your I/O semantics. After these fixes, the notebook run end-to-end and write a valid `submissiontry_water.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 145.84082) has done: 'I fix the protobuf/Keras import crash by forcing a compatible protobuf runtime (pure-Python) before importing `tf_keras`, which addresses the `MessageFactory.GetPrototype` error in this environment. Then, to move RMSE down toward your 4.60368 target (from ~66), I fix a core feature/target mismatch: your training set drops `passenger_count` but the Kaggle test features keep it, which makes the scaler/model see different distributions and can explode error; I keep feature columns aligned by dropping the same columns from the Kaggle test feature frame too. Finally, I keep the rest of your model/training loop intact and ensure the submission is written as a valid `key,fare_amount` CSV.'
- What this solution (achieved 69.5975) has done: 'I fix the execution blocker caused by forcing the pure-Python protobuf implementation, which triggers the `MessageFactory.GetPrototype` crash when importing `tf_keras` in this environment. This change is score-neutral but required to run end-to-end and actually train/predict. I also keep feature columns strictly aligned (train/val/test/Kaggle test) and ensure `key` never enters scaling, which prevents silent feature-matrix issues that can blow up RMSE. Finally, I keep your model, loss, and training loop unchanged and ensure a valid `key,fare_amount` submission CSV is always written.'
- What this solution (achieved 187.58285) has done: 'I fix the remaining import-time crash (`MessageFactory.GetPrototype`) by forcing a protobuf runtime that is compatible with the `tf_keras` stack in this Kaggle image, and do it before importing `tf_keras`. Then I address the main score issue (RMSE ~69) with a minimal, metric-aligned correction: the model is currently trained on a MinMax-scaled feature space but the target (`fare_amount`) is left unscaled, which often destabilizes regression and produces wildly miscalibrated outputs; I MinMax-scale the target using the same training split only, train/predict in scaled space, and inverse-transform predictions back to dollars for evaluation/submission. This keeps your architecture, loss, features, and training loop intact, but fixes a core scale mismatch that should move RMSE sharply down toward your ~4.6 target. Finally, I keep the submission format unchanged (`key,fare_amount`) and ensure the CSV is always written.'
- What this solution (achieved 6.08229) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by removing the forced pure-Python protobuf environment variables and importing `tf_keras` first, which is the root cause of your current runtime failure. Then I make the run deterministic across TF/Keras by setting `PYTHONHASHSEED` and TF seeds to stabilize training (score-improving but minimal, without changing the model/loop). Finally, I keep the existing feature engineering and target scaling logic intact, ensuring the pipeline completes end-to-end and always writes a valid `submissiontry_water.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("PYTHONHASHSEED", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    _SKEX_PATCHED = True
except Exception:
    _SKEX_PATCHED = False

import random
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tf_keras.callbacks import EarlyStopping
from tf_keras import optimizers
from tf_keras import regularizers
from tf_keras import backend as K

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def _resolve_path(p):
    if os.path.exists(p):
        return p
    alt = p.replace("/kaggle/input/", "/kaggle/data/")
    return alt if os.path.exists(alt) else p


TRAIN_PATH = _resolve_path("/kaggle/input/train.csv")
TEST_PATH = _resolve_path("/kaggle/input/test.csv")
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

print("sklearnex patched:", _SKEX_PATCHED)
print("TRAIN_PATH:", TRAIN_PATH, "exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH:", TEST_PATH, "exists:", os.path.exists(TEST_PATH))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3870950913.py in <cell line: 0>()
     27 from sklearn.model_selection import train_test_split
     28 
---> 29 import tensorflow as tf
     30 import tf_keras as keras
     31 from tf_keras.models import Sequential

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
def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    local_candidates = [
        "/kaggle/input/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/working/nyc_mask-74.5_-72.8_40.5_41.8.png",
    ]
    mask_path = next((p for p in local_candidates if os.path.exists(p)), None)

    if mask_path is None:
        return df

    nyc_mask = plt.imread(mask_path)[:, :, 0] > 0.9

    pickup_x, pickup_y = lonlat_to_xy(
        df.pickup_longitude,
        df.pickup_latitude,
        nyc_mask.shape[1],
        nyc_mask.shape[0],
        BB,
    )
    dropoff_x, dropoff_y = lonlat_to_xy(
        df.dropoff_longitude,
        df.dropoff_latitude,
        nyc_mask.shape[1],
        nyc_mask.shape[0],
        BB,
    )
    pickup_x = np.clip(pickup_x, 0, nyc_mask.shape[1] - 1)
    dropoff_x = np.clip(dropoff_x, 0, nyc_mask.shape[1] - 1)
    pickup_y = np.clip(pickup_y, 0, nyc_mask.shape[0] - 1)
    dropoff_y = np.clip(dropoff_y, 0, nyc_mask.shape[0] - 1)

    idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]
    return df[idx]


def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

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
    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["dropoff_longitude"] != 0)]
    df = df[(df["dropoff_latitude"] != 0)]
    print(" New size after lang lot > 0: %d" % len(df))

    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)]
    print(" New size after lang - lot > 0.001: %d" % len(df))

    print(" New size after only NYC: %d" % len(df))
    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)

    df = df[
        (nyc_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != nyc_coord[0])
    ]
    df = df[
        (nyc_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != nyc_coord[0])
    ]
    print(" New size after NY airport: %d" % len(df))

    df = df[
        (fk_coord[1] != df["pickup_longitude"]) & (df["pickup_latitude"] != fk_coord[0])
    ]
    df = df[
        (fk_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != fk_coord[0])
    ]
    print(" New size after jfk airport: %d" % len(df))

    df = df[
        (ewr_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != ewr_coord[0])
    ]
    df = df[
        (ewr_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != ewr_coord[0])
    ]
    print(" New size after ewr airport: %d" % len(df))

    df = df[
        (lga_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != lga_coord[0])
    ]
    df = df[
        (lga_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != lga_coord[0])
    ]
    print(" New size after lgr airport: %d" % len(df))

    df = df[
        (sol_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != sol_coord[0])
    ]
    df = df[
        (sol_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != sol_coord[0])
    ]
    print(" New size after sol removed: %d" % len(df))

    print("Old size: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def late_night(row):
    return 1 if (row["hour"] >= 0 and row["hour"] <= 3) else 0


def night(row):
    return (
        1
        if ((row["weekday"] < 5) and ((row["hour"] >= 20) or (row["hour"] <= 5)))
        else 0
    )


def rush_hour(row):
    if ((row["hour"] <= 20) and (row["hour"] >= 16)) and (row["weekday"] < 5):
        return 1
    else:
        return 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    if dt.isna().any():
        dt2 = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        dt = dt.fillna(dt2)

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday

    df["pickup_datetime"] = df["pickup_datetime"].astype(str)

    df["night"] = df.apply(lambda x: night(x), axis=1)
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1)
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1)
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
    df["distance"] = np.sqrt(
        np.abs(df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
        + np.abs(df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
    )
    return df


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


def distanceP(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df = df[[id_column, prediction_column]]
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df), "cols:", list(df.columns))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    if "rmse" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["rmse"])
        plt.plot(history.history["val_rmse"])
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper right")
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
    TEST_PATH, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3345740275.py in <cell line: 0>()
     11 
     12 trainKaggle = pd.read_csv(
---> 13     TRAIN_PATH,
     14     nrows=DATASET_SIZE,
     15     dtype=datatypes,

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
train_df.describe()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
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
print("test_df add_coordinate_features Disabled!")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3943412115.py in <cell line: 0>()
      1 print("train_df add_coordinate_features")
----> 2 train_df = add_coordinate_features(train_df)
      3 print("test_df add_coordinate_features Disabled!")
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
/tmp/ipykernel_11/470363350.py in <cell line: 0>()
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
_ = train_df.iloc[:2000].plot.scatter("latdiff", "londiff")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "weekday")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "night")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "late_night")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "rush_hour")
plt.show()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1506980898.py in <cell line: 0>()
----> 1 _ = train_df.iloc[:2000].plot.scatter("latdiff", "londiff")
      2 _ = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
      3 _ = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
      4 _ = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
      5 _ = train_df.iloc[:2000].plot.scatter("fare_amount", "day")

NameError: name 'train_df' is not defined

## === cell 16
dropped_columns = ["passenger_count", "pickup_datetime"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")
print("Feature columns after drop:")
print("train_df columns:", list(train_df.columns))
print("test_df columns:", list(test_df.columns))
print("testKaggle_clean columns:", list(testKaggle_clean.columns))



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2329749185.py in <cell line: 0>()
      1 dropped_columns = ["passenger_count", "pickup_datetime"]
      2 
----> 3 train_df = train_df.drop(dropped_columns, axis=1)
      4 test_df = test_df.drop(dropped_columns, axis=1)
      5 

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
train_df.describe()



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

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
train_labels = train_df["fare_amount"].values.astype("float32").reshape(-1, 1)
validation_labels = validation_df["fare_amount"].values.astype("float32").reshape(-1, 1)
test_labels = test_df["fare_amount"].values.astype("float32").reshape(-1, 1)

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

label_scaler = preprocessing.MinMaxScaler()
train_labels_scaled = label_scaler.fit_transform(train_labels).astype("float32")
validation_labels_scaled = label_scaler.transform(validation_labels).astype("float32")
test_labels_scaled = label_scaler.transform(test_labels).astype("float32")

print("Done with Labels (with target scaling)")
print("label_scaler min_:", label_scaler.data_min_, "max_:", label_scaler.data_max_)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/705118111.py in <cell line: 0>()
----> 1 train_labels = train_df["fare_amount"].values.astype("float32").reshape(-1, 1)
      2 validation_labels = validation_df["fare_amount"].values.astype("float32").reshape(-1, 1)
      3 test_labels = test_df["fare_amount"].values.astype("float32").reshape(-1, 1)
      4 
      5 train_df = train_df.drop(["fare_amount"], axis=1)

NameError: name 'train_df' is not defined

## === cell 24
test_labels.flatten()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/372165899.py in <cell line: 0>()
----> 1 test_labels.flatten()
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
for _df_name, _df in [
    ("train_df", train_df),
    ("validation_df", validation_df),
    ("test_df", test_df),
    ("testKaggle_clean", testKaggle_clean),
]:
    if "key" in _df.columns:
        print("Dropping string ID column 'key' from", _df_name)
        _df.drop(["key"], axis=1, inplace=True)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)

print(
    "Scaled shapes:",
    "train",
    train_df_scaled.shape,
    "val",
    validation_df_scaled.shape,
    "test",
    test_scaled.shape,
    "kaggle_test",
    testKaggle_scaled.shape,
)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2940917777.py in <cell line: 0>()
      1 for _df_name, _df in [
----> 2     ("train_df", train_df),
      3     ("validation_df", validation_df),
      4     ("test_df", test_df),
      5     ("testKaggle_clean", testKaggle_clean),

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
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))




## === cell 31
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
    y=train_labels_scaled,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(validation_df_scaled, validation_labels_scaled),
    shuffle=True,
)



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1353430465.py in <cell line: 0>()
----> 1 model = Sequential()
      2 model.add(
      3     Dense(
      4         256,
      5         activation="linear",

NameError: name 'Sequential' is not defined

## === cell 32
print("Training finished; history keys:", list(history.history.keys()))



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2791235592.py in <cell line: 0>()
----> 1 print("Training finished; history keys:", list(history.history.keys()))
      2 

NameError: name 'history' is not defined

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
train_pred_scaled = model.predict(train_df_scaled, batch_size=1024, verbose=0)
train_pred = label_scaler.inverse_transform(train_pred_scaled).reshape(-1)
train_true = train_labels.reshape(-1)

train_rmse_dollars = np.sqrt(np.mean((train_pred - train_true) ** 2))
train_mae_dollars = np.mean(np.abs(train_pred - train_true))
print("Train RMSE ($):", train_rmse_dollars)
print("Train MAE ($):", train_mae_dollars)



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3355987176.py in <cell line: 0>()
----> 1 train_pred_scaled = model.predict(train_df_scaled, batch_size=1024, verbose=0)
      2 train_pred = label_scaler.inverse_transform(train_pred_scaled).reshape(-1)
      3 train_true = train_labels.reshape(-1)
      4 
      5 train_rmse_dollars = np.sqrt(np.mean((train_pred - train_true) ** 2))

NameError: name 'model' is not defined

## === cell 35
val_pred_scaled = model.predict(validation_df_scaled, batch_size=1024, verbose=0)
val_pred = label_scaler.inverse_transform(val_pred_scaled).reshape(-1)
val_true = validation_labels.reshape(-1)

val_rmse_dollars = np.sqrt(np.mean((val_pred - val_true) ** 2))
val_mae_dollars = np.mean(np.abs(val_pred - val_true))
print("Validation RMSE ($):", val_rmse_dollars)
print("Validation MAE ($):", val_mae_dollars)



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2206418772.py in <cell line: 0>()
----> 1 val_pred_scaled = model.predict(validation_df_scaled, batch_size=1024, verbose=0)
      2 val_pred = label_scaler.inverse_transform(val_pred_scaled).reshape(-1)
      3 val_true = validation_labels.reshape(-1)
      4 
      5 val_rmse_dollars = np.sqrt(np.mean((val_pred - val_true) ** 2))

NameError: name 'model' is not defined

## === cell 36
test_pred_scaled = model.predict(test_scaled, batch_size=1024, verbose=0)
test_pred = label_scaler.inverse_transform(test_pred_scaled).reshape(-1)
test_true = test_labels.reshape(-1)

test_rmse_dollars = np.sqrt(np.mean((test_pred - test_true) ** 2))
test_mae_dollars = np.mean(np.abs(test_pred - test_true))
print("Test RMSE ($):", test_rmse_dollars)
print("Test MAE ($):", test_mae_dollars)



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1821488242.py in <cell line: 0>()
----> 1 test_pred_scaled = model.predict(test_scaled, batch_size=1024, verbose=0)
      2 test_pred = label_scaler.inverse_transform(test_pred_scaled).reshape(-1)
      3 test_true = test_labels.reshape(-1)
      4 
      5 test_rmse_dollars = np.sqrt(np.mean((test_pred - test_true) ** 2))

NameError: name 'model' is not defined

## === cell 37
validation_predictions = val_pred

plt.scatter(val_true, validation_predictions, s=5)
plt.xlabel("True Values [$]")
plt.ylabel("Predictions [$]")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
plt.show()



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/313906435.py in <cell line: 0>()
----> 1 validation_predictions = val_pred
      2 
      3 plt.scatter(val_true, validation_predictions, s=5)
      4 plt.xlabel("True Values [$]")
      5 plt.ylabel("Predictions [$]")

NameError: name 'val_pred' is not defined

## === cell 38
test_predictions = test_pred

plt.scatter(test_true, test_predictions, s=5)
plt.xlabel("True Values [$]")
plt.ylabel("Predictions [$]")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
plt.show()



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1260269716.py in <cell line: 0>()
----> 1 test_predictions = test_pred
      2 
      3 plt.scatter(test_true, test_predictions, s=5)
      4 plt.xlabel("True Values [$]")
      5 plt.ylabel("Predictions [$]")

NameError: name 'test_pred' is not defined

## === cell 39
predictionKaggle_scaled = model.predict(testKaggle_scaled, batch_size=128, verbose=1)
predictionKaggle = label_scaler.inverse_transform(predictionKaggle_scaled)




## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2072401307.py in <cell line: 0>()
----> 1 predictionKaggle_scaled = model.predict(testKaggle_scaled, batch_size=128, verbose=1)
      2 predictionKaggle = label_scaler.inverse_transform(predictionKaggle_scaled)
      3 
      4 

NameError: name 'model' is not defined

## === cell 40
def rmse_np(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())




## === cell 41
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_true[np.argmax(test_predictions)])
test_df.iloc[np.argmax(test_predictions)]



## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2236871954.py in <cell line: 0>()
----> 1 print(np.argmax(test_predictions))
      2 print(test_predictions[np.argmax(test_predictions)])
      3 print(test_true[np.argmax(test_predictions)])
      4 test_df.iloc[np.argmax(test_predictions)]
      5 

NameError: name 'test_predictions' is not defined

## === cell 42
print(np.argmin(test_predictions))
print(test_predictions[np.argmin(test_predictions)])
print(test_true[np.argmin(test_predictions)])
test_df.iloc[np.argmin(test_predictions)]



## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/310406520.py in <cell line: 0>()
----> 1 print(np.argmin(test_predictions))
      2 print(test_predictions[np.argmin(test_predictions)])
      3 print(test_true[np.argmin(test_predictions)])
      4 test_df.iloc[np.argmin(test_predictions)]
      5 

NameError: name 'test_predictions' is not defined

## === cell 43
fig, ax = plt.subplots()
ax.scatter(test_true, test_predictions, s=5)
ax.plot(
    [test_true.min(), test_true.max()],
    [test_true.min(), test_true.max()],
    "k--",
    lw=2,
)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()



## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4061678569.py in <cell line: 0>()
      1 fig, ax = plt.subplots()
----> 2 ax.scatter(test_true, test_predictions, s=5)
      3 ax.plot(
      4     [test_true.min(), test_true.max()],
      5     [test_true.min(), test_true.max()],

NameError: name 'test_true' is not defined

## === cell 44
predictionKaggle = np.asarray(predictionKaggle).reshape(-1, 1)
predictionKaggle = np.clip(predictionKaggle, 0, None)

assert len(predictionKaggle) == len(testKaggle), (
    len(predictionKaggle),
    len(testKaggle),
)

output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
print(
    "Wrote:",
    SUBMISSION_NAME,
    "exists:",
    os.path.exists(SUBMISSION_NAME),
    "size:",
    os.path.getsize(SUBMISSION_NAME) if os.path.exists(SUBMISSION_NAME) else None,
)

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4204816180.py in <cell line: 0>()
----> 1 predictionKaggle = np.asarray(predictionKaggle).reshape(-1, 1)
      2 predictionKaggle = np.clip(predictionKaggle, 0, None)
      3 
      4 assert len(predictionKaggle) == len(testKaggle), (
      5     len(predictionKaggle),

NameError: name 'predictionKaggle' is not defined
