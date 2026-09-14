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

4.31009

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 111.90148) has done: 'I fix the crashes caused by (1) `plt.imread()` trying to read a URL (not supported, and internet may be blocked), (2) the Keras 3 optimizer API change (`optimizers.adam` no longer exists), and (3) an optional model-plot import that doesn’t exist in this environment. I keep your model and training loop intact, but make the water-mask step safely optional/fallback so cleaning doesn’t delete everything or error, and ensure the code always reaches submission writing. I also correct two time-feature boolean bugs (`late_night`, `night`) that currently make those features incorrect (even if you don’t use them later, this is score-positive and minimal). Finally, I update paths to the provided Kaggle dataset location and ensure the submission is written as a valid `.csv` with the required columns.'
- What this solution (achieved 5.55562) has done: 'I fix the immediate runtime crash happening before your own code runs by forcing the pure-Python protobuf implementation (this resolves the `MessageFactory.GetPrototype` AttributeError seen with some tf/keras/protobuf combos on Kaggle). Then I fix the biggest logic issue causing the very poor RMSE: your model is trained on a *different cleaned/train-split distribution* than the Kaggle test set, but you scale Kaggle test using a scaler fitted on that cleaned split without applying the same cleaning to Kaggle test; I apply the exact same `clean()` pipeline to `testKaggle` before feature engineering and scaling so features are aligned. Finally, I make submission creation robust to any rows dropped by cleaning by starting from `sample_submission.csv` and merging predictions by `key`, ensuring a valid submission with all required keys and `.csv` suffix.'
- What this solution (achieved 98.67914) has done: 'I fix the TensorFlow/Keras/protobuf crash by setting the protobuf env vars before any TF/Keras import and by avoiding the problematic default `MessageFactory` path. Then I make the Kaggle-test cleaning step score-safe by not dropping rows from `test.csv` (dropping test rows forces mean-filling later, which hurts RMSE), while keeping the exact same training cleaning logic and model/training loop intact. Finally, I ensure feature engineering produces no NaNs (from bad datetimes) and keep submission creation aligned to `sample_submission.csv` keys so the output is always valid and complete.'
- What this solution (achieved 121.96193) has done: 'I fix the immediate runtime crash caused by the protobuf/TensorFlow-Keras incompatibility by ensuring the environment variables are set before *any* keras/tf import and by avoiding importing `tf_keras` when it fails. Then I keep your exact model/training loop intact but switch to the built-in `tensorflow` + `tf.keras` backend (same layers/optimizer/loss/metrics) so the notebook runs end-to-end reliably in Kaggle. Finally, I keep your current “non-dropping” test cleaning and submission merge-by-key logic, but make the inference scaling robust to any column-order mismatch (a common silent bug that can severely hurt RMSE). These changes are execution/stability fixes and a small correctness fix for feature alignment, expected to improve score toward the target without changing the modeling approach.'
- What this solution (achieved 7.78002) has done: 'I fix the crash in the very first cell caused by a known TensorFlow/Keras + protobuf incompatibility by forcing the pure-Python protobuf implementation and disabling the C++ one *before* any TF import. Then I keep your exact model/training pipeline but ensure the runtime stays within Kaggle’s limits by preventing extremely slow/blocked operations (the remote water mask) from triggering long timeouts. Finally, to move RMSE strongly toward the target (current 121.96 → target 4.31, lower is better), I correct the submission prediction scale issue that typically causes huge errors: enforce non-negative fares and clip extreme predictions to the same target range used in training cleaning (0–50), which is a minimal, metric-aligned post-processing step.'
- What this solution (achieved 8.18471) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by pinning the pure-Python protobuf implementation and also forcing the legacy Python protos path before importing TensorFlow (this is the root runtime blocker). Then I keep your exact model/training pipeline intact but fix a key feature-alignment bug: `testKaggle_clean` is currently filled with zeros **after** reindexing, which can silently wipe whole columns if they were missing/invalid; I instead impute missing values in `testKaggle_clean` using the **training-feature medians** (score-positive and consistent with your scaling). Finally, I make submission writing robust and deterministic (ensuring float dtype, correct columns, and row count), without changing your architecture, epochs, batch size, optimizer, or loss.'
- What this solution (achieved 6.8187) has done: 'I fix the protobuf/TensorFlow crash by moving the protobuf environment variables to the very top and importing TensorFlow only after that, plus adding a safe fallback to `tf_keras` if the crash still occurs in this environment. Then I correct a subtle but score-impacting preprocessing mismatch: the Kaggle test set currently uses medians computed on *unscaled* features but is passed through a scaler fitted on the training set—this is correct, but we also need to ensure the Kaggle test columns are strictly numeric and ordered identically *before* any NaN filling and scaling to avoid silent dtype/object contamination. Finally, I keep your model/training exactly the same, but I make the submission writing deterministic and guaranteed complete (all keys, correct columns, `.csv` suffix) without changing evaluation semantics.'
- What this solution (achieved 7.74861) has done: 'I fix the immediate runtime blocker (`MessageFactory.GetPrototype` protobuf issue) by forcing the pure-Python protobuf runtime and legacy/proto API *before* any TensorFlow/Keras import, and by explicitly importing `google.protobuf` early to ensure the env vars take effect. I also make the custom `rmse` metric import backend from the active Keras backend (tf.keras vs tf_keras) to avoid backend mismatches at runtime. Finally, to move RMSE down toward your target without changing the model/training loop, I fix a small but score-relevant preprocessing mismatch by applying the same “clip passenger_count to [1,6]” logic to train/validation/test splits as is already done for Kaggle test, keeping features consistent between train and inference.'
- What this solution (achieved 6.53962) has done: 'I fix the immediate runtime blocker in the first cell caused by an incompatible protobuf runtime by removing the forced pure-Python protobuf settings and instead forcing TensorFlow to use its bundled protobuf implementation (this is the standard Kaggle-safe workaround for the `MessageFactory.GetPrototype` error). I keep your model, training loop, and feature pipeline intact, but add a tiny, metric-aligned calibration step on predictions (shift by the validation-set residual mean) to reduce RMSE toward the target without changing architecture/training. I also make submission writing robust and deterministic (still using sample_submission merge-by-key) and keep the `.csv` suffix and required columns. All other logic is preserved.'
- What this solution (achieved 8.19089) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* any TensorFlow/Keras import (this is the most reliable workaround for the `MessageFactory.GetPrototype` error in Kaggle’s mixed TF/Keras environments). Then I keep your model/training loop intact, but make one minimal score-positive correction: apply the same numeric/median imputation used for Kaggle test using **training medians** (not test medians) so inference preprocessing matches training distribution more closely. Finally, I keep your submission merge-by-key logic but ensure the script always writes a valid `.csv` with the required columns even if anything upstream produces NaNs.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.pop("TF_PROTOBUF_USE_PYTHON", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
    from tensorflow.keras import optimizers, regularizers
    from tensorflow.keras.callbacks import ModelCheckpoint

    _KERAS_BACKEND = "tf.keras"
except Exception as e:
    print(
        "Warning: tensorflow import failed, trying tf_keras fallback. Error:", repr(e)
    )
    import tensorflow as tf  # still needed for random seed
    import tf_keras as keras
    from tf_keras.models import Sequential
    from tf_keras.layers import Dense, Dropout, BatchNormalization, LSTM
    from tf_keras import optimizers, regularizers
    from tf_keras.callbacks import ModelCheckpoint

    _KERAS_BACKEND = "tf_keras"

np.random.seed(1)
tf.random.set_seed(1)

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SAMPLE_SUB_PATH = (
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 1000
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 800000

print("Keras backend:", _KERAS_BACKEND)
print("Using TRAIN_PATH:", TRAIN_PATH)
print("Using TEST_PATH:", TEST_PATH)
print("Using SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("Will write submission to:", os.path.abspath(SUBMISSION_NAME))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/210213475.py in <cell line: 0>()
     21 try:
---> 22     import tensorflow as tf
     23     from tensorflow import keras

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

During handling of the above exception, another exception occurred:

ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/210213475.py in <cell line: 0>()
     32         "Warning: tensorflow import failed, trying tf_keras fallback. Error:", repr(e)
     33     )
---> 34     import tensorflow as tf  # still needed for random seed
     35     import tf_keras as keras
     36     from tf_keras.models import Sequential

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
import urllib.request
from PIL import Image


def remove_datapoints_from_water(df):
    """
    Tries to fetch an NYC land mask; if unavailable (offline), skip water filtering.
    """

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)
    url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"

    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            nyc_mask_img = Image.open(resp)
            nyc_mask = np.array(nyc_mask_img)[:, :, 0] > 0.9
    except Exception as e:
        print(
            "Warning: could not load NYC water mask (offline or blocked). Skipping water filtering. Error:",
            repr(e),
        )
        return df

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
    if "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    if "passenger_count" in df.columns:
        df["passenger_count"] = df["passenger_count"].clip(lower=1, upper=6)

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


def clean_test_no_drop(df):
    print(" Old size (test): %d" % len(df))
    df = df.copy()

    num_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "passenger_count" in df.columns:
        df["passenger_count"] = df["passenger_count"].fillna(1)
        df["passenger_count"] = df["passenger_count"].clip(lower=1, upper=6)

    for c in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]:
        if c in df.columns:
            med = df[c].median(skipna=True)
            df[c] = df[c].fillna(med)

    print(" New size (test) after non-dropping clean: %d" % len(df))
    return df


def late_night(row):
    return 1 if (row["hour"] <= 3) else 0


def night(row):
    return 1 if ((row["hour"] > 20 or row["hour"] < 6) and (row["weekday"] < 5)) else 0


def rush_hour(row):
    return (
        1
        if ((row["hour"] <= 20) and (row["hour"] >= 16) and (row["weekday"] < 5))
        else 0
    )


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=False
    )
    df["year"] = df["pickup_datetime"].dt.year.fillna(0).astype(np.int16)
    df["month"] = df["pickup_datetime"].dt.month.fillna(0).astype(np.int8)
    df["day"] = df["pickup_datetime"].dt.day.fillna(0).astype(np.int8)
    df["hour"] = df["pickup_datetime"].dt.hour.fillna(0).astype(np.int8)
    df["weekday"] = df["pickup_datetime"].dt.weekday.fillna(0).astype(np.int8)
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
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
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))


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

usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
trainKaggle = pd.read_csv(
    TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=usecols_train
)
testKaggle = pd.read_csv(
    TEST_PATH, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
)

print("Loaded trainKaggle:", trainKaggle.shape, "testKaggle:", testKaggle.shape)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2896621517.py in <cell line: 0>()
     21 ]
     22 trainKaggle = pd.read_csv(
---> 23     TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=usecols_train
     24 )
     25 testKaggle = pd.read_csv(

NameError: name 'TRAIN_PATH' is not defined

## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000].copy()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4083304854.py in <cell line: 0>()
----> 1 train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
      2 test_df = test_df[:10000].copy()
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

print("testKaggle clean (non-dropping)")
testKaggle = clean_test_no_drop(testKaggle)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/653270819.py in <cell line: 0>()
      1 print("train_df clean")
----> 2 train_df = clean(train_df)
      3 test_df = clean(test_df)
      4 
      5 print("testKaggle clean (non-dropping)")

NameError: name 'train_df' is not defined

## === cell 8
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1145226127.py in <cell line: 0>()
      1 print("train_df add_time_features")
----> 2 train_df = add_time_features(train_df)
      3 print("test_df add_time_features")
      4 test_df = add_time_features(test_df)
      5 print("testKaggle add_time_features")

NameError: name 'train_df' is not defined

## === cell 9
train_df.describe()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 10
print("train_df add_coordinate_features")
add_coordinate_features(train_df)
print("test_df add_coordinate_features Disabled!")
add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
add_coordinate_features(testKaggle)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2476012946.py in <cell line: 0>()
      1 print("train_df add_coordinate_features")
----> 2 add_coordinate_features(train_df)
      3 print("test_df add_coordinate_features Disabled!")
      4 add_coordinate_features(test_df)
      5 print("testKaggle add_coordinate_features")

NameError: name 'train_df' is not defined

## === cell 11
train_df.describe()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 12
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/470363350.py in <cell line: 0>()
      1 print("train_df add_distances_features")
----> 2 train_df = add_distances_features(train_df)
      3 print("test_df add_distances_features")
      4 test_df = add_distances_features(test_df)
      5 print("testKaggle add_distances_features")

NameError: name 'train_df' is not defined

## === cell 13
train_df.describe()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 14
try:
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")
    _ = train_df.iloc[:2000].plot.scatter("fare_amount", "weekday")
except Exception as e:
    print("Plotting skipped:", repr(e))



## === cell 15
dropped_columns = ["pickup_datetime"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4151285212.py in <cell line: 0>()
      1 dropped_columns = ["pickup_datetime"]
      2 
----> 3 train_df = train_df.drop(dropped_columns, axis=1)
      4 test_df = test_df.drop(dropped_columns, axis=1)
      5 

NameError: name 'train_df' is not defined

## === cell 16
train_df.shape



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/186828624.py in <cell line: 0>()
----> 1 train_df.shape
      2 

NameError: name 'train_df' is not defined

## === cell 17
train_df.describe()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 18
test_df.describe()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294142452.py in <cell line: 0>()
----> 1 test_df.describe()
      2 

NameError: name 'test_df' is not defined

## === cell 19
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1154705564.py in <cell line: 0>()
----> 1 train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)
      2 

NameError: name 'train_df' is not defined

## === cell 20
train_df_main = train_df
validation_df_main = validation_df



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2901810251.py in <cell line: 0>()
----> 1 train_df_main = train_df
      2 validation_df_main = validation_df
      3 

NameError: name 'train_df' is not defined

## === cell 21
validation_df.describe()



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3277078349.py in <cell line: 0>()
----> 1 validation_df.describe()
      2 

NameError: name 'validation_df' is not defined

## === cell 22
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

drop_target_and_id = ["fare_amount", "key"]
train_df = train_df.drop(drop_target_and_id, axis=1)
validation_df = validation_df.drop(drop_target_and_id, axis=1)
test_df = test_df.drop(drop_target_and_id, axis=1)

print("Done with Labels")



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1981755008.py in <cell line: 0>()
----> 1 train_labels = train_df["fare_amount"].values
      2 validation_labels = validation_df["fare_amount"].values
      3 test_labels = test_df["fare_amount"].values
      4 
      5 drop_target_and_id = ["fare_amount", "key"]

NameError: name 'train_df' is not defined

## === cell 23
test_labels



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/805156543.py in <cell line: 0>()
----> 1 test_labels
      2 

NameError: name 'test_labels' is not defined

## === cell 24
train_df.describe()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 25
validation_df.describe()



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3277078349.py in <cell line: 0>()
----> 1 validation_df.describe()
      2 

NameError: name 'validation_df' is not defined

## === cell 26
test_df.describe()



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294142452.py in <cell line: 0>()
----> 1 test_df.describe()
      2 

NameError: name 'test_df' is not defined

## === cell 27
train_feature_cols = list(train_df.columns)

testKaggle_clean = testKaggle_clean.reindex(columns=train_feature_cols)
testKaggle_clean = testKaggle_clean.apply(pd.to_numeric, errors="coerce")

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)

train_feature_medians = train_df.median(numeric_only=True)

testKaggle_clean = testKaggle_clean.fillna(train_feature_medians)

testKaggle_clean = testKaggle_clean.fillna(0)

testKaggle_scaled = scaler.transform(testKaggle_clean)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2873051234.py in <cell line: 0>()
----> 1 train_feature_cols = list(train_df.columns)
      2 
      3 testKaggle_clean = testKaggle_clean.reindex(columns=train_feature_cols)
      4 testKaggle_clean = testKaggle_clean.apply(pd.to_numeric, errors="coerce")
      5 

NameError: name 'train_df' is not defined

## === cell 28
test_scaled



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1145478310.py in <cell line: 0>()
----> 1 test_scaled
      2 

NameError: name 'test_scaled' is not defined

## === cell 29
if _KERAS_BACKEND == "tf_keras":
    from tf_keras import backend
else:
    from tensorflow.keras import backend


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))




## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/165367508.py in <cell line: 0>()
----> 1 if _KERAS_BACKEND == "tf_keras":
      2     from tf_keras import backend
      3 else:
      4     from tensorflow.keras import backend
      5 

NameError: name '_KERAS_BACKEND' is not defined

## === cell 30
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
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(
    loss="mean_squared_error", optimizer=adam, metrics=["mae", "accuracy", rmse, "mse"]
)

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



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3904324871.py in <cell line: 0>()
----> 1 checkpoint = ModelCheckpoint(filepath="my_model.h5", verbose=1, save_best_only=True)
      2 
      3 model = Sequential()
      4 model.add(
      5     Dense(

NameError: name 'ModelCheckpoint' is not defined

## === cell 31
print("Skipping model_to_dot visualization (module not available in this environment).")



## === cell 32
plot_loss_accuracy_rmse(history)



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2579130885.py in <cell line: 0>()
----> 1 plot_loss_accuracy_rmse(history)
      2 

NameError: name 'history' is not defined

## === cell 33
score = model.evaluate(train_df_scaled, train_labels, verbose=1)
print(score)
print("train mean_squared_error:", score[0])
print("train mae:", score[1])
print("train accuracy:", score[2])
print("train rmse:", score[3])
print("train mse:", score[4])



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1372304874.py in <cell line: 0>()
----> 1 score = model.evaluate(train_df_scaled, train_labels, verbose=1)
      2 print(score)
      3 print("train mean_squared_error:", score[0])
      4 print("train mae:", score[1])
      5 print("train accuracy:", score[2])

NameError: name 'model' is not defined

## === cell 34
score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print(score)
print("Validation mean_squared_error:", score[0])
print("Validation mae:", score[1])
print("Validation accuracy:", score[2])
print("Validation rmse:", score[3])
print("Validation mse:", score[4])



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4130336514.py in <cell line: 0>()
----> 1 score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
      2 print(score)
      3 print("Validation mean_squared_error:", score[0])
      4 print("Validation mae:", score[1])
      5 print("Validation accuracy:", score[2])

NameError: name 'model' is not defined

## === cell 35
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print("Test mean_squared_error:", score[0])
print("Test mae:", score[1])
print("Test accuracy:", score[2])
print("Test rmse:", score[3])
print("Test mse:", score[4])



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2752207880.py in <cell line: 0>()
----> 1 score = model.evaluate(test_scaled, test_labels, verbose=1)
      2 print(score)
      3 print("Test mean_squared_error:", score[0])
      4 print("Test mae:", score[1])
      5 print("Test accuracy:", score[2])

NameError: name 'model' is not defined

## === cell 36
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



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2392837.py in <cell line: 0>()
----> 1 validation_predictions = model.predict(validation_df_scaled).flatten()
      2 
      3 plt.scatter(validation_labels, validation_predictions)
      4 plt.xlabel("True Values")
      5 plt.ylabel("Predictions")

NameError: name 'model' is not defined

## === cell 37
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



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/481671663.py in <cell line: 0>()
----> 1 test_predictions = model.predict(test_scaled).flatten()
      2 
      3 plt.scatter(test_labels, test_predictions)
      4 plt.xlabel("True Values")
      5 plt.ylabel("Predictions")

NameError: name 'model' is not defined

## === cell 38
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_labels[np.argmax(test_predictions)])
test_df.iloc[np.argmax(test_predictions)]



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3782083718.py in <cell line: 0>()
----> 1 print(np.argmax(test_predictions))
      2 print(test_predictions[np.argmax(test_predictions)])
      3 print(test_labels[np.argmax(test_predictions)])
      4 test_df.iloc[np.argmax(test_predictions)]
      5 

NameError: name 'test_predictions' is not defined

## === cell 39
print(np.argmin(test_predictions))
print(test_predictions[np.argmin(test_predictions)])
print(test_labels[np.argmin(test_predictions)])
test_df.iloc[np.argmin(test_predictions)]



## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1775569351.py in <cell line: 0>()
----> 1 print(np.argmin(test_predictions))
      2 print(test_predictions[np.argmin(test_predictions)])
      3 print(test_labels[np.argmin(test_predictions)])
      4 test_df.iloc[np.argmin(test_predictions)]
      5 

NameError: name 'test_predictions' is not defined

## === cell 40
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



## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3918075604.py in <cell line: 0>()
      1 fig, ax = plt.subplots()
----> 2 ax.scatter(test_labels, test_predictions)
      3 ax.plot(
      4     [test_labels.min(), test_labels.max()],
      5     [test_labels.min(), test_labels.max()],

NameError: name 'test_labels' is not defined

## === cell 41
plt.figure(figsize=(20, 10))
plt.plot(validation_labels[:1000])
plt.plot(validation_predictions[:1000])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()

validation_df_scaled[52]
validation_df.iloc[52]



## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2280632997.py in <cell line: 0>()
      1 plt.figure(figsize=(20, 10))
----> 2 plt.plot(validation_labels[:1000])
      3 plt.plot(validation_predictions[:1000])
      4 plt.title("Prediction vs Actual")
      5 plt.ylabel("Fare Amount")

NameError: name 'validation_labels' is not defined

## === cell 42
plt.figure(figsize=(20, 10))
plt.plot(test_labels[:100])
plt.plot(test_predictions[:100])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()



## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1260486629.py in <cell line: 0>()
      1 plt.figure(figsize=(20, 10))
----> 2 plt.plot(test_labels[:100])
      3 plt.plot(test_predictions[:100])
      4 plt.title("Prediction vs Actual")
      5 plt.ylabel("Fare Amount")

NameError: name 'test_labels' is not defined

## === cell 43
error = validation_predictions - validation_labels
plt.hist(error, bins=100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1386752517.py in <cell line: 0>()
----> 1 error = validation_predictions - validation_labels
      2 plt.hist(error, bins=100)
      3 plt.xlabel("Prediction Error")
      4 _ = plt.ylabel("Count")
      5 

NameError: name 'validation_predictions' is not defined

## === cell 44
error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/281852328.py in <cell line: 0>()
----> 1 error = test_predictions - test_labels
      2 plt.hist(error, bins=50)
      3 plt.xlabel("Prediction Error")
      4 _ = plt.ylabel("Count")
      5 

NameError: name 'test_predictions' is not defined

## === cell 45
print(len(error))

errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
print(len(errorGreaterZero))

plt.hist(errorGreaterZero, bins=100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2451681468.py in <cell line: 0>()
----> 1 print(len(error))
      2 
      3 errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
      4 print(len(errorGreaterZero))
      5 

NameError: name 'error' is not defined

## === cell 46
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/522070667.py in <cell line: 0>()
----> 1 predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)
      2 

NameError: name 'model' is not defined

## === cell 47
val_residual_mean = float(np.mean(validation_predictions - validation_labels))
print("Validation residual mean (pred - true):", val_residual_mean)

predictionKaggle = np.asarray(predictionKaggle).reshape(-1, 1).astype(np.float32)
predictionKaggle = predictionKaggle - np.float32(val_residual_mean)

predictionKaggle = np.clip(predictionKaggle, 0.0, 50.0)

pred_df = pd.DataFrame(
    {"key": testKaggle["key"].values, "fare_amount": predictionKaggle.flatten()}
)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sub = sample_sub.merge(pred_df, on="key", how="left", suffixes=("_sample", ""))

if sub["fare_amount"].isna().any():
    fill_value = float(sample_sub["fare_amount"].mean())
    sub["fare_amount"] = sub["fare_amount"].fillna(fill_value)

sub["fare_amount"] = pd.to_numeric(sub["fare_amount"], errors="coerce").astype(
    np.float32
)
sub["fare_amount"] = (
    sub["fare_amount"]
    .fillna(float(sample_sub["fare_amount"].mean()))
    .astype(np.float32)
)

sub = sub[["key", "fare_amount"]]
sub.to_csv(SUBMISSION_NAME, index=False)
print("Output complete:", SUBMISSION_NAME, "rows:", len(sub))

print("Submission head:")
print(pd.read_csv(SUBMISSION_NAME).head())

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/826863444.py in <cell line: 0>()
----> 1 val_residual_mean = float(np.mean(validation_predictions - validation_labels))
      2 print("Validation residual mean (pred - true):", val_residual_mean)
      3 
      4 predictionKaggle = np.asarray(predictionKaggle).reshape(-1, 1).astype(np.float32)
      5 predictionKaggle = predictionKaggle - np.float32(val_residual_mean)

NameError: name 'validation_predictions' is not defined
