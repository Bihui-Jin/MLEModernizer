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

3.916084767058486

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 652.5327) has done: 'I fix the TensorFlow/Keras import issues by using `tf.keras` only (removing the incompatible standalone `keras` wrappers that cause the `MessageFactory/GetPrototype` crash) and by replacing the deprecated `tf.train.RMSPropOptimizer` with `tf.keras.optimizers.RMSprop` to run on the current Kaggle runtime. I also fix the broken input paths (`../input/my-taxi-fare-data/...`) by reading from the provided `../input/new-york-city-taxi-fare-prediction/train.csv` and `test.csv`, and I compute the missing `distance` feature directly from lat/lon so the existing feature schema stays intact. To keep runtime under control while preserving the same training approach, I stream a few chunks from the huge train file (instead of trying to load it all) and keep the original “3 partitions” idea by using three chunks. Finally, I ensure normalization uses train-derived mean/std (not per-test) and write a valid `submission_file.csv` with columns `key,fare_amount`.'
- What this solution (achieved 770.52934) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TensorFlow (this resolves the `MessageFactory.GetPrototype` error in Kaggle environments). I also remove the unused `from tensorflow import keras` aliasing pitfalls and ensure we only use `tf.keras` consistently while keeping the exact same model architecture/training logic. Finally, I make the validation split deterministic (no shuffle change) and keep the output submission formatting unchanged so a valid `submission_file.csv` is always produced. These changes are execution/stability fixes and should dramatically reduce the RMSE from the broken run behavior toward the target.'
- What this solution (achieved 565.19999) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from running by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, applied before any TensorFlow import. I also remove an unnecessary sklearn import that can trigger extra heavy imports and isn’t used (score-neutral). With TensorFlow now importing reliably, the rest of your pipeline (chunked loading, distance feature, normalization, model, grid search, training, and CSV writing) can run end-to-end and produce a valid `submission_file.csv`. This should also bring RMSE back down substantially from the broken-run behavior toward the target band.'
- What this solution (achieved 421.36355) has done: 'I fix the immediate runtime crash happening before training by forcing a compatible protobuf version in the Kaggle environment and importing TensorFlow only after that adjustment. This is a stability fix (no model/feature/training changes) so it should restore end-to-end execution and prevent the pathological RMSE you’re seeing from broken runs. I also remove the unused sklearn import in the first cell to avoid triggering heavy optional dependency paths during interpreter startup. Finally, I keep all paths, features, training loop, and submission-writing logic the same so the pipeline produces `submission_file.csv` in the required format.'
- What this solution (achieved 759.6716) has done: 'I fix the TensorFlow/protobuf import crash by removing the brittle protobuf version pre-check and forcing the pure-Python protobuf backend *before* any TensorFlow-related import, which is the direct cause of the `MessageFactory.GetPrototype` failure. I also remove the unused `GridSearchCV` import from the TensorFlow import cell (it isn’t needed there and can trigger extra heavy imports before TensorFlow stabilizes). These changes are execution/stability fixes and keep your model architecture, features, training loop, and submission formatting identical, so score behavior should return to normal (and move much closer to the target RMSE than the current broken run). Finally, I keep all paths and the output filename `submission_file.csv` unchanged to ensure Kaggle accepts the submission.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_USE_C_DESCRIPTORS", "1")

import math
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

print("Python OK")
print("TensorFlow:", tf.__version__)

np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3995999111.py in <cell line: 0>()
     15 import pandas as pd
     16 
---> 17 import tensorflow as tf
     18 from tensorflow.keras import layers
     19 

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
TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"

assert os.path.exists(TRAIN_PATH), f"Missing: {TRAIN_PATH}"
assert os.path.exists(TEST_PATH), f"Missing: {TEST_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"




## === cell 2
def add_distance_feature(df):
    """
    Bug fix: original code expects a 'distance' column, but raw NYC taxi data does not contain it.
    Minimal logic addition: compute a simple haversine distance (km) from pickup/dropoff lat/lon.
    """
    for c in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    R = 6371.0
    lat1 = np.radians(df["pickup_latitude"].values)
    lon1 = np.radians(df["pickup_longitude"].values)
    lat2 = np.radians(df["dropoff_latitude"].values)
    lon2 = np.radians(df["dropoff_longitude"].values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    dist_km = R * c

    df["distance"] = dist_km
    return df


def clean_train_df(df):
    """
    Minimal cleaning to avoid NaNs/infs and obvious outliers that break training.
    Keeps core model/training the same; only ensures numeric stability.
    """
    df = df.dropna(
        subset=[
            "fare_amount",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
    ).copy()

    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 250)]

    df = df[
        (df["pickup_longitude"].between(-74.5, -72.5))
        & (df["dropoff_longitude"].between(-74.5, -72.5))
        & (df["pickup_latitude"].between(40.0, 41.8))
        & (df["dropoff_latitude"].between(40.0, 41.8))
        & (df["passenger_count"].between(1, 6))
    ]
    return df




## === cell 3
def data_to_np_from_df(df):
    header_names = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
    ]
    df_train = df[header_names]
    np_df_train = df_train.values.astype("float32")

    df_label = df["fare_amount"]
    np_df_label = df_label.values.astype("float32")

    return np_df_train, np_df_label


def load_train_partition(part_idx, chunksize=300_000, max_chunks=1):
    """
    Bug fix: original code referenced non-existent pre-split files train_r0/1/2.
    Replacement keeps the 3-partition structure by reading successive chunks.
    """
    usecols = [
        "fare_amount",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    skiprows = None
    if part_idx > 0:
        skip_n = 1 + part_idx * chunksize * max_chunks
        skiprows = range(1, skip_n)

    reader = pd.read_csv(
        TRAIN_PATH,
        usecols=usecols,
        chunksize=chunksize,
        skiprows=skiprows,
        low_memory=False,
    )

    dfs = []
    for i, chunk in enumerate(reader):
        chunk = clean_train_df(chunk)
        chunk = add_distance_feature(chunk)
        dfs.append(chunk)
        if i + 1 >= max_chunks:
            break

    if len(dfs) == 0:
        raise RuntimeError(
            f"No data loaded for partition {part_idx}. Try adjusting chunksize/max_chunks."
        )

    df = pd.concat(dfs, axis=0, ignore_index=True)
    return data_to_np_from_df(df)




## === cell 4
def global_mean_per_column(mynp_train_list):
    sum_mean = 0.0
    for con in range(len(mynp_train_list)):
        sum_mean = sum_mean + np.mean(mynp_train_list[con], axis=0)
    mean = sum_mean / len(mynp_train_list)
    return mean


def global_std_per_column(mynp_train_list, global_mean):
    sum_mean_x2 = 0.0
    for con in range(len(mynp_train_list)):
        sum_mean_x2 += np.mean((mynp_train_list[con] - global_mean) ** 2, axis=0)
    std = np.sqrt(sum_mean_x2 / len(mynp_train_list))
    std = np.where(std == 0, 1.0, std)
    return std


def norm_mynp_train(mynp_train, mean, std):
    mynp_train_norm = (mynp_train - mean) / std
    return mynp_train_norm




## === cell 5
mynp_train_0, mynp_label_0 = load_train_partition(
    part_idx=0, chunksize=300_000, max_chunks=1
)
mynp_train_1, mynp_label_1 = load_train_partition(
    part_idx=1, chunksize=300_000, max_chunks=1
)
mynp_train_2, mynp_label_2 = load_train_partition(
    part_idx=2, chunksize=300_000, max_chunks=1
)

print(mynp_train_0.shape, mynp_label_0.shape)
print(mynp_train_1.shape, mynp_label_1.shape)
print(mynp_train_2.shape, mynp_label_2.shape)



## === cell 6
rng = np.random.default_rng(42)

for i, (X, y) in enumerate(
    [
        (mynp_train_0, mynp_label_0),
        (mynp_train_1, mynp_label_1),
        (mynp_train_2, mynp_label_2),
    ]
):
    order = rng.permutation(len(y))
    if i == 0:
        mynp_train_0, mynp_label_0 = X[order], y[order]
    elif i == 1:
        mynp_train_1, mynp_label_1 = X[order], y[order]
    else:
        mynp_train_2, mynp_label_2 = X[order], y[order]



## === cell 7
mynp_train_list = [mynp_train_0, mynp_train_1, mynp_train_2]
mynp_label_list = [mynp_label_0, mynp_label_1, mynp_label_2]

global_mean = global_mean_per_column(mynp_train_list)
global_std = global_std_per_column(mynp_train_list, global_mean)

print("global_mean:", global_mean)
print("global_std:", global_std)



## === cell 8
mynp_train_norm_0 = norm_mynp_train(mynp_train_list[0], global_mean, global_std)
mynp_train_norm_1 = norm_mynp_train(mynp_train_list[1], global_mean, global_std)
mynp_train_norm_2 = norm_mynp_train(mynp_train_list[2], global_mean, global_std)

mynp_train_concat = np.concatenate(
    (mynp_train_norm_0, mynp_train_norm_1, mynp_train_norm_2), axis=0
)
mynp_label_concat = np.concatenate((mynp_label_0, mynp_label_1, mynp_label_2), axis=0)

print("Train concat:", mynp_train_concat.shape, mynp_label_concat.shape)




## === cell 9
def build_model(shape_of_np_array):
    model = tf.keras.Sequential(
        [
            layers.Dense(64, activation=tf.nn.relu, input_shape=(shape_of_np_array,)),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(1),
        ]
    )

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])
    return model


class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            print("epoch", epoch)
        print(".")




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3269762462.py in <cell line: 0>()
     15 
     16 
---> 17 class PrintDot(tf.keras.callbacks.Callback):
     18     def on_epoch_end(self, epoch, logs=None):
     19         if epoch % 5 == 0:

NameError: name 'tf' is not defined

## === cell 10
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3177207670.py in <cell line: 0>()
----> 1 early_stop = tf.keras.callbacks.EarlyStopping(
      2     monitor="val_loss", patience=20, restore_best_weights=True
      3 )
      4 

NameError: name 'tf' is not defined

## === cell 11
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV


class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        shape_of_np_array,
        epochs=5,
        batch_size=64,
        validation_split=0.2,
        verbose=0,
    ):
        self.shape_of_np_array = int(shape_of_np_array)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.validation_split = float(validation_split)
        self.verbose = int(verbose)
        self.model_ = None

    def get_params(self, deep=True):
        return {
            "shape_of_np_array": self.shape_of_np_array,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "validation_split": self.validation_split,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y, callbacks=None):
        self.model_ = build_model(self.shape_of_np_array)
        cb = callbacks if callbacks is not None else []
        self.model_.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=self.verbose,
            callbacks=cb,
            shuffle=True,
        )
        return self

    def predict(self, X):
        pred = self.model_.predict(X, verbose=0).reshape(-1)
        return pred




## === cell 12
grid_train = mynp_train_concat[:1000]
grid_label = mynp_label_concat[:1000]
print(grid_train.shape, grid_label.shape)



## === cell 13
model_for_grid = KerasRegressorWrapper(
    shape_of_np_array=mynp_train_concat.shape[1], validation_split=0.2, verbose=0
)

epochs = [3, 5]
batches = [64, 128]
param_grid = dict(epochs=epochs, batch_size=batches)

grid = GridSearchCV(
    estimator=model_for_grid,
    param_grid=param_grid,
    n_jobs=1,
    scoring="neg_mean_absolute_error",
)

grid_result = grid.fit(grid_train, grid_label, callbacks=[early_stop, PrintDot()])



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3464424049.py in <cell line: 0>()
     14 )
     15 
---> 16 grid_result = grid.fit(grid_train, grid_label, callbacks=[early_stop, PrintDot()])
     17 

NameError: name 'early_stop' is not defined

## === cell 14
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2651115138.py in <cell line: 0>()
----> 1 print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
      2 means = grid_result.cv_results_["mean_test_score"]
      3 stds = grid_result.cv_results_["std_test_score"]
      4 params = grid_result.cv_results_["params"]
      5 for mean, stdev, param in zip(means, stds, params):

NameError: name 'grid_result' is not defined

## === cell 15
model_after_gridSearch = build_model(mynp_train_concat.shape[1])
EPOCHS = int(grid_result.best_params_["epochs"])
BATCH_SIZE = int(grid_result.best_params_["batch_size"])

print("Training final with:", EPOCHS, BATCH_SIZE)

history = model_after_gridSearch.fit(
    mynp_train_concat,
    mynp_label_concat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop, PrintDot()],
    shuffle=True,
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1313007201.py in <cell line: 0>()
----> 1 model_after_gridSearch = build_model(mynp_train_concat.shape[1])
      2 EPOCHS = int(grid_result.best_params_["epochs"])
      3 BATCH_SIZE = int(grid_result.best_params_["batch_size"])
      4 
      5 print("Training final with:", EPOCHS, BATCH_SIZE)

/tmp/ipykernel_11/3269762462.py in build_model(shape_of_np_array)
      1 def build_model(shape_of_np_array):
----> 2     model = tf.keras.Sequential(
      3         [
      4             layers.Dense(64, activation=tf.nn.relu, input_shape=(shape_of_np_array,)),
      5             layers.Dense(64, activation=tf.nn.relu),

NameError: name 'tf' is not defined

## === cell 16
df_test = pd.read_csv(TEST_PATH, sep=",", low_memory=False)
df_test = add_distance_feature(df_test)

df_test = df_test[
    [
        "key",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
    ]
].copy()

df_test_fn = df_test[
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
    ]
]

mynp_test = df_test_fn.values.astype("float32")
mynp_test = norm_mynp_train(mynp_test, global_mean, global_std)

print("Test shape:", mynp_test.shape)

test_predictions = (
    model_after_gridSearch.predict(mynp_test).reshape(-1).astype("float32")
)

test_predictions = np.clip(test_predictions, 0.0, None)

test_key_array = df_test["key"].values
df_output = pd.DataFrame({"key": test_key_array, "fare_amount": test_predictions})
print(df_output.head())

out_path = "submission_file.csv"
df_output.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("Columns:", df_output.columns.tolist())
print("Rows:", len(df_output))

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2588982399.py in <cell line: 0>()
     31 
     32 test_predictions = (
---> 33     model_after_gridSearch.predict(mynp_test).reshape(-1).astype("float32")
     34 )
     35 

NameError: name 'model_after_gridSearch' is not defined
