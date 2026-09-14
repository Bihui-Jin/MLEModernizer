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
Given time series of breaths, predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

The best submissions will take lung attributes compliance and resistance into account.

## Metric
Mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored.

## Submission Format
For each `id` in the test set, you must predict a value for the `pressure` variable. The file should contain a header and have the following format:

```
id,pressure
1,20
2,23
3,24
etc.
```

## Dataset
The ventilator data used in this competition was produced using a modified [open-source ventilator](https://pvp.readthedocs.io/) connected to an [artificial bellows test lung](https://www.ingmarmed.com/product/quicklung/) via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

![Ventilator diagram](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - globally-unique time step identifier across an entire file
- `breath_id` - globally-unique time step for breaths
- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.
- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.
- `time_step` - the actual time stamp.
- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.
- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        input/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        working/
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
```

-> data/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/ventilator-pressure-prediction/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> input/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> (stopped after 10 files for performance)

# 5. Target score

0.1388768686943292

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_tensor_float_32_execution(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for _g in gpus:
        tf.config.experimental.set_memory_growth(_g, True)
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

try:
    if not tf.config.list_physical_devices("GPU"):
        tf.config.threading.set_intra_op_parallelism_threads(
            max(1, os.cpu_count() // 2)
        )
        tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

rb = RobustScaler()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/373192805.py in <cell line: 0>()
     18 os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
     19 
---> 20 import tensorflow as tf
     21 from tensorflow import keras
     22 from tensorflow.keras import layers

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
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

train = pd.read_csv(
    TRAIN_PATH,
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
        "pressure": "float32",
    },
)
test = pd.read_csv(
    TEST_PATH,
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
    },
)




## === cell 2
assert (
    len(train) % 80 == 0
), "Train rows must be multiple of 80 (one breath = 80 time steps)."
assert (
    len(test) % 80 == 0
), "Test rows must be multiple of 80 (one breath = 80 time steps)."




## === cell 3
def add_features_fast(df: pd.DataFrame) -> None:
    n = len(df)
    n_breaths = n // 80

    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False).reshape(n_breaths, 80)

    u_in_lag1 = np.empty_like(u_in)
    u_in_lag1[:, 0] = 0.0
    u_in_lag1[:, 1:] = u_in[:, :-1]

    u_in_diff1 = u_in - u_in_lag1
    u_in_cumsum = np.cumsum(u_in, axis=1, dtype=np.float32)

    df["u_in_lag1"] = u_in_lag1.reshape(-1)
    df["u_in_diff1"] = u_in_diff1.reshape(-1)
    df["u_in_cumsum"] = u_in_cumsum.reshape(-1)


add_features_fast(train)
add_features_fast(test)




## === cell 4
targets = train["pressure"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80, 1)
test_id = test["id"].copy()

train.drop(columns=["id", "breath_id", "pressure", "time_step"], inplace=True)
test.drop(columns=["id", "breath_id", "time_step"], inplace=True)




## === cell 5
rb.fit(train)

train_new = rb.transform(train).astype(np.float32, copy=False)
test_new = rb.transform(test).astype(np.float32, copy=False)

train_re = train_new.reshape(-1, 80, train_new.shape[-1])
test_re = test_new.reshape(-1, 80, test_new.shape[-1])




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1199725517.py in <cell line: 0>()
----> 1 rb.fit(train)
      2 
      3 train_new = rb.transform(train).astype(np.float32, copy=False)
      4 test_new = rb.transform(test).astype(np.float32, copy=False)
      5 

NameError: name 'rb' is not defined

## === cell 6
def build_model():
    x_input = layers.Input(shape=([80, train_re.shape[-1]]))

    x1 = layers.Bidirectional(layers.LSTM(units=440, return_sequences=True))(x_input)
    x2 = layers.Bidirectional(
        layers.LSTM(
            units=360,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer="random_normal",
        )
    )(x1)
    x3 = layers.Bidirectional(
        layers.LSTM(
            units=240,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer="random_normal",
        )
    )(x2)
    x4 = layers.Bidirectional(
        layers.LSTM(
            units=180,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer="random_normal",
        )
    )(x3)
    x5 = layers.Bidirectional(
        layers.LSTM(
            units=100, return_sequences=True, kernel_initializer="random_normal"
        )
    )(x4)

    z2 = layers.Bidirectional(layers.GRU(units=240, return_sequences=True))(x2)

    z31 = layers.Multiply()([x3, z2])
    z31 = layers.BatchNormalization()(z31)
    z3 = layers.Bidirectional(layers.GRU(units=180, return_sequences=True))(z31)

    z41 = layers.Multiply()([x4, z3])
    z41 = layers.BatchNormalization()(z41)
    z4 = layers.Bidirectional(layers.GRU(units=100, return_sequences=True))(z41)

    z51 = layers.Multiply()([x5, z4])
    z51 = layers.BatchNormalization()(z51)
    z5 = layers.Bidirectional(layers.GRU(units=64, return_sequences=True))(z51)

    x = layers.Concatenate(axis=2)([x5, z2, z3, z4, z5])
    x = layers.Dense(units=64, activation="relu")(x)
    x_output = layers.Dense(units=1)(x)

    model = tf.keras.Model(inputs=x_input, outputs=x_output, name="Base_Model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanAbsoluteError(),
        steps_per_execution=128,
    )
    return model




## === cell 7
try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
    print("Running with TPU strategy.")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print(
        f"Running with default strategy (CPU/GPU). TPU not used: {type(e).__name__}: {e}"
    )




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1994540556.py in <cell line: 0>()
      1 try:
----> 2     tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
      3     tf.config.experimental_connect_to_cluster(tpu_resolver)

NameError: name 'tf' is not defined

During handling of the above exception, another exception occurred:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1994540556.py in <cell line: 0>()
      6     print("Running with TPU strategy.")
      7 except Exception as e:
----> 8     strategy = tf.distribute.get_strategy()
      9     print(
     10         f"Running with default strategy (CPU/GPU). TPU not used: {type(e).__name__}: {e}"

NameError: name 'tf' is not defined

## === cell 8
unique_pressures = np.unique(targets)
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = (unique_pressures[1] - unique_pressures[0]).item()
PRESSURE_MIN = sorted_pressures[0].item()
PRESSURE_MAX = sorted_pressures[-1].item()

PRESSURE_STEP, PRESSURE_MIN, PRESSURE_MAX




## === cell 9
EPOCH = 260
BATCH_SIZE = 512

reduce_lr = ReduceLROnPlateau(monitor="val_loss", verbose=1, factor=0.87, patience=8)

n_splits = 7
kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

test_fold_preds = []

AUTOTUNE = tf.data.AUTOTUNE

data_options = tf.data.Options()
data_options.experimental_deterministic = True

os.makedirs("./cache_arrays", exist_ok=True)
X_train_path = "./cache_arrays/train_re.npy"
y_path = "./cache_arrays/targets.npy"
X_test_path = "./cache_arrays/test_re.npy"

if not (
    os.path.exists(X_train_path)
    and os.path.exists(y_path)
    and os.path.exists(X_test_path)
):
    np.save(X_train_path, train_re)
    np.save(y_path, targets)
    np.save(X_test_path, test_re)

X_all = np.load(X_train_path, mmap_mode="r")
y_all = np.load(y_path, mmap_mode="r")
X_test_all = np.load(X_test_path, mmap_mode="r")

X_all_tf = tf.convert_to_tensor(X_all, dtype=tf.float32)
y_all_tf = tf.convert_to_tensor(y_all, dtype=tf.float32)
X_test_tf = tf.convert_to_tensor(X_test_all, dtype=tf.float32)


def make_ds_from_indices(indices, training=False):
    idx = tf.convert_to_tensor(indices, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices(idx).with_options(data_options)
    if training:
        ds = ds.shuffle(
            buffer_size=min(int(idx.shape[0]), 8192),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def _gather(i):
        return tf.gather(X_all_tf, i), tf.gather(y_all_tf, i)

    ds = ds.map(_gather, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def make_test_ds():
    ds = tf.data.Dataset.from_tensor_slices(X_test_tf).with_options(data_options)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds()

os.makedirs("./cache_weights", exist_ok=True)


class InMemoryBestWeights(tf.keras.callbacks.Callback):
    def __init__(self, monitor="val_loss", mode="min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best = np.inf if mode == "min" else -np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        improved = current < self.best if self.mode == "min" else current > self.best
        if improved:
            self.best = float(current)
            self.best_weights = self.model.get_weights()


with strategy.scope():
    all_idx = np.arange(X_all.shape[0], dtype=np.int32)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(all_idx, all_idx), start=1):
        print("-" * 30, ">", f"Fold {fold}", "<", "-" * 30)

        fold_w_path = f"./cache_weights/fold_{fold}.weights.h5"

        train_ds = make_ds_from_indices(train_idx, training=True)
        valid_ds = make_ds_from_indices(valid_idx, training=False)

        model = build_model()

        if os.path.exists(fold_w_path):
            model.load_weights(fold_w_path)
        else:
            best_mem = InMemoryBestWeights(monitor="val_loss", mode="min")

            history = model.fit(
                train_ds,
                validation_data=valid_ds,
                epochs=EPOCH,
                callbacks=[best_mem, reduce_lr],
                verbose=2,
            )

            if best_mem.best_weights is not None:
                model.set_weights(best_mem.best_weights)

            model.save_weights(fold_w_path)

        fold_pred = model.predict(test_ds, verbose=0).reshape(-1, 1)
        test_fold_preds.append(fold_pred)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3082607001.py in <cell line: 0>()
      2 BATCH_SIZE = 512
      3 
----> 4 reduce_lr = ReduceLROnPlateau(monitor="val_loss", verbose=1, factor=0.87, patience=8)
      5 
      6 n_splits = 7

NameError: name 'ReduceLROnPlateau' is not defined

## === cell 10
predictions_stack = np.concatenate(test_fold_preds, axis=1)  # (n_samples, n_folds)
mean_pre = np.mean(predictions_stack, axis=1)  # (n_samples,)

rounding_pre = (
    np.round((mean_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)

mean_pre.shape, clipped_pre.shape




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2413766499.py in <cell line: 0>()
----> 1 predictions_stack = np.concatenate(test_fold_preds, axis=1)  # (n_samples, n_folds)
      2 mean_pre = np.mean(predictions_stack, axis=1)  # (n_samples,)
      3 
      4 rounding_pre = (
      5     np.round((mean_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN

NameError: name 'test_fold_preds' is not defined

## === cell 11
submission_file = pd.read_csv(SAMPLE_SUB_PATH)

assert len(submission_file) == len(clipped_pre), "Submission length mismatch."

submission_file["pressure"] = clipped_pre.astype(np.float32)
submission_file.to_csv("submission.csv", index=False)

submission_file.head()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1347991711.py in <cell line: 0>()
      1 submission_file = pd.read_csv(SAMPLE_SUB_PATH)
      2 
----> 3 assert len(submission_file) == len(clipped_pre), "Submission length mismatch."
      4 
      5 submission_file["pressure"] = clipped_pre.astype(np.float32)

NameError: name 'clipped_pre' is not defined

## === cell 12
assert os.path.exists("submission.csv")
sub_check = pd.read_csv("submission.csv")
assert list(sub_check.columns) == ["id", "pressure"]
print("Wrote submission.csv with shape:", sub_check.shape)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/761393786.py in <cell line: 0>()
----> 1 assert os.path.exists("submission.csv")
      2 sub_check = pd.read_csv("submission.csv")
      3 assert list(sub_check.columns) == ["id", "pressure"]
      4 print("Wrote submission.csv with shape:", sub_check.shape)

AssertionError:
