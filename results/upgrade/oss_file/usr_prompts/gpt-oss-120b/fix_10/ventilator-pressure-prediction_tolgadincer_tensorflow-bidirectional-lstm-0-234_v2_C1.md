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

3.9

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.2363

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import mixed_precision
import multiprocessing as mp

mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)  # enable XLA JIT compilation

np.random.seed(2021)
tf.random.set_seed(2021)

mp.set_start_method("fork", force=True)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DEBUG = False

train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

if DEBUG:
    train = train[: 80 * 10000]




## === cell 2
def add_features(df):
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()
    df["u_in_cumsum"] = df["u_in"].groupby(df["breath_id"]).cumsum()
    df["u_in_lag"] = df["u_in"].shift(2).fillna(0)
    df = pd.get_dummies(df, columns=["R", "C"])
    return df


train = add_features(train)
test = add_features(test)




## === cell 3
targets = train[["pressure"]].to_numpy().reshape(-1, 80, 1)

train = train.drop(["pressure", "id", "breath_id"], axis=1)
test = test.drop(["id", "breath_id"], axis=1)




## === cell 4
from sklearn.preprocessing import RobustScaler

train = train.astype(np.float32)
test = test.astype(np.float32)

RS = RobustScaler()
train = RS.fit_transform(train)
test = RS.transform(test)




## === cell 5
train = train.astype(np.float16)
test = test.astype(np.float16)

train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, test.shape[-1])




## === cell 6
EPOCH = 30  # kept as‑is to preserve modelling intent
BATCH_SIZE = 4096  # larger batch size speeds up training without altering core logic


def get_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        return tf.distribute.experimental.TPUStrategy(tpu)
    except Exception:
        return tf.distribute.get_strategy()  # CPU / single GPU


def train_fold(args):
    fold, train_idx, valid_idx = args
    strategy = get_strategy()
    with strategy.scope():
        X_train, X_valid = train[train_idx], train[valid_idx]
        y_train, y_valid = targets[train_idx], targets[valid_idx]

        model = keras.models.Sequential(
            [
                keras.layers.Input(shape=train.shape[-2:]),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(300, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(250, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(150, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(100, return_sequences=True)
                ),
                keras.layers.Dense(50, activation="selu"),
                keras.layers.Dense(1),
            ]
        )

        steps_per_epoch = max(1, int(np.ceil(len(X_train) / BATCH_SIZE)))
        lr_schedule = ExponentialDecay(
            1e-3, decay_steps=steps_per_epoch, decay_rate=0.99, staircase=False
        )
        optimizer = Adam(learning_rate=lr_schedule)

        model.compile(optimizer=optimizer, loss="mae")

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = (
            train_ds.shuffle(buffer_size=1024)
            .batch(BATCH_SIZE, drop_remainder=True)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

        valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
        valid_ds = (
            valid_ds.batch(BATCH_SIZE, drop_remainder=True)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCH,
            verbose=0,
        )

        fold_pred = model.predict(test, batch_size=BATCH_SIZE).reshape(-1)

        tf.keras.backend.clear_session()
        return fold_pred


from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=2021)
splits = list(kf.split(train, targets))
args_list = [
    (i, train_idx, valid_idx) for i, (train_idx, valid_idx) in enumerate(splits)
]

with mp.Pool(processes=5) as pool:
    test_preds = pool.map(train_fold, args_list)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RemoteTraceback                           Traceback (most recent call last)
RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
                    ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/pool.py", line 48, in mapstar
    return list(map(*args))
           ^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/110206276.py", line 67, in train_fold
    model.fit(
  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py", line 59, in quick_execute
    except TypeError as e:
tensorflow.python.framework.errors_impl.InternalError: Graph execution error:

Detected at node gradient_tape/sequential_1/bidirectional_3_1/forward_lstm_3_1/CudnnRNNBackpropV3 defined at (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main

  File "<frozen runpy>", line 88, in _run_code

  File "/usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py", line 37, in <module>

  File "/usr/local/lib/python3.11/dist-packages/traitlets/config/application.py", line 992, in launch_instance

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelapp.py", line 712, in start

  File "/usr/local/lib/python3.11/dist-packages/tornado/platform/asyncio.py", line 211, in start

  File "/usr/lib/python3.11/asyncio/base_events.py", line 608, in run_forever

  File "/usr/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once

  File "/usr/lib/python3.11/asyncio/events.py", line 84, in _run

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 510, in dispatch_queue

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 499, in process_one

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 406, in dispatch_shell

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 730, in execute_request

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/ipkernel.py", line 383, in do_execute

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/zmqshell.py", line 528, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 2975, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3030, in _run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/async_helpers.py", line 78, in _pseudo_sync_runner

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3257, in run_cell_async

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3473, in run_ast_nodes

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code

  File "/tmp/ipykernel_55/110206276.py", line 89, in <cell line: 0>

  File "/usr/lib/python3.11/multiprocessing/context.py", line 119, in Pool

  File "/usr/lib/python3.11/multiprocessing/pool.py", line 215, in __init__

  File "/usr/lib/python3.11/multiprocessing/pool.py", line 306, in _repopulate_pool

  File "/usr/lib/python3.11/multiprocessing/pool.py", line 329, in _repopulate_pool_static

  File "/usr/lib/python3.11/multiprocessing/process.py", line 121, in start

  File "/usr/lib/python3.11/multiprocessing/context.py", line 281, in _Popen

  File "/usr/lib/python3.11/multiprocessing/popen_fork.py", line 19, in __init__

  File "/usr/lib/python3.11/multiprocessing/popen_fork.py", line 71, in _launch

  File "/usr/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap

  File "/usr/lib/python3.11/multiprocessing/process.py", line 108, in run

  File "/usr/lib/python3.11/multiprocessing/pool.py", line 125, in worker

  File "/usr/lib/python3.11/multiprocessing/pool.py", line 48, in mapstar

  File "/tmp/ipykernel_55/110206276.py", line 67, in train_fold

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 132, in multi_step_on_iterator

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 113, in one_step_on_data

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 77, in train_step

Failed to call DoRnnBackward with model config: [rnn_mode, rnn_input_mode, rnn_direction_mode]: 2, 0, 0 , [num_layers, input_size, num_units, dir_count, max_seq_length, batch_size, cell_num_units]: [1, 300, 100, 1, 80, 4096, 100] 
	 [[{{node gradient_tape/sequential_1/bidirectional_3_1/forward_lstm_3_1/CudnnRNNBackpropV3}}]] [Op:__inference_multi_step_on_iterator_9347]
"""

The above exception was the direct cause of the following exception:

InternalError                             Traceback (most recent call last)
/tmp/ipykernel_55/110206276.py in <cell line: 0>()
     88 # Parallel execution of the 5 folds
     89 with mp.Pool(processes=5) as pool:
---> 90     test_preds = pool.map(train_fold, args_list)
     91 
     92 

/usr/lib/python3.11/multiprocessing/pool.py in map(self, func, iterable, chunksize)
    365         in a list that is returned.
    366         '''
--> 367         return self._map_async(func, iterable, mapstar, chunksize).get()
    368 
    369     def starmap(self, func, iterable, chunksize=None):

/usr/lib/python3.11/multiprocessing/pool.py in get(self, timeout)
    772             return self._value
    773         else:
--> 774             raise self._value
    775 
    776     def _set(self, i, obj):

/usr/lib/python3.11/multiprocessing/pool.py in worker()
    123         job, i, func, args, kwds = task
    124         try:
--> 125             result = (True, func(*args, **kwds))
    126         except Exception as e:
    127             if wrap_exception and func is not _helper_reraises_exception:

/usr/lib/python3.11/multiprocessing/pool.py in mapstar()
     46 
     47 def mapstar(args):
---> 48     return list(map(*args))
     49 
     50 def starmapstar(args):

/tmp/ipykernel_55/110206276.py in train_fold()
     65         )
     66 
---> 67         model.fit(
     68             train_ds,
     69             validation_data=valid_ds,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler()
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute()
     51   try:
     52     ctx.ensure_initialized()
---> 53     tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
     54                                         inputs, attrs, num_outputs)
     55   except core._NotOkStatusException as e:

InternalError: Graph execution error:

Detected at node gradient_tape/sequential_1/bidirectional_3_1/forward_lstm_3_1/CudnnRNNBackpropV3 defined at (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main

  File "<frozen runpy>", line 88, in _run_code

  File "/usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py", line 37, in <module>

  File "/usr/local/lib/python3.11/dist-packages/traitlets/config/application.py", line 992, in launch_instance

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelapp.py", line 712, in start

  File "/usr/local/lib/python3.11/dist-packages/tornado/platform/asyncio.py", line 211, in start

  File "/usr/lib/python3.11/asyncio/base_events.py", line 608, in run_forever

  File "/usr/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once

  File "/usr/lib/python3.11/asyncio/events.py", line 84, in _run

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 510, in dispatch_queue

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 499, in process_one

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 406, in dispatch_shell

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 730, in execute_request

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/ipkernel.py", line 383, in do_execute

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/zmqshell.py", line 528, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 2975, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3030, in _run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/async_helpers.py", line 78, in _pseudo_sync_runner

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3257, in run_cell_async

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3473, in run_ast_nodes

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code

  File "/tmp/ipykernel_55/110206276.py", line 89, in <cell line: 0>

  File "/usr/lib/python3.11/multiprocessing/context.py", line 119, in Pool

  File "/usr/lib/python3.11/multiprocessing/pool.py", line 215, in __init__

  File "/usr/lib/python3.11/multiprocessing/pool.py", line 306, in _repopulate_pool

  File "/usr/lib/python3.11/multiprocessing/pool.py", line 329, in _repopulate_pool_static

  File "/usr/lib/python3.11/multiprocessing/process.py", line 121, in start

  File "/usr/lib/python3.11/multiprocessing/context.py", line 281, in _Popen

  File "/usr/lib/python3.11/multiprocessing/popen_fork.py", line 19, in __init__

  File "/usr/lib/python3.11/multiprocessing/popen_fork.py", line 71, in _launch

  File "/usr/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap

  File "/usr/lib/python3.11/multiprocessing/process.py", line 108, in run

  File "/usr/lib/python3.11/multiprocessing/pool.py", line 125, in worker

  File "/usr/lib/python3.11/multiprocessing/pool.py", line 48, in mapstar

  File "/tmp/ipykernel_55/110206276.py", line 67, in train_fold

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 132, in multi_step_on_iterator

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 113, in one_step_on_data

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 77, in train_step

Failed to call DoRnnBackward with model config: [rnn_mode, rnn_input_mode, rnn_direction_mode]: 2, 0, 0 , [num_layers, input_size, num_units, dir_count, max_seq_length, batch_size, cell_num_units]: [1, 300, 100, 1, 80, 4096, 100] 
	 [[{{node gradient_tape/sequential_1/bidirectional_3_1/forward_lstm_3_1/CudnnRNNBackpropV3}}]] [Op:__inference_multi_step_on_iterator_9347]

## === cell 7
submission["pressure"] = np.mean(np.column_stack(test_preds), axis=1)
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4094766021.py in <cell line: 0>()
----> 1 submission["pressure"] = np.mean(np.column_stack(test_preds), axis=1)
      2 submission.to_csv("submission.csv", index=False)

NameError: name 'test_preds' is not defined
