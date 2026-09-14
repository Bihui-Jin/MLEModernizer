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

0.1797

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

print("TensorFlow:", tf.__version__)

SEED = 2021
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
except Exception:
    pass

TFDATA_OPTIONS = tf.data.Options()
TFDATA_OPTIONS.experimental_deterministic = True
try:
    TFDATA_OPTIONS.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass

sc = StandardScaler()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"

train = pd.read_csv(
    train_path,
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
    test_path,
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
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    g = df.groupby("breath_id", sort=False)

    df["area"] = (
        (df["time_step"] * df["u_in"]).groupby(df["breath_id"], sort=False).cumsum()
    )

    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)
    u_out = df["u_out"].to_numpy(dtype=np.float32, copy=False)
    time_step = df["time_step"].to_numpy(dtype=np.float32, copy=False)

    df["cross"] = u_in * u_out
    df["cross2"] = time_step * u_out

    df["u_in_lag_2"] = g["u_in"].shift(2)
    df["u_in_diff_2"] = df["u_in"] - df["u_in_lag_2"]

    df["u_in_max"] = g["u_in"].transform("max")
    df["u_in_cumsum"] = g["u_in"].cumsum()

    cnt = g.cumcount().to_numpy(dtype=np.int32, copy=False) + 1  # 1..80
    cs = df["u_in_cumsum"].to_numpy(dtype=np.float32, copy=False)
    denom = (cnt - 1).astype(np.float32)
    expand_mean = (cs - u_in) / denom
    expand_mean[denom == 0.0] = np.nan
    df["expand_mean"] = expand_mean

    R = df["R"].to_numpy(copy=False)
    C = df["C"].to_numpy(copy=False)
    df["R_5"] = (R == 5).astype(np.int8)
    df["R_20"] = (R == 20).astype(np.int8)
    df["R_50"] = (R == 50).astype(np.int8)
    df["C_10"] = (C == 10).astype(np.int8)
    df["C_20"] = (C == 20).astype(np.int8)
    df["C_50"] = (C == 50).astype(np.int8)

    return df


train = add_features(train)
test = add_features(test)

train = train.fillna(0)
test = test.fillna(0)

train_cols = train.columns
test = test.reindex(columns=train_cols.drop("pressure"), fill_value=0)



## === cell 3
targets = train["pressure"].to_numpy().reshape(-1, 80, 1)

test_id = test["id"].copy()

train.drop(columns=["id", "breath_id", "pressure"], inplace=True)
test.drop(columns=["id", "breath_id"], inplace=True)



## === cell 4
train_np = train.to_numpy(dtype=np.float32, copy=False)
test_np = test.to_numpy(dtype=np.float32, copy=False)

sc.fit(train_np)

train_new = sc.transform(train_np).astype(np.float32, copy=False)
test_new = sc.transform(test_np).astype(np.float32, copy=False)

train_re = train_new.reshape(-1, 80, train_new.shape[-1])
test_re = test_new.reshape(-1, 80, test_new.shape[-1])

print("train_re:", train_re.shape, "targets:", targets.shape, "test_re:", test_re.shape)



## === cell 5
opt = tf.keras.optimizers.Adam(learning_rate=0.001)


def build_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Bidirectional(
            layers.LSTM(512, return_sequences=True),
            input_shape=[80, train_new.shape[-1]],
        )
    )
    model.add(
        layers.Bidirectional(layers.LSTM(256, dropout=0.2, return_sequences=True))
    )
    model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
    model.add(
        layers.Bidirectional(layers.LSTM(128, dropout=0.2, return_sequences=True))
    )
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))

    model.add(layers.Dense(32, activation="selu"))
    model.add(layers.TimeDistributed(layers.Dense(1)))

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=["mae"],
        jit_compile=True,
    )
    return model




## === cell 6
EPOCH = 400
BATCH_SIZE = 512

try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # auto-detect
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
    print("Running with TPU strategy.")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print("TPU not available; using default strategy:", type(strategy).__name__)
    print("TPU init error (informational):", repr(e))



## === cell 7
reduce_lr = ReduceLROnPlateau(monitor="val_loss", verbose=1, factor=0.85, patience=10)

BASE_SHUFFLE_BUFFER = 8192  # deterministic, much cheaper than len(indices) ~ 68k


def make_indexed_base_dataset(X, y=None):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(X)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.with_options(TFDATA_OPTIONS)
    return ds.cache()  # cached ONCE globally


base_train_ds = make_indexed_base_dataset(train_re, targets)
base_test_ds = make_indexed_base_dataset(test_re, None)


def make_dataset_from_base(base_ds, indices, batch_size, training, seed):
    indices = np.asarray(indices, dtype=np.int64)

    keys = tf.convert_to_tensor(indices, dtype=tf.int64)
    vals = tf.ones_like(keys, dtype=tf.bool)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals), default_value=False
    )

    enum_base = base_ds.enumerate()

    if isinstance(base_ds.element_spec, tuple):
        sel = enum_base.filter(lambda i, xy: table.lookup(tf.cast(i, tf.int64)))
        sel = sel.map(lambda i, xy: xy, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        sel = enum_base.filter(lambda i, x: table.lookup(tf.cast(i, tf.int64)))
        sel = sel.map(lambda i, x: x, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        sel = sel.shuffle(
            buffer_size=min(BASE_SHUFFLE_BUFFER, len(indices)),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    sel = sel.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return sel


test_ds = base_test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(
    tf.data.AUTOTUNE
)

fold_weights = []

with strategy.scope():
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_re, targets)):
        print("-" * 30, ">", f"Fold {fold+1}", "<", "-" * 30)

        tf.keras.backend.clear_session()

        ckpt_path = f"Model{fold+1}.keras"

        if os.path.exists(ckpt_path):
            print(
                f"Found existing checkpoint {ckpt_path}; skipping training for this fold."
            )
            best_model = tf.keras.models.load_model(ckpt_path)
            fold_weights.append(best_model.get_weights())
            print("\n\n")
            continue

        model = build_model()

        call_back = tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, verbose=1, monitor="val_loss", save_best_only=True
        )

        train_ds = make_dataset_from_base(
            base_train_ds, train_idx, BATCH_SIZE, training=True, seed=SEED
        )
        valid_ds = make_dataset_from_base(
            base_train_ds, valid_idx, BATCH_SIZE, training=False, seed=SEED
        )

        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCH,
            callbacks=[call_back, reduce_lr],
            verbose=2,
        )

        if os.path.exists(ckpt_path):
            best_model = tf.keras.models.load_model(ckpt_path)
            fold_weights.append(best_model.get_weights())
        else:
            fold_weights.append(model.get_weights())

        print("\n\n")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/3468253081.py in <cell line: 0>()
     79         )
     80 
---> 81         train_ds = make_dataset_from_base(
     82             base_train_ds, train_idx, BATCH_SIZE, training=True, seed=SEED
     83         )

/tmp/ipykernel_11/3468253081.py in make_dataset_from_base(base_ds, indices, batch_size, training, seed)
     23     keys = tf.convert_to_tensor(indices, dtype=tf.int64)
     24     vals = tf.ones_like(keys, dtype=tf.bool)
---> 25     table = tf.lookup.StaticHashTable(
     26         tf.lookup.KeyValueTensorInitializer(keys, vals), default_value=False
     27     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/trackable/resource.py in __call__(cls, *args, **kwargs)
    101       previous_getter = _make_getter(getter, previous_getter)
    102 
--> 103     return previous_getter(*args, **kwargs)
    104 
    105 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/trackable/resource.py in <lambda>(*a, **kw)
     96       return obj
     97 
---> 98     previous_getter = lambda *a, **kw: default_resource_creator(None, *a, **kw)
     99     resource_creator_stack = ops.get_default_graph()._resource_creator_stack
    100     for getter in resource_creator_stack[cls._resource_type()]:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/trackable/resource.py in default_resource_creator(next_creator, *a, **kw)
     93       assert next_creator is None
     94       obj = cls.__new__(cls, *a, **kw)
---> 95       obj.__init__(*a, **kw)
     96       return obj
     97 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/lookup_ops.py in __init__(self, initializer, default_value, name, experimental_is_anonymous)
    351     self._name = name or "hash_table"
    352     self._table_name = None
--> 353     super(StaticHashTable, self).__init__(default_value, initializer)
    354     self._value_shape = self._default_value.get_shape()
    355 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/lookup_ops.py in __init__(self, default_value, initializer)
    199       self._initializer = self._track_trackable(initializer, "_initializer")
    200     with ops.init_scope():
--> 201       self._resource_handle = self._create_resource()
    202     if (not context.executing_eagerly() and
    203         ops.get_default_graph()._get_control_flow_context() is not None):  # pylint: disable=protected-access

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/lookup_ops.py in _create_resource(self)
    361           name=self._name)
    362     else:
--> 363       table_ref = gen_lookup_ops.hash_table_v2(
    364           shared_name=self._shared_name,
    365           key_dtype=self._initializer.key_dtype,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_lookup_ops.py in hash_table_v2(key_dtype, value_dtype, container, shared_name, use_node_name_sharing, name)
    483       return _result
    484     except _core._NotOkStatusException as e:
--> 485       _ops.raise_from_not_ok_status(e, name)
    486     except _core._FallbackException:
    487       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

NotFoundError: Could not find device for node: {{node HashTableV2}} = HashTableV2[container="", key_dtype=DT_INT64, shared_name="736", use_node_name_sharing=false, value_dtype=DT_BOOL]
All kernels registered for op HashTableV2:
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_STRING]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_INT64]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_INT32]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_FLOAT]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_DOUBLE]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_BOOL]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_STRING]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_INT64]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_INT32]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_FLOAT]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_DOUBLE]
  device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_STRING]
  device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_INT32]
  device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_FLOAT]
  device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_DOUBLE]
 [Op:HashTableV2] name: hash_table

## === cell 8
test_preds = []

with strategy.scope():
    for i, w in enumerate(fold_weights, start=1):
        m = build_model()
        m.set_weights(w)

        pred = m.predict(test_ds, verbose=0)
        pred = np.asarray(pred, dtype=np.float32).reshape(-1, 1)
        test_preds.append(pred)

test_pred = np.mean(np.stack(test_preds, axis=0), axis=0).astype(np.float32, copy=False)
print("test_pred shape:", test_pred.shape)

if test_pred.shape[0] != len(test_id):
    raise ValueError(f"Pred rows ({test_pred.shape[0]}) != test rows ({len(test_id)})")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2783289709.py in <cell line: 0>()
     12 
     13 # Fix: robust mean even if list is length 1; guarantees test_pred is (N, 1).
---> 14 test_pred = np.mean(np.stack(test_preds, axis=0), axis=0).astype(np.float32, copy=False)
     15 print("test_pred shape:", test_pred.shape)
     16 

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    443     arrays = [asanyarray(arr) for arr in arrays]
    444     if not arrays:
--> 445         raise ValueError('need at least one array to stack')
    446 
    447     shapes = {arr.shape for arr in arrays}

ValueError: need at least one array to stack

## === cell 9
submission_file = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

if len(submission_file) != test_pred.shape[0]:
    raise ValueError(
        f"Submission rows ({len(submission_file)}) != predictions ({test_pred.shape[0]})"
    )

submission_file["pressure"] = test_pred.reshape(-1)

submission_path = "submission.csv"
submission_file.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_file.head())
print(submission_file.shape)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3572163846.py in <cell line: 0>()
      3 )
      4 
----> 5 if len(submission_file) != test_pred.shape[0]:
      6     raise ValueError(
      7         f"Submission rows ({len(submission_file)}) != predictions ({test_pred.shape[0]})"

NameError: name 'test_pred' is not defined
