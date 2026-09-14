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

0.2033095636253387

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, gc, time, logging

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

dtypes_train = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
dtypes_test = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

train = pd.read_csv(TRAIN_PATH, dtype=dtypes_train)
test = pd.read_csv(TEST_PATH, dtype=dtypes_test)
submission = pd.read_csv(SAMPLE_SUB_PATH, dtype={"id": "int32", "pressure": "float32"})

train.head(), test.head(), submission.head()



## === cell 2
assert "pressure" in train.columns
assert "pressure" not in test.columns
assert set(submission.columns) == {"id", "pressure"}

test_ids = test["id"].values



## === cell 3
train = train.drop(columns=["id"])
test = test.drop(columns=["id"])




## === cell 4
def add_features_inplace(df: pd.DataFrame) -> None:
    seq_len = 80
    n = len(df)
    assert n % seq_len == 0, "Row count must be divisible by 80"
    n_breaths = n // seq_len

    Rf = df["R"].to_numpy(dtype=np.float32, copy=False)
    Cf = df["C"].to_numpy(dtype=np.float32, copy=False)
    df["RC_sum"] = (Rf + Cf).astype(np.float32, copy=False)
    df["RC_div"] = (Rf / Cf).astype(np.float32, copy=False)

    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False).reshape(n_breaths, seq_len)
    u_out = (
        df["u_out"].to_numpy(dtype=np.float32, copy=False).reshape(n_breaths, seq_len)
    )
    t = (
        df["time_step"]
        .to_numpy(dtype=np.float32, copy=False)
        .reshape(n_breaths, seq_len)
    )

    df["u_in_cumsum"] = np.cumsum(u_in, axis=1, dtype=np.float32).reshape(-1)

    time_lag1 = np.zeros_like(t, dtype=np.float32)
    time_lag1[:, 1:] = t[:, :-1]
    df["time_lag1"] = time_lag1.reshape(-1)

    u_in_lag1 = np.zeros_like(u_in, dtype=np.float32)
    u_in_lag1[:, 1:] = u_in[:, :-1]
    df["u_in_lag1"] = u_in_lag1.reshape(-1)

    u_out_lag1 = np.zeros_like(u_out, dtype=np.float32)
    u_out_lag1[:, 1:] = u_out[:, :-1]
    df["u_out_lag1"] = u_out_lag1.reshape(-1)

    time_lag2 = np.zeros_like(t, dtype=np.float32)
    time_lag2[:, 2:] = t[:, :-2]
    df["time_lag2"] = time_lag2.reshape(-1)


add_features_inplace(train)
add_features_inplace(test)

y = train["pressure"].to_numpy(dtype=np.float32, copy=False)
train.drop(columns=["pressure"], inplace=True)

R_categories = [5, 20, 50]
C_categories = [10, 20, 50]
R_cat = pd.api.types.CategoricalDtype(categories=R_categories, ordered=False)
C_cat = pd.api.types.CategoricalDtype(categories=C_categories, ordered=False)
for _df in (train, test):
    _df["R"] = _df["R"].astype(R_cat)
    _df["C"] = _df["C"].astype(C_cat)

train = pd.get_dummies(train, columns=["R", "C"])
test = pd.get_dummies(test, columns=["R", "C"])

test = test.reindex(columns=train.columns, fill_value=0)

train.drop(columns=["breath_id"], inplace=True)
test.drop(columns=["breath_id"], inplace=True)



## === cell 5
rb = RobustScaler()
rb.fit(train.values)

train2 = rb.transform(train.values).astype(np.float32, copy=False)
test2 = rb.transform(test.values).astype(np.float32, copy=False)

del train, test
gc.collect()



## === cell 6
SEQ_LEN = 80
n_features = train2.shape[1]

assert train2.shape[0] % SEQ_LEN == 0, "Train rows not divisible by 80"
assert test2.shape[0] % SEQ_LEN == 0, "Test rows not divisible by 80"
assert y.shape[0] == train2.shape[0], "Target length mismatch"

n_breaths_train = train2.shape[0] // SEQ_LEN
n_breaths_test = test2.shape[0] // SEQ_LEN

train3 = train2.reshape(n_breaths_train, SEQ_LEN, n_features)
test3 = test2.reshape(n_breaths_test, SEQ_LEN, n_features)
y = y.reshape(n_breaths_train, SEQ_LEN)

del train2, test2, rb
gc.collect()

(train3.shape, y.shape, test3.shape, n_features)



## === cell 7
tf.get_logger().setLevel(logging.ERROR)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # will fail on non-TPU
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except Exception:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled (GPU)")
        except Exception:
            print("Mixed precision not enabled (API/version constraint)")
    strategy = (
        tf.distribute.MirroredStrategy()
        if len(gpus) > 1
        else tf.distribute.get_strategy()
    )
    print("Running on", "GPU" if len(gpus) > 0 else "CPU")

print("TensorFlow:", tf.__version__)
print("REPLICAS:", strategy.num_replicas_in_sync)




## === cell 8
def plot_hist(hist):
    plt.figure(figsize=(8, 4))
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.title("model performance")
    plt.ylabel("MAE")
    plt.xlabel("epoch")
    plt.legend(loc="upper left")
    plt.show()




## === cell 9
def create_model():
    with strategy.scope():
        model = Sequential(
            [
                Input(shape=(SEQ_LEN, n_features)),
                Bidirectional(LSTM(300, return_sequences=True)),
                Bidirectional(LSTM(256, return_sequences=True)),
                Bidirectional(LSTM(200, return_sequences=True)),
                Bidirectional(LSTM(128, return_sequences=True)),
                Dense(128, activation="selu"),
                Dropout(0.2),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mae")
    return model




## === cell 10
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

test_preds = []

AUTO = tf.data.AUTOTUNE

options = tf.data.Options()
options.deterministic = True

BATCH_SIZE = 512

base_train_ds = (
    tf.data.Dataset.from_tensor_slices((train3, y)).with_options(options).cache()
)
base_test_ds = tf.data.Dataset.from_tensor_slices(test3).with_options(options)

test_ds = base_test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

for fold, (train_idx, valid_idx) in enumerate(kf.split(train3, y)):
    print(f"****** fold: {fold+1} *******")

    train_idx_tf = tf.convert_to_tensor(train_idx, dtype=tf.int32)
    valid_idx_tf = tf.convert_to_tensor(valid_idx, dtype=tf.int32)

    train_ds = (
        base_train_ds.enumerate()
        .filter(lambda i, _: tf.reduce_any(tf.equal(i, train_idx_tf)))
        .map(lambda _, xy: xy, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTO)
    )
    valid_ds = (
        base_train_ds.enumerate()
        .filter(lambda i, _: tf.reduce_any(tf.equal(i, valid_idx_tf)))
        .map(lambda _, xy: xy, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTO)
    )

    steps_per_epoch = int(np.ceil(len(train_idx) / BATCH_SIZE))
    decay_steps = int(200 * steps_per_epoch)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=max(decay_steps, 1),
        decay_rate=1e-5,
        staircase=False,
    )

    with strategy.scope():
        model = create_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss="mae"
        )

    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=25,
        verbose=1,
        restore_best_weights=True,
    )

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=240,
        callbacks=[es],
        verbose=1,
    )

    preds = model.predict(test_ds, verbose=0).squeeze()  # (n_breaths, 80)
    test_preds.append(preds)

    del model, history, preds, train_ds, valid_ds, train_idx_tf, valid_idx_tf
    gc.collect()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4132205504.py in <cell line: 0>()
     30     train_ds = (
     31         base_train_ds.enumerate()
---> 32         .filter(lambda i, _: tf.reduce_any(tf.equal(i, train_idx_tf)))
     33         .map(lambda _, xy: xy, num_parallel_calls=AUTO)
     34         .batch(BATCH_SIZE, drop_remainder=False)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in filter(self, predicate, name)
   2561     # pylint: disable=g-import-not-at-top,protected-access
   2562     from tensorflow.python.data.ops import filter_op
-> 2563     return filter_op._filter(self, predicate, name)
   2564     # pylint: enable=g-import-not-at-top,protected-access
   2565 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/filter_op.py in _filter(input_dataset, predicate, name)
     23 
     24 def _filter(input_dataset, predicate, name=None):  # pylint: disable=redefined-builtin
---> 25   return _FilterDataset(input_dataset, predicate, name=name)
     26 
     27 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/filter_op.py in __init__(self, input_dataset, predicate, use_legacy_function, name)
     36     """See `Dataset.filter` for details."""
     37     self._input_dataset = input_dataset
---> 38     wrapped_func = structured_function.StructuredFunctionWrapper(
     39         predicate,
     40         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_filepkebuk4e.py in <lambda>(i, _)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda i, _: ag__.with_function_scope(lambda lscope: ag__.converted_call(tf.reduce_any, (ag__.converted_call(tf.equal, (i, train_idx_tf), None, lscope),), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filepkebuk4e.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda i, _: ag__.with_function_scope(lambda lscope: ag__.converted_call(tf.reduce_any, (ag__.converted_call(tf.equal, (i, train_idx_tf), None, lscope),), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    375 
    376   if not options.user_requested and conversion.is_allowlisted(f):
--> 377     return _call_unconverted(f, args, kwargs, options)
    378 
    379   # internal_convert_user_code is for example turned off when issuing a dynamic

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    458   if kwargs is not None:
    459     return f(*args, **kwargs)
--> 460   return f(*args)
    461 
    462 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map, keywords, default_type_attr_map, attrs, inputs, input_types)
    650       if input_arg.type_attr in attrs:
    651         if attrs[input_arg.type_attr] != attr_value:
--> 652           raise TypeError(
    653               f"Input '{input_name}' of '{op_type_name}' Op has type "
    654               f"{dtypes.as_dtype(attr_value).name} that does not match type "

TypeError: in user code:

    File "/tmp/ipykernel_11/4132205504.py", line 32, in None  *
        lambda i, _: tf.reduce_any(tf.equal(i, train_idx_tf))

    TypeError: Input 'y' of 'Equal' Op has type int32 that does not match type int64 of argument 'x'.


## === cell 11
pred_mean = np.mean(np.stack(test_preds, axis=0), axis=0)  # (n_breaths_test, 80)
pred_flat = pred_mean.reshape(-1)

assert len(pred_flat) == len(
    submission
), "Prediction length does not match submission length"

submission["id"] = test_ids
submission["pressure"] = pred_flat.astype(np.float32)

submission.to_csv("submission.csv", index=False)
submission.head()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1773731472.py in <cell line: 0>()
----> 1 pred_mean = np.mean(np.stack(test_preds, axis=0), axis=0)  # (n_breaths_test, 80)
      2 pred_flat = pred_mean.reshape(-1)
      3 
      4 assert len(pred_flat) == len(
      5     submission

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    443     arrays = [asanyarray(arr) for arr in arrays]
    444     if not arrays:
--> 445         raise ValueError('need at least one array to stack')
    446 
    447     shapes = {arr.shape for arr in arrays}

ValueError: need at least one array to stack

## === cell 12
print("Wrote:", os.path.abspath("submission.csv"))
subm_check = pd.read_csv("submission.csv")
print(subm_check.head())
print(subm_check.shape)
assert list(subm_check.columns) == ["id", "pressure"]
assert len(subm_check) == 603600

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/820095672.py in <cell line: 0>()
      1 print("Wrote:", os.path.abspath("submission.csv"))
----> 2 subm_check = pd.read_csv("submission.csv")
      3 print(subm_check.head())
      4 print(subm_check.shape)
      5 assert list(subm_check.columns) == ["id", "pressure"]

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'
