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

0.1543357495179031

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import KFold

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.get_logger().setLevel("ERROR")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

from sklearn.preprocessing import RobustScaler

rc = RobustScaler()




## === cell 2
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 3
def dropCols(df, cols):
    df = df.copy()
    df.drop(cols, axis=1, inplace=True)
    return df




## === cell 4
train_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
train_data = pd.read_csv(train_path, dtype=train_dtypes)

g = train_data.groupby("breath_id", sort=False)
train_data["u_in_lag1"] = g["u_in"].shift(1).fillna(0)
train_data["diff_u_in1"] = train_data["u_in"] - train_data["u_in_lag1"]
train_data["u_in_cumsum"] = g["u_in"].cumsum()




## === cell 5
cols_2_drop = ["id", "breath_id", "time_step"]




## === cell 6
train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")




## === cell 7
train_df.shape




## === cell 8
train_df.isna().sum()




## === cell 9
rc.fit(train_df)
train_df = rc.transform(train_df).astype(np.float32, copy=False)




## === cell 10
train_df = train_df.reshape(-1, 80, train_df.shape[-1])
Y = Y.values.reshape(-1, 80, 1).astype(np.float32, copy=False)




## === cell 11
train_df.shape, Y.shape




## === cell 12
def build_model():
    model = tf.keras.Sequential()

    model.add(
        layers.Bidirectional(
            layers.LSTM(120, return_sequences=True),
            input_shape=[80, train_df.shape[-1]],
        )
    )
    model.add(layers.Bidirectional(layers.LSTM(180, return_sequences=True)))
    model.add(
        layers.Bidirectional(layers.LSTM(280, dropout=0.2, return_sequences=True))
    )
    model.add(
        layers.Bidirectional(layers.LSTM(360, dropout=0.2, return_sequences=True))
    )
    model.add(
        layers.Bidirectional(layers.LSTM(500, dropout=0.2, return_sequences=True))
    )
    model.add(
        layers.Bidirectional(layers.LSTM(1000, dropout=0.2, return_sequences=True))
    )

    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.TimeDistributed(layers.Dense(1)))

    opt = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=["mae"],
        jit_compile=True,
    )
    return model




## === cell 13
build_model().summary()




## === cell 14
callback1 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.9,
    patience=10,
    verbose=1,
)




## === cell 15
EPOCH = 500
BATCH_SIZE = 1024

try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
    print("Running on TPU.")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print(
        f"Running on default strategy (CPU/GPU). TPU not used: {type(e).__name__}: {e}"
    )

AUTOTUNE = tf.data.AUTOTUNE

n_breaths = train_df.shape[0]
base_idx = np.arange(n_breaths, dtype=np.int32)

options = tf.data.Options()
options.experimental_deterministic = True

base_ds = tf.data.Dataset.from_tensor_slices((base_idx, train_df, Y)).with_options(
    options
)

trained_models = []

with strategy.scope():
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(base_idx)):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

        train_idx_tf = tf.constant(train_idx.astype(np.int32))
        valid_idx_tf = tf.constant(test_idx.astype(np.int32))

        train_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=train_idx_tf,
                values=tf.ones_like(train_idx_tf, dtype=tf.bool),
            ),
            default_value=False,
        )
        valid_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=valid_idx_tf,
                values=tf.ones_like(valid_idx_tf, dtype=tf.bool),
            ),
            default_value=False,
        )

        def _is_train(i, x, y):
            return train_table.lookup(i)

        def _is_valid(i, x, y):
            return valid_table.lookup(i)

        def _drop_i(i, x, y):
            return x, y

        train_ds = (
            base_ds.filter(_is_train)
            .map(_drop_i, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(AUTOTUNE)
        )
        valid_ds = (
            base_ds.filter(_is_valid)
            .map(_drop_i, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(AUTOTUNE)
        )

        model = build_model()

        callback0 = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"AdamPressurePreModel{fold+1}.weights.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )

        his = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCH,
            callbacks=[callback0, callback1],
            verbose=2,
        )

        if False:
            try:
                stats = pd.DataFrame(his.history)
                ax = stats.plot()
                ax.set_xlabel("epoch")
                plt.show()
            except Exception:
                pass

        model.load_weights(f"AdamPressurePreModel{fold+1}.weights.h5")
        trained_models.append(model)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/3622245489.py in <cell line: 0>()
     41         valid_idx_tf = tf.constant(test_idx.astype(np.int32))
     42 
---> 43         train_table = tf.lookup.StaticHashTable(
     44             tf.lookup.KeyValueTensorInitializer(
     45                 keys=train_idx_tf,

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

NotFoundError: Could not find device for node: {{node HashTableV2}} = HashTableV2[container="", key_dtype=DT_INT32, shared_name="857", use_node_name_sharing=false, value_dtype=DT_BOOL]
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

## === cell 16
models = trained_models
len(models)




## === cell 17
test_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}
test_data = pd.read_csv(test_path, dtype=test_dtypes)

g = test_data.groupby("breath_id", sort=False)
test_data["u_in_lag1"] = g["u_in"].shift(1).fillna(0)
test_data["diff_u_in1"] = test_data["u_in"] - test_data["u_in_lag1"]
test_data["u_in_cumsum"] = g["u_in"].cumsum()

test_data = dropCols(test_data, cols_2_drop)
test_data = rc.transform(test_data).astype(np.float32, copy=False)
test_data = test_data.reshape(-1, 80, test_data.shape[-1])




## === cell 18
test_data.shape




## === cell 19
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_data)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

mean_pred = None
for model in models:
    pred = model.predict(test_ds, verbose=0)  # (n_breaths, 80, 1)
    flat = pred.reshape(-1, 1)
    if mean_pred is None:
        mean_pred = flat
    else:
        mean_pred += flat
mean_pred = mean_pred / len(models)




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4011994227.py in <cell line: 0>()
     15     else:
     16         mean_pred += flat
---> 17 mean_pred = mean_pred / len(models)
     18 
     19 

TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'

## === cell 20
mean_pred.shape




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/917153286.py in <cell line: 0>()
----> 1 mean_pred.shape
      2 
      3 

AttributeError: 'NoneType' object has no attribute 'shape'

## === cell 21
submission_file = pd.read_csv(sample_sub)

if len(submission_file) != mean_pred.shape[0]:
    raise ValueError(
        f"Prediction length mismatch: submission has {len(submission_file)} rows but preds have {mean_pred.shape[0]} rows"
    )

submission_file["pressure"] = mean_pred.reshape(-1)
submission_file.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_file.shape)
print(submission_file.head())

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/305550105.py in <cell line: 0>()
      1 submission_file = pd.read_csv(sample_sub)
      2 
----> 3 if len(submission_file) != mean_pred.shape[0]:
      4     raise ValueError(
      5         f"Prediction length mismatch: submission has {len(submission_file)} rows but preds have {mean_pred.shape[0]} rows"

AttributeError: 'NoneType' object has no attribute 'shape'
