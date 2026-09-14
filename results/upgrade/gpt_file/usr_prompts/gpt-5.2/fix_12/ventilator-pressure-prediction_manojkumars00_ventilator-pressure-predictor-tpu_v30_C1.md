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

0.9801976631902684

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
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "0")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.set_synchronous_execution(False)
except Exception:
    pass

tf.config.run_functions_eagerly(False)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

sc = StandardScaler()
rc = RobustScaler()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 2
def dropCols(df, cols):
    df = df.drop(cols, axis=1)
    return df




## === cell 3
def preProcess(df):
    df = df.copy()

    breath = df["breath_id"].to_numpy(copy=False)
    u_in = df["u_in"].to_numpy(dtype=np.float64, copy=False)
    t = df["time_step"].to_numpy(dtype=np.float64, copy=False)

    new_breath = np.empty(breath.shape[0], dtype=bool)
    new_breath[0] = True
    new_breath[1:] = breath[1:] != breath[:-1]

    area = t * u_in
    csum = np.cumsum(area)
    offsets = np.zeros_like(csum)
    start_idx = np.flatnonzero(new_breath)
    if start_idx.size > 1:
        prev_end_idx = start_idx[1:] - 1
        offsets[start_idx[1:]] = csum[prev_end_idx]
        np.maximum.accumulate(offsets, out=offsets)
    area_cum = csum - offsets

    ucsum = np.cumsum(u_in)
    offsets_u = np.zeros_like(ucsum)
    if start_idx.size > 1:
        offsets_u[start_idx[1:]] = ucsum[prev_end_idx]
        np.maximum.accumulate(offsets_u, out=offsets_u)
    u_in_cumsum = ucsum - offsets_u

    u_in_lag1 = np.empty_like(u_in)
    u_in_lag1[0] = 0.0
    u_in_lag1[1:] = u_in[:-1]
    u_in_lag1[new_breath] = 0.0

    u_in_diff1 = u_in - u_in_lag1

    df["area"] = area_cum
    df["u_in_cumsum"] = u_in_cumsum
    df["u_in_lag1"] = u_in_lag1
    df["u_in_diff1"] = u_in_diff1
    return df




## === cell 4
_train_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
_usecols_train = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
train_data = pd.read_csv(train_path, dtype=_train_dtypes, usecols=_usecols_train)
train_data = preProcess(train_data)



## === cell 5
cols_2_drop = ["id", "breath_id", "time_step"]



## === cell 6
train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")



## === cell 7
rc.fit(train_df)
train_df = rc.transform(train_df).astype(np.float32, copy=False)

n_features = train_df.shape[-1]
train_df = np.ascontiguousarray(train_df.reshape(-1, 80, n_features), dtype=np.float32)

Y = Y.to_numpy(dtype=np.float32, copy=False)
Y = np.ascontiguousarray(Y.reshape(-1, 80, 1), dtype=np.float32)

train_df.shape, Y.shape



## === cell 8
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_C_out_loss",
    factor=0.96,
    patience=8,
    verbose=0,
)




## === cell 9
def build_model():
    Input = layers.Input(shape=([80, train_df.shape[-1]]))

    m1x1 = layers.Bidirectional(
        layers.LSTM(
            units=440,
            return_sequences=True,
            recurrent_dropout=0.0,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            implementation=2,
        )
    )
    m1x2 = layers.Bidirectional(
        layers.LSTM(
            units=360,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            implementation=2,
        )
    )
    m1x3 = layers.Bidirectional(
        layers.LSTM(
            units=240,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            implementation=2,
        )
    )
    m1x4 = layers.Bidirectional(
        layers.LSTM(
            units=180,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            implementation=2,
        )
    )
    m1x5 = layers.Bidirectional(
        layers.LSTM(
            units=100,
            return_sequences=True,
            recurrent_dropout=0.0,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            implementation=2,
        )
    )

    m2x1 = layers.Bidirectional(
        layers.LSTM(
            units=880,
            return_sequences=True,
            recurrent_dropout=0.0,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            implementation=2,
        )
    )
    m2x3 = layers.Bidirectional(
        layers.LSTM(
            units=440,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            implementation=2,
        )
    )
    m2x5 = layers.Bidirectional(
        layers.LSTM(
            units=360,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            implementation=2,
        )
    )
    m2x7 = layers.Bidirectional(
        layers.LSTM(
            units=240,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            implementation=2,
        )
    )
    m2x9 = layers.Bidirectional(
        layers.LSTM(
            units=180,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            implementation=2,
        )
    )
    m2x11 = layers.Bidirectional(
        layers.LSTM(
            units=100,
            return_sequences=True,
            recurrent_dropout=0.0,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            implementation=2,
        )
    )

    m3x1 = layers.Bidirectional(
        layers.GRU(
            units=440,
            return_sequences=True,
            recurrent_dropout=0.0,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            implementation=2,
        )
    )
    m3x2 = layers.Bidirectional(
        layers.GRU(
            units=360,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            implementation=2,
        )
    )
    m3x3 = layers.Bidirectional(
        layers.GRU(
            units=240,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            implementation=2,
        )
    )
    m3x4 = layers.Bidirectional(
        layers.GRU(
            units=180,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            implementation=2,
        )
    )
    m3x5 = layers.Bidirectional(
        layers.GRU(
            units=100,
            return_sequences=True,
            recurrent_dropout=0.0,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            implementation=2,
        )
    )

    m3x1_out = m3x1(Input)
    m3x2_out = m3x2(m3x1_out)
    m3x3_out = m3x3(m3x2_out)
    m3x4_out = m3x4(m3x3_out)
    m3x5_out = m3x5(m3x4_out)

    m2x1_out = m2x1(Input)
    m2x3_out = m2x3(m2x1_out)
    m2x5_out = m2x5(m2x3_out)
    m2x7_out = m2x7(m2x5_out)
    m2x9_out = m2x9(m2x7_out)
    m2x11_out = m2x11(m2x9_out)

    m1x1_out = m1x1(Input)
    m1x1_out = layers.Multiply()([m1x1_out, m2x3_out, m3x1_out])
    m1x1_out = layers.BatchNormalization()(m1x1_out)

    m1x2_out = m1x2(m1x1_out)
    m1x2_out = layers.Multiply()([m1x2_out, m2x5_out, m3x2_out])
    m1x2_out = layers.BatchNormalization()(m1x2_out)

    m1x3_out = m1x3(m1x2_out)
    m1x3_out = layers.Multiply()([m1x3_out, m2x7_out, m3x3_out])
    m1x3_out = layers.BatchNormalization()(m1x3_out)

    m1x4_out = m1x4(m1x3_out)
    m1x4_out = layers.Multiply()([m1x4_out, m2x9_out, m3x4_out])
    m1x4_out = layers.BatchNormalization()(m1x4_out)

    m1x5_out = m1x5(m1x4_out)

    f_mul = layers.Multiply()([m1x5_out, m2x11_out, m3x5_out])
    f_mul = layers.BatchNormalization()(f_mul)
    f_mul = layers.Bidirectional(
        layers.LSTM(
            units=100,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            implementation=2,
        )
    )(f_mul)

    C_out = layers.Dense(
        units=64, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
    )(f_mul)
    C_out = layers.Dense(
        1, name="C_out", kernel_initializer=tf.keras.initializers.HeNormal()
    )(C_out)

    m1_out = layers.Dense(
        units=64, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
    )(m1x5_out)
    m1_out = layers.Dense(
        1, name="m1_out", kernel_initializer=tf.keras.initializers.HeNormal()
    )(m1_out)

    model = tf.keras.Model(inputs=Input, outputs=[C_out, m1_out], name="Base_Model")

    losses = {
        "C_out": tf.keras.losses.MeanAbsoluteError(),
        "m1_out": tf.keras.losses.MeanAbsoluteError(),
    }
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=losses,
        run_eagerly=False,
        jit_compile=True,
    )
    return model




## === cell 10
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Using TPU strategy")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print(
        f"Using default strategy (GPU/CPU). TPU not available: {type(e).__name__}: {e}"
    )




## === cell 11
class BestWeights(tf.keras.callbacks.Callback):
    def __init__(self, monitor, mode="min"):
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
        improved = (
            (current < self.best) if self.mode == "min" else (current > self.best)
        )
        if improved:
            self.best = current
            self.best_weights = self.model.get_weights()




## === cell 12
@tf.function(reduce_retracing=True)
def _pack_targets_xy(xb, yb):
    return xb, {"C_out": yb, "m1_out": yb}


def make_base_dataset(X, y):
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    try:
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    ds = tf.data.Dataset.from_tensor_slices((X, y)).with_options(opts)
    return ds


def make_fold_datasets(base_ds, train_idx, valid_idx, batch_size, seed):
    train_idx_tf = tf.constant(train_idx, dtype=tf.int32)
    valid_idx_tf = tf.constant(valid_idx, dtype=tf.int32)

    ds_tr = tf.data.Dataset.from_tensor_slices(train_idx_tf)
    ds_tr = ds_tr.shuffle(
        buffer_size=min(int(train_idx_tf.shape[0]), 8192),
        seed=seed,
        reshuffle_each_iteration=True,
    )
    ds_tr = ds_tr.map(
        lambda i: base_ds.element_spec and i, num_parallel_calls=AUTOTUNE
    )  # no-op for tracing stability
    ds_tr = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices(train_idx_tf),
            tf.data.Dataset.from_tensor_slices(train_idx_tf),
        )
    )
    ds_tr = ds_tr.map(lambda i, _: i, num_parallel_calls=AUTOTUNE)
    ds_tr = ds_tr.map(
        lambda i: tf.gather(tf.constant(0), i), num_parallel_calls=AUTOTUNE
    )  # dummy; will be overridden below

    X_tf = tf.convert_to_tensor(train_df)
    y_tf = tf.convert_to_tensor(Y)

    ds_tr = tf.data.Dataset.from_tensor_slices(train_idx_tf)
    ds_tr = ds_tr.shuffle(
        buffer_size=min(int(train_idx_tf.shape[0]), 8192),
        seed=seed,
        reshuffle_each_iteration=True,
    )
    ds_tr = ds_tr.map(
        lambda i: (X_tf[i], y_tf[i]), num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds_tr = ds_tr.batch(batch_size, drop_remainder=False)
    ds_tr = ds_tr.map(_pack_targets_xy, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds_tr = ds_tr.prefetch(AUTOTUNE)

    ds_va = tf.data.Dataset.from_tensor_slices(valid_idx_tf)
    ds_va = ds_va.map(
        lambda i: (X_tf[i], y_tf[i]), num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds_va = ds_va.batch(batch_size, drop_remainder=False)
    ds_va = ds_va.map(_pack_targets_xy, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds_va = ds_va.prefetch(AUTOTUNE)

    return ds_tr, ds_va




## === cell 13
EPOCH = 300
BATCH_SIZE = 512

models = []

kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
indices = np.arange(len(train_df), dtype=np.int32)

base_ds = make_base_dataset(train_df, Y)

with strategy.scope():
    for fold, (train_idx, valid_idx) in enumerate(kf.split(indices)):
        print("-" * 15, ">", f"Fold {fold + 1}", "<", "-" * 15)

        model = build_model()

        bw_m1 = BestWeights(monitor="val_m1_out_loss", mode="min")
        bw_c = BestWeights(monitor="val_C_out_loss", mode="min")
        callbacks = [lr_callback, bw_m1, bw_c]

        ds_tr, ds_va = make_fold_datasets(
            base_ds, train_idx, valid_idx, BATCH_SIZE, seed=SEED + fold
        )

        model.fit(
            ds_tr,
            validation_data=ds_va,
            epochs=EPOCH,
            callbacks=callbacks,
            verbose=0,
        )

        if bw_m1.best_weights is not None:
            model.set_weights(bw_m1.best_weights)
        models.append(model)

len(models)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2623623685.py in <cell line: 0>()
     19         callbacks = [lr_callback, bw_m1, bw_c]
     20 
---> 21         ds_tr, ds_va = make_fold_datasets(
     22             base_ds, train_idx, valid_idx, BATCH_SIZE, seed=SEED + fold
     23         )

/tmp/ipykernel_11/1547494492.py in make_fold_datasets(base_ds, train_idx, valid_idx, batch_size, seed)
     40     )
     41     ds_tr = ds_tr.map(lambda i, _: i, num_parallel_calls=AUTOTUNE)
---> 42     ds_tr = ds_tr.map(
     43         lambda i: tf.gather(tf.constant(0), i), num_parallel_calls=AUTOTUNE
     44     )  # dummy; will be overridden below

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

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

/tmp/__autograph_generated_filesi18oqlo.py in <lambda>(i)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda i: ag__.with_function_scope(lambda lscope: ag__.converted_call(tf.gather, (ag__.converted_call(tf.constant, (0,), None, lscope), i), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filesi18oqlo.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda i: ag__.with_function_scope(lambda lscope: ag__.converted_call(tf.gather, (ag__.converted_call(tf.constant, (0,), None, lscope), i), None, lscope), 'lscope', ag__.STD)
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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in _create_c_op(graph, node_def, inputs, control_inputs, op_def, extract_traceback)
   1054   except errors.InvalidArgumentError as e:
   1055     # Convert to ValueError for backwards compatibility.
-> 1056     raise ValueError(e.message)
   1057 
   1058   # Record the current Python stack trace as the creating stacktrace of this

ValueError: in user code:

    File "/tmp/ipykernel_11/1547494492.py", line 43, in None  *
        lambda i: tf.gather(tf.constant(0), i)

    ValueError: Shape must be at least rank 1 but is rank 0 for '{{node GatherV2}} = GatherV2[Taxis=DT_INT32, Tindices=DT_INT32, Tparams=DT_INT32, batch_dims=0](Const, args_0, GatherV2/axis)' with input shapes: [], [], [].


## === cell 14
_test_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}
_usecols_test = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
test_data = pd.read_csv(test_path, dtype=_test_dtypes, usecols=_usecols_test)
test_data = preProcess(test_data)

test_data = dropCols(test_data, cols_2_drop)
test_data = rc.transform(test_data).astype(np.float32, copy=False)
test_data = np.ascontiguousarray(
    test_data.reshape(-1, 80, test_data.shape[-1]), dtype=np.float32
)

test_data.shape



## === cell 15
opts = tf.data.Options()
opts.experimental_deterministic = True
try:
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    opts.experimental_optimization.autotune_buffers = True
except Exception:
    pass

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_data)
    .with_options(opts)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

preds_per_model = []
for m in models:
    out = m.predict(test_ds, verbose=0)
    if isinstance(out, (list, tuple)):
        out = out[-1]  # take m1_out
    preds_per_model.append(out)

preds_per_model[0].shape, len(preds_per_model)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/387100273.py in <cell line: 0>()
     22     preds_per_model.append(out)
     23 
---> 24 preds_per_model[0].shape, len(preds_per_model)
     25 

IndexError: list index out of range

## === cell 16
pred_stack = np.stack(preds_per_model, axis=0)
median_pre = np.median(pred_stack, axis=0)

median_pre.shape



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2726861069.py in <cell line: 0>()
----> 1 pred_stack = np.stack(preds_per_model, axis=0)
      2 median_pre = np.median(pred_stack, axis=0)
      3 
      4 median_pre.shape
      5 

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    443     arrays = [asanyarray(arr) for arr in arrays]
    444     if not arrays:
--> 445         raise ValueError('need at least one array to stack')
    446 
    447     shapes = {arr.shape for arr in arrays}

ValueError: need at least one array to stack

## === cell 17
y_flat = Y.reshape(-1)
PRESSURE_MIN = float(np.min(y_flat))
PRESSURE_MAX = float(np.max(y_flat))

vals = np.unique(y_flat[::80])  # one per breath (still covers all discrete pressures)
vals.sort()
diffs = np.diff(vals)
PRESSURE_STEP = float(diffs[diffs > 0].min())

PRESSURE_STEP, PRESSURE_MIN, PRESSURE_MAX



## === cell 18
rounding_pre = (
    np.round((median_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX).reshape(-1)

clipped_pre.shape



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1739050793.py in <cell line: 0>()
      1 rounding_pre = (
----> 2     np.round((median_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
      3 )
      4 clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX).reshape(-1)
      5 

NameError: name 'median_pre' is not defined

## === cell 19
submission_file = pd.read_csv(sample_sub)
if len(submission_file) != len(clipped_pre):
    raise ValueError(
        f"Prediction length mismatch: sub={len(submission_file)} pred={len(clipped_pre)}"
    )

submission_file["pressure"] = clipped_pre.astype(np.float32)
submission_file.to_csv("submission_mediam_round.csv", index=False)

submission_file.head()

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3483830621.py in <cell line: 0>()
      1 submission_file = pd.read_csv(sample_sub)
----> 2 if len(submission_file) != len(clipped_pre):
      3     raise ValueError(
      4         f"Prediction length mismatch: sub={len(submission_file)} pred={len(clipped_pre)}"
      5     )

NameError: name 'clipped_pre' is not defined
