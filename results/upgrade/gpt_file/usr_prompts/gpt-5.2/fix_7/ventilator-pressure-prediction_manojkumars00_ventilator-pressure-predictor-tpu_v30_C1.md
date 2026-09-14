# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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



## === cell 1
sc = StandardScaler()
rc = RobustScaler()



## === cell 2
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 3
def dropCols(df, cols):
    df = df.drop(cols, axis=1)
    return df




## === cell 4
def preProcess(df):
    df = df.copy()

    breath = df["breath_id"].to_numpy()
    u_in = df["u_in"].to_numpy(dtype=np.float64, copy=False)
    t = df["time_step"].to_numpy(dtype=np.float64, copy=False)

    new_breath = np.empty_like(breath, dtype=bool)
    new_breath[0] = True
    new_breath[1:] = breath[1:] != breath[:-1]

    area = t * u_in
    csum = np.cumsum(area)
    offsets = np.zeros_like(csum)
    start_idx = np.flatnonzero(new_breath)
    if len(start_idx) > 1:
        prev_end_idx = start_idx[1:] - 1
        offsets[start_idx[1:]] = csum[prev_end_idx]
        offsets = np.maximum.accumulate(offsets)
    area_cum = csum - offsets

    ucsum = np.cumsum(u_in)
    offsets_u = np.zeros_like(ucsum)
    if len(start_idx) > 1:
        offsets_u[start_idx[1:]] = ucsum[prev_end_idx]
        offsets_u = np.maximum.accumulate(offsets_u)
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




## === cell 5
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



## === cell 6
cols_2_drop = ["id", "breath_id", "time_step"]



## === cell 7
train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")



## === cell 8
rc.fit(train_df)
train_df = rc.transform(train_df).astype(np.float32, copy=False)

train_df = np.ascontiguousarray(
    train_df.reshape(-1, 80, train_df.shape[-1]), dtype=np.float32
)
Y = np.ascontiguousarray(
    Y.to_numpy(dtype=np.float32, copy=False).reshape(-1, 80, 1), dtype=np.float32
)

train_df.shape, Y.shape



## === cell 9
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="C_out_loss",
    factor=0.96,
    patience=4,
    verbose=1,
)




## === cell 10
def build_model():
    Input = layers.Input(shape=([80, train_df.shape[-1]]))

    m1x1 = layers.Bidirectional(
        layers.LSTM(
            units=440,
            return_sequences=True,
            recurrent_dropout=0.0,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m1x2 = layers.Bidirectional(
        layers.LSTM(
            units=360,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m1x3 = layers.Bidirectional(
        layers.LSTM(
            units=240,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m1x4 = layers.Bidirectional(
        layers.LSTM(
            units=180,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m1x5 = layers.Bidirectional(
        layers.LSTM(
            units=100,
            return_sequences=True,
            recurrent_dropout=0.0,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )

    m2x1 = layers.Bidirectional(
        layers.LSTM(
            units=880,
            return_sequences=True,
            recurrent_dropout=0.0,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m2x3 = layers.Bidirectional(
        layers.LSTM(
            units=440,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m2x5 = layers.Bidirectional(
        layers.LSTM(
            units=360,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m2x7 = layers.Bidirectional(
        layers.LSTM(
            units=240,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m2x9 = layers.Bidirectional(
        layers.LSTM(
            units=180,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m2x11 = layers.Bidirectional(
        layers.LSTM(
            units=100,
            return_sequences=True,
            recurrent_dropout=0.0,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )

    m3x1 = layers.Bidirectional(
        layers.GRU(
            units=440,
            return_sequences=True,
            recurrent_dropout=0.0,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m3x2 = layers.Bidirectional(
        layers.GRU(
            units=360,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m3x3 = layers.Bidirectional(
        layers.GRU(
            units=240,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m3x4 = layers.Bidirectional(
        layers.GRU(
            units=180,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m3x5 = layers.Bidirectional(
        layers.GRU(
            units=100,
            return_sequences=True,
            recurrent_dropout=0.0,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
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
            units=100, dropout=0.2, recurrent_dropout=0.0, return_sequences=True
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
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=losses, run_eagerly=False)
    return model




## === cell 11
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




## === cell 12
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




## === cell 13
def make_fold_datasets(X, y, train_idx, valid_idx, batch_size, seed):
    X_tr = X[train_idx]
    y_tr = y[train_idx]
    X_va = X[valid_idx]
    y_va = y[valid_idx]

    opts = tf.data.Options()
    opts.experimental_deterministic = True

    ds_tr = tf.data.Dataset.from_tensor_slices((X_tr, {"C_out": y_tr, "m1_out": y_tr}))
    ds_tr = ds_tr.with_options(opts)
    ds_tr = ds_tr.cache()
    ds_tr = ds_tr.shuffle(
        buffer_size=min(len(X_tr), 8192),
        seed=seed,
        reshuffle_each_iteration=True,
    )
    ds_tr = ds_tr.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    ds_va = tf.data.Dataset.from_tensor_slices((X_va, {"C_out": y_va, "m1_out": y_va}))
    ds_va = ds_va.with_options(opts)
    ds_va = ds_va.cache()
    ds_va = ds_va.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds_tr, ds_va




## === cell 14
EPOCH = 300
BATCH_SIZE = 512

models = []

kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
indices = np.arange(len(train_df), dtype=np.int32)

with strategy.scope():
    for fold, (train_idx, valid_idx) in enumerate(kf.split(indices)):
        print("-" * 15, ">", f"Fold {fold + 1}", "<", "-" * 15)

        model = build_model()

        bw_m1 = BestWeights(monitor="val_m1_out_loss", mode="min")
        bw_c = BestWeights(monitor="val_C_out_loss", mode="min")
        callbacks = [lr_callback, bw_m1, bw_c]

        ds_tr, ds_va = make_fold_datasets(
            train_df, Y, train_idx, valid_idx, BATCH_SIZE, seed=SEED + fold
        )

        model.fit(
            ds_tr,
            validation_data=ds_va,
            epochs=EPOCH,
            callbacks=callbacks,
            verbose=2,
        )

        if bw_m1.best_weights is not None:
            model.set_weights(bw_m1.best_weights)
        models.append(model)

len(models)



## === cell 15
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



## === cell 16
opts = tf.data.Options()
opts.experimental_deterministic = True
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_data)
    .with_options(opts)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

preds_per_model = []
for m in models:
    out = m.predict(test_ds, verbose=1)
    if isinstance(out, (list, tuple)):
        out = out[-1]  # take m1_out
    preds_per_model.append(out)

preds_per_model[0].shape, len(preds_per_model)



## === cell 17
pred_stack = np.stack(preds_per_model, axis=0)
median_pre = np.median(pred_stack, axis=0)

median_pre.shape



## === cell 18
y_flat = Y.reshape(-1)
PRESSURE_MIN = float(np.min(y_flat))
PRESSURE_MAX = float(np.max(y_flat))

vals = np.unique(y_flat[::80])  # one per breath (still covers all discrete pressures)
vals.sort()
diffs = np.diff(vals)
PRESSURE_STEP = float(diffs[diffs > 0].min())

PRESSURE_STEP, PRESSURE_MIN, PRESSURE_MAX



## === cell 19
rounding_pre = (
    np.round((median_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX).reshape(-1)

clipped_pre.shape



## === cell 20
submission_file = pd.read_csv(sample_sub)
if len(submission_file) != len(clipped_pre):
    raise ValueError(
        f"Prediction length mismatch: sub={len(submission_file)} pred={len(clipped_pre)}"
    )

submission_file["pressure"] = clipped_pre.astype(np.float32)
submission_file.to_csv("submission_mediam_round.csv", index=False)

submission_file.head()
