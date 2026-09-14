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
import math
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.keras.backend.clear_session()
tf.config.run_functions_eagerly(False)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF choose
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 1
rb = RobustScaler()



## === cell 2
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"

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

read_csv_kwargs = dict(low_memory=False)
try:
    train = pd.read_csv(
        TRAIN_PATH, dtype=dtypes_train, engine="pyarrow", **read_csv_kwargs
    )
    test = pd.read_csv(
        TEST_PATH, dtype=dtypes_test, engine="pyarrow", **read_csv_kwargs
    )
except Exception:
    train = pd.read_csv(TRAIN_PATH, dtype=dtypes_train, **read_csv_kwargs)
    test = pd.read_csv(TEST_PATH, dtype=dtypes_test, **read_csv_kwargs)

print(train.shape, test.shape)
print(train.columns)




## === cell 3
def add_features_fast(df: pd.DataFrame) -> None:
    breath = df["breath_id"].to_numpy()
    u_in = df["u_in"].to_numpy()

    new_breath = np.empty_like(breath, dtype=bool)
    new_breath[0] = True
    new_breath[1:] = breath[1:] != breath[:-1]

    u_in_lag1 = np.empty_like(u_in, dtype=u_in.dtype)
    u_in_lag1[0] = 0.0
    u_in_lag1[1:] = u_in[:-1]
    u_in_lag1[new_breath] = 0.0

    u_in_diff1 = u_in - u_in_lag1

    cs = np.cumsum(u_in, dtype=np.float64)
    start_idx = np.flatnonzero(new_breath)
    offsets = np.zeros_like(cs)
    if len(start_idx) > 1:
        prev = start_idx[1:] - 1
        offsets[start_idx[1:]] = cs[prev]
    offsets = np.maximum.accumulate(offsets)
    u_in_cumsum = (cs - offsets).astype(u_in.dtype, copy=False)

    df["u_in_lag1"] = u_in_lag1
    df["u_in_diff1"] = u_in_diff1
    df["u_in_cumsum"] = u_in_cumsum


add_features_fast(train)
add_features_fast(test)



## === cell 4
targets = train["pressure"].to_numpy().reshape(-1, 80, 1)

test_id = test["id"].copy()

train.drop(columns=["id", "breath_id", "pressure", "time_step"], inplace=True)
test.drop(columns=["id", "breath_id", "time_step"], inplace=True)

print(
    "Train features:",
    train.shape,
    "Test features:",
    test.shape,
    "Targets:",
    targets.shape,
)



## === cell 5
rb.fit(train)

train_new = rb.transform(train)
test_new = rb.transform(test)

train_new = np.asarray(train_new, dtype=np.float32, order="C")
test_new = np.asarray(test_new, dtype=np.float32, order="C")

train_re = train_new.reshape(-1, 80, train_new.shape[-1])
test_re = test_new.reshape(-1, 80, test_new.shape[-1])

targets = np.asarray(targets, dtype=np.float32, order="C")

print("train_re:", train_re.shape, "test_re:", test_re.shape)




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
        run_eagerly=False,  # graph mode
        jit_compile=True,  # Speed: ensure compile uses XLA for the training step when possible.
    )
    return model




## === cell 7
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # auto-detect
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print("TPU not available, using default strategy:", type(strategy).__name__)
    print("TPU error (ignored):", repr(e))



## === cell 8
EPOCH = 260
BATCH_SIZE = 512

reduce_lr = ReduceLROnPlateau(monitor="val_loss", verbose=1, factor=0.87, patience=8)

trained_models = []

AUTOTUNE = tf.data.AUTOTUNE

opts = tf.data.Options()
opts.deterministic = True
opts.experimental_optimization.apply_default_optimizations = True
try:
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
except Exception:
    pass


def make_ds_xy_epoch(x_np: np.ndarray, y_np, training: bool, epoch: int = 0):
    if y_np is None:
        ds = tf.data.Dataset.from_tensor_slices(x_np)
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.with_options(opts)
        return ds.prefetch(AUTOTUNE)

    ds = tf.data.Dataset.from_tensor_slices((x_np, y_np))
    if training:
        buf = int(min(len(x_np), 8192))
        ds = ds.shuffle(
            buffer_size=buf,
            seed=int(SEED + epoch),  # deterministic per-epoch reshuffle
            reshuffle_each_iteration=True,
        )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.with_options(opts)
    return ds.prefetch(AUTOTUNE)


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
        if (self.mode == "min" and current < self.best) or (
            self.mode == "max" and current > self.best
        ):
            self.best = float(current)
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def train_one_fold_custom(
    model: tf.keras.Model,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    epochs: int,
    reduce_lr_cb: ReduceLROnPlateau,
    best_cb: InMemoryBestWeights,
    verbose: int = 2,
):
    optimizer = model.optimizer
    loss_fn = model.loss

    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            pred = model(xb, training=True)
            loss = loss_fn(yb, pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(xb, yb):
        pred = model(xb, training=False)
        return loss_fn(yb, pred)

    best_cb.set_model(model)
    reduce_lr_cb.set_model(model)
    reduce_lr_cb.on_train_begin({})

    for epoch in range(epochs):
        train_ds = make_ds_xy_epoch(x_tr, y_tr, training=True, epoch=epoch)
        valid_ds = make_ds_xy_epoch(x_va, y_va, training=False, epoch=epoch)

        tr_losses = []
        for xb, yb in train_ds:
            tr_losses.append(train_step(xb, yb))
        tr_loss = tf.reduce_mean(tr_losses)

        va_losses = []
        for xb, yb in valid_ds:
            va_losses.append(val_step(xb, yb))
        va_loss = tf.reduce_mean(va_losses)

        logs = {
            "loss": float(tr_loss.numpy()),
            "val_loss": float(va_loss.numpy()),
        }

        best_cb.on_epoch_end(epoch, logs)

        reduce_lr_cb.on_epoch_end(epoch, logs)

        if verbose == 2:
            print(
                f"Epoch {epoch+1}/{epochs} - loss: {logs['loss']:.6f} - val_loss: {logs['val_loss']:.6f} - lr: {float(tf.keras.backend.get_value(optimizer.learning_rate)):.6g}"
            )

    reduce_lr_cb.on_train_end({})
    best_cb.on_train_end({})


with strategy.scope():
    kf = KFold(n_splits=7, shuffle=True, random_state=SEED)

    base_model = build_model()
    initial_weights = base_model.get_weights()

    trained_weights = []  # list of weight lists, one per fold

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_re, targets)):
        print("-" * 30, ">", f"Fold {fold+1}", "<", "-" * 30)

        x_tr = train_re[train_idx]
        y_tr = targets[train_idx]
        x_va = train_re[valid_idx]
        y_va = targets[valid_idx]

        base_model.set_weights(initial_weights)

        best_cb = InMemoryBestWeights(monitor="val_loss", mode="min")

        train_one_fold_custom(
            base_model,
            x_tr,
            y_tr,
            x_va,
            y_va,
            epochs=EPOCH,
            reduce_lr_cb=reduce_lr,
            best_cb=best_cb,
            verbose=2,
        )

        trained_weights.append(base_model.get_weights())

        del x_tr, y_tr, x_va, y_va

print("Trained folds (weight sets):", len(trained_weights))



## === cell 9
unique_pressures = np.unique(targets)
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = (unique_pressures[1] - unique_pressures[0]).item()
PRESSURE_MIN = sorted_pressures[0].item()
PRESSURE_MAX = sorted_pressures[-1].item()

print("PRESSURE_STEP, MIN, MAX:", PRESSURE_STEP, PRESSURE_MIN, PRESSURE_MAX)



## === cell 10
test_re = np.asarray(test_re, dtype=np.float32, order="C")
test_ds = make_ds_xy_epoch(test_re, None, training=False)

if len(trained_weights) == 0:
    print(
        "WARNING: No trained models found; writing a valid submission with constant pressure=0."
    )
    clipped_pre = np.zeros((test_re.shape[0] * 80, 1), dtype=np.float32)
else:
    n = test_re.shape[0] * 80
    k = len(trained_weights)
    pred_mat = np.empty((n, k), dtype=np.float32)

    with strategy.scope():
        pred_model = build_model()
        for i, w in enumerate(trained_weights):
            print(f"Predicting fold {i+1} ...")
            pred_model.set_weights(w)
            pred_mat[:, i] = (
                pred_model.predict(test_ds, verbose=0)
                .reshape(-1)
                .astype(np.float32, copy=False)
            )
        del pred_model

    median_pre = np.median(pred_mat, axis=1).reshape(-1, 1)

    rounding_pre = (
        np.round((median_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
        + PRESSURE_MIN
    )
    clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)

print("Pred shape:", clipped_pre.shape)

submission_file = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
submission_file["pressure"] = clipped_pre.reshape(-1)
submission_file.to_csv("submission.csv", index=False)

print(submission_file.head())
print("Wrote submission.csv with shape:", submission_file.shape)
