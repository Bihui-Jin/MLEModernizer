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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    import multiprocessing

    ncpu = multiprocessing.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(max(1, ncpu // 2))
    tf.config.threading.set_inter_op_parallelism_threads(max(2, ncpu // 4))
except Exception:
    pass

print("TF version:", tf.__version__)




## === cell 1
def get_distribution_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # raises if no TPU
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        print("TPU initialized")
        return tf.distribute.TPUStrategy(tpu)
    except Exception as e:
        print("TPU not available, using default strategy. Reason:", repr(e))
        return tf.distribute.get_strategy()


strategy = get_distribution_strategy()
print("Num replicas in sync:", strategy.num_replicas_in_sync)



## === cell 2
TRAIN_PATH = "/kaggle/input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "/kaggle/input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"

TRAIN_DTYPES = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
TEST_DTYPES = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

train = pd.read_csv(TRAIN_PATH, dtype=TRAIN_DTYPES, low_memory=False)
test = pd.read_csv(TEST_PATH, dtype=TEST_DTYPES, low_memory=False)
sample_sub = pd.read_csv(
    SAMPLE_SUB_PATH, dtype={"id": "int32", "pressure": "float32"}, low_memory=False
)

print(train.shape, test.shape, sample_sub.shape)
train.head()




## === cell 3
def add_u_in_features_fast(df: pd.DataFrame) -> None:
    breath_id = df["breath_id"].to_numpy(copy=False)
    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)

    n = len(df)

    lag1 = np.empty(n, dtype=np.float32)
    lag1[0] = 0.0
    same_breath_prev = breath_id[1:] == breath_id[:-1]
    lag1[1:] = np.where(same_breath_prev, u_in[:-1], 0.0)
    df["u_in_lag1"] = lag1
    df["u_in_diff1"] = (u_in - lag1).astype(np.float32, copy=False)

    start = np.empty(n, dtype=bool)
    start[0] = True
    start[1:] = ~same_breath_prev
    global_csum = np.cumsum(u_in, dtype=np.float32)
    start_idx = np.flatnonzero(start)
    csum_before = np.empty_like(start_idx, dtype=np.float32)
    csum_before[0] = 0.0
    if len(start_idx) > 1:
        csum_before[1:] = global_csum[start_idx[1:] - 1]
    gid = np.cumsum(start) - 1
    df["u_in_cumsum"] = (global_csum - csum_before[gid]).astype(np.float32, copy=False)


add_u_in_features_fast(train)
add_u_in_features_fast(test)



## === cell 4
targets = train["pressure"].to_numpy().reshape(-1, 80, 1)

test_id = test["id"].copy()

train.drop(columns=["id", "breath_id", "pressure", "time_step"], inplace=True)
test.drop(columns=["id", "breath_id", "time_step"], inplace=True)

print(
    "Train features:",
    train.shape,
    "Targets:",
    targets.shape,
    "Test features:",
    test.shape,
)



## === cell 5
rb = RobustScaler()
train_arr = train.to_numpy(dtype=np.float32, copy=False)
test_arr = test.to_numpy(dtype=np.float32, copy=False)

rb.fit(train_arr)

train_new = rb.transform(train_arr).astype(np.float32, copy=False)
test_new = rb.transform(test_arr).astype(np.float32, copy=False)

train_re = train_new.reshape(-1, 80, train_new.shape[-1])
test_re = test_new.reshape(-1, 80, test_new.shape[-1])

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
        steps_per_execution=256,
    )
    return model




## === cell 7
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

EPOCH = 260
BATCH_SIZE = 512

N_SPLITS = 7
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

saved_model_paths = []

AUTOTUNE = tf.data.AUTOTUNE

DATASET_OPTIONS = tf.data.Options()
DATASET_OPTIONS.experimental_distribute.auto_shard_policy = (
    tf.data.experimental.AutoShardPolicy.DATA
)
try:
    DATASET_OPTIONS.experimental_slack = True
except Exception:
    pass
try:
    DATASET_OPTIONS.experimental_deterministic = True
except Exception:
    pass

X_np = train_re  # NumPy float32, shape (n_breaths, 80, n_feat)
y_np = targets  # NumPy float32, shape (n_breaths, 80, 1)

X_tf = tf.constant(X_np)  # once; re-used for all folds
y_tf = tf.constant(y_np)


@tf.function
def _gather_xy(i):
    return tf.gather(X_tf, i), tf.gather(y_tf, i)


@tf.function
def _gather_x(i):
    return tf.gather(X_tf, i)


def make_index_dataset(indices, training=False):
    ds = tf.data.Dataset.from_tensor_slices(indices)
    ds = ds.with_options(DATASET_OPTIONS)
    if training:
        buf = min(int(len(indices)), 8192)
        ds = ds.shuffle(buffer_size=buf, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_gather_xy, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def make_index_dataset_x(indices):
    ds = tf.data.Dataset.from_tensor_slices(indices)
    ds = ds.with_options(DATASET_OPTIONS)
    ds = ds.map(_gather_x, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


for fold, (train_idx, valid_idx) in enumerate(kf.split(X_np), start=1):
    print("-" * 30, ">", f"Fold {fold}", "<", "-" * 30)

    ckpt_path = f"NewModel{fold}.weights.h5"
    saved_model_paths.append(ckpt_path)

    train_ds = make_index_dataset(train_idx.astype(np.int32, copy=False), training=True)
    valid_ds = make_index_dataset(
        valid_idx.astype(np.int32, copy=False), training=False
    )

    with strategy.scope():
        model = build_model()

    if os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 0:
        print(f"Found existing checkpoint {ckpt_path}; loading and skipping training.")
        model.load_weights(ckpt_path)
        continue

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", verbose=1, factor=0.87, patience=8
    )
    ckpt_cb = ModelCheckpoint(
        filepath=ckpt_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        save_freq="epoch",
        verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCH,
        callbacks=[reduce_lr, ckpt_cb],
        verbose=2,
    )

    if os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 0:
        model.load_weights(ckpt_path)
    else:
        raise RuntimeError(
            f"Training finished but no checkpoint was written: {ckpt_path}"
        )

print("Saved models:", saved_model_paths)



## === cell 8
unique_pressures = np.unique(targets)
sorted_pressures = np.sort(unique_pressures)
PRESSURE_STEP = (unique_pressures[1] - unique_pressures[0]).item()
PRESSURE_MIN = sorted_pressures[0].item()
PRESSURE_MAX = sorted_pressures[-1].item()

print("Pressure min/max/step:", PRESSURE_MIN, PRESSURE_MAX, PRESSURE_STEP)


def round_to_nearest_pressure(x):
    x = np.clip(x, PRESSURE_MIN, PRESSURE_MAX)
    return np.round((x - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN




## === cell 9
preds = np.zeros((test_re.shape[0], 80, 1), dtype=np.float32)

test_indices = np.arange(test_re.shape[0], dtype=np.int32)
X_test_tf = tf.constant(test_re)


@tf.function
def _gather_test(i):
    return tf.gather(X_test_tf, i)


def make_test_dataset(indices):
    ds = tf.data.Dataset.from_tensor_slices(indices)
    ds = ds.with_options(DATASET_OPTIONS)
    ds = ds.map(_gather_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


test_ds = make_test_dataset(test_indices)

with strategy.scope():
    infer_model = build_model()

for path in saved_model_paths:
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        raise FileNotFoundError(
            f"Missing fold weights {path}. Training did not create it."
        )
    infer_model.load_weights(path)
    fold_pred = infer_model.predict(test_ds, verbose=0)
    preds += fold_pred.astype(np.float32, copy=False)

preds /= len(saved_model_paths)

preds_flat = preds.reshape(-1)
preds_flat = round_to_nearest_pressure(preds_flat)

print("Pred shape:", preds.shape, "Flat:", preds_flat.shape)

submission = pd.DataFrame({"id": test_id.values, "pressure": preds_flat})
submission = submission.sort_values("id").reset_index(drop=True)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print(submission.tail())
