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
import random
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.preprocessing import RobustScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("Python OK, TF version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))



## === cell 1
rb = RobustScaler()



## === cell 2
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"

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
test_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

read_csv_kwargs = dict()
try:
    import pyarrow  # noqa: F401

    read_csv_kwargs["engine"] = "pyarrow"
except Exception:
    pass

train = pd.read_csv(TRAIN_PATH, dtype=train_dtypes, **read_csv_kwargs)
test = pd.read_csv(TEST_PATH, dtype=test_dtypes, **read_csv_kwargs)

print(train.shape, test.shape)
print(train.columns)



## === cell 3
assert len(train) % 80 == 0, "Train rows must be multiple of 80 (timesteps per breath)."
assert len(test) % 80 == 0, "Test rows must be multiple of 80 (timesteps per breath)."




## === cell 4
def add_breath_features_fast(df: pd.DataFrame) -> pd.DataFrame:
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
    return df


train = add_breath_features_fast(train)
test = add_breath_features_fast(test)



## === cell 5
train.head()



## === cell 6
targets = train["pressure"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80, 1)
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



## === cell 7
rb.fit(train)

train_new = rb.transform(train).astype(np.float32, copy=False)
test_new = rb.transform(test).astype(np.float32, copy=False)
train_new = np.ascontiguousarray(train_new)
test_new = np.ascontiguousarray(test_new)

train_re = train_new.reshape(-1, 80, train_new.shape[-1])
test_re = test_new.reshape(-1, 80, test_new.shape[-1])

print("train_re:", train_re.shape, "test_re:", test_re.shape)



## === cell 8
unique_pressures = np.unique(targets)
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = (unique_pressures[1] - unique_pressures[0]).item()
PRESSURE_MIN = sorted_pressures[0].item()
PRESSURE_MAX = sorted_pressures[-1].item()

print(
    "Pressure grid:",
    PRESSURE_MIN,
    PRESSURE_MAX,
    "step",
    PRESSURE_STEP,
    "unique",
    len(unique_pressures),
)




## === cell 9
def build_model_n():
    input_layer = layers.Input(shape=([80, train_re.shape[-1]]))

    lstm0 = layers.Bidirectional(
        layers.LSTM(
            440, return_sequences=True, name="lstm0", kernel_initializer="random_normal"
        )
    )
    lstm1 = layers.Bidirectional(
        layers.LSTM(
            360,
            dropout=0.2,
            return_sequences=True,
            name="lstm1",
            kernel_initializer="random_normal",
        )
    )
    lstm2 = layers.Bidirectional(
        layers.LSTM(
            240,
            dropout=0.2,
            return_sequences=True,
            name="lstm2",
            kernel_initializer="random_normal",
        )
    )
    lstm3 = layers.Bidirectional(
        layers.LSTM(
            180,
            dropout=0.2,
            return_sequences=True,
            name="lstm3",
            kernel_initializer="random_normal",
        )
    )
    lstm4 = layers.Bidirectional(
        layers.LSTM(
            100, return_sequences=True, name="lstm4", kernel_initializer="random_normal"
        )
    )

    lstm_i1 = layers.Bidirectional(
        layers.LSTM(
            360,
            dropout=0.2,
            return_sequences=True,
            name="lstm_i1",
            kernel_initializer="random_normal",
        )
    )
    lstm_02 = layers.Bidirectional(
        layers.LSTM(
            240,
            dropout=0.2,
            return_sequences=True,
            name="lstm_02",
            kernel_initializer="random_normal",
        )
    )
    lstm_13 = layers.Bidirectional(
        layers.LSTM(
            180,
            dropout=0.2,
            return_sequences=True,
            name="lstm_13",
            kernel_initializer="random_normal",
        )
    )

    dense0 = layers.Dense(
        64, activation="relu", name="dense0", kernel_initializer="random_normal"
    )
    dense1 = layers.Dense(1, name="dense1")

    lstm0_out = lstm0(input_layer)
    lstm_i1_out = lstm_i1(input_layer)

    lstm1_out = lstm1(lstm0_out)
    lstm_02_out = lstm_02(lstm0_out)

    lstm_13_out = lstm_13(lstm1_out)
    lstm2_out = lstm2(
        layers.BatchNormalization()(layers.Multiply()([lstm1_out, lstm_i1_out]))
    )

    lstm3_out = lstm3(
        layers.BatchNormalization()(layers.Multiply()([lstm2_out, lstm_02_out]))
    )
    lstm4_out = lstm4(
        layers.BatchNormalization()(layers.Multiply()([lstm3_out, lstm_13_out]))
    )

    dense0_out = dense0(lstm4_out)
    dense1_out = dense1(dense0_out)

    model = tf.keras.Model(inputs=input_layer, outputs=dense1_out, name="BasicModel")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanAbsoluteError(),
        jit_compile=True,
        steps_per_execution=16,
    )
    return model




## === cell 10
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
        jit_compile=True,
        steps_per_execution=16,
    )
    return model




## === cell 11
try:
    if os.environ.get("PLOT_MODEL", "0") == "1":
        tf.keras.utils.plot_model(build_model(), show_shapes=True, to_file="model.png")
        print("Saved model plot to model.png")
    else:
        print("plot_model skipped (set PLOT_MODEL=1 to enable).")
except Exception as e:
    print("plot_model skipped due to:", repr(e))




## === cell 12
def get_strategy():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print("Running on TPU")
        return strategy
    except Exception as e:
        print("TPU not available, using default strategy. Reason:", repr(e))
        return tf.distribute.get_strategy()


strategy = get_strategy()
print("Num replicas:", strategy.num_replicas_in_sync)



## === cell 13
EPOCH = 250
BATCH_SIZE = 512

reduce_lr = ReduceLROnPlateau(monitor="val_loss", verbose=1, factor=0.87, patience=8)

n_splits = 8

saved_model_paths = [f"Model{fold+1}.weights.h5" for fold in range(n_splits)]

input_model_paths = [
    os.path.join("../input/ventilator-pressure-prediction", p)
    for p in saved_model_paths
]
resolved_model_paths = []
for w, inp in zip(saved_model_paths, input_model_paths):
    resolved_model_paths.append(w if os.path.exists(w) else inp)

need_train = any(not os.path.exists(p) for p in resolved_model_paths)
force_train = os.environ.get("FORCE_TRAIN", "0") == "1"

if need_train or force_train:
    n_samples = train_re.shape[0]
    all_idx = np.arange(n_samples, dtype=np.int32)
    fold_ids = (all_idx % n_splits).astype(np.int32)

    fold_train_idx = []
    fold_valid_idx = []
    for f in range(n_splits):
        fold_valid_idx.append(all_idx[fold_ids == f])
        fold_train_idx.append(all_idx[fold_ids != f])

    AUTOTUNE = tf.data.AUTOTUNE
    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
    except Exception:
        pass
    try:
        options.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass

    X_tf = tf.convert_to_tensor(train_re, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(targets, dtype=tf.float32)

    def make_xy_ds_from_indices(idx_np: np.ndarray, training: bool):
        idx = tf.convert_to_tensor(idx_np, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices(idx)
        if training:
            ds = ds.shuffle(
                buffer_size=min(8192, idx_np.shape[0]),
                seed=SEED,
                reshuffle_each_iteration=True,
            )
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)

        def _gather(batch_idx):
            return tf.gather(X_tf, batch_idx), tf.gather(y_tf, batch_idx)

        ds = ds.map(_gather, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.prefetch(AUTOTUNE)
        ds = ds.with_options(options)
        return ds

    with strategy.scope():
        for fold in range(n_splits):
            print("-" * 30, ">", f"Fold {fold+1}/{n_splits}", "<", "-" * 30)

            model = build_model()

            ckpt_path = saved_model_paths[fold]
            call_back = tf.keras.callbacks.ModelCheckpoint(
                ckpt_path,
                verbose=0,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
            )

            tr_idx = fold_train_idx[fold]
            va_idx = fold_valid_idx[fold]

            train_ds = make_xy_ds_from_indices(tr_idx, training=True)
            valid_ds = make_xy_ds_from_indices(va_idx, training=False)

            model.fit(
                train_ds,
                validation_data=valid_ds,
                epochs=EPOCH,
                callbacks=[call_back, reduce_lr],
                verbose=2,
            )

    resolved_model_paths = saved_model_paths
else:
    print("All fold checkpoints found; skipping training to meet runtime limit.")
    print("Resolved weight paths:", resolved_model_paths)

print("Fold model weight filenames:", saved_model_paths)



## === cell 14
models = []
with strategy.scope():
    for p in resolved_model_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected checkpoint not found: {p}")
        m = build_model()
        m.load_weights(p)
        models.append(m)

print("Loaded models:", len(models))



## === cell 15
if os.environ.get("RUN_TRAIN_EVAL", "0") == "1":
    for i, model in enumerate(models, 1):
        loss = model.evaluate(train_re, targets, verbose=0)
        print(f"Model {i} train MAE (all steps): {loss:.6f}")
else:
    print("Train evaluation skipped (set RUN_TRAIN_EVAL=1 to enable).")



## === cell 16
AUTOTUNE = tf.data.AUTOTUNE
test_ds = (
    tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(test_re, dtype=tf.float32))
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

options_test = tf.data.Options()
options_test.experimental_deterministic = True
try:
    options_test.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
except Exception:
    pass
try:
    options_test.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
test_ds = test_ds.with_options(options_test)

n_test_rows = test_re.shape[0] * 80
n_models = len(models)

preds = []
for model in models:
    pred = model.predict(test_ds, verbose=0).reshape(-1).astype(np.float32, copy=False)
    preds.append(pred)
preds = np.stack(preds, axis=1)  # shape (n_test_rows, n_models)

k1 = n_models // 2 - 1
k2 = n_models // 2
part = np.partition(preds, (k1, k2), axis=1)
median_pre = (part[:, k1] + part[:, k2]) * np.float32(0.5)

rounding_pre = (
    np.round((median_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX).astype(
    np.float32, copy=False
)

print(
    "Pred shape:", clipped_pre.shape, "min/max:", clipped_pre.min(), clipped_pre.max()
)



## === cell 17
submission_file = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv",
    dtype={"id": "int32", "pressure": "float32"},
)

if len(submission_file) != clipped_pre.reshape(-1).shape[0]:
    raise ValueError(
        f"Submission rows ({len(submission_file)}) != predictions ({clipped_pre.reshape(-1).shape[0]})."
    )

submission_file["pressure"] = clipped_pre.reshape(-1).astype(np.float32, copy=False)
submission_file.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_file.shape)



## === cell 18
submission_file.head()
