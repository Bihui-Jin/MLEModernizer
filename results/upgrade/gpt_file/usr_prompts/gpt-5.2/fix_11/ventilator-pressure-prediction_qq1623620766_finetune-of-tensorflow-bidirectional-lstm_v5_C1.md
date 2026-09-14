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
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

SEED = 2021
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass

try:
    intra = int(os.environ.get("TF_INTRA_OP_THREADS", "0"))
    inter = int(os.environ.get("TF_INTER_OP_THREADS", "0"))
    if intra <= 0:
        intra = max(1, (os.cpu_count() or 4) - 1)
    if inter <= 0:
        inter = 2
    tf.config.threading.set_intra_op_parallelism_threads(intra)
    tf.config.threading.set_inter_op_parallelism_threads(inter)
except Exception:
    pass

try:
    _GPUS = tf.config.list_physical_devices("GPU")
    HAS_GPU = len(_GPUS) > 0
except Exception:
    HAS_GPU = False

try:
    tf.config.optimizer.set_jit(True if HAS_GPU else False)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
print(
    "Intra/Inter threads:",
    getattr(tf.config.threading, "get_intra_op_parallelism_threads", lambda: None)(),
    getattr(tf.config.threading, "get_inter_op_parallelism_threads", lambda: None)(),
)



## === cell 1
DEBUG = False

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

train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv", dtype=dtypes_train
)
test = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv", dtype=dtypes_test
)
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv",
    dtype={"id": "int32", "pressure": "float32"},
)

if DEBUG:
    train = train.iloc[: 80 * 2000].copy()

print(train.shape, test.shape, submission.shape)
print(train.columns)



## === cell 2
all_pressure = np.sort(train["pressure"].unique())
PRESSURE_MIN = float(all_pressure[0])
PRESSURE_MAX = float(all_pressure[-1])
PRESSURE_STEP = float(all_pressure[1] - all_pressure[0])

print(
    "Pressure grid:",
    PRESSURE_MIN,
    PRESSURE_MAX,
    PRESSURE_STEP,
    "n_unique:",
    len(all_pressure),
)




## === cell 3
def add_features_fast_breathwise_numpy(df: pd.DataFrame, has_pressure: bool):
    df = df.copy()
    df.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    n = len(df)
    assert n % 80 == 0, "Expected fixed-length 80 timesteps per breath."
    nb = n // 80

    time_step = df["time_step"].to_numpy(np.float32, copy=False)
    u_in_flat = df["u_in"].to_numpy(np.float32, copy=False)
    u_out_flat = df["u_out"].to_numpy(np.float32, copy=False)

    u_in = u_in_flat.reshape(nb, 80)
    u_out = u_out_flat.reshape(nb, 80)
    t = time_step.reshape(nb, 80)

    area = np.cumsum(t * u_in, axis=1, dtype=np.float32).reshape(-1)
    u_in_cumsum = np.cumsum(u_in, axis=1, dtype=np.float32).reshape(-1)

    def lag_mat(x, lag):
        out = np.zeros_like(x, dtype=np.float32)
        if lag > 0:
            out[:, lag:] = x[:, :-lag]
        return out.reshape(-1)

    def back_lag_mat(x, lag):
        out = np.zeros_like(x, dtype=np.float32)
        if lag > 0:
            out[:, :-lag] = x[:, lag:]
        return out.reshape(-1)

    lags = (1, 2, 3, 4)
    u_in_lag = {lag: lag_mat(u_in, lag) for lag in lags}
    u_out_lag = {lag: lag_mat(u_out, lag) for lag in lags}
    u_in_back = {lag: back_lag_mat(u_in, lag) for lag in lags}
    u_out_back = {lag: back_lag_mat(u_out, lag) for lag in lags}

    u_in_max = np.max(u_in, axis=1).astype(np.float32, copy=False)
    u_out_max = np.max(u_out, axis=1).astype(np.float32, copy=False)
    u_in_mean = np.mean(u_in, axis=1).astype(np.float32, copy=False)

    breath_u_in_max = np.repeat(u_in_max, 80).astype(np.float32, copy=False)
    breath_u_out_max = np.repeat(u_out_max, 80).astype(np.float32, copy=False)
    breath_u_in_mean = np.repeat(u_in_mean, 80).astype(np.float32, copy=False)

    R = df["R"].to_numpy(np.int16, copy=False)
    C = df["C"].to_numpy(np.int16, copy=False)
    R_levels = (5, 20, 50)
    C_levels = (10, 20, 50)

    out = {
        "time_step": time_step,
        "u_in": u_in_flat,
        "u_out": u_out_flat,
        "area": area,
        "u_in_cumsum": u_in_cumsum,
        "breath_id__u_in__max": breath_u_in_max,
        "breath_id__u_out__max": breath_u_out_max,
        "breath_id__u_in__diffmax": (breath_u_in_max - u_in_flat).astype(
            np.float32, copy=False
        ),
        "breath_id__u_in__diffmean": (breath_u_in_mean - u_in_flat).astype(
            np.float32, copy=False
        ),
        "cross": (u_in_flat * u_out_flat).astype(np.float32, copy=False),
        "cross2": (time_step * u_out_flat).astype(np.float32, copy=False),
        "id": df["id"].to_numpy(np.int32, copy=False),
        "breath_id": df["breath_id"].to_numpy(np.int32, copy=False),
    }

    for lag in lags:
        out[f"u_in_lag{lag}"] = u_in_lag[lag]
        out[f"u_out_lag{lag}"] = u_out_lag[lag]
        out[f"u_in_lag_back{lag}"] = u_in_back[lag]
        out[f"u_out_lag_back{lag}"] = u_out_back[lag]
        out[f"u_in_diff{lag}"] = (u_in_flat - u_in_lag[lag]).astype(
            np.float32, copy=False
        )
        out[f"u_out_diff{lag}"] = (u_out_flat - u_out_lag[lag]).astype(
            np.float32, copy=False
        )

    for rv in R_levels:
        out[f"R_{rv}"] = (R == rv).astype(np.uint8)
    for cv in C_levels:
        out[f"C_{cv}"] = (C == cv).astype(np.uint8)
    for rv in R_levels:
        for cv in C_levels:
            out[f"R__C_{rv}__{cv}"] = ((R == rv) & (C == cv)).astype(np.uint8)

    if has_pressure:
        out["pressure"] = df["pressure"].to_numpy(np.float32, copy=False)

    return pd.DataFrame(out)


train_feat = add_features_fast_breathwise_numpy(train, has_pressure=True)
test_feat = add_features_fast_breathwise_numpy(test, has_pressure=False)

print("After feature engineering:", train_feat.shape, test_feat.shape)



## === cell 4
targets = train_feat[["pressure"]].to_numpy().reshape(-1, 80)

train_feat.drop(["pressure", "id", "breath_id"], axis=1, inplace=True)
test_feat.drop(["id", "breath_id"], axis=1, inplace=True)

train_cols = list(train_feat.columns)
test_cols = list(test_feat.columns)
if train_cols != test_cols:
    train_feat, test_feat = train_feat.align(
        test_feat, join="left", axis=1, fill_value=0
    )

print("Aligned shapes:", train_feat.shape, test_feat.shape)
print(
    "Any NaNs in train/test:",
    train_feat.isna().any().any(),
    test_feat.isna().any().any(),
)



## === cell 5
RS = RobustScaler()

X_train_np = np.ascontiguousarray(train_feat.to_numpy(dtype=np.float32, copy=False))
X_test_np = np.ascontiguousarray(test_feat.to_numpy(dtype=np.float32, copy=False))

train_scaled = np.ascontiguousarray(
    RS.fit_transform(X_train_np).astype(np.float32, copy=False)
)
test_scaled = np.ascontiguousarray(
    RS.transform(X_test_np).astype(np.float32, copy=False)
)

n_features = train_scaled.shape[-1]
train_scaled = train_scaled.reshape(-1, 80, n_features)
test_scaled = test_scaled.reshape(-1, 80, n_features)

print(
    "Model input shapes:",
    train_scaled.shape,
    test_scaled.shape,
    "Targets:",
    targets.shape,
)




## === cell 6
def get_strategy():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # auto-detect
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("Running on TPU:", resolver.cluster_spec().as_dict())
        return tf.distribute.TPUStrategy(resolver)
    except Exception as e:
        print("TPU not available, using default strategy. Reason:", repr(e))
        return tf.distribute.get_strategy()


strategy = get_strategy()
print("Num replicas in sync:", strategy.num_replicas_in_sync)



## === cell 7
EPOCH = 300
BATCH_SIZE = 1024
NUM_FOLDS = 10

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)
fold_splits = list(kf.split(train_scaled, targets))  # materialize once

AUTOTUNE = tf.data.AUTOTUNE
options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_slack = True
except Exception:
    pass

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_scaled)
    .batch(BATCH_SIZE, drop_remainder=False)
    .cache()
    .prefetch(AUTOTUNE)
    .with_options(options)
)


def make_fold_ds_from_numpy(X_np: np.ndarray, y_np: np.ndarray) -> tf.data.Dataset:
    return (
        tf.data.Dataset.from_tensor_slices((X_np, y_np))
        .batch(BATCH_SIZE, drop_remainder=False)
        .cache()
        .prefetch(AUTOTUNE)
        .with_options(options)
    )


def build_callbacks(ckpt_path: str):
    lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
    es = EarlyStopping(
        monitor="val_loss",
        patience=60,
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )
    mc = ModelCheckpoint(
        ckpt_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        verbose=0,
    )
    return [lr, es, mc]


JIT_COMPILE = HAS_GPU  # TPU uses XLA by default under TPUStrategy; GPU benefits from jit_compile=True.


def build_model(input_shape):
    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Bidirectional(keras.layers.LSTM(1024, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
            keras.layers.Dense(128, activation="selu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="mae",
        jit_compile=JIT_COMPILE,
    )
    return model


fold_weight_paths = []

with strategy.scope():
    input_shape = train_scaled.shape[-2:]

    for fold, (train_idx, valid_idx) in enumerate(fold_splits, start=1):
        print("-" * 15, ">", f"Fold {fold}", "<", "-" * 15)

        X_tr = train_scaled[train_idx]
        y_tr = targets[train_idx]
        X_va = train_scaled[valid_idx]
        y_va = targets[valid_idx]

        train_ds = make_fold_ds_from_numpy(X_tr, y_tr)
        valid_ds = make_fold_ds_from_numpy(X_va, y_va)

        model = build_model(input_shape)

        ckpt_path = f"fold_{fold:02d}.weights.h5"
        callbacks = build_callbacks(ckpt_path)

        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCH,
            callbacks=callbacks,
            verbose=2,
        )

        fold_weight_paths.append(ckpt_path)

test_preds = []
with strategy.scope():
    input_shape = train_scaled.shape[-2:]
    for fold, wpath in enumerate(fold_weight_paths, start=1):
        print(f"Predicting test with fold {fold} weights:", wpath)
        model = build_model(input_shape)
        model.load_weights(wpath)
        pred = model.predict(test_ds, verbose=0).squeeze()
        test_preds.append(pred)



## === cell 8
pred_mean = np.mean(np.stack(test_preds, axis=0), axis=0)  # (num_test_breaths, 80)

submission_pressure = pred_mean.reshape(-1)

submission_pressure = (
    np.round((submission_pressure - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
    + PRESSURE_MIN
)
submission_pressure = np.clip(submission_pressure, PRESSURE_MIN, PRESSURE_MAX)

submission_out = submission.copy()
submission_out["pressure"] = submission_pressure.astype(np.float32)

assert submission_out.shape[0] == test.shape[0], (submission_out.shape, test.shape)
assert list(submission_out.columns) == ["id", "pressure"], submission_out.columns

submission_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_out.shape)
print(submission_out.head())



## === cell 9
print("submission.csv exists:", os.path.exists("submission.csv"))
print(pd.read_csv("submission.csv").head())
