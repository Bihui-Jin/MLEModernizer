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
import glob
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

print("TF version:", tf.__version__)

SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
try:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
except Exception as _e:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 1
rb = RobustScaler()



## === cell 2
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

fallback_roots = [
    "/kaggle/input/ventilator-pressure-prediction",
    "/kaggle/data/ventilator-pressure-prediction",
    "/kaggle/input",
    "/kaggle/data",
]


def _resolve_path(default_path: str, filename: str) -> str:
    if os.path.exists(default_path):
        return default_path
    for root in fallback_roots:
        cand = os.path.join(root, filename)
        if os.path.exists(cand):
            return cand
    return default_path  # will fail later with a clear error if missing


TRAIN_PATH = _resolve_path(TRAIN_PATH, "train.csv")
TEST_PATH = _resolve_path(TEST_PATH, "test.csv")
SAMPLE_SUB_PATH = _resolve_path(SAMPLE_SUB_PATH, "sample_submission.csv")

print("Using paths:", TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH)

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

train = pd.read_csv(TRAIN_PATH, dtype=train_dtypes, engine="c")
test = pd.read_csv(TEST_PATH, dtype=test_dtypes, engine="c")

print(train.shape, test.shape)
print(train.columns)



## === cell 3
pass




## === cell 4
def _add_engineered_features_fast(df: pd.DataFrame) -> None:
    u = df["u_in"].to_numpy(dtype=np.float32, copy=False)
    n = u.shape[0]
    if n % 80 != 0:
        u_lag1 = df.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)
        df["u_in_lag1"] = u_lag1.astype(np.float32, copy=False)
        df["u_in_diff1"] = (df["u_in"] - df["u_in_lag1"]).astype(np.float32, copy=False)
        df["u_in_cumsum"] = (
            df.groupby("breath_id", sort=False)["u_in"]
            .cumsum()
            .astype(np.float32, copy=False)
        )
        return

    u2 = u.reshape(-1, 80)
    lag = np.zeros_like(u2, dtype=np.float32)
    lag[:, 1:] = u2[:, :-1]
    diff = u2 - lag
    cumsum = np.cumsum(u2, axis=1, dtype=np.float32)

    df["u_in_lag1"] = lag.reshape(-1)
    df["u_in_diff1"] = diff.reshape(-1)
    df["u_in_cumsum"] = cumsum.reshape(-1)


_add_engineered_features_fast(train)
_add_engineered_features_fast(test)



## === cell 5
print("Engineered features added.")



## === cell 6
targets = train["pressure"].to_numpy(dtype=np.float32).reshape(-1, 80, 1)
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



## === cell 7
train_values = train.to_numpy(dtype=np.float32, copy=False)
test_values = test.to_numpy(dtype=np.float32, copy=False)

n_breaths_train = train_values.shape[0] // 80
n_breaths_test = test_values.shape[0] // 80

fit_idx = np.arange(0, n_breaths_train * 80, 80, dtype=np.int32)
rb.fit(train_values[fit_idx])

train_new = rb.transform(train_values).astype(np.float32, copy=False)
test_new = rb.transform(test_values).astype(np.float32, copy=False)

train_re = np.ascontiguousarray(train_new.reshape(-1, 80, train_new.shape[-1]))
test_re = np.ascontiguousarray(test_new.reshape(-1, 80, test_new.shape[-1]))

print("Reshaped:", train_re.shape, test_re.shape)



## === cell 8
pass




## === cell 9
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
    )
    return model




## === cell 10
print("Model builder ready.")



## === cell 11
pass



## === cell 12
EPOCH = 260
BATCH_SIZE = 512

try:
    gpus = tf.config.list_physical_devices("GPU")
    for _g in gpus:
        tf.config.experimental.set_memory_growth(_g, True)
except Exception:
    pass

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Using TPU strategy")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print("TPU not available; using default strategy:", type(strategy).__name__)
    print("TPU connect error (ignored):", repr(e))




## === cell 13
def exp_decay(epoch):
    initial_lrate = 0.001
    k = 0.006
    lrate = initial_lrate * math.exp(-k * epoch)
    return lrate


lr_schedular = tf.keras.callbacks.LearningRateScheduler(exp_decay)



## === cell 14
reduce_lr = ReduceLROnPlateau(monitor="val_loss", verbose=1, factor=0.87, patience=8)

oof_val_losses = []
saved_model_paths = []


def _existing_fold_ckpt_paths(n_folds: int):
    candidates = []
    if "models_paths" in globals():
        candidates.extend([p for p in models_paths if isinstance(p, str) and p])

    existing = []
    seen = set()
    for p in candidates:
        if p and (p not in seen) and os.path.isfile(p):
            seen.add(p)
            existing.append(p)
            if len(existing) >= n_folds:
                return existing[:n_folds]

    limited_dirs = [
        "../input/kfolds",
        "../input/latest-2fold",
        "../input/latest-ensembles",
        "../input/ventilator-pressure-prediction",
    ]
    patterns = ("*.h5", "*.keras")
    for d in limited_dirs:
        if not os.path.isdir(d):
            continue
        for pat in patterns:
            for p in sorted(glob.glob(os.path.join(d, pat))):
                if p and (p not in seen) and os.path.isfile(p):
                    seen.add(p)
                    existing.append(p)
                    if len(existing) >= n_folds:
                        return existing[:n_folds]
    return None


N_FOLDS = 7



## === cell 15
unique_pressures = np.unique(targets)
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = (unique_pressures[1] - unique_pressures[0]).item()
PRESSURE_MIN = sorted_pressures[0].item()
PRESSURE_MAX = sorted_pressures[-1].item()

print("Pressure step/min/max:", PRESSURE_STEP, PRESSURE_MIN, PRESSURE_MAX)



## === cell 16
print(PRESSURE_STEP, PRESSURE_MIN, PRESSURE_MAX)



## === cell 17
models_paths = [
    "../input/kfolds/OldModel1.h5",
    "../input/kfolds/OldModel2.h5",
    "../input/kfolds/OldModel3.h5",
    "../input/kfolds/OldModel4.h5",
    "../input/kfolds/OldModel5.h5",
    "../input/kfolds/OldModel6.h5",
    "../input/kfolds/OldModel7.h5",
    "../input/kfolds/OldModel8.h5",
    "../input/kfolds/OldModel9.h5",
    "../input/latest-2fold/Model16049.h5",
    "../input/latest-2fold/Model16275.h5",
    "../input/kfolds/Model16277.h5",
    "../input/kfolds/Model16179.h5",
]



## === cell 18
pass



## === cell 19
pass



## === cell 20
Ensembles = ["../input/latest-ensembles/submission.csv"]



## === cell 21
pretrained = _existing_fold_ckpt_paths(N_FOLDS)
if pretrained is not None:
    saved_model_paths = pretrained
    print("Using pretrained checkpoints (skipping training):", saved_model_paths)
else:
    N_FOLDS_TRAIN = 2
    print(
        "No pretrained checkpoints found under ../input/. "
        f"Training locally with {N_FOLDS_TRAIN} folds to produce submission."
    )

    AUTOTUNE = tf.data.AUTOTUNE
    options = tf.data.Options()
    try:
        options.experimental_deterministic = True
    except Exception:
        pass

    kf = KFold(n_splits=N_FOLDS_TRAIN, shuffle=True, random_state=SEED)

    fold = 0
    for tr_idx, va_idx in kf.split(train_re):
        fold += 1
        x_tr, y_tr = train_re[tr_idx], targets[tr_idx]
        x_va, y_va = train_re[va_idx], targets[va_idx]

        tr_ds = tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
        tr_ds = (
            tr_ds.with_options(options)
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(AUTOTUNE)
        )
        va_ds = tf.data.Dataset.from_tensor_slices((x_va, y_va))
        va_ds = (
            va_ds.with_options(options)
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(AUTOTUNE)
        )

        with strategy.scope():
            model = build_model()

        ckpt_path = f"fold{fold}.keras"
        cb = [
            lr_schedular,
            reduce_lr,
            tf.keras.callbacks.ModelCheckpoint(
                ckpt_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
            ),
        ]

        hist = model.fit(
            tr_ds,
            validation_data=va_ds,
            epochs=EPOCH,
            verbose=2,
            callbacks=cb,
        )
        best_val = float(np.min(hist.history.get("val_loss", [np.nan])))
        oof_val_losses.append(best_val)
        saved_model_paths.append(ckpt_path)

        del model, hist, x_tr, y_tr, x_va, y_va, tr_ds, va_ds

    print("Training done. Fold val losses:", oof_val_losses)
    print("Saved model paths:", saved_model_paths)



## === cell 22
pass



## === cell 23
n_folds = len(saved_model_paths)
n_rows = test_re.shape[0] * test_re.shape[1]  # breaths * 80

all_fold_preds = []

for j, ckpt_path in enumerate(saved_model_paths, start=0):
    print(f"Loading best model for fold {j+1} from {ckpt_path}")
    mdl = tf.keras.models.load_model(ckpt_path, compile=False)

    y = mdl.predict(test_re, batch_size=BATCH_SIZE, verbose=0)
    fold_pred = y.reshape(-1).astype(np.float32, copy=False)

    if fold_pred.shape[0] != n_rows:
        raise ValueError(f"Fold pred length {fold_pred.shape[0]} != expected {n_rows}")
    all_fold_preds.append(fold_pred)

    del mdl, y, fold_pred

print("Num fold preds:", n_folds, "Each pred length:", n_rows)



## === cell 24
pass



## === cell 25
print("Collected fold predictions:", len(all_fold_preds), "folds")



## === cell 26
print((len(all_fold_preds), all_fold_preds[0].shape[0]))



## === cell 27
fold_preds_arr = np.stack(all_fold_preds, axis=0)  # (n_folds, n_rows)
del all_fold_preds

if n_folds % 2 == 1:
    k = n_folds // 2
    median_pre = np.partition(fold_preds_arr, kth=k, axis=0)[k, :]
else:
    k1 = n_folds // 2 - 1
    k2 = n_folds // 2
    part = np.partition(fold_preds_arr, kth=(k1, k2), axis=0)
    median_pre = (part[k1, :] + part[k2, :]) * 0.5

rounding_pre = (
    np.round((median_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)

print("Final preds shape:", clipped_pre.shape)



## === cell 28
pass



## === cell 29
print(clipped_pre.shape)



## === cell 30
submission_file = pd.read_csv(
    SAMPLE_SUB_PATH,
    dtype={"id": "int32", "pressure": "float32"},
    engine="c",
)
if len(submission_file) != len(clipped_pre):
    raise ValueError(
        f"Submission rows ({len(submission_file)}) != predictions ({len(clipped_pre)})"
    )

submission_file["pressure"] = clipped_pre.astype(np.float32, copy=False)
submission_file.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_file.shape)



## === cell 31
print(submission_file.head())
print(submission_file.tail())
