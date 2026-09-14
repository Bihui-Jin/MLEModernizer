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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ["TF_NUM_INTRAOP_THREADS"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(os.environ["TF_NUM_INTEROP_THREADS"])
    )
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass


def _pick_existing(*candidates: str) -> str:
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


train_path = _pick_existing(
    "/kaggle/input/ventilator-pressure-prediction/train.csv",
    "../input/ventilator-pressure-prediction/train.csv",
)
test_path = _pick_existing(
    "/kaggle/input/ventilator-pressure-prediction/test.csv",
    "../input/ventilator-pressure-prediction/test.csv",
)
sample_sub_path = _pick_existing(
    "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv",
    "../input/ventilator-pressure-prediction/sample_submission.csv",
)

print("Using paths:")
print("train:", train_path)
print("test :", test_path)
print("sub  :", sample_sub_path)




## === cell 1
def dropCols(df, cols):
    return df.drop(columns=cols)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    if n % 80 != 0:
        if not df.index.is_monotonic_increasing:
            df = df.reset_index(drop=True)
        df = df.sort_values(
            ["breath_id", "time_step"], kind="mergesort", ignore_index=True
        )
        g = df.groupby("breath_id", sort=False, observed=True)
        df["u_in_lag1"] = g["u_in"].shift(1, fill_value=0).astype(np.float32)
        u_in_f = df["u_in"].astype(np.float32)
        df["diff_u_in1"] = (u_in_f - df["u_in_lag1"]).astype(np.float32)
        df["u_in_cumsum"] = g["u_in"].cumsum().astype(np.float32)
        return df

    u = df["u_in"].to_numpy(dtype=np.float32, copy=False)
    u2 = u.reshape(-1, 80)

    lag = np.empty_like(u2)
    lag[:, 0] = 0.0
    lag[:, 1:] = u2[:, :-1]

    df["u_in_lag1"] = lag.reshape(-1).astype(np.float32, copy=False)
    df["diff_u_in1"] = (
        u - df["u_in_lag1"].to_numpy(dtype=np.float32, copy=False)
    ).astype(np.float32, copy=False)
    df["u_in_cumsum"] = (
        np.cumsum(u2, axis=1, dtype=np.float32)
        .reshape(-1)
        .astype(np.float32, copy=False)
    )
    return df




## === cell 2
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
usecols_train = list(train_dtypes.keys())

train_data = pd.read_csv(train_path, usecols=usecols_train, dtype=train_dtypes)
train_data = add_features(train_data)

cols_2_drop = ["id", "breath_id", "time_step"]
train_df = dropCols(train_data, cols_2_drop)

Y = train_df.pop("pressure").to_numpy(dtype=np.float32, copy=False)

rc = RobustScaler()
rc.fit(train_df)
train_X = rc.transform(train_df).astype(np.float32, copy=False)

n_features = train_X.shape[-1]
train_X = train_X.reshape(-1, 80, n_features)
train_Y = Y.reshape(-1, 80, 1)

print("train_X:", train_X.shape, "train_Y:", train_Y.shape)




## === cell 3
def build_model(n_features: int):
    model = tf.keras.Sequential()
    model.add(
        layers.Bidirectional(
            layers.LSTM(120, return_sequences=True), input_shape=[80, n_features]
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
        optimizer=opt, loss=tf.keras.losses.MeanAbsoluteError(), metrics=["mae"]
    )
    return model


callback_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.9,
    patience=10,
    verbose=1,
)




## === cell 4
EPOCH = 60
BATCH_SIZE = 1024
N_SPLITS = 3

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_mae = []
fold_models = []

_BASE_OPTIONS = tf.data.Options()
_BASE_OPTIONS.experimental_deterministic = True


def make_ds_from_arrays(X_np, y_np=None, training=False):
    if y_np is None:
        ds = tf.data.Dataset.from_tensor_slices(X_np).with_options(_BASE_OPTIONS)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X_np, y_np)).with_options(
            _BASE_OPTIONS
        )

    if training:
        shuffle_buf = min(len(X_np), 8192)
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


for fold, (tr_idx, va_idx) in enumerate(kf.split(train_X), start=1):
    print("-" * 15, ">", f"Fold {fold}", "<", "-" * 15)

    model = build_model(n_features=train_X.shape[-1])

    callback_best = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EPOCH + 1,  # never triggers early stop
        restore_best_weights=True,
        verbose=0,
    )

    X_tr, y_tr = train_X[tr_idx], train_Y[tr_idx]
    X_va, y_va = train_X[va_idx], train_Y[va_idx]

    train_ds = make_ds_from_arrays(X_tr, y_tr, training=True)
    valid_ds = make_ds_from_arrays(X_va, y_va, training=False)

    _ = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCH,
        callbacks=[callback_lr, callback_best],
        verbose=2,
    )

    fold_models.append(model)

    eval_loss, eval_mae = model.evaluate(valid_ds, verbose=0)
    fold_mae = float(eval_mae)
    oof_mae.append(fold_mae)
    print(f"Fold {fold} validation MAE (unmasked): {fold_mae:.6f}")

print(f"Mean validation MAE (unmasked): {np.mean(oof_mae):.6f}")




## === cell 5
test_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}
usecols_test = list(test_dtypes.keys())

test_data = pd.read_csv(test_path, usecols=usecols_test, dtype=test_dtypes)
test_data = add_features(test_data)
test_df = dropCols(test_data, cols_2_drop)

test_X = rc.transform(test_df).astype(np.float32, copy=False)
test_X = test_X.reshape(-1, 80, test_X.shape[-1])

print("test_X:", test_X.shape)




## === cell 6
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_X)
    .with_options(_BASE_OPTIONS)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

if len(fold_models) == 0:
    raise RuntimeError(
        "No trained fold models found; training must complete before inference."
    )

pred_list = []
for m in fold_models:
    pred = m.predict(test_ds, verbose=0).reshape(-1, 1)
    pred_list.append(pred)

predictions_table = np.concatenate(pred_list, axis=-1)
mean_pre = np.mean(predictions_table, axis=-1).reshape(-1)

print("predictions_table:", predictions_table.shape, "mean_pre:", mean_pre.shape)




## === cell 7
unique_pressures = np.unique(Y)
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = float((unique_pressures[1] - unique_pressures[0]).item())
PRESSURE_MIN = float(sorted_pressures[0].item())
PRESSURE_MAX = float(sorted_pressures[-1].item())

rounding_pre = (
    np.round((mean_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
rounding_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)

print(
    "Pressure grid:",
    PRESSURE_STEP,
    PRESSURE_MIN,
    PRESSURE_MAX,
    "preds:",
    rounding_pre.shape,
)




## === cell 8
submission_file = pd.read_csv(sample_sub_path, usecols=["id", "pressure"])
if "id" not in submission_file.columns or "pressure" not in submission_file.columns:
    raise ValueError(
        "sample_submission.csv does not have required columns: id, pressure"
    )

if len(submission_file) != len(rounding_pre):
    raise ValueError(
        f"Submission length mismatch: sample={len(submission_file)} preds={len(rounding_pre)}"
    )

submission_file["pressure"] = rounding_pre.astype(np.float32)
submission_file.to_csv("submission.csv", index=False)

print(submission_file.head())
print("Wrote submission.csv with shape:", submission_file.shape)
