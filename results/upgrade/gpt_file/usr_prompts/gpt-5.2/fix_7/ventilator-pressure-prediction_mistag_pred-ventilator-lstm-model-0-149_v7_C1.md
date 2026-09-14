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
import gc
import json
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)




## === cell 1
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

train_df = pd.read_csv(
    TRAIN_PATH,
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
        "pressure": "float32",
    },
)
test_df = pd.read_csv(
    TEST_PATH,
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
    },
)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print(train_df.shape, test_df.shape, sample_sub.shape)
print(train_df.columns)




## === cell 2
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    needs_sort = True
    b = df["breath_id"].to_numpy(copy=False)
    if b.size >= 2 and np.all(b[1:] >= b[:-1]):
        ts = df["time_step"].to_numpy(copy=False)
        same = b[1:] == b[:-1]
        if same.any():
            if np.all(ts[1:][same] >= ts[:-1][same]):
                needs_sort = False
        else:
            needs_sort = False

    if needs_sort:
        df = df.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
            drop=True
        )
    else:
        df = df.reset_index(drop=True)

    n = len(df)
    if n % 80 != 0:
        raise ValueError(f"Expected rows multiple of 80, got {n}")

    breath_ids = df["breath_id"].to_numpy(copy=False)
    unique_breaths, counts = np.unique(breath_ids, return_counts=True)
    if not np.all(counts == 80):
        raise ValueError(
            "Each breath must have exactly 80 timesteps; found counts mismatch."
        )

    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80)
    u_out = df["u_out"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80)
    time_step = df["time_step"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80)

    u_in_cumsum = np.cumsum(u_in, axis=1, dtype=np.float32)

    dt = np.empty_like(time_step, dtype=np.float32)
    dt[:, 0] = 0.0
    dt[:, 1:] = time_step[:, 1:] - time_step[:, :-1]

    u_in_lag1 = np.zeros_like(u_in, dtype=np.float32)
    u_in_lag1[:, 1:] = u_in[:, :-1]
    u_in_lag2 = np.zeros_like(u_in, dtype=np.float32)
    u_in_lag2[:, 2:] = u_in[:, :-2]
    u_in_lead1 = np.zeros_like(u_in, dtype=np.float32)
    u_in_lead1[:, :-1] = u_in[:, 1:]
    u_in_lead2 = np.zeros_like(u_in, dtype=np.float32)
    u_in_lead2[:, :-2] = u_in[:, 2:]

    u_out_lag1 = np.zeros_like(u_out, dtype=np.float32)
    u_out_lag1[:, 1:] = u_out[:, :-1]
    u_out_lead1 = np.zeros_like(u_out, dtype=np.float32)
    u_out_lead1[:, :-1] = u_out[:, 1:]

    u_in_diff1 = u_in - u_in_lag1
    u_in_diff2 = u_in_lag1 - u_in_lag2

    R = df["R"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80)
    C = df["C"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80)
    RC = (R * C).astype(np.float32, copy=False)
    u_in_R = u_in / (R + 1e-6)
    u_in_C = u_in / (C + 1e-6)

    df["u_in_cumsum"] = u_in_cumsum.reshape(-1)
    df["dt"] = dt.reshape(-1)

    df["u_in_lag1"] = u_in_lag1.reshape(-1)
    df["u_in_lag2"] = u_in_lag2.reshape(-1)
    df["u_in_lead1"] = u_in_lead1.reshape(-1)
    df["u_in_lead2"] = u_in_lead2.reshape(-1)

    df["u_out_lag1"] = u_out_lag1.reshape(-1)
    df["u_out_lead1"] = u_out_lead1.reshape(-1)

    df["u_in_diff1"] = u_in_diff1.reshape(-1)
    df["u_in_diff2"] = u_in_diff2.reshape(-1)

    df["RC"] = RC.reshape(-1)
    df["u_in_R"] = u_in_R.reshape(-1)
    df["u_in_C"] = u_in_C.reshape(-1)

    df["u_in"] = df["u_in"].astype(np.float32, copy=False)
    df["time_step"] = df["time_step"].astype(np.float32, copy=False)
    df["R"] = df["R"].astype(np.float32, copy=False)
    df["C"] = df["C"].astype(np.float32, copy=False)
    df["u_out"] = df["u_out"].astype(np.float32, copy=False)

    return df


train_fe = add_features(train_df)
test_fe = add_features(test_df)

DROP_COLS = ["id", "breath_id", "pressure"]
FEATURE_COLS = [c for c in train_fe.columns if c not in DROP_COLS]

print("n_features:", len(FEATURE_COLS))
print("Example feature cols:", FEATURE_COLS[:10])




## === cell 3
def to_sequences(df_fe: pd.DataFrame, feature_cols, with_target: bool):
    breath_ids = df_fe["breath_id"].to_numpy(copy=False)
    unique_breaths = breath_ids[::80].copy()

    feats_2d = df_fe.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)
    X = feats_2d.reshape(-1, 80, len(feature_cols))

    u_out = df_fe["u_out"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80)

    if with_target:
        y = df_fe["pressure"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80)
        return unique_breaths, X, y, u_out
    else:
        ids = df_fe["id"].to_numpy(copy=False).reshape(-1, 80)
        return unique_breaths, X, u_out, ids


train_breaths, X_train_raw, y_train, u_out_train = to_sequences(
    train_fe, FEATURE_COLS, with_target=True
)
test_breaths, X_test_raw, u_out_test, test_ids_seq = to_sequences(
    test_fe, FEATURE_COLS, with_target=False
)

print("X_train:", X_train_raw.shape, "y_train:", y_train.shape)
print("X_test:", X_test_raw.shape, "test_ids_seq:", test_ids_seq.shape)




## === cell 4
scaler = RobustScaler()

X_train_2d = X_train_raw.reshape(-1, X_train_raw.shape[-1])
scaler.fit(X_train_2d)

X_train = (
    scaler.transform(X_train_2d)
    .astype(np.float32, copy=False)
    .reshape(X_train_raw.shape)
)

X_test_2d = X_test_raw.reshape(-1, X_test_raw.shape[-1])
X_test = (
    scaler.transform(X_test_2d).astype(np.float32, copy=False).reshape(X_test_raw.shape)
)

del X_train_2d, X_test_2d, X_train_raw, X_test_raw, train_fe, test_fe
gc.collect()




## === cell 5
pressure_values = train_df["pressure"].values.astype(np.float32, copy=False)
P_MIN = float(np.min(pressure_values))
P_MAX = float(np.max(pressure_values))
P_UNIQUE = np.sort(train_df["pressure"].unique())
P_STEP = float(np.min(np.diff(P_UNIQUE)))
print("P_MIN:", P_MIN, "P_MAX:", P_MAX, "P_STEP:", P_STEP, "unique:", len(P_UNIQUE))




## === cell 6
def get_hardware_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Running on TPU", tpu.master())
    except Exception:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()

    return tpu, strategy


tpu, strategy = get_hardware_strategy()
print("Replicas:", strategy.num_replicas_in_sync)




## === cell 7
def build_model(input_shape):
    inp = keras.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dense(64, activation="selu")(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mae")
    return model


config = {
    "BATCH_SIZE": 256,
    "EPOCHS": 8,
    "NFOLDS": 5,
}
config




## === cell 8
_DATA_OPT = tf.data.Options()
_DATA_OPT.experimental_deterministic = True


def make_ds_xy_from_indices(X, y, indices, batch_size=256, training=False):
    idx = tf.convert_to_tensor(indices, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices(idx)

    if training:
        ds = ds.shuffle(
            buffer_size=min(int(idx.shape[0]), 8192),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def _gather(i):
        xi = tf.gather(X, i)
        yi = tf.gather(y, i)
        return xi, yi

    ds = ds.map(_gather, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.with_options(_DATA_OPT)
    return ds


def make_ds_x_from_tensor(X, batch_size=256):
    ds = tf.data.Dataset.from_tensor_slices(X)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.with_options(_DATA_OPT)
    return ds


gkf = GroupKFold(n_splits=config["NFOLDS"])
groups = train_breaths
fold_indices = list(gkf.split(X_train, y_train, groups=groups))

X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train[..., None], dtype=tf.float32)  # (n,80,1)
X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)

oof = np.zeros_like(y_train, dtype=np.float32)
test_preds_folds = []

ds_te_x = make_ds_x_from_tensor(X_test_tf, batch_size=config["BATCH_SIZE"])

with strategy.scope():
    for fold, (tr_idx, va_idx) in enumerate(fold_indices):
        print(
            f"\nFold {fold+1}/{config['NFOLDS']} - train breaths: {len(tr_idx)} valid breaths: {len(va_idx)}"
        )

        model = build_model(X_train.shape[1:])

        ds_tr = make_ds_xy_from_indices(
            X_train_tf,
            y_train_tf,
            tr_idx,
            batch_size=config["BATCH_SIZE"],
            training=True,
        )
        ds_va = make_ds_xy_from_indices(
            X_train_tf,
            y_train_tf,
            va_idx,
            batch_size=config["BATCH_SIZE"],
            training=False,
        )

        model.fit(
            ds_tr,
            validation_data=ds_va,
            epochs=config["EPOCHS"],
            verbose=2,
        )

        ds_va_x = ds_va.map(
            lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        )
        va_pred = (
            model.predict(ds_va_x, verbose=0).squeeze(-1).astype(np.float32, copy=False)
        )
        oof[va_idx] = va_pred

        te_pred = (
            model.predict(ds_te_x, verbose=0).squeeze(-1).astype(np.float32, copy=False)
        )
        test_preds_folds.append(te_pred)

        del model, ds_tr, ds_va, ds_va_x, va_pred, te_pred
        gc.collect()
        keras.backend.clear_session()




## === cell 9
mask = (u_out_train == 0).astype(bool)
oof_mae = np.mean(np.abs(oof[mask] - y_train[mask]))
print("OOF MAE (inspiratory only):", oof_mae)




## === cell 10
test_pred = np.median(np.stack(test_preds_folds, axis=0), axis=0)  # (n_breaths, 80)

test_pred_flat = test_pred.reshape(-1)
test_pred_flat = np.round((test_pred_flat - P_MIN) / P_STEP) * P_STEP + P_MIN
test_pred_flat = np.clip(test_pred_flat, P_MIN, P_MAX)

test_ids_flat = test_ids_seq.reshape(-1)

sub = pd.DataFrame(
    {
        "id": test_ids_flat.astype(np.int64, copy=False),
        "pressure": test_pred_flat.astype(np.float32, copy=False),
    }
)

sub = sub.sort_values("id").reset_index(drop=True)
sub = sub[["id", "pressure"]]
assert list(sub.columns) == list(sample_sub.columns), (sub.columns, sample_sub.columns)
assert len(sub) == len(sample_sub), (len(sub), len(sample_sub))

out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
out_path = os.path.join(out_dir, "submission.csv")
sub.to_csv(out_path, index=False)

print(sub.head())
print("Wrote submission.csv to:", out_path, "with shape:", sub.shape)
print(
    "File exists:",
    os.path.exists(out_path),
    "size(bytes):",
    os.path.getsize(out_path) if os.path.exists(out_path) else None,
)
