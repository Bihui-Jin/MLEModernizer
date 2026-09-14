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
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import RobustScaler

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())

DATA_DIR = "../input/ventilator-pressure-prediction"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")




## === cell 1
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    breath = df["breath_id"].to_numpy(copy=False)
    time_step = df["time_step"].to_numpy(copy=False)

    needs_sort = bool(
        np.any(breath[1:] < breath[:-1])
        or np.any((breath[1:] == breath[:-1]) & (time_step[1:] < time_step[:-1]))
    )
    if needs_sort:
        df = df.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
            drop=True
        )
        breath = df["breath_id"].to_numpy(copy=False)
        time_step = df["time_step"].to_numpy(copy=False)

    time_step = time_step.astype(np.float32, copy=False)
    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)
    u_out = df["u_out"].to_numpy(dtype=np.int16, copy=False)

    n = breath.shape[0]

    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    is_new[1:] = breath[1:] != breath[:-1]

    dt = np.empty(n, dtype=np.float32)
    dt[is_new] = 0.0
    dt[~is_new] = time_step[1:][~is_new[1:]] - time_step[:-1][~is_new[1:]]

    prod = u_in * dt
    global_cum = np.cumsum(prod, dtype=np.float32)

    start_idx = np.flatnonzero(is_new)
    prev_idx = start_idx - 1
    offsets = np.zeros(start_idx.shape[0], dtype=np.float32)
    if prev_idx.shape[0] > 0:
        offsets[1:] = global_cum[prev_idx[1:]].astype(np.float32, copy=False)

    seg_id = np.cumsum(is_new, dtype=np.int32) - 1
    u_in_cum = global_cum - offsets[seg_id]

    u_in_lag1 = np.zeros(n, dtype=np.float32)
    u_in_lag2 = np.zeros(n, dtype=np.float32)
    u_in_lead1 = np.zeros(n, dtype=np.float32)
    u_out_lag1 = np.zeros(n, dtype=np.int16)

    not_new = ~is_new
    u_in_lag1[not_new] = u_in[:-1][not_new[1:]]
    u_out_lag1[not_new] = u_out[:-1][not_new[1:]]

    is_new2 = np.empty(n, dtype=bool)
    is_new2[:2] = True
    is_new2[2:] = breath[2:] != breath[:-2]
    not_new2 = ~is_new2
    u_in_lag2[not_new2] = u_in[:-2][not_new2[2:]]

    is_last = np.empty(n, dtype=bool)
    is_last[-1] = True
    is_last[:-1] = breath[:-1] != breath[1:]
    not_last = ~is_last
    u_in_lead1[not_last] = u_in[1:][not_last[:-1]]

    u_in_diff1 = u_in - u_in_lag1

    df = df.assign(
        dt=dt,
        u_in_cum=u_in_cum,
        u_in_lag1=u_in_lag1,
        u_in_lag2=u_in_lag2,
        u_in_lead1=u_in_lead1,
        u_out_lag1=u_out_lag1,
        u_in_diff1=u_in_diff1.astype(np.float32, copy=False),
    )

    df["R"] = df["R"].astype(np.int16, copy=False)
    df["C"] = df["C"].astype(np.int16, copy=False)

    return df


def to_3d_breath_array(df_feat: pd.DataFrame, feature_cols):
    arr = df_feat[feature_cols].to_numpy(dtype=np.float32, copy=False)
    n = arr.shape[0]
    if n % 80 != 0:
        raise ValueError(
            f"Row count {n} not divisible by 80; cannot reshape to (-1,80,feat)."
        )
    return arr.reshape(-1, 80, len(feature_cols))




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
test_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

train = pd.read_csv(TRAIN_PATH, dtype=train_dtypes, usecols=list(train_dtypes.keys()))
test = pd.read_csv(TEST_PATH, dtype=test_dtypes, usecols=list(test_dtypes.keys()))

train_feat = add_features(train)
test_feat = add_features(test)

y = train_feat["pressure"].to_numpy(np.float32, copy=False)
w = 1.0 - train_feat["u_out"].to_numpy(np.float32, copy=False)

drop_cols = {"id", "breath_id", "pressure"}
feature_cols = [c for c in train_feat.columns if c not in drop_cols]

print("Num features:", len(feature_cols))
print("Feature columns:", feature_cols[:15], "..." if len(feature_cols) > 15 else "")

X_train_raw = train_feat[feature_cols].to_numpy(dtype=np.float32, copy=False)
X_test_raw = test_feat[feature_cols].to_numpy(dtype=np.float32, copy=False)

del train_feat, test_feat
gc.collect()

scaler = RobustScaler()

X_train_2d = scaler.fit_transform(X_train_raw).astype(np.float32, copy=False)
X_test_2d = scaler.transform(X_test_raw).astype(np.float32, copy=False)

del X_train_raw, X_test_raw
gc.collect()

n_features = len(feature_cols)
X_train = np.ascontiguousarray(X_train_2d.reshape(-1, 80, n_features), dtype=np.float32)
X_test = np.ascontiguousarray(X_test_2d.reshape(-1, 80, n_features), dtype=np.float32)

del X_train_2d, X_test_2d
gc.collect()

y_seq = np.ascontiguousarray(y.reshape(-1, 80, 1), dtype=np.float32)
w_seq = np.ascontiguousarray(w.reshape(-1, 80, 1), dtype=np.float32)

print(
    "X_train:",
    X_train.shape,
    "y_seq:",
    y_seq.shape,
    "w_seq:",
    w_seq.shape,
    "X_test:",
    X_test.shape,
)




## === cell 3
def get_hardware_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Running on TPU", tpu.master())
    except Exception:
        tpu = None

    if tpu is not None:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()
    return tpu, strategy


tpu, strategy = get_hardware_strategy()
print("Num replicas:", strategy.num_replicas_in_sync)




## === cell 4
@tf.function
def weighted_mae(y_true, y_pred, sample_weight):
    err = tf.abs(y_true - y_pred) * sample_weight
    denom = tf.reduce_sum(sample_weight) + 1e-8
    return tf.reduce_sum(err) / denom


class WeightedMAELoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        y = y_true[..., :1]
        w = y_true[..., 1:2]
        err = tf.abs(y - y_pred) * w
        denom = tf.reduce_sum(w) + 1e-8
        return tf.reduce_sum(err) / denom


y_pack = np.concatenate([y_seq, w_seq], axis=-1).astype(np.float32, copy=False)




## === cell 5
def build_model(n_features: int):
    inp = keras.Input(shape=(80, n_features))
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dense(128, activation="selu")(x)
    x = layers.Dense(64, activation="selu")(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    return model


BATCH_SIZE = 256
EPOCHS = 15
LR = 1e-3

with strategy.scope():
    model = build_model(len(feature_cols))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss=WeightedMAELoss(),
    )

model.summary()




## === cell 6
n_breaths = X_train.shape[0]
idx = np.arange(n_breaths)
np.random.shuffle(idx)

val_frac = 0.1
n_val = int(n_breaths * val_frac)
val_idx = idx[:n_val]
trn_idx = idx[n_val:]

X_tr, X_va = X_train[trn_idx], X_train[val_idx]
y_tr, y_va = y_pack[trn_idx], y_pack[val_idx]

del X_train, y_pack, y_seq, w_seq, idx, val_idx, trn_idx
gc.collect()

AUTOTUNE = tf.data.AUTOTUNE
options = tf.data.Options()
options.deterministic = True

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
    .with_options(options)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((X_va, y_va))
    .with_options(options)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        "best_model.keras", monitor="val_loss", save_best_only=True, verbose=1
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2,
)

model = keras.models.load_model(
    "best_model.keras", custom_objects={"WeightedMAELoss": WeightedMAELoss}
)




## === cell 7
test_options = tf.data.Options()
test_options.deterministic = True
test_ds = (
    tf.data.Dataset.from_tensor_slices(X_test)
    .with_options(test_options)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

test_pred_seq = model.predict(test_ds, verbose=2)  # (n_breaths, 80, 1)
test_pred = test_pred_seq.reshape(-1).astype(np.float32, copy=False)

p_min = float(train["pressure"].min())
p_max = float(train["pressure"].max())
test_pred = np.clip(test_pred, p_min, p_max)

print("Pred shape:", test_pred.shape, "min/max:", test_pred.min(), test_pred.max())

sub = pd.read_csv(SAMPLE_SUB_PATH)

if len(sub) != len(test_pred):
    raise ValueError(f"Submission rows {len(sub)} != predictions {len(test_pred)}")

sub["pressure"] = test_pred
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
