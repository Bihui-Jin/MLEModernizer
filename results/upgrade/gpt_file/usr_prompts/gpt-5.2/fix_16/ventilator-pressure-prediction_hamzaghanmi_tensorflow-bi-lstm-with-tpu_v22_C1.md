# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.1614826481484438

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import time
import logging

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

tf.get_logger().setLevel(logging.ERROR)

os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
np.random.seed(42)
tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    if not tf.config.list_physical_devices("GPU") and "TPU" not in str(
        tf.config.list_logical_devices()
    ):
        ncpu = os.cpu_count() or 2
        tf.config.threading.set_intra_op_parallelism_threads(max(1, ncpu))
        tf.config.threading.set_inter_op_parallelism_threads(max(1, ncpu // 2))
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)



## === cell 1
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

train = pd.read_csv(TRAIN_PATH, dtype=dtypes_train)
test = pd.read_csv(TEST_PATH, dtype=dtypes_test)



## === cell 2
print("train shape:", train.shape)
print("test shape:", test.shape)
print("train columns:", list(train.columns))
print("test columns:", list(test.columns))



## === cell 3
print(train.head(3))




## === cell 4
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    breath = df["breath_id"].to_numpy(np.int32, copy=False)
    time_step = df["time_step"].to_numpy(np.float32, copy=False)
    u_in = df["u_in"].to_numpy(np.float32, copy=False)
    u_out = df["u_out"].to_numpy(np.int8, copy=False)
    R = df["R"].to_numpy(np.int16, copy=False)
    C = df["C"].to_numpy(np.int16, copy=False)

    n = breath.shape[0]

    new_breath = np.empty(n, dtype=bool)
    new_breath[0] = True
    new_breath[1:] = breath[1:] != breath[:-1]
    start_idx = np.flatnonzero(new_breath).astype(np.int64, copy=False)

    idx = np.arange(n, dtype=np.int32)
    grp = np.cumsum(new_breath, dtype=np.int32) - 1  # 0..n_breaths-1
    grp_start = start_idx[grp].astype(np.int32, copy=False)
    count = (idx - grp_start + 1).astype(np.int16, copy=False)

    area_step = time_step * u_in
    c_area = np.cumsum(area_step, dtype=np.float64).astype(np.float32, copy=False)
    c_uin = np.cumsum(u_in, dtype=np.float64).astype(np.float32, copy=False)

    start_minus1 = grp_start - 1
    base_area = np.where(start_minus1 >= 0, c_area[start_minus1], 0.0).astype(
        np.float32, copy=False
    )
    base_uin = np.where(start_minus1 >= 0, c_uin[start_minus1], 0.0).astype(
        np.float32, copy=False
    )

    area = (c_area - base_area).astype(np.float32, copy=False)
    u_in_cumsum = (c_uin - base_uin).astype(np.float32, copy=False)
    u_in_cummean = (u_in_cumsum / count.astype(np.float32, copy=False)).astype(
        np.float32, copy=False
    )

    lag1_same = np.zeros(n, dtype=np.int8)
    lag2_same = np.zeros(n, dtype=np.int8)
    lag1_same[1:] = (breath[1:] == breath[:-1]).astype(np.int8, copy=False)
    lag2_same[2:] = (breath[2:] == breath[:-2]).astype(np.int8, copy=False)

    u_in_lag = np.zeros(n, dtype=np.float32)
    u_in_lag2 = np.zeros(n, dtype=np.float32)
    u_out_lag2 = np.zeros(n, dtype=np.float32)
    u_in_lag[1:] = u_in[:-1]
    u_in_lag2[2:] = u_in[:-2]
    u_out_lag2[2:] = u_out[:-2].astype(np.float32, copy=False)

    df["area"] = area
    u_out_f = u_out.astype(np.float32, copy=False)
    df["cross"] = (u_in * u_out_f).astype(np.float32, copy=False)
    df["cross2"] = (time_step * u_out_f).astype(np.float32, copy=False)

    df["u_in_cumsum"] = u_in_cumsum
    df["count"] = count
    df["one"] = 1
    df["u_in_cummean"] = u_in_cummean

    df["breath_id_lag"] = np.r_[0, breath[:-1]].astype(np.int32, copy=False)
    df["breath_id_lag2"] = np.r_[0, 0, breath[:-2]].astype(np.int32, copy=False)
    df["breath_id_lagsame"] = lag1_same
    df["breath_id_lag2same"] = lag2_same

    lag1_f = lag1_same.astype(np.float32, copy=False)
    lag2_f = lag2_same.astype(np.float32, copy=False)
    df["u_in_lag"] = (u_in_lag * lag1_f).astype(np.float32, copy=False)
    df["u_in_lag2"] = (u_in_lag2 * lag2_f).astype(np.float32, copy=False)
    df["u_out_lag2"] = (u_out_lag2 * lag2_f).astype(np.float32, copy=False)

    df["RC"] = (
        R.astype(np.int32, copy=False) * 1000 + C.astype(np.int32, copy=False)
    ).astype(np.int32, copy=False)
    return df


train = add_features(train)
test = add_features(test)



## === cell 5
y = train["pressure"].to_numpy(np.float32, copy=False).reshape(-1, 80)

drop_cols = [
    "pressure",
    "id",
    "breath_id",
    "one",
    "count",
    "breath_id_lag",
    "breath_id_lag2",
    "breath_id_lagsame",
    "breath_id_lag2same",
    "u_out_lag2",
]
train.drop(drop_cols, axis=1, inplace=True)
test.drop(
    [
        "id",
        "breath_id",
        "one",
        "count",
        "breath_id_lag",
        "breath_id_lag2",
        "breath_id_lagsame",
        "breath_id_lag2same",
        "u_out_lag2",
    ],
    axis=1,
    inplace=True,
)



## === cell 6
print("Features after FE (train):", train.shape, " (test):", test.shape)



## === cell 7
R_cats = np.array([5, 20, 50], dtype=np.int16)
C_cats = np.array([10, 20, 50], dtype=np.int16)
RC_cats = np.array(
    sorted({int(r) * 1000 + int(c) for r in R_cats for c in C_cats}), dtype=np.int32
)

_R_cols = [f"R_{int(x)}" for x in R_cats.tolist()]
_C_cols = [f"C_{int(x)}" for x in C_cats.tolist()]
_RC_cols = [f"RC_{int(x)}" for x in RC_cats.tolist()]


def add_fixed_onehots_inplace(df: pd.DataFrame) -> pd.DataFrame:
    Rv = df["R"].to_numpy(np.int16, copy=False)
    Cv = df["C"].to_numpy(np.int16, copy=False)
    RCv = df["RC"].to_numpy(np.int32, copy=False)

    R_oh = (Rv[:, None] == R_cats[None, :]).astype(np.float32, copy=False)
    C_oh = (Cv[:, None] == C_cats[None, :]).astype(np.float32, copy=False)
    RC_oh = (RCv[:, None] == RC_cats[None, :]).astype(np.float32, copy=False)

    for j, col in enumerate(_R_cols):
        df[col] = R_oh[:, j]
    for j, col in enumerate(_C_cols):
        df[col] = C_oh[:, j]
    for j, col in enumerate(_RC_cols):
        df[col] = RC_oh[:, j]

    df.drop(["R", "C", "RC"], axis=1, inplace=True)
    return df


train = add_fixed_onehots_inplace(train)
test = add_fixed_onehots_inplace(test)

train_cols = train.columns
test = test.reindex(columns=train_cols, fill_value=0)

rb = RobustScaler()
train_np = train.to_numpy(dtype=np.float32, copy=False)
test_np = test.to_numpy(dtype=np.float32, copy=False)
rb.fit(train_np)
train = rb.transform(train_np).astype(np.float32, copy=False)
test = rb.transform(test_np).astype(np.float32, copy=False)

del train_np, test_np
gc.collect()



## === cell 8
n_features = train.shape[-1]
train = np.ascontiguousarray(train.reshape(-1, 80, n_features), dtype=np.float32)
test = np.ascontiguousarray(test.reshape(-1, 80, n_features), dtype=np.float32)
y = np.ascontiguousarray(y, dtype=np.float32)
gc.collect()



## === cell 9
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU:", tpu.master())
except Exception:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        strategy = tf.distribute.MirroredStrategy()
        try:
            keras.mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled (GPU).")
        except Exception as e:
            print("Mixed precision not enabled:", repr(e))
    else:
        strategy = tf.distribute.get_strategy()

print("REPLICAS:", strategy.num_replicas_in_sync)




## === cell 10
def plot_hist(hist):
    return




## === cell 11
BATCH_SIZE = 512
steps_per_epoch = int(np.ceil((len(train) * 0.8) / BATCH_SIZE))


def create_model(n_features):
    with strategy.scope():
        model = keras.Sequential(
            [
                layers.Input(shape=(80, n_features)),
                layers.Bidirectional(layers.LSTM(700, return_sequences=True)),
                layers.Bidirectional(layers.LSTM(512, return_sequences=True)),
                layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
                layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
                layers.Dense(128, activation="elu"),
                layers.Dense(1),
            ]
        )

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=200 * steps_per_epoch,
            decay_rate=1e-5,
        )
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(optimizer=opt, loss="mae")
    return model




## === cell 12
AUTOTUNE = tf.data.AUTOTUNE
SHUFFLE_BUFFER = 8192

DATA_OPTIONS = tf.data.Options()
DATA_OPTIONS.experimental_deterministic = True


def make_batched_ds_from_base(base_ds, training=False):
    ds = base_ds.with_options(DATA_OPTIONS)
    if training:
        ds = ds.shuffle(
            buffer_size=min(SHUFFLE_BUFFER, n_breaths),
            seed=42,
            reshuffle_each_iteration=True,
        )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


kf = KFold(n_splits=5, shuffle=True, random_state=42)

n_features = train.shape[-1]
n_breaths = train.shape[0]

base_test_ds = (
    tf.data.Dataset.from_tensor_slices(test).with_options(DATA_OPTIONS).cache()
)
test_ds = base_test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

train_tensor = tf.convert_to_tensor(train)  # float32
y_tensor = tf.convert_to_tensor(y)  # float32

sum_test_pred = None

all_idx = np.arange(n_breaths, dtype=np.int32)
fold_indices = list(kf.split(all_idx, y))

for fold, (train_idx, valid_idx) in enumerate(fold_indices):
    print(f"****** fold: {fold+1} *******")
    t0 = time.time()

    tf.keras.backend.clear_session()

    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=35,
        verbose=0,
        restore_best_weights=True,
    )

    model = create_model(n_features)

    tr_idx_tf = tf.convert_to_tensor(train_idx, dtype=tf.int32)
    va_idx_tf = tf.convert_to_tensor(valid_idx, dtype=tf.int32)

    X_tr_tf = tf.gather(train_tensor, tr_idx_tf)
    Y_tr_tf = tf.gather(y_tensor, tr_idx_tf)
    X_va_tf = tf.gather(train_tensor, va_idx_tf)
    Y_va_tf = tf.gather(y_tensor, va_idx_tf)

    train_ds = tf.data.Dataset.from_tensor_slices((X_tr_tf, Y_tr_tf)).with_options(
        DATA_OPTIONS
    )
    train_ds = (
        train_ds.shuffle(
            buffer_size=min(SHUFFLE_BUFFER, len(train_idx)),
            seed=42,
            reshuffle_each_iteration=True,
        )
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    valid_ds = (
        tf.data.Dataset.from_tensor_slices((X_va_tf, Y_va_tf))
        .with_options(DATA_OPTIONS)
        .cache()
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=300,
        callbacks=[es],
        verbose=0,
    )

    pred = model.predict(test_ds, verbose=0)  # (n_breaths, 80, 1)
    pred = pred.squeeze(-1).reshape(-1).astype(np.float32, copy=False)  # (n_rows,)

    if sum_test_pred is None:
        sum_test_pred = pred
    else:
        sum_test_pred += pred

    plot_hist(history)

    del (
        model,
        history,
        pred,
        train_ds,
        valid_ds,
        X_tr_tf,
        Y_tr_tf,
        X_va_tf,
        Y_va_tf,
        tr_idx_tf,
        va_idx_tf,
    )
    gc.collect()
    print(f"fold time: {time.time()-t0:.1f}s")



## === cell 13
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv",
    dtype={"id": "int32", "pressure": "float32"},
)

if sum_test_pred is None:
    submission["pressure"] = 0.0
else:
    submission["pressure"] = (sum_test_pred / 5.0).astype(np.float32, copy=False)

submission.to_csv("submission_mean.csv", index=False)
print(submission.head())



## === cell 14
if "preds_mat" in locals():
    submission["pressure"] = np.median(preds_mat, axis=0).astype(np.float32, copy=False)
else:
    submission["pressure"] = submission["pressure"].to_numpy(np.float32, copy=False)
print(submission.head())



## === cell 15
pressure_unique = np.unique(y.reshape(-1).astype(np.float32, copy=False))
sub_vals = submission["pressure"].to_numpy()

idx = np.searchsorted(pressure_unique, sub_vals, side="left")
idx0 = np.clip(idx - 1, 0, len(pressure_unique) - 1)
idx1 = np.clip(idx, 0, len(pressure_unique) - 1)

cand0 = pressure_unique[idx0]
cand1 = pressure_unique[idx1]
choose1 = np.abs(cand1 - sub_vals) < np.abs(cand0 - sub_vals)
mapped = np.where(choose1, cand1, cand0)

submission["pressure"] = mapped.astype(np.float32)
submission.to_csv("submission_post_preprocessing.csv", index=False)
print(submission.head())
