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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold



## === cell 1
SEED = 2021
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF choose
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## === cell 2
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

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

train = pd.read_csv(TRAIN_PATH, dtype=TRAIN_DTYPES)
test = pd.read_csv(TEST_PATH, dtype=TEST_DTYPES)
submission = pd.read_csv(SUB_PATH, dtype={"id": "int32", "pressure": "float32"})

DEBUG = False
if DEBUG:
    n_breaths = 1000
    keep_breath_ids = train["breath_id"].unique()[:n_breaths]
    train = train[train["breath_id"].isin(keep_breath_ids)].reset_index(drop=True)

train.shape, test.shape, submission.shape




## === cell 3
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    n = len(df)
    breath_id = df["breath_id"].to_numpy()
    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)
    u_out = df["u_out"].to_numpy(dtype=np.float32, copy=False)
    t = df["time_step"].to_numpy(dtype=np.float32, copy=False)

    start = np.empty(n, dtype=bool)
    start[0] = True
    start[1:] = breath_id[1:] != breath_id[:-1]
    end = np.empty(n, dtype=bool)
    end[:-1] = start[1:]
    end[-1] = True

    grp_idx = np.arange(n, dtype=np.int32) - np.maximum.accumulate(
        np.where(start, np.arange(n, dtype=np.int32), 0)
    )

    area_step = (t * u_in).astype(np.float32, copy=False)
    area_cum = np.cumsum(area_step, dtype=np.float32)
    uin_cum = np.cumsum(u_in, dtype=np.float32)
    prev_area = np.zeros(n, dtype=np.float32)
    prev_uin = np.zeros(n, dtype=np.float32)
    if n > 1:
        prev_area[1:] = np.where(start[1:], area_cum[:-1], 0.0)
        prev_uin[1:] = np.where(start[1:], uin_cum[:-1], 0.0)
    df["area"] = (area_cum - np.maximum.accumulate(prev_area)).astype(np.float32)
    df["u_in_cumsum"] = (uin_cum - np.maximum.accumulate(prev_uin)).astype(np.float32)

    for k in (1, 2, 3, 4):
        uin_lag = np.roll(u_in, k)
        uout_lag = np.roll(u_out, k)
        invalid = grp_idx < k
        uin_lag[invalid] = 0.0
        uout_lag[invalid] = 0.0

        uin_lead = np.roll(u_in, -k)
        uout_lead = np.roll(u_out, -k)
        invalid_fwd = grp_idx > (79 - k)  # breath length is 80
        uin_lead[invalid_fwd] = 0.0
        uout_lead[invalid_fwd] = 0.0

        df[f"u_in_lag{k}"] = uin_lag.astype(np.float32, copy=False)
        df[f"u_out_lag{k}"] = uout_lag.astype(np.float32, copy=False)
        df[f"u_in_lag_back{k}"] = uin_lead.astype(np.float32, copy=False)
        df[f"u_out_lag_back{k}"] = uout_lead.astype(np.float32, copy=False)

    order = np.argsort(breath_id, kind="mergesort")
    bid_sorted = breath_id[order]
    uin_sorted = u_in[order]
    uout_sorted = u_out[order]

    gb_start = np.r_[0, np.flatnonzero(bid_sorted[1:] != bid_sorted[:-1]) + 1]
    counts = np.diff(np.r_[gb_start, len(bid_sorted)]).astype(np.int32)

    uin_max = np.maximum.reduceat(uin_sorted, gb_start).astype(np.float32)
    uout_max = np.maximum.reduceat(uout_sorted, gb_start).astype(np.float32)
    uin_sum = np.add.reduceat(uin_sorted, gb_start).astype(np.float32)
    uin_mean = (uin_sum / counts).astype(np.float32)

    group_ids_sorted = np.repeat(np.arange(len(gb_start), dtype=np.int32), counts)
    group_ids = np.empty(n, dtype=np.int32)
    group_ids[order] = group_ids_sorted

    df["breath_id__u_in__max"] = uin_max[group_ids]
    df["breath_id__u_out__max"] = uout_max[group_ids]

    df["u_in_diff1"] = (u_in - df["u_in_lag1"].to_numpy(np.float32, copy=False)).astype(
        np.float32
    )
    df["u_out_diff1"] = (
        u_out - df["u_out_lag1"].to_numpy(np.float32, copy=False)
    ).astype(np.float32)
    df["u_in_diff2"] = (u_in - df["u_in_lag2"].to_numpy(np.float32, copy=False)).astype(
        np.float32
    )
    df["u_out_diff2"] = (
        u_out - df["u_out_lag2"].to_numpy(np.float32, copy=False)
    ).astype(np.float32)

    df["breath_id__u_in__diffmax"] = (uin_max[group_ids] - u_in).astype(np.float32)
    df["breath_id__u_in__diffmean"] = (uin_mean[group_ids] - u_in).astype(np.float32)

    df["u_in_diff3"] = (u_in - df["u_in_lag3"].to_numpy(np.float32, copy=False)).astype(
        np.float32
    )
    df["u_out_diff3"] = (
        u_out - df["u_out_lag3"].to_numpy(np.float32, copy=False)
    ).astype(np.float32)
    df["u_in_diff4"] = (u_in - df["u_in_lag4"].to_numpy(np.float32, copy=False)).astype(
        np.float32
    )
    df["u_out_diff4"] = (
        u_out - df["u_out_lag4"].to_numpy(np.float32, copy=False)
    ).astype(np.float32)

    df["cross"] = (u_in * u_out).astype(np.float32)
    df["cross2"] = (t * u_out).astype(np.float32)

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["R__C"] = df["R"] + "__" + df["C"]

    df = pd.get_dummies(df)
    return df




## === cell 4
train_fe = add_features(train)
test_fe = add_features(test)

target_col = "pressure"
drop_cols_train = [target_col, "id", "breath_id"]
drop_cols_test = ["id", "breath_id"]

y = train_fe[[target_col]].to_numpy().reshape(-1, 80)

X_train_df = train_fe.drop(drop_cols_train, axis=1)
X_test_df = test_fe.drop(drop_cols_test, axis=1)

X_train_df, X_test_df = X_train_df.align(X_test_df, join="left", axis=1, fill_value=0)

X_train_df.shape, X_test_df.shape, y.shape



## === cell 5
RS = RobustScaler()
X_train_scaled = RS.fit_transform(X_train_df).astype(np.float32, copy=False)
X_test_scaled = RS.transform(X_test_df).astype(np.float32, copy=False)

n_features = X_train_scaled.shape[-1]
X_train_seq = np.ascontiguousarray(X_train_scaled.reshape(-1, 80, n_features))
X_test_seq = np.ascontiguousarray(X_test_scaled.reshape(-1, 80, n_features))

X_train_seq.shape, X_test_seq.shape



## === cell 6
try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
    USING_TPU = True
except Exception:
    strategy = tf.distribute.get_strategy()
    USING_TPU = False

print("Using TPU:", USING_TPU)
print("Num replicas:", strategy.num_replicas_in_sync)



## === cell 7
EPOCH = 300
BATCH_SIZE = 1024
NUM_FOLDS = 10

test_preds = []

AUTOTUNE = tf.data.AUTOTUNE


def make_ds(X, y=None, training=False):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(X)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y))

    opts = tf.data.Options()
    opts.deterministic = True
    ds = ds.with_options(opts)

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(X), 32768), seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_ds(X_test_seq, y=None, training=False)

with strategy.scope():
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_seq, y)):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

        X_tr, X_va = X_train_seq[tr_idx], X_train_seq[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        train_ds = make_ds(X_tr, y_tr, training=True)
        val_ds = make_ds(X_va, y_va, training=False)

        model = keras.models.Sequential(
            [
                keras.layers.Input(shape=X_train_seq.shape[-2:]),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(1024, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(512, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(256, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(128, return_sequences=True)
                ),
                keras.layers.Dense(128, activation="selu"),
                keras.layers.Dense(1),
            ]
        )

        model.compile(optimizer="adam", loss="mae", steps_per_execution=32)

        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
        es = EarlyStopping(
            monitor="val_loss",
            patience=60,
            verbose=1,
            mode="min",
            restore_best_weights=True,
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCH,
            callbacks=[lr, es],
            verbose=2,
            validation_freq=5,
        )

        pred = model.predict(test_ds, verbose=0)
        pred = pred.reshape(-1)  # (n_test_rows,)
        test_preds.append(pred)



## === cell 8
test_pred_mean = np.mean(np.vstack(test_preds), axis=0)  # (n_test_rows,)

if len(test_pred_mean) != len(submission):
    raise ValueError(
        f"Prediction length {len(test_pred_mean)} != submission length {len(submission)}"
    )

submission["pressure"] = test_pred_mean.astype(np.float32)
submission[["id", "pressure"]].to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv saved to:", os.path.abspath("submission.csv"))
