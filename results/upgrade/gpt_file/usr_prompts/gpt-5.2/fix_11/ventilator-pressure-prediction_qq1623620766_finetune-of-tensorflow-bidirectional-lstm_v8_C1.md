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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

print("TF version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)



## === cell 1
from numpy.random import seed

SEED = 2021
seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## === cell 2
DEBUG = False

DATA_DIR = "../input/ventilator-pressure-prediction"

train = pd.read_csv(
    f"{DATA_DIR}/train.csv",
    dtype={
        "id": np.int32,
        "breath_id": np.int32,
        "R": np.int16,
        "C": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
        "pressure": np.float32,
    },
)
test = pd.read_csv(
    f"{DATA_DIR}/test.csv",
    dtype={
        "id": np.int32,
        "breath_id": np.int32,
        "R": np.int16,
        "C": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
    },
)
submission = pd.read_csv(
    f"{DATA_DIR}/sample_submission.csv", dtype={"id": np.int32, "pressure": np.float32}
)

if DEBUG:
    train = train.iloc[: 80 * 1000].copy()

print(train.shape, test.shape, submission.shape)
print(train.head())




## === cell 3
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    n = len(df)
    if n % 80 != 0:
        raise ValueError(f"Expected rows multiple of 80, got {n}")
    n_breaths = n // 80

    time_step = (
        df["time_step"]
        .to_numpy(copy=False)
        .reshape(n_breaths, 80)
        .astype(np.float32, copy=False)
    )
    u_in = (
        df["u_in"]
        .to_numpy(copy=False)
        .reshape(n_breaths, 80)
        .astype(np.float32, copy=False)
    )
    u_out = (
        df["u_out"]
        .to_numpy(copy=False)
        .reshape(n_breaths, 80)
        .astype(np.float32, copy=False)
    )

    df["area"] = np.cumsum(time_step * u_in, axis=1).reshape(-1)
    df["u_in_cumsum"] = np.cumsum(u_in, axis=1).reshape(-1)

    def lag_mat(x, k: int):
        out = np.zeros_like(x)
        out[:, k:] = x[:, :-k]
        return out

    def lead_mat(x, k: int):
        out = np.zeros_like(x)
        out[:, :-k] = x[:, k:]
        return out

    u_in_lags = {}
    u_out_lags = {}
    for lag in (1, 2, 3, 4):
        u_in_lags[lag] = lag_mat(u_in, lag)
        u_out_lags[lag] = lag_mat(u_out, lag)
        df[f"u_in_lag{lag}"] = u_in_lags[lag].reshape(-1)
        df[f"u_out_lag{lag}"] = u_out_lags[lag].reshape(-1)
        df[f"u_in_lag_back{lag}"] = lead_mat(u_in, lag).reshape(-1)
        df[f"u_out_lag_back{lag}"] = lead_mat(u_out, lag).reshape(-1)

    u_in_max = np.max(u_in, axis=1).astype(np.float32, copy=False)  # (n_breaths,)
    u_out_max = np.max(u_out, axis=1).astype(np.float32, copy=False)  # (n_breaths,)
    u_in_mean = np.mean(u_in, axis=1).astype(np.float32, copy=False)  # (n_breaths,)

    u_in_max_2d = u_in_max[:, None]
    u_in_mean_2d = u_in_mean[:, None]
    u_out_max_2d = u_out_max[:, None]

    df["breath_id__u_in__max"] = np.broadcast_to(u_in_max_2d, (n_breaths, 80)).reshape(
        -1
    )
    df["breath_id__u_out__max"] = np.broadcast_to(
        u_out_max_2d, (n_breaths, 80)
    ).reshape(-1)

    df["u_in_diff1"] = (u_in - u_in_lags[1]).reshape(-1)
    df["u_out_diff1"] = (u_out - u_out_lags[1]).reshape(-1)
    df["u_in_diff2"] = (u_in - u_in_lags[2]).reshape(-1)
    df["u_out_diff2"] = (u_out - u_out_lags[2]).reshape(-1)
    df["u_in_diff3"] = (u_in - u_in_lags[3]).reshape(-1)
    df["u_out_diff3"] = (u_out - u_out_lags[3]).reshape(-1)
    df["u_in_diff4"] = (u_in - u_in_lags[4]).reshape(-1)
    df["u_out_diff4"] = (u_out - u_out_lags[4]).reshape(-1)

    df["breath_id__u_in__diffmax"] = (
        np.broadcast_to(u_in_max_2d, (n_breaths, 80)) - u_in
    ).reshape(-1)
    df["breath_id__u_in__diffmean"] = (
        np.broadcast_to(u_in_mean_2d, (n_breaths, 80)) - u_in
    ).reshape(-1)

    df["cross"] = (u_in * u_out).reshape(-1)
    df["cross2"] = (time_step * u_out).reshape(-1)

    R = df["R"].to_numpy(copy=False)
    C = df["C"].to_numpy(copy=False)

    R_str = R.astype("U3", copy=False)
    C_str = C.astype("U3", copy=False)
    df["R"] = R_str
    df["C"] = C_str
    df["R__C"] = np.char.add(np.char.add(R_str, "__"), C_str)

    for r in (5, 20, 50):
        df[f"R_{r}"] = (R == r).astype(np.int8, copy=False)
    for c in (10, 20, 50):
        df[f"C_{c}"] = (C == c).astype(np.int8, copy=False)
    for r in (5, 20, 50):
        for c in (10, 20, 50):
            df[f"R__C_{r}__{c}"] = ((R == r) & (C == c)).astype(np.int8, copy=False)

    return df


train_fe = add_features(train)
test_fe = add_features(test)

missing_in_test = set(train_fe.columns) - set(test_fe.columns)
for c in missing_in_test:
    if c != "pressure":
        test_fe[c] = 0
extra_in_test = set(test_fe.columns) - set(train_fe.columns)
if extra_in_test:
    test_fe = test_fe.drop(columns=list(extra_in_test))

test_fe = test_fe[[c for c in train_fe.columns if c != "pressure"]]

print(train_fe.shape, test_fe.shape)



## === cell 4
targets = train_fe[["pressure"]].to_numpy().reshape(-1, 80)

train_X = train_fe.drop(["pressure", "id", "breath_id"], axis=1)
test_X = test_fe.drop(["id", "breath_id"], axis=1)

for col in ["R", "C", "R__C"]:
    if col in train_X.columns:
        train_X = train_X.drop(columns=[col])
    if col in test_X.columns:
        test_X = test_X.drop(columns=[col])

print(train_X.shape, test_X.shape, targets.shape)
print(
    "Non-numeric columns remaining (should be none):",
    [c for c in train_X.columns if not pd.api.types.is_numeric_dtype(train_X[c])],
)



## === cell 5
RS = RobustScaler()

train_X = RS.fit_transform(train_X).astype(np.float32, copy=False)
test_X = RS.transform(test_X).astype(np.float32, copy=False)

n_features = train_X.shape[-1]
train_X = np.ascontiguousarray(train_X.reshape(-1, 80, n_features), dtype=np.float32)
test_X = np.ascontiguousarray(test_X.reshape(-1, 80, n_features), dtype=np.float32)
targets = np.ascontiguousarray(targets.astype(np.float32, copy=False))

print(train_X.shape, test_X.shape, targets.shape, train_X.dtype, targets.dtype)



## === cell 6
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Using TPU strategy")
except Exception as e:
    strategy = tf.distribute.MirroredStrategy()
    print(f"TPU not available, using {strategy.__class__.__name__}. Reason: {e}")



## === cell 7
EPOCH = 300
BATCH_SIZE = 1024
NUM_FOLDS = 10

AUTO = tf.data.AUTOTUNE

X_np = train_X  # (n_breaths, 80, n_features)
y_np = targets  # (n_breaths, 80)
test_X_np = test_X  # (n_test_breaths, 80, n_features)

opts_det = tf.data.Options()
opts_det.deterministic = True


def make_fold_ds_from_indices_np(
    X_all_np,
    y_all_np,
    indices_np,
    training=False,
    batch_size=1024,
):
    X_slice = X_all_np[indices_np]
    y_slice = y_all_np[indices_np]
    ds = tf.data.Dataset.from_tensor_slices((X_slice, y_slice))
    if training:
        ds = ds.shuffle(
            buffer_size=min(int(indices_np.shape[0]), 8192),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
    ds = ds.with_options(opts_det)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


def make_test_ds_np(X_all_np, batch_size=1024):
    ds = tf.data.Dataset.from_tensor_slices(X_all_np)
    ds = ds.with_options(opts_det)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


test_ds = make_test_ds_np(test_X_np, batch_size=BATCH_SIZE)

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)

test_preds = []
with strategy.scope():
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_np, y_np), start=1):
        print("-" * 15, ">", f"Fold {fold}", "<", "-" * 15)

        train_ds = make_fold_ds_from_indices_np(
            X_np, y_np, train_idx, training=True, batch_size=BATCH_SIZE
        )
        valid_ds = make_fold_ds_from_indices_np(
            X_np, y_np, valid_idx, training=False, batch_size=BATCH_SIZE
        )

        model = keras.models.Sequential(
            [
                keras.layers.Input(shape=X_np.shape[-2:]),
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
        model.compile(optimizer="adam", loss="mae")

        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=0)
        es = EarlyStopping(
            monitor="val_loss",
            patience=60,
            verbose=0,
            mode="min",
            restore_best_weights=True,
        )

        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCH,
            callbacks=[lr, es],
            verbose=0,
        )

        pred = model.predict(test_ds, verbose=0)  # (n_test_breaths, 80, 1)
        test_preds.append(pred.reshape(-1))

        keras.backend.clear_session()



## === cell 8
if not test_preds:
    raise RuntimeError(
        "No test predictions were generated; training may have failed earlier."
    )

test_pred_mean = np.mean(np.vstack(test_preds), axis=0)

assert len(test_pred_mean) == len(submission), (len(test_pred_mean), len(submission))

submission["pressure"] = test_pred_mean.astype(np.float32)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("File exists:", os.path.exists("submission.csv"))
