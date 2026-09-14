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
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
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

    u_in_max = np.max(u_in, axis=1).astype(np.float32, copy=False)
    u_out_max = np.max(u_out, axis=1).astype(np.float32, copy=False)
    u_in_mean = np.mean(u_in, axis=1).astype(np.float32, copy=False)

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
        rmask = R == r
        for c in (10, 20, 50):
            df[f"R__C_{r}__{c}"] = (rmask & (C == c)).astype(np.int8, copy=False)

    return df


FE_CACHE_DIR = "./_fe_cache"
os.makedirs(FE_CACHE_DIR, exist_ok=True)
train_fe_npz = os.path.join(FE_CACHE_DIR, "train_fe.npz")
test_fe_npz = os.path.join(FE_CACHE_DIR, "test_fe.npz")


def _save_df_npz(path, df: pd.DataFrame):
    cols = df.columns.to_numpy()
    arr = df.to_numpy()
    dtypes = np.array([str(dt) for dt in df.dtypes], dtype=object)
    np.savez_compressed(path, cols=cols, arr=arr, dtypes=dtypes)


def _load_df_npz(path) -> pd.DataFrame:
    z = np.load(path, allow_pickle=True)
    cols = z["cols"].tolist()
    arr = z["arr"]
    dtypes = z["dtypes"].tolist()
    out = pd.DataFrame(arr, columns=cols)
    for c, dt in zip(cols, dtypes):
        try:
            if dt.startswith("int") or dt.startswith("uint"):
                out[c] = out[c].astype(np.int32 if dt == "int32" else dt)
            elif dt.startswith("float"):
                out[c] = out[c].astype(np.float32 if dt == "float32" else dt)
        except Exception:
            pass
    return out


if os.path.exists(train_fe_npz) and os.path.exists(test_fe_npz) and not DEBUG:
    train_fe = _load_df_npz(train_fe_npz)
    test_fe = _load_df_npz(test_fe_npz)
else:
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

    if not DEBUG:
        _save_df_npz(train_fe_npz, train_fe)
        _save_df_npz(test_fe_npz, test_fe)

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
    n_gpus = len(tf.config.list_logical_devices("GPU"))
    if n_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"TPU not available, using MirroredStrategy({n_gpus} GPUs). Reason: {e}")
    else:
        strategy = tf.distribute.get_strategy()
        print(f"TPU not available, using default strategy. Reason: {e}")



## === cell 7
EPOCH = 300
BATCH_SIZE = 512
NUM_FOLDS = 10

AUTO = tf.data.AUTOTUNE

X_np = train_X  # (n_breaths, 80, n_features)
y_np = targets  # (n_breaths, 80)
test_X_np = test_X  # (n_test_breaths, 80, n_features)

opts = tf.data.Options()
opts.deterministic = True
try:
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
except Exception:
    pass


def make_ds_from_numpy(
    X_part, y_part=None, training=False, batch_size=512, cache=False
):
    if y_part is None:
        ds = tf.data.Dataset.from_tensor_slices(X_part)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X_part, y_part))
    ds = ds.with_options(opts)
    if training:
        ds = ds.shuffle(
            buffer_size=min(int(len(X_part)), 8192),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


test_ds = make_ds_from_numpy(
    test_X_np, y_part=None, training=False, batch_size=BATCH_SIZE, cache=True
)

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)

n_test_steps = test_X_np.shape[0] * test_X_np.shape[1]
pred_sum = np.zeros((n_test_steps,), dtype=np.float64)

with strategy.scope():
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_np, y_np), start=1):
        print("-" * 15, ">", f"Fold {fold}", "<", "-" * 15)

        X_tr, y_tr = X_np[train_idx], y_np[train_idx]
        X_va, y_va = X_np[valid_idx], y_np[valid_idx]

        train_ds = make_ds_from_numpy(
            X_tr, y_part=y_tr, training=True, batch_size=BATCH_SIZE, cache=False
        )
        valid_ds = make_ds_from_numpy(
            X_va, y_part=y_va, training=False, batch_size=BATCH_SIZE, cache=True
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
        pred_sum += pred.reshape(-1).astype(np.float64, copy=False)

        keras.backend.clear_session()

test_pred_mean = (pred_sum / float(NUM_FOLDS)).astype(np.float32, copy=False)

assert len(test_pred_mean) == len(submission), (len(test_pred_mean), len(submission))

submission["pressure"] = test_pred_mean
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("File exists:", os.path.exists("submission.csv"))
