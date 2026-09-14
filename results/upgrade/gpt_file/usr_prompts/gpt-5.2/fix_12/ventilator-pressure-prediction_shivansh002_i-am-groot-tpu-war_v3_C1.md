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

3.9

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

SEED = 2021
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)  # XLA
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

DEBUG = False

TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

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

train = pd.read_csv(TRAIN_PATH, dtype=train_dtypes)
test = pd.read_csv(TEST_PATH, dtype=test_dtypes)
submission = pd.read_csv(SUB_PATH, dtype={"id": "int32", "pressure": "float32"})

train.shape, test.shape, submission.shape




## === cell 1
def _ewm_mean_adjust_true_all(X: np.ndarray, halflife: float) -> np.ndarray:
    """
    Vectorized adjust=True EWM mean for shape (n_breaths, 80).
    Exactly equivalent to:
      y[i] = sum_{j<=i} x[j]*(1-a)^(i-j) / sum_{j<=i} (1-a)^(i-j)
    """
    alpha = 1.0 - np.exp(np.log(0.5) / halflife)
    beta = 1.0 - alpha  # decay
    X = X.astype(np.float64, copy=False)

    n_b, T = X.shape
    num = np.empty((n_b, T), dtype=np.float64)
    den = np.empty((n_b, T), dtype=np.float64)

    num[:, 0] = X[:, 0]
    den[:, 0] = 1.0
    for t in range(1, T):
        num[:, t] = X[:, t] + beta * num[:, t - 1]
        den[:, t] = 1.0 + beta * den[:, t - 1]
    return (num / den).astype(np.float32, copy=False)


def _ewm_std_adjust_true_all(
    X: np.ndarray, halflife: float, bias: bool = False
) -> np.ndarray:
    """
    Vectorized adjust=True EWM std for shape (n_breaths, 80), matching the original:
      - uses adjust=True weights ww = beta^(i-j)
      - m2 = sum ww*(x-mu)^2 / sum ww
      - bias=False applies: var = m2 * (w_sum^2 / (w_sum^2 - w2_sum))
      - first element is NaN
    """
    alpha = 1.0 - np.exp(np.log(0.5) / halflife)
    beta = 1.0 - alpha
    X = X.astype(np.float64, copy=False)

    n_b, T = X.shape
    w_sum = np.empty((n_b, T), dtype=np.float64)
    w2_sum = np.empty((n_b, T), dtype=np.float64)
    xw_sum = np.empty((n_b, T), dtype=np.float64)
    x2w_sum = np.empty((n_b, T), dtype=np.float64)

    w_sum[:, 0] = 1.0
    w2_sum[:, 0] = 1.0
    xw_sum[:, 0] = X[:, 0]
    x2w_sum[:, 0] = X[:, 0] * X[:, 0]

    for t in range(1, T):
        w_sum[:, t] = 1.0 + beta * w_sum[:, t - 1]
        w2_sum[:, t] = 1.0 + (beta * beta) * w2_sum[:, t - 1]
        xw_sum[:, t] = X[:, t] + beta * xw_sum[:, t - 1]
        x2w_sum[:, t] = (X[:, t] * X[:, t]) + beta * x2w_sum[:, t - 1]

    mu = xw_sum / w_sum
    m2 = (x2w_sum / w_sum) - (mu * mu)  # weighted second central moment (population)

    m2 = np.maximum(m2, 0.0)

    if bias:
        var = m2
    else:
        denom = w_sum * w_sum - w2_sum
        var = np.where(denom <= 0.0, np.nan, m2 * (w_sum * w_sum / denom))

    std = np.sqrt(var)
    std[:, 0] = np.nan  # match original
    return std.astype(np.float32, copy=False)


def _rolling15_max_std_all(
    X: np.ndarray, window: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact rolling max and sample std (ddof=1) over last <=window values at each time.
    Vectorized using sliding_window_view + prefix sums; matches the original behavior:
      - window for t is X[:, max(0,t-window):t+1]  (note: original used i-14 => length <=15)
      - std is NaN if window length < 2 else sample std with ddof=1
    """
    X = X.astype(np.float64, copy=False)
    n_b, T = X.shape
    pad = window - 1
    Xpad = np.pad(X, ((0, 0), (pad, 0)), mode="constant", constant_values=np.nan)

    from numpy.lib.stride_tricks import sliding_window_view

    W = sliding_window_view(Xpad, window_shape=window, axis=1)

    roll_max = np.nanmax(W, axis=2)

    valid = ~np.isnan(W)
    n = valid.sum(axis=2).astype(np.float64)

    s1 = np.nansum(W, axis=2)
    s2 = np.nansum(W * W, axis=2)

    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - (s1 * s1) / n) / (n - 1.0)
    var = np.where(n < 2.0, np.nan, np.maximum(var, 0.0))
    roll_std = np.sqrt(var)

    return roll_max.astype(np.float32, copy=False), roll_std.astype(
        np.float32, copy=False
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    n = len(out)
    if n % 80 != 0:
        raise ValueError(f"Expected row count multiple of 80, got {n}")

    time_step = out["time_step"].to_numpy(dtype=np.float64, copy=False).reshape(-1, 80)
    u_in = out["u_in"].to_numpy(dtype=np.float64, copy=False).reshape(-1, 80)

    area = (
        np.cumsum(time_step * u_in, axis=1).astype(np.float32, copy=False).reshape(-1)
    )
    u_in_cumsum = np.cumsum(u_in, axis=1).astype(np.float32, copy=False).reshape(-1)

    u_in_lag = np.zeros_like(u_in, dtype=np.float32)
    u_in_lag[:, 2:] = u_in[:, :-2].astype(np.float32, copy=False)
    u_in_lag = u_in_lag.reshape(-1)

    ewm_mean8 = _ewm_mean_adjust_true_all(u_in, halflife=8)
    ewm_std9 = _ewm_std_adjust_true_all(u_in, halflife=9, bias=False)
    ewm_std14 = _ewm_std_adjust_true_all(u_in, halflife=14, bias=False)

    ewm_u_in_mean = ewm_mean8.reshape(-1)
    ewm_u_in_std = ewm_std9.reshape(-1)

    s14 = ewm_std14.reshape(-1)
    corr = np.ones_like(s14, dtype=np.float32)
    corr[np.isnan(s14) | (s14 == 0)] = np.nan
    ewm_u_in_corr = corr.astype(np.float32, copy=False)

    roll_max, roll_std = _rolling15_max_std_all(u_in, window=15)

    out["area"] = area
    out["u_in_cumsum"] = u_in_cumsum
    out["u_in_lag"] = u_in_lag
    out["ewm_u_in_mean"] = ewm_u_in_mean
    out["ewm_u_in_std"] = ewm_u_in_std
    out["ewm_u_in_corr"] = ewm_u_in_corr
    out["15_in_max"] = roll_max.reshape(-1)
    out["15_in_std"] = roll_std.reshape(-1)

    out["R"] = out["R"].astype("category")
    out["C"] = out["C"].astype("category")
    out = pd.get_dummies(out, columns=["R", "C"], dtype=np.uint8)

    return out


train = add_features(train)
test = add_features(test)



## === cell 2
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

missing_in_test = [c for c in train.columns if c not in test.columns]
for c in missing_in_test:
    test[c] = np.uint8(0)
missing_in_train = [c for c in test.columns if c not in train.columns]
for c in missing_in_train:
    train[c] = np.uint8(0)

test = test[
    train.columns.drop("pressure") if "pressure" in train.columns else train.columns
]

train.shape, test.shape



## === cell 3
targets = train[["pressure"]].to_numpy().reshape(-1, 80)

train.drop(["pressure", "id", "breath_id"], axis=1, inplace=True)
test = test.drop(["id", "breath_id"], axis=1)

train.shape, test.shape, targets.shape



## === cell 4
RS = RobustScaler()
train_arr = train.to_numpy(dtype=np.float32, copy=False)
test_arr = test.to_numpy(dtype=np.float32, copy=False)

train_np = RS.fit_transform(train_arr)
test_np = RS.transform(test_arr)

n_features = train_np.shape[-1]
train_np = np.ascontiguousarray(train_np.reshape(-1, 80, n_features), dtype=np.float32)
test_np = np.ascontiguousarray(test_np.reshape(-1, 80, n_features), dtype=np.float32)

del train, test, train_arr, test_arr
gc.collect()

train_np.shape, test_np.shape



## === cell 5
EPOCH = 225
BATCH_SIZE = 1024


def get_strategy():
    """
    Keep same behavior: try TPU, else default.
    """
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        return tf.distribute.TPUStrategy(resolver)
    except Exception:
        return tf.distribute.get_strategy()


strategy = get_strategy()
print(
    "Using strategy:",
    type(strategy).__name__,
    "num_replicas:",
    strategy.num_replicas_in_sync,
)

test_preds = []

decay_steps = float(400 * ((len(train_np) * 0.8) / BATCH_SIZE))
initial_lr = 1e-3
decay_rate = 1e-5


def lr_schedule(epoch, lr):
    new_lr = initial_lr * (decay_rate ** (epoch / decay_steps))
    return float(new_lr)


def _ds_options_deterministic():
    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    return options


def make_fold_ds(X, y, idx, batch_size, cache_in_memory: bool):
    Xs = np.take(X, idx, axis=0)
    ys = np.take(y, idx, axis=0)
    ds = tf.data.Dataset.from_tensor_slices((Xs, ys))
    if cache_in_memory:
        ds = ds.cache()
    ds = ds.with_options(_ds_options_deterministic())
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def make_test_ds(X, batch_size=1024, cache_in_memory=True):
    ds = tf.data.Dataset.from_tensor_slices(X)
    if cache_in_memory:
        ds = ds.cache()
    ds = ds.with_options(_ds_options_deterministic())
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


ds_test = make_test_ds(
    test_np,
    batch_size=BATCH_SIZE,
    cache_in_memory=True,  # reused across folds
)

with strategy.scope():
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_np, targets)):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

        ds_train = make_fold_ds(
            train_np, targets, train_idx, BATCH_SIZE, cache_in_memory=True
        )
        ds_valid = make_fold_ds(
            train_np, targets, valid_idx, BATCH_SIZE, cache_in_memory=True
        )

        model = keras.models.Sequential(
            [
                keras.layers.Input(shape=train_np.shape[-2:]),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(500, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(375, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(200, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(100, return_sequences=True)
                ),
                keras.layers.Dense(100, activation="selu"),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mae", jit_compile=True)

        lr_cb = LearningRateScheduler(lr_schedule, verbose=0)

        model.fit(
            ds_train,
            validation_data=ds_valid,
            epochs=EPOCH,
            callbacks=[lr_cb],
            verbose=0,
        )

        pred = model.predict(ds_test, verbose=0).squeeze()
        test_preds.append(pred)

        del model, pred, ds_train, ds_valid, train_idx, valid_idx
        tf.keras.backend.clear_session()
        gc.collect()

print("done")



## === cell 6
if (not isinstance(test_preds, list)) or (len(test_preds) == 0):
    submission["pressure"] = 0.0
else:
    submission["pressure"] = np.mean(
        np.vstack([p.reshape(1, -1) for p in test_preds]), axis=0
    ).ravel()

assert len(submission) == 603600, f"Unexpected submission length: {len(submission)}"
assert list(submission.columns) == [
    "id",
    "pressure",
], f"Unexpected columns: {submission.columns.tolist()}"

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", os.path.getsize("submission.csv"), "bytes")
submission.head()
