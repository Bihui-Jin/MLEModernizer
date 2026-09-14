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

catboost==1.2.8
geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
xgboost==2.0.3

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
os.environ.setdefault("TF_DISABLE_PROFILER", "1")
os.environ.setdefault("TF_PROFILER_DISABLE", "1")

os.environ.setdefault("PYTHONHASHSEED", "228")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers.schedules import ExponentialDecay

pd.set_option("display.max_columns", None)

np.random.seed(228)
tf.keras.utils.set_random_seed(228)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

DATA_DIR = "../input/ventilator-pressure-prediction"
if not os.path.exists(os.path.join(DATA_DIR, "train.csv")):
    DATA_DIR = "../input"

train = pd.read_csv(
    f"{DATA_DIR}/train.csv",
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
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
test = pd.read_csv(
    f"{DATA_DIR}/test.csv",
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
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
ss = pd.read_csv(
    f"{DATA_DIR}/sample_submission.csv", dtype={"id": "int32", "pressure": "float32"}
)



## === cell 1
print(f"Length of TRAIN dataset: {len(train)}")
print(f"Length of TEST dataset: {len(test)}\n")

print("Missing values in TRAIN dataset")
print(train.iloc[:, 0:-1].isna().sum().to_string())
print("\nMissing values in TEST dataset")
print(test.isna().sum().to_string())
print("")
print(f'Number of breaths in train dataset: {train["breath_id"].nunique()}')
print(f'Number of breaths in test dataset: {test["breath_id"].nunique()}')
obs_per_breath = train["breath_id"].value_counts().iat[0]
print(f"The number of observations for each breath: {obs_per_breath}")



## === cell 2
pass



## === cell 3
pass




## === cell 4
def _rolling_max_window(u: np.ndarray, w: int) -> np.ndarray:
    """u: (n, T) -> (n, T) rolling max with expanding window for first w-1 steps."""
    u_pad = np.pad(u, ((0, 0), (w - 1, 0)), mode="constant", constant_values=-np.inf)
    win = np.lib.stride_tricks.sliding_window_view(u_pad, window_shape=w, axis=1)
    return win.max(axis=-1)


def _ewm_mean_var_halflife(
    u: np.ndarray, halflife: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact equivalent of:
      S0 = alpha + (1-alpha)*S0
      S1 = alpha*x + (1-alpha)*S1
      S2 = alpha*x^2 + (1-alpha)*S2
      mean = S1/S0 ; var = S2/S0 - mean^2
    with alpha defined from halflife.
    """
    alpha = 1.0 - np.exp(np.log(0.5) / float(halflife))
    one_ma = 1.0 - alpha
    n, T = u.shape

    p = one_ma ** np.arange(T, dtype=np.float64)  # (T,)
    u_div = u / p[None, :]
    u2_div = (u * u) / p[None, :]

    c1 = np.cumsum(u_div, axis=1)
    c2 = np.cumsum(u2_div, axis=1)

    S1 = alpha * p[None, :] * c1
    S2 = alpha * p[None, :] * c2

    S0 = 1.0 - (one_ma ** (np.arange(T, dtype=np.float64) + 1.0))
    mean = S1 / S0[None, :]
    var = np.maximum(S2 / S0[None, :] - mean * mean, 0.0)
    return mean, var


def _per_breath_feature_block(u_in_2d: np.ndarray, time_step_2d: np.ndarray):
    """
    u_in_2d: (n_breaths, 80)
    time_step_2d: (n_breaths, 80)
    returns dict of feature arrays (n_breaths, 80) float32
    """
    u = u_in_2d.astype(np.float64, copy=False)
    t = time_step_2d.astype(np.float64, copy=False)
    n, T = u.shape

    u_cum = np.cumsum(u, axis=1)
    area = np.cumsum(t * u, axis=1)

    u_lag2 = np.zeros_like(u)
    u_lag4 = np.zeros_like(u)
    u_lag2[:, 2:] = u[:, :-2]
    u_lag4[:, 4:] = u[:, :-4]

    w = 10
    csum = np.cumsum(u, axis=1)
    csum2 = np.cumsum(u * u, axis=1)

    csum_pad = np.concatenate([np.zeros((n, 1), dtype=np.float64), csum], axis=1)
    csum2_pad = np.concatenate([np.zeros((n, 1), dtype=np.float64), csum2], axis=1)

    idx = np.arange(T, dtype=np.int32)
    left = np.maximum(idx + 1 - w, 0)
    right = idx + 1
    win_len = (right - left).astype(np.float64)

    sum_w = csum_pad[:, right] - csum_pad[:, left]
    sum2_w = csum2_pad[:, right] - csum2_pad[:, left]

    roll_mean = sum_w / win_len[None, :]
    roll_max = _rolling_max_window(u, w)

    denom = win_len - 1.0
    var_w = sum2_w - (sum_w * sum_w) / win_len[None, :]
    roll_std = np.full_like(u, np.nan, dtype=np.float64)
    ok = denom > 0
    roll_std[:, ok] = np.sqrt(var_w[:, ok] / denom[ok][None, :])

    exp_mean = u_cum / (idx[None, :] + 1.0)
    exp_max = np.maximum.accumulate(u, axis=1)
    denom_e = idx.astype(np.float64)
    sum_e = csum
    sum2_e = csum2
    var_e = sum2_e - (sum_e * sum_e) / (idx[None, :] + 1.0)
    exp_std = np.full_like(u, np.nan, dtype=np.float64)
    ok_e = denom_e > 0
    exp_std[:, ok_e] = np.sqrt(var_e[:, ok_e] / denom_e[ok_e][None, :])

    halflife = 10.0
    ewm_mean, ewm_var = _ewm_mean_var_halflife(u, halflife=halflife)
    ewm_std = np.sqrt(ewm_var)

    ewm_corr = np.full_like(u, np.nan, dtype=np.float64)
    ewm_corr[ewm_std > 0] = 1.0

    out = {
        "area": area.astype(np.float32),
        "u_in_cumsum": u_cum.astype(np.float32),
        "u_in_lag2": u_lag2.astype(np.float32),
        "u_in_lag4": u_lag4.astype(np.float32),
        "ewm_u_in_mean": ewm_mean.astype(np.float32),
        "ewm_u_in_std": ewm_std.astype(np.float32),
        "ewm_u_in_corr": ewm_corr.astype(np.float32),
        "rolling_10_mean": roll_mean.astype(np.float32),
        "rolling_10_max": roll_max.astype(np.float32),
        "rolling_10_std": roll_std.astype(np.float32),
        "expand_mean": exp_mean.astype(np.float32),
        "expand_max": exp_max.astype(np.float32),
        "expand_std": exp_std.astype(np.float32),
    }
    return out


def features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)
    t = df["time_step"].to_numpy(dtype=np.float32, copy=False)

    n_rows = len(df)
    if n_rows % 80 != 0:
        raise ValueError(
            f"Row count {n_rows} is not divisible by 80; cannot reshape into breaths."
        )
    n_breaths = n_rows // 80

    u2 = u_in.reshape(n_breaths, 80)
    t2 = t.reshape(n_breaths, 80)

    feat = _per_breath_feature_block(u2, t2)
    for k, v in feat.items():
        df[k] = v.reshape(n_rows)

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    return df


train_feat = features(train)
test_feat = features(test)

train_feat = pd.get_dummies(train_feat, columns=["R", "C"])
test_feat = pd.get_dummies(test_feat, columns=["R", "C"])

for d in (train_feat, test_feat):
    for col in d.columns:
        if d[col].dtype == bool:
            d[col] = d[col].astype(np.int8)

feature_cols = [c for c in train_feat.columns if c not in ("pressure",)]
for c in feature_cols:
    if c not in test_feat.columns:
        test_feat[c] = 0

test_feat = test_feat[feature_cols]
train_feat = train_feat[feature_cols + ["pressure"]]

train = train_feat
test = test_feat
del train_feat, test_feat



## === cell 5
train = train.fillna(0)
test = test.fillna(0)



## === cell 6
targets = train[["pressure"]].to_numpy().reshape(-1, 80).astype(np.float32, copy=False)

train.drop(["pressure", "id", "breath_id"], axis=1, inplace=True, errors="ignore")
test_ids = test["id"].to_numpy() if "id" in test.columns else ss["id"].to_numpy()
test.drop(["id", "breath_id"], axis=1, inplace=True, errors="ignore")



## === cell 7
RS = RobustScaler()

train = RS.fit_transform(train).astype(np.float32, copy=False)
test = RS.transform(test).astype(np.float32, copy=False)



## === cell 8
n_features = train.shape[-1]
train = np.ascontiguousarray(train.reshape(-1, 80, n_features))
test = np.ascontiguousarray(test.reshape(-1, 80, n_features))
targets = np.ascontiguousarray(targets)

print(
    "Train shape:",
    train.shape,
    "Targets shape:",
    targets.shape,
    "Test shape:",
    test.shape,
)



## === cell 9
EPOCH = 300
BATCH_SIZE = 1024

try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
    print("Using TPU strategy.")
except Exception:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy on {len(gpus)} GPUs.")
    else:
        strategy = tf.distribute.get_strategy()
        print("Using default strategy (CPU or single GPU).")

AUTOTUNE = tf.data.AUTOTUNE

data_opts = tf.data.Options()
data_opts.experimental_deterministic = True
data_opts.experimental_optimization.apply_default_optimizations = True

train_tf = tf.constant(train)  # (n_breaths, 80, n_features)
targets_tf = tf.constant(targets)  # (n_breaths, 80)
test_tf = tf.constant(test)


def _make_indexed_ds(idxs, training: bool):
    idxs = np.asarray(idxs, dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices(idxs).with_options(data_opts)
    if training:
        shuffle_buf = min(200_000, len(idxs))
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=228, reshuffle_each_iteration=True
        )
    ds = ds.map(
        lambda i: (tf.gather(train_tf, i), tf.gather(targets_tf, i)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


test_ds = (
    tf.data.Dataset.from_tensor_slices(test_tf)
    .with_options(data_opts)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)


def build_model(input_shape):
    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Bidirectional(keras.layers.LSTM(400, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(300, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(200, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True)),
            keras.layers.Dense(50, activation="selu"),
            keras.layers.Dense(1),
        ]
    )
    return model


with strategy.scope():
    kf = KFold(n_splits=5, shuffle=True, random_state=228)
    test_preds = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train, targets)):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

        train_ds = _make_indexed_ds(train_idx, training=True)
        valid_ds = _make_indexed_ds(valid_idx, training=False)

        model = build_model(train.shape[-2:])

        optimizer = keras.optimizers.Adam()
        loss_fn = keras.losses.MeanAbsoluteError(
            reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )

        train_mae = keras.metrics.Mean(name="train_mae")
        val_mae = keras.metrics.Mean(name="val_mae")

        steps_per_epoch = int(np.ceil(len(train_idx) / BATCH_SIZE))
        decay_schedule = ExponentialDecay(
            1e-3, decay_steps=400 * max(1, steps_per_epoch), decay_rate=1e-5
        )

        @tf.function(jit_compile=False)
        def train_step(x, y, lr):
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss = loss_fn(y, y_pred)
            optimizer.learning_rate = lr
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_mae.update_state(loss)
            return loss

        @tf.function(jit_compile=False)
        def valid_step(x, y):
            y_pred = model(x, training=False)
            loss = loss_fn(y, y_pred)
            val_mae.update_state(loss)
            return loss

        for epoch in range(EPOCH):
            lr = tf.cast(
                decay_schedule(tf.cast(epoch * steps_per_epoch, tf.float32)), tf.float32
            )

            train_mae.reset_state()
            val_mae.reset_state()

            for xb, yb in train_ds:
                _ = train_step(xb, yb, lr)

            for xb, yb in valid_ds:
                _ = valid_step(xb, yb)

            print(
                f"Epoch {epoch+1}/{EPOCH} - "
                f"loss: {train_mae.result().numpy():.6f} - "
                f"val_loss: {val_mae.result().numpy():.6f}"
            )

        preds = model.predict(test_ds, verbose=0).reshape(-1)
        test_preds.append(preds)



## === cell 10
if len(test_preds) == 0:
    raise RuntimeError(
        "No fold predictions were produced; training likely failed earlier."
    )

pred = sum(test_preds) / len(test_preds)
pred = pred.reshape(-1)

if len(pred) != len(ss):
    raise ValueError(
        f"Prediction length {len(pred)} does not match submission length {len(ss)}"
    )

ss = ss.copy()
ss["pressure"] = pred.astype(np.float32)
ss.to_csv("submission.csv", index=False)

print("Wrote submission to submission.csv with shape:", ss.shape)
print(ss.head())
