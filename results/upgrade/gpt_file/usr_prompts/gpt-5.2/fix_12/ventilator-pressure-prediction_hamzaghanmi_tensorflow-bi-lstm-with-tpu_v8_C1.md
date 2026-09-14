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
import os, gc, time, logging

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## === cell 1
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

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
submission = pd.read_csv(SAMPLE_SUB_PATH, dtype={"id": "int32", "pressure": "float32"})

train.head(), test.head(), submission.head()



## === cell 2
assert "pressure" in train.columns
assert "pressure" not in test.columns
assert set(submission.columns) == {"id", "pressure"}

test_ids = test["id"].values



## === cell 3
train = train.drop(columns=["id"])
test = test.drop(columns=["id"])




## === cell 4
def add_features_inplace(df: pd.DataFrame) -> None:
    seq_len = 80
    n = len(df)
    assert n % seq_len == 0, "Row count must be divisible by 80"
    n_breaths = n // seq_len

    Rf = df["R"].to_numpy(dtype=np.float32, copy=False)
    Cf = df["C"].to_numpy(dtype=np.float32, copy=False)
    df["RC_sum"] = (Rf + Cf).astype(np.float32, copy=False)
    df["RC_div"] = (Rf / Cf).astype(np.float32, copy=False)

    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False).reshape(n_breaths, seq_len)
    u_out = (
        df["u_out"].to_numpy(dtype=np.float32, copy=False).reshape(n_breaths, seq_len)
    )
    t = (
        df["time_step"]
        .to_numpy(dtype=np.float32, copy=False)
        .reshape(n_breaths, seq_len)
    )

    df["u_in_cumsum"] = np.cumsum(u_in, axis=1, dtype=np.float32).reshape(-1)

    time_lag1 = np.zeros_like(t, dtype=np.float32)
    time_lag1[:, 1:] = t[:, :-1]
    df["time_lag1"] = time_lag1.reshape(-1)

    u_in_lag1 = np.zeros_like(u_in, dtype=np.float32)
    u_in_lag1[:, 1:] = u_in[:, :-1]
    df["u_in_lag1"] = u_in_lag1.reshape(-1)

    u_out_lag1 = np.zeros_like(u_out, dtype=np.float32)
    u_out_lag1[:, 1:] = u_out[:, :-1]
    df["u_out_lag1"] = u_out_lag1.reshape(-1)

    time_lag2 = np.zeros_like(t, dtype=np.float32)
    time_lag2[:, 2:] = t[:, :-2]
    df["time_lag2"] = time_lag2.reshape(-1)


add_features_inplace(train)
add_features_inplace(test)

y = train["pressure"].to_numpy(dtype=np.float32, copy=False)
train.drop(columns=["pressure"], inplace=True)

R_categories = [5, 20, 50]
C_categories = [10, 20, 50]
R_cat = pd.api.types.CategoricalDtype(categories=R_categories, ordered=False)
C_cat = pd.api.types.CategoricalDtype(categories=C_categories, ordered=False)
for _df in (train, test):
    _df["R"] = _df["R"].astype(R_cat)
    _df["C"] = _df["C"].astype(C_cat)

train = pd.get_dummies(train, columns=["R", "C"])
test = pd.get_dummies(test, columns=["R", "C"])

test = test.reindex(columns=train.columns, fill_value=0)

train.drop(columns=["breath_id"], inplace=True)
test.drop(columns=["breath_id"], inplace=True)



## === cell 5
rb = RobustScaler()
train_values = train.to_numpy(dtype=np.float32, copy=False)
test_values = test.to_numpy(dtype=np.float32, copy=False)

rb.fit(train_values)

train2 = rb.transform(train_values).astype(np.float32, copy=False)
test2 = rb.transform(test_values).astype(np.float32, copy=False)

del train, test, train_values, test_values
gc.collect()



## === cell 6
SEQ_LEN = 80
n_features = train2.shape[1]

assert train2.shape[0] % SEQ_LEN == 0, "Train rows not divisible by 80"
assert test2.shape[0] % SEQ_LEN == 0, "Test rows not divisible by 80"
assert y.shape[0] == train2.shape[0], "Target length mismatch"

n_breaths_train = train2.shape[0] // SEQ_LEN
n_breaths_test = test2.shape[0] // SEQ_LEN

train3 = np.ascontiguousarray(
    train2.reshape(n_breaths_train, SEQ_LEN, n_features), dtype=np.float32
)
test3 = np.ascontiguousarray(
    test2.reshape(n_breaths_test, SEQ_LEN, n_features), dtype=np.float32
)
y = np.ascontiguousarray(y.reshape(n_breaths_train, SEQ_LEN), dtype=np.float32)

del train2, test2, rb
gc.collect()

(train3.shape, y.shape, test3.shape, n_features)



## === cell 7
tf.get_logger().setLevel(logging.ERROR)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # will fail on non-TPU
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except Exception:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled (GPU)")
        except Exception:
            print("Mixed precision not enabled (API/version constraint)")
    strategy = (
        tf.distribute.MirroredStrategy()
        if len(gpus) > 1
        else tf.distribute.get_strategy()
    )
    print("Running on", "GPU" if len(gpus) > 0 else "CPU")

print("TensorFlow:", tf.__version__)
print("REPLICAS:", strategy.num_replicas_in_sync)




## === cell 8
def plot_hist(hist):
    plt.figure(figsize=(8, 4))
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.title("model performance")
    plt.ylabel("MAE")
    plt.xlabel("epoch")
    plt.legend(loc="upper left")
    plt.show()




## === cell 9
def create_model():
    with strategy.scope():
        model = Sequential(
            [
                Input(shape=(SEQ_LEN, n_features)),
                Bidirectional(LSTM(300, return_sequences=True)),
                Bidirectional(LSTM(256, return_sequences=True)),
                Bidirectional(LSTM(200, return_sequences=True)),
                Bidirectional(LSTM(128, return_sequences=True)),
                Dense(128, activation="selu"),
                Dropout(0.2),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mae")
    return model




## === cell 10
_DATA_OPTIONS = tf.data.Options()
_DATA_OPTIONS.experimental_deterministic = True
try:
    _DATA_OPTIONS.experimental_optimization.map_vectorization.enabled = True
    _DATA_OPTIONS.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass


def make_ds(X, Y=None, batch_size=512, training=False, cache=False):
    if Y is None:
        ds = tf.data.Dataset.from_tensor_slices(X)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, Y))

    ds = ds.with_options(_DATA_OPTIONS)
    ds = ds.batch(batch_size, drop_remainder=False)

    if cache:
        ds = ds.cache()

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

BATCH_SIZE = 512

test_ds = make_ds(test3, batch_size=BATCH_SIZE, training=False, cache=True)
test_pred_steps = int(np.ceil(len(test3) / BATCH_SIZE))

fold_indices = list(kf.split(train3, y))

test_preds = []

es = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=25,
    verbose=1,
    restore_best_weights=True,
)

for fold, (train_idx, valid_idx) in enumerate(fold_indices):
    print(f"****** fold: {fold+1} *******")

    X_tr, y_tr = train3[train_idx], y[train_idx]
    X_va, y_va = train3[valid_idx], y[valid_idx]

    steps_per_epoch = int(np.ceil(len(train_idx) / BATCH_SIZE))
    decay_steps = int(200 * steps_per_epoch)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=max(decay_steps, 1),
        decay_rate=1e-5,
        staircase=False,
    )

    with strategy.scope():
        model = create_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss="mae",
        )

    tr_ds = make_ds(X_tr, y_tr, batch_size=BATCH_SIZE, training=True, cache=False)
    va_ds = make_ds(X_va, y_va, batch_size=BATCH_SIZE, training=False, cache=True)

    history = model.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=240,
        callbacks=[es],
        verbose=1,
    )

    preds = model.predict(test_ds, verbose=0, steps=test_pred_steps).squeeze()
    if preds.ndim == 1:
        preds = preds.reshape(-1, SEQ_LEN)
    test_preds.append(preds.astype(np.float32, copy=False))

    del model, history, preds, X_tr, y_tr, X_va, y_va, tr_ds, va_ds
    gc.collect()



## === cell 11
pred_mean = np.mean(np.stack(test_preds, axis=0), axis=0)  # (n_breaths_test, 80)
pred_flat = pred_mean.reshape(-1)

assert len(pred_flat) == len(
    submission
), "Prediction length does not match submission length"

submission["id"] = test_ids
submission["pressure"] = pred_flat.astype(np.float32)

submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 12
print("Wrote:", os.path.abspath("submission.csv"))
subm_check = pd.read_csv("submission.csv")
print(subm_check.head())
print(subm_check.shape)
assert list(subm_check.columns) == ["id", "pressure"]
assert len(subm_check) == 603600
