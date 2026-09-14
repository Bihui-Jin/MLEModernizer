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
import time
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

tf.get_logger().setLevel(logging.ERROR)
print("TensorFlow:", tf.__version__)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        print(f"GPUs available: {len(gpus)}")
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 1
train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
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
    "../input/ventilator-pressure-prediction/test.csv",
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



## === cell 2
print("train shape:", train.shape, "test shape:", test.shape)
print(train.dtypes)



## === cell 3
print(test.dtypes)




## === cell 4
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(deep=False)

    n = len(df)
    if n % 80 != 0:
        raise ValueError(
            "Row count is not divisible by 80; cannot safely vectorize by breath."
        )

    breaths = n // 80

    u_in = df["u_in"].to_numpy(np.float32, copy=False).reshape(breaths, 80)
    u_out = df["u_out"].to_numpy(np.float32, copy=False).reshape(breaths, 80)
    time_step = df["time_step"].to_numpy(np.float32, copy=False).reshape(breaths, 80)

    df["area"] = np.cumsum(time_step * u_in, axis=1, dtype=np.float32).reshape(-1)
    df["cross"] = (u_in * u_out).reshape(-1).astype(np.float32, copy=False)
    df["cross2"] = (time_step * u_out).reshape(-1).astype(np.float32, copy=False)

    u_in_cumsum = np.cumsum(u_in, axis=1, dtype=np.float32)
    df["u_in_cumsum"] = u_in_cumsum.reshape(-1)

    denom = (np.arange(80, dtype=np.float32) + 1.0)[None, :]
    df["u_in_cummean"] = (u_in_cumsum / denom).reshape(-1)

    u_in_lag = np.zeros_like(u_in, dtype=np.float32)
    u_in_lag[:, 1:] = u_in[:, :-1]
    u_in_lag2 = np.zeros_like(u_in, dtype=np.float32)
    u_in_lag2[:, 2:] = u_in[:, :-2]
    u_out_lag2 = np.zeros_like(u_out, dtype=np.float32)
    u_out_lag2[:, 2:] = u_out[:, :-2]

    df["u_in_lag"] = u_in_lag.reshape(-1)
    df["u_in_lag2"] = u_in_lag2.reshape(-1)
    df["u_out_lag2"] = u_out_lag2.reshape(-1)

    b = df["breath_id"].to_numpy(np.int32, copy=False).reshape(breaths, 80)
    breath_id_lag = np.zeros_like(b, dtype=np.int32)
    breath_id_lag[:, 1:] = b[:, :-1]
    breath_id_lag2 = np.zeros_like(b, dtype=np.int32)
    breath_id_lag2[:, 2:] = b[:, :-2]
    df["breath_id_lag"] = breath_id_lag.reshape(-1)
    df["breath_id_lag2"] = breath_id_lag2.reshape(-1)

    lagsame = np.zeros((breaths, 80), dtype=np.int8)
    lagsame[:, 1:] = 1
    lag2same = np.zeros((breaths, 80), dtype=np.int8)
    lag2same[:, 2:] = 1
    df["breath_id_lagsame"] = lagsame.reshape(-1)
    df["breath_id_lag2same"] = lag2same.reshape(-1)

    df["count"] = np.broadcast_to(
        (np.arange(80, dtype=np.int16) + 1)[None, :], (breaths, 80)
    ).reshape(-1)
    df["one"] = np.int8(1)

    df["RC"] = (df["R"].astype("int16") * 100 + df["C"].astype("int16")).astype("int16")

    df["R"] = pd.Categorical(df["R"], categories=[5, 20, 50], ordered=False)
    df["C"] = pd.Categorical(df["C"], categories=[10, 20, 50], ordered=False)
    df["RC"] = pd.Categorical(
        df["RC"],
        categories=sorted(
            (
                np.array([5, 20, 50])[:, None] * 100 + np.array([10, 20, 50])[None, :]
            ).ravel()
        ),
        ordered=False,
    )

    df = pd.get_dummies(df, columns=["R", "C", "RC"], dtype="int8")
    return df


t0 = time.time()
train = add_features(train)
test = add_features(test)
print("feature engineering seconds:", round(time.time() - t0, 2))



## === cell 5
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

y = train["pressure"].to_numpy(dtype=np.float32).reshape(-1, 80)

train.drop(drop_cols, axis=1, inplace=True)
test.drop([c for c in drop_cols if c in test.columns], axis=1, inplace=True)

target_cols = train.columns
test = test.reindex(columns=target_cols, fill_value=0)



## === cell 6
print("train features shape:", train.shape, "test features shape:", test.shape)



## === cell 7
t0 = time.time()
rb = RobustScaler()

train_np = train.to_numpy(dtype=np.float32, copy=False)
test_np = test.to_numpy(dtype=np.float32, copy=False)

rb.fit(train_np)
train = np.ascontiguousarray(rb.transform(train_np), dtype=np.float32)
test = np.ascontiguousarray(rb.transform(test_np), dtype=np.float32)

del train_np, test_np
gc.collect()
print("scaling seconds:", round(time.time() - t0, 2))



## === cell 8
train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, train.shape[-1])

gc.collect()
print("train reshaped:", train.shape, "test reshaped:", test.shape)



## === cell 9
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except Exception:
    strategy = tf.distribute.MirroredStrategy()
    print("Running on CPU/GPU. Replicas:", strategy.num_replicas_in_sync)

print("REPLICAS:", strategy.num_replicas_in_sync)




## === cell 10
def plot_hist(hist):
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("model performance")
    plt.ylabel("mean_absolute_error")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()




## === cell 11
def create_model(input_dim, steps_per_epoch):
    with strategy.scope():
        model = Sequential(
            [
                Input(shape=(80, input_dim)),
                Bidirectional(LSTM(700, return_sequences=True)),
                Bidirectional(LSTM(512, return_sequences=True)),
                Bidirectional(LSTM(256, return_sequences=True)),
                Bidirectional(LSTM(128, return_sequences=True)),
                Dense(256, activation="selu"),
                Dense(1),
            ]
        )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=max(1, 200 * steps_per_epoch),
            decay_rate=1e-5,
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(optimizer=optimizer, loss="mae", steps_per_execution=128)
    return model




## === cell 12
BATCH_SIZE = 512

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

AUTO = tf.data.AUTOTUNE
ds_options = tf.data.Options()
ds_options.experimental_deterministic = True  # preserve stable order/reproducibility

X_test = test  # numpy, contiguous float32
test_ds = (
    tf.data.Dataset.from_tensor_slices(X_test)
    .with_options(ds_options)
    .batch(BATCH_SIZE * 2, drop_remainder=False)
    .prefetch(AUTO)
)

n_test = X_test.shape[0] * X_test.shape[1]
test_pred_matrix = np.empty((kf.get_n_splits(), n_test), dtype=np.float32)

for fold, (train_idx, test_idx) in enumerate(kf.split(train, y)):
    print(f"****** fold: {fold+1} *******")
    t0 = time.time()

    X_tr = train[train_idx]
    y_tr = y[train_idx]
    X_va = train[test_idx]
    y_va = y[test_idx]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
        .with_options(ds_options)
        .shuffle(
            buffer_size=min(len(X_tr), 8192), seed=SEED, reshuffle_each_iteration=True
        )
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTO)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_va, y_va))
        .with_options(ds_options)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTO)
    )

    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=35,
        verbose=1,
        restore_best_weights=True,
    )

    steps_per_epoch = int(np.ceil((len(train_idx)) / BATCH_SIZE))
    model = create_model(input_dim=train.shape[-1], steps_per_epoch=steps_per_epoch)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=300,
        callbacks=[es],
        verbose=1,
    )

    preds = model.predict(test_ds, verbose=0)
    test_pred_matrix[fold, :] = preds.reshape(-1).astype(np.float32, copy=False)

    tf.keras.backend.clear_session()
    del X_tr, y_tr, X_va, y_va, model, history, preds, train_ds, val_ds
    gc.collect()
    print("fold seconds:", round(time.time() - t0, 2))



## === cell 13
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv",
    dtype={"id": "int32", "pressure": "float32"},
)

submission["pressure"] = test_pred_matrix.mean(axis=0)
submission.to_csv("submission_mean.csv", index=False)
print(submission.head())



## === cell 14
submission_med = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv",
    dtype={"id": "int32", "pressure": "float32"},
)
submission_med["pressure"] = np.median(test_pred_matrix, axis=0)
submission_med.to_csv("submission_median.csv", index=False)
print(submission_med.head())



## === cell 15
pressure_unique = np.array(sorted(pd.Series(y.reshape(-1)).unique()), dtype=np.float32)

submission_post = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv",
    dtype={"id": "int32", "pressure": "float32"},
)
pred = submission_med["pressure"].to_numpy(dtype=np.float32)

pos = np.searchsorted(pressure_unique, pred, side="left")
pos0 = np.clip(pos - 1, 0, len(pressure_unique) - 1)
pos1 = np.clip(pos, 0, len(pressure_unique) - 1)

cand0 = pressure_unique[pos0]
cand1 = pressure_unique[pos1]
choose1 = np.abs(cand1 - pred) < np.abs(cand0 - pred)
nearest = np.where(choose1, cand1, cand0)

submission_post["pressure"] = nearest.astype(np.float32, copy=False)
submission_post.to_csv("submission.csv", index=False)
print(submission_post.head())
