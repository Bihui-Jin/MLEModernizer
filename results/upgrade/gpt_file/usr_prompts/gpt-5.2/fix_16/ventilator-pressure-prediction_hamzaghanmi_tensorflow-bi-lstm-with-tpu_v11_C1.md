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
import os, gc, logging, time
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold


os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

print("TensorFlow:", tf.__version__)

np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 8))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

tf.get_logger().setLevel(logging.ERROR)

try:
    tf.config.optimizer.set_jit(True)
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
sample_sub = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv",
    dtype={"id": "int32", "pressure": "float32"},
)

print(train.shape, test.shape, sample_sub.shape)
train.head()



## === cell 2
test.head()



## === cell 3
train_id = train["id"].copy()
test_id = test["id"].copy()




## === cell 4
def add_features_fast_inplace(df: pd.DataFrame) -> pd.DataFrame:
    Rf = df["R"].to_numpy(dtype=np.float32, copy=False)
    Cf = df["C"].to_numpy(dtype=np.float32, copy=False)
    df["RC_sum"] = (Rf + Cf).astype(np.float32, copy=False)
    df["RC_div"] = (Rf / Cf).astype(np.float32, copy=False)

    n = len(df)
    TIMESTEPS = 80
    if n % TIMESTEPS != 0:
        raise ValueError(f"Row count must be divisible by {TIMESTEPS}: rows={n}")
    nb = n // TIMESTEPS

    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False).reshape(nb, TIMESTEPS)
    u_out = df["u_out"].to_numpy(dtype=np.float32, copy=False).reshape(nb, TIMESTEPS)
    t = df["time_step"].to_numpy(dtype=np.float32, copy=False).reshape(nb, TIMESTEPS)

    df["u_in_cumsum"] = np.cumsum(u_in, axis=1, dtype=np.float32).reshape(-1)

    time_lag = np.zeros_like(t, dtype=np.float32)
    time_lag[:, 1:] = t[:, :-1]
    df["time_lag"] = time_lag.reshape(-1)

    u_in_lag = np.zeros_like(u_in, dtype=np.float32)
    u_in_lag[:, 1:] = u_in[:, :-1]
    df["u_in_lag"] = u_in_lag.reshape(-1)

    u_out_lag = np.zeros_like(u_out, dtype=np.float32)
    u_out_lag[:, 1:] = u_out[:, :-1]
    df["u_out_lag"] = u_out_lag.reshape(-1)

    time_lag2 = np.zeros_like(t, dtype=np.float32)
    time_lag2[:, 2:] = t[:, :-2]
    df["time_lag2"] = time_lag2.reshape(-1)

    return df


train = add_features_fast_inplace(train)
test = add_features_fast_inplace(test)


def add_rc_dummies_inplace(df: pd.DataFrame) -> pd.DataFrame:
    Rv = df["R"].to_numpy(copy=False)
    Cv = df["C"].to_numpy(copy=False)

    df["R_5"] = (Rv == 5).astype(np.int8)
    df["R_20"] = (Rv == 20).astype(np.int8)
    df["R_50"] = (Rv == 50).astype(np.int8)

    df["C_10"] = (Cv == 10).astype(np.int8)
    df["C_20"] = (Cv == 20).astype(np.int8)
    df["C_50"] = (Cv == 50).astype(np.int8)

    return df


train = add_rc_dummies_inplace(train)
test = add_rc_dummies_inplace(test)

y = train["pressure"].to_numpy(dtype=np.float32, copy=False)

drop_cols = ["pressure", "id", "breath_id", "R", "C"]
train = train.drop(columns=drop_cols)
test = test.drop(columns=["id", "breath_id", "R", "C"])

print("Feature columns:", train.shape[1])



## === cell 5
rb = RobustScaler()
train2 = rb.fit_transform(train).astype(np.float32, copy=False)
test2 = rb.transform(test).astype(np.float32, copy=False)



## === cell 6
TIMESTEPS = 80
n_features = train2.shape[1]

if train2.shape[0] % TIMESTEPS != 0 or test2.shape[0] % TIMESTEPS != 0:
    raise ValueError(
        f"Row counts must be divisible by {TIMESTEPS}: "
        f"train rows={train2.shape[0]}, test rows={test2.shape[0]}"
    )

n_train_breaths = train2.shape[0] // TIMESTEPS
n_test_breaths = test2.shape[0] // TIMESTEPS

train3 = np.ascontiguousarray(
    train2.reshape(n_train_breaths, TIMESTEPS, n_features), dtype=np.float32
)
test3 = np.ascontiguousarray(
    test2.reshape(n_test_breaths, TIMESTEPS, n_features), dtype=np.float32
)
y = np.ascontiguousarray(y.reshape(n_train_breaths, TIMESTEPS), dtype=np.float32)

del train2, test2, train, test
gc.collect()

print("train3:", train3.shape, "test3:", test3.shape, "y:", y.shape)



## === cell 7
strategy = tf.distribute.get_strategy()
print("REPLICAS:", strategy.num_replicas_in_sync)




## === cell 8
def create_model(lr_schedule):
    with strategy.scope():
        model = keras.Sequential(
            [
                layers.Input(shape=(80, train3.shape[2])),
                layers.Bidirectional(
                    layers.LSTM(
                        300, return_sequences=True, recurrent_activation="sigmoid"
                    )
                ),
                layers.Bidirectional(
                    layers.LSTM(
                        260, return_sequences=True, recurrent_activation="sigmoid"
                    )
                ),
                layers.Bidirectional(
                    layers.LSTM(
                        220, return_sequences=True, recurrent_activation="sigmoid"
                    )
                ),
                layers.Bidirectional(
                    layers.LSTM(
                        150, return_sequences=True, recurrent_activation="sigmoid"
                    )
                ),
                layers.Dense(100, activation="selu"),
                layers.Dropout(0.1),
                layers.Dense(1),
            ]
        )
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=opt, loss="mae")
    return model




## === cell 9
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = []

BATCH = 512


def make_train_valid_datasets(x_np, y_np, train_idx, valid_idx, batch_size, seed=42):
    x_tf = tf.convert_to_tensor(x_np)
    y_tf = tf.convert_to_tensor(y_np)

    train_idx_np = np.asarray(train_idx, dtype=np.int32)
    valid_idx_np = np.asarray(valid_idx, dtype=np.int32)

    train_idx_ds = tf.data.Dataset.from_tensor_slices(train_idx_np)
    valid_idx_ds = tf.data.Dataset.from_tensor_slices(valid_idx_np)

    train_idx_ds = train_idx_ds.shuffle(
        buffer_size=train_idx_np.shape[0],
        seed=seed,
        reshuffle_each_iteration=True,
    )

    def _gather_one(i):
        xi = tf.gather(x_tf, i)
        yi = tf.gather(y_tf, i)
        return xi, yi

    options = tf.data.Options()
    options.deterministic = True

    train_ds = train_idx_ds.map(_gather_one, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.with_options(options)
    train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(
        tf.data.AUTOTUNE
    )

    valid_ds = valid_idx_ds.map(_gather_one, num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = valid_ds.with_options(options)
    valid_ds = valid_ds.batch(batch_size, drop_remainder=False).prefetch(
        tf.data.AUTOTUNE
    )

    return train_ds, valid_ds


def make_test_dataset(x_np, batch_size):
    x_tf = tf.convert_to_tensor(x_np)
    options = tf.data.Options()
    options.deterministic = True
    ds = tf.data.Dataset.from_tensor_slices(x_tf).with_options(options)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_test_dataset(test3, BATCH)

for fold, (train_idx, valid_idx) in enumerate(kf.split(train3, y)):
    print(f"****** fold: {fold+1} *******")

    steps_per_epoch = int(np.ceil(len(train_idx) / BATCH))
    decay_steps = max(1, int(200 * steps_per_epoch))
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=decay_steps,
        decay_rate=1e-5,
        staircase=False,
    )

    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=40,
        verbose=1,
        restore_best_weights=True,
    )

    train_ds, valid_ds = make_train_valid_datasets(
        train3, y, train_idx, valid_idx, batch_size=BATCH, seed=42
    )

    model = create_model(lr_schedule)

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=340,
        callbacks=[es],
        verbose=2,
    )

    pred = model.predict(test_ds, verbose=0).squeeze().reshape(-1)
    test_preds.append(pred)

    del model, history, pred, train_ds, valid_ds, train_idx, valid_idx
    tf.keras.backend.clear_session()
    gc.collect()

    if fold >= 0:
        break

print("Got preds folds:", len(test_preds), "pred shape:", test_preds[0].shape)



## === cell 10
if len(test_preds) == 0:
    raise RuntimeError(
        "No test predictions were generated; training likely failed earlier."
    )
if test_preds[0].shape[0] != sample_sub.shape[0]:
    raise ValueError(
        f"Prediction length mismatch: pred={test_preds[0].shape[0]} vs sample_sub={sample_sub.shape[0]}"
    )

submission = sample_sub.copy()
submission["pressure"] = test_preds[0].astype(np.float32)

assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["id", "pressure"]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
submission.head()
