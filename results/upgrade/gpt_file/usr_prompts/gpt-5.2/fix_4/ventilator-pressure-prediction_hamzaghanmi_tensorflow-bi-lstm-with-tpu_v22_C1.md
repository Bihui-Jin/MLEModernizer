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

    df["area"] = (df["time_step"] * df["u_in"]).astype(np.float32)
    df["area"] = df.groupby("breath_id", sort=False)["area"].cumsum()

    df["cross"] = (df["u_in"] * df["u_out"]).astype(np.float32)
    df["cross2"] = (df["time_step"] * df["u_out"]).astype(np.float32)

    df["u_in_cumsum"] = df.groupby("breath_id", sort=False)["u_in"].cumsum()
    df["count"] = (df.groupby("breath_id", sort=False).cumcount() + 1).astype(np.int16)
    df["one"] = 1  # kept because later code drops it; values identical
    df["u_in_cummean"] = (df["u_in_cumsum"] / df["count"]).astype(np.float32)

    breath = df["breath_id"].to_numpy()
    u_in = df["u_in"].to_numpy()
    u_out = df["u_out"].to_numpy()

    lag1_same = np.zeros(len(df), dtype=np.int8)
    lag2_same = np.zeros(len(df), dtype=np.int8)
    lag1_same[1:] = breath[1:] == breath[:-1]
    lag2_same[2:] = breath[2:] == breath[:-2]

    u_in_lag = np.zeros(len(df), dtype=np.float32)
    u_in_lag2 = np.zeros(len(df), dtype=np.float32)
    u_out_lag2 = np.zeros(len(df), dtype=np.float32)

    u_in_lag[1:] = u_in[:-1]
    u_in_lag2[2:] = u_in[:-2]
    u_out_lag2[2:] = u_out[:-2]

    df["breath_id_lag"] = pd.Series(np.r_[0, breath[:-1]], index=df.index)
    df["breath_id_lag2"] = pd.Series(np.r_[0, 0, breath[:-2]], index=df.index)
    df["breath_id_lagsame"] = lag1_same
    df["breath_id_lag2same"] = lag2_same

    df["u_in_lag"] = (u_in_lag * lag1_same).astype(np.float32)
    df["u_in_lag2"] = (u_in_lag2 * lag2_same).astype(np.float32)
    df["u_out_lag2"] = (u_out_lag2 * lag2_same).astype(np.float32)

    df["R"] = df["R"].astype(np.int16)
    df["C"] = df["C"].astype(np.int16)
    df["RC"] = (df["R"].astype(np.int32) * 1000 + df["C"].astype(np.int32)).astype(
        np.int32
    )

    return df


train = add_features(train)
test = add_features(test)



## === cell 5
y = train["pressure"].to_numpy().reshape(-1, 80)

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
R_cats = [5, 20, 50]
C_cats = [10, 20, 50]
RC_cats = sorted({r * 1000 + c for r in R_cats for c in C_cats})

train["R"] = pd.Categorical(train["R"], categories=R_cats)
test["R"] = pd.Categorical(test["R"], categories=R_cats)
train["C"] = pd.Categorical(train["C"], categories=C_cats)
test["C"] = pd.Categorical(test["C"], categories=C_cats)
train["RC"] = pd.Categorical(train["RC"], categories=RC_cats)
test["RC"] = pd.Categorical(test["RC"], categories=RC_cats)

train = pd.get_dummies(train, columns=["R", "C", "RC"])
test = pd.get_dummies(test, columns=["R", "C", "RC"])

train_cols = train.columns
test = test.reindex(columns=train_cols, fill_value=0)

rb = RobustScaler()
rb.fit(train)

train = rb.transform(train)
test = rb.transform(test)



## === cell 8
n_features = train.shape[-1]
train = train.reshape(-1, 80, n_features)
test = test.reshape(-1, 80, n_features)
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

        steps_per_epoch = int(np.ceil((len(train) * 0.8) / 512))
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
BATCH_SIZE = 512

SHUFFLE_BUFFER = 8192


def make_ds(X, Y=None, training=False, cache=False):
    if Y is None:
        ds = tf.data.Dataset.from_tensor_slices(X)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, Y))

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(
            buffer_size=SHUFFLE_BUFFER, seed=42, reshuffle_each_iteration=True
        )

    if cache:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


kf = KFold(n_splits=5, shuffle=True, random_state=42)

test_preds = []
n_features = train.shape[-1]

test_ds = make_ds(test, Y=None, training=False, cache=True)

for fold, (train_idx, valid_idx) in enumerate(kf.split(train, y)):
    print(f"****** fold: {fold+1} *******")
    X_train, X_valid = train[train_idx], train[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=35,
        verbose=0,
        restore_best_weights=True,
    )

    model = create_model(n_features)

    train_ds = make_ds(X_train, y_train, training=True, cache=True)
    valid_ds = make_ds(X_valid, y_valid, training=False, cache=True)

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=300,
        callbacks=[es],
        verbose=0,
    )

    pred = model.predict(test_ds, verbose=0)  # (n_breaths, 80, 1)
    pred = pred.squeeze(-1)  # (n_breaths, 80)
    test_preds.append(pred.reshape(-1))  # (n_rows,)

    plot_hist(history)

    del X_train, X_valid, y_train, y_valid, model, history, pred, train_ds, valid_ds
    gc.collect()



## === cell 13
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv",
    dtype={"id": "int32", "pressure": "float32"},
)

if len(test_preds) == 0:
    submission["pressure"] = 0.0
else:
    submission["pressure"] = np.mean(np.vstack(test_preds), axis=0)

submission.to_csv("submission_mean.csv", index=False)
print(submission.head())



## === cell 14
if len(test_preds) == 0:
    submission["pressure"] = 0.0
else:
    submission["pressure"] = np.median(np.vstack(test_preds), axis=0)
print(submission.head())



## === cell 15
pressure_unique = np.array(sorted(np.unique(y.reshape(-1))))
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
