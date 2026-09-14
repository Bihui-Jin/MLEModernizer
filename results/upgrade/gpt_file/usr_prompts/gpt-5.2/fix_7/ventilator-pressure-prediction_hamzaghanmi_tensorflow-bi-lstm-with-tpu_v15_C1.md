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

# 5. Target score

0.1959148201672421

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
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

tf.get_logger().setLevel(logging.ERROR)

os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF choose
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

tf.keras.backend.clear_session()



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
_ = train.shape



## === cell 3
_ = test.shape




## === cell 4
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("breath_id", sort=False)

    ts = df["time_step"].to_numpy(dtype=np.float32, copy=False)
    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)
    u_out = df["u_out"].to_numpy(copy=False)
    breath_id = df["breath_id"].to_numpy(copy=False)

    df["area"] = (ts * u_in).astype(np.float32, copy=False)
    df["area"] = g["area"].cumsum().astype(np.float32)

    df["cross"] = (u_in * u_out.astype(np.float32, copy=False)).astype(
        np.float32, copy=False
    )
    df["cross2"] = (ts * u_out.astype(np.float32, copy=False)).astype(
        np.float32, copy=False
    )

    df["u_in_cumsum"] = g["u_in"].cumsum().astype(np.float32)
    cnt = (g.cumcount() + 1).to_numpy(dtype=np.float32, copy=False)
    df["u_in_cummean"] = (
        df["u_in_cumsum"].to_numpy(dtype=np.float32, copy=False) / cnt
    ).astype(np.float32, copy=False)

    lag1_same = np.zeros(len(df), dtype=bool)
    lag2_same = np.zeros(len(df), dtype=bool)
    lag1_same[1:] = breath_id[1:] == breath_id[:-1]
    lag2_same[2:] = breath_id[2:] == breath_id[:-2]

    u_in_lag = np.zeros_like(u_in, dtype=np.float32)
    u_in_lag2 = np.zeros_like(u_in, dtype=np.float32)
    u_out_lag2 = np.zeros_like(u_in, dtype=np.float32)

    u_in_lag[lag1_same] = u_in[:-1][lag1_same[1:]]
    u_in_lag2[lag2_same] = u_in[:-2][lag2_same[2:]]
    u_out_lag2[lag2_same] = u_out[:-2].astype(np.float32, copy=False)[lag2_same[2:]]

    df["u_in_lag"] = u_in_lag
    df["u_in_lag2"] = u_in_lag2
    df["u_out_lag2"] = u_out_lag2

    R = df["R"].to_numpy(copy=False).astype(np.int16, copy=False)
    C = df["C"].to_numpy(copy=False).astype(np.int16, copy=False)
    df["R_code"] = pd.Categorical(R, categories=[5, 20, 50]).codes.astype(
        np.int8, copy=False
    )
    df["C_code"] = pd.Categorical(C, categories=[10, 20, 50]).codes.astype(
        np.int8, copy=False
    )
    rc_key = R.astype(np.int32, copy=False) * 100 + C.astype(np.int32, copy=False)
    rc_cats = np.array(
        [510, 520, 550, 2010, 2020, 2050, 5010, 5020, 5050], dtype=np.int32
    )
    df["RC_code"] = pd.Categorical(rc_key, categories=rc_cats).codes.astype(
        np.int8, copy=False
    )

    return df


train = add_features(train)
test = add_features(test)



## === cell 5
y = train["pressure"].to_numpy(dtype=np.float32).reshape(-1, 80)

drop_cols = [
    "pressure",
    "id",
    "breath_id",
    "one",  # may not exist; drop safely
    "count",
    "breath_id_lag",
    "breath_id_lag2",
    "breath_id_lagsame",
    "breath_id_lag2same",
    "u_out_lag2",
]
train.drop([c for c in drop_cols if c in train.columns], axis=1, inplace=True)

drop_cols_test = [
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
test.drop([c for c in drop_cols_test if c in test.columns], axis=1, inplace=True)

test = test.reindex(columns=train.columns, fill_value=0)



## === cell 6
_ = train.shape



## === cell 7
rb = RobustScaler()
train_np = train.to_numpy(dtype=np.float32, copy=False)
test_np = test.to_numpy(dtype=np.float32, copy=False)

rb.fit(train_np)
train_np = rb.transform(train_np).astype(np.float32, copy=False)
test_np = rb.transform(test_np).astype(np.float32, copy=False)

del train, test
gc.collect()



## === cell 8
train = train_np.reshape(-1, 80, train_np.shape[-1])
test = test_np.reshape(-1, 80, train_np.shape[-1])
del train_np, test_np
gc.collect()



## === cell 9
print("TensorFlow:", tf.__version__)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except Exception:
    strategy = tf.distribute.MirroredStrategy()
    print(
        "Running with strategy:",
        type(strategy).__name__,
        "| replicas:",
        strategy.num_replicas_in_sync,
    )

print("REPLICAS: ", strategy.num_replicas_in_sync)

tf.config.run_functions_eagerly(False)




## === cell 10
def plot_hist(hist):
    return




## === cell 11
def create_model(n_features):
    with strategy.scope():
        model = Sequential(
            [
                Input(shape=(80, n_features)),
                Bidirectional(LSTM(700, return_sequences=True)),
                Bidirectional(LSTM(512, return_sequences=True)),
                Bidirectional(LSTM(256, return_sequences=True)),
                Bidirectional(LSTM(128, return_sequences=True)),
                Dense(100, activation="selu"),
                Dense(1),
            ]
        )
    return model




## === cell 12
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 512

kf = KFold(n_splits=5, shuffle=True, random_state=42)

test_preds = []
n_features = train.shape[-1]

options = tf.data.Options()
options.experimental_deterministic = True

train_tf = tf.convert_to_tensor(train, dtype=tf.float32)
y_tf = tf.convert_to_tensor(y, dtype=tf.float32)
test_tf = tf.convert_to_tensor(test, dtype=tf.float32)

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_tf)
    .with_options(options)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)


@tf.function
def _gather_xy(i):
    return train_tf[i], y_tf[i]


for fold, (train_idx, valid_idx) in enumerate(kf.split(train, y)):
    print(f"****** fold: {fold+1} *******")

    steps_per_epoch = int(np.ceil(len(train_idx) / BATCH_SIZE))
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=200 * steps_per_epoch,
        decay_rate=1e-5,
        staircase=False,
    )

    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=35,
        verbose=1,
        restore_best_weights=True,
    )

    with strategy.scope():
        model = create_model(n_features)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss="mae",
            jit_compile=True,
        )

    train_idx_tf = tf.convert_to_tensor(train_idx, dtype=tf.int32)
    valid_idx_tf = tf.convert_to_tensor(valid_idx, dtype=tf.int32)

    train_ds = (
        tf.data.Dataset.from_tensor_slices(train_idx_tf)
        .with_options(options)
        .map(_gather_xy, num_parallel_calls=AUTOTUNE, deterministic=True)
        .cache()
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )
    valid_ds = (
        tf.data.Dataset.from_tensor_slices(valid_idx_tf)
        .with_options(options)
        .map(_gather_xy, num_parallel_calls=AUTOTUNE, deterministic=True)
        .cache()
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=340,
        callbacks=[es],
        verbose=1,
    )

    pred = model.predict(test_ds, verbose=0).squeeze()
    test_preds.append(pred.reshape(-1))

    plot_hist(history)

    del model, history, pred, train_idx_tf, valid_idx_tf, train_ds, valid_ds
    tf.keras.backend.clear_session()
    gc.collect()



## === cell 13
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

test_pred_mean = np.mean(np.vstack(test_preds), axis=0)
if len(test_pred_mean) != len(submission):
    raise ValueError(
        f"Prediction length {len(test_pred_mean)} != submission length {len(submission)}"
    )

submission["pressure"] = test_pred_mean.astype(np.float32)
submission.to_csv("submission.csv", index=False)
submission
