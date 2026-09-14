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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 1
rb = RobustScaler()



## === cell 2
BASE = "../input/ventilator-pressure-prediction"
if not os.path.exists(BASE):
    BASE = "/kaggle/input/ventilator-pressure-prediction"
print("Using data base:", BASE)

train = pd.read_csv(
    f"{BASE}/train.csv",
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
    f"{BASE}/test.csv",
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

print("train shape:", train.shape, "test shape:", test.shape)




## === cell 3
def add_breath_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )

    breath = df["breath_id"].to_numpy()
    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)

    new_breath = np.empty_like(breath, dtype=bool)
    new_breath[0] = True
    new_breath[1:] = breath[1:] != breath[:-1]

    u_in_lag1 = np.empty_like(u_in, dtype=np.float32)
    u_in_lag1[0] = 0.0
    u_in_lag1[1:] = u_in[:-1]
    u_in_lag1[new_breath] = 0.0

    u_in_diff1 = u_in - u_in_lag1

    u_in_cumsum = (
        df.groupby("breath_id", sort=False)["u_in"]
        .cumsum()
        .to_numpy(dtype=np.float32, copy=False)
    )

    df["u_in_lag1"] = u_in_lag1
    df["u_in_diff1"] = u_in_diff1
    df["u_in_cumsum"] = u_in_cumsum
    return df


train = add_breath_features(train)
test = add_breath_features(test)



## === cell 4
targets = train["pressure"].to_numpy().reshape(-1, 80, 1)

test_id = test["id"].copy()

train.drop(columns=["id", "breath_id", "pressure", "time_step"], inplace=True)
test.drop(columns=["id", "breath_id", "time_step"], inplace=True)

print("Targets reshaped:", targets.shape)
print("Train cols:", train.columns.tolist())



## === cell 5
rb.fit(train)

train_new = rb.transform(train).astype(np.float32, copy=False)
test_new = rb.transform(test).astype(np.float32, copy=False)

train_re = train_new.reshape(-1, 80, train_new.shape[-1])
test_re = test_new.reshape(-1, 80, test_new.shape[-1])

print("train_re:", train_re.shape, "targets:", targets.shape, "test_re:", test_re.shape)




## === cell 6
def build_model():
    x_input = layers.Input(shape=([80, train_re.shape[-1]]))

    x1 = layers.Bidirectional(layers.LSTM(units=440, return_sequences=True))(x_input)
    x2 = layers.Bidirectional(
        layers.LSTM(
            units=360,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer="random_normal",
        )
    )(x1)
    x3 = layers.Bidirectional(
        layers.LSTM(
            units=240,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer="random_normal",
        )
    )(x2)
    x4 = layers.Bidirectional(
        layers.LSTM(
            units=180,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer="random_normal",
        )
    )(x3)
    x5 = layers.Bidirectional(
        layers.LSTM(
            units=100, return_sequences=True, kernel_initializer="random_normal"
        )
    )(x4)

    z2 = layers.Bidirectional(layers.GRU(units=240, return_sequences=True))(x2)

    z31 = layers.Multiply()([x3, z2])
    z31 = layers.BatchNormalization()(z31)
    z3 = layers.Bidirectional(layers.GRU(units=180, return_sequences=True))(z31)

    z41 = layers.Multiply()([x4, z3])
    z41 = layers.BatchNormalization()(z41)
    z4 = layers.Bidirectional(layers.GRU(units=100, return_sequences=True))(z41)

    z51 = layers.Multiply()([x5, z4])
    z51 = layers.BatchNormalization()(z51)
    z5 = layers.Bidirectional(layers.GRU(units=64, return_sequences=True))(z51)

    out_in = layers.Concatenate(axis=2)([x5, z2, z3, z4, z5])

    x1_out = layers.Dense(units=64, activation="relu")(out_in)
    x1_out = layers.Dense(1, name="out1")(x1_out)

    x2_out = layers.TimeDistributed(layers.Dense(units=128, activation="relu"))(out_in)
    x2_out = layers.TimeDistributed(layers.Dense(1), name="out2")(x2_out)

    mean_out = layers.Average(name="mean_out")([x1_out, x2_out])

    model = tf.keras.Model(
        inputs=x_input, outputs=[x1_out, x2_out, mean_out], name="Base_Model"
    )

    losses = {
        "out1": tf.keras.losses.MeanAbsoluteError(),
        "out2": tf.keras.losses.MeanAbsoluteError(),
        "mean_out": tf.keras.losses.MeanAbsoluteError(),
    }
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=losses)
    return model




## === cell 7
try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # no .connect()
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
    print("Using TPU:", tpu_resolver.master())
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print("TPU not available; using default strategy. Reason:", repr(e))



## === cell 8
EPOCH = 320
BATCH_SIZE = 512

reduce_lr = ReduceLROnPlateau(monitor="val_loss", verbose=1, factor=0.87, patience=8)

trained_models = []


def make_ds(X, y, training: bool):
    y_dict = {"out1": y, "out2": y, "mean_out": y}
    ds = tf.data.Dataset.from_tensor_slices((X, y_dict))
    if training:
        ds = ds.shuffle(min(len(X), 8192), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


with strategy.scope():
    kf = KFold(n_splits=7, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_re, targets), start=1):
        print("-" * 30, ">", f"Fold {fold}", "<", "-" * 30)

        tf.keras.backend.clear_session()

        X_train, X_valid = train_re[train_idx], train_re[valid_idx]
        y_train, y_valid = targets[train_idx], targets[valid_idx]

        train_ds = make_ds(X_train, y_train, training=True)
        valid_ds = make_ds(X_valid, y_valid, training=False)

        model = build_model()

        ckpt_path = f"NewMean{fold}.weights.h5"

        call_back_mean = tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            verbose=1,
            monitor="val_mean_out_loss",
            save_best_only=True,
            save_weights_only=True,
        )

        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCH,
            callbacks=[call_back_mean, reduce_lr],
            verbose=2,
        )

        best_model = build_model()
        best_model.load_weights(ckpt_path)
        trained_models.append(best_model)

        _ = history

print("Number of trained models:", len(trained_models))



## === cell 9
unique_pressures = np.unique(targets)
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = (unique_pressures[1] - unique_pressures[0]).item()
PRESSURE_MIN = sorted_pressures[0].item()
PRESSURE_MAX = sorted_pressures[-1].item()

print("PRESSURE_STEP, MIN, MAX:", PRESSURE_STEP, PRESSURE_MIN, PRESSURE_MAX)




## === cell 10
def make_test_ds(X):
    return (
        tf.data.Dataset.from_tensor_slices(X)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )


test_ds = make_test_ds(test_re)

test_preds = []
for i, model in enumerate(trained_models, start=1):
    print(f"Predicting fold {i}...")
    pred = model.predict(test_ds, verbose=1)
    test_preds.append(pred)



## === cell 11
preds_out1 = np.concatenate([p[0] for p in test_preds], axis=-1)  # (n,80,n_models)
preds_out2 = np.concatenate([p[1] for p in test_preds], axis=-1)
preds_mean = np.concatenate([p[2] for p in test_preds], axis=-1)

median_pre = np.median(preds_mean, axis=-1)  # (n,80)

rounding_pre = (
    np.round((median_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)

print("Final pred shape:", clipped_pre.shape)



## === cell 12
submission_file = pd.read_csv(f"{BASE}/sample_submission.csv")

pred_flat = clipped_pre.reshape(-1)
if len(pred_flat) != len(submission_file):
    raise ValueError(
        f"Prediction length {len(pred_flat)} does not match submission length {len(submission_file)}"
    )

submission_file["pressure"] = pred_flat.astype(np.float32)
submission_file.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_file.shape)



## === cell 13
submission_file.head()
