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
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.optimizer.set_jit(True)  # XLA
except Exception:
    pass

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow:", tf.__version__)



## === cell 1
rb = RobustScaler()



## === cell 2
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

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

print(train.shape, test.shape)
print(train.columns)




## === cell 3
def add_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)
    n = u_in.shape[0]
    if n % 80 != 0:
        g = df.groupby("breath_id", sort=False)
        u_in_s = df["u_in"]
        u_in_lag1 = g["u_in"].shift(1).fillna(0.0)
        u_in_lag2 = g["u_in"].shift(2).fillna(0.0)
        df["u_in_lag1"] = u_in_lag1.astype("float32")
        df["u_in_lag2"] = u_in_lag2.astype("float32")
        df["u_in_diff1"] = (u_in_s - u_in_lag1).astype("float32")
        df["u_in_diff2"] = (u_in_s - u_in_lag2).astype("float32")
        df["u_in_cumsum"] = g["u_in"].cumsum().astype("float32")
        return df

    u = u_in.reshape(-1, 80)
    u_lag1 = np.zeros_like(u, dtype=np.float32)
    u_lag2 = np.zeros_like(u, dtype=np.float32)
    u_lag1[:, 1:] = u[:, :-1]
    u_lag2[:, 2:] = u[:, :-2]

    df["u_in_lag1"] = u_lag1.reshape(-1)
    df["u_in_lag2"] = u_lag2.reshape(-1)
    df["u_in_diff1"] = (u - u_lag1).reshape(-1).astype(np.float32, copy=False)
    df["u_in_diff2"] = (u - u_lag2).reshape(-1).astype(np.float32, copy=False)
    df["u_in_cumsum"] = np.cumsum(u, axis=1, dtype=np.float32).reshape(-1)
    return df


train = add_features_fast(train)
test = add_features_fast(test)



## === cell 4
targets = train["pressure"].to_numpy().reshape(-1, 80, 1)

test_id = test["id"].copy()

train.drop(columns=["id", "breath_id", "pressure", "time_step"], inplace=True)
test.drop(columns=["id", "breath_id", "time_step"], inplace=True)

print("Train features:", train.shape, "Targets:", targets.shape)
print("Test features:", test.shape)



## === cell 5
rb.fit(train)

train_new = rb.transform(train).astype(np.float32, copy=False)
test_new = rb.transform(test).astype(np.float32, copy=False)

train_re = np.ascontiguousarray(
    train_new.reshape(-1, 80, train_new.shape[-1]), dtype=np.float32
)
test_re = np.ascontiguousarray(
    test_new.reshape(-1, 80, test_new.shape[-1]), dtype=np.float32
)
targets = np.ascontiguousarray(targets.astype(np.float32, copy=False))

print("train_re:", train_re.shape, "test_re:", test_re.shape)



## === cell 6
opt = tf.keras.optimizers.Adam(learning_rate=0.0015)




## === cell 7
def build_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Bidirectional(
            layers.LSTM(440, return_sequences=True),
            input_shape=[80, train_re.shape[-1]],
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                360,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                260,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                180,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(layers.Bidirectional(layers.LSTM(100, return_sequences=True)))

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.TimeDistributed(layers.Dense(1)))

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=["mae"],
        jit_compile=True,
        steps_per_execution=16,
    )
    return model




## === cell 8
EPOCH = 420
BATCH_SIZE = 1024

try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # auto-detect
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
    print("Running on TPU.")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print("TPU not available, running with strategy:", type(strategy).__name__)
    print("TPU error (ignored):", repr(e))



## === cell 9
reduce_lr = ReduceLROnPlateau(monitor="val_loss", verbose=1, factor=0.9, patience=10)

models = []
kf = KFold(n_splits=10, shuffle=True, random_state=42)


def make_ds(X, y=None, training=False):
    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True

    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(X).with_options(options)
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.cache()
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    ds = tf.data.Dataset.from_tensor_slices((X, y)).with_options(options)
    if training:
        ds = ds.shuffle(
            buffer_size=min(len(X), 8192), seed=42, reshuffle_each_iteration=True
        )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


with strategy.scope():
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_re, targets)):
        print("-" * 30, ">", f"Fold {fold+1}", "<", "-" * 30)
        X_train, X_valid = train_re[train_idx], train_re[valid_idx]
        y_train, y_valid = targets[train_idx], targets[valid_idx]

        train_ds = make_ds(X_train, y_train, training=True)
        valid_ds = make_ds(X_valid, y_valid, training=False)

        model = build_model()

        ckpt_path = f"Model{fold+1}.weights.h5"
        call_back = tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            verbose=0,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )

        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCH,
            callbacks=[call_back, reduce_lr],
            verbose=2,
        )

        model.load_weights(ckpt_path)
        models.append(model)

print("Trained models:", len(models))



## === cell 10
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_re)
    .batch(BATCH_SIZE, drop_remainder=False)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

pred_sum = None
for m in models:
    p = m.predict(test_ds, verbose=0)
    if pred_sum is None:
        pred_sum = p.astype(np.float32, copy=False)
    else:
        pred_sum += p.astype(np.float32, copy=False)

pred = pred_sum / len(models)  # (n_breaths, 80, 1)
print("pred shape:", pred.shape)



## === cell 11
unique_pressures = np.unique(targets)
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = (unique_pressures[1] - unique_pressures[0]).item()
PRESSURE_MIN = sorted_pressures[0].item()
PRESSURE_MAX = sorted_pressures[-1].item()

rounding_pre = (
    np.round((pred - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
rounding_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)

print("Pressure grid:", PRESSURE_MIN, PRESSURE_MAX, PRESSURE_STEP)
print("Rounded preds shape:", rounding_pre.shape)



## === cell 12
submission_file = pd.read_csv(SAMPLE_SUB_PATH)

submission_file["pressure"] = rounding_pre.reshape(-1)

assert len(submission_file) == rounding_pre.reshape(-1).shape[0], (
    len(submission_file),
    rounding_pre.reshape(-1).shape[0],
)

submission_file.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_file.shape)
print(submission_file.head())
