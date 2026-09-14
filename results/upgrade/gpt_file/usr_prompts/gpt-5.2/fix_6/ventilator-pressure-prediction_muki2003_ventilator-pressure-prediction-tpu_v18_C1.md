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
import math
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF version:", tf.__version__)




## === cell 1
def get_distribution_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # raises if no TPU
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        print("TPU initialized")
        return tf.distribute.TPUStrategy(tpu)
    except Exception as e:
        print("TPU not available, using default strategy. Reason:", repr(e))
        return tf.distribute.get_strategy()


strategy = get_distribution_strategy()
print("Num replicas in sync:", strategy.num_replicas_in_sync)




## === cell 2
TRAIN_PATH = "/kaggle/input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "/kaggle/input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"

TRAIN_DTYPES = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
TEST_DTYPES = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

train = pd.read_csv(TRAIN_PATH, dtype=TRAIN_DTYPES)
test = pd.read_csv(TEST_PATH, dtype=TEST_DTYPES)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH, dtype={"id": "int32", "pressure": "float32"})

print(train.shape, test.shape, sample_sub.shape)
train.head()




## === cell 3
g_train = train.groupby("breath_id", sort=False)["u_in"]
train_u_in_lag1 = g_train.shift(1)
train["u_in_lag1"] = train_u_in_lag1.fillna(0).astype(np.float32, copy=False)
train["u_in_diff1"] = (train["u_in"] - train["u_in_lag1"]).astype(
    np.float32, copy=False
)
train["u_in_cumsum"] = g_train.cumsum().astype(np.float32, copy=False)

g_test = test.groupby("breath_id", sort=False)["u_in"]
test_u_in_lag1 = g_test.shift(1)
test["u_in_lag1"] = test_u_in_lag1.fillna(0).astype(np.float32, copy=False)
test["u_in_diff1"] = (test["u_in"] - test["u_in_lag1"]).astype(np.float32, copy=False)
test["u_in_cumsum"] = g_test.cumsum().astype(np.float32, copy=False)




## === cell 4
targets = train["pressure"].to_numpy().reshape(-1, 80, 1)

test_id = test["id"].copy()

train.drop(columns=["id", "breath_id", "pressure", "time_step"], inplace=True)
test.drop(columns=["id", "breath_id", "time_step"], inplace=True)

print(
    "Train features:",
    train.shape,
    "Targets:",
    targets.shape,
    "Test features:",
    test.shape,
)




## === cell 5
rb = RobustScaler()
rb.fit(train)

train_new = rb.transform(train).astype(np.float32, copy=False)
test_new = rb.transform(test).astype(np.float32, copy=False)

train_re = train_new.reshape(-1, 80, train_new.shape[-1])
test_re = test_new.reshape(-1, 80, test_new.shape[-1])

print("train_re:", train_re.shape, "test_re:", test_re.shape)




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

    x = layers.Concatenate(axis=2)([x5, z2, z3, z4, z5])
    x = layers.Dense(units=64, activation="relu")(x)
    x_output = layers.Dense(units=1)(x)

    model = tf.keras.Model(inputs=x_input, outputs=x_output, name="Base_Model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanAbsoluteError(),
        steps_per_execution=16,
    )
    return model




## === cell 7
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

EPOCH = 260
BATCH_SIZE = 512

reduce_lr = ReduceLROnPlateau(monitor="val_loss", verbose=1, factor=0.87, patience=8)

N_SPLITS = 7
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

saved_model_paths = []

AUTOTUNE = tf.data.AUTOTUNE

DATASET_OPTIONS = tf.data.Options()
DATASET_OPTIONS.experimental_distribute.auto_shard_policy = (
    tf.data.experimental.AutoShardPolicy.DATA
)
try:
    DATASET_OPTIONS.experimental_deterministic = True
except Exception:
    pass

X_all = train_re
y_all = targets


def make_dataset_from_indices(indices, training=False):
    X = X_all[indices]
    y = y_all[indices]
    ds = tf.data.Dataset.from_tensor_slices((X, y)).with_options(DATASET_OPTIONS)
    if training:
        buf = min(len(indices), 8192)
        ds = ds.shuffle(buffer_size=buf, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


with strategy.scope():
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_all, y_all), start=1):
        print("-" * 30, ">", f"Fold {fold}", "<", "-" * 30)

        model = build_model()

        ckpt_path = f"NewModel{fold}.weights.h5"
        call_back = ModelCheckpoint(
            ckpt_path,
            verbose=1,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )

        train_ds = make_dataset_from_indices(train_idx, training=True)
        valid_ds = make_dataset_from_indices(valid_idx, training=False)

        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCH,
            callbacks=[call_back, reduce_lr],
            verbose=2,
        )
        saved_model_paths.append(ckpt_path)

print("Saved models:", saved_model_paths)




## === cell 8
unique_pressures = np.unique(targets)
sorted_pressures = np.sort(unique_pressures)
PRESSURE_STEP = (unique_pressures[1] - unique_pressures[0]).item()
PRESSURE_MIN = sorted_pressures[0].item()
PRESSURE_MAX = sorted_pressures[-1].item()

print("Pressure min/max/step:", PRESSURE_MIN, PRESSURE_MAX, PRESSURE_STEP)


def round_to_nearest_pressure(x):
    x = np.clip(x, PRESSURE_MIN, PRESSURE_MAX)
    return np.round((x - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN




## === cell 9
preds = np.zeros((test_re.shape[0], 80, 1), dtype=np.float32)

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_re)
    .with_options(DATASET_OPTIONS)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

with strategy.scope():
    for path in saved_model_paths:
        m = build_model()
        m.load_weights(path)
        fold_pred = m.predict(test_ds, verbose=1)
        preds += fold_pred.astype(np.float32, copy=False)

preds /= len(saved_model_paths)

preds_flat = preds.reshape(-1)
preds_flat = round_to_nearest_pressure(preds_flat)

print("Pred shape:", preds.shape, "Flat:", preds_flat.shape)




## === cell 10
submission = pd.DataFrame({"id": test_id.values, "pressure": preds_flat})

submission = submission.sort_values("id").reset_index(drop=True)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print(submission.tail())
