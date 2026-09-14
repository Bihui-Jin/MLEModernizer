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
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## === cell 1
rb = RobustScaler()



## === cell 2
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

train = pd.read_csv(
    TRAIN_PATH,
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
    TEST_PATH,
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

print(train.shape, test.shape)
print(train.columns)



## === cell 3
pass



## === cell 4
train.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")
test.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

g_tr = train.groupby("breath_id", sort=False)
train["u_in_lag1"] = g_tr["u_in"].shift(1).fillna(0.0)
train["u_in_diff1"] = train["u_in"] - train["u_in_lag1"]
train["u_in_cumsum"] = g_tr["u_in"].cumsum()

g_te = test.groupby("breath_id", sort=False)
test["u_in_lag1"] = g_te["u_in"].shift(1).fillna(0.0)
test["u_in_diff1"] = test["u_in"] - test["u_in_lag1"]
test["u_in_cumsum"] = g_te["u_in"].cumsum()



## === cell 5
print(train.head())
print(train.dtypes)



## === cell 6
targets = train["pressure"].to_numpy().reshape(-1, 80, 1)

test_id = test["id"].copy()

train.drop(columns=["id", "breath_id", "pressure", "time_step"], inplace=True)
test.drop(columns=["id", "breath_id", "time_step"], inplace=True)

print("Train features:", train.shape, "Test features:", test.shape)
print("Targets:", targets.shape)



## === cell 7
rb.fit(train)

train_new = rb.transform(train).astype(np.float32, copy=False)
test_new = rb.transform(test).astype(np.float32, copy=False)

train_re = train_new.reshape(-1, 80, train_new.shape[-1])
test_re = test_new.reshape(-1, 80, test_new.shape[-1])

print("train_re:", train_re.shape, "test_re:", test_re.shape)



## === cell 8
pass




## === cell 9
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
    )
    return model




## === cell 10
model_tmp = build_model()
print(model_tmp.name, "params:", model_tmp.count_params())



## === cell 11
pass



## === cell 12
EPOCH = 260
BATCH_SIZE = 512

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # auto-detect
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except Exception as e:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        strategy = tf.distribute.MirroredStrategy()
        print("Running on GPU(s) with MirroredStrategy:", gpus)
    else:
        strategy = tf.distribute.get_strategy()
        print("Running on CPU. TPU not available:", repr(e))




## === cell 13
def exp_decay(epoch):
    initial_lrate = 0.001
    k = 0.006
    lrate = initial_lrate * math.exp(-k * epoch)
    print(f"Learning rate : {lrate}")
    return lrate


lr_schedular = tf.keras.callbacks.LearningRateScheduler(exp_decay)



## === cell 14
Ensembles = [
    "../input/latest-ensembles/submission.csv",
    "../input/latest-ensembles/submission1.csv",
    "../input/latest-ensembles/submission2.csv",
]

test_preds_fast = []
for predicted in Ensembles:
    if os.path.exists(predicted):
        submission_file_ext = pd.read_csv(predicted)
        if "pressure" in submission_file_ext.columns and len(
            submission_file_ext
        ) == len(test_id):
            test_preds_fast.append(
                submission_file_ext["pressure"].to_numpy().reshape(-1, 1)
            )
            print("Loaded external ensemble (fast path):", predicted)
    else:
        print("Missing external ensemble (fast path skipped):", predicted)

if len(test_preds_fast) > 0:
    unique_pressures = np.unique(targets)
    sorted_pressures = np.sort(unique_pressures)

    PRESSURE_STEP = (unique_pressures[1] - unique_pressures[0]).item()
    PRESSURE_MIN = sorted_pressures[0].item()
    PRESSURE_MAX = sorted_pressures[-1].item()

    predictions_median_fast = np.concatenate(test_preds_fast, axis=-1)
    median_pre_fast = np.median(predictions_median_fast, axis=-1)

    rounding_pre_fast = (
        np.round((median_pre_fast - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
        + PRESSURE_MIN
    )
    clipped_pre_fast = np.clip(rounding_pre_fast, PRESSURE_MIN, PRESSURE_MAX)

    submission_file_out = pd.read_csv(SAMPLE_SUB_PATH)
    if "id" in submission_file_out.columns and len(submission_file_out) == len(test_id):
        if not np.array_equal(submission_file_out["id"].values, test_id.values):
            submission_file_out = pd.DataFrame(
                {"id": test_id.values, "pressure": np.zeros(len(test_id))}
            )
    submission_file_out["pressure"] = clipped_pre_fast.reshape(
        -1,
    )
    submission_file_out.to_csv("submission.csv", index=False)
    print("Wrote submission.csv (fast path) with shape:", submission_file_out.shape)

    raise SystemExit(0)



## === cell 15
reduce_lr = ReduceLROnPlateau(monitor="val_loss", verbose=1, factor=0.87, patience=8)

kf = KFold(n_splits=7, shuffle=True, random_state=SEED)
trained_model_paths = []

with strategy.scope():
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_re, targets), start=1):
        print("-" * 30, ">", f"Fold {fold}", "<", "-" * 30)
        X_train, X_valid = train_re[train_idx], train_re[valid_idx]
        y_train, y_valid = targets[train_idx], targets[valid_idx]

        model = build_model()
        ckpt_path = f"NewModel{fold}.keras"  # use native Keras format for TF/Keras 2.15+ stability
        call_back = tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, verbose=1, monitor="val_loss", save_best_only=True
        )

        _ = model.fit(
            X_train,
            y_train,
            validation_data=(X_valid, y_valid),
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            callbacks=[call_back, reduce_lr],
            verbose=2,
        )
        trained_model_paths.append(ckpt_path)

print("Saved models:", trained_model_paths)



## === cell 16
pass



## === cell 17
unique_pressures = np.unique(targets)
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = (unique_pressures[1] - unique_pressures[0]).item()
PRESSURE_MIN = sorted_pressures[0].item()
PRESSURE_MAX = sorted_pressures[-1].item()

print("PRESSURE_STEP, MIN, MAX:", PRESSURE_STEP, PRESSURE_MIN, PRESSURE_MAX)



## === cell 18
PRESSURE_STEP, PRESSURE_MIN, PRESSURE_MAX



## === cell 19
models_paths = [
    "../input/kfolds/OldModel1.h5",
    "../input/kfolds/OldModel2.h5",
    "../input/kfolds/OldModel3.h5",
    "../input/kfolds/OldModel4.h5",
    "../input/kfolds/OldModel5.h5",
    "../input/kfolds/OldModel6.h5",
    "../input/kfolds/OldModel7.h5",
    "../input/kfolds/OldModel8.h5",
    "../input/kfolds/OldModel9.h5",
    "../input/latest-2fold/Model16049.h5",
    "../input/latest-2fold/Model16275.h5",
    "../input/kfolds/Model16277.h5",
    "../input/kfolds/Model16179.h5",
]



## === cell 20
pass



## === cell 21
pass



## === cell 22
Ensembles = [
    "../input/latest-ensembles/submission.csv",
    "../input/latest-ensembles/submission1.csv",
    "../input/latest-ensembles/submission2.csv",
]



## === cell 23
test_preds = []

for predicted in Ensembles:
    if os.path.exists(predicted):
        submission_file = pd.read_csv(predicted)
        if "pressure" in submission_file.columns and len(submission_file) == len(
            test_id
        ):
            test_preds.append(submission_file["pressure"].values.reshape(-1, 1))
            print("Loaded external ensemble:", predicted)
    else:
        print("Missing external ensemble (skipped):", predicted)

if len(test_preds) == 0:
    candidate_paths = [p for p in trained_model_paths if os.path.exists(p)]

    if len(candidate_paths) == 0:
        candidate_paths = [p for p in models_paths if os.path.exists(p)]

    if len(candidate_paths) == 0:
        raise FileNotFoundError(
            "No models available to create predictions (no trained models and no provided model files found)."
        )

    for mp in candidate_paths:
        print("Predicting with model:", mp)
        m = tf.keras.models.load_model(mp, compile=False)
        pred = m.predict(test_re, batch_size=BATCH_SIZE, verbose=0).reshape(-1, 1)
        test_preds.append(pred)

print("Num prediction sources:", len(test_preds))



## === cell 24
predictions_median = np.concatenate(test_preds, axis=-1)
print("predictions_median:", predictions_median.shape)



## === cell 25
predictions_median.shape



## === cell 26
median_pre = np.median(predictions_median, axis=-1)

rounding_pre = (
    np.round((median_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)

print(
    "clipped_pre:", clipped_pre.shape, "min/max:", clipped_pre.min(), clipped_pre.max()
)



## === cell 27
pass



## === cell 28
clipped_pre.shape



## === cell 29
submission_file = pd.read_csv(SAMPLE_SUB_PATH)

if "id" in submission_file.columns and len(submission_file) == len(test_id):
    if not np.array_equal(submission_file["id"].values, test_id.values):
        submission_file = pd.DataFrame(
            {"id": test_id.values, "pressure": np.zeros(len(test_id))}
        )

submission_file["pressure"] = clipped_pre.reshape(
    -1,
)
submission_file.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_file.shape)



## === cell 30
submission_file.head()
