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
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "42"

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import random

random.seed(42)
np.random.seed(42)

import tensorflow as tf

tf.random.set_seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ["TF_NUM_INTRAOP_THREADS"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(os.environ["TF_NUM_INTEROP_THREADS"])
    )
except Exception:
    pass

tf.config.optimizer.set_jit(True)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

from tensorflow.keras import layers
from sklearn.model_selection import KFold



## === cell 1
from sklearn.preprocessing import RobustScaler

rc = RobustScaler()



## === cell 2
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 3
def dropCols_inplace_view(df, cols):
    return df.drop(columns=cols)




## === cell 4
def add_breath_features_fast(df, *, u_in_col="u_in", breath_len=80):
    u = df[u_in_col].to_numpy(dtype=np.float32, copy=False)
    n = u.shape[0]
    if n % breath_len != 0:
        raise ValueError(
            f"Unexpected row count {n} not divisible by breath_len={breath_len}"
        )

    u2 = u.reshape(-1, breath_len)
    lag = np.zeros_like(u2, dtype=np.float32)
    lag[:, 1:] = u2[:, :-1]
    diff = u2 - lag
    csum = np.cumsum(u2, axis=1, dtype=np.float32)

    df["u_in_lag1"] = lag.reshape(-1)
    df["diff_u_in1"] = diff.reshape(-1)
    df["u_in_cumsum"] = csum.reshape(-1)
    return df




## === cell 5
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
usecols_train = list(TRAIN_DTYPES.keys())

train_data = pd.read_csv(
    train_path,
    usecols=usecols_train,
    dtype=TRAIN_DTYPES,
    engine="c",
    memory_map=True,
)
train_data = add_breath_features_fast(train_data)



## === cell 6
cols_2_drop = ["id", "breath_id", "time_step"]



## === cell 7
Y_series = train_data["pressure"]  # view; no copy
train_df = dropCols_inplace_view(train_data, cols_2_drop + ["pressure"])



## === cell 8
del train_data
gc.collect()



## === cell 9
rc.fit(train_df)

Y = Y_series.to_numpy(dtype=np.float32, copy=False)

del Y_series
gc.collect()




## === cell 10
def build_model(input_n_features):
    model = tf.keras.Sequential()
    model.add(
        layers.Bidirectional(
            layers.LSTM(120, return_sequences=True),
            input_shape=[80, input_n_features],
        )
    )
    model.add(layers.Bidirectional(layers.LSTM(180, return_sequences=True)))
    model.add(
        layers.Bidirectional(layers.LSTM(280, dropout=0.2, return_sequences=True))
    )
    model.add(
        layers.Bidirectional(layers.LSTM(360, dropout=0.2, return_sequences=True))
    )
    model.add(
        layers.Bidirectional(layers.LSTM(500, dropout=0.2, return_sequences=True))
    )
    model.add(
        layers.Bidirectional(layers.LSTM(1000, dropout=0.2, return_sequences=True))
    )

    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.TimeDistributed(layers.Dense(1)))

    opt = tf.keras.optimizers.Adam()

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=["mae"],
        jit_compile=True,
    )

    model.steps_per_execution = 128
    return model




## === cell 11
callback1 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.9,
    patience=10,
    verbose=1,
)



## === cell 12
EPOCH = 50
BATCH_SIZE = 1024

model_path_dir_legacy = (
    "AdamPressurePreModel_local_best"  # legacy SavedModel directory (from older runs)
)
model_path_keras = (
    "AdamPressurePreModel_local_best.keras"  # new recommended single-file format
)

n_features = train_df.shape[-1]


def _load_existing_model_if_any():
    if os.path.isfile(model_path_keras):
        return tf.keras.models.load_model(model_path_keras)
    if os.path.exists(model_path_dir_legacy):
        return tf.keras.models.load_model(model_path_dir_legacy)
    return None


model = _load_existing_model_if_any()

if model is None:
    train_X = rc.transform(train_df)  # returns float64
    train_X = np.asarray(train_X, dtype=np.float32, order="C")
    train_X = train_X.reshape(-1, 80, n_features)
    Y_reshaped = Y.reshape(-1, 80, 1)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, valid_idx = next(kf.split(train_X))

    X_train, X_valid = train_X[train_idx], train_X[valid_idx]
    y_train, y_valid = Y_reshaped[train_idx], Y_reshaped[valid_idx]

    model = build_model(n_features)

    callback0 = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path_keras,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    options = tf.data.Options()
    options.experimental_deterministic = True

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .with_options(options)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    valid_ds = (
        tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
        .with_options(options)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    steps_per_epoch = int(np.ceil(len(X_train) / BATCH_SIZE))
    validation_steps = int(np.ceil(len(X_valid) / BATCH_SIZE))

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCH,
        callbacks=[callback0, callback1],
        verbose=2,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )

    model = tf.keras.models.load_model(model_path_keras)

    del (
        train_X,
        Y_reshaped,
        X_train,
        X_valid,
        y_train,
        y_valid,
        train_idx,
        valid_idx,
        history,
    )
    gc.collect()



## === cell 13
del train_df
gc.collect()



## === cell 14
TEST_DTYPES = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}
usecols_test = list(TEST_DTYPES.keys())

test_data = pd.read_csv(
    test_path,
    usecols=usecols_test,
    dtype=TEST_DTYPES,
    engine="c",
    memory_map=True,
)

test_data = add_breath_features_fast(test_data)

test_df = dropCols_inplace_view(test_data, cols_2_drop)

test_X = rc.transform(test_df)
test_X = np.asarray(test_X, dtype=np.float32, order="C")
test_X = test_X.reshape(-1, 80, test_X.shape[-1])

del test_data, test_df
gc.collect()



## === cell 15
models = [model]



## === cell 16
AUTOTUNE = tf.data.AUTOTUNE
options = tf.data.Options()
options.experimental_deterministic = True

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_X)
    .with_options(options)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

p = []
for m in models:
    try:
        m.steps_per_execution = 256
    except Exception:
        pass
    pred = m.predict(test_ds, verbose=1)
    p.append(pred.reshape(-1, 1))



## === cell 17
submission_files = [
    "../input/ventilatorpressure-submitted-files/sub.csv",
    "../input/ventilatorpressure-submitted-files/sub1.csv",
    "../input/ventilatorpressure-submitted-files/sub2.csv",
    "../input/ventilatorpressure-submitted-files/sub4.csv",
    "../input/ventilatorpressure-submitted-files/sub5.csv",
]

for sub_path in submission_files:
    if os.path.exists(sub_path):
        sub = pd.read_csv(sub_path)
        if "pressure" in sub.columns:
            p.append(sub["pressure"].values.reshape(-1, 1))



## === cell 18
predictions_table = np.concatenate(p, axis=-1)



## === cell 19
median_pre = np.median(predictions_table, axis=-1)
mean_pre = np.mean(predictions_table, axis=-1)



## === cell 20
unique_pressures = np.unique(Y.astype(np.float32, copy=False))
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = float((unique_pressures[1] - unique_pressures[0]).item())
PRESSURE_MIN = float(sorted_pressures[0].item())
PRESSURE_MAX = float(sorted_pressures[-1].item())



## === cell 21
rounding_pre = (
    np.round((median_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
rounding_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)



## === cell 22
submission_file = pd.read_csv(sample_sub)
if len(submission_file) != len(rounding_pre):
    raise ValueError(
        f"Prediction length mismatch: submission rows={len(submission_file)} vs preds={len(rounding_pre)}"
    )

submission_file["pressure"] = rounding_pre.astype(np.float32)
submission_file.to_csv("submission_mediam_round.csv", index=False)

print("Wrote submission_mediam_round.csv with shape:", submission_file.shape)
print(submission_file.head())
