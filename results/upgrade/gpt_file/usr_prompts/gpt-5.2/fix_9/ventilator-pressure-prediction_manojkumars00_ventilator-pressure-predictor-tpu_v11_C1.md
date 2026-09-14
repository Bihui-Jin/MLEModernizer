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
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 2) // 2)
    )
except Exception:
    pass

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

print("Python/TensorFlow:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())



## === cell 1
sc = StandardScaler()
rc = RobustScaler()



## === cell 2
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 3
def dropCols(df, cols):
    df = df.drop(columns=cols)
    return df




## === cell 4
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    b = df["breath_id"].to_numpy()
    u = df["u_in"].to_numpy()

    new_breath = np.empty_like(b, dtype=bool)
    new_breath[0] = True
    new_breath[1:] = b[1:] != b[:-1]

    u_lag1 = np.empty_like(u)
    u_lag1[0] = 0.0
    u_lag1[1:] = u[:-1]
    u_lag1[new_breath] = 0.0

    df["u_in_lag1"] = u_lag1
    df["diff_u_in1"] = u - u_lag1

    cs = np.cumsum(u, dtype=np.float64)
    cs_before_start = np.zeros_like(cs)
    start_idx = np.flatnonzero(new_breath)
    if start_idx.size:
        prev_vals = np.where(start_idx > 0, cs[start_idx - 1], 0.0)
        cs_before_start[start_idx] = prev_vals
        cs_before_start = np.maximum.accumulate(cs_before_start)

    df["u_in_cumsum"] = (cs - cs_before_start).astype(np.float32)
    return df




## === cell 5
usecols_train = ["id", "breath_id", "time_step", "u_in", "u_out", "R", "C", "pressure"]
dtype_map_train = {
    "id": "int32",
    "breath_id": "int32",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "R": "int16",
    "C": "int16",
    "pressure": "float32",
}
train_data = pd.read_csv(train_path, usecols=usecols_train, dtype=dtype_map_train)
train_data = add_features(train_data)



## === cell 6
cols_2_drop = ["id", "breath_id", "time_step"]



## === cell 7
train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")



## === cell 8
_ = train_df.shape



## === cell 9
X_np = train_df.to_numpy(copy=False)
rc.fit(X_np)
X_np = rc.transform(X_np)



## === cell 10
X_np = np.ascontiguousarray(X_np).reshape(-1, 80, X_np.shape[-1])
Y_np = np.ascontiguousarray(Y.to_numpy()).reshape(-1, 80, 1)

print("Train X/Y:", X_np.shape, Y_np.shape)




## === cell 11
def build_model():
    model = tf.keras.Sequential()

    model.add(
        layers.Bidirectional(
            layers.LSTM(120, return_sequences=True),
            input_shape=[80, X_np.shape[-1]],
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
        optimizer=opt, loss=tf.keras.losses.MeanAbsoluteError(), metrics=["mae"]
    )
    return model




## === cell 12
m_tmp = build_model()
m_tmp.summary()
del m_tmp



## === cell 13
callback1 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.9,
    patience=10,
    verbose=1,
)



## === cell 14
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Using TPU strategy")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print("TPU not available, using default strategy. Reason:", repr(e))

EPOCH = 500
BATCH_SIZE = 1024

trained_models = []

AUTOTUNE = tf.data.AUTOTUNE
STEPS_PER_EXECUTION = (
    128  # fewer Python roundtrips; same optimization target and results
)

data_opts = tf.data.Options()
data_opts.experimental_deterministic = True
try:
    data_opts.experimental_optimization.apply_default_optimizations = True
    data_opts.experimental_optimization.autotune = True
except Exception:
    pass

X_np = np.ascontiguousarray(X_np)
Y_np = np.ascontiguousarray(Y_np)


def _make_ds_xy_from_indices(X, y, idx_np):
    idx = tf.convert_to_tensor(idx_np, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices(idx).with_options(data_opts)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.map(
        lambda i: (tf.gather(X, i), tf.gather(y, i)), num_parallel_calls=AUTOTUNE
    )
    ds = ds.prefetch(AUTOTUNE)
    return ds


_COMPILE_KWARGS = dict(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanAbsoluteError(),
    metrics=["mae"],
    steps_per_execution=STEPS_PER_EXECUTION,
    jit_compile=False,
)

CKPT_DIR = "./"
os.makedirs(CKPT_DIR, exist_ok=True)

with strategy.scope():
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_np, Y_np), start=1):
        print("-" * 15, ">", f"Fold {fold}", "<", "-" * 15)

        tf.keras.backend.clear_session()

        ds_train = _make_ds_xy_from_indices(X_np, Y_np, train_idx)
        ds_valid = _make_ds_xy_from_indices(X_np, Y_np, valid_idx)

        ckpt_path = os.path.join(CKPT_DIR, f"AdamPressurePreModel{fold}.keras")

        if os.path.exists(ckpt_path):
            best_model = tf.keras.models.load_model(ckpt_path, compile=False)
            best_model.compile(**_COMPILE_KWARGS)
            trained_models.append(best_model)
            print(f"Loaded existing checkpoint: {ckpt_path}")
            print("\n\n")
            continue

        model = build_model()
        model.compile(**_COMPILE_KWARGS)

        callback0 = tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss", save_best_only=True, verbose=0
        )

        _ = model.fit(
            ds_train,
            validation_data=ds_valid,
            epochs=EPOCH,
            callbacks=[callback0, callback1],
            verbose=0,
        )

        best_model = tf.keras.models.load_model(ckpt_path, compile=False)
        best_model.compile(**_COMPILE_KWARGS)
        trained_models.append(best_model)

        print("\n\n")



## === cell 15
usecols_test = ["id", "breath_id", "time_step", "u_in", "u_out", "R", "C"]
dtype_map_test = {
    "id": "int32",
    "breath_id": "int32",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "R": "int16",
    "C": "int16",
}
test_data = pd.read_csv(test_path, usecols=usecols_test, dtype=dtype_map_test)

test_data = add_features(test_data)

test_data = dropCols(test_data, cols_2_drop)
X_test = rc.transform(test_data.to_numpy(copy=False))
X_test = np.ascontiguousarray(X_test).reshape(-1, 80, X_test.shape[-1])

print("Test X:", X_test.shape)



## === cell 16
ds_test = (
    tf.data.Dataset.from_tensor_slices(X_test)
    .with_options(data_opts)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

p = []
for m in trained_models:
    p.append(m.predict(ds_test, verbose=0).reshape(-1, 1))

predictions_table = np.concatenate(p, axis=-1)

mean_pre = np.mean(predictions_table, axis=-1)
median_pre = np.median(predictions_table, axis=-1)

print("Pred shapes:", mean_pre.shape, median_pre.shape)



## === cell 17
unique_pressures = np.unique(Y_np)
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = (unique_pressures[1] - unique_pressures[0]).item()
PRESSURE_MIN = sorted_pressures[0].item()
PRESSURE_MAX = sorted_pressures[-1].item()

print("Pressure grid:", PRESSURE_STEP, PRESSURE_MIN, PRESSURE_MAX)



## === cell 18
rounding_pre = (
    np.round((median_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
rounding_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)

print("Rounded pred shape:", rounding_pre.shape)



## === cell 19
submission_file = pd.read_csv(sample_sub)
submission_file["pressure"] = rounding_pre.astype(np.float32)
submission_file.to_csv("submission.csv", index=False)

print(submission_file.head())
print("Wrote submission.csv with shape:", submission_file.shape)
