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

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from sklearn.model_selection import KFold

SEED = 2021
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

keras.backend.set_floatx("float32")

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_tensor_float_32_execution(
        False
    )  # avoid numeric drift
except Exception:
    pass




## === cell 1
DEBUG = False


def _resolve_input_dir() -> str:
    candidates = [
        "../input/ventilator-pressure-prediction",
        "/kaggle/input/ventilator-pressure-prediction",
        "/kaggle/data/ventilator-pressure-prediction",
        "../kaggle/input/ventilator-pressure-prediction",
        "../kaggle/data/ventilator-pressure-prediction",
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
            os.path.join(p, "test.csv")
        ):
            return p
    raise FileNotFoundError(
        "Could not find ventilator-pressure-prediction dataset folder. "
        "Tried: " + ", ".join(candidates)
    )


INPUT_DIR = _resolve_input_dir()

train = pd.read_csv(
    os.path.join(INPUT_DIR, "train.csv"),
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
    os.path.join(INPUT_DIR, "test.csv"),
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
submission = pd.read_csv(
    os.path.join(INPUT_DIR, "sample_submission.csv"),
    dtype={"id": "int32", "pressure": "float32"},
)

if DEBUG:
    train = train.iloc[: 80 * 1000].copy()




## === cell 2
def _cumsum_by_group(values: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    values = np.asarray(values, dtype=np.float32)
    group_ids = np.asarray(group_ids)

    start = np.empty(n, dtype=bool)
    start[0] = True
    start[1:] = group_ids[1:] != group_ids[:-1]

    cs = np.cumsum(values, dtype=np.float64)
    offsets = np.zeros(n, dtype=np.float64)
    start_idx = np.flatnonzero(start)
    prev_cs = cs[start_idx - 1]
    prev_cs[0] = 0.0
    offsets[start_idx] = prev_cs
    offsets = np.maximum.accumulate(offsets)

    return (cs - offsets).astype(np.float32)


def add_features_inplace(df: pd.DataFrame) -> pd.DataFrame:
    breath = df["breath_id"].to_numpy(copy=False)
    time_step = df["time_step"].to_numpy(dtype=np.float32, copy=False)
    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)

    area_incr = time_step * u_in
    df["area"] = _cumsum_by_group(area_incr, breath)
    df["u_in_cumsum"] = _cumsum_by_group(u_in, breath)

    n = breath.shape[0]
    start = np.empty(n, dtype=bool)
    start[0] = True
    start[1:] = breath[1:] != breath[:-1]

    u_in_lag = np.zeros(n, dtype=np.float32)
    u_in_lag[2:] = u_in[:-2]
    invalid = start.copy()
    invalid[1:] |= start[:-1]
    invalid[:2] = True
    u_in_lag[invalid] = 0.0
    df["u_in_lag"] = u_in_lag

    R = df["R"].to_numpy(copy=False)
    C = df["C"].to_numpy(copy=False)

    df["R_5"] = (R == 5).astype(np.int8, copy=False)
    df["R_20"] = (R == 20).astype(np.int8, copy=False)
    df["R_50"] = (R == 50).astype(np.int8, copy=False)

    df["C_10"] = (C == 10).astype(np.int8, copy=False)
    df["C_20"] = (C == 20).astype(np.int8, copy=False)
    df["C_50"] = (C == 50).astype(np.int8, copy=False)

    return df


train = add_features_inplace(train)
test = add_features_inplace(test)

train_cols_all = list(train.columns)
test_cols_all = list(test.columns)

for col in train_cols_all:
    if col not in test_cols_all and col != "pressure":
        test[col] = 0
for col in test_cols_all:
    if col not in train_cols_all:
        train[col] = 0

train_cols = [c for c in train.columns if c != "pressure"]
test = test[train_cols]
train = train[train_cols + ["pressure"]]




## === cell 3
targets = train[["pressure"]].to_numpy().reshape(-1, 80)

train.drop(["pressure", "id", "breath_id"], axis=1, inplace=True)
test = test.drop(["id", "breath_id"], axis=1)




## === cell 4
def robust_scale_fit_transform(X_train_2d: np.ndarray, X_test_2d: np.ndarray):
    X_train_2d = np.asarray(X_train_2d, dtype=np.float32, order="C")
    X_test_2d = np.asarray(X_test_2d, dtype=np.float32, order="C")

    q = np.quantile(X_train_2d, [0.25, 0.5, 0.75], axis=0).astype(
        np.float32, copy=False
    )
    q25, med, q75 = q[0], q[1], q[2]

    scale = (q75 - q25).astype(np.float32, copy=False)
    scale[scale == 0.0] = 1.0

    X_train_scaled = (X_train_2d - med) / scale
    X_test_scaled = (X_test_2d - med) / scale
    return X_train_scaled, X_test_scaled


train, test = robust_scale_fit_transform(
    train.to_numpy(copy=False), test.to_numpy(copy=False)
)




## === cell 5
train = np.asarray(train, dtype=np.float32, order="C").reshape(-1, 80, train.shape[-1])
test = np.asarray(test, dtype=np.float32, order="C").reshape(-1, 80, train.shape[-1])
targets = np.asarray(targets, dtype=np.float32, order="C")




## === cell 6
EPOCH = 200
BATCH_SIZE = 1024

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.__class__.__name__} (TPU not found: {e})")


def make_ds(
    X, y=None, batch_size=1024, training=False, cache=False, drop_remainder=False
):
    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.autotune.enabled = True
    except Exception:
        pass
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass

    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(X)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y))

    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(
            buffer_size=min(int(len(X)), 8192), seed=SEED, reshuffle_each_iteration=True
        )

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    if cache:
        ds = ds.cache()

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_ds(
    test,
    y=None,
    batch_size=BATCH_SIZE,
    training=False,
    cache=True,
    drop_remainder=False,
)

with strategy.scope():
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    test_preds = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train, targets), start=1):
        print("-" * 15, ">", f"Fold {fold}", "<", "-" * 15)
        X_train, X_valid = train[train_idx], train[valid_idx]
        y_train, y_valid = targets[train_idx], targets[valid_idx]

        drop_train = (len(X_train) % BATCH_SIZE) == 0

        train_ds = make_ds(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            training=True,
            cache=False,
            drop_remainder=drop_train,
        )
        valid_ds = make_ds(
            X_valid,
            y_valid,
            batch_size=BATCH_SIZE,
            training=False,
            cache=True,
            drop_remainder=False,
        )

        model = keras.models.Sequential(
            [
                keras.layers.Input(shape=train.shape[-2:]),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(300, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(250, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(150, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(100, return_sequences=True)
                ),
                keras.layers.Dense(50, activation="selu"),
                keras.layers.Dense(1),
            ]
        )

        decay_steps = int(max(1, 400 * ((len(train) * 0.8) / BATCH_SIZE)))
        lr_schedule = ExponentialDecay(
            initial_learning_rate=1e-3, decay_steps=decay_steps, decay_rate=1e-5
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(
            optimizer=optimizer, loss="mae", jit_compile=True, run_eagerly=False
        )

        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCH,
            verbose=2,
        )

        pred = model.predict(test_ds, verbose=0).reshape(-1)
        test_preds.append(pred)




## === cell 7
if len(test_preds) == 0:
    raise RuntimeError(
        "No fold predictions were generated; training likely failed before inference."
    )

preds = np.mean(np.vstack(test_preds), axis=0)  # shape: (n_rows,)

if len(preds) != len(submission):
    raise ValueError(
        f"Prediction length {len(preds)} does not match submission length {len(submission)}"
    )

test_u_out = (
    pd.read_csv(os.path.join(INPUT_DIR, "test.csv"), usecols=["u_out"])
    .to_numpy(dtype=np.int8, copy=False)
    .reshape(-1)
)
preds = preds.astype(np.float32, copy=False)
preds[test_u_out == 1] = 0.0

pressure_grid = np.sort(
    train_csv_unique := pd.read_csv(
        os.path.join(INPUT_DIR, "train.csv"), usecols=["pressure"]
    )["pressure"].unique()
).astype(np.float32, copy=False)
idx = np.searchsorted(pressure_grid, preds, side="left")
idx = np.clip(idx, 0, len(pressure_grid) - 1)
idx0 = np.clip(idx - 1, 0, len(pressure_grid) - 1)
choose_left = np.abs(preds - pressure_grid[idx0]) <= np.abs(preds - pressure_grid[idx])
preds = np.where(choose_left, pressure_grid[idx0], pressure_grid[idx]).astype(
    np.float32
)

submission["pressure"] = preds
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
