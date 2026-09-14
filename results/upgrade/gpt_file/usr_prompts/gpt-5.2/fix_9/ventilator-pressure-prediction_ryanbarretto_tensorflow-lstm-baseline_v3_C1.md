# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
lightgbm==4.6.0
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

# 5. Target score

0.4333

# 6. Current score

17.89756

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 17.89756) has done: 'The timeout is dominated by training compute: 200 epochs of a large 3×BiLSTM model on ~68k breaths. We keep the exact same model/optimizer/loss and number of epochs, but remove input-pipeline overhead and expensive XLA recompilation triggers by (1) switching to `TFRecord`-like in-memory feeding via `keras.utils.Sequence` with deterministic shuffling, and (2) using fixed shapes by dropping the last partial batch (no data loss in semantics because we compensate with deterministic padding-free batching over the full dataset). We also reduce pandas/groupby overhead and memory churn by using vectorized `shift`/`cumsum` with `sort=False` already and by avoiding unnecessary DataFrame copies. These changes preserve the algorithm and training loop semantics while cutting per-epoch overhead and stabilizing compilation, which is typically what pushes this notebook over 600s.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import pandas as pd
import numpy as np

from IPython.display import display

import tensorflow as tf
from tensorflow import keras

import lightgbm  # noqa: F401
import optuna  # noqa: F401
from sklearn.model_selection import train_test_split, GroupKFold  # noqa: F401
from sklearn.metrics import mean_absolute_error  # noqa: F401

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF:", tf.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

DTYPES_TRAIN = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
DTYPES_TEST = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

train_df = pd.read_csv(train_path, dtype=DTYPES_TRAIN, low_memory=False)
test_df = pd.read_csv(test_path, dtype=DTYPES_TEST, low_memory=False)
submission = pd.read_csv(sub_path)

display(train_df.head())
display(test_df.head())
display(submission.head())

print(
    "train shape:",
    train_df.shape,
    " test shape:",
    test_df.shape,
    " submission shape:",
    submission.shape,
)




## === cell 2
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("breath_id", sort=False, observed=True)

    df["u_in_cumsum"] = g["u_in"].cumsum().astype(np.float32, copy=False)
    df["u_in_lag"] = g["u_in"].shift(2).fillna(0.0).astype(np.float32, copy=False)
    return df


train_df = add_features(train_df)
test_df = add_features(test_df)




## === cell 3
FEATURE_COLS = ["R", "C", "time_step", "u_in", "u_in_cumsum", "u_in_lag"]


def to_breath_tensor(df: pd.DataFrame, feature_cols, target_col=None):
    breath_ids = df["breath_id"].to_numpy(copy=False)

    change = np.empty(breath_ids.shape[0], dtype=bool)
    change[0] = True
    np.not_equal(breath_ids[1:], breath_ids[:-1], out=change[1:])
    start_idx = np.flatnonzero(change)
    n_breaths = start_idx.shape[0]

    end_idx = np.empty_like(start_idx)
    end_idx[:-1] = start_idx[1:]
    end_idx[-1] = breath_ids.shape[0]
    counts = end_idx - start_idx

    if counts.min() != 80 or counts.max() != 80:
        raise ValueError(
            f"Expected 80 rows per breath, got min={counts.min()} max={counts.max()}"
        )

    breaths = breath_ids[start_idx]
    n_feat = len(feature_cols)

    X_flat = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    X = np.ascontiguousarray(X_flat.reshape(n_breaths, 80, n_feat))

    if target_col is None:
        return X, breaths, df

    y_flat = df[target_col].to_numpy(dtype=np.float32, copy=False)
    y = np.ascontiguousarray(y_flat.reshape(n_breaths, 80))
    return X, y, breaths, df


X_train, y_train, train_breaths, train_sorted_df = to_breath_tensor(
    train_df, FEATURE_COLS, target_col="pressure"
)
X_test, test_breaths, test_sorted_df = to_breath_tensor(
    test_df, FEATURE_COLS, target_col=None
)

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test:", X_test.shape)
assert X_train.shape[0] == y_train.shape[0]
assert X_train.shape[1] == 80 and X_test.shape[1] == 80




## === cell 4
BATCH_SIZE = 1024
EPOCHS = 200
VAL_FRAC = 0.2

n = len(X_train)
n_val = int(round(n * VAL_FRAC))
n_train = n - n_val

idx = np.arange(n)
rng = np.random.default_rng(SEED)
rng.shuffle(idx)

train_idx = idx[:n_train]
val_idx = idx[n_train:]

X_tr = X_train[train_idx]
y_tr = y_train[train_idx]
X_va = X_train[val_idx]
y_va = y_train[val_idx]

if not X_tr.flags["C_CONTIGUOUS"]:
    X_tr = np.ascontiguousarray(X_tr)
if not y_tr.flags["C_CONTIGUOUS"]:
    y_tr = np.ascontiguousarray(y_tr)
if not X_va.flags["C_CONTIGUOUS"]:
    X_va = np.ascontiguousarray(X_va)
if not y_va.flags["C_CONTIGUOUS"]:
    y_va = np.ascontiguousarray(y_va)

steps_per_epoch = int(np.ceil(n_train / BATCH_SIZE))
steps_per_epoch = max(1, steps_per_epoch)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=200 * steps_per_epoch,
    decay_rate=1e-5,
)

model = keras.models.Sequential(
    [
        keras.layers.Input(shape=(80, len(FEATURE_COLS))),
        keras.layers.Bidirectional(keras.layers.LSTM(200, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(150, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True)),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(1),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="mae",
    jit_compile=True,
    run_eagerly=False,
)


class BreathSequence(keras.utils.Sequence):
    def __init__(self, X, y, batch_size, shuffle, seed):
        self.X = X
        self.y = y
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.n = len(X)
        self.index = np.arange(self.n, dtype=np.int32)
        self.epoch = 0
        self.full_batches = self.n // self.batch_size

    def __len__(self):
        return self.full_batches

    def on_epoch_end(self):
        if self.shuffle:
            r = np.random.default_rng(self.seed + self.epoch)
            r.shuffle(self.index)
        self.epoch += 1

    def __getitem__(self, i):
        sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
        ids = self.index[sl]
        return self.X[ids], self.y[ids]


train_seq = BreathSequence(X_tr, y_tr, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
val_seq = BreathSequence(X_va, y_va, batch_size=BATCH_SIZE, shuffle=False, seed=SEED)

_ = model(tf.convert_to_tensor(X_tr[: min(len(X_tr), 2)]), training=False)

history = model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=EPOCHS,
    verbose=2,
    workers=1,
    use_multiprocessing=False,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2683458531.py in <cell line: 0>()
     93 _ = model(tf.convert_to_tensor(X_tr[: min(len(X_tr), 2)]), training=False)
     94 
---> 95 history = model.fit(
     96     train_seq,
     97     validation_data=val_seq,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 5
pred = model.predict(X_test, batch_size=1024, verbose=1).squeeze(-1)  # (n_breaths, 80)
pred_flat = pred.reshape(-1).astype(np.float32, copy=False)

test_sorted_df = test_sorted_df.copy(deep=False)
test_sorted_df["pressure"] = pred_flat

out = test_sorted_df[["id", "pressure"]].sort_values("id").reset_index(drop=True)

assert len(out) == len(submission), (len(out), len(submission))

out.to_csv("submission.csv", index=False)
print(out.head())
print("Wrote submission.csv with shape:", out.shape)
