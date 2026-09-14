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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
tqdm==4.67.1

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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import random
import tensorflow as tf

random.seed(7)
np.random.seed(7)
tf.random.set_seed(7)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    n = max(1, (os.cpu_count() or 2) // 2)
    tf.config.threading.set_intra_op_parallelism_threads(n)
    tf.config.threading.set_inter_op_parallelism_threads(n)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense
import tensorflow.keras

print("TensorFlow:", tf.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)



## === cell 1
directory = "/kaggle/input/ventilator-pressure-prediction"

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

train = pd.read_csv(
    os.path.join(directory, "train.csv"),
    dtype=dtypes_train,
    usecols=list(dtypes_train.keys()),
)
test = pd.read_csv(
    os.path.join(directory, "test.csv"),
    dtype=dtypes_test,
    usecols=list(dtypes_test.keys()),
)
sub = pd.read_csv(
    os.path.join(directory, "sample_submission.csv"),
    dtype={"id": "int32", "pressure": "float32"},
)

print(train.shape, test.shape, sub.shape)
print(train.head())



## === cell 2
FEATURES = ["R", "C", "time_step", "u_in", "u_out"]
sample_length = 80


def _assert_ordered_by_breath_and_time_fast(df: pd.DataFrame, seq_len: int):
    b = df["breath_id"].to_numpy()
    t = df["time_step"].to_numpy()

    ok = np.all((b[1:] > b[:-1]) | ((b[1:] == b[:-1]) & (t[1:] >= t[:-1])))
    if not ok:
        raise ValueError(
            "Input df is not ordered by (breath_id, time_step); cannot use fast reshape safely."
        )

    change_idx = np.flatnonzero(b[1:] != b[:-1]) + 1
    bounds = np.concatenate(([0], change_idx, [len(b)]))
    lengths = np.diff(bounds)
    if not np.all(lengths == seq_len):
        raise ValueError(
            f"Unexpected breath length encountered; expected all breaths to have {seq_len} rows."
        )


def to_breath_sequences(df: pd.DataFrame, features, seq_len=80, y_col=None):
    _assert_ordered_by_breath_and_time_fast(df, seq_len)

    x = df[features].to_numpy(dtype=np.float32, copy=False)
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x, dtype=np.float32)

    n = (len(df) // seq_len) * seq_len
    x = x[:n].reshape(-1, seq_len, len(features))

    if y_col is None:
        return x, None

    y = df[y_col].to_numpy(dtype=np.float32, copy=False)[:n]
    if not y.flags["C_CONTIGUOUS"]:
        y = np.ascontiguousarray(y, dtype=np.float32)
    y = y.reshape(-1, seq_len, 1)
    return x, y


try:
    _assert_ordered_by_breath_and_time_fast(train, sample_length)
except Exception:
    train = train.sort_values(["breath_id", "time_step"], kind="quicksort").reset_index(
        drop=True
    )

try:
    _assert_ordered_by_breath_and_time_fast(test, sample_length)
except Exception:
    test = test.sort_values(["breath_id", "time_step"], kind="quicksort").reset_index(
        drop=True
    )

inputs, targets = to_breath_sequences(
    train, FEATURES, seq_len=sample_length, y_col="pressure"
)
test_inputs, _ = to_breath_sequences(test, FEATURES, seq_len=sample_length, y_col=None)

print("train inputs:", inputs.shape, "train targets:", targets.shape)
print("test inputs:", test_inputs.shape)

assert inputs.shape[1] == 80 and test_inputs.shape[1] == 80
assert inputs.shape[2] == 5 and test_inputs.shape[2] == 5




## === cell 3
def lstm_model():
    model = Sequential()
    model.add(Input(shape=(inputs.shape[1], inputs.shape[2])))
    model.add(LSTM(320, return_sequences=True))
    model.add(LSTM(320, return_sequences=True))
    model.add(Dense(160, activation="relu"))
    model.add(Dense(1, kernel_initializer="normal"))
    model.compile(
        loss="mae",
        optimizer="adam",
        metrics=[tensorflow.keras.metrics.RootMeanSquaredError()],
    )
    return model


def bi_lstm_model():
    model = Sequential()
    model.add(Input(shape=(inputs.shape[1], inputs.shape[2])))
    model.add(tf.keras.layers.Bidirectional(LSTM(640, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(LSTM(320, return_sequences=True)))
    model.add(LSTM(320, return_sequences=True))
    model.add(LSTM(320, return_sequences=True))
    model.add(Dense(320, activation="relu"))
    model.add(Dense(160, activation="relu"))
    model.add(Dense(1, kernel_initializer="normal"))
    model.compile(
        loss="mae",
        optimizer="adam",
        metrics=[tensorflow.keras.metrics.RootMeanSquaredError()],
    )
    return model


def cnn_lstm_model():
    model = Sequential()
    model.add(Input(shape=(inputs.shape[1], inputs.shape[2])))
    model.add(
        Conv1D(
            filters=320,
            kernel_size=3,
            strides=1,
            padding="causal",
            activation="relu",
        )
    )
    model.add(LSTM(320, return_sequences=True, activation="tanh"))
    model.add(LSTM(320, return_sequences=True, activation="tanh"))
    model.add(Dense(160, activation="relu"))
    model.add(Dense(1, kernel_initializer="normal"))

    model.compile(
        loss="mae",
        optimizer=tensorflow.keras.optimizers.Adam(),
        metrics=[tensorflow.keras.metrics.RootMeanSquaredError()],
        steps_per_execution=4096,
    )
    return model




## === cell 4
model = cnn_lstm_model()
model.summary()

BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

opts = tf.data.Options()
opts.experimental_deterministic = True
opts.experimental_optimization.apply_default_optimizations = True
opts.experimental_optimization.map_parallelization = False

train_ds = tf.data.Dataset.from_tensor_slices((inputs, targets)).with_options(opts)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(
    buffer_size=min(len(inputs), 8192), seed=7, reshuffle_each_iteration=True
)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)

try:
    train_ds = train_ds.apply(
        tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=AUTOTUNE)
    )
except Exception:
    train_ds = train_ds.prefetch(AUTOTUNE)

history = model.fit(
    train_ds,
    epochs=500,
    verbose=0,
)



## === cell 5
opts = tf.data.Options()
opts.experimental_deterministic = True
opts.experimental_optimization.apply_default_optimizations = True

test_ds = tf.data.Dataset.from_tensor_slices(test_inputs).with_options(opts)
test_ds = test_ds.cache()
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)

try:
    test_ds = test_ds.apply(
        tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=AUTOTUNE)
    )
except Exception:
    test_ds = test_ds.prefetch(AUTOTUNE)

pred_seq = model.predict(test_ds, verbose=0)  # (n_breaths, 80, 1)
print("pred_seq shape:", pred_seq.shape)

sub_array = pred_seq.reshape(-1)  # numpy array
print("num predictions:", sub_array.shape[0], "num test rows:", len(test))
assert sub_array.shape[0] == len(test)



## === cell 6
test_ordered = test
assert sub_array.shape[0] == len(test_ordered)

submission = pd.DataFrame(
    {
        "id": test_ordered["id"].to_numpy(dtype=np.int64, copy=False),
        "pressure": sub_array.astype(np.float32, copy=False),
    }
)

submission = submission.sort_values("id", kind="quicksort").reset_index(drop=True)

assert submission.shape[0] == sub.shape[0]
assert list(submission.columns) == ["id", "pressure"]

print(submission.head())



## === cell 7
out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", submission.shape)
print(submission.tail())
