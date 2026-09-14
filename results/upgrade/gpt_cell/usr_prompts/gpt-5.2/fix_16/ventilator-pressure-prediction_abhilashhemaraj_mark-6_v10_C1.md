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
import numpy as np
import pandas as pd

import os
import random
import time

from tqdm.auto import tqdm

tqdm.pandas()

import matplotlib.pyplot as plt
import seaborn as sns

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
random.seed(7)
np.random.seed(7)



## === cell 1
directory = "../input/ventilator-pressure-prediction"

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

train = pd.read_csv(os.path.join(directory, "train.csv"), dtype=TRAIN_DTYPES)
test = pd.read_csv(os.path.join(directory, "test.csv"), dtype=TEST_DTYPES)
sub = pd.read_csv(
    os.path.join(directory, "sample_submission.csv"),
    dtype={"id": "int32", "pressure": "float32"},
)



## === cell 2
_ = train  # keep cell structure; no side effects



## === cell 3
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("PYTHONHASHSEED", "7")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

tf.keras.utils.set_random_seed(7)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## === cell 4
FEATURE_COLS = ["R", "C", "time_step", "u_in", "u_out"]
TARGET_COL = "pressure"
sample_length = 80


def make_breath_sequences_fast(
    df: pd.DataFrame, feature_cols, target_col=None, seq_len=80
):
    n_rows = len(df)
    if n_rows % seq_len != 0:
        raise ValueError(f"Row count {n_rows} not divisible by seq_len={seq_len}")

    breath_ids = df["breath_id"].to_numpy(copy=False)
    n_breaths = n_rows // seq_len

    breath_block = breath_ids.reshape(n_breaths, seq_len)

    if not np.all(breath_block[:, 0:1] == breath_block):
        df_sorted = df.sort_values(
            ["breath_id", "time_step"], kind="mergesort"
        ).reset_index(drop=True)
        breath_ids2 = df_sorted["breath_id"].to_numpy(copy=False)
        breath_block2 = breath_ids2.reshape(n_breaths, seq_len)
        if not np.all(breath_block2[:, 0:1] == breath_block2):
            raise ValueError(
                "Found breaths that are not contiguous blocks of length seq_len even after sorting."
            )
        df_use = df_sorted
    else:
        df_use = df

    X = (
        df_use[feature_cols]
        .to_numpy(dtype=np.float32, copy=False)
        .reshape(n_breaths, seq_len, len(feature_cols))
    )

    if target_col is None:
        return X, df_use  # return df in the exact order used for reshaping

    y = (
        df_use[target_col]
        .to_numpy(dtype=np.float32, copy=False)
        .reshape(n_breaths, seq_len, 1)
    )
    return X, y, df_use


inputs, targets, train_sorted = make_breath_sequences_fast(
    train, FEATURE_COLS, TARGET_COL, seq_len=sample_length
)
test_inputs, test_sorted = make_breath_sequences_fast(
    test, FEATURE_COLS, target_col=None, seq_len=sample_length
)

print("Train sequences:", inputs.shape, "Targets:", targets.shape)
print("Test sequences:", test_inputs.shape)



## === cell 5
_ = train.shape




## === cell 6
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
        run_eagerly=False,
    )
    return model




## === cell 7
def bi_lstm_model():
    model = Sequential()
    model.add(Input(shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Bidirectional(LSTM(640, return_sequences=True)))
    model.add(Bidirectional(LSTM(320, return_sequences=True)))
    model.add(LSTM(320, return_sequences=True))
    model.add(LSTM(320, return_sequences=True))
    model.add(Dense(320, activation="relu"))
    model.add(Dense(160, activation="relu"))
    model.add(Dense(1, kernel_initializer="normal"))
    model.compile(
        loss="mae",
        optimizer="adam",
        metrics=[tensorflow.keras.metrics.RootMeanSquaredError()],
        run_eagerly=False,
    )
    return model




## === cell 8
def cnn_lstm_model():
    model = Sequential()
    model.add(Input(shape=(inputs.shape[1], inputs.shape[2])))
    model.add(
        Conv1D(
            filters=320, kernel_size=3, strides=1, padding="causal", activation="relu"
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
        run_eagerly=False,
    )
    return model




## === cell 9
BATCH_SIZE = 512

options = tf.data.Options()
options.experimental_deterministic = True

train_ds = tf.data.Dataset.from_tensor_slices((inputs, targets)).with_options(options)

shuffle_buf = min(len(inputs), 8192)
train_ds = train_ds.shuffle(
    buffer_size=shuffle_buf, seed=7, reshuffle_each_iteration=True
)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.cache()
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

model = cnn_lstm_model()

WEIGHTS_PATH = "cnn_lstm_ep500.weights.h5"

if os.path.exists(WEIGHTS_PATH):
    print(f"Loading cached weights from {WEIGHTS_PATH}")
    model.load_weights(WEIGHTS_PATH)
    history = None
else:
    t0 = time.time()
    history = model.fit(train_ds, epochs=500, verbose=2)
    print("Training seconds:", time.time() - t0)
    model.save_weights(WEIGHTS_PATH)
    print(f"Saved weights to {WEIGHTS_PATH}")



## === cell 10
_ = 4024000 / 80



## === cell 11
test_options = tf.data.Options()
test_options.experimental_deterministic = True

test_ds = tf.data.Dataset.from_tensor_slices(test_inputs).with_options(test_options)
test_ds = test_ds.batch(4096, drop_remainder=False).cache().prefetch(tf.data.AUTOTUNE)

pred_seq = model.predict(test_ds, verbose=1)  # (n_breaths, 80, 1)
pred_flat = pred_seq.reshape(-1)

if len(pred_flat) != len(test_sorted):
    raise ValueError(
        f"Prediction length {len(pred_flat)} != test rows {len(test_sorted)}"
    )

submission = pd.DataFrame(
    {"id": test_sorted["id"].to_numpy(copy=False), "pressure": pred_flat}
)
submission = submission.sort_values("id").reset_index(drop=True)

submission.to_csv("mark_5.csv", index=False)
print(submission.head())
print("Saved:", "mark_5.csv", "rows:", len(submission))
