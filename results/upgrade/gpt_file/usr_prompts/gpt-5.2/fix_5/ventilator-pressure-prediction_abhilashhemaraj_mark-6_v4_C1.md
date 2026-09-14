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
import random
import time

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

tqdm.pandas()

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))



## === cell 1
directory = "../input/ventilator-pressure-prediction"

features = ["R", "C", "time_step", "u_in", "u_out"]
train_cols = ["breath_id", "id"] + features + ["pressure"]
test_cols = ["breath_id", "id"] + features

dtypes_train = {
    "breath_id": np.int32,
    "id": np.int32,
    "R": "category",
    "C": "category",
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": "category",
    "pressure": np.float32,
}
dtypes_test = {
    "breath_id": np.int32,
    "id": np.int32,
    "R": "category",
    "C": "category",
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": "category",
}

read_csv_kwargs = dict(engine="pyarrow")

train = pd.read_csv(
    os.path.join(directory, "train.csv"),
    usecols=train_cols,
    dtype=dtypes_train,
    **read_csv_kwargs,
)
test = pd.read_csv(
    os.path.join(directory, "test.csv"),
    usecols=test_cols,
    dtype=dtypes_test,
    **read_csv_kwargs,
)
sub = pd.read_csv(os.path.join(directory, "sample_submission.csv"), **read_csv_kwargs)

for col in ["R", "C", "u_out"]:
    if pd.api.types.is_categorical_dtype(train[col]):
        train[col] = train[col].cat.codes.astype(np.int16)
    if pd.api.types.is_categorical_dtype(test[col]):
        test[col] = test[col].cat.codes.astype(np.int16)

print(train.shape, test.shape, sub.shape)
print(train.head())



## === cell 2
sample_length = 80

X_train_raw = train[features].to_numpy(dtype=np.float32, copy=False)
y_train_raw = train["pressure"].to_numpy(dtype=np.float32, copy=False)

n_train_rows = X_train_raw.shape[0]
assert (
    n_train_rows % sample_length == 0
), "Train rows not divisible by 80; cannot reshape into breaths."
n_train_breaths = n_train_rows // sample_length

X_train_seq = X_train_raw.reshape(n_train_breaths, sample_length, len(features))
y_train_seq = y_train_raw.reshape(n_train_breaths, sample_length, 1)

print("Train sequences:", X_train_seq.shape, y_train_seq.shape)

assert np.allclose(X_train_seq[0, 0, :], X_train_raw[0, :])
assert np.allclose(y_train_seq[1, 0, 0], y_train_raw[sample_length])



## === cell 3
from sklearn.model_selection import train_test_split



## === cell 4
train.shape




## === cell 5
def lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(320, return_sequences=True))
    model.add(LSTM(320, return_sequences=True))
    model.add(Dense(160, activation="relu"))
    model.add(Dense(1, kernel_initializer="normal"))
    model.compile(
        loss="mae",
        optimizer="adam",
        metrics=[tensorflow.keras.metrics.RootMeanSquaredError()],
        jit_compile=True,
    )
    return model




## === cell 6
idx = np.arange(n_train_breaths)
train_idx, val_idx = train_test_split(
    idx, test_size=0.1, random_state=SEED, shuffle=True
)

X_tr, y_tr = X_train_seq[train_idx], y_train_seq[train_idx]
X_va, y_va = X_train_seq[val_idx], y_train_seq[val_idx]

model_cache_path = "model_cache.keras"

if os.path.exists(model_cache_path):
    print(f"Loading cached model from: {model_cache_path}")
    model = tf.keras.models.load_model(model_cache_path, compile=True)
else:
    model = lstm_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    model.summary()

    batch_size = 5000
    epochs = 500
    batch_size = min(batch_size, X_tr.shape[0])

    AUTO = tf.data.AUTOTUNE

    train_opts = tf.data.Options()
    try:
        train_opts.autotune.enabled = True
    except Exception:
        pass
    train_opts.experimental_deterministic = True

    train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).with_options(train_opts)
    train_ds = train_ds.batch(batch_size, drop_remainder=False).cache().prefetch(AUTO)

    val_opts = tf.data.Options()
    try:
        val_opts.autotune.enabled = True
    except Exception:
        pass
    val_opts.experimental_deterministic = True

    val_ds = tf.data.Dataset.from_tensor_slices((X_va, y_va)).with_options(val_opts)
    val_ds = val_ds.batch(batch_size, drop_remainder=False).cache().prefetch(AUTO)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
    )

    print(f"Saving trained model to: {model_cache_path}")
    model.save(model_cache_path)



## === cell 7
pass



## === cell 8
X_test_raw = test[features].to_numpy(dtype=np.float32, copy=False)
n_test_rows = X_test_raw.shape[0]
assert (
    n_test_rows % sample_length == 0
), "Test rows not divisible by 80; cannot reshape into breaths."
n_test_breaths = n_test_rows // sample_length

X_test_seq = X_test_raw.reshape(n_test_breaths, sample_length, len(features))
print("Test sequences:", X_test_seq.shape)

test_opts = tf.data.Options()
test_opts.experimental_deterministic = True
test_ds = tf.data.Dataset.from_tensor_slices(X_test_seq).with_options(test_opts)

pred_batch = min(4096, n_test_breaths)
test_ds = test_ds.batch(pred_batch, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

pred_seq = model.predict(test_ds, verbose=1)  # (n_breaths, 80, 1)



## === cell 9
sub_array = pred_seq.reshape(-1)
assert len(sub_array) == len(test), "Prediction length mismatch with test rows."



## === cell 10
submission = pd.DataFrame(
    {
        "id": test["id"].astype(np.int64),
        "pressure": sub_array.astype(np.float32, copy=False),
    }
)
submission = submission[["id", "pressure"]]

print(submission.head())
print(submission.shape)



## === cell 11
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(pd.read_csv(submission_path).head())
