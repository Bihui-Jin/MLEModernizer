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
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

sitecustomize_path = Path("sitecustomize.py")
sitecustomize_code = r"""
import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]
"""
sitecustomize_path.write_text(sitecustomize_code)

if "" not in sys.path:
    sys.path.insert(0, "")

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

TF_AVAILABLE = True
tf_import_error = None

import numpy as np
import pandas as pd

np.random.seed(42)

try:
    import tensorflow as tf
    from tensorflow.keras import layers

    try:
        tf.random.set_seed(42)
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
except Exception as e:
    TF_AVAILABLE = False
    tf_import_error = repr(e)

print("TF_AVAILABLE:", TF_AVAILABLE)
if not TF_AVAILABLE:
    print("TensorFlow import error:", tf_import_error)




## === cell 1
def resolve_path(p: str) -> str:
    if os.path.exists(p):
        return p
    alt = p.replace("../input", "/kaggle/input")
    if os.path.exists(alt):
        return alt
    alt2 = p.replace("../input", "/kaggle/data")
    if os.path.exists(alt2):
        return alt2
    return p


train_path = resolve_path("../input/ventilator-pressure-prediction/train.csv")
test_path = resolve_path("../input/ventilator-pressure-prediction/test.csv")
sample_sub = resolve_path(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

print("train_path exists:", os.path.exists(train_path), train_path)
print("test_path exists:", os.path.exists(test_path), test_path)
print("sample_sub exists:", os.path.exists(sample_sub), sample_sub)




## === cell 2
def dropCols(df, cols):
    df = df.copy()
    df.drop(cols, axis=1, inplace=True)
    return df




## === cell 3
if not TF_AVAILABLE:
    test_data = pd.read_csv(test_path, usecols=["id"])
    test_ids = test_data["id"].values

    submission_file = pd.read_csv(sample_sub)
    submission_file = submission_file[["id"]].copy()
    submission_file["pressure"] = 0.0

    submission_file.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_file.shape)
    print(submission_file.head())
    raise SystemExit(0)



## === cell 4
train_usecols = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
train_dtypes = {
    "id": np.int32,
    "breath_id": np.int32,
    "R": np.int16,
    "C": np.int16,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
    "pressure": np.float32,
}
train_data = pd.read_csv(train_path, usecols=train_usecols, dtype=train_dtypes)

u_in = train_data["u_in"].to_numpy()
breath = train_data["breath_id"].to_numpy()
prev_u_in = np.empty_like(u_in)
prev_u_in[0] = 0.0
same_breath = breath[1:] == breath[:-1]
prev_u_in[1:] = np.where(same_breath, u_in[:-1], 0.0).astype(u_in.dtype, copy=False)
train_data["diff_u_in"] = u_in - prev_u_in



## === cell 5
cols_2_drop = ["id", "breath_id"]



## === cell 6
Y = train_data["pressure"].to_numpy().reshape(-1, 80, 1)

X = np.column_stack(
    [
        train_data["R"].to_numpy(),
        train_data["C"].to_numpy(),
        train_data["time_step"].to_numpy(),
        train_data["u_in"].to_numpy(),
        train_data["u_out"].to_numpy(),
        train_data["diff_u_in"].to_numpy(),
    ]
).astype(np.float32, copy=False)

train_df = X.reshape(-1, 80, 6)
print("train_df:", train_df.shape, "Y:", Y.shape)




## === cell 7
def build_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Bidirectional(
            layers.LSTM(128, return_sequences=True), input_shape=[80, 6]
        )
    )
    model.add(layers.LSTM(64, dropout=0.2, return_sequences=True))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1, activation=None))

    opt = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opt, loss=tf.keras.losses.MeanAbsoluteError(), metrics=["mae"]
    )
    return model




## === cell 8
model = build_model()
model.summary()



## === cell 9
n_breaths = train_df.shape[0]
idx = np.arange(n_breaths)
rng = np.random.RandomState(42)
rng.shuffle(idx)

n_valid = int(n_breaths * 0.2)
valid_idx = idx[:n_valid]
train_idx = idx[n_valid:]

X_train = train_df[train_idx]
Y_train = Y[train_idx]
X_valid = train_df[valid_idx]
Y_valid = Y[valid_idx]

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    .batch(32, drop_remainder=False)
    .cache()
    .prefetch(AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((X_valid, Y_valid))
    .batch(32, drop_remainder=False)
    .cache()
    .prefetch(AUTOTUNE)
)

print("train_ds/valid_ds ready:", X_train.shape, X_valid.shape)



## === cell 10
callback0 = tf.keras.callbacks.ModelCheckpoint(
    "AdamPressurePreModel.keras", monitor="val_loss", save_best_only=True
)



## === cell 11
history = model.fit(
    train_ds, validation_data=valid_ds, epochs=5, callbacks=[callback0], verbose=1
)

model = tf.keras.models.load_model("AdamPressurePreModel.keras")



## === cell 12
"""
stats = pd.DataFrame(history.history) 
stats.plot()
"""



## === cell 13
test_usecols = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
test_dtypes = {
    "id": np.int32,
    "breath_id": np.int32,
    "R": np.int16,
    "C": np.int16,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
}
test_data = pd.read_csv(test_path, usecols=test_usecols, dtype=test_dtypes)

u_in_t = test_data["u_in"].to_numpy()
breath_t = test_data["breath_id"].to_numpy()
prev_u_in_t = np.empty_like(u_in_t)
prev_u_in_t[0] = 0.0
same_breath_t = breath_t[1:] == breath_t[:-1]
prev_u_in_t[1:] = np.where(same_breath_t, u_in_t[:-1], 0.0).astype(
    u_in_t.dtype, copy=False
)
test_data["diff_u_in"] = u_in_t - prev_u_in_t

test_ids = test_data["id"].values

X_test = np.column_stack(
    [
        test_data["R"].to_numpy(),
        test_data["C"].to_numpy(),
        test_data["time_step"].to_numpy(),
        test_data["u_in"].to_numpy(),
        test_data["u_out"].to_numpy(),
        test_data["diff_u_in"].to_numpy(),
    ]
).astype(np.float32, copy=False)

test_feat = X_test.reshape(-1, 80, 6)



## === cell 14
p = model.predict(test_feat, batch_size=256, verbose=1).reshape(-1)

u_out = test_data["u_out"].values.astype(np.int8)
p = np.where(u_out == 1, 0.0, p)

submission_file = pd.read_csv(sample_sub)

if not np.array_equal(submission_file["id"].values, test_ids):
    pred_df = pd.DataFrame({"id": test_ids, "pressure": p})
    submission_file = submission_file[["id"]].merge(pred_df, on="id", how="left")
else:
    submission_file = submission_file[["id"]].copy()
    submission_file["pressure"] = p

submission_file.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_file.shape)
print(submission_file.head())
