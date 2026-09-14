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

# 5. Target score

0.5427281114016237

# 6. Current score

0.88289

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.93269) has done: 'I fix the TensorFlow import crash by setting a safe protobuf implementation environment variable before importing TensorFlow (this resolves the `MessageFactory.GetPrototype` error in many Kaggle images). I also remove the failing attempt to load a non-existent pretrained `.h5` and instead build the same model architecture and train it using the already-prepared `tf.data.Dataset` pipeline, so `model` is defined for inference. I keep the existing feature logic (including `diff_u_in`) and shapes, but make `diff_u_in` breath-aware to avoid leakage across breath boundaries (score-positive and still consistent with the same feature intent). Finally, I ensure predictions are written into `submission.csv` with the exact `id,pressure` format.'
- What this solution (achieved 0.86253) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* disabling the C++ one **before any TensorFlow/protobuf import**, plus clearing any already-imported `google.protobuf` modules to ensure the env var actually takes effect in Kaggle. I also make the train/valid split deterministic and leakage-safe by shuffling at the *breath* level (still the same model, loss, and features), which typically improves MAE versus taking the first 20% breaths as validation (distribution shift). Finally, I keep the same feature engineering and model architecture/training loop, and ensure the submission is written as `submission.csv` with correct `id,pressure` alignment from `test.csv`.'
- What this solution (achieved 0.89403) has done: 'I fix the TensorFlow/protobuf crash by ensuring the environment variables are set before any TensorFlow-related import and by forcing a compatible protobuf package version in-process (Kaggle images sometimes ship an incompatible protobuf that triggers `MessageFactory.GetPrototype`). I also add a safe fallback path: if TensorFlow still cannot import, the script still generate a valid `submission.csv` by outputting a simple baseline (so you always get a valid file). The modeling/training logic, features, split strategy, and submission alignment remain unchanged when TensorFlow loads successfully, so the score behavior stays comparable while unblocking end-to-end execution.'
- What this solution (achieved 0.89095) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* any protobuf/TensorFlow import and by removing the in-notebook `pip install protobuf` attempt that can leave the runtime in a broken mixed state (root cause of the `MessageFactory.GetPrototype` error). If TensorFlow still cannot import, the existing fallback still write a valid `submission.csv` so you always get an output file. I also make the `ModelCheckpoint` use the native `.keras` format to avoid HDF5/serialization pitfalls and keep the rest of your feature engineering, model architecture, split strategy, training loop, and submission alignment unchanged (score-impact should be neutral to slightly positive from stability). All paths and the required `id,pressure` submission format are preserved.'
- What this solution (achieved 0.88656) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation early and additionally setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, which is a common Kaggle-safe workaround for the `MessageFactory.GetPrototype` error. I also make the fallback submission write *and continue* (without raising) in case TF still fails, ensuring you always get a valid `submission.csv`. To move the MAE score toward your target (lower is better) with minimal semantic change, I align inference with the competition metric by setting predicted pressure to 0 during the expiratory phase (`u_out==1`), which is not scored and typically reduces model confusion/noise. All other core modeling, features, training loop, and paths remain unchanged.'
- What this solution (achieved 0.94325) has done: 'I fix the TensorFlow/protobuf import crash that currently prevents the model from training/inferencing by ensuring the pure-Python protobuf implementation is enforced before any protobuf/TensorFlow import and by forcing a compatible protobuf runtime behavior in-process (without adding new packages). To keep score improvements minimal but meaningful toward your target (lower MAE), I preserve the same model/training loop and features, but change only the final layer activation from ReLU to linear so the model can predict the full pressure range (your labels include values below many ReLU-biased outputs), which typically reduces MAE without changing the architecture depth/units. I also keep the `u_out==1 -> 0` post-processing (not scored phase) and ensure the script always writes a valid `submission.csv` even if TF still cannot load (fallback). All paths and the submission format (`id,pressure`) remain unchanged.'
- What this solution (achieved 17.68919) has done: 'Main runtime bottlenecks are (1) very slow `groupby().shift()` over 5.4M rows, (2) materializing large `X_train/X_valid` arrays before building `tf.data`, and (3) suboptimal `tf.data` input pipeline (no caching/prefetch, no parallelism). The optimized version computes `diff_u_in` with a fully-vectorized NumPy method that is exactly equivalent to the per-breath shift, avoids copying/dropping DataFrames repeatedly, and builds `tf.data` directly from the full arrays via index gather (no extra huge intermediate arrays). It also adds `cache()`/`prefetch()` and deterministic settings to reduce input overhead while keeping the same model, loss, epochs, split logic, and prediction semantics intact. File paths and submission logic are unchanged.'
- What this solution (achieved 17.38412) has done: 'The timeout is dominated by LSTM training over ~68k breaths with eager execution overhead and suboptimal input pipeline; the rest (CSV load + feature prep + inference) is comparatively small. I keep the exact same model architecture, loss, epochs, and split logic, but make training faster by enabling XLA JIT compilation, ensuring tf.data uses efficient options (no expensive `.cache()` on huge in-memory tensors, add `.shuffle()` and `.prefetch()` properly, and avoid extra copies), and by feeding the full array directly where it’s provably equivalent. I also reduce pandas overhead by computing features directly into NumPy arrays without adding extra DataFrame columns, keeping identical values. These changes preserve evaluation semantics and determinism while significantly reducing wall-clock time.'
- What this solution (achieved 0.87118) has done: 'Main runtime is dominated by training (5 epochs over ~68k breaths) and by expensive dataframe/array materialization. I keep the exact same model and training loop but reduce input-pipeline and memory overhead by avoiding creation of large intermediate arrays/copies, using `tf.data.Dataset.from_tensor_slices` directly on the full arrays with index-based splitting, and enabling dataset cache (in-memory) so each epoch doesn’t rebuild/rehit Python objects. I also remove the forced pure-Python protobuf setting (it slows TF startup/graph building significantly) while keeping determinism via fixed seeds and deterministic TF ops. Finally, I make prediction input a `tf.data` pipeline (same batch size) to reduce peak memory and speed host→TF feeding without changing outputs.'
- What this solution (achieved 0.89817) has done: 'I fix the immediate runtime crash by enforcing the pure-Python protobuf implementation *before any TensorFlow/protobuf import* and by clearing any already-imported `google.protobuf` modules so the env var reliably takes effect in Kaggle (this directly addresses the `MessageFactory.GetPrototype` error). I keep your model, features, training loop, epochs, and post-processing the same, but remove the now-harmful `os.environ.pop(...)` lines that undo the protobuf workaround. I also keep the existing fallback that writes a valid `submission.csv` if TensorFlow still fails to import, ensuring an end-to-end run always produces a submission file. These changes are primarily correctness/stability and should also move the score back toward your target by restoring the TensorFlow training path instead of crashing.'
- What this solution (achieved 0.88289) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by enforcing a protobuf/TensorFlow-safe environment *before any protobuf/TensorFlow import* and by explicitly selecting the Python protobuf implementation while also disabling the C++ one; this is the root cause blocking training/inference and currently forcing your pipeline to fail. We keep your exact model architecture, features (including breath-aware `diff_u_in`), training loop, and post-processing intact so score behavior stays consistent, but actually runs end-to-end using TensorFlow rather than falling back. We also make the TF-availability probe robust: if TF still cannot import for any reason, the script still write a valid `submission.csv` and exit cleanly. No changes are made to paths, columns, epochs, or core learning setup.'

# 9. Code solution

## === cell 0
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION", "1")

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import numpy as np
import pandas as pd

np.random.seed(42)

TF_AVAILABLE = True
tf_import_error = None

try:
    import tensorflow as tf
    from tensorflow.keras import layers

    try:
        tf.random.set_seed(42)
    except Exception:
        pass

    try:
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        pass

    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass

except Exception as e:
    TF_AVAILABLE = False
    tf_import_error = repr(e)

print("TF_AVAILABLE:", TF_AVAILABLE)
if not TF_AVAILABLE:
    print("TensorFlow import error:", tf_import_error)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

u_in = train_data["u_in"].to_numpy(copy=False)
breath = train_data["breath_id"].to_numpy(copy=False)

prev_u_in = np.empty_like(u_in)
prev_u_in[0] = 0.0
same_breath = breath[1:] == breath[:-1]
prev_u_in[1:] = u_in[:-1]
prev_u_in[1:][~same_breath] = 0.0
diff_u_in = u_in - prev_u_in



## === cell 5
cols_2_drop = ["id", "breath_id"]



## === cell 6
n_rows = train_data.shape[0]
X = np.empty((n_rows, 6), dtype=np.float32)

X[:, 0] = train_data["R"].to_numpy(copy=False)
X[:, 1] = train_data["C"].to_numpy(copy=False)
X[:, 2] = train_data["time_step"].to_numpy(copy=False)
X[:, 3] = u_in
X[:, 4] = train_data["u_out"].to_numpy(copy=False)
X[:, 5] = diff_u_in

train_df = X.reshape(-1, 80, 6)
Y = (
    train_data["pressure"]
    .to_numpy(copy=False)
    .astype(np.float32, copy=False)
    .reshape(-1, 80, 1)
)

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
    model.compile(optimizer=opt, loss=tf.keras.losses.MeanAbsoluteError())
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

AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.deterministic = True

X_tf = tf.convert_to_tensor(train_df)
Y_tf = tf.convert_to_tensor(Y)

train_idx_tf = tf.constant(train_idx, dtype=tf.int64)
valid_idx_tf = tf.constant(valid_idx, dtype=tf.int64)

train_ds = tf.data.Dataset.from_tensor_slices(
    (tf.gather(X_tf, train_idx_tf), tf.gather(Y_tf, train_idx_tf))
).with_options(options)

valid_ds = tf.data.Dataset.from_tensor_slices(
    (tf.gather(X_tf, valid_idx_tf), tf.gather(Y_tf, valid_idx_tf))
).with_options(options)

train_ds = (
    train_ds.shuffle(
        buffer_size=min(len(train_idx), 8192),
        seed=42,
        reshuffle_each_iteration=True,
    )
    .batch(32, drop_remainder=False)
    .cache()
    .prefetch(AUTOTUNE)
)

valid_ds = valid_ds.batch(32, drop_remainder=False).cache().prefetch(AUTOTUNE)

print("train_ds/valid_ds ready:", len(train_idx), len(valid_idx))



## === cell 10
callback0 = tf.keras.callbacks.ModelCheckpoint(
    "AdamPressurePreModel.keras", monitor="val_loss", save_best_only=True
)



## === cell 11
try:
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        steps_per_execution=32,
    )
except Exception:
    pass

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

u_in_t = test_data["u_in"].to_numpy(copy=False)
breath_t = test_data["breath_id"].to_numpy(copy=False)

prev_u_in_t = np.empty_like(u_in_t)
prev_u_in_t[0] = 0.0
same_breath_t = breath_t[1:] == breath_t[:-1]
prev_u_in_t[1:] = u_in_t[:-1]
prev_u_in_t[1:][~same_breath_t] = 0.0
diff_u_in_t = u_in_t - prev_u_in_t

test_ids = test_data["id"].to_numpy(copy=False)

n_test_rows = test_data.shape[0]
X_test = np.empty((n_test_rows, 6), dtype=np.float32)
X_test[:, 0] = test_data["R"].to_numpy(copy=False)
X_test[:, 1] = test_data["C"].to_numpy(copy=False)
X_test[:, 2] = test_data["time_step"].to_numpy(copy=False)
X_test[:, 3] = u_in_t
X_test[:, 4] = test_data["u_out"].to_numpy(copy=False)
X_test[:, 5] = diff_u_in_t

test_feat = X_test.reshape(-1, 80, 6)



## === cell 14
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_feat).batch(256).prefetch(tf.data.AUTOTUNE)
)
p = model.predict(test_ds, verbose=1).reshape(-1)

u_out = test_data["u_out"].to_numpy(copy=False).astype(np.int8, copy=False)
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
