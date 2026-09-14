# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

0.5486

# 6. Current score

0.76503

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.60133) has done: 'Diagnosis: Cell 9 crashes because it tries to load a local pretrained model file `./AdamPressurePreModel.h5` that does not exist in this environment, resulting in a `FileNotFoundError`. The intended fallback (commented out) is to build a fresh model using the already-defined `build_model()` function. To keep the notebook running deterministically without changing the model architecture/training semantics, we should conditionally load the file if present, otherwise instantiate the model via `build_model()`.

Patch summary: In cell 9 only, add an `os.path.exists` guard around `tf.keras.models.load_model`, and fall back to `build_model()` when the file is missing. Keep the `model` variable name and `model.summary()` call unchanged so later cells continue to work.

Updated cells: Only cell 9 is modified.

Compatibility notes for cell k+1: Cell 10 expects `model` to exist; this patch guarantees `model` is defined (either loaded or freshly built). No changes to `train_df`, `Y`, or dataset creation are introduced.

Assumptions: The pretrained `.h5` file is genuinely absent in the runtime filesystem; building an untrained model is acceptable to proceed to subsequent training/evaluation steps in later cells.'
- What this solution (achieved 0.68925) has done: 'Your current score is far worse than the target (lower-is-better), and the main issue is that the model is never trained in this script, so predictions come from an untrained/random-initialized network. I make the smallest change that preserves your exact model architecture and loss: add a single `model.fit(...)` call using your existing `train_ds`, `valid_ds`, and `ModelCheckpoint`, then reload the best checkpoint before predicting. I also fix the `diff_u_in` feature to be computed within each `breath_id` (both train and test), because the current global `.shift(1)` leaks across breath boundaries and harms MAE; this is a minimal feature-correctness fix without changing what features you use. These changes should move the score dramatically toward the target band while keeping the core logic intact and still producing a valid `submission.csv`.'
- What this solution (achieved 0.70653) has done: 'You’re currently worse than the target (lower MAE is better), so the smallest safe move toward the target is to improve generalization without changing your model or loss. I (1) fix the validation split to be by `breath_id` (not by row order) to avoid leakage/shift artifacts and to better reflect the metric, and (2) add an `u_out==1 => pressure=0` post-processing step at prediction time since expiratory steps are not scored and this typically reduces overall MAE without changing training semantics. These are minimal, metric-aligned changes that keep your architecture/training loop intact and should reduce the gap toward 0.5486 while still producing a valid `submission.csv`.'
- What this solution (achieved 0.68014) has done: 'Your current MAE (0.70653) is worse than the target (0.5486), so we should make small, metric-aligned changes that improve predictions without changing your model architecture or training loop. The biggest low-risk gain here is to quantize/clip predictions to the discrete pressure grid seen in training, which is known to reduce MAE for this competition while preserving the same model and loss. I also add a minimal post-processing clip to the observed train pressure range to avoid out-of-range outputs. Everything else (features, model, loss, epochs, split strategy, and u_out handling) stays the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.75903) has done: 'Your current MAE (0.68014) is worse than the target (0.5486), so we should make a small, metric-aligned improvement without changing the model architecture or training loop. The most impactful minimal fix here is to stop training on expiratory timesteps (`u_out==1`), since those rows are not scored; we can do this by switching the loss/metric to a masked MAE that ignores `u_out==1` while keeping everything else the same. This preserves the same inputs/features and training procedure, but aligns optimization with the competition metric and typically reduces MAE toward your target. I keep your existing `u_out==1 => pressure=0` prediction post-processing and pressure-grid snapping intact, and still write a valid `submission.csv`.'
- What this solution (achieved 0.76503) has done: 'Your current MAE (0.75903) is worse than the target (0.5486), and the most likely cause is that training/validation splitting is currently done on already-reshaped arrays, which can unintentionally mix timesteps and harm generalization relative to the breath-level metric. I make the smallest metric-aligned fix: split by `breath_id` *before* reshaping, so entire breaths go either to train or validation, and then reshape consistently. I also ensure `diff_u_in` is computed within each breath (already correct) and keep your masked loss, architecture, epochs, u_out post-processing, and pressure-grid snapping unchanged. This should move the MAE down toward the target while preserving the core logic and still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_version
except Exception:
    _pb_version = None


def _pb_major(ver):
    try:
        return int(str(ver).split(".", 1)[0])
    except Exception:
        return None


if _pb_version is None or (
    _pb_major(_pb_version) is not None and _pb_major(_pb_version) >= 6
):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )
    import importlib
    import google.protobuf

    importlib.reload(google.protobuf)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers

tf.keras.utils.set_random_seed(42)



## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 2
def dropCols(df, cols):
    df = df.copy()
    df.drop(cols, axis=1, inplace=True)
    return df




## === cell 3
train_data = pd.read_csv(train_path)

train_data["diff_u_in"] = train_data["u_in"] - train_data.groupby("breath_id")[
    "u_in"
].shift(1).fillna(0)



## === cell 4
cols_2_drop = ["id", "breath_id"]



## === cell 5
breath_ids = train_data["breath_id"].values
unique_breath_ids = np.unique(breath_ids)

rng = np.random.default_rng(42)
rng.shuffle(unique_breath_ids)

n_valid = int(len(unique_breath_ids) * 0.25)
valid_breath_ids = set(unique_breath_ids[:n_valid])
is_valid = np.isin(breath_ids, list(valid_breath_ids))

train_data_tr = train_data.loc[~is_valid].copy()
train_data_va = train_data.loc[is_valid].copy()




## === cell 6
def make_xy(df, cols_2_drop):
    df2 = dropCols(df, cols_2_drop)
    pressure = df2.pop("pressure").astype(np.float32).values
    u_out = (
        df2["u_out"].astype(np.float32).values
    )  # keep as feature too (unchanged feature set)
    Y = np.stack([pressure, u_out], axis=-1).astype(np.float32)
    X = df2.values.astype(np.float32)
    X = X.reshape(-1, 80, X.shape[-1]).astype(np.float32)
    Y = Y.reshape(-1, 80, 2).astype(np.float32)
    return X, Y


X_train, y_train = make_xy(train_data_tr, cols_2_drop)
X_valid, y_valid = make_xy(train_data_va, cols_2_drop)

train_pressure_values = np.sort(train_data["pressure"].unique()).astype(np.float32)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape




## === cell 7
def masked_mae(y_true, y_pred):
    true_p = y_true[..., 0:1]
    u_out_true = y_true[..., 1:2]
    mask = tf.cast(tf.equal(u_out_true, 0.0), tf.float32)
    mae = tf.abs(true_p - y_pred) * mask
    denom = tf.reduce_sum(mask) + 1e-7
    return tf.reduce_sum(mae) / denom


def build_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Bidirectional(
            layers.LSTM(128, return_sequences=True), input_shape=[80, 6]
        )
    )
    model.add(
        layers.Bidirectional(layers.LSTM(128, dropout=0.2, return_sequences=True))
    )
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1, activation="relu"))

    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss=masked_mae, metrics=[masked_mae])
    return model




## === cell 8
model_path = "./AdamPressurePreModel.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(
        model_path, custom_objects={"masked_mae": masked_mae}
    )
else:
    model = build_model()

model.summary()



## === cell 9
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(4096, seed=42, reshuffle_each_iteration=True)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)
valid_ds = (
    tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)



## === cell 10
callback0 = tf.keras.callbacks.ModelCheckpoint(
    "AdamPressurePreModel.h5", monitor="val_loss", save_best_only=True
)



## === cell 11
his = model.fit(
    train_ds, validation_data=valid_ds, epochs=10, callbacks=[callback0], verbose=2
)



## === cell 12
"""stats = pd.DataFrame(his.history)
stats.plot()"""



## === cell 13
if os.path.exists("AdamPressurePreModel.h5"):
    model = tf.keras.models.load_model(
        "AdamPressurePreModel.h5", custom_objects={"masked_mae": masked_mae}
    )

test_data = pd.read_csv(test_path)

test_data["diff_u_in"] = test_data["u_in"] - test_data.groupby("breath_id")[
    "u_in"
].shift(1).fillna(0)

test_u_out = test_data["u_out"].values.reshape(-1, 80)

test_data = dropCols(test_data, cols_2_drop)
test_data = test_data.values.reshape(-1, 80, test_data.shape[-1]).astype(np.float32)

p = model.predict(test_data, batch_size=256, verbose=1).reshape(-1, 80)

p = np.where(test_u_out == 1, 0.0, p)

pressure_values = train_pressure_values

p = np.clip(p, pressure_values.min(), pressure_values.max()).astype(np.float32)

idx = np.searchsorted(pressure_values, p, side="left")
idx = np.clip(idx, 0, len(pressure_values) - 1)

idx0 = np.clip(idx - 1, 0, len(pressure_values) - 1)
v0 = pressure_values[idx0]
v1 = pressure_values[idx]
p = np.where(np.abs(p - v0) <= np.abs(p - v1), v0, v1)

submission_file = pd.read_csv(sample_sub)
submission_file["pressure"] = p.reshape(-1)
submission_file.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_file.shape)
