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

# 5. Target score

1.9106

# 6. Current score

0.64066

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 17.67478) has done: 'The timeout is coming from training a very large stacked (Bi)LSTM for up to 500 epochs over ~68k breaths; that is not feasible within 600 seconds. To preserve core logic and accuracy, the main speed fix is to deterministically stage and load pretrained weights (same model, same inference semantics) and avoid the expensive “train-if-missing” path by hard-failing early if weights are not available. Additionally, we remove avoidable input-pipeline overhead (in-memory `.cache()` duplication, unnecessary `.cache()` on huge datasets) and switch inference to a direct NumPy `model.predict(...)` call (no tf.data build cost), while keeping determinism settings and identical model architecture/loss.'
- What this solution (achieved 0.64066) has done: 'You’re hitting two separate blockers: (1) TensorFlow import crashes due to an incompatibility between TF 2.18 and the installed protobuf 6.x, and (2) the script hard-fails when pretrained weights aren’t present, so it never produces a submission. I fix the TF/protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow (a common Kaggle workaround for this exact error) while keeping the same TF version and model code. Then I remove the “must-have weights” hard failure and instead run a short, feasible training of the same architecture (same loss/optimizer/loop semantics) so the notebook completes within 600s and outputs `submission.csv`. Finally, I also ensure the split is done by `breath_id` (not by individual timesteps) to prevent leakage and improve MAE toward your target.'
- What this solution (achieved 0.64066) has done: 'You’re blocked immediately by a TensorFlow import crash caused by protobuf 6.x incompatibility; I fix this by forcing the pure-Python protobuf runtime and (critically) disabling TF’s C++ protobuf implementation before importing TensorFlow. Then I keep your same model/training/inference logic, but make the runtime more reliable by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` and guarding the determinism call so the notebook always reaches submission writing. Finally, I ensure the output `submission.csv` is created even if something odd happens with sorting/alignment by validating row counts and preserving the original test row order for the `id` mapping (sorting only by `id` at the very end as you already do).'
- What this solution (achieved 0.64066) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x incompatibility by forcing a compatible protobuf runtime **before** any TensorFlow-related import (including indirect imports). Then I keep your exact model/training/inference logic intact, but make the environment setup more robust by clearing conflicting protobuf/TF settings and importing TensorFlow only after the workaround is applied. Finally, I keep the same submission writing, but add a small safety check so the notebook always reaches `submission.csv` creation if TensorFlow loads successfully.'

# 9. Code solution

## === cell 0
import os
import random
import shutil
import glob
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense

random.seed(7)
np.random.seed(7)
tf.random.set_seed(7)

print("TF version:", tf.__version__)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism enable warning:", e)

try:
    tf.config.optimizer.set_jit(True)
    print("XLA JIT: enabled")
except Exception as e:
    print("XLA JIT enable warning:", e)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("GPUs:", gpus)
    except Exception as e:
        print("GPU setup warning:", e)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
directory = "/kaggle/input/ventilator-pressure-prediction"

train = pd.read_csv(
    os.path.join(directory, "train.csv"),
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
    dtype={
        "id": np.int32,
        "breath_id": np.int32,
        "R": np.int16,
        "C": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
        "pressure": np.float32,
    },
)
test = pd.read_csv(
    os.path.join(directory, "test.csv"),
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
    dtype={
        "id": np.int32,
        "breath_id": np.int32,
        "R": np.int16,
        "C": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
    },
)
sub = pd.read_csv(os.path.join(directory, "sample_submission.csv"))

print(train.shape, test.shape, sub.shape)
train.head()



## === cell 2
FEATURES = ["R", "C", "time_step", "u_in", "u_out"]
TARGET = "pressure"
SEQLEN = 80

assert len(train) % SEQLEN == 0, "Train rows not divisible by SEQLEN."
assert len(test) % SEQLEN == 0, "Test rows not divisible by SEQLEN."
n_train_breaths = len(train) // SEQLEN
n_test_breaths = len(test) // SEQLEN

train_breath_ids = train["breath_id"].to_numpy()
test_breath_ids = test["breath_id"].to_numpy()
assert np.all(
    train_breath_ids[::SEQLEN] == train_breath_ids.reshape(-1, SEQLEN)[:, 0]
), "Train breath_id not constant within breath."
assert np.all(
    test_breath_ids[::SEQLEN] == test_breath_ids.reshape(-1, SEQLEN)[:, 0]
), "Test breath_id not constant within breath."

train[FEATURES].values[:5]




## === cell 3
def make_breath_tensor(df, features, seq_len=80, target_col=None):
    n = len(df)
    n_breaths = n // seq_len
    X = (
        df[features]
        .to_numpy(dtype=np.float32, copy=False)
        .reshape(n_breaths, seq_len, len(features))
    )
    if target_col is None:
        return X
    y = (
        df[target_col]
        .to_numpy(dtype=np.float32, copy=False)
        .reshape(n_breaths, seq_len, 1)
    )
    return X, y


inputs, targets = make_breath_tensor(train, FEATURES, SEQLEN, TARGET)
test_inputs = make_breath_tensor(test, FEATURES, SEQLEN, target_col=None)

print("Train X/y:", inputs.shape, targets.shape)
print("Test X:", test_inputs.shape)




## === cell 4
def bi_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
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
    )
    return model


model = bi_lstm_model((inputs.shape[1], inputs.shape[2]))
model.summary()



## === cell 5
breath_ids = train["breath_id"].to_numpy()[::SEQLEN]
assert len(breath_ids) == inputs.shape[0]

rng = np.random.RandomState(7)
unique_breaths = breath_ids.copy()
idx = np.arange(len(unique_breaths))
rng.shuffle(idx)

val_size = int(0.1 * len(idx))
val_idx = np.sort(idx[:val_size])
tr_idx = np.sort(idx[val_size:])

X_tr, y_tr = inputs[tr_idx], targets[tr_idx]
X_val, y_val = inputs[val_idx], targets[val_idx]

print(X_tr.shape, y_tr.shape, X_val.shape, y_val.shape)

BATCH_SIZE = 80



## === cell 6
WEIGHTS_PATH = "model_weights.weights.h5"


def _try_stage_weights_from_input(dst_path: str) -> bool:
    if os.path.exists(dst_path):
        return True

    basename = os.path.basename(dst_path)

    likely_paths = [
        os.path.join("/kaggle/input", basename),
        os.path.join("/kaggle/input/ventilator-pressure-prediction", basename),
        os.path.join("/kaggle/input", "ventilator-pressure-prediction", basename),
        os.path.join("/kaggle/input", "model-weights", basename),
        os.path.join("/kaggle/input", "weights", basename),
    ]
    for p in likely_paths:
        if os.path.exists(p):
            try:
                shutil.copyfile(p, dst_path)
                print(f"Staged pretrained weights from: {p} -> {dst_path}")
                return True
            except Exception as e:
                print("Failed to stage weights from likely path:", p, "err:", e)
                return False

    try:
        matches = glob.glob(
            os.path.join("/kaggle/input", "**", basename), recursive=True
        )
        matches.sort(key=len)
        for candidate in matches:
            if os.path.isfile(candidate):
                try:
                    shutil.copyfile(candidate, dst_path)
                    print(f"Staged pretrained weights from: {candidate} -> {dst_path}")
                    return True
                except Exception as e:
                    print(
                        "Failed to stage weights from glob candidate:",
                        candidate,
                        "err:",
                        e,
                    )
                    return False
    except Exception as e:
        print("Weights glob warning:", e)

    return False


have_weights = _try_stage_weights_from_input(WEIGHTS_PATH)

if have_weights and os.path.exists(WEIGHTS_PATH):
    print(f"Found {WEIGHTS_PATH}; loading weights.")
    model.load_weights(WEIGHTS_PATH)
else:
    EPOCHS = 12
    print(
        f"Pretrained weights not found; training model for {EPOCHS} epochs to produce a valid submission."
    )
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    try:
        model.save_weights(WEIGHTS_PATH)
        print("Saved weights to:", WEIGHTS_PATH)
    except Exception as e:
        print("Warning: could not save weights:", e)



## === cell 7
test_pred = model.predict(test_inputs, batch_size=BATCH_SIZE, verbose=1)
print("Raw pred shape:", test_pred.shape)

test_pred_flat = test_pred.reshape(-1)

ts = test["time_step"].to_numpy()
assert np.all(
    ts.reshape(-1, SEQLEN)[:, 1:] >= ts.reshape(-1, SEQLEN)[:, :-1]
), "time_step not non-decreasing within breath."

test_sorted = test  # already in correct order

assert len(test_sorted) == len(
    test_pred_flat
), "Prediction length mismatch with test rows."



## === cell 8
submission = pd.DataFrame(
    {
        "id": test_sorted["id"].astype(np.int64, copy=False),
        "pressure": test_pred_flat.astype(np.float32, copy=False),
    }
)

submission = submission.sort_values("id").reset_index(drop=True)

assert submission.shape[0] == sub.shape[0]
assert list(submission.columns) == ["id", "pressure"]

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
submission.head()



## === cell 9
print(submission.describe(include="all"))
print("Any NaNs:", submission.isna().any().to_dict())
print("Submission rows:", len(submission))
print("Min/Max id:", submission["id"].min(), submission["id"].max())
