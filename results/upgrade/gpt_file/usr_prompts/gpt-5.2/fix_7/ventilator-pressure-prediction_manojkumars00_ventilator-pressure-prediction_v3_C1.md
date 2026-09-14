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

0.94325

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.93269) has done: 'I fix the TensorFlow import crash by setting a safe protobuf implementation environment variable before importing TensorFlow (this resolves the `MessageFactory.GetPrototype` error in many Kaggle images). I also remove the failing attempt to load a non-existent pretrained `.h5` and instead build the same model architecture and train it using the already-prepared `tf.data.Dataset` pipeline, so `model` is defined for inference. I keep the existing feature logic (including `diff_u_in`) and shapes, but make `diff_u_in` breath-aware to avoid leakage across breath boundaries (score-positive and still consistent with the same feature intent). Finally, I ensure predictions are written into `submission.csv` with the exact `id,pressure` format.'
- What this solution (achieved 0.86253) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* disabling the C++ one **before any TensorFlow/protobuf import**, plus clearing any already-imported `google.protobuf` modules to ensure the env var actually takes effect in Kaggle. I also make the train/valid split deterministic and leakage-safe by shuffling at the *breath* level (still the same model, loss, and features), which typically improves MAE versus taking the first 20% breaths as validation (distribution shift). Finally, I keep the same feature engineering and model architecture/training loop, and ensure the submission is written as `submission.csv` with correct `id,pressure` alignment from `test.csv`.'
- What this solution (achieved 0.89403) has done: 'I fix the TensorFlow/protobuf crash by ensuring the environment variables are set before any TensorFlow-related import and by forcing a compatible protobuf package version in-process (Kaggle images sometimes ship an incompatible protobuf that triggers `MessageFactory.GetPrototype`). I also add a safe fallback path: if TensorFlow still cannot import, the script still generate a valid `submission.csv` by outputting a simple baseline (so you always get a valid file). The modeling/training logic, features, split strategy, and submission alignment remain unchanged when TensorFlow loads successfully, so the score behavior stays comparable while unblocking end-to-end execution.'
- What this solution (achieved 0.89095) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* any protobuf/TensorFlow import and by removing the in-notebook `pip install protobuf` attempt that can leave the runtime in a broken mixed state (root cause of the `MessageFactory.GetPrototype` error). If TensorFlow still cannot import, the existing fallback still write a valid `submission.csv` so you always get an output file. I also make the `ModelCheckpoint` use the native `.keras` format to avoid HDF5/serialization pitfalls and keep the rest of your feature engineering, model architecture, split strategy, training loop, and submission alignment unchanged (score-impact should be neutral to slightly positive from stability). All paths and the required `id,pressure` submission format are preserved.'
- What this solution (achieved 0.88656) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation early and additionally setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, which is a common Kaggle-safe workaround for the `MessageFactory.GetPrototype` error. I also make the fallback submission write *and continue* (without raising) in case TF still fails, ensuring you always get a valid `submission.csv`. To move the MAE score toward your target (lower is better) with minimal semantic change, I align inference with the competition metric by setting predicted pressure to 0 during the expiratory phase (`u_out==1`), which is not scored and typically reduces model confusion/noise. All other core modeling, features, training loop, and paths remain unchanged.'
- What this solution (achieved 0.94325) has done: 'I fix the TensorFlow/protobuf import crash that currently prevents the model from training/inferencing by ensuring the pure-Python protobuf implementation is enforced before any protobuf/TensorFlow import and by forcing a compatible protobuf runtime behavior in-process (without adding new packages). To keep score improvements minimal but meaningful toward your target (lower MAE), I preserve the same model/training loop and features, but change only the final layer activation from ReLU to linear so the model can predict the full pressure range (your labels include values below many ReLU-biased outputs), which typically reduces MAE without changing the architecture depth/units. I also keep the `u_out==1 -> 0` post-processing (not scored phase) and ensure the script always writes a valid `submission.csv` even if TF still cannot load (fallback). All paths and the submission format (`id,pressure`) remain unchanged.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

TF_AVAILABLE = True
tf_import_error = None

try:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras import layers
except Exception as e:
    TF_AVAILABLE = False
    tf_import_error = repr(e)

import numpy as np
import pandas as pd

np.random.seed(42)
if TF_AVAILABLE:
    try:
        tf.random.set_seed(42)
    except Exception:
        pass

print("TF_AVAILABLE:", TF_AVAILABLE)
if not TF_AVAILABLE:
    print("TensorFlow import error:", tf_import_error)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
if not TF_AVAILABLE:
    test_data = pd.read_csv(test_path)
    test_ids = test_data["id"].values

    submission_file = pd.read_csv(sample_sub)
    submission_file = submission_file[["id"]].copy()
    submission_file["pressure"] = 0.0

    submission_file.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_file.shape)
    print(submission_file.head())
    raise SystemExit(0)



## === cell 4
train_data = pd.read_csv(train_path)

train_data["diff_u_in"] = train_data["u_in"] - train_data.groupby("breath_id")[
    "u_in"
].shift(1).fillna(0)



## === cell 5
cols_2_drop = ["id", "breath_id"]



## === cell 6
train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")



## === cell 7
train_df = train_df.values.reshape(-1, 80, train_df.shape[-1])
Y = Y.values.reshape(-1, 80, 1)

print("train_df:", train_df.shape, "Y:", Y.shape)




## === cell 8
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




## === cell 9
model = build_model()
model.summary()



## === cell 10
n_breaths = train_df.shape[0]
idx = np.arange(n_breaths)
rng = np.random.RandomState(42)
rng.shuffle(idx)

n_valid = int(n_breaths * 0.2)
valid_idx = idx[:n_valid]
train_idx = idx[n_valid:]

X_train, y_train = train_df[train_idx], Y[train_idx]
X_valid, y_valid = train_df[valid_idx], Y[valid_idx]

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(32)



## === cell 11
callback0 = tf.keras.callbacks.ModelCheckpoint(
    "AdamPressurePreModel.keras", monitor="val_loss", save_best_only=True
)



## === cell 12
history = model.fit(
    train_ds, validation_data=valid_ds, epochs=5, callbacks=[callback0], verbose=1
)

model = tf.keras.models.load_model("AdamPressurePreModel.keras")



## === cell 13
"""
stats = pd.DataFrame(history.history) 
stats.plot()
"""



## === cell 14
test_data = pd.read_csv(test_path)

test_data["diff_u_in"] = test_data["u_in"] - test_data.groupby("breath_id")[
    "u_in"
].shift(1).fillna(0)

test_ids = test_data["id"].values

test_feat = dropCols(test_data, cols_2_drop)
test_feat = test_feat.values.reshape(-1, 80, test_feat.shape[-1])



## === cell 15
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
