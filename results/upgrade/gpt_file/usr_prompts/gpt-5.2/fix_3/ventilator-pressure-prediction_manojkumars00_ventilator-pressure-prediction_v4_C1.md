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

0.72706

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.7276) has done: 'I fix the TensorFlow import crash by pinning a compatible pure-Python protobuf implementation before importing TF, which resolves the `MessageFactory.GetPrototype` error in this environment. Then I remove the dependency on a missing pre-trained `.h5` file by building the same model and training it (same architecture/loss/loop semantics) before inference. I also fix the time-series feature bug where `diff_u_in` was incorrectly computed across breath boundaries by computing it per `breath_id`, which is a minimal logic fix that should materially improve MAE toward your target. Finally, I ensure the submission is written as `submission.csv` with the exact required `id,pressure` columns and correct row alignment.'
- What this solution (achieved 0.72706) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation *before* TensorFlow is imported, and I also add a small compatibility pin for `protobuf<5` (installed via pip) which is the typical root cause of the `MessageFactory.GetPrototype` failure in Kaggle TF environments. Then I keep your same feature engineering, model, and training loop, but adjust the final activation from `relu` to `linear` so the model can predict the full pressure range (your current `relu` unnecessarily clips negatives/shape and usually hurts MAE for this task), which should move the score toward your target. Finally, I keep the submission creation logic but add safety checks on shape/alignment and ensure `submission.csv` is always produced with the required `id,pressure` columns.'

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys
import subprocess


def _pip_install(pkg: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as pb_ver

    major = int(pb_ver.split(".", 1)[0])
    if major >= 5:
        _pip_install("protobuf<5")
except Exception:
    _pip_install("protobuf<5")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 2
def dropCols(df, cols):
    df = df.copy()
    df.drop(cols, axis=1, inplace=True)
    return df


def add_diff_u_in_per_breath(df):
    df = df.copy()
    df["diff_u_in"] = df.groupby("breath_id")["u_in"].diff().fillna(0.0)
    return df




## === cell 3
train_data = pd.read_csv(train_path)
train_data = add_diff_u_in_per_breath(train_data)

cols_2_drop = ["id", "breath_id"]

train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")

train_df = train_df.values.reshape(-1, 80, train_df.shape[-1])
Y = Y.values.reshape(-1, 80, 1)

print("train_df:", train_df.shape, "Y:", Y.shape)




## === cell 4
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

    model.add(layers.Dense(1, activation="linear"))

    opt = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opt, loss=tf.keras.losses.MeanAbsoluteError(), metrics=["mae"]
    )
    return model


model = build_model()
model.summary()



## === cell 5
ds = tf.data.Dataset.from_tensor_slices((train_df, Y))

n = train_df.shape[0]
n_valid = int(n * 0.25)
n_train = n - n_valid

ds = ds.shuffle(buffer_size=min(n, 20000), seed=42, reshuffle_each_iteration=False)

valid_ds = ds.take(n_valid).batch(32).prefetch(tf.data.AUTOTUNE)
train_ds = ds.skip(n_valid).batch(32).prefetch(tf.data.AUTOTUNE)

callback0 = tf.keras.callbacks.ModelCheckpoint(
    "AdamPressurePreModel.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
)



## === cell 6
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=10,
    callbacks=[callback0],
    verbose=2,
)

model = tf.keras.models.load_model("AdamPressurePreModel.h5", compile=True)



## === cell 7
test_data = pd.read_csv(test_path)
test_data = add_diff_u_in_per_breath(test_data)

test_ids = test_data["id"].values  # preserve alignment for submission
test_data = dropCols(test_data, cols_2_drop)
test_data = test_data.values.reshape(-1, 80, test_data.shape[-1])

print("test_data:", test_data.shape)



## === cell 8
p = model.predict(test_data, batch_size=256, verbose=1)
p = p.reshape(-1)

print("pred shape:", p.shape, "min/max:", float(np.min(p)), float(np.max(p)))

if len(p) != len(test_ids):
    raise ValueError(f"Prediction length {len(p)} != test_ids length {len(test_ids)}")



## === cell 9
submission_file = pd.read_csv(sample_sub)

sub = pd.DataFrame({"id": test_ids, "pressure": p})
sub = sub.sort_values("id").reset_index(drop=True)

submission_file = submission_file.sort_values("id").reset_index(drop=True)

if len(submission_file) != len(sub):
    raise ValueError(
        f"sample_submission rows {len(submission_file)} != predictions rows {len(sub)}"
    )

submission_file["pressure"] = sub["pressure"].values
submission_file.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_file.shape)
print(submission_file.head())
