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

3.10

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
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "42"

import random

random.seed(42)
np.random.seed(42)

import tensorflow as tf

tf.random.set_seed(42)

tf.config.optimizer.set_jit(True)

from tensorflow.keras import layers
from sklearn.model_selection import KFold



## === cell 1
from sklearn.preprocessing import RobustScaler

rc = RobustScaler()



## === cell 2
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 3
def dropCols(df, cols):
    df = df.copy()
    df.drop(cols, axis=1, inplace=True)
    return df




## === cell 4
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
usecols_train = list(TRAIN_DTYPES.keys())
train_data = pd.read_csv(train_path, usecols=usecols_train, dtype=TRAIN_DTYPES)

g = train_data.groupby("breath_id", sort=False)
train_data["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0).astype("float32")
train_data["diff_u_in1"] = (train_data["u_in"] - train_data["u_in_lag1"]).astype(
    "float32"
)
train_data["u_in_cumsum"] = g["u_in"].cumsum().astype("float32")



## === cell 5
cols_2_drop = ["id", "breath_id", "time_step"]



## === cell 6
train_df = dropCols(train_data, cols_2_drop)
Y_series = train_df.pop(
    "pressure"
)  # keep original 1D series for later unique pressure computation



## === cell 7
rc.fit(train_df)
train_X = rc.transform(train_df).astype(np.float32, copy=False)

Y = Y_series.to_numpy(dtype=np.float32, copy=False)



## === cell 8
n_features = train_X.shape[-1]
train_X = train_X.reshape(-1, 80, n_features)
Y = Y.reshape(-1, 80, 1)




## === cell 9
def build_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Bidirectional(
            layers.LSTM(120, return_sequences=True),
            input_shape=[80, train_X.shape[-1]],
        )
    )
    model.add(layers.Bidirectional(layers.LSTM(180, return_sequences=True)))
    model.add(
        layers.Bidirectional(layers.LSTM(280, dropout=0.2, return_sequences=True))
    )
    model.add(
        layers.Bidirectional(layers.LSTM(360, dropout=0.2, return_sequences=True))
    )
    model.add(
        layers.Bidirectional(layers.LSTM(500, dropout=0.2, return_sequences=True))
    )
    model.add(
        layers.Bidirectional(layers.LSTM(1000, dropout=0.2, return_sequences=True))
    )

    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.TimeDistributed(layers.Dense(1)))

    opt = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opt, loss=tf.keras.losses.MeanAbsoluteError(), metrics=["mae"]
    )
    return model




## === cell 10
callback1 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.9,
    patience=10,
    verbose=1,
)



## === cell 11
EPOCH = 50
BATCH_SIZE = 1024

kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_idx, valid_idx = next(kf.split(train_X))

X_train, X_valid = train_X[train_idx], train_X[valid_idx]
y_train, y_valid = Y[train_idx], Y[valid_idx]

model = build_model()

callback0 = tf.keras.callbacks.ModelCheckpoint(
    "AdamPressurePreModel_local_best.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).cache().prefetch(AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=False).cache().prefetch(AUTOTUNE)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCH,
    callbacks=[callback0, callback1],
    verbose=2,
)

model = tf.keras.models.load_model("AdamPressurePreModel_local_best.h5")



## === cell 12
TEST_DTYPES = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}
usecols_test = list(TEST_DTYPES.keys())
test_data = pd.read_csv(test_path, usecols=usecols_test, dtype=TEST_DTYPES)

gtest = test_data.groupby("breath_id", sort=False)
test_data["u_in_lag1"] = gtest["u_in"].shift(1).fillna(0.0).astype("float32")
test_data["diff_u_in1"] = (test_data["u_in"] - test_data["u_in_lag1"]).astype("float32")
test_data["u_in_cumsum"] = gtest["u_in"].cumsum().astype("float32")

test_data = dropCols(test_data, cols_2_drop)
test_X = rc.transform(test_data).astype(np.float32, copy=False)
test_X = test_X.reshape(-1, 80, test_X.shape[-1])



## === cell 13
models = [model]



## === cell 14
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_X)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

p = []
for m in models:
    p.append(m.predict(test_ds, verbose=1).reshape(-1, 1))



## === cell 15
submission_files = [
    "../input/ventilatorpressure-submitted-files/sub.csv",
    "../input/ventilatorpressure-submitted-files/sub1.csv",
    "../input/ventilatorpressure-submitted-files/sub2.csv",
    "../input/ventilatorpressure-submitted-files/sub4.csv",
    "../input/ventilatorpressure-submitted-files/sub5.csv",
]

for sub_path in submission_files:
    if os.path.exists(sub_path):
        sub = pd.read_csv(sub_path)
        if "pressure" in sub.columns:
            p.append(sub["pressure"].values.reshape(-1, 1))



## === cell 16
predictions_table = np.concatenate(p, axis=-1)



## === cell 17
median_pre = np.median(predictions_table, axis=-1)
mean_pre = np.mean(predictions_table, axis=-1)



## === cell 18
unique_pressures = np.unique(Y_series.to_numpy(dtype=np.float32, copy=False))
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = float((unique_pressures[1] - unique_pressures[0]).item())
PRESSURE_MIN = float(sorted_pressures[0].item())
PRESSURE_MAX = float(sorted_pressures[-1].item())



## === cell 19
rounding_pre = (
    np.round((median_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
rounding_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)



## === cell 20
submission_file = pd.read_csv(sample_sub)
if len(submission_file) != len(rounding_pre):
    raise ValueError(
        f"Prediction length mismatch: submission rows={len(submission_file)} vs preds={len(rounding_pre)}"
    )

submission_file["pressure"] = rounding_pre.astype(np.float32)
submission_file.to_csv("submission_mediam_round.csv", index=False)

print("Wrote submission_mediam_round.csv with shape:", submission_file.shape)
print(submission_file.head())
