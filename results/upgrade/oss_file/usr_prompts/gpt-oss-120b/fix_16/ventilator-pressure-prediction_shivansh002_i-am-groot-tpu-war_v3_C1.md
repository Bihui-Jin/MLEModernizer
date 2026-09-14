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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.2173

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import optuna

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

tf.config.threading.set_inter_op_parallelism_threads(tf.data.AUTOTUNE)
tf.config.threading.set_intra_op_parallelism_threads(tf.data.AUTOTUNE)




## === cell 1
DEBUG = False

dtype_dict = {
    "R": "int8",
    "C": "int8",
    "breath_id": "int32",
    "id": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv", dtype=dtype_dict
)
test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv", dtype=dtype_dict)
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)




## === cell 2
def add_features(df):
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()

    df["u_in_cumsum"] = df["u_in"].groupby(df["breath_id"]).cumsum()

    df["u_in_lag"] = df["u_in"].shift(2).fillna(0)

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df = pd.get_dummies(df)

    grp = df.groupby("breath_id")["u_in"]
    df["ewm_u_in_mean"] = grp.ewm(halflife=8).mean().reset_index(level=0, drop=True)
    df["ewm_u_in_std"] = grp.ewm(halflife=9).std().reset_index(level=0, drop=True)
    df["ewm_u_in_corr"] = grp.ewm(halflife=14).corr().reset_index(level=0, drop=True)

    roll = df.groupby("breath_id")["u_in"].rolling(window=15, min_periods=1)
    roll_agg = roll.agg(["max", "std"]).rename(
        columns={"max": "15_in_max", "std": "15_in_std"}
    )
    df[["15_in_max", "15_in_std"]] = roll_agg.reset_index(level=0, drop=True)

    return df


train = add_features(train)
test = add_features(test)




## === cell 3
train = train.fillna(0)
test = test.fillna(0)




## === cell 4
train_pressure_mean = train["pressure"].mean()

targets = train["pressure"].values.reshape(-1, 80, 1)

train = train.drop(["pressure", "id", "breath_id"], axis=1)
test = test.drop(["id", "breath_id"], axis=1)




## === cell 5
RS = RobustScaler()
train = RS.fit_transform(train).astype(np.float32)
test = RS.transform(test).astype(np.float32)

train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, test.shape[-1])




## === cell 6
EPOCH = 60  # early‑stopping will likely stop earlier
BATCH_SIZE = 2048  # reduced to avoid possible OOM and improve training stability
tf.config.optimizer.set_jit(True)


def make_dataset(x, y=None, shuffle=False):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(x)
    else:
        ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)  # removed .cache()
    return ds


kf = KFold(n_splits=3, shuffle=True, random_state=2021)
test_preds = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(train, targets)):
    print("-" * 15, f"Fold {fold+1}", "-" * 15)
    X_train, X_valid = train[train_idx], train[valid_idx]
    y_train, y_valid = targets[train_idx], targets[valid_idx]

    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=X_train.shape[1:]),
            keras.layers.Bidirectional(keras.layers.LSTM(250, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(200, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(50, return_sequences=True)),
            keras.layers.Dense(100, activation="selu"),
            keras.layers.Dense(1, dtype="float32"),
        ]
    )

    steps_per_epoch = max(1, (len(X_train) * 0.8) // BATCH_SIZE)
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=steps_per_epoch * 400,
        decay_rate=0.5,
        staircase=False,
    )
    lr_callback = LearningRateScheduler(
        lambda epoch, lr: float(lr_schedule(epoch)), verbose=1
    )

    es = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )

    model.compile(optimizer="adam", loss="mae")

    train_ds = make_dataset(X_train, y_train, shuffle=True)
    valid_ds = make_dataset(X_valid, y_valid, shuffle=False)

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCH,
        callbacks=[lr_callback, es],
        verbose=2,
    )

    val_pred = model.predict(valid_ds, verbose=0)  # (samples, 80, 1)
    val_pred_mean = np.mean(val_pred)
    y_valid_mean = np.mean(y_valid)
    if val_pred_mean != 0:
        scale_factor = y_valid_mean / val_pred_mean
    else:
        scale_factor = 1.0

    test_ds = make_dataset(test, shuffle=False)
    fold_pred = model.predict(test_ds, verbose=0)  # (samples, 80, 1)
    fold_pred = np.squeeze(fold_pred, axis=-1)  # (samples, 80)
    fold_pred = fold_pred * scale_factor  # calibrated
    test_preds.append(fold_pred)

    tf.keras.backend.clear_session()




## === cell 7
print("Training complete.")




## === cell 8
avg_pred = np.mean(np.stack(test_preds), axis=0)  # (samples, 80)

avg_pred = pd.DataFrame(avg_pred).rolling(window=5, min_periods=1, axis=1).mean().values

avg_pred = np.clip(avg_pred, 0, 50)

submission["pressure"] = avg_pred.reshape(-1)  # flatten to rows
submission.to_csv("submission.csv", index=False)
print("submission.csv written, shape:", submission.shape)
