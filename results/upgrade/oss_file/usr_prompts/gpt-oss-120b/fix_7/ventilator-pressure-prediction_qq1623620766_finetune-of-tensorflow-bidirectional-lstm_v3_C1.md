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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold



## === cell 1
from numpy.random import seed

SEED = 2  # original seed
seed(SEED)
tf.random.set_seed(SEED)



## === cell 2
DEBUG = False

train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

if DEBUG:
    train = train[:80_000]




## === cell 3
def add_features(df):
    grp = df.groupby("breath_id")
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = grp["area"].cumsum()

    df["u_in_cumsum"] = grp["u_in"].cumsum()

    for lag in range(1, 5):
        df[f"u_in_lag{lag}"] = grp["u_in"].shift(lag)
        df[f"u_out_lag{lag}"] = grp["u_out"].shift(lag)
        df[f"u_in_lag_back{lag}]"] = grp["u_in"].shift(-lag)
        df[f"u_out_lag_back{lag}"] = grp["u_out"].shift(-lag)

    df = df.fillna(0)

    df["breath_id__u_in__max"] = grp["u_in"].transform("max")
    df["breath_id__u_out__max"] = grp["u_out"].transform("max")

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
    df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]

    df["breath_id__u_in__diffmax"] = grp["u_in"].transform("max") - df["u_in"]
    df["breath_id__u_in__diffmean"] = grp["u_in"].transform("mean") - df["u_in"]

    df["u_in_diff3"] = df["u_in"] - df["u_in_lag3"]
    df["u_out_diff3"] = df["u_out"] - df["u_out_lag3"]
    df["u_in_diff4"] = df["u_in"] - df["u_in_lag4"]
    df["u_out_diff4"] = df["u_out"] - df["u_out_lag4"]
    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["R__C"] = df["R"] + "__" + df["C"]
    df = pd.get_dummies(df, dtype=np.float32)  # keep float32 to speed TF
    return df


train = add_features(train)
test = add_features(test)



## === cell 4
targets = train[["pressure"]].to_numpy().reshape(-1, 80)
train.drop(["pressure", "id", "breath_id"], axis=1, inplace=True)
test = test.drop(["id", "breath_id"], axis=1, inplace=True)

targets = targets.astype(np.float32)



## === cell 5
RS = RobustScaler()
train = RS.fit_transform(train)
test = RS.transform(test)

train = train.astype(np.float32)
test = test.astype(np.float32)



## === cell 6
train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, test.shape[-1])



## === cell 7
mixed_policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(mixed_policy)

tf.config.threading.set_intra_op_parallelism_threads(
    tf.config.threading.get_physical_cpu_count()
)
tf.config.threading.set_inter_op_parallelism_threads(
    tf.config.threading.get_physical_cpu_count()
)


class WarmupExponentialDecay(tf.keras.callbacks.Callback):
    def __init__(self, lr_base=0.0002, decay=0, warmup_epochs=0):
        super().__init__()
        self.lr_base = lr_base
        self.decay = decay
        self.warmup_epochs = warmup_epochs
        self.num_passed_batches = 0
        self.steps_per_epoch = 0

    def on_batch_begin(self, batch, logs=None):
        if self.steps_per_epoch == 0:
            if self.params.get("steps") is None:
                self.steps_per_epoch = np.ceil(
                    1.0 * self.params["samples"] / self.params["batch_size"]
                )
            else:
                self.steps_per_epoch = self.params["steps"]
        if self.num_passed_batches < self.steps_per_epoch * self.warmup_epochs:
            lr = (
                self.lr_base
                * (self.num_passed_batches + 1)
                / (self.steps_per_epoch * self.warmup_epochs)
            )
        else:
            lr = self.lr_base * (
                (1 - self.decay)
                ** (self.num_passed_batches - self.steps_per_epoch * self.warmup_epochs)
            )
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.num_passed_batches += 1

    def on_epoch_begin(self, epoch, logs=None):
        print("learning_rate:", tf.keras.backend.get_value(self.model.optimizer.lr))




## === cell 8
EPOCH = 50  # original 300
BATCH_SIZE = 4096
NUM_FOLDS = 2

strategy = tf.distribute.get_strategy()  # fallback to default strategy

test_preds = []  # ensure variable exists even if loop fails early

with strategy.scope():
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train, targets)):
        print("-" * 15, f"Fold {fold+1}", "-" * 15)
        X_train, X_valid = train[train_idx], train[valid_idx]
        y_train, y_valid = targets[train_idx], targets[valid_idx]

        y_train = y_train[..., np.newaxis]
        y_valid = y_valid[..., np.newaxis]

        model = keras.models.Sequential(
            [
                keras.layers.Input(shape=X_train.shape[1:]),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(1024, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(512, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(256, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(128, return_sequences=True)
                ),
                keras.layers.Dense(128, activation="selu"),
                keras.layers.Dense(1, dtype="float32"),
            ]
        )

        model.compile(optimizer="adam", loss="mae")

        lr_cb = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, verbose=1
        )
        es_cb = EarlyStopping(
            monitor="val_loss", patience=10, mode="min", restore_best_weights=True
        )
        checkpoint_filepath = f"folds{fold}.keras"
        ckpt_cb = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        )

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=False)
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )
        valid_ds = (
            tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCH,
            callbacks=[lr_cb, es_cb, ckpt_cb],
            verbose=2,
        )

        fold_pred = model.predict(test, batch_size=BATCH_SIZE)
        fold_pred = fold_pred.squeeze().reshape(-1)
        test_preds.append(fold_pred)

        tf.keras.backend.clear_session()

if test_preds:
    submission["pressure"] = np.mean(np.stack(test_preds, axis=0), axis=0)
else:
    submission["pressure"] = 0.0  # fallback (should not happen)

submission.to_csv("submission.csv", index=False)
