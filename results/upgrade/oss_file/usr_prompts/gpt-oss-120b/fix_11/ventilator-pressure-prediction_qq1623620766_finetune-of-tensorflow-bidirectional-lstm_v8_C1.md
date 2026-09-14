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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

tf.config.optimizer.set_jit(True)
try:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass




## === cell 1
SEED = 2021
np.random.seed(SEED)
tf.random.set_seed(SEED)




## === cell 2
DATA_PATH = "../input/ventilator-pressure-prediction/"

train = pd.read_csv(f"{DATA_PATH}train.csv")
test = pd.read_csv(f"{DATA_PATH}test.csv")
submission = pd.read_csv(f"{DATA_PATH}sample_submission.csv")




## === cell 3
def add_features(df):
    gb = df.groupby("breath_id", sort=False)

    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = gb["area"].cumsum()

    df["u_in_cumsum"] = gb["u_in"].cumsum()

    for lag in range(1, 5):
        df[[f"u_in_lag{lag}", f"u_out_lag{lag}"]] = (
            gb[["u_in", "u_out"]].shift(lag).fillna(0).values
        )
        df[[f"u_in_lag_back{lag}", f"u_out_lag_back{lag}"]] = (
            gb[["u_in", "u_out"]].shift(-lag).fillna(0).values
        )

    for lag in range(1, 5):
        df[f"u_in_diff{lag}"] = df["u_in"] - df[f"u_in_lag{lag}"]
        df[f"u_out_diff{lag}"] = df["u_out"] - df[f"u_out_lag{lag}"]

    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]

    df["R"] = df["R"].astype("category")
    df["C"] = df["C"].astype("category")
    df["R__C"] = (df["R"].astype(str) + "__" + df["C"].astype(str)).astype("category")
    return df


train = add_features(train)
test = add_features(test)

combined = pd.concat(
    [train.drop(columns=["pressure"]), test], axis=0, ignore_index=True
)

combined = pd.get_dummies(combined, dtype=np.uint8)

train = combined.iloc[: len(train), :].copy()
test = combined.iloc[len(train) :, :].copy()

train["pressure"] = pd.read_csv(f"{DATA_PATH}train.csv")["pressure"]




## === cell 4
targets = train[["pressure"]].to_numpy().reshape(-1, 80)

train = train.drop(["pressure", "id", "breath_id"], axis=1)
test = test.drop(["id", "breath_id"], axis=1)




## === cell 5
scaler = RobustScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

train = train.astype(np.float32)
test = test.astype(np.float32)




## === cell 6
train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, test.shape[-1])




## === cell 7
BATCH_SIZE = 4096
EPOCH = 150
NUM_FOLDS = 3  # keep original cross‑validation scheme

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)


def train_one_fold(fold, train_idx, valid_idx):
    np.random.seed(SEED + fold)
    tf.random.set_seed(SEED + fold)
    tf.keras.backend.clear_session()

    X_train, X_valid = train[train_idx], train[valid_idx]
    y_train, y_valid = targets[train_idx], targets[valid_idx]

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = (
        train_ds.cache()
        .shuffle(1024, seed=SEED + fold)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    valid_ds = valid_ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=train.shape[-2:]),
            keras.layers.Bidirectional(keras.layers.LSTM(1024, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
            keras.layers.Dense(128, activation="selu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mae")

    lr_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
    es_cb = EarlyStopping(
        monitor="val_loss",
        patience=5,  # tighter patience to halt training earlier
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCH,
        callbacks=[lr_cb, es_cb],
        verbose=0,
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices(test)
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    pred = model.predict(test_ds, verbose=0).squeeze()
    pred = pred.reshape(-1)
    return pred


fold_results = [
    train_one_fold(fold, train_idx, valid_idx)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train))
]

test_preds = list(fold_results)




## === cell 8
final_pred = np.mean(np.stack(test_preds, axis=0), axis=0)

submission["pressure"] = final_pred
submission.to_csv("submission.csv", index=False)

print("Submission saved to submission.csv")
