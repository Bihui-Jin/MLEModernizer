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
import os, gc, time, logging

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)



## === cell 1
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
submission = pd.read_csv(SAMPLE_SUB_PATH)

train.head(), test.head(), submission.head()



## === cell 2
assert "pressure" in train.columns
assert "pressure" not in test.columns
assert set(submission.columns) == {"id", "pressure"}

test_ids = test["id"].values



## === cell 3
train = train.drop(columns=["id"])
test = test.drop(columns=["id"])




## === cell 4
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["RC_sum"] = df["R"] + df["C"]
    df["RC_div"] = df["R"] / df["C"]

    gb = df.groupby("breath_id", sort=False)
    df["u_in_cumsum"] = gb["u_in"].cumsum()

    df["time_lag1"] = gb["time_step"].shift(1).fillna(0)
    df["u_in_lag1"] = gb["u_in"].shift(1).fillna(0)
    df["u_out_lag1"] = gb["u_out"].shift(1).fillna(0)

    df["time_lag2"] = gb["time_step"].shift(2).fillna(0)

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)

    return df


train = add_features(train)
test = add_features(test)

y = train["pressure"].to_numpy()
train.drop(columns=["pressure"], inplace=True)

train = pd.get_dummies(train)
test = pd.get_dummies(test)

test = test.reindex(columns=train.columns, fill_value=0)

train.drop(columns=["breath_id"], inplace=True)
test.drop(columns=["breath_id"], inplace=True)



## === cell 5
rb = RobustScaler()
rb.fit(train)

train2 = rb.transform(train)
test2 = rb.transform(test)

del train, test
gc.collect()



## === cell 6
SEQ_LEN = 80
n_features = train2.shape[1]

assert train2.shape[0] % SEQ_LEN == 0, "Train rows not divisible by 80"
assert test2.shape[0] % SEQ_LEN == 0, "Test rows not divisible by 80"
assert y.shape[0] == train2.shape[0], "Target length mismatch"

n_breaths_train = train2.shape[0] // SEQ_LEN
n_breaths_test = test2.shape[0] // SEQ_LEN

train3 = train2.reshape(n_breaths_train, SEQ_LEN, n_features)
test3 = test2.reshape(n_breaths_test, SEQ_LEN, n_features)
y = y.reshape(n_breaths_train, SEQ_LEN)

del train2, test2, rb
gc.collect()

(train3.shape, y.shape, test3.shape, n_features)



## === cell 7
tf.get_logger().setLevel(logging.ERROR)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # will fail on non-TPU
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except Exception:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled (GPU)")
        except Exception:
            print("Mixed precision not enabled (API/version constraint)")
    strategy = (
        tf.distribute.MirroredStrategy()
        if len(gpus) > 1
        else tf.distribute.get_strategy()
    )
    print("Running on", "GPU" if len(gpus) > 0 else "CPU")

print("TensorFlow:", tf.__version__)
print("REPLICAS:", strategy.num_replicas_in_sync)




## === cell 8
def plot_hist(hist):
    plt.figure(figsize=(8, 4))
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.title("model performance")
    plt.ylabel("MAE")
    plt.xlabel("epoch")
    plt.legend(loc="upper left")
    plt.show()




## === cell 9
def create_model():
    with strategy.scope():
        model = Sequential(
            [
                Input(shape=(SEQ_LEN, n_features)),
                Bidirectional(LSTM(300, return_sequences=True)),
                Bidirectional(LSTM(256, return_sequences=True)),
                Bidirectional(LSTM(200, return_sequences=True)),
                Bidirectional(LSTM(128, return_sequences=True)),
                Dense(128, activation="selu"),
                Dropout(0.2),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mae")
    return model




## === cell 10
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

test_preds = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(train3, y)):
    print(f"****** fold: {fold+1} *******")
    X_train, X_valid = train3[train_idx], train3[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    decay_steps = int(200 * ((len(train3) * 0.8) / 512))
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=max(decay_steps, 1),
        decay_rate=1e-5,
        staircase=False,
    )

    with strategy.scope():
        model = create_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss="mae"
        )

    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=25,
        verbose=1,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=240,
        batch_size=512,
        callbacks=[es],
        verbose=1,
    )

    preds = model.predict(test3, batch_size=512, verbose=0).squeeze()  # (n_breaths, 80)
    test_preds.append(preds)

    plot_hist(history)

    del X_train, X_valid, y_train, y_valid, model, history, preds
    gc.collect()



## === cell 11
pred_mean = np.mean(np.stack(test_preds, axis=0), axis=0)  # (n_breaths_test, 80)
pred_flat = pred_mean.reshape(-1)

assert len(pred_flat) == len(
    submission
), "Prediction length does not match submission length"

submission["id"] = test_ids
submission["pressure"] = pred_flat.astype(np.float32)

submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 12
print("Wrote:", os.path.abspath("submission.csv"))
subm_check = pd.read_csv("submission.csv")
print(subm_check.head())
print(subm_check.shape)
assert list(subm_check.columns) == ["id", "pressure"]
assert len(subm_check) == 603600
