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
import random

random.seed(42)
np.random.seed(42)

dtype = {
    "id": np.int64,
    "breath_id": np.int64,
    "R": np.float32,
    "C": np.float32,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.float32,
    "pressure": np.float32,
}

train_path = "/kaggle/input/ventilator-pressure-prediction/train.csv"
test_path = "/kaggle/input/ventilator-pressure-prediction/test.csv"
sample_sub_path = "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"

usecols = ["breath_id", "R", "C", "u_in", "u_out", "pressure"]
df = pd.read_csv(train_path, dtype=dtype, usecols=usecols)
df.shape



## === cell 1
df.isna().sum().sum()



## === cell 2
graal = "pressure"

features = df.columns.to_list()
features.remove(graal)
features



## === cell 3
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 6)
plt.close("all")



## === cell 4
plt.close("all")



## === cell 5
plt.close("all")



## === cell 6
from sklearn.preprocessing import MinMaxScaler

scalerX = MinMaxScaler(feature_range=(0, 1))
scalerY = MinMaxScaler(feature_range=(0, 1))

X_np = np.asarray(df[features].to_numpy(copy=False), dtype=np.float32, order="C")
y_np = np.asarray(df[[graal]].to_numpy(copy=False), dtype=np.float32, order="C")


def _fit_minmax_scaler_and_transform(scaler: MinMaxScaler, X: np.ndarray) -> np.ndarray:
    data_min = X.min(axis=0)
    data_max = X.max(axis=0)
    data_range = data_max - data_min
    scale = np.empty_like(data_range, dtype=np.float32)
    min_ = np.empty_like(data_range, dtype=np.float32)

    nonzero = data_range != 0
    scale[nonzero] = 1.0 / data_range[nonzero]
    scale[~nonzero] = 0.0
    min_[nonzero] = -data_min[nonzero] * scale[nonzero]
    min_[~nonzero] = 0.0

    scaler.data_min_ = data_min.astype(np.float32, copy=False)
    scaler.data_max_ = data_max.astype(np.float32, copy=False)
    scaler.data_range_ = data_range.astype(np.float32, copy=False)
    scaler.scale_ = scale
    scaler.min_ = min_
    scaler.n_features_in_ = X.shape[1]
    scaler.feature_names_in_ = None

    return (X * scaler.scale_ + scaler.min_).astype(np.float32, copy=False)


X_all_scaled = _fit_minmax_scaler_and_transform(scalerX, X_np)
y_all_scaled = _fit_minmax_scaler_and_transform(scalerY, y_np)

breath_id_all = df["breath_id"].to_numpy(dtype=np.int64, copy=False)
breath_id_all[:5]




## === cell 7
def _reshape_by_breath_from_array(arr: np.ndarray, steps: int = 80) -> np.ndarray:
    n = arr.shape[0]
    if n % steps != 0:
        raise ValueError(
            f"Row count {n} not divisible by steps={steps}; cannot reshape safely."
        )
    arr = np.ascontiguousarray(arr)
    return arr.reshape(n // steps, steps, -1)


n_rows = X_all_scaled.shape[0]
n_breaths = n_rows // 80
train_breaths = int(n_breaths * 0.92)
split_row = train_breaths * 80

X_train = _reshape_by_breath_from_array(X_all_scaled[:split_row], steps=80)
X_test = _reshape_by_breath_from_array(X_all_scaled[split_row:], steps=80)
y_train = _reshape_by_breath_from_array(y_all_scaled[:split_row], steps=80)
y_test = _reshape_by_breath_from_array(y_all_scaled[split_row:], steps=80)

print("train :", X_train.shape, " -> ", y_train.shape)
print("test  :", X_test.shape, " -> ", y_test.shape)



## === cell 8
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    LayerNormalization,
    Dropout,
    UpSampling1D,
)

tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, min(4, (os.cpu_count() or 2) // 2))
    )
except Exception:
    pass


def build_model(ksize, kernel, dense, dropout):
    input_shape = X_train.shape[1:]
    activation = "relu"

    model = Sequential()

    for layer in range(0, 4):
        model.add(
            Conv1D(
                kernel,
                ksize,
                padding="same",
                strides=2,
                activation=activation,
                input_shape=input_shape,
            )
        )
        model.add(LayerNormalization())

    for layer in range(0, 4):
        model.add(UpSampling1D(size=2))
        model.add(Conv1D(kernel, ksize, padding="same", activation=activation))

    for layer in range(1, dense):
        model.add(Dropout(dropout))
        model.add(Dense(1))

    model.add(Dropout(dropout))
    model.add(Dense(1))
    return model


learning_rate = 0.00018
model = build_model(7, 256, 1, 0.1)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_absolute_error")
model.summary()



## === cell 9
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

model_file = "checkpoint.weights.h5"

early = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=4, mode="auto", restore_best_weights=True
)
checkpoint = ModelCheckpoint(
    model_file,
    monitor="val_loss",
    save_best_only=True,
    verbose=0,
    save_weights_only=True,
    mode="auto",
    save_freq="epoch",
)
rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.9, patience=2, verbose=1)

callbacks = [rlrop, checkpoint, early]

batch_size = 32


class PerEpochShuffledSequence(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, seed=42):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.seed = int(seed)
        self.n = X.shape[0]
        self.idxs = np.arange(self.n, dtype=np.int32)
        self.epoch = 0

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        rng.shuffle(self.idxs)
        self.epoch += 1

    def __getitem__(self, i):
        sl = slice(i * self.batch_size, min((i + 1) * self.batch_size, self.n))
        b = self.idxs[sl]
        return self.X[b], self.y[b]


train_seq = PerEpochShuffledSequence(X_train, y_train, batch_size=batch_size, seed=42)

val_options = tf.data.Options()
val_options.experimental_deterministic = True
val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_ds = (
    val_ds.with_options(val_options)
    .batch(256, drop_remainder=False)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

hist = model.fit(
    train_seq,
    validation_data=val_ds,
    verbose=1,
    epochs=100,
    callbacks=callbacks,
)



## === cell 10
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(hist.history["loss"], label="mean absolute error")
plt.plot(hist.history["val_loss"], label="val mean absolute error")
plt.ylabel("Metric")
plt.xlabel("Epoch")
plt.legend(loc="upper left")
plt.close()



## === cell 11
from sklearn.metrics import mean_absolute_error

if os.path.exists(model_file):
    model.load_weights(model_file)

y_test_pred = model.predict(X_test, verbose=1, batch_size=1024)  # (n_breaths, 80, 1)
y_test_pred_flat = y_test_pred.reshape(-1, 1)
y_test_true_flat = y_test.reshape(-1, 1)

y_test_pred_inv = scalerY.inverse_transform(y_test_pred_flat).reshape(-1)
y_test_true_inv = scalerY.inverse_transform(y_test_true_flat).reshape(-1)

val_mae = mean_absolute_error(y_test_true_inv, y_test_pred_inv)
print(f"Validation MAE (all timesteps): {val_mae:.6f}")



## === cell 12
df_score = pd.read_csv(
    test_path,
    dtype={k: v for k, v in dtype.items() if k != "pressure"},
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
)

ids = df_score["id"].to_numpy(dtype=np.int64, copy=False)
u_out_score = df_score["u_out"].to_numpy(dtype=np.int32, copy=False)

X_score_np = np.asarray(
    df_score[features].to_numpy(copy=False), dtype=np.float32, order="C"
)



## === cell 13
X_score_scaled = scalerX.transform(X_score_np).astype(np.float32, copy=False)
X_score = np.ascontiguousarray(X_score_scaled).reshape(
    len(df_score) // 80, 80, len(features)
)
print("X_score:", X_score.shape)

y_score_pred = model.predict(X_score, verbose=1, batch_size=1024)  # (n_breaths, 80, 1)
y_score_pred_flat = y_score_pred.reshape(-1, 1)
y_score_pred_inv = scalerY.inverse_transform(y_score_pred_flat).reshape(-1)

y_score_pred_inv = y_score_pred_inv.copy()
y_score_pred_inv[u_out_score == 1] = 0.0

print("Pred shape flat:", y_score_pred_inv.shape, " expected:", len(df_score))



## === cell 14
submission = pd.DataFrame({"id": ids, "pressure": y_score_pred_inv.astype(np.float32)})
submission.to_csv("submission.csv", index=False)

submission.head()



## === cell 15
out_path = os.path.abspath("submission.csv")
print("Saved:", out_path)
sub_check = pd.read_csv(out_path)
print(sub_check.head())
print(sub_check.columns.tolist(), "rows:", len(sub_check))
print(
    "id min/max:",
    sub_check["id"].min(),
    sub_check["id"].max(),
    "unique:",
    sub_check["id"].nunique(),
)
