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

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
np.random.seed(0)

dtype = {
    "id": np.int32,
    "breath_id": np.int32,
    "R": np.float32,
    "C": np.float32,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.float32,
    "pressure": np.float32,
}

usecols = ["breath_id", "R", "C", "u_in", "u_out", "pressure"]
df = pd.read_csv(
    "/kaggle/input/ventilator-pressure-prediction/train.csv",
    dtype=dtype,
    usecols=usecols,
    index_col=["breath_id"],
)
df



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

df_graph = df.iloc[0:1200].reset_index()
df_graph[["R", "C", "u_out", "u_in", "pressure"]].plot(subplots=True)



## === cell 4
df_graph["div"] = df_graph["u_in"] / df_graph["pressure"]
df_graph["div"].plot()



## === cell 5
df_graph = df.iloc[0:80].reset_index()
df_graph[["pressure", "u_in"]].plot()



## === cell 6
from sklearn.preprocessing import MinMaxScaler

scalerX = MinMaxScaler(feature_range=(0, 1))
scalerY = MinMaxScaler(feature_range=(0, 1))

df_scaled = pd.DataFrame(
    scalerX.fit_transform(df[features]).astype(np.float32, copy=False),
    columns=features,
    index=df.index,
)
df_scaled[graal] = scalerY.fit_transform(df[[graal]]).astype(np.float32, copy=False)
df_scaled




## === cell 7
def split_fixed_80(df1: pd.DataFrame) -> np.ndarray:
    arr = df1.to_numpy(dtype=np.float32, copy=False)
    n = arr.shape[0]
    if n % 80 != 0:
        return np.stack(
            [
                g.to_numpy(dtype=np.float32, copy=False)
                for _, g in df1.groupby(level=0, sort=False)
            ]
        )
    return arr.reshape(n // 80, 80, arr.shape[1])


breath_ids = df_scaled.index.unique()
train_breaths = int(len(breath_ids) * 0.92)

train_ids = breath_ids[:train_breaths]
test_ids = breath_ids[train_breaths:]

df_train = df_scaled.loc[train_ids]
df_test = df_scaled.loc[test_ids]

X_train, X_test = split_fixed_80(df_train[features]), split_fixed_80(df_test[features])
y_train, y_test = split_fixed_80(df_train[[graal]]), split_fixed_80(df_test[[graal]])

print("train :", X_train.shape, " -> ", y_train.shape)
print("test :", X_test.shape, " -> ", y_test.shape)



## === cell 8
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys

for m in list(sys.modules.keys()):
    if m == "google.protobuf" or m.startswith("google.protobuf."):
        del sys.modules[m]

try:
    import google.protobuf.message_factory as _mf

    def _patch_message_factory_class(cls):
        if cls is None or hasattr(cls, "GetPrototype"):
            return
        if hasattr(cls, "GetMessageClass"):

            def GetPrototype(self, descriptor):
                return self.GetMessageClass(descriptor)

            cls.GetPrototype = GetPrototype
        else:
            def GetPrototype(self, descriptor):
                return None

            cls.GetPrototype = GetPrototype

    _patch_message_factory_class(getattr(_mf, "MessageFactory", None))
    gen = getattr(_mf, "_GENERATED_MESSAGE_FACTORY", None)
    if gen is not None:
        _patch_message_factory_class(gen.__class__)
except Exception:
    pass

import tensorflow as tf

tf.keras.utils.set_random_seed(0)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    Conv1DTranspose,
    LayerNormalization,
    Dropout,
)


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
        model.add(
            Conv1DTranspose(
                kernel, ksize, padding="same", strides=2, activation=activation
            )
        )

    for layer in range(1, dense):
        model.add(Dropout(0.1))
        model.add(Dense(1))

    model.add(Dropout(0.1))
    model.add(Dense(1))
    return model


model = build_model(10, 10, 2, 0.1)
model.summary()


## === cell 9
learning_rate = 0.00018


def tuner_model(hp):
    ksize = hp.Choice("ksize", [2, 5, 7])
    kernel = hp.Choice("kernel", [64, 128, 256])
    dense = hp.Choice("dense", [1, 2, 3])
    dropout = hp.Choice("dropout", [0.1, 0.2, 0.3, 0.5])

    model = build_model(ksize, kernel, dense, dropout)
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mean_absolute_error")
    return model


RUN_TUNER = os.environ.get("RUN_TUNER", "0") == "1"

best_model = None
if RUN_TUNER:
    os.system("rm -Rf untitled_project/")
    tuner = keras_tuner.BayesianOptimization(
        tuner_model, objective="val_loss", max_trials=30, overwrite=True
    )
    tuner.search(
        X_train, y_train, validation_data=(X_test, y_test), epochs=1, verbose=1
    )
    tuner.results_summary()
    best_model = tuner.get_best_models()[0]



## === cell 10
if best_model is not None:
    model = best_model
else:
    model = build_model(7, 256, 1, 0.1)

model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_absolute_error")
model.summary()


## === cell 11
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

model_file = "checkpoint.h5"

early = EarlyStopping(monitor="val_loss", min_delta=0, patience=4, mode="auto")
checkpoint = ModelCheckpoint(
    model_file,
    monitor="val_loss",
    save_best_only=True,
    verbose=0,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
)
rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.9, patience=2, verbose=1)

callbacks = [rlrop, checkpoint, early]

hist = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    verbose=1,
    batch_size=32,
    epochs=100,
    callbacks=callbacks,
)



## === cell 12
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(hist.history["loss"], label="mean absolute error")
plt.plot(hist.history["val_loss"], label="val mean absolute error")
plt.ylabel("Metric")
plt.xlabel("Epoch")
plt.legend(loc="upper left")
plt.show()



## === cell 13
from tensorflow import keras
from sklearn.metrics import mean_absolute_error

model = keras.models.load_model(model_file)

y_test_pred = model.predict(X_test, verbose=1)
y_test_pred = scalerY.inverse_transform(y_test_pred.reshape(-1, y_test_pred.shape[-1]))



## === cell 14
df_test_pred = pd.DataFrame(index=df_test.index)
df_test_pred["pressure"] = scalerY.inverse_transform(df_test[[graal]])
df_test_pred["pred"] = y_test_pred
df_test_pred["diff"] = df_test_pred["pressure"] - df_test_pred["pred"]
df_test_pred



## === cell 15
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 6)

df_graph = df_test_pred.iloc[0:1200].reset_index()
df_graph[["pressure", "pred"]].plot()
df_graph[["diff"]].plot()



## === cell 16
trainScore = mean_absolute_error(df_test_pred["pressure"], df_test_pred["pred"])
print("Test Score: %.2f MAE" % (trainScore))



## === cell 17
usecols_test = ["breath_id", "R", "C", "u_in", "u_out"]
df_score = pd.read_csv(
    "/kaggle/input/ventilator-pressure-prediction/test.csv",
    dtype=dtype,
    usecols=usecols_test,
    index_col=["breath_id"],
)
df_score



## === cell 18
from tensorflow import keras

model = keras.models.load_model(model_file)

df_scaled_score = pd.DataFrame(
    scalerX.transform(df_score[features]).astype(np.float32, copy=False),
    columns=features,
    index=df_score.index,
)
X_score = split_fixed_80(df_scaled_score)

y_score_pred = model.predict(X_score, verbose=1)
y_score_pred = scalerY.inverse_transform(
    y_score_pred.reshape(-1, y_score_pred.shape[-1])
)



## === cell 19
submission = pd.read_csv(
    "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"
)
submission["pressure"] = y_score_pred
submission.to_csv("submission.csv", index=False)
