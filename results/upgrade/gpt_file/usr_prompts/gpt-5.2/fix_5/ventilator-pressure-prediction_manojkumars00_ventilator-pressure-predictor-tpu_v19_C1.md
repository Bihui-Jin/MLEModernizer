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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_UPB"] = "1"

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 2
def dropCols(df, cols):
    df = df.copy()
    df.drop(cols, axis=1, inplace=True)
    return df


def preProcess(df):
    df = df.copy()
    g = df.groupby("breath_id", sort=False)
    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0)
    df["diff_u_in1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_cumsum"] = g["u_in"].cumsum()
    return df




## === cell 3
train_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
train_data = pd.read_csv(train_path, dtype=train_dtypes)
train_data = preProcess(train_data)

cols_2_drop = ["id", "breath_id", "time_step"]

train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")

rc = RobustScaler()
rc.fit(train_df)

train_df = rc.transform(train_df).astype(np.float32, copy=False)

train_df = train_df.reshape(-1, 80, train_df.shape[-1])
Y = Y.values.astype(np.float32, copy=False).reshape(-1, 80, 1)

print("X shape:", train_df.shape, "Y shape:", Y.shape)




## === cell 4
def build_model0(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(
        layers.Bidirectional(
            layers.LSTM(440, return_sequences=True, input_shape=input_shape)
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                360,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                260,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                180,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(layers.Bidirectional(layers.LSTM(100, return_sequences=True)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1, name=name))
    return model


def build_model1(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(
        layers.Bidirectional(
            layers.LSTM(440, return_sequences=True, input_shape=input_shape)
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                360,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                260,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                180,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                100,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(layers.Bidirectional(layers.LSTM(80, return_sequences=True)))
    model.add(
        layers.Dense(
            64, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
        )
    )
    model.add(layers.Dense(1, name=name))
    return model


def build_model2(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(
        layers.Bidirectional(
            layers.LSTM(420, return_sequences=True, input_shape=input_shape)
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                340,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                260,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                180,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                100,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(layers.Bidirectional(layers.LSTM(20, return_sequences=True)))
    model.add(
        layers.TimeDistributed(
            layers.Dense(
                64,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )
        )
    )
    model.add(layers.TimeDistributed(layers.Dense(1, name=name)))
    return model


def build_model3(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(
        layers.Bidirectional(
            layers.LSTM(280, return_sequences=True, input_shape=input_shape)
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                200,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                160,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                140,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                100,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(layers.Bidirectional(layers.LSTM(60, return_sequences=True)))
    model.add(
        layers.Dense(
            64, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
        )
    )
    model.add(layers.Dense(1, name=name))
    return model


def build_ensembleModel0(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(
        layers.Dense(
            8,
            activation="relu",
            kernel_initializer=tf.keras.initializers.HeNormal(),
            input_shape=input_shape,
        )
    )
    model.add(layers.Dense(1, name=name))
    return model


def build_ensembleModel1(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(
        layers.Dense(
            8,
            activation="relu",
            kernel_initializer=tf.keras.initializers.HeNormal(),
            input_shape=input_shape,
        )
    )
    model.add(layers.Dense(1, name=name))
    return model


def buildMasterEnsemble(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(
        layers.Dense(
            4, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
        )
    )
    model.add(layers.Dense(1, name=name))
    return model




## === cell 5
def build_model(input_size=(80, train_df.shape[-1])):
    model0 = build_model0(input_size, name="model0")
    model1 = build_model1(input_size, name="model1")
    model2 = build_model2(input_size, name="model2")
    model3 = build_model2(input_size, name="model3")

    ensemblemodel0 = build_ensembleModel0((80, 5), name="ensemble0")
    ensemblemodel1 = build_ensembleModel1((80, 5), name="ensemble1")
    masterensemble = buildMasterEnsemble((80, 2), name="master_ensemble")

    Input = tf.keras.layers.Input(input_size)

    model0_out = model0(Input)
    model1_out = model1(Input)
    model2_out = model2(Input)
    model3_out = model3(Input)

    mean_out = layers.Add()([model0_out, model1_out, model2_out, model3_out])
    mean_out = layers.Lambda(lambda x: x / 4, name="mean")(mean_out)

    concatenateModelsout = layers.Concatenate()(
        [model0_out, model1_out, model2_out, model3_out, mean_out]
    )
    ensemble_out0 = ensemblemodel0(concatenateModelsout)
    ensemble_out1 = ensemblemodel1(concatenateModelsout)

    concatenateensemble_out = layers.Concatenate()([ensemble_out0, ensemble_out1])
    master_ensemble_out = masterensemble(concatenateensemble_out)

    model = tf.keras.Model(
        inputs=Input,
        outputs=[
            model0_out,
            model1_out,
            model2_out,
            model3_out,
            mean_out,
            ensemble_out0,
            ensemble_out1,
            master_ensemble_out,
        ],
    )

    opt = tf.keras.optimizers.Adam()

    losses = {
        "model0": tf.keras.losses.MeanAbsoluteError(),
        "model1": tf.keras.losses.MeanAbsoluteError(),
        "model2": tf.keras.losses.MeanAbsoluteError(),
        "model3": tf.keras.losses.MeanAbsoluteError(),
        "mean": tf.keras.losses.MeanAbsoluteError(),
        "ensemble0": tf.keras.losses.MeanAbsoluteError(),
        "ensemble1": tf.keras.losses.MeanAbsoluteError(),
        "master_ensemble": tf.keras.losses.MeanAbsoluteError(),
    }

    model.compile(
        optimizer=opt,
        loss=losses,
        loss_weights={
            "model0": 1.0,
            "model1": 1.0,
            "model2": 1.0,
            "model3": 1.0,
            "mean": 0.8,
            "ensemble0": 0.6,
            "ensemble1": 0.6,
            "master_ensemble": 1.0,
        },
    )
    return model


model = build_model()
model.summary()



## === cell 6
X_train, X_valid, y_train, y_valid = train_test_split(
    train_df, Y, test_size=0.2, random_state=42
)

lr_reducer_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.8,
    patience=25,
    verbose=1,
)

callback1 = tf.keras.callbacks.ModelCheckpoint(
    "Bestmodel0.weights.h5",
    monitor="val_model0_loss",
    save_best_only=True,
    verbose=1,
    save_weights_only=True,
)
callback2 = tf.keras.callbacks.ModelCheckpoint(
    "Bestmodel1.weights.h5",
    monitor="val_model1_loss",
    save_best_only=True,
    verbose=1,
    save_weights_only=True,
)
callback3 = tf.keras.callbacks.ModelCheckpoint(
    "Bestmodel2.weights.h5",
    monitor="val_model2_loss",
    save_best_only=True,
    verbose=1,
    save_weights_only=True,
)
callback4 = tf.keras.callbacks.ModelCheckpoint(
    "Bestmodel3.weights.h5",
    monitor="val_model3_loss",
    save_best_only=True,
    verbose=1,
    save_weights_only=True,
)
callback5 = tf.keras.callbacks.ModelCheckpoint(
    "Bestensemble0.weights.h5",
    monitor="val_ensemble0_loss",
    save_best_only=True,
    verbose=1,
    save_weights_only=True,
)
callback6 = tf.keras.callbacks.ModelCheckpoint(
    "Bestensemble1.weights.h5",
    monitor="val_ensemble1_loss",
    save_best_only=True,
    verbose=1,
    save_weights_only=True,
)
callback7 = tf.keras.callbacks.ModelCheckpoint(
    "BestEnsembleModel.weights.h5",
    monitor="val_master_ensemble_loss",
    save_best_only=True,
    verbose=1,
    save_weights_only=True,
)
callback8 = tf.keras.callbacks.ModelCheckpoint(
    "BestMeanModel.weights.h5",
    monitor="val_mean_loss",
    save_best_only=True,
    verbose=1,
    save_weights_only=True,
)

callbacks = [
    lr_reducer_callback,
    callback1,
    callback2,
    callback3,
    callback4,
    callback5,
    callback6,
    callback7,
    callback8,
]



## === cell 7
EPOCH = 50
BATCH_SIZE = 1024

y_train_multi = {
    "model0": y_train,
    "model1": y_train,
    "model2": y_train,
    "model3": y_train,
    "mean": y_train,
    "ensemble0": y_train,
    "ensemble1": y_train,
    "master_ensemble": y_train,
}
y_valid_multi = {
    "model0": y_valid,
    "model1": y_valid,
    "model2": y_valid,
    "model3": y_valid,
    "mean": y_valid,
    "ensemble0": y_valid,
    "ensemble1": y_valid,
    "master_ensemble": y_valid,
}

AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_multi))
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).cache().prefetch(AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid_multi))
valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=False).cache().prefetch(AUTOTUNE)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCH,
    callbacks=callbacks,
    verbose=2,
)



## === cell 8
test_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}
test_data = pd.read_csv(test_path, dtype=test_dtypes)
test_data = preProcess(test_data)
test_features = dropCols(test_data, cols_2_drop)
test_features = rc.transform(test_features).astype(np.float32, copy=False)
test_features = test_features.reshape(-1, 80, test_features.shape[-1])

print("Test features shape:", test_features.shape)



## === cell 9
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_features)
    .batch(1024)
    .prefetch(tf.data.AUTOTUNE)
)
preds = model.predict(test_ds, verbose=1)

pred_model0 = preds[0].reshape(-1, 1)
pred_model1 = preds[1].reshape(-1, 1)
pred_model2 = preds[2].reshape(-1, 1)
pred_model3 = preds[3].reshape(-1, 1)
pred_mean = preds[4].reshape(-1, 1)
pred_ens0 = preds[5].reshape(-1, 1)
pred_ens1 = preds[6].reshape(-1, 1)
pred_master = preds[7].reshape(-1, 1)

predictions_table = np.concatenate(
    [
        pred_model0,
        pred_model1,
        pred_model2,
        pred_model3,
        pred_ens0,
        pred_ens1,
        pred_mean,
        pred_master,
    ],
    axis=-1,
)
median_pre = np.median(predictions_table, axis=-1)

print("Median preds shape:", median_pre.shape)



## === cell 10
unique_pressures = np.unique(Y)
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = (unique_pressures[1] - unique_pressures[0]).item()
PRESSURE_MIN = sorted_pressures[0].item()
PRESSURE_MAX = sorted_pressures[-1].item()

rounding_pre = (
    np.round((median_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
rounding_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)

print("Rounded preds shape:", rounding_pre.shape)



## === cell 11
submission_file = pd.read_csv(sample_sub_path)
if len(submission_file) != len(rounding_pre):
    raise ValueError(
        f"Submission length mismatch: sample={len(submission_file)} preds={len(rounding_pre)}"
    )

submission_file["pressure"] = rounding_pre.astype(np.float32)
submission_file.to_csv("submission.csv", index=False)

print(submission_file.head())
print("Wrote submission.csv", os.path.getsize("submission.csv"), "bytes")
