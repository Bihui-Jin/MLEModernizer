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

# 5. Target score

0.1638596430578128

# 6. Current score

17.65185

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 17.6111) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error seen in your first cell. Then I remove the dependency on missing pre-trained `.h5` files by running the existing (already provided) training code on CPU/GPU in this environment and using the trained model directly for prediction, preserving the same model architecture, losses, and output/rounding logic. I also correct the inference calls so we predict once with the multi-output model (instead of repeatedly and indexing mismatched outputs), which is score-neutral and prevents shape/index bugs. Finally, I ensure a valid `submission.csv` with `id,pressure` is always written.'
- What this solution (achieved 17.66404) has done: 'The timeout is dominated by TensorFlow training: you’re training a very large multi-output BiLSTM ensemble for 50 epochs on ~67k sequences, plus heavy callback checkpointing overhead. To keep identical training semantics while making it finish faster, I (1) enable fast TensorFlow execution (XLA JIT) and deterministic ops, (2) switch training/prediction inputs to efficient `tf.data` pipelines with caching/prefetching to eliminate Python/NumPy feeding overhead, and (3) reduce checkpoint I/O overhead by saving weights only (same “best” selection criteria, same monitored metrics) and avoiding legacy HDF5 serialization cost. I also speed up pandas feature engineering via vectorized groupby operations without changing the features produced, and reduce memory copies by using `float32` arrays throughout (training remains float32 in TF anyway). No model layers, losses, epoch count, batch size, feature definitions, or evaluation logic are changed.'
- What this solution (achieved 17.68145) has done: 'The timeout is dominated by extremely heavy training (four large BiLSTM towers plus ensembles) run for 50 epochs, so the only safe way to fit into 600s without changing the learning algorithm is to reduce per-step overhead and input pipeline costs. I keep the exact same model architecture, losses, and epoch loop, but speed it up by (1) removing expensive `Dataset.cache()` of huge in-memory tensors, (2) enabling deterministic-but-faster dataset traversal with `reshuffle_each_iteration=False`, (3) using a single compiled `train_one_epoch/valid_one_epoch` to cut Python loop overhead, and (4) avoiding the very expensive `tf.stack + broadcast_to` inside the loss by computing the same weighted MAE via a mathematically equivalent accumulation. I also avoid unnecessary pandas copies in feature engineering by operating on NumPy arrays where possible while keeping identical features and scaling.'
- What this solution (achieved 17.65185) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow (the current code unsets it, which triggers the `MessageFactory.GetPrototype` error). Then I fix the `tf.function` training crash by removing the symbolic loop index and computing the exact same weighted MAE via a Python-level unrolled sum over the 8 outputs (same loss semantics, just graph-safe). Finally, I keep your model/training/inference logic unchanged otherwise, ensuring training runs end-to-end and a valid `submission.csv` is written in the required `id,pressure` format.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_UPB"] = "1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.preprocessing import RobustScaler

np.random.seed(42)
tf.random.set_seed(42)

try:
    physical_gpus = tf.config.list_physical_devices("GPU")
    for g in physical_gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())
print("GPUs:", tf.config.list_physical_devices("GPU"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 2
def dropCols(df, cols):
    return df.drop(columns=cols)


def preProcess(df):
    g = df.groupby("breath_id", sort=False, observed=True)
    uin = df["u_in"]
    uin_lag1 = g["u_in"].shift(1)
    df["u_in_lag1"] = uin_lag1.fillna(0.0).astype(uin.dtype, copy=False)
    df["diff_u_in1"] = uin - df["u_in_lag1"]
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

train_data = pd.read_csv(train_path, dtype=train_dtypes, low_memory=False)
train_data = preProcess(train_data)

cols_2_drop = ["id", "breath_id", "time_step"]

train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")

rc = RobustScaler()
rc.fit(train_df)

X_np = rc.transform(train_df)
X_np = np.asarray(X_np, dtype=np.float32, order="C")
Y_np = Y.to_numpy(dtype=np.float32, copy=False)
del train_df, train_data, Y  # free memory early

X_np = X_np.reshape(-1, 80, X_np.shape[-1])
Y_np = Y_np.reshape(-1, 80, 1)

train_df = X_np
Y = Y_np

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

    losses = [
        tf.keras.losses.MeanAbsoluteError(),
        tf.keras.losses.MeanAbsoluteError(),
        tf.keras.losses.MeanAbsoluteError(),
        tf.keras.losses.MeanAbsoluteError(),
        tf.keras.losses.MeanAbsoluteError(),
        tf.keras.losses.MeanAbsoluteError(),
        tf.keras.losses.MeanAbsoluteError(),
        tf.keras.losses.MeanAbsoluteError(),
    ]
    loss_weights = tf.constant([1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.6, 1.0], tf.float32)

    model.compile(optimizer=opt, steps_per_execution=16)
    return model, opt, losses, loss_weights


model, opt, losses_list, loss_weights_vec = build_model()
model.summary()



## === cell 6
n = train_df.shape[0]
rng = np.random.RandomState(42)
perm = rng.permutation(n)
n_valid = int(round(0.2 * n))
valid_idx = perm[:n_valid]
train_idx = perm[n_valid:]

X_train = train_df[train_idx]
X_valid = train_df[valid_idx]
y_train = Y[train_idx]
y_valid = Y[valid_idx]

lr_reducer_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.8,
    patience=25,
    verbose=0,
)

callbacks = [lr_reducer_callback]



## === cell 7
EPOCH = 50
BATCH_SIZE = 1024

AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.deterministic = True

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).with_options(options)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).with_options(options)
valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
val_loss_metric = tf.keras.metrics.Mean(name="val_loss")


@tf.function(jit_compile=True)
def _total_weighted_mae(y_true, outs_list):
    y_true = tf.cast(y_true, tf.float32)
    total = tf.zeros([], tf.float32)

    o0 = tf.cast(outs_list[0], tf.float32)
    o1 = tf.cast(outs_list[1], tf.float32)
    o2 = tf.cast(outs_list[2], tf.float32)
    o3 = tf.cast(outs_list[3], tf.float32)
    o4 = tf.cast(outs_list[4], tf.float32)
    o5 = tf.cast(outs_list[5], tf.float32)
    o6 = tf.cast(outs_list[6], tf.float32)
    o7 = tf.cast(outs_list[7], tf.float32)

    total += tf.reduce_mean(tf.abs(y_true - o0)) * loss_weights_vec[0]
    total += tf.reduce_mean(tf.abs(y_true - o1)) * loss_weights_vec[1]
    total += tf.reduce_mean(tf.abs(y_true - o2)) * loss_weights_vec[2]
    total += tf.reduce_mean(tf.abs(y_true - o3)) * loss_weights_vec[3]
    total += tf.reduce_mean(tf.abs(y_true - o4)) * loss_weights_vec[4]
    total += tf.reduce_mean(tf.abs(y_true - o5)) * loss_weights_vec[5]
    total += tf.reduce_mean(tf.abs(y_true - o6)) * loss_weights_vec[6]
    total += tf.reduce_mean(tf.abs(y_true - o7)) * loss_weights_vec[7]
    return total


@tf.function(jit_compile=True)
def train_step(x, y):
    with tf.GradientTape() as tape:
        outs = model(x, training=True)
        total_loss = _total_weighted_mae(y, outs)
    grads = tape.gradient(total_loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    train_loss_metric.update_state(total_loss)


@tf.function(jit_compile=True)
def valid_step(x, y):
    outs = model(x, training=False)
    total_loss = _total_weighted_mae(y, outs)
    val_loss_metric.update_state(total_loss)


@tf.function(jit_compile=True)
def train_one_epoch(ds):
    for xb, yb in ds:
        train_step(xb, yb)


@tf.function(jit_compile=True)
def valid_one_epoch(ds):
    for xb, yb in ds:
        valid_step(xb, yb)


history = {"loss": [], "val_loss": []}

lr_reducer_callback.set_model(model)
lr_reducer_callback.on_train_begin()

for epoch in range(EPOCH):
    train_loss_metric.reset_state()
    val_loss_metric.reset_state()

    train_one_epoch(train_ds)
    valid_one_epoch(valid_ds)

    epoch_train_loss = float(train_loss_metric.result().numpy())
    epoch_val_loss = float(val_loss_metric.result().numpy())
    history["loss"].append(epoch_train_loss)
    history["val_loss"].append(epoch_val_loss)

    lr_reducer_callback.on_epoch_end(epoch, logs={"val_loss": epoch_val_loss})

    print(
        f"Epoch {epoch+1}/{EPOCH} - loss: {epoch_train_loss:.6f} - val_loss: {epoch_val_loss:.6f}"
    )

lr_reducer_callback.on_train_end()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/1375688662.py in <cell line: 0>()
     82     val_loss_metric.reset_state()
     83 
---> 84     train_one_epoch(train_ds)
     85     valid_one_epoch(valid_ds)
     86 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Detected unsupported operations when trying to compile graph __inference_train_one_epoch_96273[_XlaMustCompile=true,config_proto=4469435546627048257,executor_type=11160318154034397263] on XLA_CPU_JIT: ScanDataset (No registered 'ScanDataset' OpKernel for XLA_CPU_JIT devices compatible with node {{node ScanDataset}}){{node ScanDataset}}
The op is created at: 
File "<frozen runpy>", line 198, in _run_module_as_main
File "<frozen runpy>", line 88, in _run_code
File "/usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py", line 37, in <module>
File "/usr/local/lib/python3.11/dist-packages/traitlets/config/application.py", line 992, in launch_instance
File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelapp.py", line 712, in start
File "/usr/local/lib/python3.11/dist-packages/tornado/platform/asyncio.py", line 211, in start
File "/usr/lib/python3.11/asyncio/base_events.py", line 608, in run_forever
File "/usr/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once
File "/usr/lib/python3.11/asyncio/events.py", line 84, in _run
File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 510, in dispatch_queue
File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 499, in process_one
File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 406, in dispatch_shell
File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 730, in execute_request
File "/usr/local/lib/python3.11/dist-packages/ipykernel/ipkernel.py", line 383, in do_execute
File "/usr/local/lib/python3.11/dist-packages/ipykernel/zmqshell.py", line 528, in run_cell
File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 2975, in run_cell
File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3030, in _run_cell
File "/usr/local/lib/python3.11/dist-packages/IPython/core/async_helpers.py", line 78, in _pseudo_sync_runner
File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3257, in run_cell_async
File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3473, in run_ast_nodes
File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code
File "/tmp/ipykernel_11/1375688662.py", line 84, in <cell line: 0>
File "/tmp/ipykernel_11/1375688662.py", line 65, in train_one_epoch
	tf2xla conversion failed while converting __inference_train_one_epoch_96273[_XlaMustCompile=true,config_proto=4469435546627048257,executor_type=11160318154034397263]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions. [Op:__inference_train_one_epoch_96273]

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

test_data = pd.read_csv(test_path, dtype=test_dtypes, low_memory=False)
test_data = preProcess(test_data)

test_ids = test_data["id"].to_numpy(copy=False)

test_features = dropCols(test_data, cols_2_drop)

test_features = rc.transform(test_features)
test_features = np.asarray(test_features, dtype=np.float32, order="C")
del test_data

test_features = test_features.reshape(-1, 80, test_features.shape[-1])

print("Test features shape:", test_features.shape)



## === cell 9
PRED_BATCH = 2048

pred_options = tf.data.Options()
pred_options.deterministic = True

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_features)
    .with_options(pred_options)
    .batch(PRED_BATCH, drop_remainder=False)
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
unique_pressures = np.unique(Y.ravel())
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
if len(test_ids) != len(rounding_pre):
    raise ValueError(
        f"Submission length mismatch: test={len(test_ids)} preds={len(rounding_pre)}"
    )

submission_file = pd.DataFrame(
    {"id": test_ids, "pressure": rounding_pre.astype(np.float32)}
)
submission_file.to_csv("submission.csv", index=False)

print(submission_file.head())
print("Wrote submission.csv", os.path.getsize("submission.csv"), "bytes")
