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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler

print("Python:", os.sys.version)
print("TF:", tf.__version__)

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide sensibly
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## === cell 1
sc = StandardScaler()
rc = RobustScaler()




## === cell 2
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 3
def dropCols(df, cols):
    return df.drop(cols, axis=1)




## === cell 4
def preProcess(df):
    """
    Runtime: use groupby(...).cumsum/shift on Series directly (vectorized) and minimize copies.
    Correctness: identical feature definitions:
      - area = cumsum(time_step * u_in) per breath_id
      - u_in_cumsum = cumsum(u_in) per breath_id
      - u_in_lag1 = previous u_in within breath, 0 for first step
      - u_in_diff1 = u_in - u_in_lag1
    """
    df = df.copy()
    g = df.groupby("breath_id", sort=False)

    df["area"] = (
        (df["time_step"] * df["u_in"]).groupby(df["breath_id"], sort=False).cumsum()
    )
    df["u_in_cumsum"] = g["u_in"].cumsum()
    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0).astype(np.float32, copy=False)
    df["u_in_diff1"] = (df["u_in"] - df["u_in_lag1"]).astype(np.float32, copy=False)
    return df




## === cell 5
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




## === cell 6
cols_2_drop = ["id", "breath_id", "time_step"]




## === cell 7
train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")

print("Train features:", train_df.shape, "Target:", Y.shape)




## === cell 8
rc.set_params(copy=False)
X_train_np = train_df.to_numpy(copy=False)
rc.fit(X_train_np)
train_df = rc.transform(X_train_np)




## === cell 9
train_df = np.ascontiguousarray(train_df, dtype=np.float32)
Y = np.ascontiguousarray(Y.to_numpy(copy=False), dtype=np.float32)

train_df = train_df.reshape(-1, 80, train_df.shape[-1])
Y = Y.reshape(-1, 80, 1)

print("Train tensors:", train_df.shape, Y.shape)




## === cell 10
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_C_out_loss",
    factor=0.96,
    patience=4,
    verbose=1,
)




## === cell 11
def build_model():
    Input = layers.Input(shape=([80, train_df.shape[-1]]))

    m1x1 = layers.Bidirectional(
        layers.LSTM(
            units=440,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m1x2 = layers.Bidirectional(
        layers.LSTM(
            units=360,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m1x3 = layers.Bidirectional(
        layers.LSTM(
            units=240,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m1x4 = layers.Bidirectional(
        layers.LSTM(
            units=180,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m1x5 = layers.Bidirectional(
        layers.LSTM(
            units=100,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )

    m2x1 = layers.Bidirectional(
        layers.LSTM(
            units=880,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m2x3 = layers.Bidirectional(
        layers.LSTM(
            units=440,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )  #
    m2x5 = layers.Bidirectional(
        layers.LSTM(
            units=360,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )  #
    m2x7 = layers.Bidirectional(
        layers.LSTM(
            units=240,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )  #
    m2x9 = layers.Bidirectional(
        layers.LSTM(
            units=180,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )  #
    m2x11 = layers.Bidirectional(
        layers.LSTM(
            units=100,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )

    m3x1 = layers.Bidirectional(
        layers.GRU(
            units=440,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m3x2 = layers.Bidirectional(
        layers.GRU(
            units=360,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m3x3 = layers.Bidirectional(
        layers.GRU(
            units=240,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m3x4 = layers.Bidirectional(
        layers.GRU(
            units=180,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m3x5 = layers.Bidirectional(
        layers.GRU(
            units=100,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )

    m3x1_out = m3x1(Input)
    m3x2_out = m3x2(m3x1_out)
    m3x3_out = m3x3(m3x2_out)
    m3x4_out = m3x4(m3x3_out)
    m3x5_out = m3x5(m3x4_out)

    m2x1_out = m2x1(Input)
    m2x3_out = m2x3(m2x1_out)
    m2x5_out = m2x5(m2x3_out)
    m2x7_out = m2x7(m2x5_out)
    m2x9_out = m2x9(m2x7_out)
    m2x11_out = m2x11(m2x9_out)

    m1x1_out = m1x1(Input)
    m1x1_out = layers.Multiply()([m1x1_out, m2x3_out, m3x1_out])
    m1x1_out = layers.BatchNormalization()(m1x1_out)

    m1x2_out = m1x2(m1x1_out)
    m1x2_out = layers.Multiply()([m1x2_out, m2x5_out, m3x2_out])
    m1x2_out = layers.BatchNormalization()(m1x2_out)

    m1x3_out = m1x3(m1x2_out)
    m1x3_out = layers.Multiply()([m1x3_out, m2x7_out, m3x3_out])
    m1x3_out = layers.BatchNormalization()(m1x3_out)

    m1x4_out = m1x4(m1x3_out)
    m1x4_out = layers.Multiply()([m1x4_out, m2x9_out, m3x4_out])
    m1x4_out = layers.BatchNormalization()(m1x4_out)

    m1x5_out = m1x5(m1x4_out)

    f_mul = layers.Multiply()([m1x5_out, m2x11_out, m3x5_out])
    f_mul = layers.BatchNormalization()(f_mul)
    f_mul = layers.Bidirectional(
        layers.LSTM(units=100, dropout=0.2, return_sequences=True)
    )(f_mul)

    C_out = layers.Dense(
        units=64, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
    )(f_mul)
    C_out = layers.Dense(
        1, name="C_out", kernel_initializer=tf.keras.initializers.HeNormal()
    )(C_out)

    m1_out = layers.Dense(
        units=64, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
    )(m1x5_out)
    m1_out = layers.Dense(
        1, name="m1_out", kernel_initializer=tf.keras.initializers.HeNormal()
    )(m1_out)

    model = tf.keras.Model(inputs=Input, outputs=[C_out, m1_out], name="Base_Model")

    losses = {
        "C_out": tf.keras.losses.MeanAbsoluteError(),
        "m1_out": tf.keras.losses.MeanAbsoluteError(),
    }

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=losses)
    return model




## === cell 12
pass




## === cell 13
pass




## === cell 14
EPOCH = 300
BATCH_SIZE = 512

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print(
        "TPU not available, using default strategy:",
        type(strategy).__name__,
        "|",
        repr(e),
    )


def make_dataset(X, y, batch_size, training):
    """
    Runtime: use tf.data with prefetch and deterministic options; avoid per-epoch Python overhead.
    Correctness: same data, same shuffle seed, same reshuffle_each_iteration behavior, same batching.
    Note: no caching here to avoid double RAM usage on large arrays; keeping semantics identical.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, {"C_out": y, "m1_out": y}))
    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)
    if training:
        ds = ds.shuffle(
            buffer_size=min(len(X), 8192), seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def resolve_model_paths(base_names):
    candidates = []
    for bn in base_names:
        local_p = bn
        input_p = os.path.join("../input", bn)
        if os.path.exists(local_p):
            candidates.append(local_p)
        elif os.path.exists(input_p):
            candidates.append(input_p)
        else:
            candidates.append(local_p)  # default for saving during training
    return candidates


models_paths = resolve_model_paths(
    [f"M1_PressurePreModel{fold}.h5" for fold in [1, 2, 3]]
)
c_paths = resolve_model_paths([f"C_PressurePreModel{fold}.h5" for fold in [1, 2, 3]])

need_train = not all(os.path.exists(p) for p in (models_paths + c_paths))
print("Need training:", need_train)

if need_train:
    with strategy.scope():
        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
            print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)
            X_train, X_valid = train_df[train_idx], train_df[valid_idx]
            y_train, y_valid = Y[train_idx], Y[valid_idx]

            model = build_model()

            callback0 = tf.keras.callbacks.ModelCheckpoint(
                f"M1_PressurePreModel{fold+1}.h5",
                monitor="val_m1_out_loss",
                save_best_only=True,
                verbose=1,
            )
            callback1 = tf.keras.callbacks.ModelCheckpoint(
                f"C_PressurePreModel{fold+1}.h5",
                monitor="val_C_out_loss",
                save_best_only=True,
                verbose=1,
            )

            callbacks = [lr_callback, callback0, callback1]

            train_ds = make_dataset(X_train, y_train, BATCH_SIZE, training=True)
            valid_ds = make_dataset(X_valid, y_valid, BATCH_SIZE, training=False)

            history = model.fit(
                train_ds,
                validation_data=valid_ds,
                epochs=EPOCH,
                callbacks=callbacks,
                verbose=2,
            )




## === cell 15
models_paths = resolve_model_paths(
    [f"M1_PressurePreModel{fold}.h5" for fold in [1, 2, 3]]
)
existing_paths = [p for p in models_paths if os.path.exists(p)]
if not existing_paths:
    raise FileNotFoundError(
        f"No trained model checkpoints found. Looked for: {models_paths}"
    )

models = [tf.keras.models.load_model(p, compile=False) for p in existing_paths]
print("Loaded models:", existing_paths)




## === cell 16
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

test_ids = test_data["id"].values  # keep for sanity
test_data = dropCols(test_data, cols_2_drop)

X_test_np = test_data.to_numpy(copy=False)
test_data = rc.transform(X_test_np)
test_data = np.ascontiguousarray(test_data, dtype=np.float32)
test_data = test_data.reshape(-1, 80, test_data.shape[-1])

print("Test tensor:", test_data.shape, "First/last id:", test_ids[0], test_ids[-1])




## === cell 17
def make_test_dataset(X, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(X)
    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_test_dataset(test_data, batch_size=2048)

all_preds = []
for model in models:
    out = model.predict(test_ds, verbose=0)
    all_preds.append(out[1].astype(np.float32, copy=False))
preds = np.stack(all_preds, axis=0)
print("Preds stacked:", preds.shape)




## === cell 18
median_pre = np.median(preds, axis=0)
median_pre_flat = median_pre.reshape(-1)
print("Median pred flat:", median_pre_flat.shape)




## === cell 19
PRESSURE_STEP = 0.070302145
PRESSURE_MIN = 0.0
PRESSURE_MAX = 100.0

rounding_pre = (
    np.round((median_pre_flat - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
    + PRESSURE_MIN
)
clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)




## === cell 20
submission_file = pd.read_csv(sample_sub)

if len(submission_file) != len(clipped_pre):
    raise ValueError(
        f"Submission rows ({len(submission_file)}) != predictions ({len(clipped_pre)})"
    )

submission_file["pressure"] = clipped_pre.astype(np.float32)
submission_file.to_csv("submission.csv", index=False)

print(submission_file.head())
print("Wrote submission.csv with shape:", submission_file.shape)




## === cell 21
submission_file
