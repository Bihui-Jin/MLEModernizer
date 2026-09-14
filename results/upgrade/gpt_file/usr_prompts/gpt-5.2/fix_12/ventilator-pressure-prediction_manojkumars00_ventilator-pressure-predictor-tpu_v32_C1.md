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
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import math
import gc
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_experimental_options(
        {
            "layout_optimizer": True,
            "constant_folding": True,
            "shape_optimization": True,
            "remapping": True,
            "arithmetic_optimization": True,
            "dependency_optimization": True,
        }
    )
except Exception:
    pass

rc = RobustScaler()




## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 2
def dropCols(df, cols):
    return df.drop(columns=cols)




## === cell 3
def preProcess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    n_rows = len(df)
    if n_rows % 80 != 0:
        u_in = df["u_in"].astype(np.float32, copy=False)
        time_step = df["time_step"].astype(np.float32, copy=False)
        g = df.groupby("breath_id", sort=False)

        df["area"] = (
            (time_step * u_in)
            .groupby(df["breath_id"], sort=False)
            .cumsum()
            .astype(np.float32)
        )
        df["u_in_cumsum"] = g["u_in"].cumsum().astype(np.float32)

        u_in_lag1 = g["u_in"].shift(1).fillna(0.0).astype(np.float32)
        df["u_in_lag1"] = u_in_lag1
        df["u_in_diff1"] = (u_in - u_in_lag1).astype(np.float32)
        return df

    n_breaths = n_rows // 80

    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False).reshape(n_breaths, 80)
    time_step = (
        df["time_step"].to_numpy(dtype=np.float32, copy=False).reshape(n_breaths, 80)
    )

    area = np.cumsum(time_step * u_in, axis=1, dtype=np.float32)
    u_in_cumsum = np.cumsum(u_in, axis=1, dtype=np.float32)

    u_in_lag1 = np.zeros_like(u_in, dtype=np.float32)
    u_in_lag1[:, 1:] = u_in[:, :-1]
    u_in_diff1 = (u_in - u_in_lag1).astype(np.float32, copy=False)

    df["area"] = area.reshape(-1)
    df["u_in_cumsum"] = u_in_cumsum.reshape(-1)
    df["u_in_lag1"] = u_in_lag1.reshape(-1)
    df["u_in_diff1"] = u_in_diff1.reshape(-1)

    return df




## === cell 4
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
train_data = pd.read_csv(train_path, dtype=train_dtypes, engine="c")
train_data = preProcess(train_data)




## === cell 5
cols_2_drop = ["id", "breath_id", "time_step"]




## === cell 6
train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")




## === cell 7
rc.fit(train_df)
train_arr_2d = rc.transform(train_df)  # float64 by sklearn; cast once below

del train_data, train_df
gc.collect()

train_arr = np.ascontiguousarray(train_arr_2d, dtype=np.float32).reshape(
    -1, 80, train_arr_2d.shape[-1]
)
del train_arr_2d
Y = np.ascontiguousarray(Y.to_numpy(dtype=np.float32)).reshape(-1, 80, 1)

print("train_df:", train_arr.shape, "Y:", Y.shape)




## === cell 8
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="C_out_loss",
    factor=0.96,
    patience=4,
    verbose=1,
)




## === cell 9
def build_model():
    Input = layers.Input(shape=([80, train_arr.shape[-1]]))

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
    )
    m2x5 = layers.Bidirectional(
        layers.LSTM(
            units=360,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m2x7 = layers.Bidirectional(
        layers.LSTM(
            units=240,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
    m2x9 = layers.Bidirectional(
        layers.LSTM(
            units=180,
            dropout=0.2,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
    )
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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=losses,
        run_eagerly=False,
        jit_compile=True,
    )
    return model




## === cell 10
try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # auto-detect
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
    print("Running on TPU")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print("Running on default strategy (CPU/GPU). TPU not available:", str(e)[:200])




## === cell 11
EPOCH = 300
BATCH_SIZE = 512

best_m1_paths = []
best_c_paths = []

AUTOTUNE = tf.data.AUTOTUNE


def make_index_dataset(X, y, indices, batch_size, training: bool):
    idx = tf.convert_to_tensor(indices, dtype=tf.int32)

    def _gather(i):
        xi = tf.gather(X, i)
        yi = tf.gather(y, i)
        return xi, {"C_out": yi, "m1_out": yi}

    ds = tf.data.Dataset.from_tensor_slices(idx)
    if training:
        ds = ds.shuffle(
            buffer_size=idx.shape[0], seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds.map(_gather, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


with strategy.scope():
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_arr)):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

        model = build_model()

        m1_path = f"M1_PressurePreModel{fold+1}.h5"
        c_path = f"C_PressurePreModel{fold+1}.h5"

        callback0 = tf.keras.callbacks.ModelCheckpoint(
            m1_path, monitor="val_m1_out_loss", save_best_only=True, verbose=1
        )
        callback1 = tf.keras.callbacks.ModelCheckpoint(
            c_path, monitor="val_C_out_loss", save_best_only=True, verbose=1
        )
        callbacks = [lr_callback, callback0, callback1]

        if os.path.exists(m1_path) and os.path.exists(c_path):
            print(f"Found existing checkpoints for fold {fold+1}; skipping fit().")
        else:
            train_ds = make_index_dataset(
                train_arr, Y, train_idx, BATCH_SIZE, training=True
            )
            valid_ds = make_index_dataset(
                train_arr, Y, valid_idx, BATCH_SIZE, training=False
            )

            model.fit(
                train_ds,
                validation_data=valid_ds,
                epochs=EPOCH,
                callbacks=callbacks,
                verbose=2,
            )

            del train_ds, valid_ds
            gc.collect()

        best_m1_paths.append(m1_path)
        best_c_paths.append(c_path)

        del model
        gc.collect()
        tf.keras.backend.clear_session()




## === cell 12
test_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}
test_data = pd.read_csv(test_path, dtype=test_dtypes, engine="c")
test_data = preProcess(test_data)

test_data = dropCols(test_data, cols_2_drop)
test_arr_2d = rc.transform(test_data)

del test_data
gc.collect()

test_arr = np.ascontiguousarray(test_arr_2d, dtype=np.float32).reshape(
    -1, 80, test_arr_2d.shape[-1]
)
del test_arr_2d
gc.collect()

print("test_data:", test_arr.shape)




## === cell 13
models = []
for fold in range(3):
    m1_path = f"M1_PressurePreModel{fold+1}.h5"
    if os.path.exists(m1_path):
        models.append(tf.keras.models.load_model(m1_path, compile=False))
    else:
        raise FileNotFoundError(f"Missing checkpoint: {m1_path}")

print("Models to ensemble:", len(models))




## === cell 14
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_arr).batch(512).prefetch(tf.data.AUTOTUNE)
)

preds_c = []
preds_m1 = []
for model in models:
    pred = model.predict(test_ds, verbose=0)
    if isinstance(pred, (list, tuple)) and len(pred) == 2:
        preds_c.append(pred[0][..., 0])  # (n,80)
        preds_m1.append(pred[1][..., 0])  # (n,80)
    else:
        preds_c.append(pred[..., 0])
        preds_m1.append(pred[..., 0])

preds_c = np.stack(preds_c, axis=-1)  # (n,80,3)
preds_m1 = np.stack(preds_m1, axis=-1)  # (n,80,3)

median_pre0 = np.median(preds_c, axis=-1)  # (n,80)
median_pre1 = np.median(preds_m1, axis=-1)  # (n,80)

del preds_c, preds_m1, test_ds
gc.collect()




## === cell 15
y_flat = Y.reshape(-1)
unique_pressures = np.unique(y_flat)
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = (unique_pressures[1] - unique_pressures[0]).item()
PRESSURE_MIN = sorted_pressures[0].item()
PRESSURE_MAX = sorted_pressures[-1].item()

rounding_pre0 = (
    np.round((median_pre0 - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
    + PRESSURE_MIN
)
rounding_pre1 = (
    np.round((median_pre1 - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
    + PRESSURE_MIN
)

clipped_pre0 = np.clip(rounding_pre0, PRESSURE_MIN, PRESSURE_MAX).reshape(-1, 1)
clipped_pre1 = np.clip(rounding_pre1, PRESSURE_MIN, PRESSURE_MAX).reshape(-1, 1)

print("clipped_pre0:", clipped_pre0.shape, "clipped_pre1:", clipped_pre1.shape)




## === cell 16
submission_file = pd.read_csv(sample_sub, engine="c")

submission_file["pressure"] = clipped_pre0
submission_file.to_csv("submission0.csv", index=False)

submission_file["pressure"] = clipped_pre1
submission_file.to_csv("submission1.csv", index=False)

submission_file.to_csv("submission.csv", index=False)

print("Wrote: submission.csv, submission0.csv, submission1.csv")
print(submission_file.head())
