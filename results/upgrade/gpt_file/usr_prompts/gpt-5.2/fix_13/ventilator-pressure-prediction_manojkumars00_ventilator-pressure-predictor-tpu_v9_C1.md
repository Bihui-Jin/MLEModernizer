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
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import KFold

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.get_logger().setLevel("ERROR")

try:
    from tensorflow.keras import mixed_precision

    gpus = tf.config.list_physical_devices("GPU")
    tpus = tf.config.list_logical_devices("TPU")
    if gpus or tpus:
        mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## === cell 1
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

from sklearn.preprocessing import RobustScaler

rc = RobustScaler()




## === cell 2
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 3
def dropCols(df, cols):
    return df.drop(columns=cols)




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
train_data = pd.read_csv(train_path, dtype=train_dtypes)

u_in = train_data["u_in"].to_numpy(dtype=np.float32, copy=False)
n_rows = u_in.shape[0]
if n_rows % 80 != 0:
    raise ValueError(f"Expected train rows multiple of 80, got {n_rows}")
u_in_2d = u_in.reshape(-1, 80)

u_in_lag1_2d = np.empty_like(u_in_2d)
u_in_lag1_2d[:, 0] = 0.0
u_in_lag1_2d[:, 1:] = u_in_2d[:, :-1]
train_data["u_in_lag1"] = u_in_lag1_2d.reshape(-1)

train_data["diff_u_in1"] = (u_in_2d - u_in_lag1_2d).reshape(-1)
train_data["u_in_cumsum"] = np.cumsum(u_in_2d, axis=1).reshape(-1)




## === cell 5
cols_2_drop = ["id", "breath_id", "time_step"]




## === cell 6
train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")




## === cell 7
train_df.shape




## === cell 8
train_df.isna().sum()




## === cell 9
rc.fit(train_df)
train_df = np.ascontiguousarray(rc.transform(train_df), dtype=np.float32)




## === cell 10
train_df = np.ascontiguousarray(
    train_df.reshape(-1, 80, train_df.shape[-1]), dtype=np.float32
)
Y = np.ascontiguousarray(
    Y.to_numpy(dtype=np.float32, copy=False).reshape(-1, 80, 1), dtype=np.float32
)




## === cell 11
train_df.shape, Y.shape




## === cell 12
def build_model():
    model = tf.keras.Sequential()

    model.add(
        layers.Bidirectional(
            layers.LSTM(120, return_sequences=True),
            input_shape=[80, train_df.shape[-1]],
        )
    )
    model.add(layers.Bidirectional(layers.LSTM(180, return_sequences=True)))
    model.add(
        layers.Bidirectional(layers.LSTM(280, dropout=0.2, return_sequences=True))
    )
    model.add(
        layers.Bidirectional(layers.LSTM(360, dropout=0.2, return_sequences=True))
    )
    model.add(
        layers.Bidirectional(layers.LSTM(500, dropout=0.2, return_sequences=True))
    )
    model.add(
        layers.Bidirectional(layers.LSTM(1000, dropout=0.2, return_sequences=True))
    )

    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.TimeDistributed(layers.Dense(1)))

    opt = tf.keras.optimizers.Adam()

    if tf.keras.mixed_precision.global_policy().compute_dtype == "float16":
        model.layers[-1].dtype_policy = tf.keras.mixed_precision.Policy("float32")

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=["mae"],
        jit_compile=True,
        steps_per_execution=32,
    )
    return model




## === cell 13
build_model().summary()




## === cell 14
callback1 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.9,
    patience=10,
    verbose=1,
)




## === cell 15
EPOCH = 500
BATCH_SIZE = 1024

try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
    print("Running on TPU.")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print(
        f"Running on default strategy (CPU/GPU). TPU not used: {type(e).__name__}: {e}"
    )

AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.threading.private_threadpool_size = 0
options.threading.max_intra_op_parallelism = 0

n_breaths = train_df.shape[0]
base_idx = np.arange(n_breaths, dtype=np.int32)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(iter(kf.split(base_idx)))


def make_ds_from_indices_slice(x_np, y_np, idx_np, training: bool):
    x_sel = np.ascontiguousarray(x_np[idx_np], dtype=np.float32)
    y_sel = np.ascontiguousarray(y_np[idx_np], dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((x_sel, y_sel)).with_options(options)

    if training:
        ds = ds.shuffle(
            buffer_size=min(int(x_sel.shape[0]), 8192),
            seed=42,
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = (
        ds.cache()
    )  # fixed fold data; speeds up repeated epochs without changing samples
    ds = ds.prefetch(AUTOTUNE)
    return ds


trained_models = []

with strategy.scope():
    fold = 0
    print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

    train_ds = make_ds_from_indices_slice(train_df, Y, train_idx, training=True)
    valid_ds = make_ds_from_indices_slice(train_df, Y, test_idx, training=False)

    model = build_model()

    callback0 = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"AdamPressurePreModel{fold+1}.weights.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    his = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCH,
        callbacks=[callback0, callback1],
        verbose=2,
    )

    if False:
        try:
            stats = pd.DataFrame(his.history)
            ax = stats.plot()
            ax.set_xlabel("epoch")
            plt.show()
        except Exception:
            pass

    model.load_weights(f"AdamPressurePreModel{fold+1}.weights.h5")
    trained_models.append(model)




## === cell 16
models = trained_models
len(models)




## === cell 17
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

u_in_t = test_data["u_in"].to_numpy(dtype=np.float32, copy=False)
n_rows_t = u_in_t.shape[0]
if n_rows_t % 80 != 0:
    raise ValueError(f"Expected test rows multiple of 80, got {n_rows_t}")
u_in_t2d = u_in_t.reshape(-1, 80)

u_in_lag1_t2d = np.empty_like(u_in_t2d)
u_in_lag1_t2d[:, 0] = 0.0
u_in_lag1_t2d[:, 1:] = u_in_t2d[:, :-1]
test_data["u_in_lag1"] = u_in_lag1_t2d.reshape(-1)
test_data["diff_u_in1"] = (u_in_t2d - u_in_lag1_t2d).reshape(-1)
test_data["u_in_cumsum"] = np.cumsum(u_in_t2d, axis=1).reshape(-1)

test_data = dropCols(test_data, cols_2_drop)
test_data = np.ascontiguousarray(rc.transform(test_data), dtype=np.float32)
test_data = np.ascontiguousarray(
    test_data.reshape(-1, 80, test_data.shape[-1]), dtype=np.float32
)




## === cell 18
test_data.shape




## === cell 19
if len(models) == 0:
    raise RuntimeError(
        "No trained models found; training likely failed before producing fold models."
    )

PRED_BATCH_SIZE = 4096

pred_options = tf.data.Options()
pred_options.experimental_deterministic = True
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_data)
    .with_options(pred_options)
    .batch(PRED_BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

mean_pred = None
for model in models:
    pred = model.predict(test_ds, verbose=0)
    if mean_pred is None:
        mean_pred = pred
    else:
        mean_pred += pred

mean_pred = mean_pred / len(models)
mean_pred = mean_pred.reshape(-1, 1)




## === cell 20
mean_pred.shape




## === cell 21
submission_file = pd.read_csv(sample_sub)

if len(submission_file) != mean_pred.shape[0]:
    raise ValueError(
        f"Prediction length mismatch: submission has {len(submission_file)} rows but preds have {mean_pred.shape[0]} rows"
    )

submission_file["pressure"] = mean_pred.reshape(-1)
submission_file.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_file.shape)
print(submission_file.head())
