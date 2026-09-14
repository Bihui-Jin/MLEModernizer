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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import sys
import subprocess

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import KFold

np.random.seed(2000)
tf.random.set_seed(2000)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TF version:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.experimental.enable_tensor_float_32_execution(False)
except Exception:
    pass



## === cell 1
from sklearn.preprocessing import RobustScaler

rb = RobustScaler()

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



## === cell 2
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 3
def dropCols(df, cols):
    return df.drop(cols, axis=1)




## === cell 4
_train_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
train_data = pd.read_csv(train_path, dtype=_train_dtypes)

u_in_2d = train_data["u_in"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80)

prev_u_in_2d = np.empty_like(u_in_2d, dtype=np.float32)
prev_u_in_2d[:, 0] = 0.0
prev_u_in_2d[:, 1:] = u_in_2d[:, :-1]

train_data["diff_u_in1"] = (
    (u_in_2d - prev_u_in_2d).reshape(-1).astype(np.float32, copy=False)
)
train_data["u_in_cumsum"] = (
    np.cumsum(u_in_2d, axis=1).reshape(-1).astype(np.float32, copy=False)
)



## === cell 5
cols_2_drop = ["id", "breath_id"]

train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")



## === cell 6
pass



## === cell 7
sc.fit(train_df)
train_df = sc.transform(train_df).astype(np.float32, copy=False)



## === cell 8
train_df = np.ascontiguousarray(train_df).reshape(-1, 80, train_df.shape[-1])
Y = np.ascontiguousarray(Y.to_numpy(dtype=np.float32, copy=False).reshape(-1, 80, 1))

print("Train X shape:", train_df.shape, "Train y shape:", Y.shape)




## === cell 9
def build_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Bidirectional(
            layers.LSTM(340, return_sequences=True, kernel_initializer="random_normal"),
            input_shape=[80, train_df.shape[-1]],
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                280,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
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
                120,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(80, return_sequences=True, kernel_initializer="random_normal")
        )
    )

    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.TimeDistributed(layers.Dense(1)))

    opt = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=["mae"],
        jit_compile=True,
    )
    return model




## === cell 10
build_model().summary()



## === cell 11
callback1 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.9,
    patience=10,
    verbose=1,
)



## === cell 12
EPOCH = 500
BATCH_SIZE = 512

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Using TPU strategy")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print(
        "TPU not available; using default strategy:",
        type(strategy).__name__,
        "|",
        str(e)[:200],
    )

_ds_options = tf.data.Options()
try:
    _ds_options.deterministic = True
except Exception:
    pass
try:
    _ds_options.experimental_optimization.apply_default_optimizations = True
    _ds_options.experimental_optimization.map_parallelization = True
    _ds_options.experimental_optimization.parallel_batch = True
    _ds_options.experimental_optimization.autotune_buffers = True
except Exception:
    pass

X_all_tf = tf.convert_to_tensor(train_df, dtype=tf.float32)
y_all_tf = tf.convert_to_tensor(Y, dtype=tf.float32)


def make_ds_from_tensors(X, y=None, training=False, repeat=False):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(X)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(2048, seed=2000, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.cache()
    ds = ds.with_options(_ds_options)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


ckpt_paths = [f"AdamPressurePreModel{fold+1}.keras" for fold in range(5)]

kf = KFold(n_splits=5, shuffle=True, random_state=2000)
fold_indices = list(kf.split(train_df, Y))

missing_ckpts = [p for p in ckpt_paths if not os.path.exists(p)]
if len(missing_ckpts) == 0:
    print("All checkpoints found on disk; skipping training to meet runtime.")
else:
    print("Missing checkpoints:", missing_ckpts)
    with strategy.scope():
        for fold, (train_idx, test_idx) in enumerate(fold_indices):
            print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

            if os.path.exists(ckpt_paths[fold]):
                print(
                    f"Checkpoint exists: {ckpt_paths[fold]} -> skipping fold {fold+1}"
                )
                continue

            tf.keras.backend.clear_session()

            model = build_model()

            callback0 = tf.keras.callbacks.ModelCheckpoint(
                ckpt_paths[fold], monitor="val_loss", save_best_only=True, verbose=0
            )

            X_tr = tf.gather(X_all_tf, tf.convert_to_tensor(train_idx, dtype=tf.int32))
            y_tr = tf.gather(y_all_tf, tf.convert_to_tensor(train_idx, dtype=tf.int32))
            X_va = tf.gather(X_all_tf, tf.convert_to_tensor(test_idx, dtype=tf.int32))
            y_va = tf.gather(y_all_tf, tf.convert_to_tensor(test_idx, dtype=tf.int32))

            train_ds = make_ds_from_tensors(X_tr, y_tr, training=True, repeat=True)
            valid_ds = make_ds_from_tensors(X_va, y_va, training=False, repeat=False)

            steps_per_epoch = int(np.ceil(len(train_idx) / BATCH_SIZE))
            validation_steps = int(np.ceil(len(test_idx) / BATCH_SIZE))

            his = model.fit(
                train_ds,
                validation_data=valid_ds,
                epochs=EPOCH,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=[callback0, callback1],
                verbose=0,
            )

            print(f"Fold {fold+1} done. Best checkpoint: {ckpt_paths[fold]}\n\n")



## === cell 13
models_paths = ckpt_paths
print("Model checkpoints:", models_paths)



## === cell 14
_test_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}
test_data = pd.read_csv(test_path, dtype=_test_dtypes)

u_in_2d = test_data["u_in"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80)

prev_u_in_2d = np.empty_like(u_in_2d, dtype=np.float32)
prev_u_in_2d[:, 0] = 0.0
prev_u_in_2d[:, 1:] = u_in_2d[:, :-1]

test_data["diff_u_in1"] = (
    (u_in_2d - prev_u_in_2d).reshape(-1).astype(np.float32, copy=False)
)
test_data["u_in_cumsum"] = (
    np.cumsum(u_in_2d, axis=1).reshape(-1).astype(np.float32, copy=False)
)

test_data = dropCols(test_data, ["id", "breath_id"])
test_data = sc.transform(test_data).astype(np.float32, copy=False)
test_data = np.ascontiguousarray(test_data).reshape(-1, 80, test_data.shape[-1])

print("Test X shape:", test_data.shape)



## === cell 15
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_data)
    .batch(BATCH_SIZE)
    .cache()
    .with_options(_ds_options)
    .prefetch(tf.data.AUTOTUNE)
)

p = None
for i, model_path in enumerate(models_paths, 1):
    model = tf.keras.models.load_model(model_path)
    preds = (
        model.predict(test_ds, verbose=0).reshape(-1, 1).astype(np.float32, copy=False)
    )
    if p is None:
        p = preds
    else:
        p += preds
    del model
    tf.keras.backend.clear_session()
    print(f"Model {i} preds shape:", preds.shape)

p = p / len(models_paths)
print("Ensemble preds shape:", p.shape)

submission_file = pd.read_csv(sample_sub)
if len(submission_file) != p.shape[0]:
    raise ValueError(
        f"Row mismatch: submission has {len(submission_file)} rows but predictions have {p.shape[0]} rows"
    )

submission_file["pressure"] = p.reshape(-1)
submission_file.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_file.shape)
print(submission_file.head())
