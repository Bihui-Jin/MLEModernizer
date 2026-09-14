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
from sklearn.preprocessing import RobustScaler

print("TF version:", tf.__version__)

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    import multiprocessing

    _cpu = max(1, multiprocessing.cpu_count())
    tf.config.threading.set_intra_op_parallelism_threads(min(8, _cpu))
    tf.config.threading.set_inter_op_parallelism_threads(min(2, _cpu))
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

rb = RobustScaler()

train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"

AUTOTUNE = tf.data.AUTOTUNE
DATASET_OPTS = tf.data.Options()
DATASET_OPTS.deterministic = True
try:
    DATASET_OPTS.experimental_optimization.map_and_batch_fusion = True
    DATASET_OPTS.experimental_optimization.parallel_batch = True
    DATASET_OPTS.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass




## === cell 1
def dropCols_inplace_view(df, cols):
    return df.drop(columns=cols)




## === cell 2
def add_engineered_features_sorted(df: pd.DataFrame) -> pd.DataFrame:
    breath = df["breath_id"].to_numpy()
    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)

    start = np.empty(len(df), dtype=bool)
    start[0] = True
    start[1:] = breath[1:] != breath[:-1]

    prev = np.empty_like(u_in)
    prev[0] = 0.0
    prev[1:] = u_in[:-1]
    prev[start] = 0.0
    df["diff_u_in1"] = u_in - prev

    cs = np.cumsum(u_in, dtype=np.float32)
    base = np.zeros_like(cs)
    idx_start = np.flatnonzero(start)
    if len(idx_start) > 1:
        base[idx_start[1:]] = cs[idx_start[1:] - 1]
    base_ff = np.maximum.accumulate(base)
    df["u_in_cumsum"] = cs - base_ff

    return df




## === cell 3
usecols = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
train_data = pd.read_csv(
    train_path,
    usecols=usecols,
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
        "pressure": "float32",
    },
)

train_data.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")
train_data = add_engineered_features_sorted(train_data)

cols_2_drop = ["id", "breath_id"]
train_df = dropCols_inplace_view(train_data, cols_2_drop)
Y = train_df.pop("pressure")




## === cell 4
rb.fit(train_df)
train_arr = rb.transform(train_df).astype(np.float32, copy=False)
train_arr = np.ascontiguousarray(train_arr.reshape(-1, 80, train_arr.shape[-1]))

Y_arr = Y.to_numpy(dtype=np.float32, copy=False)
Y_arr = np.ascontiguousarray(Y_arr.reshape(-1, 80, 1))

print("Train shapes:", train_arr.shape, Y_arr.shape)




## === cell 5
def build_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Bidirectional(
            layers.LSTM(300, return_sequences=True),
            input_shape=[80, train_arr.shape[-1]],
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                150,
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

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1))

    opt = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=["mae"],
        run_eagerly=False,
    )
    return model


def scheduler(epoch, lr):
    if epoch > 200 and epoch % 10 == 0:
        return lr * tf.math.exp(-0.01)
    else:
        return lr


callback1 = tf.keras.callbacks.LearningRateScheduler(scheduler)




## === cell 6
EPOCH = 400
BATCH_SIZE = 1024

try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # raises if none
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
    print("Using TPU strategy")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print("TPU not available; using default strategy. Reason:", repr(e))


train_x_tf = tf.convert_to_tensor(train_arr)
train_y_tf = tf.convert_to_tensor(Y_arr)


def make_fold_ds_from_indices(indices_np: np.ndarray, training: bool):
    idx = tf.convert_to_tensor(indices_np, dtype=tf.int32)
    x = tf.gather(train_x_tf, idx, axis=0)
    y = tf.gather(train_y_tf, idx, axis=0)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).with_options(DATASET_OPTS)
    if training:
        buf = min(int(indices_np.shape[0]), 8192)
        ds = ds.shuffle(buffer_size=buf, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


saved_weight_paths = []

with strategy.scope():
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(kf.split(np.empty((train_arr.shape[0], 1), dtype=np.uint8)))

    for fold, (train_idx, test_idx) in enumerate(folds):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

        ckpt_path = f"AdamPressurePreModel{fold+1}.weights.h5"

        if os.path.exists(ckpt_path):
            print(f"Found existing weights, skipping training: {ckpt_path}")
            saved_weight_paths.append(ckpt_path)
            print("\n\n")
            continue

        train_ds = make_fold_ds_from_indices(train_idx, training=True)
        valid_ds = make_fold_ds_from_indices(test_idx, training=False)

        model = build_model()
        model.build((None, 80, train_arr.shape[-1]))

        callback0 = tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )

        try:
            model.compile(
                optimizer=model.optimizer,
                loss=model.loss,
                metrics=["mae"],
                run_eagerly=False,
                jit_compile=True,
            )
        except Exception:
            pass

        his = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCH,
            callbacks=[callback0, callback1],
            verbose=2,
        )

        if os.path.exists(ckpt_path):
            saved_weight_paths.append(ckpt_path)

        print("\n\n")

print("Saved weight files:", saved_weight_paths)




## === cell 7
models_paths = [
    "AdamPressurePreModel1.weights.h5",
    "AdamPressurePreModel2.weights.h5",
    "AdamPressurePreModel3.weights.h5",
    "AdamPressurePreModel4.weights.h5",
    "AdamPressurePreModel5.weights.h5",
]

models = []
for mp in models_paths:
    if not os.path.exists(mp):
        print(f"WARNING: missing weights file, skipping: {mp}")
        continue
    m = build_model()
    m.build((None, 80, train_arr.shape[-1]))
    m.load_weights(mp)
    try:
        m.compile(
            optimizer=m.optimizer,
            loss=m.loss,
            metrics=["mae"],
            run_eagerly=False,
            jit_compile=True,
        )
    except Exception:
        pass
    models.append(m)

if len(models) == 0:
    raise RuntimeError(
        "No model weight files were found. Training likely failed before saving checkpoints."
    )

print("Loaded models:", len(models))




## === cell 8
test_usecols = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
test_data = pd.read_csv(
    test_path,
    usecols=test_usecols,
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
    },
)
test_data.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")
test_data = add_engineered_features_sorted(test_data)
test_data = dropCols_inplace_view(test_data, cols_2_drop)

test_arr = rb.transform(test_data).astype(np.float32, copy=False)
test_arr = np.ascontiguousarray(test_arr.reshape(-1, 80, test_arr.shape[-1]))
print("Test shape:", test_arr.shape)

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_arr)
    .with_options(DATASET_OPTS)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

p = None
for i, model in enumerate(models, start=1):
    pred = model.predict(test_ds, verbose=0).reshape(-1).astype(np.float64, copy=False)
    if p is None:
        p = pred
    else:
        p += pred

p /= len(models)
p = p.astype(np.float32, copy=False)
print("Prediction shape:", p.shape)

submission_file = pd.read_csv(sample_sub)
if len(submission_file) != len(p):
    raise ValueError(
        f"Submission rows ({len(submission_file)}) != predictions ({len(p)})"
    )

submission_file["pressure"] = p
submission_file.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_file.shape)
print(submission_file.head())
