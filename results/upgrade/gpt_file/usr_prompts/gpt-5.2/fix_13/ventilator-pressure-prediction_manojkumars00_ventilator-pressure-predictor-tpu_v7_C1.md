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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

print("TensorFlow:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))

np.random.seed(42)
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
    cpu_count = os.cpu_count() or 4
    if len(tf.config.list_physical_devices("GPU")) == 0:
        tf.config.threading.set_intra_op_parallelism_threads(max(1, cpu_count // 2))
        tf.config.threading.set_inter_op_parallelism_threads(2)
    else:
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## === cell 1
rb = RobustScaler()




## === cell 2
def _resolve_path(p_candidates):
    for p in p_candidates:
        if os.path.exists(p):
            return p
    return p_candidates[0]


train_path = _resolve_path(
    [
        "/kaggle/input/ventilator-pressure-prediction/train.csv",
        "../input/ventilator-pressure-prediction/train.csv",
    ]
)
test_path = _resolve_path(
    [
        "/kaggle/input/ventilator-pressure-prediction/test.csv",
        "../input/ventilator-pressure-prediction/test.csv",
    ]
)
sample_sub = _resolve_path(
    [
        "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv",
        "../input/ventilator-pressure-prediction/sample_submission.csv",
    ]
)

print("Resolved train_path:", train_path)
print("Resolved test_path :", test_path)
print("Resolved sample_sub:", sample_sub)




## === cell 3
def dropCols(df, cols):
    return df.drop(columns=cols)




## === cell 4
def add_breath_features_inplace(df: pd.DataFrame) -> pd.DataFrame:
    breath = df["breath_id"].to_numpy()
    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)

    new_breath = np.empty_like(breath, dtype=bool)
    new_breath[0] = True
    new_breath[1:] = breath[1:] != breath[:-1]

    u_prev = np.empty_like(u_in, dtype=np.float32)
    u_prev[0] = 0.0
    u_prev[1:] = u_in[:-1]
    u_prev[new_breath] = 0.0
    df["diff_u_in1"] = u_in - u_prev

    cs = np.cumsum(u_in, dtype=np.float32)
    start_idx = np.flatnonzero(new_breath)
    offsets = np.empty(start_idx.shape[0], dtype=np.float32)
    offsets[0] = 0.0
    prev_start = start_idx[1:] - 1
    offsets[1:] = cs[prev_start]
    end_idx = np.r_[start_idx[1:], len(u_in)]
    counts = end_idx - start_idx
    offset_per_row = np.repeat(offsets, counts)
    df["u_in_cumsum"] = cs - offset_per_row

    return df




## === cell 5
train_dtypes = {
    "id": np.int32,
    "breath_id": np.int32,
    "R": np.int16,
    "C": np.int16,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
    "pressure": np.float32,
}
train_data = pd.read_csv(train_path, dtype=train_dtypes)
train_data = add_breath_features_inplace(train_data)



## === cell 6
cols_2_drop = ["id", "breath_id"]



## === cell 7
train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")



## === cell 8
rb.fit(train_df)
train_df = rb.transform(train_df).astype(np.float32, copy=False)



## === cell 9
train_df = train_df.reshape(-1, 80, train_df.shape[-1])
Y = Y.values.astype(np.float32).reshape(-1, 80, 1)



## === cell 10
train_df.shape, Y.shape




## === cell 11
def build_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Bidirectional(
            layers.LSTM(300, return_sequences=True),
            input_shape=[80, train_df.shape[-1]],
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

    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1))

    opt = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=["mae"],
        run_eagerly=False,
    )
    return model




## === cell 12
def scheduler(epoch, lr):
    if epoch > 200 and epoch % 10 == 0:
        return lr * tf.math.exp(-0.01)
    else:
        return lr


callback1 = tf.keras.callbacks.LearningRateScheduler(scheduler)



## === cell 13
EPOCH = 100
BATCH_SIZE = 512

try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
    print("Using TPU strategy")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print(
        "Using default strategy (CPU/GPU). TPU not available:",
        type(e).__name__,
        str(e)[:120],
    )

kf = KFold(n_splits=2, shuffle=True, random_state=42)

fold_model_paths = []
last_X_valid, last_y_valid = None, None

options = tf.data.Options()
options.deterministic = True


def make_indexed_ds(
    X_np: np.ndarray,
    y_np: np.ndarray,
    idx_np: np.ndarray,
    cache: bool,
    shuffle: bool,
    seed: int,
    shuffle_buffer: int = 8192,
):
    X_t = tf.convert_to_tensor(X_np)
    y_t = tf.convert_to_tensor(y_np)
    idx_ds = tf.data.Dataset.from_tensor_slices(idx_np.astype(np.int32, copy=False))

    if shuffle:
        buf = int(min(shuffle_buffer, idx_np.shape[0]))
        idx_ds = idx_ds.shuffle(
            buffer_size=buf, seed=seed, reshuffle_each_iteration=True
        )

    ds = idx_ds.map(
        lambda i: (tf.gather(X_t, i), tf.gather(y_t, i)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    if cache:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(options)
    return ds


with strategy.scope():
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df, Y)):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

        last_X_valid, last_y_valid = train_df[valid_idx], Y[valid_idx]

        model = build_model()

        model_path = f"AdamPressurePreModel{fold+1}.weights.h5"
        fold_model_paths.append(model_path)

        callback0 = tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", save_best_only=True, save_weights_only=True
        )

        train_ds = make_indexed_ds(
            train_df, Y, train_idx, cache=False, shuffle=True, seed=42 + fold
        )
        valid_ds = make_indexed_ds(
            train_df, Y, valid_idx, cache=True, shuffle=False, seed=42
        )

        his = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCH,
            callbacks=[callback0, callback1],
            verbose=2,
        )

        print("\n\n")



## === cell 14
models = []
for pth in fold_model_paths:
    m = build_model()
    m.load_weights(pth)
    models.append(m)
print("Loaded models (weights):", fold_model_paths)



## === cell 15
test_dtypes = {
    "id": np.int32,
    "breath_id": np.int32,
    "R": np.int16,
    "C": np.int16,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
}
test_raw = pd.read_csv(test_path, dtype=test_dtypes)

test_data = add_breath_features_inplace(test_raw)

test_data = dropCols(test_data, cols_2_drop)
test_data = rb.transform(test_data).astype(np.float32, copy=False)
test_data = test_data.reshape(-1, 80, test_data.shape[-1])



## === cell 16
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_data)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(options)
)

p_list = []
for m in models:
    p_list.append(m.predict(test_ds, verbose=0).reshape(-1, 1))

p = np.mean(np.stack(p_list, axis=0), axis=0)  # (n_rows, 1)
p.shape



## === cell 17
submission_file = pd.DataFrame({"id": test_raw["id"].values, "pressure": p.reshape(-1)})
submission_file.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_file.shape)
print(submission_file.head())
print(submission_file.tail())
assert submission_file.shape[0] == test_raw.shape[0]
assert list(submission_file.columns) == ["id", "pressure"]
assert submission_file[
    "id"
].is_monotonic_increasing, "id should be in the same order as test.csv"
