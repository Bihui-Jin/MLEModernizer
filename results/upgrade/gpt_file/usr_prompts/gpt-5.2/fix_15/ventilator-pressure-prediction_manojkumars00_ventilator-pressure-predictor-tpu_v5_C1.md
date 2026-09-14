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
import sys
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import KFold

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

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

print("TF version:", tf.__version__)



## === cell 1
from sklearn.preprocessing import RobustScaler

rb = RobustScaler()




## === cell 2
def _resolve_base_dir():
    candidates = [
        "/kaggle/input/ventilator-pressure-prediction",
        "/kaggle/data/ventilator-pressure-prediction",
        "/kaggle/data",
        "/kaggle/input",
        "../input/ventilator-pressure-prediction",
        "../input",
        "./",
    ]
    for base in candidates:
        train_p = os.path.join(base, "train.csv")
        test_p = os.path.join(base, "test.csv")
        sub_p = os.path.join(base, "sample_submission.csv")
        if os.path.exists(train_p) and os.path.exists(test_p) and os.path.exists(sub_p):
            return base
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv/sample_submission.csv in expected directories."
    )


BASE_DIR = _resolve_base_dir()
print("Resolved BASE_DIR:", BASE_DIR)

train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")
sample_sub = os.path.join(BASE_DIR, "sample_submission.csv")




## === cell 3
def dropCols(df, cols):
    return df.drop(cols, axis=1)




## === cell 4
TRAIN_DTYPES = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "u_out": "int8",
    "u_in": "float32",
    "time_step": "float32",
    "pressure": "float32",
}
train_data = pd.read_csv(train_path, dtype=TRAIN_DTYPES)

g_breath_uin = train_data["u_in"].groupby(train_data["breath_id"], sort=False)
train_data["diff_u_in1"] = (train_data["u_in"] - g_breath_uin.shift(1)).fillna(0.0)
train_data["u_in_cumsum"] = g_breath_uin.cumsum()



## === cell 5
cols_2_drop = ["id", "breath_id"]



## === cell 6
train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")



## === cell 7
rb.fit(train_df)
train_df = rb.transform(train_df).astype(np.float32, copy=False)



## === cell 8
n_features = train_df.shape[-1]
train_df = np.ascontiguousarray(train_df.reshape(-1, 80, n_features))
Y = np.ascontiguousarray(Y.values.reshape(-1, 80, 1).astype(np.float32, copy=False))



## === cell 9
train_df.shape, Y.shape




## === cell 10
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
        optimizer=opt, loss=tf.keras.losses.MeanAbsoluteError(), metrics=["mae"]
    )
    return model




## === cell 11
import math


def scheduler(epoch, lr):
    if epoch > 200 and epoch % 10 == 0:
        return lr * math.exp(-0.01)
    else:
        return lr


callback1 = tf.keras.callbacks.LearningRateScheduler(scheduler)



## === cell 12
EPOCH = 80
BATCH_SIZE = 512
N_FOLDS = 2

try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
    print("Using TPU strategy.")
except Exception as e:
    print("TPU not available; using default strategy. Details:", repr(e))
    strategy = tf.distribute.get_strategy()

AUTOTUNE = tf.data.AUTOTUNE
tfdata_opts = tf.data.Options()
tfdata_opts.deterministic = True
tfdata_opts.experimental_optimization.apply_default_optimizations = True

x_all_np = train_df  # (n_breaths, 80, n_features), contiguous float32
y_all_np = Y  # (n_breaths, 80, 1), contiguous float32

STEPS_PER_EXECUTION = 32

x_all_tf = tf.convert_to_tensor(x_all_np, dtype=tf.float32)
y_all_tf = tf.convert_to_tensor(y_all_np, dtype=tf.float32)


def make_ds_from_indices(idx_np, training=False):
    idx_np = np.asarray(idx_np, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices(idx_np).with_options(tfdata_opts)
    if training:
        ds = ds.shuffle(
            buffer_size=min(int(idx_np.shape[0]), 8192),
            seed=SEED,
            reshuffle_each_iteration=False,  # keep determinism behavior aligned with original
        )

    def _gather_batch(batch_idx):
        x = tf.gather(x_all_tf, batch_idx)
        y = tf.gather(y_all_tf, batch_idx)
        return x, y

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.map(_gather_batch, num_parallel_calls=AUTOTUNE, deterministic=True)

    if not training:
        ds = ds.cache()

    ds = ds.prefetch(AUTOTUNE)
    return ds


kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

weights_paths = []

with strategy.scope():
    for fold, (train_idx, valid_idx) in enumerate(
        kf.split(np.arange(x_all_np.shape[0]))
    ):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

        train_ds = make_ds_from_indices(train_idx, training=True)
        valid_ds = make_ds_from_indices(valid_idx, training=False)

        model = build_model()
        model.steps_per_execution = STEPS_PER_EXECUTION

        weights_path = f"AdamPressurePreModel{fold+1}.weights.h5"
        weights_paths.append(weights_path)

        callback0 = tf.keras.callbacks.ModelCheckpoint(
            filepath=weights_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            verbose=0,
        )

        _ = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCH,
            callbacks=[callback0, callback1],
            verbose=2,
        )

        print("\n\n")



## === cell 13
models = []
with strategy.scope():
    for i, wp in enumerate(weights_paths, start=1):
        m = build_model()
        m.load_weights(wp)
        models.append(m)
len(models)



## === cell 14
TEST_DTYPES = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "u_out": "int8",
    "u_in": "float32",
    "time_step": "float32",
}
test_data_raw = pd.read_csv(test_path, dtype=TEST_DTYPES)

g_breath_uin_t = test_data_raw["u_in"].groupby(test_data_raw["breath_id"], sort=False)
test_data_raw["diff_u_in1"] = (test_data_raw["u_in"] - g_breath_uin_t.shift(1)).fillna(
    0.0
)
test_data_raw["u_in_cumsum"] = g_breath_uin_t.cumsum()

test_data = dropCols(test_data_raw.copy(), cols_2_drop)
test_data = rb.transform(test_data).astype(np.float32, copy=False)
test_data = np.ascontiguousarray(test_data.reshape(-1, 80, test_data.shape[-1]))

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_data)
    .with_options(tfdata_opts)
    .batch(BATCH_SIZE, drop_remainder=False)
    .cache()
    .prefetch(AUTOTUNE)
)



## === cell 15
p = []
for m in models:
    preds = m.predict(test_ds, verbose=0).reshape(-1, 1)  # (n_rows, 1)
    p.append(preds)

p = np.mean(np.stack(p, axis=0), axis=0)  # (n_rows, 1)
p.shape



## === cell 16
submission_file = pd.read_csv(sample_sub)

test_ids = test_data_raw["id"].to_numpy()
sub_ids = submission_file["id"].to_numpy()

if len(submission_file) != len(test_ids) or not np.array_equal(sub_ids, test_ids):
    submission_file = pd.DataFrame({"id": test_ids})
    print(
        "sample_submission id order/length mismatch; rebuilt submission ids from test.csv"
    )
else:
    print("sample_submission ids match test.csv order")

submission_file["pressure"] = p.reshape(-1).astype(np.float32, copy=False)
submission_file.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_file.shape)
print(submission_file.head())
print(submission_file.tail())
