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
os.environ.setdefault("PYTHONHASHSEED", "42")

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

tf.get_logger().setLevel("ERROR")
tf.keras.utils.set_random_seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() or 1))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

DATA_OPTS = tf.data.Options()
DATA_OPTS.deterministic = True




## === cell 1
rb = RobustScaler()




## === cell 2
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 3
def dropCols(df, cols):
    df = df.copy()
    df.drop(cols, axis=1, inplace=True)
    return df




## === cell 4
def add_breath_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    breath = df["breath_id"].to_numpy(copy=False)
    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)

    is_new = np.empty(len(df), dtype=bool)
    is_new[0] = True
    is_new[1:] = breath[1:] != breath[:-1]

    prev_u = np.empty_like(u_in)
    prev_u[0] = 0.0
    prev_u[1:] = u_in[:-1]
    prev_u[is_new] = 0.0
    df["diff_u_in1"] = u_in - prev_u

    cs = np.cumsum(u_in, dtype=np.float64)
    start_idx = np.flatnonzero(is_new)

    prev_cs_at_start = np.empty_like(start_idx, dtype=np.float64)
    prev_cs_at_start[0] = 0.0
    prev_cs_at_start[1:] = cs[start_idx[1:] - 1]

    breath_offsets = prev_cs_at_start - u_in[start_idx].astype(np.float64, copy=False)
    counts = np.diff(np.append(start_idx, len(df)))
    offset = np.repeat(breath_offsets, counts)

    df["u_in_cumsum"] = (cs - offset).astype(np.float32)
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
train_data = add_breath_features_fast(train_data)




## === cell 6
cols_2_drop = ["id", "breath_id"]




## === cell 7
train_df = train_data.drop(cols_2_drop, axis=1)
Y = train_df.pop("pressure")




## === cell 8
rb.fit(train_df)
train_arr = rb.transform(train_df)




## === cell 9
train_arr = np.asarray(train_arr, dtype=np.float32, order="C")
Y = np.asarray(Y.to_numpy(copy=False), dtype=np.float32, order="C")

train_arr = train_arr.reshape(-1, 80, train_arr.shape[-1])
Y = Y.reshape(-1, 80, 1)




## === cell 10
train_arr.shape, Y.shape




## === cell 11
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

    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1))

    opt = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=["mae"],
        steps_per_execution=128,
        jit_compile=True,
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
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
except Exception:
    tpu_strategy = tf.distribute.get_strategy()

DO_PLOT = False


def make_ds(X, y, batch_size, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(buffer_size=len(X), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.with_options(DATA_OPTS)
    ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


with tpu_strategy.scope():
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(train_arr, Y)):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

        fold_path = f"AdamPressurePreModel{fold+1}.h5"
        if os.path.exists(fold_path):
            model = tf.keras.models.load_model(fold_path, compile=False)
            print(f"Loaded existing model: {fold_path} (skipping training)")
            continue

        pretrained_path = "./AdamPressurePreModel3.h5"
        if os.path.exists(pretrained_path):
            model = tf.keras.models.load_model(pretrained_path, compile=False)
            opt = tf.keras.optimizers.Adam()
            model.compile(
                optimizer=opt,
                loss=tf.keras.losses.MeanAbsoluteError(),
                metrics=["mae"],
                steps_per_execution=128,
                jit_compile=True,
                run_eagerly=False,
            )
        else:
            model = build_model()

        weight_ckpt_path = fold_path + ".weights.h5"
        callback0 = tf.keras.callbacks.ModelCheckpoint(
            weight_ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )

        callback_bnr = tf.keras.callbacks.BackupAndRestore(
            backup_dir=f"./backup_fold_{fold+1}"
        )

        X_tr = train_arr[train_idx]
        y_tr = Y[train_idx]
        X_va = train_arr[test_idx]
        y_va = Y[test_idx]

        ds_tr = make_ds(X_tr, y_tr, BATCH_SIZE, training=True)
        ds_va = make_ds(X_va, y_va, BATCH_SIZE, training=False)

        his = model.fit(
            ds_tr,
            validation_data=ds_va,
            epochs=EPOCH,
            shuffle=False,  # shuffle handled by tf.data for identical semantics
            callbacks=[callback0, callback1, callback_bnr],
            verbose=1,
        )

        model.load_weights(weight_ckpt_path)
        model.save(fold_path, include_optimizer=False)

        if DO_PLOT:
            import matplotlib.pyplot as plt

            stats = pd.DataFrame(his.history)
            stats.plot()
            plt.show()
        print("\n\n")




## === cell 14
models_paths = [
    "AdamPressurePreModel1.h5",
    "AdamPressurePreModel2.h5",
    "AdamPressurePreModel3.h5",
    "AdamPressurePreModel4.h5",
    "AdamPressurePreModel5.h5",
]

models = [
    tf.keras.models.load_model(model_path, compile=False) for model_path in models_paths
]




## === cell 15
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
test_data = add_breath_features_fast(test_data)

test_df = test_data.drop(cols_2_drop, axis=1)
test_arr = rb.transform(test_df)
test_arr = np.asarray(test_arr, dtype=np.float32, order="C")
test_arr = test_arr.reshape(-1, 80, test_arr.shape[-1])




## === cell 16
PRED_BATCH_SIZE = 1024

ds_test = (
    tf.data.Dataset.from_tensor_slices(test_arr)
    .batch(PRED_BATCH_SIZE)
    .with_options(DATA_OPTS)
    .prefetch(tf.data.AUTOTUNE)
)

p = None
for model in models:
    pred = model.predict(ds_test, verbose=0).reshape(-1, 1)
    if p is None:
        p = pred
    else:
        p += pred
p = p / len(models)




## === cell 17
p.shape




## === cell 18
p




## === cell 19
submission_file = pd.read_csv(sample_sub)
submission_file["pressure"] = p.reshape(-1)
submission_file.to_csv("submission.csv", index=False)
