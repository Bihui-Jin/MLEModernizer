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

# 5. Target score

0.1591

# 6. Current score

8.61059

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 8.61059) has done: 'The timeout is dominated by the 5-fold training loop (5 × 500 epochs on a large BiLSTM), plus some avoidable input-pipeline overhead per fold from rebuilding and caching datasets after NumPy slicing. To keep identical core logic while reducing wall time, I (1) eliminate the expensive per-fold `tf.data` shuffling/caching steps by using Keras’s built-in `validation_split` within each fold (still same fold semantics) and (2) add a time-budget callback that stops training exactly at the 600s limit (this does not change the algorithm, it only prevents overruns). I also reduce avoidable overhead by precomputing fold indices once, ensuring arrays are contiguous, and using `tf.data` only for test-time (where caching is beneficial across 5 predicts). These changes preserve model architecture, loss, optimizer, epochs target (until time limit), and evaluation semantics, while preventing a hard timeout.'
- What this solution (achieved 8.61059) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by forcing the pure-Python protobuf implementation before TensorFlow loads. Then I correct the feature-engineering bug in `u_in_cumsum` that currently uses a wrong repeat-length calculation, which severely harms the model’s signal and is the most likely reason for the very high MAE (8.61 vs target 0.1591). These changes keep the same model architecture, loss, optimizer, training loop, and submission format, but should materially improve the score while ensuring the notebook runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 8.66824) has done: 'We fix the immediate runtime crash before TensorFlow imports by removing the protobuf-implementation override that is incompatible with the Kaggle environment’s protobuf 6.x, which is what triggers the `MessageFactory.GetPrototype` error. Then we keep the model/training logic intact but correct the data paths to match your provided filesystem (your current `../input/...` paths won’t resolve here), ensuring the pipeline can actually load train/test and write `submission.csv`. These changes are execution-critical and score-positive (they allow the existing improved feature engineering and training to run). No architecture, loss, optimizer, or fold logic is changed.'
- What this solution (achieved 8.61059) has done: 'We fix the TensorFlow/protobuf crash by pinning the pure-Python protobuf implementation **before** importing TensorFlow; this resolves the `MessageFactory.GetPrototype` AttributeError in this environment. We also make the input-path resolution robust to your provided filesystem (`/kaggle/data` and `/kaggle/input/ventilator-pressure-prediction/...`) without changing the modeling/training logic. Finally, we keep the existing feature engineering and 5-fold training/inference unchanged, ensuring the script runs end-to-end and writes a valid `submission.csv` with the required `id,pressure` columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import KFold

print("TF version:", tf.__version__)

tf.keras.utils.set_random_seed(2000)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

tf.config.run_functions_eagerly(False)

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.preprocessing import RobustScaler

rb = RobustScaler()

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()




## === cell 2
def _resolve_path(*candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


train_path = _resolve_path(
    "/kaggle/data/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/input/ventilator-pressure-prediction/train.csv",
    "/kaggle/data/ventilator-pressure-prediction/train.csv",
)
test_path = _resolve_path(
    "/kaggle/data/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/input/ventilator-pressure-prediction/test.csv",
    "/kaggle/data/ventilator-pressure-prediction/test.csv",
)
sample_sub = _resolve_path(
    "/kaggle/data/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv",
    "/kaggle/data/ventilator-pressure-prediction/sample_submission.csv",
)

assert train_path is not None, "Missing train.csv in expected locations"
assert test_path is not None, "Missing test.csv in expected locations"
assert sample_sub is not None, "Missing sample_submission.csv in expected locations"

print("Resolved paths:")
print(" train:", train_path)
print(" test :", test_path)
print(" sub  :", sample_sub)




## === cell 3
def dropCols(df, cols):
    return df.drop(cols, axis=1)




## === cell 4
def add_breath_features_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX: correct per-breath cumulative sum for u_in.
    The previous implementation computed repeat lengths incorrectly (np.diff on flatnonzero(start)
    without including the last boundary), producing wrong offsets and thus wrong u_in_cumsum.
    This is score-critical but keeps core feature intent identical: cumulative u_in within each breath.
    """
    breath = df["breath_id"].to_numpy()
    u = df["u_in"].to_numpy(np.float32, copy=False)

    start = np.empty(len(df), dtype=bool)
    start[0] = True
    start[1:] = breath[1:] != breath[:-1]

    du = np.empty_like(u, dtype=np.float32)
    du[0] = 0.0
    np.subtract(u[1:], u[:-1], out=du[1:])
    du[start] = 0.0

    cs = np.cumsum(u, dtype=np.float32)
    start_idx = np.flatnonzero(start)
    ends = np.append(start_idx[1:], len(df))
    lengths = ends - start_idx  # per-breath lengths, sum == len(df)

    offsets = np.repeat(cs[start_idx] - u[start_idx], lengths).astype(
        np.float32, copy=False
    )
    u_cum = cs - offsets

    df["diff_u_in1"] = du
    df["u_in_cumsum"] = u_cum.astype(np.float32, copy=False)
    return df




## === cell 5
TRAIN_DTYPES = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int8",
    "C": "int8",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
train_data = pd.read_csv(train_path, dtype=TRAIN_DTYPES)

train_data.sort_values(
    ["breath_id", "time_step"], inplace=True, kind="mergesort", ignore_index=True
)
train_data = add_breath_features_sorted(train_data)



## === cell 6
cols_2_drop = ["id", "breath_id"]



## === cell 7
train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")



## === cell 8
train_df.isna().sum()



## === cell 9
sc.fit(train_df)
train_df = sc.transform(train_df).astype(np.float32, copy=False)



## === cell 10
train_df = np.ascontiguousarray(train_df.reshape(-1, 80, train_df.shape[-1]))
Y = np.ascontiguousarray(Y.values.reshape(-1, 80, 1).astype(np.float32, copy=False))



## === cell 11
train_df.shape, Y.shape




## === cell 12
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
        optimizer=opt, loss=tf.keras.losses.MeanAbsoluteError(), metrics=["mae"]
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
BATCH_SIZE = 512
AUTOTUNE = tf.data.AUTOTUNE

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Using TPU strategy")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print("TPU not available, using default strategy:", type(strategy).__name__)
    print("TPU connect error:", repr(e))


class BestWeights(tf.keras.callbacks.Callback):
    def __init__(self, monitor="val_loss", mode="min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best = np.inf if mode == "min" else -np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        v = logs.get(self.monitor, None)
        if v is None:
            return
        if (self.mode == "min" and v < self.best) or (
            self.mode == "max" and v > self.best
        ):
            self.best = float(v)
            self.best_weights = self.model.get_weights()


class TimeLimit(tf.keras.callbacks.Callback):
    def __init__(self, seconds: float):
        super().__init__()
        self.seconds = float(seconds)
        self.t0 = None

    def on_train_begin(self, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if (time.time() - self.t0) >= self.seconds:
            self.model.stop_training = True


with strategy.scope():
    kf = KFold(n_splits=5, shuffle=True, random_state=2000)
    folds = list(kf.split(train_df, Y))

    model = build_model()
    initial_weights = model.get_weights()

    best_weights_per_fold = []

    total_budget_sec = 600.0
    safety_overhead_sec = 60.0  # test predict + I/O buffer
    train_budget_sec = max(1.0, total_budget_sec - safety_overhead_sec)
    per_fold_budget_sec = train_budget_sec / len(folds)

    for fold, (train_idx, test_idx) in enumerate(folds):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

        model.set_weights(initial_weights)

        X_train, X_valid = train_df[train_idx], train_df[test_idx]
        y_train, y_valid = Y[train_idx], Y[test_idx]

        best_cb = BestWeights(monitor="val_loss", mode="min")
        time_cb = TimeLimit(seconds=per_fold_budget_sec)

        his = model.fit(
            X_train,
            y_train,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            shuffle=True,
            validation_data=(X_valid, y_valid),
            callbacks=[best_cb, callback1, time_cb],
            verbose=2,
        )

        if best_cb.best_weights is None:
            best_weights_per_fold.append(model.get_weights())
        else:
            best_weights_per_fold.append(best_cb.best_weights)

        print("\n\n")



## === cell 16
with strategy.scope():
    infer_model = build_model()

print(
    f"Prepared inference model; will run {len(best_weights_per_fold)} fold-weight sets."
)



## === cell 17
TEST_DTYPES = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int8",
    "C": "int8",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}
test_data = pd.read_csv(test_path, dtype=TEST_DTYPES)

test_data.sort_values(
    ["breath_id", "time_step"], inplace=True, kind="mergesort", ignore_index=True
)
test_data = add_breath_features_sorted(test_data)



## === cell 18
test_data = dropCols(test_data, cols_2_drop)
test_data = sc.transform(test_data).astype(np.float32, copy=False)
test_data = np.ascontiguousarray(test_data.reshape(-1, 80, test_data.shape[-1]))



## === cell 19
test_data.shape



## === cell 20
options = tf.data.Options()
options.deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
try:
    options.experimental_slack = True
except Exception:
    pass

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_data)
    .with_options(options)
    .batch(BATCH_SIZE, drop_remainder=False)
    .cache()  # reuse cached batches across 5 predict passes (one per fold)
    .prefetch(tf.data.AUTOTUNE)
)

preds = []
for w in best_weights_per_fold:
    infer_model.set_weights(w)
    preds.append(infer_model.predict(test_ds, verbose=0).reshape(-1, 1))

p = np.mean(np.stack(preds, axis=0), axis=0)  # (n, 1)
print("Pred shape:", p.shape)



## === cell 21
submission_file = pd.read_csv(sample_sub)
if len(submission_file) != p.shape[0]:
    raise ValueError(
        f"Prediction length {p.shape[0]} does not match submission length {len(submission_file)}"
    )

submission_file["pressure"] = p.reshape(-1)
submission_file.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_file.shape)
print(submission_file.head())
