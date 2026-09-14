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
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

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
except Exception as _:
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
g = train_data.groupby("breath_id", sort=False)
train_data["diff_u_in1"] = g["u_in"].diff().fillna(0.0).astype("float32")
train_data["u_in_cumsum"] = g["u_in"].cumsum().astype("float32")




## === cell 5
cols_2_drop = ["id", "breath_id"]




## === cell 6
train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")




## === cell 7
train_df.isna().sum()




## === cell 8
sc.fit(train_df)
train_df = sc.transform(train_df).astype(np.float32, copy=False)




## === cell 9
train_df = np.ascontiguousarray(train_df.reshape(-1, 80, train_df.shape[-1]))
Y = np.ascontiguousarray(Y.values.reshape(-1, 80, 1).astype(np.float32, copy=False))




## === cell 10
train_df.shape, Y.shape




## === cell 11
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




## === cell 12
build_model().summary()




## === cell 13
callback1 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.9,
    patience=10,
    verbose=1,
)




## === cell 14
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


class PeriodicValidationMAE(tf.keras.callbacks.Callback):
    def __init__(self, valid_ds, every_n_epochs=5):
        super().__init__()
        self.valid_ds = valid_ds
        self.every_n_epochs = int(every_n_epochs)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        res = self.model.evaluate(self.valid_ds, verbose=0, return_dict=True)
        if logs is not None:
            logs["val_loss"] = float(res["loss"])
            logs["val_mae"] = float(res.get("mae", res["loss"]))


def make_ds(X, y=None, training=False):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(X)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y))

    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    try:
        options.experimental_slack = True
    except Exception:
        pass
    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(min(len(X), 8192), seed=2000, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


with strategy.scope():
    kf = KFold(n_splits=5, shuffle=True, random_state=2000)

    folds = list(kf.split(train_df, Y))

    model = build_model()
    initial_weights = model.get_weights()

    best_weights_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(folds):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

        model.set_weights(initial_weights)

        X_train, X_valid = train_df[train_idx], train_df[test_idx]
        y_train, y_valid = Y[train_idx], Y[test_idx]

        train_ds = make_ds(X_train, y_train, training=True)
        valid_ds = make_ds(X_valid, y_valid, training=False)

        best_cb = BestWeights(monitor="val_loss", mode="min")
        val_cb = PeriodicValidationMAE(valid_ds=valid_ds, every_n_epochs=5)

        his = model.fit(
            train_ds,
            epochs=EPOCH,
            callbacks=[val_cb, best_cb, callback1],
            verbose=2,
        )

        if best_cb.best_weights is None:
            best_weights_per_fold.append(model.get_weights())
        else:
            best_weights_per_fold.append(best_cb.best_weights)

        print("\n\n")




## === cell 15
with strategy.scope():
    infer_model = build_model()

print(
    f"Prepared inference model; will run {len(best_weights_per_fold)} fold-weight sets."
)




## === cell 16
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
g = test_data.groupby("breath_id", sort=False)
test_data["diff_u_in1"] = g["u_in"].diff().fillna(0.0).astype("float32")
test_data["u_in_cumsum"] = g["u_in"].cumsum().astype("float32")




## === cell 17
test_data = dropCols(test_data, cols_2_drop)
test_data = sc.transform(test_data).astype(np.float32, copy=False)
test_data = np.ascontiguousarray(test_data.reshape(-1, 80, test_data.shape[-1]))




## === cell 18
test_data.shape




## === cell 19
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
    .prefetch(tf.data.AUTOTUNE)
)

preds = []
for w in best_weights_per_fold:
    infer_model.set_weights(w)
    preds.append(infer_model.predict(test_ds, verbose=0).reshape(-1, 1))

p = np.mean(np.stack(preds, axis=0), axis=0)  # (n, 1)
print("Pred shape:", p.shape)




## === cell 20
submission_file = pd.read_csv(sample_sub)
if len(submission_file) != p.shape[0]:
    raise ValueError(
        f"Prediction length {p.shape[0]} does not match submission length {len(submission_file)}"
    )

submission_file["pressure"] = p.reshape(-1)
submission_file.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_file.shape)
print(submission_file.head())
