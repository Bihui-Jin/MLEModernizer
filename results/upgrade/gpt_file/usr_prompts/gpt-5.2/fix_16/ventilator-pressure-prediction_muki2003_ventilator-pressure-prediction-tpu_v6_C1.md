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
seaborn==0.12.2
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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

print("TensorFlow:", tf.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    import multiprocessing

    ncpu = multiprocessing.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(max(1, min(8, ncpu)))
    tf.config.threading.set_inter_op_parallelism_threads(max(1, min(2, ncpu // 4)))
except Exception:
    pass

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

try:
    tf.config.experimental.enable_tensor_float_32_execution(True)
except Exception:
    pass



## === cell 1
rb = RobustScaler()



## === cell 2
DATA_DIR = "../input/ventilator-pressure-prediction"

train = pd.read_csv(
    f"{DATA_DIR}/train.csv",
    dtype={
        "id": np.int32,
        "breath_id": np.int32,
        "R": np.int16,
        "C": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
        "pressure": np.float32,
    },
)
test = pd.read_csv(
    f"{DATA_DIR}/test.csv",
    dtype={
        "id": np.int32,
        "breath_id": np.int32,
        "R": np.int16,
        "C": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
    },
)

print(train.shape, test.shape)
train.head()



## === cell 3
pass




## === cell 4
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    breath_id = df["breath_id"].to_numpy(copy=False)
    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)

    is_new = np.empty(len(df), dtype=bool)
    is_new[0] = True
    is_new[1:] = breath_id[1:] != breath_id[:-1]

    u_in_lag1 = np.empty_like(u_in, dtype=np.float32)
    u_in_lag1[0] = 0.0
    u_in_lag1[1:] = u_in[:-1]
    u_in_lag1[is_new] = 0.0

    u_in_lag2 = np.empty_like(u_in, dtype=np.float32)
    u_in_lag2[:2] = 0.0
    u_in_lag2[2:] = u_in[:-2]
    u_in_lag2[is_new] = 0.0
    prev_is_new = np.empty_like(is_new)
    prev_is_new[0] = False
    prev_is_new[1:] = is_new[:-1]
    u_in_lag2[prev_is_new] = 0.0

    global_cum = np.cumsum(u_in, dtype=np.float32)
    start_idx = np.flatnonzero(is_new)
    start_cum = global_cum[start_idx] - u_in[start_idx]
    offsets = np.repeat(start_cum, np.diff(np.append(start_idx, len(df))))
    u_in_cumsum = global_cum - offsets

    df["u_in_lag1"] = u_in_lag1
    df["u_in_lag2"] = u_in_lag2
    df["u_in_diff1"] = (u_in - u_in_lag1).astype(np.float32, copy=False)
    df["u_in_diff2"] = (u_in - u_in_lag2).astype(np.float32, copy=False)
    df["u_in_cumsum"] = u_in_cumsum.astype(np.float32, copy=False)
    return df


n_train = len(train)

train = add_features(train)
test = add_features(test)

train.head()



## === cell 5
targets = train["pressure"].to_numpy(dtype=np.float32).reshape(-1, 80, 1)

test_id = test["id"].copy()

train.drop(columns=["id", "breath_id", "pressure"], inplace=True)
drop_cols_test = ["id", "breath_id"]
if "pressure" in test.columns:
    drop_cols_test.append("pressure")
test.drop(columns=drop_cols_test, inplace=True)

missing_in_test = [c for c in train.columns if c not in test.columns]
missing_in_train = [c for c in test.columns if c not in train.columns]
if missing_in_test or missing_in_train:
    raise ValueError(
        f"Train/test feature mismatch. Missing in test: {missing_in_test}. Missing in train: {missing_in_train}."
    )
test = test[train.columns]

print(
    "train features:",
    train.shape,
    "targets:",
    targets.shape,
    "test features:",
    test.shape,
)



## === cell 6
train_np = np.ascontiguousarray(train.to_numpy(dtype=np.float32, copy=False))
test_np = np.ascontiguousarray(test.to_numpy(dtype=np.float32, copy=False))

rb.fit(train_np)

train_new = rb.transform(train_np)
test_new = rb.transform(test_np)

train_new = np.ascontiguousarray(train_new.astype(np.float32, copy=False))
test_new = np.ascontiguousarray(test_new.astype(np.float32, copy=False))

if train_new.shape[0] % 80 != 0 or test_new.shape[0] % 80 != 0:
    raise ValueError(
        f"Row count not divisible by 80. train_rows={train_new.shape[0]}, test_rows={test_new.shape[0]}"
    )

train_re = train_new.reshape(-1, 80, train_new.shape[-1])
test_re = test_new.reshape(-1, 80, test_new.shape[-1])

train_re = np.ascontiguousarray(train_re)
test_re = np.ascontiguousarray(test_re)

print("train_re:", train_re.shape, "test_re:", test_re.shape)




## === cell 7
def build_model0(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                440,
                return_sequences=True,
                input_shape=input_shape,
                dropout=0.0,
                recurrent_dropout=0.0,
                unroll=False,
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                360,
                dropout=0.2,
                recurrent_dropout=0.0,
                unroll=False,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                260,
                dropout=0.25,
                recurrent_dropout=0.0,
                unroll=False,
                return_sequences=True,
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                180,
                dropout=0.2,
                recurrent_dropout=0.0,
                unroll=False,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                100,
                return_sequences=True,
                dropout=0.0,
                recurrent_dropout=0.0,
                unroll=False,
            )
        )
    )
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.TimeDistributed(layers.Dense(1, name=name)))
    return model


def build_model1(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                520,
                return_sequences=True,
                input_shape=input_shape,
                dropout=0.0,
                recurrent_dropout=0.0,
                unroll=False,
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                400,
                dropout=0.2,
                recurrent_dropout=0.0,
                unroll=False,
                return_sequences=True,
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                320,
                dropout=0.3,
                recurrent_dropout=0.0,
                unroll=False,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                240,
                return_sequences=True,
                dropout=0.0,
                recurrent_dropout=0.0,
                unroll=False,
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                160,
                dropout=0.25,
                recurrent_dropout=0.0,
                unroll=False,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                80,
                return_sequences=True,
                dropout=0.0,
                recurrent_dropout=0.0,
                unroll=False,
            )
        )
    )
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.TimeDistributed(layers.Dense(1, name=name)))
    return model


def build_model2(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                440,
                return_sequences=True,
                input_shape=input_shape,
                dropout=0.0,
                recurrent_dropout=0.0,
                unroll=False,
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                360,
                return_sequences=True,
                dropout=0.0,
                recurrent_dropout=0.0,
                unroll=False,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                260,
                dropout=0.3,
                recurrent_dropout=0.0,
                unroll=False,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                180,
                return_sequences=True,
                dropout=0.0,
                recurrent_dropout=0.0,
                unroll=False,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                120,
                return_sequences=True,
                dropout=0.0,
                recurrent_dropout=0.0,
                unroll=False,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(layers.Dense(64, activation="linear"))
    model.add(layers.Dense(1, activation="linear", name=name))
    return model


def build_ensembleModel(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(layers.Dense(16, activation="relu", input_shape=input_shape))
    model.add(layers.Dense(1, name=name))
    return model




## === cell 8
def build_model(input_size=(80, None)):
    model0 = build_model0(input_size, name="model0")
    model1 = build_model1(input_size, name="model1")
    model2 = build_model1(input_size, name="model2")
    dupmodel2 = build_model2(input_size, name="dupmodel2")

    ensemblemodel = build_ensembleModel((80, 5), name="ensemble")

    Input = tf.keras.layers.Input(shape=input_size)

    model0_out = model0(Input)
    model1_out = model1(Input)
    model2_out = model2(Input)
    dupmodel_2 = dupmodel2(Input)

    mean_out = layers.Add()([model0_out, model1_out, model2_out, dupmodel_2])
    mean_out = layers.Lambda(lambda x: x / 4.0, name="mean")(mean_out)

    concatenateModelsout = layers.Concatenate()(
        [model0_out, model1_out, model2_out, dupmodel_2, mean_out]
    )
    ensemble_out = ensemblemodel(concatenateModelsout)

    model = tf.keras.Model(
        inputs=Input,
        outputs=[
            model0_out,
            model1_out,
            model2_out,
            dupmodel_2,
            mean_out,
            ensemble_out,
        ],
    )

    opt = tf.keras.optimizers.Adam()

    model.compile(
        optimizer=opt,
        loss=[tf.keras.losses.MeanAbsoluteError()] * 6,
        metrics=[["mae"]] * 6,
        run_eagerly=False,
        jit_compile=True,
        steps_per_execution=16,
    )
    return model




## === cell 9
EPOCH = 400
BATCH_SIZE = 768

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print("Running on default strategy (CPU/GPU). TPU not available:", repr(e))



## === cell 10
y_all = [targets] * 6

with strategy.scope():
    model = build_model(input_size=(80, train_re.shape[-1]))

idx = np.arange(train_re.shape[0])
idx_train, idx_valid = train_test_split(
    idx, test_size=0.2, shuffle=True, random_state=42
)

X_train, X_valid = train_re[idx_train], train_re[idx_valid]

y_tensor = targets  # (n_breaths, 80, 1)
y_train = y_tensor[idx_train]
y_valid = y_tensor[idx_valid]
y_train_list = [y_train] * 6
y_valid_list = [y_valid] * 6


def make_ds(X, y_list, training: bool):
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True

    X_t = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tup = tuple(tf.convert_to_tensor(y, dtype=tf.float32) for y in y_list)

    ds = tf.data.Dataset.from_tensor_slices((X_t, y_tup)).with_options(options)
    if training:
        ds = ds.shuffle(
            buffer_size=min(int(X.shape[0]), 8192),
            seed=42,
            reshuffle_each_iteration=True,
        )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_ds(X_train, y_train_list, training=True)
valid_ds = make_ds(X_valid, y_valid_list, training=False)

reduce_lr = ReduceLROnPlateau(monitor="val_loss", verbose=0, factor=0.9, patience=10)

ckpt_path = "LModel.weights.h5"
save_call = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path,
    verbose=0,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    save_weights_only=True,
)

his = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCH,
    shuffle=False,  # shuffle handled by dataset
    callbacks=[save_call, reduce_lr],
    verbose=2,
)

stats = pd.DataFrame(his.history)
print(stats.tail(3))



## === cell 11
if "ckpt_path" in globals() and os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)
    print("Loaded best weights from", ckpt_path)
else:
    print("Checkpoint not found, using in-memory model.")



## === cell 12
options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True

test_t = tf.convert_to_tensor(test_re, dtype=tf.float32)
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_t)
    .with_options(options)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_pred = model.predict(test_ds, verbose=1)
ensemble_pred = np.asarray(test_pred[-1]).reshape(-1)
print("Pred shape:", ensemble_pred.shape)



## === cell 13
submission_file = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

if len(submission_file) != len(ensemble_pred):
    raise ValueError(
        f"Submission length mismatch: sample={len(submission_file)} pred={len(ensemble_pred)}"
    )

submission_file["pressure"] = ensemble_pred.astype(np.float32)
submission_path = "submission.csv"
submission_file.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_file.head())
