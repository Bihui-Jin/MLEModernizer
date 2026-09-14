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

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler

np.random.seed(42)
tf.random.set_seed(42)

rb = RobustScaler()

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    import multiprocessing

    _cpu = multiprocessing.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(max(1, _cpu // 2))
    tf.config.threading.set_inter_op_parallelism_threads(max(1, _cpu // 2))
except Exception:
    pass

try:
    tf.config.optimizer.set_experimental_options(
        {
            "layout_optimizer": True,
            "constant_folding": True,
            "shape_optimization": True,
            "remapping": True,
            "dependency_optimization": True,
            "arithmetic_optimization": True,
        }
    )
except Exception:
    pass



## === cell 1
train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "u_out": "int8",
        "time_step": "float32",
        "u_in": "float32",
        "pressure": "float32",
    },
)
test = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv",
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "u_out": "int8",
        "time_step": "float32",
        "u_in": "float32",
    },
)




## === cell 2
def add_u_in_features_fast(df: pd.DataFrame) -> None:
    breath = df["breath_id"].to_numpy(copy=False)
    u_in = df["u_in"].to_numpy(copy=False).astype(np.float32, copy=False)

    n = len(df)
    new_breath = np.empty(n, dtype=bool)
    new_breath[0] = True
    new_breath[1:] = breath[1:] != breath[:-1]

    lag1 = np.empty(n, dtype=np.float32)
    lag1[0] = 0.0
    lag1[1:] = u_in[:-1]
    lag1[new_breath] = 0.0

    lag2 = np.empty(n, dtype=np.float32)
    if n >= 2:
        lag2[:2] = 0.0
        lag2[2:] = u_in[:-2]
    else:
        lag2[:] = 0.0
    new_breath_shift1 = np.empty(n, dtype=bool)
    new_breath_shift1[0] = True
    new_breath_shift1[1:] = new_breath[:-1]
    lag2[new_breath | new_breath_shift1] = 0.0

    diff1 = u_in - lag1
    diff2 = u_in - lag2

    cs = np.cumsum(u_in, dtype=np.float32)
    start_idx = np.flatnonzero(new_breath)
    prev_cs = np.empty_like(start_idx, dtype=np.float32)
    prev_cs[0] = 0.0
    if len(start_idx) > 1:
        prev_cs[1:] = cs[start_idx[1:] - 1]
    offset = np.repeat(prev_cs, np.diff(np.append(start_idx, n)))
    cumsum = cs - offset

    df["u_in_lag1"] = lag1
    df["u_in_lag2"] = lag2
    df["u_in_diff1"] = diff1
    df["u_in_diff2"] = diff2
    df["u_in_cumsum"] = cumsum


add_u_in_features_fast(train)
add_u_in_features_fast(test)



## === cell 3
targets = train["pressure"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80, 1)
test_id = test["id"].to_numpy(copy=False)

train.drop(columns=["id", "breath_id", "pressure"], inplace=True)
test.drop(columns=["id", "breath_id"], inplace=True)



## === cell 4
train_np = train.to_numpy(copy=False)
test_np = test.to_numpy(copy=False)

rb.fit(train_np)

train_new = rb.transform(train_np)
test_new = rb.transform(test_np)

train_re = np.ascontiguousarray(
    train_new.reshape(-1, 80, train_new.shape[-1]).astype(np.float32, copy=False)
)
test_re = np.ascontiguousarray(
    test_new.reshape(-1, 80, test_new.shape[-1]).astype(np.float32, copy=False)
)




## === cell 5
def build_model0(input_shape, name):
    model = tf.keras.Sequential(name=name)

    model.add(
        layers.Bidirectional(
            layers.LSTM(440, return_sequences=True, input_shape=input_shape)
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                360,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(layers.Dropout(0.2))

    model.add(layers.Bidirectional(layers.LSTM(260, return_sequences=True)))
    model.add(layers.Dropout(0.25))

    model.add(
        layers.Bidirectional(
            layers.LSTM(
                180,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(layers.Dropout(0.2))

    model.add(layers.Bidirectional(layers.LSTM(100, return_sequences=True)))

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.TimeDistributed(layers.Dense(1, name=name)))

    return model


def build_model1(input_shape, name):
    model = tf.keras.Sequential(name=name)

    model.add(
        layers.Bidirectional(
            layers.LSTM(520, return_sequences=True, input_shape=input_shape)
        )
    )
    model.add(layers.Bidirectional(layers.LSTM(400, return_sequences=True)))
    model.add(layers.Dropout(0.2))

    model.add(
        layers.Bidirectional(
            layers.LSTM(
                320,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(layers.Dropout(0.3))

    model.add(layers.Bidirectional(layers.LSTM(240, return_sequences=True)))

    model.add(
        layers.Bidirectional(
            layers.LSTM(
                160,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(layers.Dropout(0.25))

    model.add(layers.Bidirectional(layers.LSTM(80, return_sequences=True)))

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.TimeDistributed(layers.Dense(1, name=name)))

    return model


def build_model2(input_shape, name):
    model = tf.keras.Sequential(name=name)

    model.add(
        layers.Bidirectional(
            layers.LSTM(440, return_sequences=True, input_shape=input_shape)
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(360, return_sequences=True, kernel_initializer="random_normal")
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                260,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Bidirectional(
            layers.LSTM(180, return_sequences=True, kernel_initializer="random_normal")
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(120, return_sequences=True, kernel_initializer="random_normal")
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




## === cell 6
def build_model(input_size=[80, train_re.shape[-1]]):
    model0 = build_model0(input_size, name="model0")
    model1 = build_model1(input_size, name="model1")
    model2 = build_model1(input_size, name="model2")
    dupmodel2 = build_model2(input_size, name="dupmodel2")

    ensemblemodel = build_ensembleModel([80, 5], name="ensemble")

    Input = tf.keras.layers.Input(input_size)

    model0_out = model0(Input)
    model1_out = model1(Input)
    model2_out = model2(Input)
    dupmodel_2 = dupmodel2(Input)

    mean_out = layers.Add()([model0_out, model1_out, model2_out, dupmodel_2])
    mean_out = layers.Lambda(lambda x: x / 4, name="mean")(mean_out)

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
    losses = {
        "model0": tf.keras.losses.MeanAbsoluteError(),
        "model1": tf.keras.losses.MeanAbsoluteError(),
        "model2": tf.keras.losses.MeanAbsoluteError(),
        "dupmodel2": tf.keras.losses.MeanAbsoluteError(),
        "mean": tf.keras.losses.MeanAbsoluteError(),
        "ensemble": tf.keras.losses.MeanAbsoluteError(),
    }
    metrics = {
        "model0": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        "model1": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        "model2": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        "dupmodel2": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        "mean": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        "ensemble": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
    }

    model.compile(
        optimizer=opt,
        loss=losses,
        metrics=metrics,
        run_eagerly=False,
        jit_compile=True,
    )
    return model




## === cell 7
EPOCH = 400
BATCH_SIZE = 768

try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
    print("Using TPU strategy")
except Exception as e:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print("Using MirroredStrategy on multiple GPUs")
    else:
        strategy = tf.distribute.get_strategy()
        print("Using default strategy (CPU/GPU). TPU not available:", repr(e))



## === cell 8
checkpoint_path = "LModel.keras"


def make_train_val_datasets(X_train, y_train, X_val, y_val, batch_size, seed=42):
    y_train_multi = {
        "model0": y_train,
        "model1": y_train,
        "model2": y_train,
        "dupmodel2": y_train,
        "mean": y_train,
        "ensemble": y_train,
    }
    y_val_multi = {
        "model0": y_val,
        "model1": y_val,
        "model2": y_val,
        "dupmodel2": y_val,
        "mean": y_val,
        "ensemble": y_val,
    }

    options = tf.data.Options()
    options.experimental_deterministic = True

    shuffle_buf = int(min(len(X_train), max(batch_size * 32, 8192)))

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_multi))
    train_ds = train_ds.shuffle(
        buffer_size=shuffle_buf, seed=seed, reshuffle_each_iteration=True
    )
    train_ds = (
        train_ds.batch(batch_size, drop_remainder=False)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    train_ds = train_ds.with_options(options)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_multi))
    val_ds = (
        val_ds.batch(batch_size, drop_remainder=False)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = val_ds.with_options(options)
    return train_ds, val_ds


with strategy.scope():
    model = build_model()

    n = train_re.shape[0]
    rng = np.random.RandomState(42)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1.0 - 0.2))
    tr_idx = idx[:split]
    va_idx = idx[split:]

    X_train = np.ascontiguousarray(train_re[tr_idx])
    y_train = np.ascontiguousarray(targets[tr_idx])
    X_val = np.ascontiguousarray(train_re[va_idx])
    y_val = np.ascontiguousarray(targets[va_idx])

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", verbose=0, factor=0.9, patience=10
    )

    save_call = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        verbose=0,
        monitor="val_ensemble_mae",
        save_best_only=True,
    )

    train_ds, val_ds = make_train_val_datasets(
        X_train, y_train, X_val, y_val, BATCH_SIZE, seed=42
    )

    his = model.fit(
        train_ds,
        validation_data=val_ds,
        validation_freq=10,
        epochs=EPOCH,
        callbacks=[save_call, reduce_lr],
        verbose=2,
    )



## === cell 9
try:
    tf.keras.backend.clear_session()
except Exception:
    pass

if os.path.exists(checkpoint_path):
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
else:
    print("Warning: checkpoint not found; using current in-memory model.")



## === cell 10
pred_options = tf.data.Options()
pred_options.experimental_deterministic = True

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_re)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)
test_ds = test_ds.with_options(pred_options)

test_pred = model.predict(test_ds, verbose=1)



## === cell 11
pred_pressure = np.asarray(test_pred[-2]).reshape(-1).astype(np.float32)



## === cell 12
submission_file = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

if not submission_file["id"].is_monotonic_increasing:
    submission_file = submission_file.sort_values("id").reset_index(drop=True)

if len(submission_file) != len(pred_pressure):
    raise ValueError(
        f"Prediction length mismatch: submission has {len(submission_file)} rows, "
        f"pred has {len(pred_pressure)}"
    )

submission_file["pressure"] = pred_pressure
submission_file.to_csv("submission.csv", index=False)
print(submission_file.head())
print("Wrote submission.csv with shape:", submission_file.shape)
