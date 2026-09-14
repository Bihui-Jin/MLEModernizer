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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")  # keep original intent

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

rc = RobustScaler()



## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 2
def dropCols(df, cols):
    df = df.copy()
    df.drop(cols, axis=1, inplace=True)
    return df




## === cell 3
def preProcess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    breath_id = df["breath_id"].to_numpy()
    u_in = df["u_in"].to_numpy(dtype=np.float64, copy=False)
    time_step = df["time_step"].to_numpy(dtype=np.float64, copy=False)

    new_breath = np.empty_like(breath_id, dtype=bool)
    new_breath[0] = True
    new_breath[1:] = breath_id[1:] != breath_id[:-1]

    u_in_lag1 = np.empty_like(u_in, dtype=np.float64)
    u_in_lag1[0] = 0.0
    u_in_lag1[1:] = u_in[:-1]
    u_in_lag1[new_breath] = 0.0

    df["u_in_lag1"] = u_in_lag1
    df["diff_u_in1"] = u_in - u_in_lag1

    area_step = time_step * u_in
    csum_area = np.cumsum(area_step)
    csum_uin = np.cumsum(u_in)

    start_idx = np.flatnonzero(new_breath)
    start_area = csum_area[start_idx] - area_step[start_idx]
    start_uin = csum_uin[start_idx] - u_in[start_idx]

    offsets_area = np.empty_like(csum_area)
    offsets_uin = np.empty_like(csum_uin)
    offsets_area[start_idx] = start_area
    offsets_uin[start_idx] = start_uin

    offsets_area = np.maximum.accumulate(offsets_area)
    offsets_uin = np.maximum.accumulate(offsets_uin)

    df["area"] = csum_area - offsets_area
    df["u_in_cumsum"] = csum_uin - offsets_uin
    return df




## === cell 4
train_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "u_out": "int8",
    "u_in": "float32",
    "time_step": "float32",
    "pressure": "float32",
}
train_data = pd.read_csv(train_path, dtype=train_dtypes)
train_data = preProcess(train_data)



## === cell 5
cols_2_drop = ["id", "breath_id", "time_step"]



## === cell 6
train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")



## === cell 7
assert train_df.shape[0] == Y.shape[0]
assert train_df.isna().sum().sum() == 0



## === cell 8
rc.fit(train_df)
train_df = rc.transform(train_df).astype(np.float32, copy=False)

train_df = np.ascontiguousarray(train_df.reshape(-1, 80, train_df.shape[-1]))
Y = np.ascontiguousarray(Y.values.astype(np.float32, copy=False).reshape(-1, 80, 1))

train_df.shape, Y.shape




## === cell 9
def _import_tf():
    import tensorflow as tf
    from tensorflow.keras import layers

    try:
        tf.keras.utils.set_random_seed(42)
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    try:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

    try:
        cpu = os.cpu_count() or 2
        tf.config.threading.set_inter_op_parallelism_threads(min(4, cpu))
        tf.config.threading.set_intra_op_parallelism_threads(max(1, cpu - 2))
    except Exception:
        pass

    return tf, layers


tf, layers = _import_tf()




## === cell 10
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
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                260,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                180,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(layers.Bidirectional(layers.LSTM(100, return_sequences=True)))
    model.add(layers.TimeDistributed(layers.Dense(64, activation="relu")))
    model.add(layers.TimeDistributed(layers.Dense(1, name=name)))
    return model


def build_model1(input_shape, name):
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
                dropout=0.2,
                return_sequences=True,
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                260,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                180,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                100,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                80,
                return_sequences=True,
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )
        )
    )
    model.add(
        layers.TimeDistributed(
            layers.Dense(
                64,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )
        )
    )
    model.add(layers.TimeDistributed(layers.Dense(1, name=name)))
    return model


def build_ensembleModel(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(
        layers.Dense(
            16,
            activation="relu",
            kernel_initializer=tf.keras.initializers.HeNormal(),
            input_shape=input_shape,
        )
    )
    model.add(layers.Dense(1, name=name))
    return model




## === cell 11
def build_model(input_size=None):
    if input_size is None:
        input_size = [80, train_df.shape[-1]]

    model0 = build_model0(input_size, name="model0")
    dupmodel0 = build_model0(input_size, name="dupmodel0")
    model1 = build_model1(input_size, name="model1")
    ensemblemodel = build_ensembleModel([80, 4], name="ensemble")

    Input = tf.keras.layers.Input(input_size)

    model0_out = model0(Input)
    dupmodel_0 = dupmodel0(Input)
    model1_out = model1(Input)

    mean3_out = layers.Add()([model0_out, dupmodel_0, model1_out])
    mean3_out = layers.Lambda(lambda x: x / 3.0, name="mean")(mean3_out)

    concatenateModelsout = layers.Concatenate()(
        [model0_out, dupmodel_0, model1_out, mean3_out]
    )
    ensemble_out = ensemblemodel(concatenateModelsout)

    model = tf.keras.Model(
        inputs=Input,
        outputs=[model0_out, dupmodel_0, model1_out, mean3_out, ensemble_out],
    )

    opt = tf.keras.optimizers.Adam()
    losses = {
        "model0": tf.keras.losses.MeanAbsoluteError(),
        "dupmodel0": tf.keras.losses.MeanAbsoluteError(),
        "model1": tf.keras.losses.MeanAbsoluteError(),
        "mean": tf.keras.losses.MeanAbsoluteError(),
        "ensemble": tf.keras.losses.MeanAbsoluteError(),
    }
    metrics = {
        "model0": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        "dupmodel0": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        "model1": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        "mean": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        "ensemble": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
    }

    model.compile(optimizer=opt, loss=losses, metrics=metrics, jit_compile=True)
    return model




## === cell 12
callback1 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.9,
    patience=15,
    verbose=1,
)



## === cell 13
EPOCH = 35
BATCH_SIZE = 256

np.random.seed(42)
tf.random.set_seed(42)


def make_indexed_ds(X, y_list, idx, batch_size, cache: bool):
    options = tf.data.Options()
    options.deterministic = True

    idx = tf.convert_to_tensor(idx, dtype=tf.int32)

    X_tf = tf.convert_to_tensor(X)
    y_tf = tuple(tf.convert_to_tensor(y) for y in y_list)

    def _gather(i):
        x = tf.gather(X_tf, i)
        ys = tuple(tf.gather(t, i) for t in y_tf)
        return x, ys

    ds = tf.data.Dataset.from_tensor_slices(idx).with_options(options)
    ds = ds.map(_gather, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


kf = KFold(n_splits=3, shuffle=True, random_state=42)
trained_model_paths = []
histories = []

Y_multi = (Y, Y, Y, Y, Y)

best_models = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
    print("-" * 15, ">", f"Fold {fold}", "<", "-" * 15)

    train_ds = make_indexed_ds(train_df, Y_multi, train_idx, BATCH_SIZE, cache=False)
    valid_ds = make_indexed_ds(train_df, Y_multi, valid_idx, BATCH_SIZE, cache=True)

    tf.keras.backend.clear_session()
    model = build_model()

    fold_path = f"AdamPressurePreModel{fold}.h5"
    callback0 = tf.keras.callbacks.ModelCheckpoint(
        fold_path,
        monitor="val_ensemble_mae",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    his = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCH,
        callbacks=[callback0, callback1],
        verbose=2,
    )
    histories.append(his.history)
    trained_model_paths.append(fold_path)

    best_m = tf.keras.models.load_model(fold_path)
    best_models.append(best_m)



## === cell 14
models = best_models
[m.name for m in models]



## === cell 15
test_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "u_out": "int8",
    "u_in": "float32",
    "time_step": "float32",
}
test_raw = pd.read_csv(test_path, dtype=test_dtypes)
test_data = preProcess(test_raw)
test_data = dropCols(test_data, cols_2_drop)
test_data = rc.transform(test_data).astype(np.float32, copy=False)
test_data = np.ascontiguousarray(test_data.reshape(-1, 80, test_data.shape[-1]))

test_data.shape



## === cell 16
preds = []
for m in models:
    test_ds = (
        tf.data.Dataset.from_tensor_slices(test_data)
        .batch(512)
        .prefetch(tf.data.AUTOTUNE)
    )
    p = m.predict(test_ds, verbose=0)  # list of 5 outputs
    predictions_table = np.concatenate(p, axis=-1)  # (n_breaths, 80, 5)
    mean_pre = np.mean(predictions_table, axis=-1)  # (n_breaths, 80)
    preds.append(mean_pre.reshape(-1, 1))  # (n_rows, 1)

preds[0].shape, len(preds)



## === cell 17
predictions_table = np.concatenate(preds, axis=-1)  # (n_rows, 3)
median_pre = np.median(predictions_table, axis=-1)  # (n_rows,)
median_pre.shape



## === cell 18
Y_flat = Y.reshape(-1)
unique_pressures = np.unique(Y_flat)
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = (sorted_pressures[1] - sorted_pressures[0]).item()
PRESSURE_MIN = sorted_pressures[0].item()
PRESSURE_MAX = sorted_pressures[-1].item()

rounding_pre = (
    np.round((median_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
rounding_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)

rounding_pre.shape



## === cell 19
submission_file = pd.read_csv(sample_sub)
assert len(submission_file) == rounding_pre.shape[0], (
    len(submission_file),
    rounding_pre.shape[0],
)

submission_file["pressure"] = rounding_pre.astype(np.float32)
submission_file.to_csv("submission.csv", index=False)

print(submission_file.head())
print("Wrote submission.csv with shape:", submission_file.shape)
