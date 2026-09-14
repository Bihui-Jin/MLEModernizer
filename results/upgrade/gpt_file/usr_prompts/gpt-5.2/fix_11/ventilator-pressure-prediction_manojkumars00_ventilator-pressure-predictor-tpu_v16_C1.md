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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("PYTHONHASHSEED", "42")

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

rc = RobustScaler()



## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 2
def dropCols(df, cols):
    return df.drop(columns=cols)




## === cell 3
def preProcess(df: pd.DataFrame) -> pd.DataFrame:
    breath_id = df["breath_id"].to_numpy(copy=False)

    u_in = df["u_in"].to_numpy(dtype=np.float64, copy=False)
    time_step = df["time_step"].to_numpy(dtype=np.float64, copy=False)

    new_breath = np.empty(breath_id.shape[0], dtype=bool)
    new_breath[0] = True
    new_breath[1:] = breath_id[1:] != breath_id[:-1]

    u_in_lag1 = np.empty_like(u_in, dtype=np.float64)
    u_in_lag1[0] = 0.0
    u_in_lag1[1:] = u_in[:-1]
    u_in_lag1[new_breath] = 0.0

    df["u_in_lag1"] = u_in_lag1
    df["diff_u_in1"] = u_in - u_in_lag1

    area_step = time_step * u_in

    start_idx = np.flatnonzero(new_breath)
    end_idx = np.append(start_idx[1:], breath_id.shape[0])
    sizes = end_idx - start_idx

    csum_area = np.cumsum(area_step, dtype=np.float64)
    csum_uin = np.cumsum(u_in, dtype=np.float64)

    start_area = csum_area[start_idx] - area_step[start_idx]
    start_uin = csum_uin[start_idx] - u_in[start_idx]

    start_area_rep = np.repeat(start_area, sizes)
    start_uin_rep = np.repeat(start_uin, sizes)

    df["area"] = csum_area - start_area_rep
    df["u_in_cumsum"] = csum_uin - start_uin_rep
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
X_scaled = rc.transform(train_df)
X_scaled = np.asarray(X_scaled, dtype=np.float32, order="C")

n_features = X_scaled.shape[-1]
X = X_scaled.reshape(-1, 80, n_features)

Y_arr = Y.to_numpy(dtype=np.float32, copy=False).reshape(-1, 80, 1)

del train_data, train_df, X_scaled  # free RAM early

print("X shape:", X.shape, "Y shape:", Y_arr.shape)




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
        tf.config.threading.set_intra_op_parallelism_threads(max(1, cpu - 1))
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
        input_size = [80, X.shape[-1]]

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


def make_numpy_ds(X_np, y_np, batch_size, cache: bool):
    options = tf.data.Options()
    options.deterministic = True
    y_tuple = (y_np, y_np, y_np, y_np, y_np)
    ds = tf.data.Dataset.from_tensor_slices((X_np, y_tuple)).with_options(options)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


kf = KFold(n_splits=3, shuffle=True, random_state=42)
histories = []
best_models = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
    print("-" * 15, ">", f"Fold {fold}", "<", "-" * 15)

    X_tr = X[train_idx]
    X_va = X[valid_idx]
    Y_tr = Y_arr[train_idx]
    Y_va = Y_arr[valid_idx]

    train_ds = make_numpy_ds(X_tr, Y_tr, BATCH_SIZE, cache=True)
    valid_ds = make_numpy_ds(X_va, Y_va, BATCH_SIZE, cache=True)

    tf.keras.backend.clear_session()
    model = build_model()

    fold_path = f"AdamPressurePreModel{fold}.weights.h5"
    callback0 = tf.keras.callbacks.ModelCheckpoint(
        fold_path,
        monitor="val_ensemble_mae",
        mode="min",
        save_best_only=True,
        verbose=1,
        save_weights_only=True,
    )

    his = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCH,
        callbacks=[callback0, callback1],
        verbose=2,
    )
    histories.append(his.history)

    model.load_weights(fold_path)
    best_models.append(model)

models = best_models
print("Trained models:", [m.name for m in models])



## === cell 14
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

test_scaled = rc.transform(test_data)
test_scaled = np.asarray(test_scaled, dtype=np.float32, order="C")
test_data = test_scaled.reshape(-1, 80, test_scaled.shape[-1])

del test_raw, test_scaled

print("Test breath tensor shape:", test_data.shape)



## === cell 15
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_data).batch(1024).prefetch(tf.data.AUTOTUNE)
)


@tf.function(reduce_retracing=True)
def _predict_ensemble(model, x_batch):
    outs = model(x_batch, training=False)
    return outs[4]


preds = []
for m in models:
    out_batches = []
    for xb in test_ds:
        out_batches.append(_predict_ensemble(m, xb))
    ensemble_out = tf.concat(out_batches, axis=0).numpy()  # (n_breaths, 80, 1)
    preds.append(ensemble_out.reshape(-1, 1))

print("Per-model flattened pred shape:", preds[0].shape, "n_models:", len(preds))



## === cell 16
predictions_table = np.concatenate(preds, axis=-1)  # (n_rows, 3)
median_pre = np.median(predictions_table, axis=-1)  # (n_rows,)
print("Median pred shape:", median_pre.shape)



## === cell 17
Y_flat = Y_arr.reshape(-1)
unique_pressures = np.unique(Y_flat)
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = (sorted_pressures[1] - sorted_pressures[0]).item()
PRESSURE_MIN = sorted_pressures[0].item()
PRESSURE_MAX = sorted_pressures[-1].item()

rounding_pre = (
    np.round((median_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
rounding_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)

print("Rounded pred shape:", rounding_pre.shape)



## === cell 18
submission_file = pd.read_csv(sample_sub)
assert len(submission_file) == rounding_pre.shape[0], (
    len(submission_file),
    rounding_pre.shape[0],
)

submission_file["pressure"] = rounding_pre.astype(np.float32)
submission_file.to_csv("submission.csv", index=False)

print(submission_file.head())
print("Wrote submission.csv with shape:", submission_file.shape)
