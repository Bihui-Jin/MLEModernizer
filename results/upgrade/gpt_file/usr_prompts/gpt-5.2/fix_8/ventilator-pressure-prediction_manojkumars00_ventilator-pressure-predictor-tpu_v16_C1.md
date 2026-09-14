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

# 5. Target score

0.1832862225159837

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

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
X_scaled = rc.transform(train_df).astype(np.float32, copy=False)

n_features = X_scaled.shape[-1]
X = X_scaled.reshape(-1, 80, n_features)
Y_arr = Y.to_numpy(dtype=np.float32, copy=False).reshape(-1, 80, 1)

del train_data, train_df, X_scaled  # free RAM early

X.shape, Y_arr.shape




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




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


kf = KFold(n_splits=3, shuffle=True, random_state=42)
histories = []
best_models = []

tf.keras.backend.clear_session()
template_model = build_model()
template_model_config = template_model.get_config()
del template_model

for fold, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
    print("-" * 15, ">", f"Fold {fold}", "<", "-" * 15)

    X_tr = X[train_idx]
    X_va = X[valid_idx]
    Y_tr = Y_arr[train_idx]
    Y_va = Y_arr[valid_idx]

    train_ds = make_numpy_ds(X_tr, Y_tr, BATCH_SIZE, cache=False)
    valid_ds = make_numpy_ds(X_va, Y_va, BATCH_SIZE, cache=True)

    tf.keras.backend.clear_session()
    model = tf.keras.Model.from_config(template_model_config, custom_objects=None)
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

    best_m = tf.keras.Model.from_config(template_model_config, custom_objects=None)
    best_m.compile(optimizer=opt, loss=losses, metrics=metrics, jit_compile=True)
    best_m.load_weights(fold_path)
    best_models.append(best_m)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3598908594.py in <cell line: 0>()
     44 
     45     tf.keras.backend.clear_session()
---> 46     model = tf.keras.Model.from_config(template_model_config, custom_objects=None)
     47     # Must compile exactly as build_model does (same optimizer/loss/metrics/jit)
     48     opt = tf.keras.optimizers.Adam()

/usr/local/lib/python3.11/dist-packages/keras/src/models/model.py in from_config(cls, config, custom_objects)
    580             from keras.src.models.functional import functional_from_config
    581 
--> 582             return functional_from_config(
    583                 cls, config, custom_objects=custom_objects
    584             )

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in functional_from_config(cls, config, custom_objects)
    549     # First, we create all layers and enqueue nodes to be processed
    550     for layer_data in functional_config["layers"]:
--> 551         process_layer(layer_data)
    552 
    553     # Then we process nodes in order of layer depth.

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in process_layer(layer_data)
    521             )
    522         else:
--> 523             layer = serialization_lib.deserialize_keras_object(
    524                 layer_data, custom_objects=custom_objects
    525             )

/usr/local/lib/python3.11/dist-packages/keras/src/saving/serialization_lib.py in deserialize_keras_object(config, custom_objects, safe_mode, **kwargs)
    716     with custom_obj_scope, safe_mode_scope:
    717         try:
--> 718             instance = cls.from_config(inner_config)
    719         except TypeError as e:
    720             raise TypeError(

/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/lambda_layer.py in from_config(cls, config, custom_objects, safe_mode)
    188             and fn_config["class_name"] == "__lambda__"
    189         ):
--> 190             cls._raise_for_lambda_deserialization("function", safe_mode)
    191             inner_config = fn_config["config"]
    192             fn = python_utils.func_load(

/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/lambda_layer.py in _raise_for_lambda_deserialization(arg_name, safe_mode)
    170     def _raise_for_lambda_deserialization(arg_name, safe_mode):
    171         if safe_mode:
--> 172             raise ValueError(
    173                 "The `{arg_name}` of this `Lambda` layer is a Python lambda. "
    174                 "Deserializing it is unsafe. If you trust the source of the "

ValueError: The `{arg_name}` of this `Lambda` layer is a Python lambda. Deserializing it is unsafe. If you trust the source of the config artifact, you can override this error by passing `safe_mode=False` to `from_config()`, or calling `keras.config.enable_unsafe_deserialization().

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
test_scaled = rc.transform(test_data).astype(np.float32, copy=False)
test_data = test_scaled.reshape(-1, 80, test_scaled.shape[-1])

del test_raw, test_scaled

test_data.shape



## === cell 16
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_data).batch(512).prefetch(tf.data.AUTOTUNE)
)

preds = []
for m in models:
    p = m.predict(test_ds, verbose=0)  # list of 5 outputs
    predictions_table = np.concatenate(p, axis=-1)  # (n_breaths, 80, 5)
    mean_pre = np.mean(predictions_table, axis=-1)  # (n_breaths, 80)
    preds.append(mean_pre.reshape(-1, 1))  # (n_rows, 1)

preds[0].shape, len(preds)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/2168287438.py in <cell line: 0>()
     10     preds.append(mean_pre.reshape(-1, 1))  # (n_rows, 1)
     11 
---> 12 preds[0].shape, len(preds)
     13 

IndexError: list index out of range

## === cell 17
predictions_table = np.concatenate(preds, axis=-1)  # (n_rows, 3)
median_pre = np.median(predictions_table, axis=-1)  # (n_rows,)
median_pre.shape



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2913669540.py in <cell line: 0>()
----> 1 predictions_table = np.concatenate(preds, axis=-1)  # (n_rows, 3)
      2 median_pre = np.median(predictions_table, axis=-1)  # (n_rows,)
      3 median_pre.shape
      4 

ValueError: need at least one array to concatenate

## === cell 18
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

rounding_pre.shape



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1985019423.py in <cell line: 0>()
      8 
      9 rounding_pre = (
---> 10     np.round((median_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
     11 )
     12 rounding_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)

NameError: name 'median_pre' is not defined

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

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1249828538.py in <cell line: 0>()
      1 submission_file = pd.read_csv(sample_sub)
----> 2 assert len(submission_file) == rounding_pre.shape[0], (
      3     len(submission_file),
      4     rounding_pre.shape[0],
      5 )

NameError: name 'rounding_pre' is not defined
