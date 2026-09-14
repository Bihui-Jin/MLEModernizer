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

DATA_DIR = "/kaggle/input"
COMP_DIR = "/kaggle/input/ventilator-pressure-prediction"

train_path = os.path.join(COMP_DIR, "train.csv")
test_path = os.path.join(COMP_DIR, "test.csv")
sample_sub_path = os.path.join(COMP_DIR, "sample_submission.csv")
if not os.path.exists(train_path):
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

print("train_path:", train_path)
print("test_path:", test_path)
print("sample_sub_path:", sample_sub_path)




## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, activations, initializers, backend

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

tf.config.run_functions_eagerly(False)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as _:
    pass

print("TF version:", tf.__version__)




## === cell 2
train = pd.read_csv(
    train_path,
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
        "pressure": "float32",
    },
)
test = pd.read_csv(
    test_path,
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
    },
)
sample_sub = pd.read_csv(sample_sub_path)

print(train.shape, test.shape, sample_sub.shape)
print(train.columns)

SEQ_LEN = 80

train = train.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
test = test.sort_values(["breath_id", "time_step"]).reset_index(drop=True)




## === cell 3
RC_COMBOS = sorted(train[["R", "C"]].drop_duplicates().apply(tuple, axis=1).tolist())
rc_to_idx = {rc: i for i, rc in enumerate(RC_COMBOS)}
RC_DIM = 15  # as required by the architecture

_lut_keys = np.array([r * 1000 + c for (r, c) in RC_COMBOS], dtype=np.int32)
_lut_vals = np.arange(len(RC_COMBOS), dtype=np.int32)
_sorter = np.argsort(_lut_keys)
_lut_keys_sorted = _lut_keys[_sorter]
_lut_vals_sorted = _lut_vals[_sorter]


def _breath_first_indices(breath_id_arr: np.ndarray) -> np.ndarray:
    change = np.empty_like(breath_id_arr, dtype=np.bool_)
    change[0] = True
    change[1:] = breath_id_arr[1:] != breath_id_arr[:-1]
    return np.flatnonzero(change)


def build_rc_onehot(df: pd.DataFrame) -> np.ndarray:
    breath_id = df["breath_id"].to_numpy(np.int32, copy=False)
    first_idx = _breath_first_indices(breath_id)

    R_first = (
        df["R"].to_numpy(np.int16, copy=False)[first_idx].astype(np.int32, copy=False)
    )
    C_first = (
        df["C"].to_numpy(np.int16, copy=False)[first_idx].astype(np.int32, copy=False)
    )

    key = R_first * 1000 + C_first
    pos = np.searchsorted(_lut_keys_sorted, key)
    idx = _lut_vals_sorted[pos].astype(np.int32, copy=False)

    onehot = np.zeros((idx.shape[0], RC_DIM), dtype=np.int32)
    onehot[np.arange(idx.shape[0]), idx] = 1
    return onehot


def build_other_features(df: pd.DataFrame) -> np.ndarray:
    feats = df[["time_step", "u_in", "u_out"]].to_numpy(dtype=np.float32, copy=False)
    n_rows = feats.shape[0]
    assert n_rows % SEQ_LEN == 0, "Rows must be multiple of 80 for breath sequences."
    return feats.reshape(-1, SEQ_LEN, feats.shape[-1])


def build_targets(df: pd.DataFrame) -> np.ndarray:
    y = df["pressure"].to_numpy(dtype=np.float32, copy=False)
    assert y.shape[0] % SEQ_LEN == 0
    return y.reshape(-1, SEQ_LEN, 1)


def build_uout_mask(df: pd.DataFrame) -> np.ndarray:
    u_out = df["u_out"].to_numpy(dtype=np.float32, copy=False).reshape(-1, SEQ_LEN)
    return (u_out == 0).astype(np.float32, copy=False).reshape(-1, SEQ_LEN, 1)


train_x_rc = build_rc_onehot(train)
train_x_other = build_other_features(train)
train_y = build_targets(train)
train_w = build_uout_mask(train)

test_x_rc = build_rc_onehot(test)
test_x_other = build_other_features(test)

print("train_x_rc:", train_x_rc.shape, train_x_rc.dtype)
print("train_x_other:", train_x_other.shape, train_x_other.dtype)
print("train_y:", train_y.shape, train_y.dtype)
print("train_w:", train_w.shape, train_w.dtype)
print("test_x_rc:", test_x_rc.shape, test_x_rc.dtype)
print("test_x_other:", test_x_other.shape, test_x_other.dtype)




## === cell 4
other_flat = train_x_other.reshape(-1, train_x_other.shape[-1])
means = other_flat.mean(axis=0)
stds = other_flat.std(axis=0) + 1e-6

scale_idx = np.array([0, 1], dtype=np.int32)


def apply_scaling_inplace(x: np.ndarray):
    x[..., scale_idx] -= means[scale_idx]
    x[..., scale_idx] /= stds[scale_idx]
    return x


train_x_other = apply_scaling_inplace(train_x_other.astype(np.float32, copy=False))
test_x_other = apply_scaling_inplace(test_x_other.astype(np.float32, copy=False))




## === cell 5
class ExpandTileLayer(layers.Layer):
    def __init__(self):
        super(ExpandTileLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        return backend.tile(backend.expand_dims(inputs, axis=-2), (1, 80, 1))




## === cell 6
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Using TPU")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print("Using default strategy (CPU/GPU). Reason TPU not used:", repr(e))




## === cell 7
with strategy.scope():
    rc_input = keras.Input(shape=(15,), dtype="int32", name="rc_input_layer")
    other_x_input = keras.Input(
        shape=(80, train_x_other.shape[-1]), dtype="float32", name="other_x_input"
    )

    rc_embedding_layer = layers.Dense(
        units=10, use_bias=False, activation=activations.selu, name="rc_embed_layer"
    )
    rc_embedding = rc_embedding_layer(rc_input)
    rc_embedding = ExpandTileLayer()(rc_embedding)

    x_input = layers.Concatenate(axis=-1)([other_x_input, rc_embedding])

    conv1d_1 = layers.Conv1D(filters=256, kernel_size=5, padding="same")(x_input)
    conv1d_1 = layers.Activation(activations.relu)(conv1d_1)
    conv1d_1 = layers.BatchNormalization()(conv1d_1)

    conv1d_2 = layers.Conv1D(filters=192, kernel_size=5, padding="same")(conv1d_1)
    conv1d_output = layers.Activation(activations.relu)(conv1d_2)
    conv1d_output = layers.BatchNormalization()(conv1d_output)

    the_feature = layers.Concatenate()([conv1d_output, x_input])

    ox_1 = layers.Bidirectional(
        layers.LSTM(
            units=672,
            return_sequences=True,
            kernel_initializer=initializers.GlorotUniform(),
        ),
        merge_mode="concat",
    )(the_feature)

    ox_2 = layers.Bidirectional(
        layers.LSTM(
            units=512,
            return_sequences=True,
            kernel_initializer=initializers.GlorotUniform(),
        ),
        merge_mode="concat",
    )(ox_1)

    ox = layers.Bidirectional(
        layers.LSTM(
            units=384,
            return_sequences=True,
            kernel_initializer=initializers.GlorotUniform(),
        ),
        merge_mode="concat",
    )(ox_2)

    ox = layers.Bidirectional(layers.GRU(units=192, return_sequences=True))(ox)

    output = layers.Concatenate(axis=-1)([ox, conv1d_output])
    output = layers.Dense(
        units=128,
        activation=activations.selu,
        kernel_initializer=initializers.GlorotUniform(),
    )(output)
    output = layers.Dense(units=1, kernel_initializer=initializers.GlorotUniform())(
        output
    )

    my_model = models.Model(inputs=[rc_input, other_x_input], outputs=[output])

    my_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mae")

my_model.summary()




## === cell 8
n_breaths = train_x_other.shape[0]
idx = np.arange(n_breaths)
rng = np.random.default_rng(SEED)
rng.shuffle(idx)

val_size = int(0.05 * n_breaths)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

x_tr_rc = train_x_rc[tr_idx]
x_tr_other = train_x_other[tr_idx]
y_tr = train_y[tr_idx]
w_tr = train_w[tr_idx]

x_va_rc = train_x_rc[val_idx]
x_va_other = train_x_other[val_idx]
y_va = train_y[val_idx]
w_va = train_w[val_idx]

BATCH_SIZE = 256

opts = tf.data.Options()
opts.deterministic = True

train_ds = tf.data.Dataset.from_tensor_slices(((x_tr_rc, x_tr_other), y_tr, w_tr))
train_ds = train_ds.with_options(opts).cache()
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(((x_va_rc, x_va_other), y_va, w_va))
val_ds = val_ds.with_options(opts).cache()
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

history = my_model.fit(
    x=train_ds,
    validation_data=val_ds,
    epochs=3,
    verbose=2,
)




## === cell 9
PRED_BS = 512
opts = tf.data.Options()
opts.deterministic = True

test_ds = tf.data.Dataset.from_tensor_slices((test_x_rc, test_x_other))
test_ds = (
    test_ds.with_options(opts)
    .batch(PRED_BS, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

pred = my_model.predict(test_ds, verbose=1).reshape(-1)

pressure_step = 0.07030248641967773
p_min = -1.7551400036622216

pred = np.round((pred - p_min) / pressure_step) * pressure_step + p_min
pred = pred.astype(np.float32)




## === cell 10
sub = pd.DataFrame(
    {"id": test["id"].values.astype(np.int32, copy=False), "pressure": pred}
)
sub = sub.sort_values("id").reset_index(drop=True)

assert sub.shape[0] == test.shape[0]
assert sub["id"].is_unique
assert sub["id"].min() == test["id"].min() and sub["id"].max() == test["id"].max()

out_path = "./submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sub))
print(sub.head())
