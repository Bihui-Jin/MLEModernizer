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

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
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


def _assert_breath_grouping_ok(df: pd.DataFrame, seq_len: int, name: str):
    n = len(df)
    assert n % seq_len == 0, f"{name}: rows must be multiple of {seq_len}"
    b = df["breath_id"].to_numpy(np.int32, copy=False).reshape(-1, seq_len)
    assert np.all(
        b == b[:, :1]
    ), f"{name}: breath_id not constant within sequences; sorting would be required"


_assert_breath_grouping_ok(train, SEQ_LEN, "train")
_assert_breath_grouping_ok(test, SEQ_LEN, "test")



## === cell 3
RC_DIM = 15  # as required by the architecture

rc_keys = train["R"].astype(np.int32).to_numpy(copy=False) * 1000 + train["C"].astype(
    np.int32
).to_numpy(copy=False)
uniq_keys = np.unique(rc_keys)
RC_COMBOS = sorted([(int(k // 1000), int(k % 1000)) for k in uniq_keys.tolist()])
rc_to_idx = {r * 1000 + c: i for i, (r, c) in enumerate(RC_COMBOS)}


def build_rc_onehot(df: pd.DataFrame) -> np.ndarray:
    breath_id = df["breath_id"].to_numpy(np.int32, copy=False)
    change = np.empty_like(breath_id, dtype=np.bool_)
    change[0] = True
    change[1:] = breath_id[1:] != breath_id[:-1]
    first_idx = np.flatnonzero(change)

    R_first = (
        df["R"].to_numpy(np.int16, copy=False)[first_idx].astype(np.int32, copy=False)
    )
    C_first = (
        df["C"].to_numpy(np.int16, copy=False)[first_idx].astype(np.int32, copy=False)
    )
    key = (R_first * 1000 + C_first).astype(np.int32, copy=False)

    uniq, inv = np.unique(key, return_inverse=True)
    lut = np.empty(uniq.shape[0], dtype=np.int32)
    for j, k in enumerate(uniq.tolist()):
        lut[j] = rc_to_idx[int(k)]
    idx = lut[inv]

    onehot = np.zeros((idx.shape[0], RC_DIM), dtype=np.int32)
    onehot[np.arange(idx.shape[0]), idx] = 1
    return np.ascontiguousarray(onehot)


def build_other_features(df: pd.DataFrame) -> np.ndarray:
    feats = df[["time_step", "u_in", "u_out"]].to_numpy(dtype=np.float32, copy=False)
    n_rows = feats.shape[0]
    assert n_rows % SEQ_LEN == 0, "Rows must be multiple of 80 for breath sequences."
    x = feats.reshape(-1, SEQ_LEN, feats.shape[-1])
    return np.ascontiguousarray(x)


def build_targets(df: pd.DataFrame) -> np.ndarray:
    y = df["pressure"].to_numpy(dtype=np.float32, copy=False)
    assert y.shape[0] % SEQ_LEN == 0
    y = y.reshape(-1, SEQ_LEN, 1)
    return np.ascontiguousarray(y)


def build_uout_mask(df: pd.DataFrame) -> np.ndarray:
    u_out = df["u_out"].to_numpy(dtype=np.float32, copy=False).reshape(-1, SEQ_LEN)
    w = (u_out == 0).astype(np.float32, copy=False).reshape(-1, SEQ_LEN, 1)
    return np.ascontiguousarray(w)


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
other_arr = train_x_other  # already float32 contiguous
means = other_arr.reshape(-1, other_arr.shape[-1]).mean(axis=0)
stds = other_arr.reshape(-1, other_arr.shape[-1]).std(axis=0) + 1e-6

scale_idx = np.array([0, 1], dtype=np.int32)


def apply_scaling_inplace(x: np.ndarray):
    x[..., scale_idx] -= means[scale_idx]
    x[..., scale_idx] /= stds[scale_idx]
    return x


train_x_other = apply_scaling_inplace(train_x_other.astype(np.float32, copy=False))
test_x_other = apply_scaling_inplace(test_x_other.astype(np.float32, copy=False))

train_x_other = np.ascontiguousarray(train_x_other)
test_x_other = np.ascontiguousarray(test_x_other)




## === cell 5
class ExpandTileLayer(layers.Layer):
    def __init__(self):
        super(ExpandTileLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        return backend.tile(backend.expand_dims(inputs, axis=-2), (1, 80, 1))




## === cell 6
strategy = None
tpu_name = os.environ.get("KAGGLE_TPU_NAME", "").strip()
if tpu_name:
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Using TPU:", tpu_name)
    except Exception as e:
        strategy = tf.distribute.get_strategy()
        print("Using default strategy (CPU/GPU). Reason TPU not used:", repr(e))
else:
    strategy = tf.distribute.get_strategy()
    print("Using default strategy (CPU/GPU). TPU not requested via KAGGLE_TPU_NAME.")



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

BATCH_SIZE = 256
AUTOTUNE = tf.data.AUTOTUNE

tr_x_rc = train_x_rc[tr_idx]
tr_x_other = train_x_other[tr_idx]
tr_y = train_y[tr_idx]
tr_w = train_w[tr_idx]

va_x_rc = train_x_rc[val_idx]
va_x_other = train_x_other[val_idx]
va_y = train_y[val_idx]
va_w = train_w[val_idx]


def make_dataset_from_arrays(x_rc, x_other, y, w, batch_size: int):
    ds = tf.data.Dataset.from_tensor_slices(((x_rc, x_other), y, w))
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


ds_tr = make_dataset_from_arrays(tr_x_rc, tr_x_other, tr_y, tr_w, BATCH_SIZE)
ds_va = make_dataset_from_arrays(va_x_rc, va_x_other, va_y, va_w, BATCH_SIZE)

history = my_model.fit(
    ds_tr,
    validation_data=ds_va,
    epochs=3,
    shuffle=False,
    verbose=2,
)



## === cell 9
PRED_BS = 512

pred = my_model.predict(
    (test_x_rc, test_x_other), batch_size=PRED_BS, verbose=1
).reshape(-1)

pressure_step = 0.07030248641967773
p_min = -1.7551400036622216

pred = np.round((pred - p_min) / pressure_step) * pressure_step + p_min
pred = pred.astype(np.float32)

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
