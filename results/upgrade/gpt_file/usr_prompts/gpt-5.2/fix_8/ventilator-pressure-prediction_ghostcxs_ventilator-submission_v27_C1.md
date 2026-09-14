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

INPUT_DIR = "/kaggle/input"
WORKING_DIR = "/kaggle/working"

for dirname in [INPUT_DIR]:
    try:
        files = sorted(os.listdir(dirname))[:50]
        for f in files:
            print(os.path.join(dirname, f))
    except Exception:
        pass



## === cell 1
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import (
    layers,
    models,
    losses,
    optimizers,
    activations,
    initializers,
    backend,
)

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 2
DATA_ROOT = "/kaggle/input/ventilator-pressure-prediction"
if not os.path.exists(os.path.join(DATA_ROOT, "train.csv")):
    DATA_ROOT = "/kaggle/input"

train_path = os.path.join(DATA_ROOT, "train.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")
sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")

SEQ_LEN = 80

train_usecols = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
test_usecols = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "id"]
sample_usecols = ["id", "pressure"]

train_dtypes = {
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
test_dtypes = {
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "id": "int32",
}
sample_dtypes = {"id": "int32", "pressure": "float32"}

test_df = pd.read_csv(test_path, usecols=test_usecols, dtype=test_dtypes)
sample_sub = pd.read_csv(sample_path, usecols=sample_usecols, dtype=sample_dtypes)

print("test:", test_df.shape, "sample:", sample_sub.shape)
print("test cols:", test_df.columns.tolist())


def make_rc_input(df: pd.DataFrame):
    r_vals = np.array([5, 20, 50], dtype=np.int16)
    c_vals = np.array([10, 20, 50], dtype=np.int16)

    n_rows = df.shape[0]
    n_breaths = n_rows // SEQ_LEN

    R_col = df["R"].to_numpy(copy=False).astype(np.int16, copy=False)
    C_col = df["C"].to_numpy(copy=False).astype(np.int16, copy=False)

    R = R_col.reshape(n_breaths, SEQ_LEN)[:, 0]
    C = C_col.reshape(n_breaths, SEQ_LEN)[:, 0]

    r_oh = (R[:, None] == r_vals[None, :]).astype(np.int32, copy=False)  # (n,3)
    c_oh = (C[:, None] == c_vals[None, :]).astype(np.int32, copy=False)  # (n,3)

    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False).reshape(n_breaths, SEQ_LEN)
    u_out = (
        df["u_out"].to_numpy(dtype=np.float32, copy=False).reshape(n_breaths, SEQ_LEN)
    )

    u_in_sum = u_in.sum(axis=1)
    u_in_max = u_in.max(axis=1)
    u_in_mean = u_in.mean(axis=1)
    u_out_sum = u_out.sum(axis=1)

    feats = np.stack([u_in_sum, u_in_max, u_in_mean, u_out_sum], axis=1)
    feats = np.clip(np.round(feats), -(2**31), 2**31 - 1).astype(np.int32, copy=False)

    rc = np.concatenate([r_oh, c_oh, feats], axis=1)  # (n, 10)
    if rc.shape[1] < 15:
        rc = np.pad(
            rc, ((0, 0), (0, 15 - rc.shape[1])), mode="constant", constant_values=0
        )
    elif rc.shape[1] > 15:
        rc = rc[:, :15]
    return rc.astype(np.int32, copy=False)


def _robust_transform(X: np.ndarray, med: np.ndarray, scale: np.ndarray):
    return ((X - med) / scale).astype(np.float32, copy=False)


def make_other_x(df: pd.DataFrame, scaler: dict = None, fit: bool = False):
    n_rows = df.shape[0]
    n_breaths = n_rows // SEQ_LEN

    time_step = (
        df["time_step"]
        .to_numpy(dtype=np.float32, copy=False)
        .reshape(n_breaths, SEQ_LEN)
    )
    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False).reshape(n_breaths, SEQ_LEN)
    u_out = (
        df["u_out"].to_numpy(dtype=np.float32, copy=False).reshape(n_breaths, SEQ_LEN)
    )

    u_in_cum = np.cumsum(u_in, axis=1, dtype=np.float32)

    u_in_lag1 = np.zeros_like(u_in, dtype=np.float32)
    u_in_lag1[:, 1:] = u_in[:, :-1]

    u_out_lag1 = np.zeros_like(u_out, dtype=np.float32)
    u_out_lag1[:, 1:] = u_out[:, :-1]

    dt = np.zeros_like(time_step, dtype=np.float32)
    dt[:, 1:] = time_step[:, 1:] - time_step[:, :-1]

    X = np.stack(
        [time_step, u_in, u_out, u_in_cum, u_in_lag1, u_out_lag1, dt], axis=-1
    ).astype(
        np.float32, copy=False
    )  # (n, 80, 7)

    if scaler is None:
        scaler = {}
    if fit:
        raise RuntimeError(
            "Scaler fitting is handled by the optimized in-memory fitter below."
        )
    X_out = _robust_transform(X, scaler["center_"], scaler["scale_"])
    return X_out, scaler


test_x_rc = make_rc_input(test_df)


def _quantiles_from_hist(
    counts: np.ndarray, bin_edges: np.ndarray, probs=(0.25, 0.5, 0.75)
) -> np.ndarray:
    cdf = np.cumsum(counts, dtype=np.int64)
    total = int(cdf[-1])
    out = np.empty((len(probs),), dtype=np.float32)
    for i, p in enumerate(probs):
        target = p * (total - 1)
        idx = int(np.searchsorted(cdf, target, side="right"))
        if idx <= 0:
            out[i] = float(bin_edges[0])
        else:
            prev = int(cdf[idx - 1])
            curr = int(cdf[idx])
            if curr == prev:
                out[i] = float(bin_edges[idx])
            else:
                frac = (target - prev) / (curr - prev)
                out[i] = float(
                    bin_edges[idx] + frac * (bin_edges[idx + 1] - bin_edges[idx])
                )
    return out


def _fit_robust_scaler_from_feature_tensor(X: np.ndarray) -> dict:
    Xf = X.reshape(-1, X.shape[-1]).astype(np.float32, copy=False)  # view when possible
    q = np.quantile(Xf, [0.25, 0.5, 0.75], axis=0, method="linear").astype(np.float32)
    q1, med, q3 = q[0], q[1], q[2]
    scale = (q3 - q1).astype(np.float32, copy=False)
    scale = np.where(scale == 0.0, 1.0, scale).astype(np.float32, copy=False)
    return {"center_": med.astype(np.float32, copy=False), "scale_": scale}


test_x_other_raw, _ = make_other_x(
    test_df,
    scaler={"center_": np.zeros((7,), np.float32), "scale_": np.ones((7,), np.float32)},
    fit=False,
)
scaler = _fit_robust_scaler_from_feature_tensor(test_x_other_raw)
test_x_other = _robust_transform(test_x_other_raw, scaler["center_"], scaler["scale_"])
del test_x_other_raw

test_x_rc = np.ascontiguousarray(test_x_rc, dtype=np.int32)
test_x_other = np.ascontiguousarray(test_x_other, dtype=np.float32)

print("test_x_rc:", test_x_rc.shape, test_x_rc.dtype)
print("test_x_other:", test_x_other.shape, test_x_other.dtype)




## === cell 3
class ExpandTileLayer(layers.Layer):
    def __init__(self):
        super(ExpandTileLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        return backend.tile(backend.expand_dims(inputs, axis=-2), (1, 80, 1))




## === cell 4
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Using TPU strategy")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print("Using default strategy (no TPU). Reason:", repr(e))



## === cell 5
with strategy.scope():
    rc_input = keras.Input(shape=(15,), dtype="int32", name="rc_input_layer")
    other_x_input = keras.Input(
        shape=(80, test_x_other.shape[-1]), dtype="float32", name="other_x_input"
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

my_model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3), loss=losses.MeanAbsoluteError()
)
my_model.summary()




## === cell 6
def load_train_arrays(train_csv: str, scaler: dict):
    train_df_small = pd.read_csv(train_csv, usecols=train_usecols, dtype=train_dtypes)
    n_rows = train_df_small.shape[0]
    n_breaths = n_rows // SEQ_LEN

    x_rc = make_rc_input(train_df_small)

    x_other_raw, _ = make_other_x(
        train_df_small,
        scaler={
            "center_": np.zeros((7,), np.float32),
            "scale_": np.ones((7,), np.float32),
        },
        fit=False,
    )
    x_other = _robust_transform(x_other_raw, scaler["center_"], scaler["scale_"])
    x_rc = np.ascontiguousarray(x_rc, dtype=np.int32)
    x_other = np.ascontiguousarray(x_other, dtype=np.float32)

    y = (
        train_df_small["pressure"]
        .to_numpy(dtype=np.float32, copy=False)
        .reshape(n_breaths, SEQ_LEN, 1)
    )

    u_out = (
        train_df_small["u_out"]
        .to_numpy(dtype=np.int8, copy=False)
        .reshape(n_breaths, SEQ_LEN)
    )
    sw = (u_out == 0).astype(np.float32).reshape(n_breaths, SEQ_LEN)

    return x_rc, x_other, y, sw


weights_loaded = False
pre_pressure = []

weights_dir = "/kaggle/input/model-weights"
BATCH_SIZE_PRED = 8192

for fold in range(1):
    w_path = os.path.join(weights_dir, f"model_final_{fold}.h5")
    if os.path.exists(w_path):
        my_model.load_weights(w_path)
        weights_loaded = True
        print("Loaded weights:", w_path)

        pre_y_ = my_model.predict(
            [test_x_rc, test_x_other], batch_size=BATCH_SIZE_PRED, verbose=1
        )
        pre_y = np.asarray(pre_y_, dtype=np.float32).reshape(-1)
        pre_pressure.append(pre_y)
    else:
        print("Weights not found:", w_path)

if not weights_loaded:
    print("No external weights found; running fallback training (single pass).")
    train_x_rc, train_x_other, y_train, sw_train = load_train_arrays(
        train_path, scaler=scaler
    )

    n_breaths = train_x_rc.shape[0]
    split = int(n_breaths * 0.9)
    xtr = [train_x_rc[:split], train_x_other[:split]]
    ytr = y_train[:split]
    swtr = sw_train[:split]

    xva = [train_x_rc[split:], train_x_other[split:]]
    yva = y_train[split:]
    swva = sw_train[split:]

    my_model.fit(
        xtr,
        ytr,
        sample_weight=swtr,
        validation_data=(xva, yva, swva),
        epochs=1,
        batch_size=256,
        verbose=2,
        shuffle=False,
    )

    pre_y_ = my_model.predict(
        [test_x_rc, test_x_other], batch_size=BATCH_SIZE_PRED, verbose=1
    )
    pre_y = np.asarray(pre_y_, dtype=np.float32).reshape(-1)
    pre_pressure.append(pre_y)

print(
    "Num prediction arrays:",
    len(pre_pressure),
    "Pred length:",
    pre_pressure[0].shape[0],
)



## === cell 7
pressure_step = 0.07030248641967773
p_min = -1.7551400036622216
p_max = 64.82099173863328

sub_pred = np.mean(np.vstack(pre_pressure), axis=0)

sub_pred = np.round((sub_pred - p_min) / pressure_step) * pressure_step + p_min
sub_pred = np.clip(sub_pred, p_min, p_max).astype(np.float32)

sub = pd.DataFrame(
    {"id": test_df["id"].values.astype(np.int32, copy=False), "pressure": sub_pred}
)

assert sub.shape[0] == sample_sub.shape[0], (sub.shape, sample_sub.shape)
assert list(sub.columns) == ["id", "pressure"]

sub_path = os.path.join(WORKING_DIR, "submission.csv")
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub.head())
print(sub.tail())
