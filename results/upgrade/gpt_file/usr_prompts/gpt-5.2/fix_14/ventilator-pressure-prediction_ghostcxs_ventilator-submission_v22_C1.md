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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, activations, initializers, backend

tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

from sklearn.preprocessing import RobustScaler

BASE = "/kaggle/input/ventilator-pressure-prediction"
if not os.path.exists(BASE):
    BASE = "/kaggle/input"

test_path = os.path.join(BASE, "test.csv")
train_path = os.path.join(BASE, "train.csv")
sample_path = os.path.join(BASE, "sample_submission.csv")

dtypes_train = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
dtypes_test = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

test_ori = pd.read_csv(test_path, dtype=dtypes_test, engine="c", low_memory=False)

BREATH_LEN = 80


def build_features_and_target(
    df: pd.DataFrame, has_target: bool, breath_len: int = BREATH_LEN
):
    breath_id = df["breath_id"].to_numpy(copy=False)
    time_step = df["time_step"].to_numpy(copy=False)
    u_in = df["u_in"].to_numpy(copy=False)
    u_out = df["u_out"].to_numpy(copy=False)
    R = df["R"].to_numpy(copy=False)
    C = df["C"].to_numpy(copy=False)

    n = breath_id.shape[0]
    n_breaths = n // breath_len

    ts = time_step.reshape(n_breaths, breath_len)
    ui = u_in.reshape(n_breaths, breath_len)

    area = np.cumsum(ts * ui, axis=1, dtype=np.float32).reshape(n)
    u_in_cumsum = np.cumsum(ui, axis=1, dtype=np.float32).reshape(n)

    u_in_lag_m = np.zeros((n_breaths, breath_len, 4), dtype=np.float32)
    u_in_lag_back_m = np.zeros((n_breaths, breath_len, 4), dtype=np.float32)

    u_in_lag_m[:, 1:, 0] = ui[:, :-1]
    u_in_lag_m[:, 2:, 1] = ui[:, :-2]
    u_in_lag_m[:, 3:, 2] = ui[:, :-3]
    u_in_lag_m[:, 4:, 3] = ui[:, :-4]

    u_in_lag_back_m[:, :-1, 0] = ui[:, 1:]
    u_in_lag_back_m[:, :-2, 1] = ui[:, 2:]
    u_in_lag_back_m[:, :-3, 2] = ui[:, 3:]
    u_in_lag_back_m[:, :-4, 3] = ui[:, 4:]

    u_in_lag1 = u_in_lag_m[:, :, 0].reshape(n)
    u_in_lag2 = u_in_lag_m[:, :, 1].reshape(n)
    u_in_lag3 = u_in_lag_m[:, :, 2].reshape(n)
    u_in_lag4 = u_in_lag_m[:, :, 3].reshape(n)

    u_in_lag_back1 = u_in_lag_back_m[:, :, 0].reshape(n)
    u_in_lag_back2 = u_in_lag_back_m[:, :, 1].reshape(n)
    u_in_lag_back3 = u_in_lag_back_m[:, :, 2].reshape(n)
    u_in_lag_back4 = u_in_lag_back_m[:, :, 3].reshape(n)

    u_in_max = np.max(ui, axis=1).astype(np.float32, copy=False)
    breath_id__u_in__max = np.repeat(u_in_max, breath_len).astype(
        np.float32, copy=False
    )

    u_in_flat = ui.reshape(n)
    u_in_diff1 = (u_in_flat - u_in_lag1).astype(np.float32, copy=False)
    u_in_diff2 = (u_in_flat - u_in_lag2).astype(np.float32, copy=False)
    u_in_diff3 = (u_in_flat - u_in_lag3).astype(np.float32, copy=False)
    u_in_diff4 = (u_in_flat - u_in_lag4).astype(np.float32, copy=False)

    u_out_f = u_out.astype(np.float32, copy=False)
    cross = (u_in_flat * u_out_f).astype(np.float32, copy=False)
    cross2 = (time_step * u_out_f).astype(np.float32, copy=False)

    R_5 = (R == 5).astype(np.uint8, copy=False)
    R_20 = (R == 20).astype(np.uint8, copy=False)
    R_50 = (R == 50).astype(np.uint8, copy=False)

    C_10 = (C == 10).astype(np.uint8, copy=False)
    C_20 = (C == 20).astype(np.uint8, copy=False)
    C_50 = (C == 50).astype(np.uint8, copy=False)

    RC_5_10 = ((R == 5) & (C == 10)).astype(np.uint8, copy=False)
    RC_5_20 = ((R == 5) & (C == 20)).astype(np.uint8, copy=False)
    RC_5_50 = ((R == 5) & (C == 50)).astype(np.uint8, copy=False)
    RC_20_10 = ((R == 20) & (C == 10)).astype(np.uint8, copy=False)
    RC_20_20 = ((R == 20) & (C == 20)).astype(np.uint8, copy=False)
    RC_20_50 = ((R == 20) & (C == 50)).astype(np.uint8, copy=False)
    RC_50_10 = ((R == 50) & (C == 10)).astype(np.uint8, copy=False)
    RC_50_20 = ((R == 50) & (C == 20)).astype(np.uint8, copy=False)
    RC_50_50 = ((R == 50) & (C == 50)).astype(np.uint8, copy=False)

    other = np.empty((n, 20), dtype=np.float32)
    other[:, 0] = time_step
    other[:, 1] = u_in
    other[:, 2] = u_out_f
    other[:, 3] = area
    other[:, 4] = u_in_cumsum
    other[:, 5] = u_in_lag1
    other[:, 6] = u_in_lag_back1
    other[:, 7] = u_in_lag2
    other[:, 8] = u_in_lag_back2
    other[:, 9] = u_in_lag3
    other[:, 10] = u_in_lag_back3
    other[:, 11] = u_in_lag4
    other[:, 12] = u_in_lag_back4
    other[:, 13] = breath_id__u_in__max
    other[:, 14] = u_in_diff1
    other[:, 15] = u_in_diff2
    other[:, 16] = u_in_diff3
    other[:, 17] = u_in_diff4
    other[:, 18] = cross
    other[:, 19] = cross2

    rc = np.empty((n, 15), dtype=np.float32)
    rc[:, 0] = R_5
    rc[:, 1] = R_20
    rc[:, 2] = R_50
    rc[:, 3] = C_10
    rc[:, 4] = C_20
    rc[:, 5] = C_50
    rc[:, 6] = RC_5_10
    rc[:, 7] = RC_5_20
    rc[:, 8] = RC_5_50
    rc[:, 9] = RC_20_10
    rc[:, 10] = RC_20_20
    rc[:, 11] = RC_20_50
    rc[:, 12] = RC_50_10
    rc[:, 13] = RC_50_20
    rc[:, 14] = RC_50_50

    y = None
    if has_target:
        y = df["pressure"].to_numpy(copy=False).reshape(n_breaths, breath_len, 1)

    other_3d = other.reshape(n_breaths, breath_len, other.shape[-1]).astype(
        np.float32, copy=False
    )
    rc_2d = rc.reshape(n_breaths, breath_len, rc.shape[-1])[:, 0, :].astype(
        np.float32, copy=False
    )
    return other_3d, rc_2d, y


test_x_other, test_x_rc, _ = build_features_and_target(test_ori, has_target=False)


def _quantile_linear_1d_exact_via_partition(x: np.ndarray, q: float) -> float:
    n = x.size
    if n == 0:
        return np.nan
    if n == 1:
        return float(x[0])
    pos = (n - 1) * q
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(np.partition(x, lo)[lo])
    w = pos - lo
    x_lo = float(np.partition(x, lo)[lo])
    x_hi = float(np.partition(x, hi)[hi])
    return x_lo + (x_hi - x_lo) * w


def _count_rows_fast(csv_path: str) -> int:
    with open(csv_path, "rb") as f:
        n = 0
        buf = f.read(1024 * 1024)
        while buf:
            n += buf.count(b"\n")
            buf = f.read(1024 * 1024)
    return max(0, n - 1)  # subtract header


def fit_robust_scaler_exact_fast(train_csv_path: str, chunk_breaths: int = 4096):
    usecols = ["breath_id", "R", "C", "time_step", "u_in", "u_out"]
    chunksize = chunk_breaths * BREATH_LEN

    nrows = _count_rows_fast(train_csv_path)
    nrows = (nrows // BREATH_LEN) * BREATH_LEN  # ensure full breaths only
    if nrows <= 0:
        raise ValueError("train.csv appears empty or invalid.")

    n_features = 20
    buffer = np.empty((nrows, n_features), dtype=np.float32)
    write_pos = 0

    carry = None
    reader = pd.read_csv(
        train_csv_path,
        dtype=dtypes_train,
        engine="c",
        low_memory=False,
        usecols=usecols,
        chunksize=chunksize,
    )

    for chunk in reader:
        if carry is not None and len(carry) > 0:
            chunk = pd.concat([carry, chunk], axis=0, ignore_index=True)
            carry = None

        rem = (len(chunk) // BREATH_LEN) * BREATH_LEN
        if rem != len(chunk):
            carry = chunk.iloc[rem:].copy()
            chunk = chunk.iloc[:rem]

        if len(chunk) == 0:
            continue

        other_3d, _, _ = build_features_and_target(chunk, has_target=False)
        X = other_3d.reshape(-1, other_3d.shape[-1])  # float32

        end = write_pos + X.shape[0]
        if end > nrows:
            X = X[: (nrows - write_pos)]
            end = nrows

        buffer[write_pos:end, :] = X
        write_pos = end
        if write_pos >= nrows:
            break

        del other_3d, X, chunk

    if write_pos != nrows:
        buffer = buffer[:write_pos, :]
        nrows = write_pos

    q25 = np.empty(n_features, dtype=np.float64)
    q50 = np.empty(n_features, dtype=np.float64)
    q75 = np.empty(n_features, dtype=np.float64)

    for j in range(n_features):
        col = buffer[:, j]
        col_copy = np.array(col, copy=True)
        q25[j] = _quantile_linear_1d_exact_via_partition(col_copy, 0.25)
        col_copy = np.array(col, copy=True)
        q50[j] = _quantile_linear_1d_exact_via_partition(col_copy, 0.50)
        col_copy = np.array(col, copy=True)
        q75[j] = _quantile_linear_1d_exact_via_partition(col_copy, 0.75)

    scale_ = q75 - q25
    scale_[scale_ == 0.0] = 1.0

    scaler = RobustScaler()
    scaler.center_ = q50
    scaler.scale_ = scale_
    scaler.n_features_in_ = n_features
    scaler.feature_names_in_ = None
    return scaler


SCALER_CACHE = "/kaggle/working/robust_scaler_exact_v1.npz"

if os.path.exists(SCALER_CACHE):
    data = np.load(SCALER_CACHE)
    scale = RobustScaler()
    scale.center_ = data["center_"].astype(np.float64, copy=False)
    scale.scale_ = data["scale_"].astype(np.float64, copy=False)
    scale.n_features_in_ = int(data["n_features_in_"])
    scale.feature_names_in_ = None
else:
    scale = fit_robust_scaler_exact_fast(train_path)
    np.savez(
        SCALER_CACHE,
        center_=scale.center_.astype(np.float64, copy=False),
        scale_=scale.scale_.astype(np.float64, copy=False),
        n_features_in_=np.int64(scale.n_features_in_),
    )

n_breaths_te, bl, nfeat = test_x_other.shape
test_x_other_2d = test_x_other.reshape(-1, nfeat)
test_x_other_2d = scale.transform(test_x_other_2d).astype("float32", copy=False)
test_x_other = test_x_other_2d.reshape(n_breaths_te, bl, nfeat)

print("test_x_other :", test_x_other.shape)
print("test_x_rc    :", test_x_rc.shape)




## === cell 1
class ExpandTileLayer(layers.Layer):
    def __init__(self):
        super(ExpandTileLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        return backend.tile(backend.expand_dims(inputs, axis=-2), (1, 80, 1))




## === cell 2
rc_input = keras.Input(shape=(15,), dtype="float32", name="rc_input_layer")
other_x_input = keras.Input(
    shape=(80, test_x_other.shape[-1]), dtype="float32", name="other_x_input"
)

rc_embedding_layer = layers.Dense(units=16, use_bias=False, name="rc_embed_layer")
rc_embedding = rc_embedding_layer(rc_input)
rc_embedding = ExpandTileLayer()(rc_embedding)

x_input = layers.Concatenate(axis=-1)([other_x_input, rc_embedding])

conv1d_1 = layers.Conv1D(filters=256, kernel_size=5, padding="same", use_bias=False)(
    x_input
)
conv1d_1 = layers.Activation(activations.selu)(conv1d_1)

conv1d_2 = layers.Conv1D(filters=128, use_bias=False, kernel_size=5, padding="same")(
    conv1d_1
)
conv1d_output = layers.Activation(activations.selu)(conv1d_2)

ox = layers.Bidirectional(
    layers.LSTM(
        units=512,
        return_sequences=True,
        kernel_initializer=initializers.GlorotUniform(),
    ),
    merge_mode="concat",
)(conv1d_output)

ox = layers.Bidirectional(
    layers.LSTM(
        units=384,
        return_sequences=True,
        kernel_initializer=initializers.GlorotUniform(),
    ),
    merge_mode="concat",
)(ox)

ox = layers.Bidirectional(
    layers.LSTM(
        units=256,
        return_sequences=True,
        kernel_initializer=initializers.GlorotUniform(),
    ),
    merge_mode="concat",
)(ox)

ox = layers.Dense(units=128, kernel_initializer=initializers.GlorotUniform())(ox)
lstm_output = layers.ELU()(ox)

output = layers.Concatenate(axis=-1)([lstm_output, conv1d_output])
output = layers.Dense(
    units=256,
    activation=activations.selu,
    kernel_initializer=initializers.GlorotUniform(),
)(output)
output = layers.Dense(units=1, kernel_initializer=initializers.GlorotUniform())(output)

my_model = models.Model(inputs=[rc_input, other_x_input], outputs=[output])
my_model.compile(optimizer=keras.optimizers.Adam(), loss="mae")
my_model.summary()



## === cell 3
pre_pressure = []

weights_dir = "/kaggle/input/model-weights"
if not os.path.exists(weights_dir):
    weights_dir = "../input/model-weights"


def predict_on_test(model, x_rc, x_other, batch_size: int):
    pre_y_ = model.predict([x_rc, x_other], batch_size=batch_size, verbose=0)
    return np.asarray(pre_y_, dtype=np.float32).reshape(-1)


found_any_weights = False
for fold in range(7):
    w_path = f"{weights_dir}/model_conv_stack_lstm_res_{fold}.h5"
    if os.path.exists(w_path):
        found_any_weights = True
        my_model.load_weights(w_path)
        pre_pressure.append(
            predict_on_test(my_model, test_x_rc, test_x_other, batch_size=2048)
        )
    else:
        print(f"WARNING: weights not found: {w_path}")

if not found_any_weights:
    print("No external model weights found; training model from scratch as fallback.")
    train_ori = pd.read_csv(
        train_path, dtype=dtypes_train, engine="c", low_memory=False
    )
    train_x_other, train_x_rc, y_seq = build_features_and_target(
        train_ori, has_target=True
    )
    n_breaths_tr, bl, nfeat = train_x_other.shape
    train_x_other_2d = train_x_other.reshape(-1, nfeat)
    train_x_other_2d = scale.transform(train_x_other_2d).astype("float32", copy=False)
    train_x_other = train_x_other_2d.reshape(n_breaths_tr, bl, nfeat)

    n_breaths = train_x_other.shape[0]
    idx = np.arange(n_breaths)
    split = int(n_breaths * 0.95)
    tr_idx, va_idx = idx[:split], idx[split:]

    my_model.fit(
        x=[train_x_rc[tr_idx], train_x_other[tr_idx]],
        y=y_seq[tr_idx],
        validation_data=([train_x_rc[va_idx], train_x_other[va_idx]], y_seq[va_idx]),
        epochs=2,
        batch_size=128,
        verbose=2,
        shuffle=True,
    )

    pre_pressure = [predict_on_test(my_model, test_x_rc, test_x_other, batch_size=2048)]



## === cell 4
pressure_step = 0.07030248641967773
p_min = -1.7551400036622216
p_max = 64.82099173863328

sub_medclip = np.median(np.vstack(pre_pressure), axis=0)

sub_medclip = np.clip(sub_medclip, p_min, p_max)
sub_medclip = np.round((sub_medclip - p_min) / pressure_step) * pressure_step + p_min

sub = pd.DataFrame(
    {
        "id": test_ori["id"].astype("int32").to_numpy(copy=False),
        "pressure": sub_medclip.astype("float32", copy=False),
    }
)

sub.to_csv("./submission.csv", index=False)
print(sub.head())
print("Wrote submission to ./submission.csv")
