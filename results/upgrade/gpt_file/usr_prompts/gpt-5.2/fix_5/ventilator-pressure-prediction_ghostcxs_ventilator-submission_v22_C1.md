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



## === cell 1
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
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



## === cell 2
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

train_ori = pd.read_csv(train_path, dtype=dtypes_train)
test_ori = pd.read_csv(test_path, dtype=dtypes_test)


def add_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    breath_id = df["breath_id"].to_numpy(copy=False)
    time_step = df["time_step"].to_numpy(dtype=np.float32, copy=False)
    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)
    u_out = df["u_out"].to_numpy(dtype=np.int8, copy=False)
    R = df["R"].to_numpy(dtype=np.int16, copy=False)
    C = df["C"].to_numpy(dtype=np.int16, copy=False)

    first = breath_id[0]
    breath_len = int(np.argmax(breath_id != first))
    if breath_len == 0:
        breath_len = 80
    n_breaths = n // breath_len

    ts = time_step.reshape(n_breaths, breath_len)
    ui = u_in.reshape(n_breaths, breath_len)
    uo = u_out.reshape(n_breaths, breath_len)

    area = (
        np.cumsum(ts * ui, axis=1, dtype=np.float32)
        .reshape(n)
        .astype(np.float32, copy=False)
    )

    u_in_cumsum = (
        np.cumsum(ui, axis=1, dtype=np.float32)
        .reshape(n)
        .astype(np.float32, copy=False)
    )

    def lag(mat, k: int):
        out = np.zeros_like(mat, dtype=np.float32)
        if k > 0:
            out[:, k:] = mat[:, :-k]
        return out

    def lag_back(mat, k: int):
        out = np.zeros_like(mat, dtype=np.float32)
        if k > 0:
            out[:, :-k] = mat[:, k:]
        return out

    u_in_lag1 = lag(ui, 1).reshape(n)
    u_in_lag2 = lag(ui, 2).reshape(n)
    u_in_lag3 = lag(ui, 3).reshape(n)
    u_in_lag4 = lag(ui, 4).reshape(n)

    u_in_lag_back1 = lag_back(ui, 1).reshape(n)
    u_in_lag_back2 = lag_back(ui, 2).reshape(n)
    u_in_lag_back3 = lag_back(ui, 3).reshape(n)
    u_in_lag_back4 = lag_back(ui, 4).reshape(n)

    u_in_max = np.max(ui, axis=1).astype(np.float32, copy=False)
    breath_id__u_in__max = np.repeat(u_in_max, breath_len).astype(
        np.float32, copy=False
    )

    u_in_flat = ui.reshape(n).astype(np.float32, copy=False)
    u_in_diff1 = (u_in_flat - u_in_lag1).astype(np.float32, copy=False)
    u_in_diff2 = (u_in_flat - u_in_lag2).astype(np.float32, copy=False)
    u_in_diff3 = (u_in_flat - u_in_lag3).astype(np.float32, copy=False)
    u_in_diff4 = (u_in_flat - u_in_lag4).astype(np.float32, copy=False)

    cross = (u_in_flat * u_out.astype(np.float32, copy=False)).astype(
        np.float32, copy=False
    )
    cross2 = (time_step * u_out.astype(np.float32, copy=False)).astype(
        np.float32, copy=False
    )

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

    out = df[["id", "breath_id", "time_step", "u_in", "u_out"]].copy()
    out["area"] = area
    out["u_in_cumsum"] = u_in_cumsum

    out["u_in_lag1"] = u_in_lag1
    out["u_in_lag_back1"] = u_in_lag_back1
    out["u_in_lag2"] = u_in_lag2
    out["u_in_lag_back2"] = u_in_lag_back2
    out["u_in_lag3"] = u_in_lag3
    out["u_in_lag_back3"] = u_in_lag_back3
    out["u_in_lag4"] = u_in_lag4
    out["u_in_lag_back4"] = u_in_lag_back4

    out["breath_id__u_in__max"] = breath_id__u_in__max

    out["u_in_diff1"] = u_in_diff1
    out["u_in_diff2"] = u_in_diff2
    out["u_in_diff3"] = u_in_diff3
    out["u_in_diff4"] = u_in_diff4

    out["cross"] = cross
    out["cross2"] = cross2

    out["R_5"] = R_5
    out["R_20"] = R_20
    out["R_50"] = R_50
    out["C_10"] = C_10
    out["C_20"] = C_20
    out["C_50"] = C_50

    out["R__C_5__10"] = RC_5_10
    out["R__C_5__20"] = RC_5_20
    out["R__C_5__50"] = RC_5_50
    out["R__C_20__10"] = RC_20_10
    out["R__C_20__20"] = RC_20_20
    out["R__C_20__50"] = RC_20_50
    out["R__C_50__10"] = RC_50_10
    out["R__C_50__20"] = RC_50_20
    out["R__C_50__50"] = RC_50_50

    if "pressure" in df.columns:
        out["pressure"] = df["pressure"].to_numpy(dtype=np.float32, copy=False)

    return out


train = add_features_fast(train_ori)
y = train["pressure"].to_numpy(dtype="float32", copy=False)
train.drop(["pressure", "id", "breath_id"], axis=1, inplace=True)

test = add_features_fast(test_ori)
test.drop(["id", "breath_id"], axis=1, inplace=True)

train_np = train.to_numpy(copy=False)
test_np = test.to_numpy(copy=False)

train_x_other = train_np[:, :-15]
train_x_rc = train_np[:, -15:]
test_x_other = test_np[:, :-15]
test_x_rc = test_np[:, -15:]

scale = RobustScaler()
scale.fit(train_x_other)
train_x_other = scale.transform(train_x_other).astype("float32", copy=False)
test_x_other = scale.transform(test_x_other).astype("float32", copy=False)

train_x_other = train_x_other.reshape((-1, 80, train_x_other.shape[-1]))
test_x_other = test_x_other.reshape((-1, 80, test_x_other.shape[-1]))

train_x_rc = train_x_rc.reshape((-1, 80, train_x_rc.shape[-1]))[:, 0, :].astype(
    "float32", copy=False
)
test_x_rc = test_x_rc.reshape((-1, 80, test_x_rc.shape[-1]))[:, 0, :].astype(
    "float32", copy=False
)

y_seq = y.reshape((-1, 80, 1)).astype("float32", copy=False)

print("train_x_other:", train_x_other.shape)
print("train_x_rc   :", train_x_rc.shape)
print("y_seq        :", y_seq.shape)
print("test_x_other :", test_x_other.shape)
print("test_x_rc    :", test_x_rc.shape)




## === cell 3
class ExpandTileLayer(layers.Layer):
    def __init__(self):
        super(ExpandTileLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        return backend.tile(backend.expand_dims(inputs, axis=-2), (1, 80, 1))




## === cell 4
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



## === cell 5
pre_pressure = []

weights_dir = "/kaggle/input/model-weights"
if not os.path.exists(weights_dir):
    weights_dir = "../input/model-weights"


def make_test_ds(x_rc, x_other, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((x_rc, x_other))
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_test_ds(test_x_rc, test_x_other, batch_size=4096)

found_any_weights = False

for fold in range(7):
    w_path = f"{weights_dir}/model_conv_stack_lstm_res_{fold}.h5"
    if os.path.exists(w_path):
        found_any_weights = True
        my_model.load_weights(w_path)
        pre_y_ = my_model.predict(test_ds, verbose=0)
        pre_y = np.asarray(pre_y_, dtype=np.float32).reshape(-1)
        pre_pressure.append(pre_y)
    else:
        print(f"WARNING: weights not found: {w_path}")

if not found_any_weights:
    print("No external model weights found; training model from scratch as fallback.")
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

    pre_y_ = my_model.predict(test_ds, verbose=0)
    pre_pressure = [np.asarray(pre_y_, dtype=np.float32).reshape(-1)]



## === cell 6
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
