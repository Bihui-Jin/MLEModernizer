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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.pop("TF_PROTOBUF_USE_CPP_IMPLEMENTATION", None)

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

VER = 81
FIRST_FOLD_ONLY = True
TRAIN_MODEL = False

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error

print("TF version:", tf.__version__)



## === cell 1
import random

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)
random.seed(SEED_GLOBAL)
tf.random.set_seed(SEED_GLOBAL)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 2
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

train_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
test_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

train = pd.read_csv(TRAIN_PATH, dtype=train_dtypes)
test = pd.read_csv(TEST_PATH, dtype=test_dtypes)
submission = pd.read_csv(SAMPLE_PATH)




## === cell 3
def _roll2d(a2d, k):
    out = np.roll(a2d, shift=k, axis=1)
    if k > 0:
        out[:, :k] = 0
    elif k < 0:
        out[:, k:] = 0
    return out


def _cumsum2d(a2d):
    return np.cumsum(a2d, axis=1)


def _ewm_mean_2d(a2d, halflife):
    alpha = 1.0 - np.exp(np.log(0.5) / float(halflife))
    n, t = a2d.shape
    num = np.zeros((n, t), dtype=np.float64)
    den = np.zeros((n, t), dtype=np.float64)
    num[:, 0] = a2d[:, 0]
    den[:, 0] = 1.0
    a = 1.0 - alpha
    for i in range(1, t):
        num[:, i] = a * num[:, i - 1] + a2d[:, i]
        den[:, i] = a * den[:, i - 1] + 1.0
    return (num / den).astype(np.float32)


def _rolling_agg_2d(a2d, window):
    n, t = a2d.shape
    out_sum = np.empty((n, t), dtype=np.float32)
    out_min = np.empty((n, t), dtype=np.float32)
    out_max = np.empty((n, t), dtype=np.float32)
    for i in range(t):
        s = max(0, i - window + 1)
        w = a2d[:, s : i + 1]
        out_sum[:, i] = np.sum(w, axis=1, dtype=np.float64)
        out_min[:, i] = np.min(w, axis=1)
        out_max[:, i] = np.max(w, axis=1)
    out_mean = (out_sum / np.minimum(np.arange(1, t + 1), window)[None, :]).astype(
        np.float32
    )
    return out_sum, out_min, out_max, out_mean


def add_features(df):
    n_rows = len(df)
    n_breaths = n_rows // 80

    u_in = df["u_in"].to_numpy(dtype=np.float32).reshape(n_breaths, 80)
    u_out = df["u_out"].to_numpy(dtype=np.float32).reshape(n_breaths, 80)
    time_step = df["time_step"].to_numpy(dtype=np.float32).reshape(n_breaths, 80)

    cross = (u_in * u_out).reshape(-1)
    cross2 = (time_step * u_out).reshape(-1)
    area = _cumsum2d(time_step * u_in).reshape(-1)
    time_step_cumsum = _cumsum2d(time_step).reshape(-1)
    u_in_cumsum = _cumsum2d(u_in).reshape(-1)

    print("Step-1...Completed")

    u_in_lag1 = _roll2d(u_in, 1).reshape(-1)
    u_out_lag1 = _roll2d(u_out, 1).reshape(-1)
    u_in_lag_back1 = _roll2d(u_in, -1).reshape(-1)
    u_out_lag_back1 = _roll2d(u_out, -1).reshape(-1)

    u_in_lag2 = _roll2d(u_in, 2).reshape(-1)
    u_out_lag2 = _roll2d(u_out, 2).reshape(-1)
    u_in_lag_back2 = _roll2d(u_in, -2).reshape(-1)
    u_out_lag_back2 = _roll2d(u_out, -2).reshape(-1)

    u_in_lag3 = _roll2d(u_in, 3).reshape(-1)
    u_out_lag3 = _roll2d(u_out, 3).reshape(-1)
    u_in_lag_back3 = _roll2d(u_in, -3).reshape(-1)
    u_out_lag_back3 = _roll2d(u_out, -3).reshape(-1)

    u_in_lag4 = _roll2d(u_in, 4).reshape(-1)
    u_out_lag4 = _roll2d(u_out, 4).reshape(-1)
    u_in_lag_back4 = _roll2d(u_in, -4).reshape(-1)
    u_out_lag_back4 = _roll2d(u_out, -4).reshape(-1)

    print("Step-2...Completed")

    u_in_max = np.max(u_in, axis=1).astype(np.float32)
    u_in_mean = np.mean(u_in, axis=1).astype(np.float32)
    breath_id__u_in__max = np.repeat(u_in_max, 80)
    breath_id__u_in__mean = np.repeat(u_in_mean, 80)
    breath_id__u_in__diffmax = breath_id__u_in__max - df["u_in"].to_numpy(
        dtype=np.float32
    )
    breath_id__u_in__diffmean = breath_id__u_in__mean - df["u_in"].to_numpy(
        dtype=np.float32
    )
    print("Step-3...Completed")

    u_in_flat = df["u_in"].to_numpy(dtype=np.float32)
    u_out_flat = df["u_out"].to_numpy(dtype=np.float32)
    u_in_diff1 = u_in_flat - u_in_lag1
    u_out_diff1 = u_out_flat - u_out_lag1
    u_in_diff2 = u_in_flat - u_in_lag2
    u_out_diff2 = u_out_flat - u_out_lag2
    u_in_diff3 = u_in_flat - u_in_lag3
    u_out_diff3 = u_out_flat - u_out_lag3
    u_in_diff4 = u_in_flat - u_in_lag4
    u_out_diff4 = u_out_flat - u_out_lag4
    print("Step-4...Completed")

    count = np.tile(np.arange(1, 81, dtype=np.float32), n_breaths)
    u_in_cummean = u_in_cumsum / count

    breath_id__u_in_lag = u_in_lag1  # already 0 at breath boundaries
    breath_id__u_in_lag2 = u_in_lag2
    print("Step-5...Completed")

    time_step_diff = (
        np.diff(time_step, axis=1, prepend=0.0).astype(np.float32).reshape(-1)
    )

    ewm_u_in_mean = _ewm_mean_2d(u_in.astype(np.float64), halflife=9).reshape(-1)

    rsum, rmin, rmax, rmean = _rolling_agg_2d(u_in, window=15)
    in_15_sum = rsum.reshape(-1)
    in_15_min = rmin.reshape(-1)
    in_15_max = rmax.reshape(-1)
    in_15_mean = rmean.reshape(-1)
    print("Step-6...Completed")

    u_in_lagback_diff1 = u_in_flat - u_in_lag_back1
    u_out_lagback_diff1 = u_out_flat - u_out_lag_back1
    u_in_lagback_diff2 = u_in_flat - u_in_lag_back2
    u_out_lagback_diff2 = u_out_flat - u_out_lag_back2
    print("Step-7...Completed")

    df = df.copy()
    df["cross"] = cross
    df["cross2"] = cross2
    df["area"] = area
    df["time_step_cumsum"] = time_step_cumsum
    df["u_in_cumsum"] = u_in_cumsum

    df["u_in_lag1"] = u_in_lag1
    df["u_out_lag1"] = u_out_lag1
    df["u_in_lag_back1"] = u_in_lag_back1
    df["u_out_lag_back1"] = u_out_lag_back1
    df["u_in_lag2"] = u_in_lag2
    df["u_out_lag2"] = u_out_lag2
    df["u_in_lag_back2"] = u_in_lag_back2
    df["u_out_lag_back2"] = u_out_lag_back2
    df["u_in_lag3"] = u_in_lag3
    df["u_out_lag3"] = u_out_lag3
    df["u_in_lag_back3"] = u_in_lag_back3
    df["u_out_lag_back3"] = u_out_lag_back3
    df["u_in_lag4"] = u_in_lag4
    df["u_out_lag4"] = u_out_lag4
    df["u_in_lag_back4"] = u_in_lag_back4
    df["u_out_lag_back4"] = u_out_lag_back4

    df["breath_id__u_in__max"] = breath_id__u_in__max
    df["breath_id__u_in__mean"] = breath_id__u_in__mean
    df["breath_id__u_in__diffmax"] = breath_id__u_in__diffmax
    df["breath_id__u_in__diffmean"] = breath_id__u_in__diffmean

    df["u_in_diff1"] = u_in_diff1
    df["u_out_diff1"] = u_out_diff1
    df["u_in_diff2"] = u_in_diff2
    df["u_out_diff2"] = u_out_diff2
    df["u_in_diff3"] = u_in_diff3
    df["u_out_diff3"] = u_out_diff3
    df["u_in_diff4"] = u_in_diff4
    df["u_out_diff4"] = u_out_diff4

    df["one"] = 1
    df["count"] = count.astype(np.float32)
    df["u_in_cummean"] = u_in_cummean.astype(np.float32)

    df["breath_id_lag"] = 0
    df["breath_id_lag2"] = 0
    df["breath_id_lagsame"] = 0
    df["breath_id_lag2same"] = 0
    df["breath_id__u_in_lag"] = breath_id__u_in_lag
    df["breath_id__u_in_lag2"] = breath_id__u_in_lag2

    df["time_step_diff"] = time_step_diff
    df["ewm_u_in_mean"] = ewm_u_in_mean
    df["15_in_sum"] = in_15_sum
    df["15_in_min"] = in_15_min
    df["15_in_max"] = in_15_max
    df["15_in_mean"] = in_15_mean

    df["u_in_lagback_diff1"] = u_in_lagback_diff1
    df["u_out_lagback_diff1"] = u_out_lagback_diff1
    df["u_in_lagback_diff2"] = u_in_lagback_diff2
    df["u_out_lagback_diff2"] = u_out_lagback_diff2
    print("Step-7.5...Completed")

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["R__C"] = df["R"].astype(str) + "__" + df["C"].astype(str)
    df = pd.get_dummies(df)
    print("Step-8...Completed")
    return df


train = add_features(train)
test = add_features(test)



## === cell 4
print("Train shape is now:", train.shape)
train.head()



## === cell 5
missing_in_test = [c for c in train.columns if c not in test.columns]
for c in missing_in_test:
    test[c] = 0
extra_in_test = [c for c in test.columns if c not in train.columns]
if extra_in_test:
    test = test.drop(columns=extra_in_test)
test = test[train.columns]



## === cell 6
train_pressure = (
    pd.read_csv(TRAIN_PATH, usecols=["pressure"], dtype={"pressure": "float32"})[
        "pressure"
    ]
    .to_numpy(dtype=np.float32)
    .reshape(-1, 80)
)
pressure_diff = np.diff(train_pressure, axis=1, prepend=0.0).astype(np.float32)
pressure_integral = (np.cumsum(train_pressure, axis=1) / 200.0).astype(np.float32)
targets = np.stack([train_pressure, pressure_diff, pressure_integral], axis=-1)

train.drop(
    [
        "pressure",
        "id",
        "breath_id",
        "one",
        "count",
        "breath_id_lag",
        "breath_id_lag2",
        "breath_id_lagsame",
        "breath_id_lag2same",
    ],
    axis=1,
    inplace=True,
)

test.drop(
    [
        "id",
        "breath_id",
        "one",
        "count",
        "breath_id_lag",
        "breath_id_lag2",
        "breath_id_lagsame",
        "breath_id_lag2same",
    ],
    axis=1,
    inplace=True,
)



## === cell 7
print("Targets shape is", targets.shape)



## === cell 8
print("Train features:", train.shape, "Test features:", test.shape)



## === cell 9
COL_ORDER = (
    list(train.columns[:3]) + list(train.columns[-15:]) + list(train.columns[3:-15])
)
train = train[COL_ORDER]
test = test[COL_ORDER]

print("Train columns:")
np.array(COL_ORDER)



## === cell 10
assert train.shape[1] == test.shape[1]



## === cell 11
RS = RobustScaler()
train_np = train.to_numpy(dtype=np.float32, copy=False)
test_np = test.to_numpy(dtype=np.float32, copy=False)

train_np = RS.fit_transform(train_np)
test_np = RS.transform(test_np)

train = train_np.reshape(-1, 80, train_np.shape[-1]).astype(np.float32, copy=False)
test = test_np.reshape(-1, 80, train_np.shape[-1]).astype(np.float32, copy=False)



## === cell 12
print("Reshaped train/test:", train.shape, test.shape)



## === cell 13
U_OUT_IDX = 2  # keep for mask computation on X_valid as in original logic
u_out_raw_seq = (
    pd.read_csv(TRAIN_PATH, usecols=["u_out"], dtype={"u_out": "int8"})["u_out"]
    .to_numpy()
    .reshape(-1, 80)
    .astype(np.float32)
)

y_weight = (u_out_raw_seq == 0).astype(np.float32)[..., None]  # (n_breaths, 80, 1)



## === cell 14
train.shape, targets.shape, y_weight.shape



## === cell 15
print(
    "Weight stats:", y_weight.min(), y_weight.max(), "fraction kept:", y_weight.mean()
)



## === cell 16
if os.environ["CUDA_VISIBLE_DEVICES"].count(",") == 0:
    gpu_strategy = tf.distribute.get_strategy()
    print("single strategy")
else:
    gpu_strategy = tf.distribute.MirroredStrategy()
    print("multiple strategy")



## === cell 17
try:
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    print("Mixed precision enabled (experimental)")
except Exception as e:
    print("Mixed precision not enabled:", repr(e))



## === cell 18
tf.keras.backend.set_floatx("float32")




## === cell 19
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="gelu"),
                layers.Dense(feat_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, training=None):
        x = inputs
        if isinstance(x, tf.SparseTensor):
            x = tf.sparse.to_dense(x)
        x = tf.ensure_shape(x, (None, 80, self.feat_dim))
        x = tf.cast(x, tf.float32)

        attn_output = self.att(query=x, value=x, key=x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)




## === cell 20
feat_dim = train.shape[-1] + 32
embed_dim = 64
num_heads = 8
ff_dim = 128
dropout_rate = 0.0
num_blocks = 12


def build_model():
    inputs = layers.Input(shape=train.shape[-2:], dtype="float32", sparse=False)

    x = layers.Dense(feat_dim)(inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    for k in range(num_blocks):
        x_old = x
        transformer_block = TransformerBlock(
            embed_dim, feat_dim, num_heads, ff_dim, dropout_rate
        )
        x = transformer_block(x)
        x = 0.7 * x + 0.3 * x_old

    x = layers.Dense(128, activation="selu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(3, activation="linear", dtype="float32")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model




## === cell 21
with gpu_strategy.scope():
    _m = build_model()
_m.summary()



## === cell 22
import math
import matplotlib.pyplot as plt

LR_START = 1e-6
LR_MAX = 6e-4
LR_MIN = 1e-6
LR_RAMPUP_EPOCHS = 0
LR_SUSTAIN_EPOCHS = 0
EPOCHS = 420
STEPS = [60, 120, 240]


def lrfn(epoch):
    if epoch < STEPS[0]:
        epoch2 = epoch
        EPOCHS2 = STEPS[0]
    elif epoch < STEPS[0] + STEPS[1]:
        epoch2 = epoch - STEPS[0]
        EPOCHS2 = STEPS[1]
    elif epoch < STEPS[0] + STEPS[1] + STEPS[2]:
        epoch2 = epoch - STEPS[0] - STEPS[1]
        EPOCHS2 = STEPS[2]
    else:
        epoch2 = epoch - (STEPS[0] + STEPS[1] + STEPS[2])
        EPOCHS2 = max(1, STEPS[-1])

    if epoch2 < LR_RAMPUP_EPOCHS and LR_RAMPUP_EPOCHS > 0:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch2 + LR_START
    elif epoch2 < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        decay_total_epochs = max(1, EPOCHS2 - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1)
        decay_epoch_index = epoch2 - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        phase = math.pi * decay_epoch_index / decay_total_epochs
        cosine_decay = 0.5 * (1 + math.cos(phase))
        lr = (LR_MAX - LR_MIN) * cosine_decay + LR_MIN
    return float(lr)


rng = [i for i in range(EPOCHS)]
lr_y = [lrfn(x) for x in rng]
plt.figure(figsize=(10, 4))
plt.plot(rng, lr_y, "-o")
print(
    "Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(
        lr_y[0], max(lr_y), lr_y[-1]
    )
)
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
plt.xlabel("Epoch", size=14)
plt.ylabel("Learning Rate", size=14)
plt.show()



## === cell 23
import glob


def _find_weight_file(fold, ver):
    p = f"../input/vent-tranformer/folds{fold}_{ver}.weights.h5"
    if os.path.exists(p):
        return p
    p2 = f"folds{fold}_{ver}.weights.h5"
    if os.path.exists(p2):
        return p2
    return None




## === cell 24
EPOCH = EPOCHS
BATCH_SIZE = 64
NUM_FOLDS = 11
SEED = 42
VERBOSE = 1

_pretrained_any = _find_weight_file(0, VER) is not None
if not TRAIN_MODEL and not _pretrained_any:
    print(
        "Pretrained weights not found; switching TRAIN_MODEL=True to ensure a valid submission is produced."
    )
    TRAIN_MODEL = True


def make_ds(X, y=None, sw=None, batch_size=64, training=False):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(X)
    elif sw is None:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y, sw))
    if training:
        pass
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


with gpu_strategy.scope():
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    test_preds = []
    oof_preds = []
    oof_true = []
    all_mask = []
    test_folds = []

    test_ds = make_ds(test, batch_size=BATCH_SIZE, training=False)

    for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)
        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]
        test_folds.append(test_idx)

        checkpoint_filepath = f"folds{fold}_{VER}.weights.h5"

        model = build_model()
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss="mae")

        sv = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="auto",
            save_freq="epoch",
        )

        if TRAIN_MODEL:
            train_ds = make_ds(
                X_train,
                y_train,
                y_weight[train_idx, :, :1],
                batch_size=BATCH_SIZE,
                training=True,
            )
            valid_ds = make_ds(
                X_valid,
                y_valid,
                y_weight[test_idx, :, :1],
                batch_size=BATCH_SIZE,
                training=False,
            )
            history = model.fit(
                train_ds,
                verbose=VERBOSE,
                validation_data=valid_ds,
                epochs=EPOCH,
                callbacks=[lr_callback, sv],
            )
            if os.path.exists(checkpoint_filepath):
                model.load_weights(checkpoint_filepath)
        else:
            wpath = _find_weight_file(fold, VER)
            if wpath is None:
                raise FileNotFoundError(
                    f"Pretrained weights not found for fold {fold}, VER {VER}. "
                    f"Expected at ../input/vent-tranformer/folds{fold}_{VER}.weights.h5"
                )
            model.load_weights(wpath)

        print("Predicting Test...")
        test_preds.append(model.predict(test_ds, verbose=VERBOSE)[:, :, 0].reshape(-1))

        print("Predicting OOF...")
        valid_ds_pred = make_ds(X_valid, batch_size=BATCH_SIZE, training=False)
        oof_preds.append(
            model.predict(valid_ds_pred, verbose=VERBOSE)[:, :, 0].reshape(-1, 1)
        )
        oof_true.append(y_valid[:, :, 0].reshape(-1, 1))

        u_out_valid_raw = u_out_raw_seq[test_idx].reshape(-1)
        mask = np.where(u_out_valid_raw == 0)[0]
        all_mask.append(mask)

        score_all = mean_absolute_error(oof_true[-1], oof_preds[-1])
        score_insp = mean_absolute_error(oof_true[-1][mask], oof_preds[-1][mask])
        print(f"Fold-{fold+1} | OOF all timesteps MAE: {score_all}")
        print(f"Fold-{fold+1} | OOF inspiratory (u_out=0) MAE: {score_insp}")

        np.save(f"oof_v{VER}_trans", oof_preds)

        if FIRST_FOLD_ONLY:
            break



## === cell 25
num_done_folds = len(oof_preds)
print("Folds completed:", num_done_folds)



## === cell 26
NUM_FOLDS_USED = num_done_folds



## === cell 27
assert NUM_FOLDS_USED > 0, "No folds were run; cannot create submission."



## === cell 28
t = 0.0
for k in range(NUM_FOLDS_USED):
    mask = all_mask[k]
    mae = np.mean(np.abs(oof_preds[k].flatten()[mask] - oof_true[k].flatten()[mask]))
    t += mae
    print("Fold", k, "has inspiratory MAE =", mae)
print("Overall CV inspiratory MAE =", t / NUM_FOLDS_USED)



## === cell 29
t = 0.0
for k in range(NUM_FOLDS_USED):
    mask_k = all_mask[k]
    oof = oof_preds[k].copy()
    oof2 = (
        np.round((oof + 1.895744294564641) / 0.07030214545121005) * 0.07030214545121005
        - 1.895744294564641
    )
    mae = np.mean(np.abs(oof2.flatten()[mask_k] - oof_true[k].flatten()[mask_k]))
    t += mae
    print("Fold", k, "has inspiratory MAE with PP =", mae)
print("Overall CV inspiratory MAE with PP =", t / NUM_FOLDS_USED)



## === cell 30
if len(test_preds) == 0:
    raise RuntimeError("No test predictions were generated; cannot write submission.")
print("Num test_preds:", len(test_preds), "each length:", test_preds[0].shape[0])



## === cell 31
train_df = pd.read_csv(
    TRAIN_PATH,
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
    dtype=train_dtypes,
)

folds = test_folds.copy()
for k in range(len(folds)):
    folds[k] = np.ones_like(folds[k]) * k
folds = np.hstack(folds)
folds = np.repeat(folds, 80)

test_folds_flat = np.hstack(test_folds)
test_folds_flat = 80 * np.repeat(test_folds_flat, 80)
shifter = np.tile(np.arange(80), len(test_folds_flat) // 80)
test_folds_flat += shifter

train_df = train_df.loc[test_folds_flat].copy()

oof_stack = np.vstack(oof_preds)  # (N,1)
train_df["oof"] = oof_stack.squeeze()
train_df["fold"] = folds[: len(train_df)]

train_df.head()



## === cell 32
train_df["id"] = train_df["id"].astype("int32")
train_df["oof"] = train_df["oof"].astype("float32")
train_df["fold"] = train_df["fold"].astype("int8")
train_df[["id", "oof", "fold"]].to_csv(f"oof_v{VER}.csv", index=False)
train_df[["id", "oof", "fold"]].head()



## === cell 33
test_pred_mean = np.mean(np.vstack(test_preds), axis=0)



## === cell 34
submission = pd.read_csv(SAMPLE_PATH)
submission["pressure"] = test_pred_mean
submission.to_csv(f"submission_mean_{VER}.csv", index=False)
print("Wrote:", f"submission_mean_{VER}.csv", "shape:", submission.shape)



## === cell 35
submission = pd.read_csv(SAMPLE_PATH)
submission["pressure"] = np.median(np.vstack(test_preds), axis=0)
submission.to_csv(f"submission_median_{VER}.csv", index=False)
print("Wrote:", f"submission_median_{VER}.csv", "shape:", submission.shape)



## === cell 36
submission.head()



## === cell 37
submission["pressure"] = (
    np.round((submission["pressure"] + 1.895744294564641) / 0.07030214545121005)
    * 0.07030214545121005
    - 1.895744294564641
)
submission.to_csv(f"submission_median_snap_{VER}.csv", index=False)
print("Wrote:", f"submission_median_snap_{VER}.csv", "shape:", submission.shape)



## === cell 38
submission.head()
