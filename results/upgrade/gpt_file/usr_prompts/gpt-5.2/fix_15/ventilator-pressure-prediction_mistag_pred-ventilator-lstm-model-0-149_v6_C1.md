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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import gc
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, Bidirectional, LSTM

from sklearn.preprocessing import RobustScaler

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    import multiprocessing as _mp

    _NCPU = max(1, _mp.cpu_count())
    tf.config.threading.set_intra_op_parallelism_threads(min(8, _NCPU))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.run_functions_eagerly(False)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__ if hasattr(keras, "__version__") else "bundled")




## === cell 1
def _resolve_path(p: str) -> str:
    if os.path.exists(p):
        return p
    alt = p.replace("../input/", "/kaggle/input/")
    if os.path.exists(alt):
        return alt
    alt2 = p.replace("../input/ventilator-pressure-prediction/", "/kaggle/input/")
    if os.path.exists(alt2):
        return alt2
    return p  # will raise later in read_csv if truly missing


TRAIN_PATH = _resolve_path("../input/ventilator-pressure-prediction/train.csv")
TEST_PATH = _resolve_path("../input/ventilator-pressure-prediction/test.csv")
SAMPLE_SUB_PATH = _resolve_path(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

DTYPES_TRAIN = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
DTYPES_TEST = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

_READ_KW = {}
try:
    import pyarrow  # noqa: F401

    _READ_KW["engine"] = "pyarrow"
except Exception:
    _READ_KW["low_memory"] = False

train_ori = pd.read_csv(TRAIN_PATH, dtype=DTYPES_TRAIN, **_READ_KW)
test_ori = pd.read_csv(TEST_PATH, dtype=DTYPES_TEST, **_READ_KW)
print("train/test shapes:", train_ori.shape, test_ori.shape)
print("train columns:", list(train_ori.columns))




## === cell 2
def build_feature_matrix(df: pd.DataFrame, *, sort: bool) -> np.ndarray:
    if sort:
        bid = df["breath_id"].to_numpy(copy=False)
        if len(bid) >= 160:
            sample = bid[:160]
            needs_sort = np.any(sample[1:] < sample[:-1])
        else:
            needs_sort = np.any(bid[1:] < bid[:-1])
        if needs_sort:
            df = df.sort_values(
                ["breath_id", "time_step"], kind="mergesort"
            ).reset_index(drop=True)

    n = len(df)
    if n % 80 != 0:
        raise AssertionError(
            "Rows not divisible by 80; per-breath vectorization assumes 80 steps."
        )
    n_breaths = n // 80

    R = df["R"].to_numpy(copy=False).astype(np.float32, copy=False)
    C = df["C"].to_numpy(copy=False).astype(np.float32, copy=False)
    time_step = df["time_step"].to_numpy(copy=False).astype(np.float32, copy=False)

    u_in = (
        df["u_in"]
        .to_numpy(copy=False)
        .astype(np.float32, copy=False)
        .reshape(n_breaths, 80)
    )
    u_out = (
        df["u_out"]
        .to_numpy(copy=False)
        .astype(np.float32, copy=False)
        .reshape(n_breaths, 80)
    )

    u_in_cumsum = np.cumsum(u_in, axis=1)

    u_in_lag1 = np.empty_like(u_in)
    u_out_lag1 = np.empty_like(u_out)
    u_in_lag1[:, 0] = 0.0
    u_out_lag1[:, 0] = 0.0
    u_in_lag1[:, 1:] = u_in[:, :-1]
    u_out_lag1[:, 1:] = u_out[:, :-1]

    u_in_diff1 = u_in - u_in_lag1
    u_out_diff1 = u_out - u_out_lag1
    u_in_u_out = u_in * (1.0 - u_out)

    RC = (R * C).astype(np.float32, copy=False)

    X = np.empty((n, 12), dtype=np.float32)
    X[:, 0] = R
    X[:, 1] = C
    X[:, 2] = time_step
    X[:, 3] = u_in.reshape(-1)
    X[:, 4] = u_out.reshape(-1)
    X[:, 5] = u_in_cumsum.reshape(-1)
    X[:, 6] = u_in_lag1.reshape(-1)
    X[:, 7] = u_out_lag1.reshape(-1)
    X[:, 8] = u_in_diff1.reshape(-1)
    X[:, 9] = u_out_diff1.reshape(-1)
    X[:, 10] = u_in_u_out.reshape(-1)
    X[:, 11] = RC
    return X




## === cell 3
X_train_raw = build_feature_matrix(train_ori, sort=True)
y_train = train_ori["pressure"].to_numpy(dtype=np.float32, copy=False)
X_test_raw = build_feature_matrix(test_ori, sort=False)

RS = RobustScaler()
X_train_scaled = RS.fit_transform(X_train_raw).astype(np.float32, copy=False)
X_test_scaled = RS.transform(X_test_raw).astype(np.float32, copy=False)

del X_train_raw, X_test_raw
gc.collect()

n_features = X_train_scaled.shape[1]
assert X_train_scaled.shape[0] % 80 == 0, "Train rows not divisible by 80"
assert X_test_scaled.shape[0] % 80 == 0, "Test rows not divisible by 80"

X_train_3d = X_train_scaled.reshape(-1, 80, n_features)
y_train_3d = y_train.reshape(-1, 80, 1)
X_test_3d = X_test_scaled.reshape(-1, 80, n_features)

del X_train_scaled, X_test_scaled
gc.collect()

print(
    "X_train_3d:",
    X_train_3d.shape,
    "y_train_3d:",
    y_train_3d.shape,
    "X_test_3d:",
    X_test_3d.shape,
)



## === cell 4
unique_breath_ids = train_ori["breath_id"].drop_duplicates().to_numpy()
rng = np.random.RandomState(SEED)
rng.shuffle(unique_breath_ids)

val_frac = 0.1
n_val = int(len(unique_breath_ids) * val_frac)
val_breath_ids = unique_breath_ids[:n_val]

breath_id_order = train_ori["breath_id"].to_numpy(copy=False)[::80]
val_mask = np.isin(breath_id_order, val_breath_ids, assume_unique=False)
train_mask = ~val_mask

X_tr, y_tr = X_train_3d[train_mask], y_train_3d[train_mask]
X_va, y_va = X_train_3d[val_mask], y_train_3d[val_mask]

uout_train = (
    train_ori["u_out"]
    .to_numpy(copy=False)
    .astype(np.float32, copy=False)
    .reshape(-1, 80)
)
w_all = (1.0 - uout_train).reshape(-1, 80, 1)
w_tr, w_va = w_all[train_mask], w_all[val_mask]
del uout_train, w_all
gc.collect()

print("Train breaths:", X_tr.shape[0], "Val breaths:", X_va.shape[0])
print("Weights shapes:", w_tr.shape, w_va.shape)




## === cell 5
def create_lstm_model(input_shape):
    x0 = Input(shape=input_shape)

    lstm_layers = 4  # number of LSTM layers
    lstm_units = [940, 540, 462, 316]
    x = Bidirectional(LSTM(lstm_units[0], return_sequences=True))(x0)
    for i in range(lstm_layers - 1):
        x = Bidirectional(LSTM(lstm_units[i + 1], return_sequences=True))(x)

    x = Dropout(0.002)(x)
    x = Dense(lstm_units[-1], activation="swish")(x)
    x = Dense(1)(x)

    model = keras.Model(inputs=x0, outputs=x)
    model.compile(optimizer="adam", loss="mae", jit_compile=True)
    return model




## === cell 6
def get_hardware_strategy():
    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Running on TPU", tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()
    return tpu, strategy


tpu, strategy = get_hardware_strategy()
print("Num replicas:", strategy.num_replicas_in_sync)



## === cell 7
config = {
    "BATCH_SIZE": 256,
    "EPOCHS": 8,  # keep identical
    "VERBOSE": 2,
}
print(config)


def make_dataset(X, y=None, w=None, batch_size=256):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(X)
    else:
        if w is None:
            ds = tf.data.Dataset.from_tensor_slices((X, y))
        else:
            ds = tf.data.Dataset.from_tensor_slices((X, y, w))

    opt = tf.data.Options()
    opt.deterministic = True
    ds = ds.with_options(opt)

    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(X_tr, y_tr, w=w_tr, batch_size=config["BATCH_SIZE"])
val_ds = make_dataset(X_va, y_va, w=w_va, batch_size=config["BATCH_SIZE"])
test_ds = make_dataset(X_test_3d, y=None, w=None, batch_size=config["BATCH_SIZE"])




## === cell 8
def train_one_model(seed_offset: int):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)

    with strategy.scope():
        model = create_lstm_model(input_shape=(X_tr.shape[1], X_tr.shape[2]))

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["EPOCHS"],
        verbose=config["VERBOSE"],
    )
    return model




## === cell 9
n_rows_test = X_test_3d.shape[0] * 80
test_preds = np.empty((3, n_rows_test), dtype=np.float32)

for i in range(3):
    print(f"\nTraining model {i+1}/3")
    model = train_one_model(seed_offset=100 * i)

    print(f"Predicting with model {i+1}/3")
    pred = model.predict(test_ds, verbose=2)  # (breaths, 80, 1)
    test_preds[i] = pred.reshape(-1).astype(np.float32, copy=False)

    del model, pred
    tf.keras.backend.clear_session()
    gc.collect()

test_pred = np.median(test_preds, axis=0)
del test_preds
gc.collect()
print("test_pred shape:", test_pred.shape)



## === cell 10
pressure_unique = np.sort(train_ori["pressure"].unique())
P_MIN = float(pressure_unique.min())
P_MAX = float(pressure_unique.max())
P_STEP = float(np.median(np.diff(pressure_unique)))

print("Min pressure:", P_MIN)
print("Max pressure:", P_MAX)
print("Pressure step:", P_STEP)
print("Unique values:", pressure_unique.shape[0])



## === cell 11
test_ids = test_ori["id"].to_numpy(copy=False)
assert test_ids.shape[0] == test_pred.shape[0]

submission = pd.DataFrame(
    {
        "id": test_ids.astype(np.int64, copy=False),
        "pressure": test_pred.astype(np.float32, copy=False),
    }
)

submission["pressure"] = (
    np.round((submission["pressure"] - P_MIN) / P_STEP) * P_STEP + P_MIN
)
submission["pressure"] = np.clip(submission["pressure"], P_MIN, P_MAX)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))
