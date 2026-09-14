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
import gc
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
from sklearn.model_selection import GroupKFold

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "../input/ventilator-pressure-prediction"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass


def add_features_fast(df: pd.DataFrame, n_steps: int = 80) -> pd.DataFrame:
    n = len(df)
    assert n % n_steps == 0, f"Row count {n} not divisible by n_steps {n_steps}"

    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False).reshape(-1, n_steps)
    u_out = df["u_out"].to_numpy(dtype=np.int8, copy=False).reshape(-1, n_steps)

    u_in_cumsum = np.cumsum(u_in, axis=1, dtype=np.float32)

    u_in_lag1 = np.zeros_like(u_in, dtype=np.float32)
    u_in_lag1[:, 1:] = u_in[:, :-1]

    u_in_lag2 = np.zeros_like(u_in, dtype=np.float32)
    u_in_lag2[:, 2:] = u_in[:, :-2]

    u_out_lag1 = np.zeros_like(u_out, dtype=np.int8)
    u_out_lag1[:, 1:] = u_out[:, :-1]

    u_in_diff1 = u_in - u_in_lag1
    u_in_diff2 = u_in_lag1 - u_in_lag2

    df = df.copy(deep=False)
    df["u_in_cumsum"] = u_in_cumsum.reshape(-1)
    df["u_in_lag1"] = u_in_lag1.reshape(-1)
    df["u_in_lag2"] = u_in_lag2.reshape(-1)
    df["u_out_lag1"] = u_out_lag1.reshape(-1)
    df["u_in_diff1"] = u_in_diff1.reshape(-1)
    df["u_in_diff2"] = u_in_diff2.reshape(-1)
    return df


def to_breath_tensor_from_array(X: np.ndarray, n_steps: int = 80) -> np.ndarray:
    n_rows = X.shape[0]
    assert (
        n_rows % n_steps == 0
    ), f"Row count {n_rows} not divisible by n_steps {n_steps}"
    n_breaths = n_rows // n_steps
    return X.reshape(n_breaths, n_steps, X.shape[1])


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

train_df = pd.read_csv(TRAIN_PATH, dtype=train_dtypes)
test_df = pd.read_csv(TEST_PATH, dtype=test_dtypes)

unique_pressures = np.sort(train_df["pressure"].unique())
P_MIN = float(unique_pressures.min())
P_MAX = float(unique_pressures.max())
P_STEP = float(np.round(np.min(np.diff(unique_pressures)), 8))
print(
    "P_MIN:",
    P_MIN,
    "P_MAX:",
    P_MAX,
    "P_STEP:",
    P_STEP,
    "n_unique:",
    unique_pressures.shape[0],
)

train_fe = add_features_fast(train_df, n_steps=80)
test_fe = add_features_fast(test_df, n_steps=80)

num_cols = [
    "time_step",
    "u_in",
    "u_out",
    "u_in_cumsum",
    "u_in_lag1",
    "u_in_lag2",
    "u_out_lag1",
    "u_in_diff1",
    "u_in_diff2",
]

R_vals = np.array([5, 20, 50], dtype=np.int16)
C_vals = np.array([10, 20, 50], dtype=np.int16)


def rc_onehot_from_df(df: pd.DataFrame) -> np.ndarray:
    r = df["R"].to_numpy(dtype=np.int16, copy=False)
    c = df["C"].to_numpy(dtype=np.int16, copy=False)

    r_idx = (r == 20).astype(np.int16) + 2 * (r == 50).astype(np.int16)
    c_idx = (c == 20).astype(np.int16) + 2 * (c == 50).astype(np.int16)

    k = (r_idx * 3 + c_idx).astype(np.int16, copy=False)  # 0..8
    out = np.zeros((k.shape[0], 9), dtype=np.float32)
    out[np.arange(k.shape[0]), k] = 1.0
    return out


rc_train_ohe = rc_onehot_from_df(train_fe)
rc_test_ohe = rc_onehot_from_df(test_fe)

rc_feature_names = [f"R_C_{int(r)}_{int(c)}" for r in R_vals for c in C_vals]

X_train_num = train_fe[num_cols].to_numpy(dtype=np.float32, copy=False)
X_test_num = test_fe[num_cols].to_numpy(dtype=np.float32, copy=False)

X_train_full = np.concatenate([X_train_num, rc_train_ohe], axis=1).astype(
    np.float32, copy=False
)
X_test_full = np.concatenate([X_test_num, rc_test_ohe], axis=1).astype(
    np.float32, copy=False
)

feature_cols = num_cols + rc_feature_names


def robust_scale_fit_transform(
    X_train: np.ndarray, X_test: np.ndarray, q_min=25.0, q_max=75.0
):
    X_train = np.asarray(X_train, dtype=np.float32, order="C")
    X_test = np.asarray(X_test, dtype=np.float32, order="C")
    med = np.median(X_train, axis=0).astype(np.float32, copy=False)
    q1 = np.percentile(X_train, q_min, axis=0).astype(np.float32, copy=False)
    q3 = np.percentile(X_train, q_max, axis=0).astype(np.float32, copy=False)
    iqr = (q3 - q1).astype(np.float32, copy=False)
    iqr[iqr == 0.0] = 1.0  # same spirit as sklearn (avoid division by zero)
    X_train_s = ((X_train - med) / iqr).astype(np.float32, copy=False)
    X_test_s = ((X_test - med) / iqr).astype(np.float32, copy=False)
    return X_train_s, X_test_s


X_train_full, X_test_full = robust_scale_fit_transform(X_train_full, X_test_full)

X = to_breath_tensor_from_array(X_train_full, n_steps=80)
y = train_df["pressure"].to_numpy(dtype=np.float32).reshape(-1, 80, 1)
X_test = to_breath_tensor_from_array(X_test_full, n_steps=80)

groups = train_df["breath_id"].to_numpy(copy=False).reshape(-1, 80)[:, 0]

print(
    "X:", X.shape, "y:", y.shape, "X_test:", X_test.shape, "n_groups:", groups.shape[0]
)

del (
    train_fe,
    test_fe,
    X_train_num,
    X_test_num,
    rc_train_ohe,
    rc_test_ohe,
    X_train_full,
    X_test_full,
)
gc.collect()




## === cell 1
def create_lstm_model(train_tensor):
    x0 = tf.keras.layers.Input(shape=(train_tensor.shape[-2], train_tensor.shape[-1]))

    lstm_layers = 4  # number of LSTM layers
    lstm_units = [940, 540, 462, 316]
    lstm = Bidirectional(keras.layers.LSTM(lstm_units[0], return_sequences=True))(x0)
    for i in range(lstm_layers - 1):
        lstm = Bidirectional(
            keras.layers.LSTM(lstm_units[i + 1], return_sequences=True)
        )(lstm)
    lstm = Dropout(0.002)(lstm)
    lstm = Dense(lstm_units[-1], activation="swish")(lstm)
    lstm = Dense(1)(lstm)

    model = keras.Model(inputs=x0, outputs=lstm)
    model.compile(optimizer="adam", loss="mae")
    return model




## === cell 2
def get_hardware_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Running on TPU", tpu.master())
    except Exception:
        tpu = None

    if tpu is not None:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        try:
            tf.config.optimizer.set_jit(True)
        except Exception:
            pass
    else:
        strategy = tf.distribute.get_strategy()
    return tpu, strategy


tpu, strategy = get_hardware_strategy()
print("Replicas:", strategy.num_replicas_in_sync)



## === cell 3
config = {
    "BATCH_SIZE": 256,
    "EPOCHS": 15,
    "N_SPLITS": 3,
}

gkf = GroupKFold(n_splits=config["N_SPLITS"])

_DATASET_OPTS = tf.data.Options()
try:
    _DATASET_OPTS.deterministic = True
except Exception:
    pass
try:
    _DATASET_OPTS.experimental_optimization.map_parallelization = True
    _DATASET_OPTS.experimental_optimization.parallel_batch = True
    _DATASET_OPTS.experimental_optimization.autotune_buffers = True
    _DATASET_OPTS.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    _DATASET_OPTS.experimental_slack = True
except Exception:
    pass
try:
    _DATASET_OPTS.threading.private_threadpool_size = 0
    _DATASET_OPTS.threading.max_intra_op_parallelism = 0
except Exception:
    pass


def make_xy_dataset(X_arr, y_arr, batch_size, shuffle, seed):
    ds = tf.data.Dataset.from_tensor_slices((X_arr, y_arr)).with_options(_DATASET_OPTS)
    if shuffle:
        buf = min(int(X_arr.shape[0]), 8192)
        ds = ds.shuffle(buffer_size=buf, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_x_dataset(X_arr, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(X_arr).with_options(_DATASET_OPTS)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_x_dataset(X_test, config["BATCH_SIZE"])

test_preds = []
oof_mae = []

with strategy.scope():
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        print(
            f"\nFold {fold}/{config['N_SPLITS']} - train breaths: {len(tr_idx)}, valid breaths: {len(va_idx)}"
        )
        model = create_lstm_model(X)

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        train_ds = make_xy_dataset(
            X_tr, y_tr, config["BATCH_SIZE"], shuffle=True, seed=SEED + fold
        )
        valid_ds = make_xy_dataset(
            X_va, y_va, config["BATCH_SIZE"], shuffle=False, seed=SEED + fold
        )

        del X_tr, y_tr
        gc.collect()

        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=config["EPOCHS"],
            verbose=2,
        )

        va_pred = model.predict(valid_ds, verbose=0)
        mae = float(
            tf.keras.metrics.mean_absolute_error(
                y_va.reshape(-1, 1), va_pred.reshape(-1, 1)
            )
            .numpy()
            .mean()
        )
        oof_mae.append(mae)
        print(f"Fold {fold} val MAE (raw): {mae:.6f}")

        pred_test = model.predict(test_ds, verbose=0).reshape(-1)
        test_preds.append(pred_test)

        keras.backend.clear_session()
        del model, history, va_pred, pred_test, X_va, y_va, train_ds, valid_ds
        gc.collect()

print("\nMean CV MAE (raw):", float(np.mean(oof_mae)))



## === cell 4
test_pred = np.median(np.vstack(test_preds), axis=0)

test_pred = np.round((test_pred - P_MIN) / P_STEP) * P_STEP + P_MIN
test_pred = np.clip(test_pred, P_MIN, P_MAX)

sub = pd.read_csv(SAMPLE_SUB_PATH)
assert len(sub) == len(
    test_df
), f"sample_submission rows {len(sub)} != test rows {len(test_df)}"
sub["pressure"] = test_pred.astype(np.float32)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
