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
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder

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




## === cell 1
def add_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("breath_id", sort=False, observed=True)

    u_in = df["u_in"]
    u_out = df["u_out"]

    u_in_cumsum = g["u_in"].cumsum()
    u_in_lag1 = g["u_in"].shift(1)
    u_in_lag2 = g["u_in"].shift(2)
    u_out_lag1 = g["u_out"].shift(1)

    df["u_in_cumsum"] = u_in_cumsum
    df["u_in_lag1"] = u_in_lag1.fillna(0.0)
    df["u_in_lag2"] = u_in_lag2.fillna(0.0)
    df["u_out_lag1"] = u_out_lag1.fillna(0).astype(np.int8)

    df["u_in_diff1"] = u_in - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_lag1"] - df["u_in_lag2"]

    return df


def to_breath_tensor_from_array(X: np.ndarray, n_steps: int = 80) -> np.ndarray:
    n_rows = X.shape[0]
    assert (
        n_rows % n_steps == 0
    ), f"Row count {n_rows} not divisible by n_steps {n_steps}"
    n_breaths = n_rows // n_steps
    return X.reshape(n_breaths, n_steps, X.shape[1])




## === cell 2
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

train_fe = add_features_fast(train_df)
test_fe = add_features_fast(test_df)

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

rc_train = train_fe[["R", "C"]].astype(str)
rc_test = test_fe[["R", "C"]].astype(str)
rc_all = pd.concat([rc_train, rc_test], axis=0, ignore_index=True)

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
ohe.fit(rc_all)

rc_train_ohe = ohe.transform(rc_train)
rc_test_ohe = ohe.transform(rc_test)
rc_feature_names = list(ohe.get_feature_names_out(["R", "C"]))

X_train_num = train_fe[num_cols].to_numpy(dtype=np.float32, copy=False)
X_test_num = test_fe[num_cols].to_numpy(dtype=np.float32, copy=False)

X_train_full = np.concatenate([X_train_num, rc_train_ohe], axis=1).astype(
    np.float32, copy=False
)
X_test_full = np.concatenate([X_test_num, rc_test_ohe], axis=1).astype(
    np.float32, copy=False
)

feature_cols = num_cols + rc_feature_names

scaler = RobustScaler(
    with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)
)
scaler.fit(X_train_full)
X_train_full = scaler.transform(X_train_full).astype(np.float32, copy=False)
X_test_full = scaler.transform(X_test_full).astype(np.float32, copy=False)

X = to_breath_tensor_from_array(X_train_full, n_steps=80)
y = train_df["pressure"].to_numpy(dtype=np.float32).reshape(-1, 80, 1)
X_test = to_breath_tensor_from_array(X_test_full, n_steps=80)

groups = train_df["breath_id"].values.reshape(-1, 80)[:, 0]

print(
    "X:", X.shape, "y:", y.shape, "X_test:", X_test.shape, "n_groups:", groups.shape[0]
)

del (
    train_fe,
    test_fe,
    X_train_num,
    X_test_num,
    rc_all,
    rc_train,
    rc_test,
    rc_train_ohe,
    rc_test_ohe,
    X_train_full,
    X_test_full,
)
gc.collect()




## === cell 3
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




## === cell 4
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



## === cell 5
config = {
    "BATCH_SIZE": 256,
    "EPOCHS": 15,
    "N_SPLITS": 3,
}

gkf = GroupKFold(n_splits=config["N_SPLITS"])


def make_array_dataset(X_arr, y_arr, batch_size, training: bool):
    opts = tf.data.Options()
    try:
        opts.deterministic = True
    except Exception:
        pass

    ds = tf.data.Dataset.from_tensor_slices((X_arr, y_arr))
    if training:
        ds = ds.shuffle(
            buffer_size=len(X_arr), seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds.with_options(opts)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_x_dataset(X_arr, batch_size):
    opts = tf.data.Options()
    try:
        opts.deterministic = True
    except Exception:
        pass
    ds = tf.data.Dataset.from_tensor_slices(X_arr).with_options(opts)
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

        train_ds = make_array_dataset(X_tr, y_tr, config["BATCH_SIZE"], training=True)
        valid_ds = make_array_dataset(X_va, y_va, config["BATCH_SIZE"], training=False)

        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=config["EPOCHS"],
            verbose=2,
        )

        va_pred = model.predict(valid_ds, verbose=0)
        mae = float(np.mean(np.abs(va_pred - y_va)))
        oof_mae.append(mae)
        print(f"Fold {fold} val MAE (raw): {mae:.6f}")

        pred_test = model.predict(test_ds, verbose=0).reshape(-1)
        test_preds.append(pred_test)

        keras.backend.clear_session()
        del (
            model,
            history,
            va_pred,
            pred_test,
            train_ds,
            valid_ds,
            X_tr,
            y_tr,
            X_va,
            y_va,
        )
        gc.collect()

print("\nMean CV MAE (raw):", float(np.mean(oof_mae)))



## === cell 6
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
