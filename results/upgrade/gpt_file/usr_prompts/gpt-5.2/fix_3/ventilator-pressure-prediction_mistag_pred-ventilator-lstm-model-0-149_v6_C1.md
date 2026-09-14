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

print("TensorFlow:", tf.__version__)



## === cell 1
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

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

train_ori = pd.read_csv(TRAIN_PATH, dtype=DTYPES_TRAIN)
test_ori = pd.read_csv(TEST_PATH, dtype=DTYPES_TEST)
print(train_ori.shape, test_ori.shape)
print(train_ori.columns)




## === cell 2
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("breath_id", sort=False)

    u_in = df["u_in"]
    u_out = df["u_out"]

    df["u_in_cumsum"] = g["u_in"].cumsum()
    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0.0)

    df["u_in_diff1"] = u_in - df["u_in_lag1"]
    df["u_out_diff1"] = u_out - df["u_out_lag1"]

    df["u_in_u_out"] = u_in * (1 - u_out)
    df["RC"] = df["R"] * df["C"]

    return df




## === cell 3
train = add_features(train_ori)
test = add_features(test_ori)

train.sort_values(["breath_id", "time_step"], kind="mergesort", inplace=True)
test.sort_values(["breath_id", "time_step"], kind="mergesort", inplace=True)

drop_cols_train = ["id", "breath_id", "pressure"]
drop_cols_test = ["id", "breath_id"]

X_train_df = train.drop(columns=drop_cols_train)
y_train = train["pressure"].to_numpy(dtype=np.float32, copy=False)

X_test_df = test.drop(columns=drop_cols_test)

RS = RobustScaler()
X_train_scaled = RS.fit_transform(X_train_df).astype(np.float32, copy=False)
X_test_scaled = RS.transform(X_test_df).astype(np.float32, copy=False)

n_features = X_train_scaled.shape[1]

assert X_train_scaled.shape[0] % 80 == 0, "Train rows not divisible by 80"
assert X_test_scaled.shape[0] % 80 == 0, "Test rows not divisible by 80"

X_train_3d = X_train_scaled.reshape(-1, 80, n_features)
y_train_3d = y_train.reshape(-1, 80, 1)
X_test_3d = X_test_scaled.reshape(-1, 80, n_features)

print(
    "X_train_3d:",
    X_train_3d.shape,
    "y_train_3d:",
    y_train_3d.shape,
    "X_test_3d:",
    X_test_3d.shape,
)

del X_train_df, X_test_df, X_train_scaled, X_test_scaled
gc.collect()



## === cell 4
unique_breath_ids = train["breath_id"].drop_duplicates().to_numpy()
rng = np.random.RandomState(SEED)
rng.shuffle(unique_breath_ids)

val_frac = 0.1
n_val = int(len(unique_breath_ids) * val_frac)
val_breath_ids = unique_breath_ids[:n_val]

breath_id_order = train["breath_id"].to_numpy()[::80]  # breath_id per 80-step block
val_mask = np.isin(breath_id_order, val_breath_ids, assume_unique=False)
train_mask = ~val_mask

X_tr, y_tr = X_train_3d[train_mask], y_train_3d[train_mask]
X_va, y_va = X_train_3d[val_mask], y_train_3d[val_mask]

print("Train breaths:", X_tr.shape[0], "Val breaths:", X_va.shape[0])




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
    model.compile(optimizer="adam", loss="mae")
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

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
    .batch(config["BATCH_SIZE"], drop_remainder=False)
    .prefetch(AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((X_va, y_va))
    .batch(config["BATCH_SIZE"], drop_remainder=False)
    .prefetch(AUTOTUNE)
)




## === cell 8
def train_one_model(seed_offset: int):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)

    with strategy.scope():
        model = create_lstm_model(input_shape=(X_tr.shape[1], X_tr.shape[2]))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["EPOCHS"],
        verbose=config["VERBOSE"],
    )
    return model




## === cell 9
models = []
for i in range(3):
    print(f"\nTraining model {i+1}/3")
    model = train_one_model(seed_offset=100 * i)
    models.append(model)
    gc.collect()



## === cell 10
test_preds = []
for i, model in enumerate(models):
    print(f"Predicting with model {i+1}/3")
    pred = model.predict(
        X_test_3d, batch_size=config["BATCH_SIZE"], verbose=2
    )  # (breaths, 80, 1)
    test_preds.append(pred.reshape(-1))  # flatten to rows

test_pred = np.median(np.vstack(test_preds), axis=0)
print("test_pred shape:", test_pred.shape)



## === cell 11
pressure_unique = np.sort(train_ori["pressure"].unique())
P_MIN = float(pressure_unique.min())
P_MAX = float(pressure_unique.max())
P_STEP = float(np.median(np.diff(pressure_unique)))

print("Min pressure:", P_MIN)
print("Max pressure:", P_MAX)
print("Pressure step:", P_STEP)
print("Unique values:", pressure_unique.shape[0])



## === cell 12
test_sorted_ids = test["id"].to_numpy()
assert test_sorted_ids.shape[0] == test_pred.shape[0]

submission = pd.DataFrame(
    {
        "id": test_sorted_ids.astype(np.int64, copy=False),
        "pressure": test_pred.astype(np.float32, copy=False),
    }
)

submission["pressure"] = (
    np.round((submission["pressure"] - P_MIN) / P_STEP) * P_STEP + P_MIN
)
submission["pressure"] = np.clip(submission["pressure"], P_MIN, P_MAX)

submission.sort_values("id", inplace=True)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
