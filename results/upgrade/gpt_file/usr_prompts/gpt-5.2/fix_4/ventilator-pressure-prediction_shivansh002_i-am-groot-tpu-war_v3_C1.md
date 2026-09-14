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

3.9

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

SEED = 2021
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)  # XLA
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## === cell 1
DEBUG = False

TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

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
submission = pd.read_csv(SUB_PATH, dtype={"id": "int32", "pressure": "float32"})

train.shape, test.shape, submission.shape




## === cell 2
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    gb = df.groupby("breath_id", sort=False)

    df["area"] = (df["time_step"] * df["u_in"]).astype(np.float32)
    df["area"] = gb["area"].cumsum()

    df["u_in_cumsum"] = gb["u_in"].cumsum()

    df["u_in_lag"] = df["u_in"].shift(2).fillna(0).astype(np.float32)

    df["R"] = df["R"].astype("category")
    df["C"] = df["C"].astype("category")
    df = pd.get_dummies(df, columns=["R", "C"], dtype=np.uint8)

    ewm_gb = df.groupby("breath_id", sort=False)["u_in"].ewm

    df["ewm_u_in_mean"] = (
        ewm_gb(halflife=8).mean().reset_index(level=0, drop=True).astype(np.float32)
    )
    df["ewm_u_in_std"] = (
        ewm_gb(halflife=9).std().reset_index(level=0, drop=True).astype(np.float32)
    )
    df["ewm_u_in_corr"] = (
        ewm_gb(halflife=14).corr().reset_index(level=0, drop=True).astype(np.float32)
    )

    roll = (
        gb["u_in"]
        .rolling(window=15, min_periods=1)
        .agg(["max", "std"])
        .reset_index(level=0, drop=True)
    )
    df["15_in_max"] = roll["max"].astype(np.float32)
    df["15_in_std"] = roll["std"].astype(np.float32)

    return df


train = add_features(train)
test = add_features(test)




## === cell 3
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)
train.shape, test.shape




## === cell 4
targets = train[["pressure"]].to_numpy().reshape(-1, 80)

train.drop(["pressure", "id", "breath_id"], axis=1, inplace=True)
test = test.drop(["id", "breath_id"], axis=1)

train.shape, test.shape, targets.shape




## === cell 5
RS = RobustScaler()
train_np = RS.fit_transform(train.to_numpy())
test_np = RS.transform(test.to_numpy())

train_np = train_np.reshape(-1, 80, train_np.shape[-1]).astype(np.float32, copy=False)
test_np = test_np.reshape(-1, 80, train_np.shape[-1]).astype(np.float32, copy=False)

del train, test
gc.collect()

train_np.shape, test_np.shape




## === cell 6
EPOCH = 225
BATCH_SIZE = 1024


def get_strategy():
    """
    Keep same behavior: try TPU, else default.
    """
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        return tf.distribute.TPUStrategy(resolver)
    except Exception:
        return tf.distribute.get_strategy()


strategy = get_strategy()
print(
    "Using strategy:",
    type(strategy).__name__,
    "num_replicas:",
    strategy.num_replicas_in_sync,
)

test_preds = []

decay_steps = float(400 * ((len(train_np) * 0.8) / BATCH_SIZE))
initial_lr = 1e-3
decay_rate = 1e-5


def lr_schedule(epoch, lr):
    new_lr = initial_lr * (decay_rate ** (epoch / decay_steps))
    return float(new_lr)


def make_ds(X, y=None, batch_size=1024, training=False):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(X)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


with strategy.scope():
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_np, targets)):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

        X_train, X_valid = train_np[train_idx], train_np[valid_idx]
        y_train, y_valid = targets[train_idx], targets[valid_idx]

        ds_train = make_ds(X_train, y_train, batch_size=BATCH_SIZE, training=True)
        ds_valid = make_ds(X_valid, y_valid, batch_size=BATCH_SIZE, training=False)
        ds_test = make_ds(test_np, y=None, batch_size=BATCH_SIZE, training=False)

        model = keras.models.Sequential(
            [
                keras.layers.Input(shape=train_np.shape[-2:]),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(500, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(375, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(200, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(100, return_sequences=True)
                ),
                keras.layers.Dense(100, activation="selu"),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mae", jit_compile=True)

        lr_cb = LearningRateScheduler(lr_schedule, verbose=1)

        model.fit(
            ds_train,
            validation_data=ds_valid,
            epochs=EPOCH,
            callbacks=[lr_cb],
            verbose=1,
        )

        pred = model.predict(ds_test, verbose=1).squeeze()
        test_preds.append(pred)

        del model, X_train, X_valid, y_train, y_valid, pred, ds_train, ds_valid, ds_test
        gc.collect()

print("done")




## === cell 7
if (not isinstance(test_preds, list)) or (len(test_preds) == 0):
    submission["pressure"] = 0.0
else:
    submission["pressure"] = np.mean(
        np.vstack([p.reshape(1, -1) for p in test_preds]), axis=0
    ).ravel()

assert len(submission) == 603600, f"Unexpected submission length: {len(submission)}"
assert list(submission.columns) == [
    "id",
    "pressure",
], f"Unexpected columns: {submission.columns.tolist()}"

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", os.path.getsize("submission.csv"), "bytes")
submission.head()
