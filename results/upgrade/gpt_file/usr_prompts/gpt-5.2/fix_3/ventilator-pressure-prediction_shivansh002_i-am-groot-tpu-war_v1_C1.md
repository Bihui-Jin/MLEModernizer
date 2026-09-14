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
lightgbm==4.6.0
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
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold

SEED = 2021
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF version:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))



## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

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

train = pd.read_csv(train_path, dtype=train_dtypes)
test = pd.read_csv(test_path, dtype=test_dtypes)
submission = pd.read_csv(sub_path, dtype={"id": "int32", "pressure": "float32"})

print(train.shape, test.shape, submission.shape)
display(train.head())



## === cell 2
g_train = train.groupby("breath_id", sort=False, observed=True)
g_test = test.groupby("breath_id", sort=False, observed=True)

train["u_in_cumsum"] = g_train["u_in"].cumsum()
test["u_in_cumsum"] = g_test["u_in"].cumsum()



## === cell 3
train["u_in_lag"] = g_train["u_in"].shift(2).fillna(0.0)
test["u_in_lag"] = g_test["u_in"].shift(2).fillna(0.0)

train = train.fillna(0)
test = test.fillna(0)



## === cell 4
targets = train[["pressure"]].to_numpy(dtype=np.float32).reshape(-1, 80)

train_feat = train.drop(["pressure", "id", "breath_id", "u_out"], axis=1)
test_feat = test.drop(["id", "breath_id", "u_out"], axis=1)

assert (
    train_feat.shape[1] == 6
), f"Expected 6 features, got {train_feat.shape[1]} with cols {train_feat.columns.tolist()}"
assert (
    test_feat.shape[1] == 6
), f"Expected 6 features, got {test_feat.shape[1]} with cols {test_feat.columns.tolist()}"

n_train_breaths = train_feat.shape[0] // 80
n_test_breaths = test_feat.shape[0] // 80

train_arr = train_feat.to_numpy(dtype=np.float32, copy=False).reshape(
    n_train_breaths, 80, 6
)
test_arr = test_feat.to_numpy(dtype=np.float32, copy=False).reshape(
    n_test_breaths, 80, 6
)

print(
    "train_arr:",
    train_arr.shape,
    "targets:",
    targets.shape,
    "test_arr:",
    test_arr.shape,
)



## === cell 5
display(train_feat.head())
print(
    "Unique breath_id in train:",
    train["breath_id"].nunique(),
    "-> breaths:",
    n_train_breaths,
)
print(
    "Unique breath_id in test :",
    test["breath_id"].nunique(),
    "-> breaths:",
    n_test_breaths,
)



## === cell 6
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # will throw if no TPU
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except Exception as e:
    strategy = tf.distribute.get_strategy()  # default (GPU/CPU)
    print("TPU not available, using default strategy. Reason:", str(e)[:200])

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
test_preds = []

GLOBAL_BATCH_SIZE = 1024
AUTO = tf.data.AUTOTUNE


def make_ds(X, y=None, batch_size=GLOBAL_BATCH_SIZE, training=False):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(X)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(AUTO)
    return ds


with strategy.scope():
    for fold, (train_idx, valid_idx) in enumerate(
        kf.split(train_arr, targets), start=1
    ):
        print("-" * 15, ">", f"Fold {fold}", "<", "-" * 15)

        X_train, X_valid = train_arr[train_idx], train_arr[valid_idx]
        y_train, y_valid = targets[train_idx], targets[valid_idx]

        train_ds = make_ds(X_train, y_train, training=True)
        valid_ds = make_ds(X_valid, y_valid, training=False)

        model = keras.models.Sequential(
            [
                keras.layers.Input(shape=(80, 6)),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(300, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(250, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(150, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(50, return_sequences=True)
                ),
                keras.layers.Dense(50, activation="selu"),
                keras.layers.Dense(1),
            ]
        )

        model.compile(optimizer="adam", loss="mae")

        steps_per_epoch = int(np.ceil(len(X_train) / GLOBAL_BATCH_SIZE))
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=400 * steps_per_epoch,
            decay_rate=1e-5,
            staircase=False,
        )
        model.optimizer.learning_rate = lr_schedule

        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=200,
            verbose=2,
        )

        test_ds = make_ds(test_arr, y=None, training=False)
        pred = model.predict(test_ds, verbose=0).squeeze(-1)  # (n_test_breaths, 80)
        test_preds.append(pred)



## === cell 7
test_pred_mean = np.mean(np.stack(test_preds, axis=0), axis=0)  # (n_test_breaths, 80)
submission["pressure"] = test_pred_mean.reshape(-1)

assert len(submission) == len(
    test
), f"Submission length {len(submission)} != test rows {len(test)}"
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
display(submission.head())
