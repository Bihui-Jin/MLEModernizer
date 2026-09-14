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
import pandas as pd
import numpy as np
from tqdm import tqdm
import time, logging, gc, os, random
from sklearn.preprocessing import RobustScaler


from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.callbacks import *

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

tf.config.threading.set_intra_op_parallelism_threads(min(os.cpu_count(), 8))
tf.config.threading.set_inter_op_parallelism_threads(min(os.cpu_count(), 8))



## === cell 1
train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")




## === cell 2
def add_features(df):
    grp = df.groupby("breath_id", sort=False)

    df["area"] = (df["time_step"] * df["u_in"]).groupby(df["breath_id"]).cumsum()
    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]
    df["u_in_cumsum"] = grp["u_in"].cumsum()
    df["count"] = grp.cumcount() + 1
    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]

    df["breath_id_lag"] = df["breath_id"].shift(1).fillna(0)
    df["breath_id_lag2"] = df["breath_id"].shift(2).fillna(0)
    df["breath_id_lagsame"] = (df["breath_id_lag"] == df["breath_id"]).astype(np.int8)
    df["breath_id_lag2same"] = (df["breath_id_lag2"] == df["breath_id"]).astype(np.int8)

    df["u_in_lag"] = df["u_in"].shift(1).fillna(0) * df["breath_id_lagsame"]
    df["u_in_lag2"] = df["u_in"].shift(2).fillna(0) * df["breath_id_lag2same"]
    df["u_out_lag2"] = df["u_out"].shift(2).fillna(0) * df["breath_id_lag2same"]
    return df


train = add_features(train)
test = add_features(test)



## === cell 3
y = train["pressure"].to_numpy().reshape(-1, 80).astype(np.float32)
train.drop(
    [
        "pressure",
        "id",
        "breath_id",
        "count",
        "breath_id_lag",
        "breath_id_lag2",
        "breath_id_lagsame",
        "breath_id_lag2same",
        "u_out_lag2",
    ],
    axis=1,
    inplace=True,
)
test.drop(
    [
        "id",
        "breath_id",
        "count",
        "breath_id_lag",
        "breath_id_lag2",
        "breath_id_lagsame",
        "breath_id_lag2same",
        "u_out_lag2",
    ],
    axis=1,
    inplace=True,
)



## === cell 4
rb = RobustScaler()
rb.fit(train)
train = rb.transform(train)
test = rb.transform(test)

train = train.astype(np.float32)
test = test.astype(np.float32)



## === cell 5
train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, train.shape[-1])
gc.collect()



## === cell 6
print(tf.version.VERSION)
tf.get_logger().setLevel(logging.ERROR)
try:  # detect TPU
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:  # fallback to GPU / CPU
    strategy = tf.distribute.MirroredStrategy()  # works on GPU and multi‑GPU
    from tensorflow.keras.mixed_precision import Policy

    policy = Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
    tf.config.optimizer.set_jit(True)  # XLA compilation
    print("Mixed precision enabled")
print("REPLICAS: ", strategy.num_replicas_in_sync)




## === cell 7
def create_model():
    with strategy.scope():
        model = Sequential(
            [
                Input(shape=(80, train.shape[-1])),
                Bidirectional(LSTM(700, return_sequences=True)),
                Bidirectional(LSTM(512, return_sequences=True)),
                Bidirectional(LSTM(256, return_sequences=True)),
                Bidirectional(LSTM(128, return_sequences=True)),
                Dense(128, activation="elu"),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mae")
    return model




## === cell 8
BATCH_SIZE = 8192  # larger batch → fewer gradient steps per epoch
EPOCHS = 30  # reduced upper bound to keep total runtime < 600 s
PATIENCE = 15  # stop sooner if no improvement

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

test_preds = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(train, y)):
    print(f"****** fold: {fold+1} *******")
    X_train, X_valid = train[train_idx], train[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(10000, seed=SEED)
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    valid_ds = (
        tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=PATIENCE,
        verbose=1,
        restore_best_weights=True,
    )

    model = create_model()

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=[es],
        verbose=0,
    )
    pred_ds = tf.data.Dataset.from_tensor_slices(test).batch(BATCH_SIZE)
    preds = model.predict(pred_ds, verbose=0).squeeze()
    test_preds.append(preds.reshape(-1, 1).squeeze())

    tf.keras.backend.clear_session()
    del X_train, X_valid, y_train, y_valid, model, train_ds, valid_ds, pred_ds
    gc.collect()



## === cell 9
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
submission["pressure"] = np.mean(np.column_stack(test_preds), axis=1)
submission.to_csv("submission_mean.csv", index=False)
submission



## === cell 10
if test_preds:
    submission["pressure"] = np.median(np.column_stack(test_preds), axis=1)
    submission.to_csv("submission_median.csv", index=False)
submission



## === cell 11
train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
pressure_unique = np.array(sorted(train["pressure"].unique()))
submission["pressure"] = submission["pressure"].map(
    lambda x: pressure_unique[np.abs(pressure_unique - x).argmin()]
)
submission.to_csv("submission_post_preprocessing.csv", index=False)
submission
