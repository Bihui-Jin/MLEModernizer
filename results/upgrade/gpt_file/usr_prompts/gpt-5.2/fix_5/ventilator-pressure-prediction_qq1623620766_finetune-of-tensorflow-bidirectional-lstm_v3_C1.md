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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

os.environ.setdefault("PYTHONHASHSEED", "0")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 1
from numpy.random import seed

SEED = 2021
seed(SEED)
tf.random.set_seed(SEED)



## === cell 2
DEBUG = False

TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

train = pd.read_csv(
    TRAIN_PATH,
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
        "pressure": "float32",
    },
)
test = pd.read_csv(
    TEST_PATH,
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
    },
)
submission = pd.read_csv(SUB_PATH, dtype={"id": "int32", "pressure": "float32"})

if DEBUG:
    train = train[: 80 * 1000].copy()




## === cell 3
def add_features_fast(
    df: pd.DataFrame, *, rc_categories=None, r_categories=None, c_categories=None
) -> pd.DataFrame:
    df = df.copy()

    df.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    n = len(df)
    if n % 80 != 0:
        raise ValueError(
            f"Row count {n} is not divisible by 80; cannot reshape safely."
        )

    u_in = df["u_in"].to_numpy(np.float32).reshape(-1, 80)
    u_out = df["u_out"].to_numpy(np.float32).reshape(-1, 80)
    t = df["time_step"].to_numpy(np.float32).reshape(-1, 80)

    dt = np.zeros_like(t, dtype=np.float32)
    dt[:, 1:] = t[:, 1:] - t[:, :-1]
    area = np.cumsum(dt * u_in, axis=1, dtype=np.float32)

    u_in_cumsum = np.cumsum(u_in, axis=1, dtype=np.float32)

    def lag_pad(a, k):
        if k <= 0:
            return a
        out = np.zeros_like(a, dtype=np.float32)
        out[:, k:] = a[:, :-k]
        return out

    def back_lag_pad(a, k):
        if k <= 0:
            return a
        out = np.zeros_like(a, dtype=np.float32)
        out[:, :-k] = a[:, k:]
        return out

    feats = {}
    feats["area"] = area.reshape(-1)
    feats["u_in_cumsum"] = u_in_cumsum.reshape(-1)

    for lag in (1, 2, 3, 4):
        feats[f"u_in_lag{lag}"] = lag_pad(u_in, lag).reshape(-1)
        feats[f"u_out_lag{lag}"] = lag_pad(u_out, lag).reshape(-1)
        feats[f"u_in_lag_back{lag}"] = back_lag_pad(u_in, lag).reshape(-1)
        feats[f"u_out_lag_back{lag}"] = back_lag_pad(u_out, lag).reshape(-1)

    u_in_max = np.max(u_in, axis=1, keepdims=True).astype(np.float32)  # (B,1)
    u_out_max = np.max(u_out, axis=1, keepdims=True).astype(np.float32)  # (B,1)

    feats["breath_id__u_in__max"] = (
        u_in_max + np.zeros((u_in.shape[0], 80), dtype=np.float32)
    ).reshape(-1)
    feats["breath_id__u_out__max"] = (
        u_out_max + np.zeros((u_out.shape[0], 80), dtype=np.float32)
    ).reshape(-1)

    for lag in (1, 2, 3, 4):
        feats[f"u_in_diff{lag}"] = (u_in - lag_pad(u_in, lag)).reshape(-1)
        feats[f"u_out_diff{lag}"] = (u_out - lag_pad(u_out, lag)).reshape(-1)

    u_in_mean = np.mean(u_in, axis=1, keepdims=True).astype(np.float32)
    u_in_flat = u_in.reshape(-1)
    feats["breath_id__u_in__diffmax"] = (
        u_in_max + np.zeros((u_in.shape[0], 80), dtype=np.float32)
    ).reshape(-1) - u_in_flat
    feats["breath_id__u_in__diffmean"] = (
        u_in_mean + np.zeros((u_in.shape[0], 80), dtype=np.float32)
    ).reshape(-1) - u_in_flat

    feats["cross"] = (u_in * u_out).reshape(-1)
    feats["cross2"] = (t * u_out).reshape(-1)

    for k, v in feats.items():
        df[k] = v

    if r_categories is None:
        r_categories = sorted(df["R"].unique().tolist())
    if c_categories is None:
        c_categories = sorted(df["C"].unique().tolist())

    df["R"] = pd.Categorical(
        df["R"].astype("int16").astype(str), categories=[str(x) for x in r_categories]
    )
    df["C"] = pd.Categorical(
        df["C"].astype("int16").astype(str), categories=[str(x) for x in c_categories]
    )

    if rc_categories is None:
        rc_categories = [
            f"{r}__{c}"
            for r in [str(x) for x in r_categories]
            for c in [str(x) for x in c_categories]
        ]

    df["R__C"] = pd.Categorical(
        (df["R"].astype(str) + "__" + df["C"].astype(str)), categories=rc_categories
    )

    df = pd.get_dummies(df)
    return df


r_cats = sorted(pd.concat([train["R"], test["R"]]).unique().tolist())
c_cats = sorted(pd.concat([train["C"], test["C"]]).unique().tolist())
rc_cats = [
    f"{r}__{c}" for r in [str(x) for x in r_cats] for c in [str(x) for x in c_cats]
]

train = add_features_fast(
    train, rc_categories=rc_cats, r_categories=r_cats, c_categories=c_cats
)
test = add_features_fast(
    test, rc_categories=rc_cats, r_categories=r_cats, c_categories=c_cats
)



## === cell 4
target_col = "pressure"
meta_cols_train = [target_col, "id", "breath_id"]
meta_cols_test = ["id", "breath_id"]

train_features = train.drop(columns=meta_cols_train, errors="ignore")
test_features = test.drop(columns=meta_cols_test, errors="ignore")

train_features, test_features = train_features.align(
    test_features, join="left", axis=1, fill_value=0
)

targets = train[[target_col]].to_numpy().reshape(-1, 80)



## === cell 5
RS = RobustScaler()
train_scaled = RS.fit_transform(train_features.to_numpy(copy=False))
test_scaled = RS.transform(test_features.to_numpy(copy=False))

train_scaled = np.ascontiguousarray(
    train_scaled.reshape(-1, 80, train_scaled.shape[-1])
)
test_scaled = np.ascontiguousarray(test_scaled.reshape(-1, 80, train_scaled.shape[-1]))



## === cell 6
EPOCH = 80
BATCH_SIZE = 1024
NUM_FOLDS = 2

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # auto-detect TPU
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.__class__.__name__} (TPU not available): {e}")

AUTOTUNE = tf.data.AUTOTUNE



## === cell 7
with strategy.scope():
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)
    test_preds = []

    options = tf.data.Options()
    options.experimental_deterministic = True

    test_ds = (
        tf.data.Dataset.from_tensor_slices(test_scaled)
        .batch(BATCH_SIZE, drop_remainder=False)
        .with_options(options)
        .prefetch(AUTOTUNE)
    )

    SHUFFLE_BUFFER = min(
        100_000, train_scaled.shape[0]
    )  # breaths-level buffer; deterministic with seed

    for fold, (train_idx, valid_idx) in enumerate(
        kf.split(train_scaled, targets), start=1
    ):
        print("-" * 15, ">", f"Fold {fold}", "<", "-" * 15)

        X_train, X_valid = train_scaled[train_idx], train_scaled[valid_idx]
        y_train, y_valid = targets[train_idx], targets[valid_idx]

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(
                buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
            )
            .batch(BATCH_SIZE, drop_remainder=False)
            .with_options(options)
            .prefetch(AUTOTUNE)
        )
        valid_ds = (
            tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
            .batch(BATCH_SIZE, drop_remainder=False)
            .with_options(options)
            .prefetch(AUTOTUNE)
        )

        model = keras.models.Sequential(
            [
                keras.layers.Input(shape=train_scaled.shape[-2:]),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(1024, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(512, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(256, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(128, return_sequences=True)
                ),
                keras.layers.Dense(128, activation="selu"),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mae", jit_compile=True)

        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
        es = EarlyStopping(
            monitor="val_loss",
            patience=60,
            verbose=1,
            mode="min",
            restore_best_weights=True,
        )

        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCH,
            callbacks=[lr, es],
            verbose=2,
        )

        pred = model.predict(test_ds, verbose=0).squeeze()
        test_preds.append(pred)



## === cell 8
submission = submission.copy()
submission["pressure"] = np.mean(
    np.vstack([p.reshape(1, -1) for p in test_preds]), axis=0
).reshape(-1)

submission = submission[["id", "pressure"]]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
