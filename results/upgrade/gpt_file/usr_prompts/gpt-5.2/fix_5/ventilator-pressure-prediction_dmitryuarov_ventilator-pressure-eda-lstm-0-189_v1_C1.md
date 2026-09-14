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

catboost==1.2.8
geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
xgboost==2.0.3

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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "228")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay

pd.set_option("display.max_columns", None)

np.random.seed(228)
tf.keras.utils.set_random_seed(228)

DATA_DIR = "../input/ventilator-pressure-prediction"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
ss = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")



## === cell 1
print(f"Length of TRAIN dataset: {len(train)}")
print(f"Length of TEST dataset: {len(test)}\n")

print("Missing values in TRAIN dataset")
print(train.iloc[:, 0:-1].isna().sum().to_string())
print("\nMissing values in TEST dataset")
print(test.isna().sum().to_string())
print("")
print(f'Number of breaths in train dataset: {train["breath_id"].nunique()}')
print(f'Number of breaths in test dataset: {test["breath_id"].nunique()}')
obs_per_breath = train["breath_id"].value_counts().iat[0]
print(f"The number of observations for each breath: {obs_per_breath}")



## === cell 2
pass



## === cell 3
pass



## === cell 4
pass




## === cell 5
def features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["area"] = (
        (df["time_step"] * df["u_in"]).groupby(df["breath_id"], sort=False).cumsum()
    )
    df["u_in_cumsum"] = df["u_in"].groupby(df["breath_id"], sort=False).cumsum()

    df["u_in_lag2"] = df["u_in"].shift(2).fillna(0)
    df["u_in_lag4"] = df["u_in"].shift(4).fillna(0)

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)

    g = df.groupby("breath_id", sort=False)["u_in"]

    ewm_g = g.ewm(halflife=10)
    df["ewm_u_in_mean"] = ewm_g.mean().reset_index(level=0, drop=True)
    df["ewm_u_in_std"] = ewm_g.std().reset_index(level=0, drop=True)
    df["ewm_u_in_corr"] = ewm_g.corr().reset_index(level=0, drop=True)

    roll_g = g.rolling(window=10, min_periods=1)
    df["rolling_10_mean"] = roll_g.mean().reset_index(level=0, drop=True)
    df["rolling_10_max"] = roll_g.max().reset_index(level=0, drop=True)
    df["rolling_10_std"] = roll_g.std().reset_index(level=0, drop=True)

    exp_g = g.expanding(2)
    df["expand_mean"] = exp_g.mean().reset_index(level=0, drop=True)
    df["expand_max"] = exp_g.max().reset_index(level=0, drop=True)
    df["expand_std"] = exp_g.std().reset_index(level=0, drop=True)

    return df


train_feat = features(train)
test_feat = features(test)

train_feat = pd.get_dummies(train_feat, columns=["R", "C"])
test_feat = pd.get_dummies(test_feat, columns=["R", "C"])

train_feat, test_feat = train_feat.align(test_feat, join="left", axis=1, fill_value=0)
if "pressure" in test_feat.columns:
    test_feat = test_feat.drop(columns=["pressure"])

train = train_feat
test = test_feat
del train_feat, test_feat



## === cell 6
train = train.fillna(0)
test = test.fillna(0)



## === cell 7
targets = train[["pressure"]].to_numpy().reshape(-1, 80).astype(np.float32, copy=False)

train.drop(["pressure", "id", "breath_id"], axis=1, inplace=True)
test_ids = test["id"].to_numpy()
test.drop(["id", "breath_id"], axis=1, inplace=True)



## === cell 8
RS = RobustScaler()

train = RS.fit_transform(train).astype(np.float32, copy=False)
test = RS.transform(test).astype(np.float32, copy=False)



## === cell 9
n_features = train.shape[-1]
train = np.ascontiguousarray(train.reshape(-1, 80, n_features))
test = np.ascontiguousarray(test.reshape(-1, 80, n_features))
targets = np.ascontiguousarray(targets)

print(
    "Train shape:",
    train.shape,
    "Targets shape:",
    targets.shape,
    "Test shape:",
    test.shape,
)



## === cell 10
EPOCH = 300
BATCH_SIZE = 1024

try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
    print("Using TPU strategy.")
except Exception:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy on {len(gpus)} GPUs.")
    else:
        strategy = tf.distribute.get_strategy()
        print("Using default strategy (CPU or single GPU).")

AUTOTUNE = tf.data.AUTOTUNE

with strategy.scope():
    kf = KFold(n_splits=5, shuffle=True, random_state=228)
    test_preds = []

    test_ds = (
        tf.data.Dataset.from_tensor_slices(test)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train, targets)):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

        X_train, X_valid = train[train_idx], train[valid_idx]
        y_train, y_valid = targets[train_idx], targets[valid_idx]

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(AUTOTUNE)
        )
        valid_ds = (
            tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(AUTOTUNE)
        )

        model = keras.models.Sequential(
            [
                keras.layers.Input(shape=train.shape[-2:]),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(400, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(300, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(200, return_sequences=True)
                ),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(100, return_sequences=True)
                ),
                keras.layers.Dense(50, activation="selu"),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mae")

        steps_per_epoch = int(np.ceil(len(X_train) / BATCH_SIZE))
        decay_schedule = ExponentialDecay(
            1e-3, decay_steps=400 * max(1, steps_per_epoch), decay_rate=1e-5
        )

        def lr_fn(epoch, lr):
            return float(decay_schedule(epoch * steps_per_epoch).numpy())

        lr_cb = LearningRateScheduler(lr_fn, verbose=1)

        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCH,
            callbacks=[lr_cb],
            verbose=2,
        )

        preds = model.predict(test_ds, verbose=0).reshape(-1)
        test_preds.append(preds)



## === cell 11
if len(test_preds) == 0:
    raise RuntimeError(
        "No fold predictions were produced; training likely failed earlier."
    )

pred = sum(test_preds) / len(test_preds)
pred = pred.reshape(-1)

if len(pred) != len(ss):
    raise ValueError(
        f"Prediction length {len(pred)} does not match submission length {len(ss)}"
    )

ss["pressure"] = pred.astype(np.float32)
ss.to_csv("submission.csv", index=False)

print("Wrote submission to submission.csv with shape:", ss.shape)
print(ss.head())
