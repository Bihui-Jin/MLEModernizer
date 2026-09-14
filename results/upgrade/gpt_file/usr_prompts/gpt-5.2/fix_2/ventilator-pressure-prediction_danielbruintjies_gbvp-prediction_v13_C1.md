# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.1534471572034798

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import time
import numpy as np
import pandas as pd
import math
import random
import gc
import warnings

warnings.filterwarnings("ignore")


def set_seed(seeed: int = 42):
    from numpy.random import seed as np_seed

    np_seed(seeed)
    os.environ["PYTHONHASHSEED"] = str(seeed)
    random.seed(seeed)
    np.random.seed(seeed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seeed)
    except Exception:
        pass


start_time = time.time()
gc.enable()



## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

import matplotlib.pyplot as plt
import seaborn as sns


def display(x):
    try:
        from pandas import DataFrame, Series

        if isinstance(x, (DataFrame, Series)):
            print(x.head())
        else:
            print(x)
    except Exception:
        print(x)


def swish(x, beta=1):
    return x * sigmoid(beta * x)


get_custom_objects().update({"swish": Activation(swish)})

sns.set_style("darkgrid")
sns.set_palette("dark")
pd.set_option("display.max_columns", 300)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def connect_to_tpu(tpu_address: str = None):
    if tpu_address is not None:
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_address
        )
        if tpu_address not in ("", "local"):
            tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
        print("Running on TPU ", cluster_resolver.master())
        print("REPLICAS: ", strategy.num_replicas_in_sync)
        return cluster_resolver, strategy
    else:
        try:
            cluster_resolver = (
                tf.distribute.cluster_resolver.TPUClusterResolver.connect()
            )
            strategy = tf.distribute.TPUStrategy(cluster_resolver)
            print("Running on TPU ", cluster_resolver.master())
            print("REPLICAS: ", strategy.num_replicas_in_sync)
            return cluster_resolver, strategy
        except Exception:
            print("WARNING: No TPU detected. Using MirroredStrategy/CPU.")
            mirrored_strategy = tf.distribute.MirroredStrategy()
            return None, mirrored_strategy


cluster_resolver, strategy = connect_to_tpu()



## === cell 3
DEBUG = False
TRAIN_MODEL = True

TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"



## === cell 4
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
submission = pd.read_csv(SAMPLE_SUB_PATH)

train["bilstm_pred"] = 0.0
test["bilstm_pred"] = 0.0

bilstm_train_path = "../input/gbvpp-predictions1/bilstm_train.csv/bilstm_train.csv"
bilstm_test_path = "../input/gbvpp-predictions1/bilstm_test.csv/bilstm_test.csv"
if os.path.exists(bilstm_train_path) and os.path.exists(bilstm_test_path):
    train_bilstm = pd.read_csv(bilstm_train_path)
    test_bilstm = pd.read_csv(bilstm_test_path)
    if "pressure" in train_bilstm.columns and len(train_bilstm) == len(train):
        train["bilstm_pred"] = train_bilstm["pressure"].values
    if "pressure" in test_bilstm.columns and len(test_bilstm) == len(test):
        test["bilstm_pred"] = test_bilstm["pressure"].values
    del train_bilstm, test_bilstm
gc.collect()

if DEBUG:
    train = train[: 80 * 1000]



## === cell 5
print(f"Length of TRAIN dataset: {len(train)}")
print(f"Length of TEST dataset: {len(test)}")
print(f'Number of breaths in train dataset: {train["breath_id"].nunique()}')
print(f'Number of breaths in test dataset: {test["breath_id"].nunique()}')



## === cell 6
train_gf = pd.read_csv(TRAIN_PATH, usecols=["pressure"])
all_pressure = np.sort(train_gf.pressure.unique())
del train_gf
gc.collect()

PRESSURE_MIN = float(all_pressure[0].item())
PRESSURE_MAX = float(all_pressure[-1].item())
PRESSURE_STEP = float((all_pressure[1] - all_pressure[0]).item())

print(
    "PRESSURE_MIN",
    PRESSURE_MIN,
    "PRESSURE_MAX",
    PRESSURE_MAX,
    "PRESSURE_STEP",
    PRESSURE_STEP,
)



## === cell 7
train["log_u_in"] = np.log1p(train.u_in)
test["log_u_in"] = np.log1p(test.u_in)

train["time_step_class"] = pd.qcut(
    train.time_step, q=80, labels=range(0, 80), duplicates="drop"
)
test["time_step_class"] = pd.qcut(
    test.time_step, q=80, labels=range(0, 80), duplicates="drop"
)

piv = train.pivot_table(
    index="breath_id",
    columns="time_step_class",
    values="log_u_in",
    fill_value=0,
    aggfunc="mean",
)
piv_test = test.pivot_table(
    index="breath_id",
    columns="time_step_class",
    values="log_u_in",
    fill_value=0,
    aggfunc="mean",
)

pca = PCA(n_components=2, random_state=42)
pca.fit(piv)

train_pca = pca.transform(piv)
test_pca = pca.transform(piv_test)

train_pca = pd.DataFrame(train_pca, columns=["c0", "c1"], index=piv.index)
test_pca = pd.DataFrame(test_pca, columns=["c0", "c1"], index=piv_test.index)

km = KMeans(
    n_clusters=5, random_state=42, max_iter=200, init="k-means++", tol=0.0001, n_init=10
)
y_km = km.fit_predict(train_pca)
y_km_test = km.predict(test_pca)

train_pca["cluster"] = y_km
test_pca["cluster"] = y_km_test

train_pca["breath_id"] = train_pca.index
train_pca = train_pca[["breath_id", "cluster"]].reset_index(drop=True)
train = pd.merge(train, train_pca, how="left", on="breath_id")

test_pca["breath_id"] = test_pca.index
test_pca = test_pca[["breath_id", "cluster"]].reset_index(drop=True)
test = pd.merge(test, test_pca, how="left", on="breath_id")




## === cell 8
def find_cluster_r_c(df):
    fig, ax = plt.subplots(5, 2, figsize=(15, 10))
    for c in range(5):
        for r_c in range(2):
            col = "R" if r_c == 0 else "C"
            tmp = df.loc[df.cluster == c, col]
            sns.countplot(data=pd.DataFrame({col: tmp.values}), x=col, ax=ax[c][r_c])
            ax[c][r_c].set_title(f"Cluster={c}")
    plt.tight_layout()


def find_cluster_transition(df, is_train=True):
    fig, ax = plt.subplots(5, 5, figsize=(15, 10))
    for c in range(5):
        x = df.loc[df.cluster == c]
        breath = x.breath_id.unique()
        for n in range(5):
            if len(breath) <= n:
                continue
            if is_train:
                xx = x.loc[
                    x.breath_id == breath[n], ["time_step", "u_in", "u_out", "pressure"]
                ]
            else:
                xx = x.loc[x.breath_id == breath[n], ["time_step", "u_in", "u_out"]]
            xx.set_index("time_step").plot(ax=ax[c][n])
            ax[c][n].set_title(f"breath_id={breath[n]}")
            ax[c][n].set_xticks([])
            if n == 0:
                ax[c][n].set_ylabel(f"Cluster={c}")
    plt.tight_layout()


if DEBUG:
    find_cluster_r_c(train)
    find_cluster_r_c(test)
    find_cluster_transition(train, True)
    find_cluster_transition(test, False)



## === cell 9
train.drop(["time_step_class", "log_u_in"], axis=1, inplace=True)
test.drop(["time_step_class", "log_u_in"], axis=1, inplace=True)




## === cell 10
def add_features(df):
    df = df.copy()

    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()

    for lag in [1, 2, 3, 4]:
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag)
        df[f"u_out_lag{lag}"] = df.groupby("breath_id")["u_out"].shift(lag)
        df[f"u_in_lag_back{lag}"] = df.groupby("breath_id")["u_in"].shift(-lag)
        df[f"u_out_lag_back{lag}"] = df.groupby("breath_id")["u_out"].shift(-lag)

    df = df.fillna(0)

    df["breath_id__u_in__max"] = df.groupby("breath_id")["u_in"].transform("max")

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
    df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]
    df["u_in_diff3"] = df["u_in"] - df["u_in_lag3"]
    df["u_out_diff3"] = df["u_out"] - df["u_out_lag3"]
    df["u_in_diff4"] = df["u_in"] - df["u_in_lag4"]
    df["u_out_diff4"] = df["u_out"] - df["u_out_lag4"]

    df["breath_id__u_in__diffmax"] = (
        df.groupby("breath_id")["u_in"].transform("max") - df["u_in"]
    )
    df["breath_id__u_in__diffmean"] = (
        df.groupby("breath_id")["u_in"].transform("mean") - df["u_in"]
    )

    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]

    df["bilstm_pred_lag1"] = df.groupby("breath_id")["bilstm_pred"].shift(1)
    df["bilstm_pred_lag2"] = df.groupby("breath_id")["bilstm_pred"].shift(2)
    df["bilstm_pred_lag_back1"] = df.groupby("breath_id")["bilstm_pred"].shift(-1)
    df["bilstm_pred_lag_back2"] = df.groupby("breath_id")["bilstm_pred"].shift(-2)

    df = df.fillna(0)

    df["cluster__u_in__diffmean"] = (
        df.groupby("cluster")["u_in"].transform("mean") - df["u_in"]
    )
    df["cluster__u_in__diffmax"] = (
        df.groupby("cluster")["u_in"].transform("max") - df["u_in"]
    )

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["R__C"] = df["R"].astype(str) + "__" + df["C"].astype(str)
    df["cluster"] = df["cluster"].astype(str)

    df = pd.get_dummies(df)
    df = df.fillna(0)
    return df


train_fe = add_features(train)
test_fe = add_features(test)

for c in set(train_fe.columns) - set(test_fe.columns):
    test_fe[c] = 0
for c in set(test_fe.columns) - set(train_fe.columns):
    train_fe[c] = 0
train_fe = train_fe.sort_index(axis=1)
test_fe = test_fe.sort_index(axis=1)



## === cell 11
targets = train_fe[["pressure"]].to_numpy().reshape(-1, 80)

train_ids = train_fe[["id", "breath_id"]].copy()
test_ids = test_fe[["id", "breath_id"]].copy()

train_fe.drop(["pressure", "id", "breath_id"], axis=1, inplace=True)
test_fe.drop(["id", "breath_id"], axis=1, inplace=True)

train_fe.replace([np.inf, -np.inf], 0, inplace=True)
test_fe.replace([np.inf, -np.inf], 0, inplace=True)

RS = RobustScaler()
train_arr = RS.fit_transform(train_fe)
test_arr = RS.transform(test_fe)

train_arr = train_arr.reshape(-1, 80, train_arr.shape[-1])
test_arr = test_arr.reshape(-1, 80, train_arr.shape[-1])

print(
    "train_arr", train_arr.shape, "targets", targets.shape, "test_arr", test_arr.shape
)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4234590904.py in <cell line: 0>()
     12 RS = RobustScaler()
     13 train_arr = RS.fit_transform(train_fe)
---> 14 test_arr = RS.transform(test_fe)
     15 
     16 train_arr = train_arr.reshape(-1, 80, train_arr.shape[-1])

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in transform(self, X)
   1575         """
   1576         check_is_fitted(self)
-> 1577         X = self._validate_data(
   1578             X,
   1579             accept_sparse=("csr", "csc"),

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    546             validated.
    547         """
--> 548         self._check_feature_names(X, reset=reset)
    549 
    550         if y is None and self._get_tags()["requires_y"]:

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _check_feature_names(self, X, reset)
    479                 )
    480 
--> 481             raise ValueError(message)
    482 
    483     def _validate_data(

ValueError: The feature names should match those that were passed during fit.
Feature names unseen at fit time:
- pressure


## === cell 12
class Attention(Layer):
    def __init__(self, units=(80, 256), **kwargs):
        self.units = units
        super().__init__(**kwargs)

    def __call__(self, inputs):
        hidden_states = inputs
        hidden_size = int(hidden_states.shape[2])
        score_first_part = Dense(
            hidden_size, use_bias=False, name="attention_score_vec"
        )(hidden_states)
        h_t = Lambda(lambda x: x[:, -1, :], name="last_hidden_state")(hidden_states)
        score = Dot(axes=[1, 2], name="attention_score")([h_t, score_first_part])
        attention_weights = Activation("softmax", name="attention_weight")(score)
        context_vector = Dot(axes=[1, 1], name="context_vector")(
            [hidden_states, attention_weights]
        )
        pre_activation = Concatenate(name="attention_output")([context_vector, h_t])
        attention_vector = Dense(
            self.units, use_bias=False, activation="selu", name="attention_vector"
        )(pre_activation)
        return attention_vector

    def get_config(self):
        return {"units": self.units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)




## === cell 13
set_seed(23)

BATCH_SIZE = 1024
NUM_FOLDS = 10
EPOCHS = 300
if DEBUG:
    EPOCHS = 1


def fit_lstm(train_arr, test_arr, targets):
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)
    test_preds = []
    oof_preds = np.zeros_like(targets, dtype=np.float32)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_arr, targets)):
        K.clear_session()
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

        X_train, X_valid = train_arr[train_idx], train_arr[valid_idx]
        y_train, y_valid = targets[train_idx], targets[valid_idx]

        checkpoint_filepath = f"folds{fold}.hdf5"

        with strategy.scope():
            model = keras.models.Sequential(
                [
                    keras.layers.Input(shape=train_arr.shape[-2:]),
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
                    Attention(256),
                    keras.layers.Dense(128, activation="selu"),
                    Dense(units=1),
                ]
            )
            model.compile(optimizer="adam", loss="mae")

        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=0)
        es = EarlyStopping(
            monitor="val_loss",
            patience=60,
            verbose=1,
            mode="min",
            restore_best_weights=True,
        )
        sv = ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
        )

        if TRAIN_MODEL:
            model.fit(
                X_train,
                y_train,
                validation_data=(X_valid, y_valid),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
                callbacks=[lr, es, sv],
            )
        else:
            model = keras.models.load_model(
                checkpoint_filepath,
                custom_objects={"Attention": Attention, "swish": swish},
            )

        pred_valid = model.predict(X_valid, batch_size=BATCH_SIZE, verbose=0).reshape(
            -1, 80
        )
        oof_preds[valid_idx] = pred_valid

        pred_test = model.predict(test_arr, batch_size=BATCH_SIZE, verbose=0).reshape(
            -1, 80
        )
        test_preds.append(pred_test)

        del X_train, X_valid, y_train, y_valid, pred_valid, pred_test
        gc.collect()

    return oof_preds, test_preds




## === cell 14
oof_preds, test_preds = fit_lstm(train_arr, test_arr, targets)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3285784255.py in <cell line: 0>()
      1 # Train full CV and predict full test (no early next(generator) stop)
----> 2 oof_preds, test_preds = fit_lstm(train_arr, test_arr, targets)
      3 

NameError: name 'test_arr' is not defined

## === cell 15
test_pred = np.median(np.stack(test_preds, axis=0), axis=0).reshape(-1)

submission["pressure"] = test_pred
submission["pressure"] = (
    np.round((submission["pressure"] - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
    + PRESSURE_MIN
)
submission["pressure"] = np.clip(submission["pressure"], PRESSURE_MIN, PRESSURE_MAX)

submission = submission[["id", "pressure"]]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1404260470.py in <cell line: 0>()
      1 # Ensemble folds with median, then round to known pressure grid and clip
----> 2 test_pred = np.median(np.stack(test_preds, axis=0), axis=0).reshape(-1)
      3 
      4 submission["pressure"] = test_pred
      5 submission["pressure"] = (

NameError: name 'test_preds' is not defined

## === cell 16
print(
    "submission pressure stats:",
    submission["pressure"].min(),
    submission["pressure"].max(),
    submission["pressure"].nunique(),
)
print("Elapsed seconds:", round(time.time() - start_time, 2))
