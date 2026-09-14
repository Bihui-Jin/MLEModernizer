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

0.1491524544004632

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
gc.enable()


def display(x):
    try:
        print(x.head())
        print(f"(shape={x.shape})")
    except Exception:
        print(x)




## === cell 1
from tqdm import tqdm
from tqdm.keras import TqdmCallback

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
sns.set_palette("dark")
pd.set_option("display.max_columns", 300)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Bidirectional, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation


def swish(x, beta=1):
    return x * sigmoid(beta * x)


get_custom_objects().update({"swish": Activation(swish)})




## === cell 3
def set_seed(seeed: int):
    from numpy.random import seed as npseed

    npseed(seeed)
    tf.random.set_seed(seeed)
    os.environ["PYTHONHASHSEED"] = str(seeed)
    np.random.seed(seeed)
    random.seed(seeed)


start_time = time.time()




## === cell 4
def connect_to_tpu(tpu_address: str = None):
    if tpu_address is not None:  # When using GCP
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_address
        )
        if tpu_address not in ("", "local"):
            tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
        print("Running on TPU ", cluster_resolver.master())
        print("REPLICAS: ", strategy.num_replicas_in_sync)
        return cluster_resolver, strategy
    else:  # When using Colab or Kaggle
        try:
            cluster_resolver = (
                tf.distribute.cluster_resolver.TPUClusterResolver.connect()
            )
            strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
            print("Running on TPU ", cluster_resolver.master())
            print("REPLICAS: ", strategy.num_replicas_in_sync)
            return cluster_resolver, strategy
        except Exception:
            print("WARNING: No TPU detected.")
            mirrored_strategy = tf.distribute.MirroredStrategy()
            return None, mirrored_strategy


cluster_resolver, strategy = connect_to_tpu()



## === cell 5
DEBUG = False
TRAIN_MODEL = True



## === cell 6
DATA_DIR = "../input/ventilator-pressure-prediction"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

if DEBUG:
    train = train[: 80 * 1000]
    test = test[: 80 * 100]
    submission = submission[: 80 * 100]



## === cell 7
print("TRAIN\n")
display(train)
print("\n\nTEST\n")
display(test)



## === cell 8
print(f"Length of TRAIN dataset: {len(train)}")
print(f"Length of TEST dataset: {len(test)}")
print(f'Number of breaths in train dataset: {train["breath_id"].nunique()}')
print(f'Number of breaths in test dataset: {test["breath_id"].nunique()}')
print(
    f'The number of observations for each breath: {train["breath_id"].value_counts().reset_index()["breath_id"].unique()[0]}'
)



## === cell 9
display(test[test["breath_id"] == 0])



## === cell 10
train_gf = pd.read_csv(f"{DATA_DIR}/train.csv")
if not DEBUG:
    sample_vals = train_gf.sample(100_000, random_state=42).pressure.values
    plt.title("Histogram of Train Pressures", size=14)
    plt.hist(sample_vals, bins=100)
    plt.show()
print(
    "Max pressure =", train_gf.pressure.max(), "Min pressure =", train_gf.pressure.min()
)

all_pressure = np.sort(train_gf.pressure.unique())
del train_gf
gc.collect()

PRESSURE_MIN = all_pressure[0].item()
PRESSURE_MAX = all_pressure[-1].item()
PRESSURE_STEP = (all_pressure[1] - all_pressure[0]).item()

print(
    "PRESSURE_MIN:",
    PRESSURE_MIN,
    "PRESSURE_MAX:",
    PRESSURE_MAX,
    "PRESSURE_STEP:",
    PRESSURE_STEP,
)



## === cell 11
train["log_u_in"] = np.log1p(train.u_in)
test["log_u_in"] = np.log1p(test.u_in)

train["time_step_class"] = pd.qcut(train.time_step, q=80, labels=range(0, 80))
test["time_step_class"] = pd.qcut(test.time_step, q=80, labels=range(0, 80))

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

if DEBUG:
    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.grid()
    plt.xlabel("n_components")
    plt.ylabel("explained_variance_ratio_")
    plt.xticks([0, 1])
    plt.show()

train_pca = pca.transform(piv)
test_pca = pca.transform(piv_test)

train_pca = pd.DataFrame(
    train_pca, columns=["c" + str(c) for c in range(2)], index=piv.index
)
test_pca = pd.DataFrame(
    test_pca, columns=["c" + str(c) for c in range(2)], index=piv_test.index
)

km = KMeans(
    n_clusters=5, random_state=42, max_iter=200, init="k-means++", tol=0.0001, n_init=10
)
y_km = km.fit_predict(train_pca)
y_km_test = km.predict(test_pca)

train_pca["cluster"] = y_km
test_pca["cluster"] = y_km_test



## === cell 12
train_pca["breath_id"] = train_pca.index
train_pca.drop(["c0", "c1"], axis=1, inplace=True)
train_pca = train_pca.reset_index(drop=True)
train = pd.merge(train, train_pca, how="left", on="breath_id")

test_pca["breath_id"] = test_pca.index
test_pca.drop(["c0", "c1"], axis=1, inplace=True)
test_pca = test_pca.reset_index(drop=True)
test = pd.merge(test, test_pca, how="left", on="breath_id")




## === cell 13
def find_cluster_r_c(df):
    fig, ax = plt.subplots(5, 2, figsize=(15, 10))
    for c in range(5):
        for r_c in range(2):
            x = df.loc[df.cluster == c, "R" if r_c == 0 else "C"]
            sns.countplot(x=x, ax=ax[c][r_c])
            ax[c][r_c].set_title(f"Cluster={c}")
    plt.tight_layout()


def find_cluster_transition(df, is_train=True):
    fig, ax = plt.subplots(5, 5, figsize=(15, 10))
    for c in range(5):
        x = df.loc[df.cluster == c]
        breath = x.breath_id.unique()
        for n in range(min(5, len(breath))):
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
    sns.countplot(x=train["cluster"])
    plt.show()
    sns.countplot(x=test["cluster"])
    plt.show()
    find_cluster_r_c(train)
    find_cluster_r_c(test)
    find_cluster_transition(train)
    find_cluster_transition(test, False)



## === cell 14
train.drop(["time_step_class", "log_u_in"], axis=1, inplace=True)
test.drop(["time_step_class", "log_u_in"], axis=1, inplace=True)




## === cell 15
def add_features(dff):
    df = dff.copy()
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()
    df["u_in_cumsum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()

    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1)
    df["u_out_lag1"] = df.groupby("breath_id")["u_out"].shift(1)
    df["u_in_lag_back1"] = df.groupby("breath_id")["u_in"].shift(-1)
    df["u_out_lag_back1"] = df.groupby("breath_id")["u_out"].shift(-1)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2)
    df["u_out_lag2"] = df.groupby("breath_id")["u_out"].shift(2)
    df["u_in_lag_back2"] = df.groupby("breath_id")["u_in"].shift(-2)
    df["u_in_lag3"] = df.groupby("breath_id")["u_in"].shift(3)
    df["u_in_lag_back3"] = df.groupby("breath_id")["u_in"].shift(-3)
    df["u_in_lag8"] = df.groupby("breath_id")["u_in"].shift(8)
    df["u_in_lag_back8"] = df.groupby("breath_id")["u_in"].shift(-8)
    df["u_out_lag_back8"] = df.groupby("breath_id")["u_out"].shift(-8)
    df["time_step_diff3"] = df.groupby("breath_id")["time_step"].diff(3)
    df["u_in_pct"] = df.groupby("breath_id")["u_in"].pct_change()
    df["u_in_pct10"] = df.groupby("breath_id")["u_in"].pct_change(10)
    df["u_in_rolling8"] = list(
        df.groupby("breath_id")["u_in"].rolling(window=8).mean().fillna(0)
    )
    df["u_out_rolling8"] = list(
        df.groupby("breath_id")["u_out"].rolling(window=8).mean().fillna(0)
    )
    df["u_in_rolling4"] = list(
        df.groupby("breath_id")["u_in"].rolling(window=4).mean().fillna(0)
    )
    df["u_in_expanding5"] = list(
        df.groupby("breath_id")["u_in"].expanding(5).mean().fillna(0)
    )
    df["u_in_expanding2"] = list(
        df.groupby("breath_id")["u_in"].expanding(2).mean().fillna(0)
    )
    df = df.fillna(0)
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
    df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]
    df["breath_id__u_in__max"] = df.groupby(["breath_id"])["u_in"].transform("max")
    df["breath_id__u_in__diffmax"] = (
        df.groupby(["breath_id"])["u_in"].transform("max") - df["u_in"]
    )
    df["breath_id__u_in__diffmean"] = (
        df.groupby(["breath_id"])["u_in"].transform("mean") - df["u_in"]
    )

    df["u_in_change"] = df.groupby("breath_id")["u_in"].diff(-1)
    df["delta_time"] = df.groupby("breath_id")["time_step"].diff(-1)
    df["area_u_in"] = df["u_in"] * df["delta_time"]
    df["area_u_in_abs"] = df["u_in_change"] * df["delta_time"]
    df["uin_in_time"] = df["u_in_change"] / df["delta_time"]

    cvc = df["cluster"].value_counts()
    df["cvc"] = df["cluster"].apply(lambda x: cvc[x])
    df["cluster__u_in__diffmean"] = (
        df.groupby(["cluster"])["u_in"].transform("mean") - df["u_in"]
    )

    df["mean_RC"] = df.groupby(["R", "C"])["u_in"].transform("mean") - df["u_in"]
    df["max_RC"] = df.groupby(["R", "C"])["u_in"].transform("max") - df["u_in"]
    df["mean_RC_u_in_rolling8_mean"] = list(
        df.groupby("breath_id")["mean_RC"].rolling(window=8).mean().fillna(0)
    )
    df["max_RC_u_in_rolling8_max"] = list(
        df.groupby("breath_id")["max_RC"].rolling(window=8).mean().fillna(0)
    )

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["R__C"] = df["R"].astype(str) + "__" + df["C"].astype(str)
    df["cluster"] = df["cluster"].astype(str)

    df = pd.get_dummies(df)
    df = df.fillna(0)
    return df


train_ = add_features(train)
test_ = add_features(test)



## === cell 16
targets = train_[["pressure"]].to_numpy().reshape(-1, 80)

train_.drop(["pressure", "id", "breath_id"], axis=1, inplace=True)
test_ = test_.drop(["id", "breath_id"], axis=1)

train_.replace([np.inf, -np.inf], 0, inplace=True)
test_.replace([np.inf, -np.inf], 0, inplace=True)

for col in [c for c in train_.columns if train_[c].dtype == "float64"]:
    train_[col] = train_[col].astype("float32")



## === cell 17
RS = RobustScaler()
train_ = RS.fit_transform(train_)
test_ = RS.transform(test_)

train_ = train_.reshape(-1, 80, train_.shape[-1])
test_ = test_.reshape(-1, 80, train_.shape[-1])

np.savez_compressed("gbvpp_reshaped_tt", a=train_, b=test_)

print(
    "train_ reshaped:",
    train_.shape,
    "targets:",
    targets.shape,
    "test_ reshaped:",
    test_.shape,
)



## === cell 18
set_seed(23)

BATCH_SIZE = 1024
NUM_FOLDS = 10
EPOCHS = 300

if DEBUG:
    EPOCHS = 3
    test_ = test_[: 80 * 100]
    NUM_FOLDS = 2


def fit_lstm(train_arr, test_arr):
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)
    test_preds = []
    train_preds = train[["id", "breath_id", "pressure"]].copy()
    train_preds.loc[:, "modified_breath_id"] = [
        i for i in range(len(train_arr)) for _ in range(80)
    ]

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_arr, targets)):
        K.clear_session()
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)
        X_train, X_valid = train_arr[train_idx], train_arr[valid_idx]
        y_train, y_valid = targets[train_idx], targets[valid_idx]

        checkpoint_filepath = f"folds{fold}.hdf5"

        if TRAIN_MODEL:
            model = keras.models.Sequential(
                [
                    keras.layers.Input(shape=train_arr.shape[-2:]),
                    Bidirectional(LSTM(1024, return_sequences=True)),
                    Bidirectional(LSTM(512, return_sequences=True)),
                    Bidirectional(LSTM(256, return_sequences=True)),
                    Bidirectional(LSTM(128, return_sequences=True)),
                    Dense(128, activation="selu"),
                    Dense(1),
                ]
            )
            model.compile(optimizer="adam", loss="mae")
            if fold == 0:
                model.summary()

            lr = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, verbose=0
            )
            es = EarlyStopping(
                monitor="val_loss",
                patience=60,
                verbose=2,
                mode="min",
                restore_best_weights=True,
            )
            sv = ModelCheckpoint(
                checkpoint_filepath,
                monitor="val_loss",
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode="min",
                save_freq="epoch",
            )

            model.fit(
                X_train,
                y_train,
                validation_data=(X_valid, y_valid),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
                callbacks=[lr, es, sv, TqdmCallback(verbose=0)],
            )
        else:
            model = keras.models.load_model(
                "../input/finetune-of-tensorflow-bidirectional-lstm/"
                + checkpoint_filepath
            )

        test_preds.append(
            model.predict(test_arr, batch_size=BATCH_SIZE, verbose=2).ravel()
        )
        train_preds.loc[
            train_preds.loc[:, "modified_breath_id"].isin(valid_idx), "pressure"
        ] = model.predict(X_valid, batch_size=BATCH_SIZE, verbose=2).ravel()

        del X_train, X_valid, y_train, y_valid
        gc.collect()

    return test_preds, train_preds




## === cell 19
gc.collect()
set_seed(23)

test_preds, train_preds = fit_lstm(train_, test_)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/235330818.py in <cell line: 0>()
      2 set_seed(23)
      3 
----> 4 test_preds, train_preds = fit_lstm(train_, test_)
      5 

/tmp/ipykernel_11/2179133444.py in fit_lstm(train_arr, test_arr)
     53                 restore_best_weights=True,
     54             )
---> 55             sv = ModelCheckpoint(
     56                 checkpoint_filepath,
     57                 monitor="val_loss",

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    192                 self.filepath.endswith(ext) for ext in (".keras", ".h5")
    193             ):
--> 194                 raise ValueError(
    195                     "The filepath provided must end in `.keras` "
    196                     "(Keras model format). Received: "

ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=folds0.hdf5

## === cell 20
pred_mean = sum(test_preds) / NUM_FOLDS
pred_median = np.median(np.vstack(test_preds), axis=0)

submission_mean = submission.copy()
submission_mean["pressure"] = pred_mean
submission_mean.to_csv("submission_mean.csv", index=False)

submission_median = submission.copy()
submission_median["pressure"] = pred_median
submission_median.to_csv("submission_median.csv", index=False)

submission_final = submission.copy()
submission_final["pressure"] = pred_median
submission_final["pressure"] = (
    np.round((submission_final.pressure - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
    + PRESSURE_MIN
)
submission_final["pressure"] = np.clip(
    submission_final.pressure, PRESSURE_MIN, PRESSURE_MAX
)

submission_final.to_csv("submission.csv", index=False)
submission_final.to_csv("submission_median_round.csv", index=False)

test = test.copy()
test["pred1"] = pred_median
test["pred1"] = (
    np.round((test.pred1 - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
test["pred1"] = np.clip(test.pred1, PRESSURE_MIN, PRESSURE_MAX)

train_preds[["id", "pressure"]].to_csv("train_p2.csv", index=False)
test[["id", "pred1"]].to_csv("test_p2.csv", index=False)

print(
    "Wrote submission.csv with columns:",
    submission_final.columns.tolist(),
    "shape:",
    submission_final.shape,
)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1448523426.py in <cell line: 0>()
      1 # Ensemble + rounding to known pressure grid (as in original)
----> 2 pred_mean = sum(test_preds) / NUM_FOLDS
      3 pred_median = np.median(np.vstack(test_preds), axis=0)
      4 
      5 submission_mean = submission.copy()

NameError: name 'test_preds' is not defined

## === cell 21
plt.title("Histogram of Test Predicted Pressures (submission.csv)", size=14)
plt.hist(submission_final.sample(10_000, random_state=42).pressure.values, bins=100)
plt.show()
print(
    "Max pressure =",
    submission_final.pressure.max(),
    "Min pressure =",
    submission_final.pressure.min(),
)

print("Elapsed seconds:", time.time() - start_time)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4063063314.py in <cell line: 0>()
      4 # Quick sanity checks:
      5 plt.title("Histogram of Test Predicted Pressures (submission.csv)", size=14)
----> 6 plt.hist(submission_final.sample(10_000, random_state=42).pressure.values, bins=100)
      7 plt.show()
      8 print(

NameError: name 'submission_final' is not defined
