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

from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
from numpy.random import seed
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
from sklearn.model_selection import KFold, GroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error as mae
from scipy import stats
from scipy.stats import pearsonr
import scipy as sc

import lightgbm as lgb

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import *
    from tensorflow.keras.layers import *
    from tensorflow.keras.callbacks import *
    from tensorflow.keras.optimizers.schedules import ExponentialDecay
    from tqdm.keras import TqdmCallback
    from tensorflow.keras.backend import sigmoid
    from tensorflow.keras.utils import get_custom_objects
    from tensorflow.keras.layers import Activation

    def swish(x, beta=1):
        return x * sigmoid(beta * x)

    get_custom_objects().update({"swish": Activation(swish)})
except Exception:
    tf = None
    keras = None

    class DummyK:
        @staticmethod
        def clear_session():
            pass

    K = DummyK()

    class DummyCallback:
        pass

    TqdmCallback = DummyCallback

from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.matlib

import pickle
from pickle import dump, load

import warnings

warnings.filterwarnings("ignore")
pd.set_option("max_columns", 300)
sns.set_style("darkgrid")
sns.set_palette("dark")
gc.enable()


def set_seed(seeed):
    seed(seeed)
    random.seed(seeed)
    np.random.seed(seeed)
    if tf is not None:
        tf.random.set_seed(seeed)
    os.environ["PYTHONHASHSEED"] = str(seeed)


start_time = time.time()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
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
        except:
            print("WARNING: No TPU detected.")
            mirrored_strategy = tf.distribute.MirroredStrategy()
            return None, mirrored_strategy


cluster_resolver, strategy = connect_to_tpu()



## === cell 2
DEBUG = False
TRAIN_MODEL = True



## === cell 3
train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

if DEBUG:
    train = train[: 80 * 1000]
    test = test[: 80 * 100]
    submission = submission[: 80 * 100]



## === cell 4
print("TRAIN\n")
display(train)
print("\n\nTEST\n")
display(test)



## === cell 5
print(f"Length of TRAIN dataset: {len(train)}")
print(f"Length of TEST dataset: {len(test)}")
print(f'Number of breaths in train dataset: {train["breath_id"].nunique()}')
print(f'Number of breaths in test dataset: {test["breath_id"].nunique()}')
print(
    f'The number of observations for each breath: {train["breath_id"].value_counts().reset_index()["breath_id"].unique()[0]}'
)



## === cell 6
display(test[test["breath_id"] == 0])



## === cell 7
train_gf = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
plt.title("Histogram of Train Pressures", size=14)
plt.hist(train_gf.sample(100_000).pressure.values, bins=100)
plt.show()
print(
    "Max pressure =", train_gf.pressure.max(), "Min pressure =", train_gf.pressure.min()
)



## === cell 8
all_pressure = np.sort(train_gf.pressure.unique())
del train_gf
print("The first 25 unique pressures...")
PRESSURE_MIN = all_pressure[0].item()
PRESSURE_MAX = all_pressure[-1].item()
all_pressure[:25]



## === cell 9
print("The differences between first 25 pressures...")
PRESSURE_STEP = (all_pressure[1] - all_pressure[0]).item()
all_pressure[1:26] - all_pressure[:25]



## === cell 10
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
piv.head()



## === cell 11
pca = PCA(n_components=2, random_state=42)
pca.fit(piv)

plt.plot(pca.explained_variance_ratio_.cumsum())
plt.grid()
plt.xlabel("n_components")
plt.ylabel("explained_variance_ratio_")
plt.xticks([0, 1])
plt.show()



## === cell 12
train_pca = pca.transform(piv)
test_pca = pca.transform(piv_test)

train_pca = pd.DataFrame(
    train_pca, columns=["c" + str(c) for c in range(2)], index=piv.index
)
test_pca = pd.DataFrame(
    test_pca, columns=["c" + str(c) for c in range(2)], index=piv_test.index
)
train_pca.head()



## === cell 13
sns.scatterplot(data=train_pca, x="c0", y="c1")
plt.show()



## === cell 14
km = KMeans(n_clusters=5, random_state=42, max_iter=200, init="k-means++", tol=0.0001)
y_km = km.fit_predict(train_pca)
y_km_test = km.predict(test_pca)



## === cell 15
train_pca["cluster"] = y_km
test_pca["cluster"] = y_km_test

center = km.cluster_centers_
sns.scatterplot(data=train_pca, x="c0", y="c1", hue="cluster")
for i in range(5):
    plt.plot(center[i, 0], center[i, 1], "ro")
plt.show()



## === cell 16
train_pca["breath_id"] = train_pca.index
train_pca.drop(["c0", "c1"], axis=1, inplace=True)
train_pca = train_pca.reset_index(drop=True)
train = pd.merge(train, train_pca, how="left", on="breath_id")

test_pca["breath_id"] = test_pca.index
test_pca.drop(["c0", "c1"], axis=1, inplace=True)
test_pca = test_pca.reset_index(drop=True)
test = pd.merge(test, test_pca, how="left", on="breath_id")


def find_cluster_r_c(df):
    fig, ax = plt.subplots(5, 2, figsize=(15, 10))
    for c in range(5):
        for r_c in range(2):
            col = "R" if r_c == 0 else "C"
            mask = df["cluster"] == c
            x = df.loc[mask, col]
            sns.countplot(x=x, ax=ax[c][r_c])
            ax[c][r_c].set_title(f"Cluster={c}")
    plt.tight_layout()


def find_cluster_transition(df, is_train=True):
    fig, ax = plt.subplots(5, 5, figsize=(15, 10))
    for c in range(5):
        x = df.loc[df.cluster == c]
        breath = x.breath_id.unique()
        for n in range(5):
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




## === cell 17
sns.countplot(train["cluster"])



## === cell 18
sns.countplot(test.cluster)



## === cell 19
find_cluster_r_c(train)



## === cell 20
find_cluster_r_c(test)



## === cell 21
find_cluster_transition(train)



## === cell 22
find_cluster_transition(test, False)



## === cell 23
train.drop(["time_step_class", "log_u_in"], axis=1, inplace=True)
test.drop(["time_step_class", "log_u_in"], axis=1, inplace=True)



## === cell 24
display(test)
print(test.shape)
display(train)
print(train.shape)




## === cell 25
def add_features(dff):
    df = dff.copy()
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()
    df["u_in_cumsum"] = df["u_in"].groupby(df["breath_id"]).cumsum()

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
    df["R__C"] = df["R"] + "__" + df["C"]
    df["cluster"] = df["cluster"].astype(str)
    df = pd.get_dummies(df)
    df = df.fillna(0)
    return df


train_ = add_features(train)
test_ = add_features(test)



## === cell 26
display(train_.head())
print(train_.shape)
display(test_)
print(test_.shape)




## === cell 27
def getDuplicateColumns(df):
    duplicateColumnNames = set()
    for x in range(df.shape[1]):
        col = df.iloc[:, x]
        for y in range(x + 1, df.shape[1]):
            otherCol = df.iloc[:, y]
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
    return list(duplicateColumnNames)




## === cell 28
dc = getDuplicateColumns(train_)
dc



## === cell 29
targets = train_[["pressure"]].to_numpy().reshape(-1, 80)
train_.drop(["pressure", "id", "breath_id"], axis=1, inplace=True)
test_ = test_.drop(["id", "breath_id"], axis=1)

train_.replace([np.inf, -np.inf], 0, inplace=True)
test_.replace([np.inf, -np.inf], 0, inplace=True)



## === cell 30
for col in [c for c in train_.columns if train_[c].dtype == "float64"]:
    train_[col] = train_[col].astype("float32")



## === cell 31
RS = RobustScaler()
train_ = RS.fit_transform(train_)
test_ = RS.transform(test_)



## === cell 32
train_ = train_.reshape(-1, 80, train_.shape[-1])
test_ = test_.reshape(-1, 80, train_.shape[-1])



## === cell 33
np.savez_compressed("gbvpp_reshaped_tt", a=train_, b=test_)



## === cell 34
set_seed(23)

BATCH_SIZE = 1024
NUM_FOLDS = 10
EPOCHS = 300

if DEBUG:
    EPOCHS = 3
    test_ = test_[: 80 * 100]
    NUM_FOLDS = 2


def fit_lgb(train_, test_, targets):
    """Train LightGBM with K-Fold and return list of test predictions."""
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)
    test_preds = []
    X_test_flat = test_.reshape(-1, test_.shape[-1])
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_, targets)):
        print(f"Fold {fold+1}/{NUM_FOLDS}")
        X_tr = train_[train_idx].reshape(-1, train_.shape[-1])
        y_tr = targets[train_idx].reshape(-1)
        X_val = train_[val_idx].reshape(-1, train_.shape[-1])
        y_val = targets[val_idx].reshape(-1)

        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "verbosity": -1,
            "seed": 23,
        }

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_val],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        pred = model.predict(X_test_flat, num_iteration=model.best_iteration)
        test_preds.append(pred)
        del X_tr, X_val, y_tr, y_val, lgb_train, lgb_val, model
        gc.collect()
    return test_preds




## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2069788471.py in <cell line: 0>()
----> 1 set_seed(23)
      2 
      3 BATCH_SIZE = 1024
      4 NUM_FOLDS = 10
      5 EPOCHS = 300

NameError: name 'set_seed' is not defined

## === cell 35
gc.collect()



## === cell 36
set_seed(23)
test_preds = fit_lgb(train_, test_, targets)



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/41297089.py in <cell line: 0>()
----> 1 set_seed(23)
      2 test_preds = fit_lgb(train_, test_, targets)
      3 

NameError: name 'set_seed' is not defined

## === cell 37
submission["pressure"] = np.mean(np.vstack(test_preds), axis=0)
submission["pressure"] = (
    np.round((submission.pressure - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
    + PRESSURE_MIN
)
submission.pressure = np.clip(submission.pressure, PRESSURE_MIN, PRESSURE_MAX)
submission.to_csv("submission.csv", index=False)

print("Submission file written to 'submission.csv'.")

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/460439510.py in <cell line: 0>()
----> 1 submission["pressure"] = np.mean(np.vstack(test_preds), axis=0)
      2 submission["pressure"] = (
      3     np.round((submission.pressure - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
      4     + PRESSURE_MIN
      5 )

NameError: name 'test_preds' is not defined
