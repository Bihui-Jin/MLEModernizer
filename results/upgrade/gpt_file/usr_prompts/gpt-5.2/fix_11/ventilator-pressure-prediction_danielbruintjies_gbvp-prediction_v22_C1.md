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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import time
import numpy as np
import pandas as pd
import math
import random
import gc
import warnings

warnings.filterwarnings("ignore")

from numpy.random import seed
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Activation
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects

TQDM_CALLBACK = None


def swish(x, beta=1):
    return x * sigmoid(beta * x)


get_custom_objects().update({"swish": Activation(swish)})

pd.set_option("display.max_columns", 300)
sns.set_style("darkgrid")
sns.set_palette("dark")
gc.enable()


def set_seed(seeed: int):
    seed(seeed)
    tf.random.set_seed(seeed)
    os.environ["PYTHONHASHSEED"] = str(seeed)
    np.random.seed(seeed)
    random.seed(seeed)


start_time = time.time()




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
        except Exception:
            print("WARNING: No TPU detected. Using MirroredStrategy/CPU-GPU.")
            mirrored_strategy = tf.distribute.MirroredStrategy()
            return None, mirrored_strategy


cluster_resolver, strategy = connect_to_tpu()



## === cell 2
DEBUG = False

TRAIN_MODEL = False

PRETRAIN_DIR = "../input/finetune-of-tensorflow-bidirectional-lstm"
if (not TRAIN_MODEL) and (not os.path.isdir(PRETRAIN_DIR)):
    print(f"WARNING: Pretrained directory not found: {PRETRAIN_DIR}")
    print("Switching TRAIN_MODEL=True to train model in this run.")
    TRAIN_MODEL = True



## === cell 3
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
submission = pd.read_csv(SUB_PATH)

if DEBUG:
    train = train[: 80 * 1000]
    test = test[: 80 * 100]
    submission = submission[: 80 * 100]



## === cell 4
print("TRAIN head():\n", train.head())
print("\nTEST head():\n", test.head())



## === cell 5
print(f"Length of TRAIN dataset: {len(train)}")
print(f"Length of TEST dataset: {len(test)}")
print(f"Number of breaths in train dataset: {train['breath_id'].nunique()}")
print(f"Number of breaths in test dataset: {test['breath_id'].nunique()}")
print(
    f"The number of observations for each breath: {train['breath_id'].value_counts().iloc[0]}"
)



## === cell 6
print("Rows with test breath_id==0:", len(test[test["breath_id"] == 0]))



## === cell 7
if DEBUG:
    plt.title("Histogram of Train Pressures", size=14)
    plt.hist(train.sample(100_000, random_state=0).pressure.values, bins=100)
    plt.show()
print("Max pressure =", train.pressure.max(), "Min pressure =", train.pressure.min())

all_pressure = np.sort(train.pressure.unique())

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




## === cell 8
def _compute_breath_matrix_log_uin(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    gb_size = df.groupby("breath_id", sort=False).size()
    breath_ids = gb_size.index.to_numpy()
    counts = gb_size.to_numpy()
    if not np.all(counts == 80):
        raise ValueError("Expected exactly 80 rows per breath.")
    mat = np.log1p(df["u_in"].to_numpy(dtype=np.float32, copy=False)).reshape(-1, 80)
    return breath_ids, mat


train_breath_ids, train_mat = _compute_breath_matrix_log_uin(train)
test_breath_ids, test_mat = _compute_breath_matrix_log_uin(test)

pca = PCA(n_components=2, random_state=42)
pca.fit(train_mat)
train_pca_arr = pca.transform(train_mat)
test_pca_arr = pca.transform(test_mat)

train_pca = pd.DataFrame(train_pca_arr, columns=["c0", "c1"])
test_pca = pd.DataFrame(test_pca_arr, columns=["c0", "c1"])

km = KMeans(
    n_clusters=5,
    random_state=42,
    max_iter=200,
    init="k-means++",
    tol=0.0001,
)
y_km = km.fit_predict(train_pca[["c0", "c1"]])
y_km_test = km.predict(test_pca[["c0", "c1"]])

train_pca["cluster"] = y_km
test_pca["cluster"] = y_km_test

train_pca["breath_id"] = train_breath_ids
test_pca["breath_id"] = test_breath_ids

train = pd.merge(train, train_pca[["breath_id", "cluster"]], how="left", on="breath_id")
test = pd.merge(test, test_pca[["breath_id", "cluster"]], how="left", on="breath_id")


def find_cluster_r_c(df):
    fig, ax = plt.subplots(5, 2, figsize=(15, 10))
    for c in range(5):
        for r_c in range(2):
            colname = "R" if r_c == 0 else "C"
            tmp = df.loc[df.cluster == c, [colname]].copy()
            sns.countplot(data=tmp, x=colname, ax=ax[c][r_c])
            ax[c][r_c].set_title(f"Cluster={c}")
    plt.tight_layout()


def find_cluster_transition(df, is_train=True):
    fig, ax = plt.subplots(5, 5, figsize=(15, 10))
    for c in range(5):
        x = df.loc[df.cluster == c]
        breath = x.breath_id.unique()
        if len(breath) == 0:
            continue
        for n in range(5):
            if n >= len(breath):
                break
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
    sns.countplot(data=train, x="cluster")
    plt.show()
    sns.countplot(data=test, x="cluster")
    plt.show()
    find_cluster_r_c(train)
    find_cluster_r_c(test)
    find_cluster_transition(train)
    find_cluster_transition(test, False)



## === cell 9
for col in ["time_step_class", "log_u_in"]:
    if col in train.columns:
        train.drop([col], axis=1, inplace=True)
    if col in test.columns:
        test.drop([col], axis=1, inplace=True)

print("test shape:", test.shape)
print("train shape:", train.shape)




## === cell 10
def add_features(dff):
    df = dff  # shallow reference; we never mutate original columns in-place below

    gb = df.groupby("breath_id", sort=False)

    u_in_roll8 = gb["u_in"].rolling(window=8).mean().reset_index(level=0, drop=True)
    u_out_roll8 = gb["u_out"].rolling(window=8).mean().reset_index(level=0, drop=True)
    u_in_roll4 = gb["u_in"].rolling(window=4).mean().reset_index(level=0, drop=True)
    u_in_exp5 = gb["u_in"].expanding(5).mean().reset_index(level=0, drop=True)
    u_in_exp2 = gb["u_in"].expanding(2).mean().reset_index(level=0, drop=True)
    u_in_roll8_max = gb["u_in"].rolling(window=8).max().reset_index(level=0, drop=True)

    out = pd.DataFrame(index=df.index)
    out["R"] = df["R"]
    out["C"] = df["C"]
    out["cluster"] = df["cluster"]
    out["time_step"] = df["time_step"]
    out["u_in"] = df["u_in"]
    out["u_out"] = df["u_out"]

    out["area"] = (
        (df["time_step"] * df["u_in"]).groupby(df["breath_id"], sort=False).cumsum()
    )
    out["u_in_cumsum"] = gb["u_in"].cumsum()

    out["u_in_lag1"] = gb["u_in"].shift(1)
    out["u_out_lag1"] = gb["u_out"].shift(1)
    out["u_in_lag_back1"] = gb["u_in"].shift(-1)
    out["u_out_lag_back1"] = gb["u_out"].shift(-1)
    out["u_in_lag2"] = gb["u_in"].shift(2)
    out["u_out_lag2"] = gb["u_out"].shift(2)
    out["u_in_lag_back2"] = gb["u_in"].shift(-2)
    out["u_in_lag3"] = gb["u_in"].shift(3)
    out["u_in_lag_back3"] = gb["u_in"].shift(-3)
    out["u_in_lag8"] = gb["u_in"].shift(8)
    out["u_in_lag_back8"] = gb["u_in"].shift(-8)
    out["u_out_lag_back8"] = gb["u_out"].shift(-8)

    out["time_step_diff3"] = gb["time_step"].diff(3)
    out["u_in_pct"] = gb["u_in"].pct_change()
    out["u_in_pct10"] = gb["u_in"].pct_change(10)

    out["u_in_rolling8"] = u_in_roll8.fillna(0)
    out["u_out_rolling8"] = u_out_roll8.fillna(0)
    out["u_in_rolling4"] = u_in_roll4.fillna(0)
    out["u_in_expanding5"] = u_in_exp5.fillna(0)
    out["u_in_expanding2"] = u_in_exp2.fillna(0)

    out = out.fillna(0)

    out["u_in_diff1"] = out["u_in"] - out["u_in_lag1"]
    out["u_out_diff1"] = out["u_out"] - out["u_out_lag1"]
    out["u_in_diff2"] = out["u_in"] - out["u_in_lag2"]
    out["u_out_diff2"] = out["u_out"] - out["u_out_lag2"]

    uin_max = gb["u_in"].transform("max")
    uin_mean = gb["u_in"].transform("mean")
    out["breath_id__u_in__max"] = uin_max
    out["breath_id__u_in__diffmax"] = uin_max - out["u_in"]
    out["breath_id__u_in__diffmean"] = uin_mean - out["u_in"]

    out["u_in_change"] = gb["u_in"].diff(-1)
    out["delta_time"] = gb["time_step"].diff(-1)
    out["area_u_in"] = out["u_in"] * out["delta_time"]
    out["area_u_in_abs"] = out["u_in_change"] * out["delta_time"]
    out["uin_in_time"] = out["u_in_change"] / out["delta_time"]

    cvc = out["cluster"].value_counts()
    out["cvc"] = out["cluster"].map(cvc).astype(np.int32)

    out["cluster__u_in__diffmean"] = (
        out.groupby(["cluster"], sort=False)["u_in"].transform("mean") - out["u_in"]
    )
    out["mean_RC"] = (
        out.groupby(["R", "C"], sort=False)["u_in"].transform("mean") - out["u_in"]
    )
    out["max_RC"] = (
        out.groupby(["R", "C"], sort=False)["u_in"].transform("max") - out["u_in"]
    )

    out["mean_RC_u_in_rolling8_mean"] = out["u_in_rolling8"] - out["u_in"]
    out["max_RC_u_in_rolling8_max"] = u_in_roll8_max.fillna(0) - out["u_in"]

    out["R"] = out["R"].astype("category")
    out["C"] = out["C"].astype("category")
    out["R__C"] = (out["R"].astype(str) + "__" + out["C"].astype(str)).astype(
        "category"
    )
    out["cluster"] = out["cluster"].astype("category")

    out = pd.get_dummies(out, dtype=np.int8)
    out = out.fillna(0)
    return out


train_ = add_features(train)
test_ = add_features(test)

train_, test_ = train_.align(test_, join="left", axis=1, fill_value=0)

print("train_ shape:", train_.shape, "test_ shape:", test_.shape)



## === cell 11
breath_ids_train = train.groupby("breath_id", sort=False).size().index.to_numpy()
counts_train = train.groupby("breath_id", sort=False).size().to_numpy()
if not np.all(counts_train == 80):
    raise ValueError("Expected exactly 80 rows per breath for all breaths (train).")

breath_ids_test = test.groupby("breath_id", sort=False).size().index.to_numpy()
counts_test = test.groupby("breath_id", sort=False).size().to_numpy()
if not np.all(counts_test == 80):
    raise ValueError("Expected exactly 80 rows per breath for all breaths (test).")

targets = train["pressure"].to_numpy(dtype=np.float32).reshape(-1, 80)

train_.replace([np.inf, -np.inf], 0, inplace=True)
test_.replace([np.inf, -np.inf], 0, inplace=True)

X_train_flat = train_.to_numpy(dtype=np.float32, copy=False)
X_test_flat = test_.to_numpy(dtype=np.float32, copy=False)

RS = RobustScaler()
X_train_flat = RS.fit_transform(X_train_flat)
X_test_flat = RS.transform(X_test_flat)

n_features = X_train_flat.shape[1]
train_ = X_train_flat.reshape(-1, 80, n_features)
test_ = X_test_flat.reshape(-1, 80, n_features)

print(
    "train_ seq shape:",
    train_.shape,
    "test_ seq shape:",
    test_.shape,
    "targets shape:",
    targets.shape,
)



## === cell 12
set_seed(23)

BATCH_SIZE = 1024
NUM_FOLDS = 10
EPOCHS = 300

if DEBUG:
    EPOCHS = 3
    test_ = test_[: 80 * 100]
    NUM_FOLDS = 2


def fit_lstm(train_, test_):
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)
    test_preds = []

    train_preds = train[["id", "breath_id", "pressure"]].copy()

    breath_ids = train.groupby("breath_id", sort=False).size().index.to_numpy()
    if len(breath_ids) != train_.shape[0]:
        raise ValueError(
            f"Mismatch: number of breath_ids ({len(breath_ids)}) != train_ sequences ({train_.shape[0]})."
        )

    counts = train.groupby("breath_id", sort=False).size().to_numpy()
    if not np.all(counts == 80):
        raise ValueError("Expected exactly 80 rows per breath for all breaths.")
    offsets = np.arange(len(breath_ids), dtype=np.int64) * 80

    test_ds = (
        tf.data.Dataset.from_tensor_slices(test_)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    for fold, (train_idx, test_idx) in enumerate(kf.split(train_, targets)):
        K.clear_session()
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)
        X_train, X_valid = train_[train_idx], train_[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]

        checkpoint_filepath = f"folds{fold}.keras"

        if TRAIN_MODEL:
            with strategy.scope():
                model = keras.models.Sequential(
                    [
                        keras.layers.Input(shape=train_.shape[-2:]),
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
                mode="auto",
                save_freq="epoch",
            )

            callbacks = [lr, es, sv]
            if TQDM_CALLBACK is not None:
                callbacks.append(TQDM_CALLBACK)

            model.fit(
                X_train,
                y_train,
                validation_data=(X_valid, y_valid),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
                callbacks=callbacks,
            )
        else:
            model_path = os.path.join(PRETRAIN_DIR, checkpoint_filepath)
            model = keras.models.load_model(model_path)

        test_preds.append(model.predict(test_ds, verbose=0).ravel())

        valid_ds = (
            tf.data.Dataset.from_tensor_slices(X_valid)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )
        valid_pred = model.predict(valid_ds, verbose=0).ravel()

        row_idx = (
            offsets[test_idx][:, None] + np.arange(80, dtype=np.int64)[None, :]
        ).ravel()
        train_preds.iloc[row_idx, train_preds.columns.get_loc("pressure")] = valid_pred

        del X_train, X_valid, y_train, y_valid, valid_pred, valid_ds
        gc.collect()

    return test_preds, train_preds




## === cell 13
gc.collect()
set_seed(23)

test_preds, train_preds = fit_lstm(train_, test_)



## === cell 14
submission["pressure"] = np.median(np.vstack(test_preds), axis=0)

submission["pressure"] = (
    np.round((submission["pressure"] - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
    + PRESSURE_MIN
)
submission["pressure"] = np.clip(submission["pressure"], PRESSURE_MIN, PRESSURE_MAX)

submission[["id", "pressure"]].to_csv("submission.csv", index=False)
submission[["id", "pressure"]].to_csv("submission_median_round.csv", index=False)

train_preds[["id", "pressure"]].to_csv("train_p2.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(
    "Pressure min/max in submission:",
    submission["pressure"].min(),
    submission["pressure"].max(),
)



## === cell 15
if DEBUG:
    plt.title("Histogram of Test Pressures", size=14)
    plt.hist(submission.sample(10_000, random_state=0).pressure.values, bins=100)
    plt.show()

elapsed = time.time() - start_time
print(f"Done. Elapsed seconds: {elapsed:.1f}")
