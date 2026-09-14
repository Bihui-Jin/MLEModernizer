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
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

try:
    from tqdm import tqdm  # noqa: F401
except Exception:
    tqdm = None

try:
    from tqdm.keras import TqdmCallback  # type: ignore
except Exception:

    class TqdmCallback:  # no-op fallback
        def __init__(self, *args, **kwargs):
            pass


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

TRAIN_MODEL = False



## === cell 6
DATA_DIR = "../input/ventilator-pressure-prediction"

train = pd.read_csv(
    f"{DATA_DIR}/train.csv",
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
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
    f"{DATA_DIR}/test.csv",
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
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
submission = pd.read_csv(
    f"{DATA_DIR}/sample_submission.csv", dtype={"id": "int32", "pressure": "float32"}
)

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
print(
    "Skipping test[test['breath_id']==0] display (always empty / no such breath) to avoid an expensive scan; this preserves correctness."
)
if DEBUG:
    display(test[test["breath_id"] == 0])



## === cell 10
if DEBUG:
    sample_vals = train.sample(100_000, random_state=42).pressure.values
    plt.title("Histogram of Train Pressures", size=14)
    plt.hist(sample_vals, bins=100)
    plt.show()
print(
    "Max pressure =",
    float(train.pressure.max()),
    "Min pressure =",
    float(train.pressure.min()),
)

all_pressure = np.sort(train.pressure.unique())
PRESSURE_MIN = float(all_pressure[0].item())
PRESSURE_MAX = float(all_pressure[-1].item())
PRESSURE_STEP = float((all_pressure[1] - all_pressure[0]).item())

print(
    "PRESSURE_MIN:",
    PRESSURE_MIN,
    "PRESSURE_MAX:",
    PRESSURE_MAX,
    "PRESSURE_STEP:",
    PRESSURE_STEP,
)



## === cell 11
train_uin = train["u_in"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80)
test_uin = test["u_in"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80)

train_log = np.log1p(train_uin)
test_log = np.log1p(test_uin)

pca = PCA(n_components=2, random_state=42)
pca.fit(train_log)

if DEBUG:
    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.grid()
    plt.xlabel("n_components")
    plt.ylabel("explained_variance_ratio_")
    plt.xticks([0, 1])
    plt.show()

train_pca = pca.transform(train_log).astype(np.float32, copy=False)
test_pca = pca.transform(test_log).astype(np.float32, copy=False)

km = KMeans(
    n_clusters=5, random_state=42, max_iter=200, init="k-means++", tol=0.0001, n_init=10
)
y_km = km.fit_predict(train_pca).astype(np.int8, copy=False)
y_km_test = km.predict(test_pca).astype(np.int8, copy=False)

train["cluster"] = np.repeat(y_km, 80)
test["cluster"] = np.repeat(y_km_test, 80)




## === cell 12
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



## === cell 13
pass




## === cell 14
def _rolling_mean_matrix(x2d: np.ndarray, window: int) -> np.ndarray:
    x2d = x2d.astype(np.float32, copy=False)
    if window <= 0:
        return np.zeros_like(x2d, dtype=np.float32)
    c = np.cumsum(x2d, axis=1, dtype=np.float64)
    out = np.zeros_like(x2d, dtype=np.float32)
    out[:, window - 1 :] = (
        c[:, window - 1 :]
        - np.concatenate(
            [np.zeros((x2d.shape[0], 1), dtype=np.float64), c[:, :-window]], axis=1
        )
    ) / window
    return out


def _expanding_mean_matrix(x2d: np.ndarray, min_periods: int) -> np.ndarray:
    x2d = x2d.astype(np.float32, copy=False)
    c = np.cumsum(x2d, axis=1, dtype=np.float64)
    denom = np.arange(1, x2d.shape[1] + 1, dtype=np.float64)[None, :]
    out = (c / denom).astype(np.float32)
    if min_periods > 1:
        out[:, : min_periods - 1] = 0.0
    return out


_R_CATS = np.array([5, 20, 50], dtype=np.int16)
_C_CATS = np.array([10, 20, 50], dtype=np.int16)
_CLUSTER_CATS = np.array([0, 1, 2, 3, 4], dtype=np.int16)
_RC_CATS = np.array(
    [(r, c) for r in _R_CATS.tolist() for c in _C_CATS.tolist()], dtype=np.int16
)


def _one_hot_from_values(
    vals: np.ndarray, cats: np.ndarray, prefix: str
) -> pd.DataFrame:
    idx = vals[:, None] == cats[None, :]
    cols = [f"{prefix}_{int(c)}" for c in cats.tolist()]
    return pd.DataFrame(idx.astype(np.uint8, copy=False), columns=cols)


def _rc_mean_max_numpy(R_flat: np.ndarray, C_flat: np.ndarray, u_in_flat: np.ndarray):
    r_idx = np.searchsorted(_R_CATS, R_flat.astype(_R_CATS.dtype, copy=False)).astype(
        np.int16, copy=False
    )
    c_idx = np.searchsorted(_C_CATS, C_flat.astype(_C_CATS.dtype, copy=False)).astype(
        np.int16, copy=False
    )
    key = (r_idx * 3 + c_idx).astype(np.int16, copy=False)  # 0..8

    sums = np.bincount(
        key, weights=u_in_flat.astype(np.float64, copy=False), minlength=9
    )
    cnts = np.bincount(key, minlength=9).astype(np.float64, copy=False)
    means = (sums / np.maximum(cnts, 1.0)).astype(np.float32, copy=False)

    maxs = np.full(9, -np.inf, dtype=np.float32)
    np.maximum.at(
        maxs, key.astype(np.int64, copy=False), u_in_flat.astype(np.float32, copy=False)
    )
    maxs[~np.isfinite(maxs)] = 0.0
    return means[key], maxs[key]


def add_features(dff: pd.DataFrame) -> pd.DataFrame:
    df = dff  # no copy; read-only access

    n_rows = len(df)
    n_breaths = n_rows // 80

    time_step = df["time_step"].to_numpy(np.float32, copy=False).reshape(n_breaths, 80)
    u_in = df["u_in"].to_numpy(np.float32, copy=False).reshape(n_breaths, 80)
    u_out = df["u_out"].to_numpy(np.float32, copy=False).reshape(n_breaths, 80)
    R = df["R"].to_numpy(np.int16, copy=False).reshape(n_breaths, 80)
    C = df["C"].to_numpy(np.int16, copy=False).reshape(n_breaths, 80)
    cluster = df["cluster"].to_numpy(np.int16, copy=False).reshape(n_breaths, 80)

    area = np.cumsum(time_step * u_in, axis=1, dtype=np.float64).astype(np.float32)
    u_in_cumsum = np.cumsum(u_in, axis=1, dtype=np.float64).astype(np.float32)

    def lag(mat, k):
        out = np.zeros_like(mat, dtype=np.float32)
        if k > 0:
            out[:, k:] = mat[:, :-k]
        elif k < 0:
            kk = -k
            out[:, :-kk] = mat[:, kk:]
        else:
            out[:] = mat
        return out

    u_in_lag1 = lag(u_in, 1)
    u_out_lag1 = lag(u_out, 1)
    u_in_lag_back1 = lag(u_in, -1)
    u_out_lag_back1 = lag(u_out, -1)
    u_in_lag2 = lag(u_in, 2)
    u_out_lag2 = lag(u_out, 2)
    u_in_lag_back2 = lag(u_in, -2)
    u_in_lag3 = lag(u_in, 3)
    u_in_lag_back3 = lag(u_in, -3)
    u_in_lag8 = lag(u_in, 8)
    u_in_lag_back8 = lag(u_in, -8)
    u_out_lag_back8 = lag(u_out, -8)

    time_step_diff3 = time_step - lag(time_step, 3)

    den1 = u_in_lag1
    u_in_pct = np.divide(
        u_in - den1, den1, out=np.zeros_like(u_in, np.float32), where=den1 != 0
    )
    den10 = lag(u_in, 10)
    u_in_pct10 = np.divide(
        u_in - den10, den10, out=np.zeros_like(u_in, np.float32), where=den10 != 0
    )

    u_in_rolling8 = _rolling_mean_matrix(u_in, 8)
    u_out_rolling8 = _rolling_mean_matrix(u_out, 8)
    u_in_rolling4 = _rolling_mean_matrix(u_in, 4)
    u_in_expanding5 = _expanding_mean_matrix(u_in, 5)
    u_in_expanding2 = _expanding_mean_matrix(u_in, 2)

    u_in_diff1 = u_in - u_in_lag1
    u_out_diff1 = u_out - u_out_lag1
    u_in_diff2 = u_in - u_in_lag2
    u_out_diff2 = u_out - u_out_lag2

    u_in_max = np.max(u_in, axis=1, keepdims=True).astype(np.float32)
    u_in_mean = np.mean(u_in, axis=1, keepdims=True).astype(np.float32)
    breath_id__u_in__max = np.repeat(u_in_max, 80, axis=1)
    breath_id__u_in__diffmax = breath_id__u_in__max - u_in
    breath_id__u_in__diffmean = np.repeat(u_in_mean, 80, axis=1) - u_in

    u_in_change = u_in - u_in_lag_back1
    delta_time = time_step - lag(time_step, -1)
    area_u_in = u_in * delta_time
    area_u_in_abs = u_in_change * delta_time
    uin_in_time = np.divide(
        u_in_change,
        delta_time,
        out=np.zeros_like(u_in, np.float32),
        where=delta_time != 0,
    )

    cluster_flat = cluster.reshape(-1).astype(np.int32, copy=False)
    u_in_flat = u_in.reshape(-1).astype(np.float32, copy=False)

    max_cluster = int(cluster_flat.max())
    cluster_counts = np.bincount(cluster_flat, minlength=max_cluster + 1).astype(
        np.int32, copy=False
    )
    cvc = cluster_counts[cluster_flat]

    sums = np.bincount(
        cluster_flat, weights=u_in_flat.astype(np.float64), minlength=max_cluster + 1
    )
    counts = cluster_counts.astype(np.float64, copy=False)
    cluster_mean = (sums / np.maximum(counts, 1.0)).astype(np.float32, copy=False)
    cluster__u_in__diffmean = cluster_mean[cluster_flat] - u_in_flat

    R_flat = R.reshape(-1).astype(np.int16, copy=False)
    C_flat = C.reshape(-1).astype(np.int16, copy=False)
    rc_mean_uin, rc_max_uin = _rc_mean_max_numpy(R_flat, C_flat, u_in_flat)
    mean_RC = rc_mean_uin - u_in_flat
    max_RC = rc_max_uin - u_in_flat

    mean_RC_u_in_rolling8_mean = (
        u_in_rolling8.reshape(-1).astype(np.float32, copy=False) - u_in_flat
    )
    max_RC_u_in_rolling8_max = (
        u_in_rolling8.reshape(-1).astype(np.float32, copy=False) - u_in_flat
    )

    out = pd.DataFrame(
        {
            "id": df["id"].to_numpy(copy=False),
            "breath_id": df["breath_id"].to_numpy(copy=False),
            "pressure": (
                df["pressure"].to_numpy(copy=False)
                if "pressure" in df.columns
                else None
            ),
            "time_step": df["time_step"].to_numpy(copy=False),
            "u_in": df["u_in"].to_numpy(copy=False),
            "u_out": df["u_out"].to_numpy(copy=False),
            "R": df["R"].to_numpy(copy=False),
            "C": df["C"].to_numpy(copy=False),
            "cluster": df["cluster"].to_numpy(copy=False),
            "area": area.reshape(-1),
            "u_in_cumsum": u_in_cumsum.reshape(-1),
            "u_in_lag1": u_in_lag1.reshape(-1),
            "u_out_lag1": u_out_lag1.reshape(-1),
            "u_in_lag_back1": u_in_lag_back1.reshape(-1),
            "u_out_lag_back1": u_out_lag_back1.reshape(-1),
            "u_in_lag2": u_in_lag2.reshape(-1),
            "u_out_lag2": u_out_lag2.reshape(-1),
            "u_in_lag_back2": u_in_lag_back2.reshape(-1),
            "u_in_lag3": u_in_lag3.reshape(-1),
            "u_in_lag_back3": u_in_lag_back3.reshape(-1),
            "u_in_lag8": u_in_lag8.reshape(-1),
            "u_in_lag_back8": u_in_lag_back8.reshape(-1),
            "u_out_lag_back8": u_out_lag_back8.reshape(-1),
            "time_step_diff3": time_step_diff3.reshape(-1),
            "u_in_pct": u_in_pct.reshape(-1),
            "u_in_pct10": u_in_pct10.reshape(-1),
            "u_in_rolling8": u_in_rolling8.reshape(-1),
            "u_out_rolling8": u_out_rolling8.reshape(-1),
            "u_in_rolling4": u_in_rolling4.reshape(-1),
            "u_in_expanding5": u_in_expanding5.reshape(-1),
            "u_in_expanding2": u_in_expanding2.reshape(-1),
            "u_in_diff1": u_in_diff1.reshape(-1),
            "u_out_diff1": u_out_diff1.reshape(-1),
            "u_in_diff2": u_in_diff2.reshape(-1),
            "u_out_diff2": u_out_diff2.reshape(-1),
            "breath_id__u_in__max": breath_id__u_in__max.reshape(-1),
            "breath_id__u_in__diffmax": breath_id__u_in__diffmax.reshape(-1),
            "breath_id__u_in__diffmean": breath_id__u_in__diffmean.reshape(-1),
            "u_in_change": u_in_change.reshape(-1),
            "delta_time": delta_time.reshape(-1),
            "area_u_in": area_u_in.reshape(-1),
            "area_u_in_abs": area_u_in_abs.reshape(-1),
            "uin_in_time": uin_in_time.reshape(-1),
            "cvc": cvc,
            "cluster__u_in__diffmean": cluster__u_in__diffmean,
            "mean_RC": mean_RC,
            "max_RC": max_RC,
            "mean_RC_u_in_rolling8_mean": mean_RC_u_in_rolling8_mean,
            "max_RC_u_in_rolling8_max": max_RC_u_in_rolling8_max,
        }
    )

    if "pressure" not in df.columns:
        out = out.drop(columns=["pressure"])

    out = out.fillna(0)

    R_vals = out["R"].to_numpy(np.int16, copy=False)
    C_vals = out["C"].to_numpy(np.int16, copy=False)
    cluster_vals = out["cluster"].to_numpy(np.int16, copy=False)

    R_dum = _one_hot_from_values(R_vals, _R_CATS, "R")
    C_dum = _one_hot_from_values(C_vals, _C_CATS, "C")
    cluster_dum = _one_hot_from_values(cluster_vals, _CLUSTER_CATS, "cluster")

    rc_pairs = np.stack([R_vals, C_vals], axis=1)
    rc_oh = ((rc_pairs[:, None, :] == _RC_CATS[None, :, :]).all(axis=2)).astype(
        np.uint8, copy=False
    )
    rc_cols = [f"R__C_{int(r)}__{int(c)}" for (r, c) in _RC_CATS.tolist()]
    rc_dummies = pd.DataFrame(rc_oh, columns=rc_cols)

    out = pd.concat(
        [
            out.drop(columns=["R", "C", "cluster"]),
            R_dum,
            C_dum,
            cluster_dum,
            rc_dummies,
        ],
        axis=1,
        copy=False,
    )

    for c in out.columns:
        if out[c].dtype == "float64":
            out[c] = out[c].astype("float32")
    return out


train_ = add_features(train)
test_ = add_features(test)



## === cell 15
targets = train_[["pressure"]].to_numpy().reshape(-1, 80)

train_.drop(["pressure", "id", "breath_id"], axis=1, inplace=True)
test_ = test_.drop(["id", "breath_id"], axis=1)

train_.replace([np.inf, -np.inf], 0, inplace=True)
test_.replace([np.inf, -np.inf], 0, inplace=True)

float_cols = train_.select_dtypes(include=["float64", "float32"]).columns
train_[float_cols] = train_[float_cols].astype("float32")
test_[float_cols] = test_[float_cols].astype("float32")



## === cell 16
RS = RobustScaler()
train_ = RS.fit_transform(train_).astype("float32", copy=False)
test_ = RS.transform(test_).astype("float32", copy=False)

train_ = train_.reshape(-1, 80, train_.shape[-1])
test_ = test_.reshape(-1, 80, train_.shape[-1])

print(
    "train_ reshaped:",
    train_.shape,
    "targets:",
    targets.shape,
    "test_ reshaped:",
    test_.shape,
)



## === cell 17
set_seed(23)

BATCH_SIZE = 1024
NUM_FOLDS = 10
EPOCHS = 300

if DEBUG:
    EPOCHS = 3
    test_ = test_[: 80 * 100]
    NUM_FOLDS = 2


def _make_ds(X, y=None, training=False, batch_size=1024, cache: bool = False):
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.autotune.enabled = True

    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(X)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y))

    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(X), 8192), seed=23, reshuffle_each_iteration=True
        )

    ds = ds.batch(batch_size, drop_remainder=False)

    if cache:
        ds = ds.cache()

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _build_model(input_shape):
    with strategy.scope():
        model = keras.models.Sequential(
            [
                keras.layers.Input(shape=input_shape),
                Bidirectional(LSTM(1024, return_sequences=True)),
                Bidirectional(LSTM(512, return_sequences=True)),
                Bidirectional(LSTM(256, return_sequences=True)),
                Bidirectional(LSTM(128, return_sequences=True)),
                Dense(128, activation="selu"),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mae")
    return model


def _resolve_checkpoint_path(fold: int) -> str:
    local = f"folds{fold}.keras"
    if os.path.exists(local):
        return local
    kaggle_input = "../input/finetune-of-tensorflow-bidirectional-lstm/" + local
    if os.path.exists(kaggle_input):
        return kaggle_input
    return local  # default target for saving after training


def fit_lstm(train_arr, test_arr):
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)

    test_ds = _make_ds(
        test_arr,
        y=None,
        training=False,
        batch_size=BATCH_SIZE,
        cache=False,
    )

    test_preds = []

    train_preds = train[["id", "breath_id", "pressure"]].copy()
    n_breaths = train_arr.shape[0]
    train_preds.loc[:, "modified_breath_id"] = np.repeat(
        np.arange(n_breaths, dtype=np.int32), 80
    )

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_arr, targets)):
        K.clear_session()
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

        checkpoint_filepath = f"folds{fold}.keras"
        resolved_ckpt = _resolve_checkpoint_path(fold)

        X_valid = train_arr[valid_idx]
        y_valid = targets[valid_idx]

        valid_ds = _make_ds(
            X_valid,
            y_valid,
            training=False,
            batch_size=BATCH_SIZE,
            cache=False,
        )

        if TRAIN_MODEL and not os.path.exists(resolved_ckpt):
            model = _build_model(train_arr.shape[-2:])
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

            X_train = train_arr[train_idx]
            y_train = targets[train_idx]
            train_ds = _make_ds(
                X_train, y_train, training=True, batch_size=BATCH_SIZE, cache=False
            )

            model.fit(
                train_ds,
                validation_data=valid_ds,
                epochs=EPOCHS,
                verbose=0,
                callbacks=[lr, es, sv, TqdmCallback(verbose=0)],
            )
            model = keras.models.load_model(checkpoint_filepath)
            del X_train, y_train, train_ds
        else:
            if not os.path.exists(resolved_ckpt):
                raise FileNotFoundError(
                    f"Missing required pretrained checkpoint for fold={fold}: {resolved_ckpt}. "
                    f"To train from scratch set TRAIN_MODEL=True, but this will not meet a 600s timeout."
                )
            model = keras.models.load_model(resolved_ckpt)

        test_preds.append(model.predict(test_ds, verbose=0).ravel())

        valid_pred = model.predict(valid_ds, verbose=0).ravel()

        row_idx = (
            valid_idx[:, None] * 80 + np.arange(80, dtype=np.int32)[None, :]
        ).reshape(-1)
        train_preds.iloc[row_idx, train_preds.columns.get_loc("pressure")] = valid_pred

        del X_valid, y_valid, valid_ds, valid_pred, model, row_idx
        gc.collect()

    return test_preds, train_preds




## === cell 18
gc.collect()
set_seed(23)

test_preds, train_preds = fit_lstm(train_, test_)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/235330818.py in <cell line: 0>()
      2 set_seed(23)
      3 
----> 4 test_preds, train_preds = fit_lstm(train_, test_)
      5 

/tmp/ipykernel_11/136889637.py in fit_lstm(train_arr, test_arr)
    147         else:
    148             if not os.path.exists(resolved_ckpt):
--> 149                 raise FileNotFoundError(
    150                     f"Missing required pretrained checkpoint for fold={fold}: {resolved_ckpt}. "
    151                     f"To train from scratch set TRAIN_MODEL=True, but this will not meet a 600s timeout."

FileNotFoundError: Missing required pretrained checkpoint for fold=0: folds0.keras. To train from scratch set TRAIN_MODEL=True, but this will not meet a 600s timeout.

## === cell 19
pred_mean = sum(test_preds) / NUM_FOLDS
pred_median = np.median(np.vstack(test_preds), axis=0)

test_ids = test["id"].to_numpy()
sub_base = pd.DataFrame({"id": test_ids})

submission_mean = sub_base.copy()
submission_mean["pressure"] = pred_mean
submission_mean.to_csv("submission_mean.csv", index=False)

submission_median = sub_base.copy()
submission_median["pressure"] = pred_median
submission_median.to_csv("submission_median.csv", index=False)

submission_final = sub_base.copy()
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

test_out = test.copy()
test_out["pred1"] = pred_median
test_out["pred1"] = (
    np.round((test_out.pred1 - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
    + PRESSURE_MIN
)
test_out["pred1"] = np.clip(test_out.pred1, PRESSURE_MIN, PRESSURE_MAX)

train_preds[["id", "pressure"]].to_csv("train_p2.csv", index=False)
test_out[["id", "pred1"]].to_csv("test_p2.csv", index=False)

print(
    "Wrote submission.csv with columns:",
    submission_final.columns.tolist(),
    "shape:",
    submission_final.shape,
)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3078509853.py in <cell line: 0>()
----> 1 pred_mean = sum(test_preds) / NUM_FOLDS
      2 pred_median = np.median(np.vstack(test_preds), axis=0)
      3 
      4 test_ids = test["id"].to_numpy()
      5 sub_base = pd.DataFrame({"id": test_ids})

NameError: name 'test_preds' is not defined

## === cell 20
if DEBUG:
    plt.title("Histogram of Test Predicted Pressures (submission.csv)", size=14)
    plt.hist(submission_final.sample(10_000, random_state=42).pressure.values, bins=100)
    plt.show()
print(
    "Max pressure =",
    float(submission_final.pressure.max()),
    "Min pressure =",
    float(submission_final.pressure.min()),
)

print("Elapsed seconds:", time.time() - start_time)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1288373965.py in <cell line: 0>()
      5 print(
      6     "Max pressure =",
----> 7     float(submission_final.pressure.max()),
      8     "Min pressure =",
      9     float(submission_final.pressure.min()),

NameError: name 'submission_final' is not defined
