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
import time
import numpy as np
import pandas as pd
import math
import random
import gc
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")


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

pd.set_option("display.max_columns", 300)




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

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ["TF_NUM_INTRAOP_THREADS"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(os.environ["TF_NUM_INTEROP_THREADS"])
    )
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.keras.utils.set_random_seed(23)
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 3
DEBUG = False

TRAIN_MODEL = True

TRAIN_PATH = "/kaggle/input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "/kaggle/input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"



## === cell 4
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
submission = pd.read_csv(SAMPLE_SUB_PATH, dtype={"id": "int32", "pressure": "float32"})

train["bilstm_pred"] = np.float32(0.0)
test["bilstm_pred"] = np.float32(0.0)

bilstm_train_path = "/kaggle/input/gbvpp-predictions1/bilstm_train.csv/bilstm_train.csv"
bilstm_test_path = "/kaggle/input/gbvpp-predictions1/bilstm_test.csv/bilstm_test.csv"
if os.path.exists(bilstm_train_path) and os.path.exists(bilstm_test_path):
    train_bilstm = pd.read_csv(
        bilstm_train_path, usecols=["pressure"], dtype={"pressure": "float32"}
    )
    test_bilstm = pd.read_csv(
        bilstm_test_path, usecols=["pressure"], dtype={"pressure": "float32"}
    )
    if len(train_bilstm) == len(train):
        train["bilstm_pred"] = train_bilstm["pressure"].values
    if len(test_bilstm) == len(test):
        test["bilstm_pred"] = test_bilstm["pressure"].values
    del train_bilstm, test_bilstm
gc.collect()

if DEBUG:
    train = train[: 80 * 1000].copy()



## === cell 5
print(f"Length of TRAIN dataset: {len(train)}")
print(f"Length of TEST dataset: {len(test)}")
print(f'Number of breaths in train dataset: {train["breath_id"].nunique()}')
print(f'Number of breaths in test dataset: {test["breath_id"].nunique()}')



## === cell 6
all_pressure = np.sort(train["pressure"].unique())
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
train = train.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
    drop=True
)
test = test.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
    drop=True
)

train_log_u_in = np.log1p(train["u_in"].to_numpy(np.float32, copy=False))
test_log_u_in = np.log1p(test["u_in"].to_numpy(np.float32, copy=False))
train["log_u_in"] = train_log_u_in
test["log_u_in"] = test_log_u_in

train_qcut = pd.qcut(train.time_step, q=80, retbins=True, duplicates="drop")
train["time_step_class"] = train_qcut[0].cat.codes.astype("int16")
bins = train_qcut[1]
test["time_step_class"] = pd.cut(
    test.time_step, bins=bins, include_lowest=True
).cat.codes.astype("int16")


def _per_breath_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    assert n % 80 == 0
    nb = n // 80
    mat = df["log_u_in"].to_numpy(np.float32, copy=False).reshape(nb, 80)
    breath_ids = df["breath_id"].to_numpy(np.int32, copy=False)[::80]
    return mat, breath_ids


piv, train_breath_ids = _per_breath_matrix(train)
piv_test, test_breath_ids = _per_breath_matrix(test)

pca = PCA(n_components=2, random_state=42)
pca.fit(piv)

train_pca = pca.transform(piv)
test_pca = pca.transform(piv_test)

try:
    km = KMeans(
        n_clusters=5,
        random_state=42,
        max_iter=200,
        init="k-means++",
        tol=0.0001,
        n_init="auto",
        algorithm="lloyd",
    )
except TypeError:
    km = KMeans(
        n_clusters=5,
        random_state=42,
        max_iter=200,
        init="k-means++",
        tol=0.0001,
        n_init=10,
        algorithm="lloyd",
    )

y_km = km.fit_predict(train_pca)
y_km_test = km.predict(test_pca)

train["cluster"] = np.repeat(y_km.astype("int16", copy=False), 80)
test["cluster"] = np.repeat(y_km_test.astype("int16", copy=False), 80)



## === cell 8
pass



## === cell 9
train.drop(["time_step_class", "log_u_in"], axis=1, inplace=True)
test.drop(["time_step_class", "log_u_in"], axis=1, inplace=True)




## === cell 10
def add_features_fast(df: pd.DataFrame, already_sorted: bool = True) -> pd.DataFrame:
    if not already_sorted:
        df = df.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
            drop=True
        )

    n = len(df)
    assert n % 80 == 0, "Expected 80 timesteps per breath."
    nb = n // 80

    u_in = df["u_in"].to_numpy(np.float32, copy=False)
    u_out = df["u_out"].to_numpy(np.float32, copy=False)
    t = df["time_step"].to_numpy(np.float32, copy=False)
    bilstm_pred = df["bilstm_pred"].to_numpy(np.float32, copy=False)
    cluster = df["cluster"].to_numpy(np.int16, copy=False)

    u_in_2d = u_in.reshape(nb, 80)
    u_out_2d = u_out.reshape(nb, 80)
    t_2d = t.reshape(nb, 80)
    bilstm_2d = bilstm_pred.reshape(nb, 80)

    area = np.cumsum(t_2d * u_in_2d, axis=1, dtype=np.float32).reshape(-1)
    u_in_cumsum = np.cumsum(u_in_2d, axis=1, dtype=np.float32).reshape(-1)

    def lag_pad_2d(x2d: np.ndarray, k: int) -> np.ndarray:
        out = np.empty_like(x2d)
        out[:, :k] = 0.0
        out[:, k:] = x2d[:, :-k]
        return out

    def lead_pad_2d(x2d: np.ndarray, k: int) -> np.ndarray:
        out = np.empty_like(x2d)
        out[:, -k:] = 0.0
        out[:, :-k] = x2d[:, k:]
        return out

    u_in_lag = {}
    u_out_lag = {}
    u_in_lead = {}
    u_out_lead = {}
    for k in (1, 2, 3, 4):
        u_in_lag[k] = lag_pad_2d(u_in_2d, k).reshape(-1)
        u_out_lag[k] = lag_pad_2d(u_out_2d, k).reshape(-1)
        u_in_lead[k] = lead_pad_2d(u_in_2d, k).reshape(-1)
        u_out_lead[k] = lead_pad_2d(u_out_2d, k).reshape(-1)

    u_in_max = u_in_2d.max(axis=1).astype(np.float32, copy=False)
    u_in_mean = u_in_2d.mean(axis=1).astype(np.float32, copy=False)
    rep_max = np.repeat(u_in_max, 80)
    rep_mean = np.repeat(u_in_mean, 80)

    c = cluster.astype(np.int32, copy=False)
    cmax = int(c.max()) if c.size else 0
    counts = np.bincount(c, minlength=cmax + 1).astype(np.float32)
    sums = np.bincount(
        c, weights=u_in.astype(np.float64, copy=False), minlength=cmax + 1
    )
    means = (sums / np.maximum(counts, 1.0)).astype(np.float32)

    if c.size:
        order = np.argsort(c, kind="mergesort")
        c_sorted = c[order]
        u_sorted = u_in[order]
        seg_starts = np.r_[0, np.flatnonzero(c_sorted[1:] != c_sorted[:-1]) + 1]
        seg_ids = c_sorted[seg_starts]
        seg_max = np.maximum.reduceat(u_sorted, seg_starts).astype(
            np.float32, copy=False
        )
        maxs = np.full(cmax + 1, -np.inf, dtype=np.float32)
        maxs[seg_ids] = seg_max
    else:
        maxs = np.full(cmax + 1, -np.inf, dtype=np.float32)

    cluster_u_in_mean = means[c]
    cluster_u_in_max = maxs[c]

    df["area"] = area
    df["u_in_cumsum"] = u_in_cumsum

    for k in (1, 2, 3, 4):
        df[f"u_in_lag{k}"] = u_in_lag[k]
        df[f"u_out_lag{k}"] = u_out_lag[k]
        df[f"u_in_lag_back{k}"] = u_in_lead[k]
        df[f"u_out_lag_back{k}"] = u_out_lead[k]

    df["breath_id__u_in__max"] = rep_max
    df["breath_id__u_in__diffmax"] = rep_max - u_in
    df["breath_id__u_in__diffmean"] = rep_mean - u_in

    for k in (1, 2, 3, 4):
        df[f"u_in_diff{k}"] = u_in - u_in_lag[k]
        df[f"u_out_diff{k}"] = u_out - u_out_lag[k]

    df["cross"] = (u_in * u_out).astype(np.float32, copy=False)
    df["cross2"] = (t * u_out).astype(np.float32, copy=False)

    df["bilstm_pred_lag1"] = lag_pad_2d(bilstm_2d, 1).reshape(-1)
    df["bilstm_pred_lag2"] = lag_pad_2d(bilstm_2d, 2).reshape(-1)
    df["bilstm_pred_lag_back1"] = lead_pad_2d(bilstm_2d, 1).reshape(-1)
    df["bilstm_pred_lag_back2"] = lead_pad_2d(bilstm_2d, 2).reshape(-1)

    df["cluster__u_in__diffmean"] = cluster_u_in_mean - u_in
    df["cluster__u_in__diffmax"] = cluster_u_in_max - u_in

    R = df["R"].astype("int16", copy=False)
    C = df["C"].astype("int16", copy=False)
    df["R"] = R.astype(np.float32, copy=False)
    df["C"] = C.astype(np.float32, copy=False)
    df["R__C"] = (R.astype("int32") * 1000 + C.astype("int32")).astype(np.int32)
    df["cluster"] = df["cluster"].astype("int16", copy=False)

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df


train_fe = add_features_fast(train, already_sorted=True)
test_fe = add_features_fast(test, already_sorted=True)

del train, test
gc.collect()

train_feature_cols = [c for c in train_fe.columns if c != "pressure"]
test_feature_cols = list(test_fe.columns)

for c in set(train_feature_cols) - set(test_feature_cols):
    test_fe[c] = 0
for c in set(test_feature_cols) - set(train_feature_cols):
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

del train_fe, test_fe, RS
gc.collect()

train_arr = train_arr.reshape(-1, 80, train_arr.shape[-1])
test_arr = test_arr.reshape(-1, 80, train_arr.shape[-1])

print(
    "train_arr", train_arr.shape, "targets", targets.shape, "test_arr", test_arr.shape
)




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

NUM_FOLDS = 5
EPOCHS = 80
if DEBUG:
    EPOCHS = 1


def _make_ds(X, y=None, batch_size=1024):
    opt = tf.data.Options()
    try:
        opt.experimental_deterministic = True
    except Exception:
        pass

    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(X)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y))

    ds = ds.with_options(opt)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _predict_dataset(model, ds) -> np.ndarray:
    return model.predict(ds, verbose=0)


def fit_lstm(train_arr, test_arr, targets):
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)
    test_preds = []
    oof_preds = np.zeros_like(targets, dtype=np.float32)

    n_breaths = train_arr.shape[0]
    idx_all = np.arange(n_breaths)

    test_ds = _make_ds(test_arr, y=None, batch_size=BATCH_SIZE)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(idx_all)):
        K.clear_session()
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)

        X_train, X_valid = train_arr[train_idx], train_arr[valid_idx]
        y_train, y_valid = targets[train_idx], targets[valid_idx]

        train_ds = _make_ds(X_train, y_train, batch_size=BATCH_SIZE)
        valid_ds = _make_ds(X_valid, y_valid, batch_size=BATCH_SIZE)

        checkpoint_filepath = f"folds{fold}.keras"

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
            model.compile(
                optimizer="adam",
                loss="mae",
                jit_compile=True,
                steps_per_execution=16,
            )

        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=0)
        es = EarlyStopping(
            monitor="val_loss",
            patience=25,
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

        if (not TRAIN_MODEL) and os.path.exists(checkpoint_filepath):
            model = keras.models.load_model(
                checkpoint_filepath,
                custom_objects={"Attention": Attention, "swish": swish},
            )
        else:
            model.fit(
                train_ds,
                validation_data=valid_ds,
                epochs=EPOCHS,
                verbose=0,
                callbacks=[lr, es, sv],
            )
            model = keras.models.load_model(
                checkpoint_filepath,
                custom_objects={"Attention": Attention, "swish": swish},
            )

        pred_valid = _predict_dataset(model, valid_ds).reshape(-1, 80)
        oof_preds[valid_idx] = pred_valid

        pred_test = _predict_dataset(model, test_ds).reshape(-1, 80)
        test_preds.append(pred_test)

        del (
            X_train,
            X_valid,
            y_train,
            y_valid,
            pred_valid,
            pred_test,
            model,
            train_ds,
            valid_ds,
        )
        gc.collect()

    return oof_preds, test_preds




## === cell 14
oof_preds, test_preds = fit_lstm(train_arr, test_arr, targets)



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



## === cell 16
print(
    "submission pressure stats:",
    submission["pressure"].min(),
    submission["pressure"].max(),
    submission["pressure"].nunique(),
)
print("Elapsed seconds:", round(time.time() - start_time, 2))
