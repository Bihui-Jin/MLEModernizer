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

cudf-cu12==25.2.2
cudf-polars-cu12==25.6.0
cuml-cu12==25.2.1
cupy-cuda12x==13.6.0
dask-cudf-cu12==25.2.2
geopandas==0.14.4
google-api-python-client==2.177.0
imageio==2.37.0
imageio-ffmpeg==0.6.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
libcudf-cu12==25.2.2
libcuml-cu12==25.2.1
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numba==0.60.0
numba-cuda==0.2.0
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
plotly==5.24.1
plotly-express==0.4.1
protobuf==6.33.0
pylibcudf-cu12==25.2.2
requests==2.32.5
requests-oauthlib==2.0.0
requests-toolbelt==1.0.0
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
tqdm==4.67.1

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

0.6841

# 6. Current score

17.32907

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 17.32907) has done: 'We fix the immediate import-time crash caused by an incompatibility between TensorFlow 2.18 and the installed protobuf 6.x by forcing protobuf to use the pure-Python implementation before importing TensorFlow. Then we keep your pipeline intact but add a small safety fix to align the submission `id` ordering exactly to `sample_submission.csv` to avoid any accidental misalignment. Finally, we keep all modeling/feature logic unchanged so the produced submission is valid and your score can be obtained (currently you have no score because the notebook crashes before training/inference).'

# 9. Code solution

## === cell 0
print("\n... IMPORTS STARTING ...\n")

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gc
import sys
import math
import time
import random
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

import tensorflow as tf
from tqdm import tqdm

print("\n\tVERSION INFORMATION")
print(f"\t\t– TENSORFLOW VERSION: {tf.__version__}")
print(f"\t\t– NUMPY VERSION: {np.__version__}")
print(f"\t\t– PANDAS VERSION: {pd.__version__}")

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


def seed_it_all(seed=7):
    """Attempt to be reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


print("\n\n... IMPORTS COMPLETE ...\n")
print("\n... SEEDING FOR DETERMINISTIC BEHAVIOUR ...\n")
seed_it_all(7)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
print(f"\n... ACCELERATOR SETUP STARTING ...\n")

try:
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()
except Exception:
    TPU = None

if TPU:
    print(f"\n... RUNNING ON TPU - {TPU.master()} ...\n")
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    strategy = tf.distribute.experimental.TPUStrategy(TPU)
else:
    strategy = tf.distribute.get_strategy()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        print("\n ... RUNNING ON GPU ...\n")
    else:
        print("\n ... RUNNING ON CPU ...\n")

N_REPLICAS = strategy.num_replicas_in_sync
print(f"... # OF REPLICAS: {N_REPLICAS} ...\n")
print(f"\n... ACCELERATOR SETUP COMPLETED ...\n")



## === cell 2
print("\n... DATA ACCESS SETUP STARTED ...\n")

DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/kaggle/data/ventilator-pressure-prediction"

print(f"\n... DATA DIRECTORY PATH IS:\n\t--> {DATA_DIR}")

print(f"\n... IMMEDIATE CONTENTS OF DATA DIRECTORY IS:")
for file in tf.io.gfile.glob(os.path.join(DATA_DIR, "*")):
    print(f"\t--> {file}")

print("\n\n... DATA ACCESS SETUP COMPLETED ...\n")



## === cell 3
print("\n... BASIC DATA SETUP STARTING ...\n\n")

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SS_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
ss_df = pd.read_csv(SS_CSV)

ROWS_PER_BREATH = 80

train_df["breath_step"] = np.tile(
    np.arange(ROWS_PER_BREATH, dtype=np.int16), len(train_df) // ROWS_PER_BREATH
)
test_df["breath_step"] = np.tile(
    np.arange(ROWS_PER_BREATH, dtype=np.int16), len(test_df) // ROWS_PER_BREATH
)

POSSIBLE_PRESSURES = np.sort(train_df["pressure"].unique())
PRESSURE_DELTA_STEP = float(np.mean(POSSIBLE_PRESSURES[1:] - POSSIBLE_PRESSURES[:-1]))

print(f"... train shape: {train_df.shape}, test shape: {test_df.shape}")
print(
    f"... unique pressures: {len(POSSIBLE_PRESSURES)}, delta_step≈{PRESSURE_DELTA_STEP:.6f}"
)

print("\n\n... BASIC DATA SETUP FINISHING ...\n")



## === cell 4
print(f"\n... XLA OPTIMIZATIONS STARTING ...\n")
tf.config.optimizer.set_jit(True)
print(f"\n... XLA OPTIMIZATIONS COMPLETED ...\n")




## === cell 5
def flatten_l_o_l(nested_list):
    return [item for sublist in nested_list for item in sublist]


def compute_weighted_median(_values, _weights, redux_factor=1):
    return np.median(
        flatten_l_o_l(
            [[_v] * int(_w // redux_factor) for _v, _w in zip(_values, _weights)]
        )
    )


def add_features(
    df: pd.DataFrame,
    U_IN_N_FORWARD=3,
    U_IN_N_BACKWARD=3,
    U_OUT_N_FORWARD=1,
    U_OUT_N_BACKWARD=1,
    use_rc=True,
) -> pd.DataFrame:
    """Feature engineering (kept same intent)."""

    df = df.copy()

    df["uin_auc"] = (df["time_step"] * df["u_in"]).groupby(df["breath_id"]).cumsum()
    df["uin_csum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()

    df["cross3"] = df["time_step"] * df["u_in"]
    df["cross3_sqd_1"] = df["time_step"] * (df["u_in"] ** 2)
    df["cross3_sqd_2"] = (df["time_step"] ** 2) * df["u_in"]
    df["cross3_cubed_1"] = df["time_step"] * (df["u_in"] ** 3)
    df["cross3_cubed_2"] = (df["time_step"] ** 3) * df["u_in"]

    for i in range(1, U_IN_N_BACKWARD + 1):
        df[f"u_in_{i}_back"] = df.groupby("breath_id")["u_in"].shift(i).fillna(0.0)
    for i in range(1, U_IN_N_FORWARD + 1):
        df[f"u_in_{i}_forw"] = df.groupby("breath_id")["u_in"].shift(-i).fillna(0.0)

    for i in range(1, U_OUT_N_BACKWARD + 1):
        df[f"u_out_{i}_back"] = df.groupby("breath_id")["u_out"].shift(i).fillna(0.0)
    for i in range(1, U_OUT_N_FORWARD + 1):
        df[f"u_out_{i}_forw"] = df.groupby("breath_id")["u_out"].shift(-i).fillna(0.0)

    for i in range(1, U_IN_N_BACKWARD + 1):
        df[f"u_in_diff_{i}_back"] = df["u_in"] - df[f"u_in_{i}_back"]
    for i in range(1, U_OUT_N_BACKWARD + 1):
        df[f"u_out_diff_{i}_back"] = df["u_out"] - df[f"u_out_{i}_back"]

    for i in range(1, U_IN_N_FORWARD + 1):
        df[f"u_in_diff_{i}_forw"] = df["u_in"] - df[f"u_in_{i}_forw"]
    for i in range(1, U_OUT_N_FORWARD + 1):
        df[f"u_out_diff_{i}_forw"] = df["u_out"] - df[f"u_out_{i}_forw"]

    if use_rc:
        df["_R"] = df["R"].astype(str)
        df["_C"] = df["C"].astype(str)
        df["R"] = df["R"] / 50.0
        df["C"] = df["C"] / 50.0
        df = pd.get_dummies(df, columns=["_R", "_C"])

    for c in df.columns:
        if c in ["u_out", "breath_step"]:
            df[c] = df[c].astype("uint8")
        elif df[c].dtype == "float64":
            df[c] = df[c].astype("float32")

    return df




## === cell 6
print("\n... ADDING FEATURES TO TRAIN/TEST DATAFRAMES ...\n")

train_feat = add_features(train_df, use_rc=True)
test_feat = add_features(test_df, use_rc=True)

missing_in_test = [c for c in train_feat.columns if c not in test_feat.columns]
missing_in_train = [c for c in test_feat.columns if c not in train_feat.columns]

for c in missing_in_test:
    test_feat[c] = 0
for c in missing_in_train:
    train_feat[c] = 0

test_feat = test_feat[train_feat.columns]

print(f"... train_feat shape: {train_feat.shape}, test_feat shape: {test_feat.shape}")



## === cell 7
print("\n... ADDING BREATH-LEVEL META FEATURES ...\n")


def add_breath_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["breath_duration"] = (
        df.groupby("breath_id")["time_step"].transform("max").astype("float32")
    )
    df["exhale_steps"] = (
        df.groupby("breath_id")["u_out"].transform("sum").astype("float32")
    )
    df["inhale_steps"] = (
        ROWS_PER_BREATH - df.groupby("breath_id")["u_out"].transform("sum")
    ).astype("float32")
    return df


train_feat = add_breath_meta(train_feat)
test_feat = add_breath_meta(test_feat)

print("... done")



## === cell 8
print("\n... ROBUST SCALING FEATURES ...\n")

from sklearn.preprocessing import RobustScaler

RS = RobustScaler()

FEATURE_COLS = [
    c for c in train_feat.columns if c not in ["id", "breath_id", "pressure"]
]
train_feat[FEATURE_COLS] = RS.fit_transform(train_feat[FEATURE_COLS]).astype("float32")
test_feat[FEATURE_COLS] = RS.transform(test_feat[FEATURE_COLS]).astype("float32")

print("... done")



## === cell 9
print("\n... COMPRESSING TO BREATH-LEVEL (80 steps -> 1 row per breath) ...\n")


def compress_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert timestep-level rows into breath-level rows by flattening selected columns into 80-length vectors.

    FIX: avoid pandas pivot_table failure "Grouper for 'breath_step' not 1-dimensional"
    by (1) de-duplicating columns, (2) forcing breath_step to be a single Series,
    (3) using set_index + unstack which is robust and fast.
    """
    df = df.copy()

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    if "breath_step" not in df.columns:
        raise KeyError("breath_step missing from dataframe")
    df["breath_step"] = pd.Series(df["breath_step"].to_numpy(), index=df.index).astype(
        np.int16
    )

    exclude_cols = {"id", "breath_id", "pressure"}

    never_seq = {"R", "C", "breath_duration", "exhale_steps", "inhale_steps"}
    never_seq |= {c for c in df.columns if c.startswith("_R_") or c.startswith("_C_")}

    seq_cols = sorted(
        {
            c
            for c in df.columns
            if (c not in exclude_cols)
            and (c not in never_seq)
            and (
                ("_back" in c)
                or ("_forw" in c)
                or ("cross" in c)
                or ("u_in" in c)
                or ("u_out" in c)
                or ("uin" in c)
                or (c == "time_step")
            )
        }
    )

    out = pd.DataFrame({"breath_id": np.sort(df["breath_id"].unique())})

    idx_cols = ["breath_id", "breath_step"]
    base = df[
        idx_cols + seq_cols + (["pressure"] if "pressure" in df.columns else [])
    ].copy()

    base = base.sort_values(["breath_id", "breath_step"], kind="mergesort")
    base = base.drop_duplicates(subset=["breath_id", "breath_step"], keep="first")

    base = base.set_index(["breath_id", "breath_step"])

    for j, c in enumerate(seq_cols):
        wide = base[c].unstack("breath_step")
        wide = wide.reindex(columns=np.arange(ROWS_PER_BREATH))
        wide.columns = [
            f"{chr(97 + (j % 26))}{j // 26}_{i}" for i in range(ROWS_PER_BREATH)
        ]
        wide = wide.reset_index()
        out = out.merge(wide, on="breath_id", how="left")

    if "pressure" in df.columns:
        wide_p = base["pressure"].unstack("breath_step")
        wide_p = wide_p.reindex(columns=np.arange(ROWS_PER_BREATH))
        wide_p.columns = [f"z_{i}" for i in range(ROWS_PER_BREATH)]
        wide_p = wide_p.reset_index()
        out = out.merge(wide_p, on="breath_id", how="left")

    repeat_cols = [
        c
        for c in df.columns
        if c in ["R", "C", "breath_duration", "exhale_steps", "inhale_steps"]
        or c.startswith("_R_")
        or c.startswith("_C_")
    ]
    if repeat_cols:
        rep = df.groupby("breath_id")[repeat_cols].first().reset_index()
        out = out.merge(rep, on="breath_id", how="left")

    return out


train_b = compress_df(train_feat)
test_b = compress_df(test_feat)

print(f"... train_b shape: {train_b.shape}, test_b shape: {test_b.shape}")



## === cell 10
print("\n... KNN SETUP ...\n")

from sklearn.neighbors import NearestNeighbors

PRESSURE_COLS = [f"z_{i}" for i in range(ROWS_PER_BREATH)]
USE_COLS = [c for c in train_b.columns if c not in ["breath_id"] + PRESSURE_COLS]

NN_TO_USE = 10
RESTRICT_RC = True

train_b[USE_COLS] = train_b[USE_COLS].fillna(0.0)
test_b[USE_COLS] = test_b[USE_COLS].fillna(0.0)

print(
    f"... Using {len(USE_COLS)} features, neighbors={NN_TO_USE}, restrict_rc={RESTRICT_RC}"
)



## === cell 11
print("\n... RUNNING KNN INFERENCE (BREATH-LEVEL) ...\n")

rc_cols = [c for c in train_b.columns if c.startswith("_R_") or c.startswith("_C_")]
use_onehot_rc = (len(rc_cols) > 0) and RESTRICT_RC


def rc_key_from_row(df: pd.DataFrame, idx: int) -> Tuple:
    if use_onehot_rc:
        r = tuple(
            int(df.loc[idx, c])
            for c in sorted([c for c in rc_cols if c.startswith("_R_")])
        )
        cc = tuple(
            int(df.loc[idx, c])
            for c in sorted([c for c in rc_cols if c.startswith("_C_")])
        )
        return (r, cc)
    return (float(df.loc[idx, "R"]), float(df.loc[idx, "C"]))


train_groups = {}
for i in range(len(train_b)):
    k = rc_key_from_row(train_b, i)
    train_groups.setdefault(k, []).append(i)

test_groups = {}
for i in range(len(test_b)):
    k = rc_key_from_row(test_b, i)
    test_groups.setdefault(k, []).append(i)

mean_pred = np.zeros((len(test_b), ROWS_PER_BREATH), dtype=np.float32)
median_pred = np.zeros((len(test_b), ROWS_PER_BREATH), dtype=np.float32)

for k, test_idx_list in tqdm(test_groups.items(), total=len(test_groups)):
    train_idx_list = train_groups.get(k, None)
    if train_idx_list is None or len(train_idx_list) < 2:
        train_idx_list = list(range(len(train_b)))

    X_train = train_b.loc[train_idx_list, USE_COLS].values
    Y_train = train_b.loc[train_idx_list, PRESSURE_COLS].values

    X_test = test_b.loc[test_idx_list, USE_COLS].values

    model = NearestNeighbors(
        n_neighbors=min(NN_TO_USE, len(train_idx_list)), metric="manhattan"
    )
    model.fit(X_train)

    distances, indices = model.kneighbors(X_test, return_distance=True)

    eps = 1e-6
    w = 1.0 / (distances + eps)  # shape (n_test_group, k)
    w = w / (w.sum(axis=1, keepdims=True) + eps)

    for row_i, test_global_idx in enumerate(test_idx_list):
        nn_idx = indices[row_i]
        nn_press = Y_train[nn_idx]  # (k, 80)

        mean_pred[test_global_idx] = (
            (w[row_i][:, None] * nn_press).sum(axis=0).astype(np.float32)
        )
        median_pred[test_global_idx] = np.median(nn_press, axis=0).astype(np.float32)

print("... done")



## === cell 12
print("\n... EXPANDING BREATH-LEVEL PREDICTIONS BACK TO ROW-LEVEL ...\n")

test_map = test_df[["id", "breath_id", "breath_step"]].copy()
test_map = test_map.reset_index(drop=True)

breath_id_to_idx = {bid: i for i, bid in enumerate(test_b["breath_id"].values)}

pred_pressure = np.zeros(len(test_map), dtype=np.float32)
for i in range(len(test_map)):
    bid = test_map.loc[i, "breath_id"]
    step = int(test_map.loc[i, "breath_step"])
    bidx = breath_id_to_idx[bid]
    pred_pressure[i] = 0.5 * (mean_pred[bidx, step] + median_pred[bidx, step])

sub = pd.DataFrame({"id": test_map["id"].values, "pressure": pred_pressure})

sub = ss_df[["id"]].merge(sub, on="id", how="left")
sub["pressure"] = sub["pressure"].fillna(0.0).astype("float32")

assert list(sub.columns) == ["id", "pressure"]
assert len(sub) == len(ss_df)

print(sub.head())
print(sub.tail())



## === cell 13
print("\n... WRITING SUBMISSION ...\n")

SUB_PATH = "submission.csv"
sub.to_csv(SUB_PATH, index=False)

print(f"... wrote {SUB_PATH} with shape {sub.shape}")
print("\nDone.")
