# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
print("\n... IMPORTS STARTING ...\n")

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    _pb_major = int(_pb_ver.split(".", 1)[0])
except Exception:
    _pb_major = None

if _pb_major is not None and _pb_major >= 5:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])
    import importlib

    importlib.invalidate_caches()
    for _m in list(sys.modules):
        if _m.startswith("google.protobuf"):
            del sys.modules[_m]

try:
    import weightedstats as ws
except Exception:
    ws = None

import gc, random, time, math
import numpy as np
import pandas as pd
import tensorflow as tf

pd.options.mode.chained_assignment = None

print("\n\tVERSION INFORMATION")
print(f"\t\t– TENSORFLOW VERSION: {tf.__version__}")
print(f"\t\t– NUMPY VERSION: {np.__version__}")


def seed_it_all(seed=7):
    """Attempt to be Reproducible"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


print("\n\n... IMPORTS COMPLETE ...\n")
print("\n... SEEDING FOR DETERMINISTIC BEHAVIOUR ...\n")
seed_it_all()


## === cell 1
print(f"\n... ACCELERATOR SETUP STARTING ...\n")

try:
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()
except ValueError:
    TPU = None

if TPU:
    print(f"\n... RUNNING ON TPU - {TPU.master()} ...\n")
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    strategy = tf.distribute.experimental.TPUStrategy(TPU)
else:
    strategy = tf.distribute.get_strategy()
    if tf.config.experimental.list_physical_devices("GPU"):
        print(f"\n ... RUNNING ON GPU ...\n")
        import cudf, cupy
        import cuml
        from cuml.neighbors import NearestNeighbors
        import cuml.preprocessing
    else:
        print(f"\n ... RUNNING ON CPU ...\n")
        import pandas as cudf  # dummy to avoid crashes

        NearestNeighbors = None

N_REPLICAS = strategy.num_replicas_in_sync
print(f"... # OF REPLICAS: {N_REPLICAS} ...\n")
print(f"\n... ACCELERATOR SETUP COMPLTED ...\n")



## === cell 2
print("\n... DATA ACCESS SETUP STARTED ...\n")

from kaggle_datasets import KaggleDatasets

if TPU:
    DATA_DIR = KaggleDatasets().get_gcs_path("ventilator-pressure-prediction")
else:
    DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"

print(f"\n... DATA DIRECTORY PATH IS:\n\t--> {DATA_DIR}")
print("\n\n... DATA ACCESS SETUP COMPLETED ...\n")



## === cell 3
print("\n... BASIC DATA SETUP STARTING ...\n\n")

DO_CLUSTERING = True

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SS_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

if DO_CLUSTERING:
    train_df = cudf.read_csv(TRAIN_CSV)
    test_df = cudf.read_csv(TEST_CSV)
else:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

ss_df = pd.read_csv(SS_CSV)

ROWS_PER_BREATH = 80

train_df["breath_step"] = cudf.Series(
    cupy.arange(len(train_df), dtype=cupy.int32) % ROWS_PER_BREATH
)
test_df["breath_step"] = cudf.Series(
    cupy.arange(len(test_df), dtype=cupy.int32) % ROWS_PER_BREATH
)

POSSIBLE_PRESSURES = train_df["pressure"].unique().sort_values().values
PRESSURE_DELTA_STEP = float((POSSIBLE_PRESSURES[1:] - POSSIBLE_PRESSURES[:-1]).mean())

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


def add_features_cudf(
    df,
    U_IN_N_FORWARD=3,
    U_IN_N_BACKWARD=3,
    U_OUT_N_FORWARD=1,
    U_OUT_N_BACKWARD=1,
    use_rc=True,
):
    """
    Runtime optimization (equivalent): do feature engineering directly in cuDF to avoid
    multi-million-row cudf<->pandas roundtrips. Groupby/shift/cumsum semantics preserved.
    """
    df["time_step"] = df["time_step"].astype("float32")
    df["u_in"] = df["u_in"].astype("float32")
    df["u_out"] = df["u_out"].astype("int8")

    df["uin_auc"] = (df["time_step"] * df["u_in"]).astype("float32")
    df["uin_auc"] = df.groupby("breath_id")["uin_auc"].cumsum()
    df["uin_csum"] = df.groupby("breath_id")["u_in"].cumsum().astype("float32")

    df["cross3"] = (df["time_step"] * df["u_in"]).astype("float32")
    df["cross3_sqd_1"] = (df["time_step"] * (df["u_in"] ** 2)).astype("float32")
    df["cross3_sqd_2"] = ((df["time_step"] ** 2) * df["u_in"]).astype("float32")
    df["cross3_cubed_1"] = (df["time_step"] * (df["u_in"] ** 3)).astype("float32")
    df["cross3_cubed_2"] = ((df["time_step"] ** 3) * df["u_in"]).astype("float32")

    gb = df.groupby("breath_id")

    for i in range(1, U_IN_N_BACKWARD + 1):
        df[f"u_in_{i}_back"] = gb["u_in"].shift(i).fillna(0).astype("float32")
    for i in range(1, U_IN_N_FORWARD + 1):
        df[f"u_in_{i}_forw"] = gb["u_in"].shift(-i).fillna(0).astype("float32")

    for i in range(1, U_OUT_N_BACKWARD + 1):
        df[f"u_out_{i}_back"] = gb["u_out"].shift(i).fillna(0).astype("float32")
    for i in range(1, U_OUT_N_FORWARD + 1):
        df[f"u_out_{i}_forw"] = gb["u_out"].shift(-i).fillna(0).astype("float32")

    for i in range(1, U_IN_N_BACKWARD + 1):
        df[f"u_in_diff_{i}_back"] = (df["u_in"] - df[f"u_in_{i}_back"]).astype(
            "float32"
        )
    for i in range(1, U_OUT_N_BACKWARD + 1):
        df[f"u_out_diff_{i}_back"] = (df["u_out"] - df[f"u_out_{i}_back"]).astype(
            "float32"
        )
    for i in range(1, U_IN_N_FORWARD + 1):
        df[f"u_in_diff_{i}_back"] = (df["u_in"] - df[f"u_in_{i}_forw"]).astype(
            "float32"
        )
    for i in range(1, U_OUT_N_FORWARD + 1):
        df[f"u_out_diff_{i}_forw"] = (df["u_out"] - df[f"u_out_{i}_forw"]).astype(
            "float32"
        )

    if use_rc:
        df["_R"] = df["R"].astype("int32").astype("str")
        df["_C"] = df["C"].astype("int32").astype("str")
        df["R"] = (df["R"].astype("float32") / 50.0).astype("float32")
        df["C"] = (df["C"].astype("float32") / 50.0).astype("float32")

        for r in ["5", "20", "50"]:
            df[f"_R_{r}"] = (df["_R"] == r).astype("uint8")
        for c in ["10", "20", "50"]:
            df[f"_C_{c}"] = (df["_C"] == c).astype("uint8")

        df = df.drop(columns=["_R", "_C"])

    df["breath_step"] = df["breath_step"].astype("uint8")
    df["u_out"] = df["u_out"].astype("uint8")

    return df




## === cell 6
print(
    "\n... ADDING FEATURES TO TRAIN/TEST DATAFRAMES (GPU, NO PANDAS ROUNDTRIPS) ...\n"
)

train_df = add_features_cudf(train_df, use_rc=True)
test_df = add_features_cudf(test_df, use_rc=True)

for _df in (train_df, test_df):
    g = _df.groupby("breath_id")
    _df["breath_duration"] = g["time_step"].transform("max").astype("float32")
    _df["exhale_steps"] = g["u_out"].transform("sum").astype("float32")
    _df["inhale_steps"] = (80.0 - _df["exhale_steps"]).astype("float32")

gc.collect()



## === cell 7
RS = cuml.preprocessing.RobustScaler()
FEATURE_COLS = [
    _c for _c in train_df.columns if _c not in ["id", "breath_id", "pressure"]
]
train_df[FEATURE_COLS] = RS.fit_transform(train_df[FEATURE_COLS])
test_df[FEATURE_COLS] = RS.transform(test_df[FEATURE_COLS])

gc.collect()




## === cell 8
def compress_df_fast(df):
    """
    Runtime optimization (equivalent): replace nested Python loops (80 * num_features * num_breaths)
    with cuDF list explosion/expand. Keeps identical per-breath flattening semantics.
    """
    cdf = df.groupby("breath_id").collect().reset_index()

    flatten_cols = list(
        set(
            [_c for _c in cdf.columns if "_back" in _c]
            + [_c for _c in cdf.columns if "_forw" in _c]
            + [_c for _c in cdf.columns if "cross" in _c]
            + [_c for _c in cdf.columns if "u_in" in _c]
            + [_c for _c in cdf.columns if "u_out" in _c]
            + [_c for _c in cdf.columns if "uin" in _c]
            + ["breath_step", "time_step"]
        )
    )

    flatten_cols = sorted(flatten_cols)

    for j, col in enumerate(flatten_cols):
        expanded = cdf[col].list.leaves.values.reshape((-1, ROWS_PER_BREATH))
        colnames = [f"{chr(97 + j)}_{i}" for i in range(ROWS_PER_BREATH)]
        cdf = cudf.concat([cdf, cudf.DataFrame(expanded, columns=colnames)], axis=1)

    if "pressure" in cdf.columns:
        pexp = cdf["pressure"].list.leaves.values.reshape((-1, ROWS_PER_BREATH))
        cdf = cudf.concat(
            [
                cdf,
                cudf.DataFrame(
                    pexp, columns=[f"z_{i}" for i in range(ROWS_PER_BREATH)]
                ),
            ],
            axis=1,
        )
        flatten_cols = flatten_cols + ["pressure"]

    drop_cols = [c for c in flatten_cols if c in cdf.columns]
    if "id" in cdf.columns:
        drop_cols.append("id")
    cdf = cdf.drop(columns=drop_cols)

    REPEAT_COLS = [
        "R",
        "C",
        "_R_20",
        "_R_5",
        "_R_50",
        "_C_10",
        "_C_20",
        "_C_50",
        "breath_duration",
        "exhale_steps",
        "inhale_steps",
    ]
    for rc in REPEAT_COLS:
        if rc in cdf.columns:
            cdf[rc] = cdf[rc].list.get(0)
        else:
            cdf[rc] = cudf.Series(cupy.zeros(len(cdf), dtype=cupy.float32))

    return cdf


train_df = compress_df_fast(train_df)
test_df = compress_df_fast(test_df)

gc.collect()


## === cell 9
PRESSURE_COLS = [f"z_{i}" for i in range(80)]
USE_COLS = [
    _c for _c in train_df.columns if _c not in ["breath_id", "id"] + PRESSURE_COLS
]
BLEND_NEIGHBORS = 100
RESTRICT_RC = True



## === cell 10
RUN_DIAGNOSTICS = False

if RUN_DIAGNOSTICS:
    import matplotlib.pyplot as plt
    from tqdm.notebook import tqdm

    BREATH_ID = 1
    gt_breath = train_df.loc[train_df.breath_id == BREATH_ID].to_pandas()
    BREATH_R = gt_breath.R.values[0]
    BREATH_C = gt_breath.C.values[0]

    df_to_use = (
        train_df.copy()[
            (train_df.R == BREATH_R) & (train_df.C == BREATH_C)
        ].reset_index(drop=True)
        if RESTRICT_RC
        else train_df.copy()
    )

    model = NearestNeighbors(n_neighbors=BLEND_NEIGHBORS, metric="l1")
    model.fit(df_to_use[USE_COLS])

    distances, indices = model.kneighbors(gt_breath[USE_COLS])
    distances, indices = distances[0, 1:], indices[0, 1:]

    gt_breath_pressure = gt_breath[PRESSURE_COLS].squeeze().values
    nn_pressure_df = df_to_use.iloc[indices][PRESSURE_COLS]
    nn_pressure_df.insert(loc=0, name="nn_distance", value=distances)
    nn_pressures = nn_pressure_df[PRESSURE_COLS].values




## === cell 11
if RUN_DIAGNOSTICS:
    pass



## === cell 12
if RUN_DIAGNOSTICS:
    pass



## === cell 13
if RUN_DIAGNOSTICS:
    pass



## === cell 14
if RUN_DIAGNOSTICS:
    pass



## === cell 15
pass



## === cell 16
NN_TO_USE = 10

breath_ids_all = []
mean_test_pressures_all = []
median_test_pressures_all = []

rc_pairs = test_df[["R", "C"]].drop_duplicates().to_pandas().values.tolist()

for _R, _C in rc_pairs:
    test_df_to_use = test_df[(test_df.R == _R) & (test_df.C == _C)].reset_index(
        drop=True
    )
    train_df_to_use = train_df[(train_df.R == _R) & (train_df.C == _C)].reset_index(
        drop=True
    )

    model = NearestNeighbors(n_neighbors=NN_TO_USE, metric="l1")
    model.fit(train_df_to_use[USE_COLS])

    distances, indices = model.kneighbors(test_df_to_use[USE_COLS])
    distances_cp = distances.values  # cupy ndarray shape [n_test_group, NN]
    indices_cp = indices.values  # cupy ndarray shape [n_test_group, NN]

    nn_pressures = train_df_to_use.iloc[indices_cp.reshape(-1)][PRESSURE_COLS].values
    nn_pressures = nn_pressures.reshape(
        (len(test_df_to_use), NN_TO_USE, ROWS_PER_BREATH)
    )

    w = distances_cp.astype(cupy.float32)
    wsum = w.sum(axis=1, keepdims=True)
    mean_pressures = (nn_pressures * w[:, :, None]).sum(axis=1) / wsum

    median_pressures = cupy.median(nn_pressures, axis=1)

    breath_ids_all.append(test_df_to_use["breath_id"].to_pandas().values)
    mean_test_pressures_all.append(cupy.asnumpy(mean_pressures))
    median_test_pressures_all.append(cupy.asnumpy(median_pressures))

breath_ids = np.concatenate(breath_ids_all, axis=0)
mean_test_pressures = np.concatenate(mean_test_pressures_all, axis=0)
median_test_pressures = np.concatenate(median_test_pressures_all, axis=0)



## === cell 17
ss_df



## === cell 18
ss_df = pd.merge(
    left=ss_df,
    right=pd.read_csv(
        "../input/ventilator-pressure-prediction/test.csv", usecols=["id", "breath_id"]
    ),
    on="id",
)

ss_df



## === cell 19
pass



## === cell 20
pred_breath = pd.DataFrame(
    {
        "breath_id": breath_ids.astype(np.int64),
        "mean_vec": list(mean_test_pressures),
        "median_vec": list(median_test_pressures),
    }
)

pred_breath = pred_breath.sort_values("breath_id").reset_index(drop=True)

mean_flat = np.vstack(pred_breath["mean_vec"].to_numpy()).reshape(-1)
median_flat = np.vstack(pred_breath["median_vec"].to_numpy()).reshape(-1)

test_meta = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv", usecols=["id", "breath_id"]
)
test_meta["breath_step"] = np.arange(len(test_meta), dtype=np.int32) % ROWS_PER_BREATH

breath_to_idx = pd.Series(
    pred_breath.index.values, index=pred_breath["breath_id"].values
)

row_block = breath_to_idx.loc[test_meta["breath_id"].values].values.astype(np.int64)
row_idx = row_block * ROWS_PER_BREATH + test_meta["breath_step"].values

ss_df["median_pressure"] = median_flat[row_idx]
ss_df["mean_pressure"] = mean_flat[row_idx]
ss_df["pressure"] = (ss_df["mean_pressure"] + ss_df["median_pressure"]) / 2.0

ss_df



## === cell 21
for x in ["median", "mean", None]:
    p_name = "pressure" if not x else f"{x}_pressure"
    ss_df[["id", p_name]].to_csv(f"{p_name}_submission.csv", index=False)



## === cell 22
if RUN_DIAGNOSTICS:
    BLEND_NEIGHBORS = 1000
    gt_breath = train_df.iloc[:1].to_pandas()  # placeholder
    model = NearestNeighbors(n_neighbors=BLEND_NEIGHBORS, metric="l1")
    model.fit(train_df[USE_COLS])
    distances, indices = model.kneighbors(gt_breath[USE_COLS])
    distances, indices = distances[0, 1:], indices[0, 1:]
