# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.10

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

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

        cupy = None
        cuml = None
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

if DO_CLUSTERING and (cuml is not None):
    train_df = cudf.read_csv(TRAIN_CSV)
    test_df = cudf.read_csv(TEST_CSV)
else:
    DO_CLUSTERING = False
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

ss_df = pd.read_csv(SS_CSV)

ROWS_PER_BREATH = 80

if DO_CLUSTERING:
    train_df["breath_step"] = cudf.Series(
        cupy.arange(len(train_df), dtype=cupy.int32) % ROWS_PER_BREATH
    )
    test_df["breath_step"] = cudf.Series(
        cupy.arange(len(test_df), dtype=cupy.int32) % ROWS_PER_BREATH
    )
    POSSIBLE_PRESSURES = train_df["pressure"].unique().sort_values().values
else:
    train_df["breath_step"] = (
        np.arange(len(train_df), dtype=np.int32) % ROWS_PER_BREATH
    ).astype(np.int16)
    test_df["breath_step"] = (
        np.arange(len(test_df), dtype=np.int32) % ROWS_PER_BREATH
    ).astype(np.int16)
    POSSIBLE_PRESSURES = np.sort(train_df["pressure"].unique())

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
        df[f"u_in_diff_{i}_forw"] = (df["u_in"] - df[f"u_in_{i}_forw"]).astype(
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

if not DO_CLUSTERING:
    raise RuntimeError(
        "This notebook requires GPU (cuDF/cuML) for NearestNeighbors and fast feature pipeline. "
        "Enable GPU in the Kaggle notebook settings."
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



## --- ERROR in cell 8, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNotImplementedError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3788262457.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     65[0m [0;34m[0m[0m
[1;32m     66[0m [0;34m[0m[0m
[0;32m---> 67[0;31m [0mtrain_df[0m [0;34m=[0m [0mcompress_df_fast[0m[0;34m([0m[0mtrain_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     68[0m [0mtest_df[0m [0;34m=[0m [0mcompress_df_fast[0m[0;34m([0m[0mtest_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     69[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3788262457.py[0m in [0;36mcompress_df_fast[0;34m(df)[0m
[1;32m     27[0m     [0;32mif[0m [0;34m"pressure"[0m [0;32min[0m [0mcdf[0m[0;34m.[0m[0mcolumns[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     28[0m         [0mpexp[0m [0;34m=[0m [0mcdf[0m[0;34m[[0m[0;34m"pressure"[0m[0;34m][0m[0;34m.[0m[0mlist[0m[0;34m.[0m[0mleaves[0m[0;34m.[0m[0mvalues[0m[0;34m.[0m[0mreshape[0m[0;34m([0m[0;34m([0m[0;34m-[0m[0;36m1[0m[0;34m,[0m [0mROWS_PER_BREATH[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 29[0;31m         cdf = cudf.concat(
[0m[1;32m     30[0m             [
[1;32m     31[0m                 [0mcdf[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/cudf/core/reshape.py[0m in [0;36mconcat[0;34m(objs, axis, join, ignore_index, keys, levels, names, verify_integrity, sort)[0m
[1;32m    414[0m                 [0;32mfor[0m [0mname[0m[0;34m,[0m [0mcol[0m [0;32min[0m [0mo[0m[0;34m.[0m[0m_column_labels_and_values[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    415[0m                     [0;32mif[0m [0mname[0m [0;32min[0m [0mresult_data[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 416[0;31m                         raise NotImplementedError(
[0m[1;32m    417[0m                             [0;34mf"A Column with duplicate name found: {name}, cuDF "[0m[0;34m[0m[0;34m[0m[0m
[1;32m    418[0m                             [0;34mf"doesn't support having multiple columns with "[0m[0;34m[0m[0;34m[0m[0m

[0;31mNotImplementedError[0m: A Column with duplicate name found: z_0, cuDF doesn't support having multiple columns with same names yet.

## === cell 9
PRESSURE_COLS = [f"z_{i}" for i in range(80)]
USE_COLS = [
    _c for _c in train_df.columns if _c not in ["breath_id", "id"] + PRESSURE_COLS
]
BLEND_NEIGHBORS = 100
RESTRICT_RC = True
