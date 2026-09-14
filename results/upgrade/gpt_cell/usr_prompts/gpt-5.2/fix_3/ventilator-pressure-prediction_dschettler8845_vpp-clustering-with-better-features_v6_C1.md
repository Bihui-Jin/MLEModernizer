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

# 5. Code solution

## === cell 0
print("\n... IMPORTS STARTING ...\n")

import os, sys, gc, math, time, random, warnings
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

import tensorflow as tf

print("\n\tVERSION INFORMATION")
print(f"\t\t– TENSORFLOW VERSION: {tf.__version__}")
print(f"\t\t– NUMPY VERSION: {np.__version__}")

from tqdm.notebook import tqdm

tqdm.pandas()


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
        import cudf, cuml, cupy
        from cuml.neighbors import NearestNeighbors
    else:
        print(f"\n ... RUNNING ON CPU ...\n")
        raise RuntimeError(
            "This solution expects a GPU RAPIDS environment (cudf/cuml)."
        )

N_REPLICAS = strategy.num_replicas_in_sync
print(f"... # OF REPLICAS: {N_REPLICAS} ...\n")
print(f"\n... ACCELERATOR SETUP COMPLETED ...\n")



## === cell 2
print("\n... DATA ACCESS SETUP STARTED ...\n")

if TPU:
    from kaggle_datasets import KaggleDatasets

    DATA_DIR = KaggleDatasets().get_gcs_path("ventilator-pressure-prediction")
else:
    DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"

print(f"\n... DATA DIRECTORY PATH IS:\n\t--> {DATA_DIR}")
print(f"\n... IMMEDIATE CONTENTS OF DATA DIRECTORY IS:")
for file in tf.io.gfile.glob(os.path.join(DATA_DIR, "*")):
    print(f"\t--> {file}")

print("\n\n... DATA ACCESS SETUP COMPLETED ...\n")



## === cell 3
print("\n... BASIC DATA SETUP STARTING ...\n\n")

DO_CLUSTERING = True

print("\n... TRAIN DATAFRAME ..\n")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
train_df = cudf.read_csv(TRAIN_CSV) if DO_CLUSTERING else pd.read_csv(TRAIN_CSV)
display(train_df.head())

print("\n... TEST DATAFRAME ..\n")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
test_df = cudf.read_csv(TEST_CSV) if DO_CLUSTERING else pd.read_csv(TEST_CSV)
display(test_df.head())

print("\n... SAMPLE SUBMISSION DATAFRAME ..\n")
SS_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
ss_df = pd.read_csv(SS_CSV)
display(ss_df.head())

ROWS_PER_BREATH = 80

test_df["breath_step"] = np.arange(ROWS_PER_BREATH).tolist() * (
    len(test_df) // ROWS_PER_BREATH
)
train_df["breath_step"] = np.arange(ROWS_PER_BREATH).tolist() * (
    len(train_df) // ROWS_PER_BREATH
)

POSSIBLE_PRESSURES = train_df.pressure.unique().sort_values().values
PRESSURE_DELTA_STEP = float((POSSIBLE_PRESSURES[1:] - POSSIBLE_PRESSURES[:-1]).mean())

print("\n\n... BASIC DATA SETUP FINISHING ...\n")



## === cell 4
print(f"\n... XLA OPTIMIZATIONS STARTING ...\n")
print(f"\n... CONFIGURE JIT (JUST IN TIME) COMPILATION ...\n")
tf.config.optimizer.set_jit(True)
print(f"\n... XLA OPTIMIZATIONS COMPLETED ...\n")




## === cell 5
def flatten_l_o_l(nested_list):
    """Flatten a list of lists"""
    return [item for sublist in nested_list for item in sublist]


def compute_weighted_median(_values, _weights, redux_factor=1):
    """Compute a simple weighted median approximation via replication (kept for core-logic compatibility)."""
    return np.median(
        flatten_l_o_l(
            [[_v] * int(_w // redux_factor) for _v, _w in zip(_values, _weights)]
        )
    )


def add_features(
    df,
    U_IN_N_FORWARD=3,
    U_IN_N_BACKWARD=3,
    U_OUT_N_FORWARD=1,
    U_OUT_N_BACKWARD=1,
    use_rc=True,
):
    print("\n... Add general features ...\n")
    df["uin_auc"] = df["time_step"] * df["u_in"]
    df["uin_auc"] = df.groupby("breath_id")["uin_auc"].cumsum()
    df["uin_csum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
    df["cross3"] = df["time_step"] * df["u_in"]
    df["cross3_sqd_1"] = df["time_step"] * df["u_in"] ** 2
    df["cross3_sqd_2"] = df["time_step"] ** 2 * df["u_in"]
    df["cross3_cubed_1"] = df["time_step"] * df["u_in"] ** 3
    df["cross3_cubed_2"] = df["time_step"] ** 3 * df["u_in"]

    print("\t... Add lag and advance UIN features ...")
    for i in range(1, U_IN_N_BACKWARD + 1):
        df[f"u_in_{i}_back"] = df.groupby("breath_id")["u_in"].shift(i).fillna(0)
    for i in range(1, U_IN_N_FORWARD + 1):
        df[f"u_in_{i}_forw"] = df.groupby("breath_id")["u_in"].shift(-i).fillna(0)

    print("\t... Add lag and advance UOUT features ...")
    for i in range(1, U_OUT_N_BACKWARD + 1):
        df[f"u_out_{i}_back"] = df.groupby("breath_id")["u_out"].shift(i).fillna(0)
    for i in range(1, U_OUT_N_FORWARD + 1):
        df[f"u_out_{i}_forw"] = df.groupby("breath_id")["u_out"].shift(-i).fillna(0)

    print("\t... Add UIN and UOUT `diff` features ...")
    for i in range(1, U_IN_N_BACKWARD + 1):
        df[f"u_in_diff_{i}_back"] = df["u_in"] - df[f"u_in_{i}_back"]
    for i in range(1, U_OUT_N_BACKWARD + 1):
        df[f"u_out_diff_{i}_back"] = df["u_out"] - df[f"u_out_{i}_back"]
    for i in range(1, U_IN_N_FORWARD + 1):
        df[f"u_in_diff_{i}_back"] = df["u_in"] - df[f"u_in_{i}_forw"]
    for i in range(1, U_OUT_N_FORWARD + 1):
        df[f"u_out_diff_{i}_forw"] = df["u_out"] - df[f"u_out_{i}_forw"]

    print("\t... Add categorical features ...")
    if use_rc:
        df["_R"] = df["R"].astype(str)
        df["_C"] = df["C"].astype(str)
        df["R"] = df["R"] / 50
        df["C"] = df["C"] / 50
        df = pd.get_dummies(df)

    print("\t... Reset dtypes for lower memory usage ...")
    for c in df.columns:
        if c in ["u_out", "breath_step"]:
            df[c] = df[c].astype("uint8")
        elif df[c].dtype == "float64":
            df[c] = df[c].astype("float32")

    gc.collect()
    gc.collect()
    return df




## === cell 6
print("\n... ADDING FEATURES TO TRAIN DATAFRAME ...\n")
train_df = add_features(train_df.to_pandas(), use_rc=True)
train_df = cudf.from_pandas(train_df)

print("\n... ADDING FEATURES TO TEST DATAFRAME ...\n")
test_df = add_features(test_df.to_pandas(), use_rc=True)
test_df = cudf.from_pandas(test_df)

display(train_df.head(3))
display(test_df.head(3))



## === cell 7
test_df_pd = test_df.to_pandas()
test_df_pd["breath_duration"] = test_df_pd.groupby("breath_id")[
    ["time_step"]
].transform("max")
test_df_pd["exhale_steps"] = test_df_pd.groupby("breath_id")[["u_out"]].transform("sum")
test_df_pd["inhale_steps"] = 80 - test_df_pd.groupby("breath_id")[["u_out"]].transform(
    "sum"
)
test_df = cudf.from_pandas(test_df_pd)

train_df_pd = train_df.to_pandas()
train_df_pd["breath_duration"] = train_df_pd.groupby("breath_id")[
    ["time_step"]
].transform("max")
train_df_pd["exhale_steps"] = train_df_pd.groupby("breath_id")[["u_out"]].transform(
    "sum"
)
train_df_pd["inhale_steps"] = 80 - train_df_pd.groupby("breath_id")[
    ["u_out"]
].transform("sum")
train_df = cudf.from_pandas(train_df_pd)

display(train_df.head())
display(test_df.head())



## === cell 8
RS = cuml.preprocessing.RobustScaler()
FEATURE_COLS = [
    _c for _c in train_df.columns if _c not in ["id", "breath_id", "pressure"]
]
train_df[FEATURE_COLS] = RS.fit_transform(train_df[FEATURE_COLS])
test_df[FEATURE_COLS] = RS.transform(test_df[FEATURE_COLS])

display(train_df.head())
display(test_df.head())




## === cell 9
def compress_df(df):
    cdf = df.groupby("breath_id").collect().reset_index()

    flatten_cols = list(
        set(
            [_c for _c in cdf.columns if "_back" in _c]
            + [_c for _c in cdf.columns if "_forw" in _c]
            + [_c for _c in cdf.columns if "cross" in _c]
            + [_c for _c in cdf.columns if "u_in" in _c]
            + [_c for _c in cdf.columns if "u_out" in _c]
            + [_c for _c in cdf.columns if "uin" in _c]
            + [_c for _c in cdf.columns if "uout" in _c]
            + ["breath_step", "time_step"]
        )
    )

    for j, _c in enumerate(flatten_cols):
        for i in range(ROWS_PER_BREATH):
            cdf[f"{chr(97 + j)}_{i}"] = cdf[_c].list.get(i)

    if "pressure" in cdf.columns:
        for i in range(ROWS_PER_BREATH):
            cdf[f"z_{i}"] = cdf["pressure"].list.get(i)
        flatten_cols.append("pressure")

    cdf.drop(columns=flatten_cols + ["id"], inplace=True)

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
    for _rc in REPEAT_COLS:
        if _rc in cdf.columns:
            cdf[_rc] = cdf[_rc].list.get(0)

    return cdf


train_df = compress_df(train_df)
test_df = compress_df(test_df)

display(train_df.head())
display(test_df.head())



## === cell 10
PRESSURE_COLS = [f"z_{i}" for i in range(80)]
USE_COLS = [
    _c for _c in train_df.columns if _c not in ["breath_id", "id"] + PRESSURE_COLS
]

NN_TO_USE = 10
RESTRICT_RC = True

EPS = 1e-6

print("USE_COLS:", len(USE_COLS), "PRESSURE_COLS:", len(PRESSURE_COLS))



## === cell 11

breath_ids = []
mean_test_pressures = []
median_test_pressures = []

unique_R = test_df["R"].unique().to_pandas().tolist()
unique_C = test_df["C"].unique().to_pandas().tolist()

for _R in unique_R:
    for _C in unique_C:
        test_df_to_use = test_df[(test_df.R == _R) & (test_df.C == _C)].reset_index(
            drop=True
        )
        train_df_to_use = train_df[(train_df.R == _R) & (train_df.C == _C)].reset_index(
            drop=True
        )

        if len(test_df_to_use) == 0:
            continue

        if len(train_df_to_use) == 0:
            train_df_to_use = train_df

        model = NearestNeighbors(n_neighbors=NN_TO_USE, metric="l1")
        model.fit(train_df_to_use[USE_COLS])

        distances, indices = model.kneighbors(test_df_to_use[USE_COLS])
        distances = distances.values
        indices = indices.values

        for i in range(len(test_df_to_use)):
            BREATH_ID = int(test_df_to_use.iloc[i]["breath_id"])
            breath_ids.append(BREATH_ID)

            _distances = distances[i].get()
            _indices = indices[i].get()

            nn_pressures = train_df_to_use.iloc[_indices][
                PRESSURE_COLS
            ].values.get()  # (k, 80)

            w = 1.0 / (_distances + EPS)
            mean_test_pressures.append(np.average(nn_pressures, axis=0, weights=w))
            median_test_pressures.append(np.median(nn_pressures, axis=0))

print(
    "Predicted breaths:", len(breath_ids), "Expected test breaths:", int(len(test_df))
)



## === cell 12

test_id_breath = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv"), usecols=["id", "breath_id"]
)
sub = ss_df.copy()
sub = sub.merge(test_id_breath, on="id", how="left")

order = np.argsort(np.array(breath_ids))
breath_ids_sorted = np.array(breath_ids)[order]
mean_sorted = np.array(mean_test_pressures, dtype=np.float32)[order]
median_sorted = np.array(median_test_pressures, dtype=np.float32)[order]

mean_map = {int(b): mean_sorted[i] for i, b in enumerate(breath_ids_sorted)}
median_map = {int(b): median_sorted[i] for i, b in enumerate(breath_ids_sorted)}

within_breath_pos = (
    test_id_breath.groupby("breath_id").cumcount().values
)  # 0..79 for each breath
breath_id_arr = test_id_breath["breath_id"].values

mean_pred = np.empty(len(sub), dtype=np.float32)
median_pred = np.empty(len(sub), dtype=np.float32)

for i in range(len(sub)):
    b = int(breath_id_arr[i])
    t = int(within_breath_pos[i])
    mean_pred[i] = mean_map[b][t]
    median_pred[i] = median_map[b][t]

sub["mean_pressure"] = mean_pred
sub["median_pressure"] = median_pred
sub["pressure"] = (sub["mean_pressure"] + sub["median_pressure"]) / 2.0

display(sub.head())



## === cell 13
submission = sub[["id", "pressure"]].copy()
submission.to_csv("/kaggle/working/submission.csv", index=False)

sub[["id", "mean_pressure"]].rename(columns={"mean_pressure": "pressure"}).to_csv(
    "/kaggle/working/mean_pressure_submission.csv", index=False
)
sub[["id", "median_pressure"]].rename(columns={"median_pressure": "pressure"}).to_csv(
    "/kaggle/working/median_pressure_submission.csv", index=False
)

print("Wrote:", "/kaggle/working/submission.csv")
display(pd.read_csv("/kaggle/working/submission.csv").head())


## === cell 14
chk = pd.read_csv("/kaggle/working/submission.csv")
print("Rows:", len(chk), "Cols:", chk.columns.tolist())
print("NaNs:", chk.isna().sum().to_dict())
print("Pressure stats:", chk["pressure"].describe())
