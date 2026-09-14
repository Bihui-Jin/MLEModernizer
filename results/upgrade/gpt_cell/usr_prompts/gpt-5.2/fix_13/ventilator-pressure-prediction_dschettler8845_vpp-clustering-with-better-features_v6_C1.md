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

0.6658

# 6. Current score

9.34912

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'The timeout is dominated by (1) repeated cuDF→pandas→cuDF conversions during feature engineering and per-breath aggregations, (2) the extremely expensive `compress_df` Python-loop that creates ~tens of thousands of columns one-by-one, and (3) the inner Python loop over every test breath when computing NN predictions and later mapping predictions back to rows. The refactor keeps the exact same features, scaling, nearest-neighbor model, and prediction formulas, but computes them with GPU-native cuDF operations (groupby shifts, cumsums, transforms) and replaces the column-by-column flattening with a single vectorized reshape to 2D matrices. Finally, NN prediction is vectorized per (R,C) group (no per-breath Python loop) and submission construction uses array indexing rather than dict lookups and row loops, preserving identical semantics with negligible float differences.'
- What this solution (achieved 17.65486) has done: 'Diagnosis: The crash happens immediately when importing TensorFlow in cell 0, before any of your notebook logic runs. This specific `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` is a known incompatibility between TensorFlow 2.18 and newer `protobuf` (here `protobuf==6.33.0`). Since we cannot change installed packages, we must force TensorFlow to use the pure-Python protobuf implementation before importing it.

Patch summary: In cell 0 only, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version fallback) in `os.environ` before `import tensorflow as tf`. This avoids the failing C++ protobuf path that triggers the missing `GetPrototype` attribute. No other logic, model code, or downstream variables are changed.

Updated cells:'
- What this solution (achieved 17.65486) has done: 'The crash happens immediately after importing TensorFlow due to an incompatibility between TensorFlow 2.18 and the currently installed `protobuf` (6.x), which triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The existing environment-variable workaround forcing the pure-Python protobuf implementation does not resolve this with protobuf 6.x. The minimal deterministic fix is to pin protobuf to the supported major version range at runtime by installing a compatible protobuf (<5) before importing TensorFlow, then importing TensorFlow normally. This change is localized to cell 0 and preserves all downstream variables and behavior expected by cell 1.'
- What this solution (achieved 10.28217) has done: 'Diagnosis: The crash happens in cell 9 because recent cuDF versions no longer provide `cudf.DataFrame.from_gpu_matrix`, so attempting to build a DataFrame from a CuPy matrix via that API raises `AttributeError`. The core logic expects to convert the reshaped CuPy arrays `X` (and `y`) into cuDF columns efficiently. We can preserve the exact same data/shape semantics by converting the CuPy matrices to a pandas DataFrame (via NumPy on host) and then into cuDF, which is supported and deterministic. This change is localized to the buggy cell and keeps the output schema (`breath_id`, feature columns, and `z_` pressure columns) unchanged for cell 10.

Patch summary: Replace `cudf.DataFrame.from_gpu_matrix(...)` calls with a small helper that converts a 2D CuPy array into a cuDF DataFrame via `cupy.asnumpy` + `pd.DataFrame` + `cudf.from_pandas`, then assign those columns exactly as before. No other logic, columns, or ordering are changed.

Updated cells:'
- What this solution (achieved 8.68171) has done: 'Your MAE is very high mainly because the current pipeline predicts pressures for the expiratory phase too, while Kaggle only scores inspiratory timesteps (`u_out==0`); setting expiratory predictions to 0 (or any value) can heavily hurt MAE if left uncontrolled. To move the score toward your target with minimal logic change, I keep the same feature engineering, scaling, nearest-neighbor retrieval, and weighted/median blending, but post-process the final per-row predictions by forcing `pressure=0` wherever `u_out==1` (using the true `u_out` from test). I also fix a small bug in `add_features_cudf` where a forward `u_in_diff` feature accidentally overwrites a backward one, which should modestly improve neighbor matching without changing the overall approach. Finally, I add a safety clip to the known discrete pressure grid (common for this competition) to reduce MAE from off-grid values while keeping semantics intact.'
- What this solution (achieved 9.353) has done: 'Your current MAE is still far from the target, so the highest-impact minimal changes are to (1) stop distorting the R/C grouping by scaling `R` and `C` then using exact equality (this can break grouping and neighbor search), and (2) ensure expiratory predictions don’t introduce unnecessary error by holding the last inspiratory pressure rather than forcing 0. These keep the same core approach (feature engineering → robust scaling → kNN → mean/median blend → optional snapping) but fix two issues that commonly inflate MAE. I also make the pressure snapping respect the “expiratory not scored” logic by applying it after the expiratory overwrite. All I/O paths and submission schema stay the same.'
- What this solution (achieved 8.62694) has done: 'I make two minimal, score-relevant fixes that should reduce MAE substantially without changing your core kNN + feature-engineering approach. First, I preserve `R` and `C` from being scaled (they are used for exact equality grouping; scaling breaks the grouping and can silently force fallback to all-train, hurting neighbors). Second, I fix the `sub.merge(... on="id")` alignment issue: your `id` column is only 1..2000 repeated per breath, so merging on `id` corrupts the row mapping and makes predictions land on the wrong rows, inflating MAE. The rest of the pipeline (features, RobustScaler on other features, kNN, mean/median blend, expiratory hold, snapping to pressure grid, and submission writing) remains the same.'
- What this solution (achieved 10.31485) has done: 'Your current score is still far from the target (lower-is-better), so we need a genuine accuracy lift while keeping the kNN+feature pipeline intact. The biggest remaining issue is that you’re computing nearest neighbors across the entire 80-step breath including expiratory timesteps, even though only inspiratory (`u_out==0`) is scored; this can distort neighbor selection and the averaged breath-shape. With minimal changes, we (1) add a per-timestep inspiratory mask into the flattened features so neighbor matching emphasizes the scored portion, and (2) compute distances/weights using only the inspiratory positions (by zeroing expiratory feature blocks for both train and test) while keeping the same kNN model, columns, blending, expiratory hold, and pressure snapping. These changes preserve your core logic and should move MAE substantially downward toward the target.'
- What this solution (achieved 8.6133) has done: 'Your score is far worse than the target (lower-is-better), so we need a real accuracy lift while keeping your kNN + flattened-breath feature core intact. The biggest remaining issue is that the inspiratory mask is applied after scaling, so it no longer behaves like a clean 0/1 mask and the “zero expiratory blocks” step becomes inconsistent; we preserve a binary `insp_mask_raw` and use it for masking after flattening. Next, we apply the mask only to truly time-varying flattened blocks (not the repeated per-breath constant columns like R/C/duration), so we don’t accidentally erase useful grouping signals. Finally, we compute mean/median breath predictions using inspiratory-only weights (distances derived from inspiratory parts) while keeping the same neighbor model and blending, which should reduce MAE toward your target without changing the fundamental approach.'
- What this solution (achieved 9.34912) has done: 'Diagnosis: `apply_insp_mask_on_flattened_features()` assumes `USE_COLS` contains only time-flattened (80-step) features, so it errors when `len(USE_COLS) % 80 != 0`. However, `compress_df_fast()` appends per-breath constant columns (e.g., `R`, `C`, `breath_duration`, etc.) that are not repeated 80 times, and these are included in `USE_COLS`, breaking the divisibility check. The core intent of this cell is to apply the inspiratory mask only to time-varying flattened blocks, so we should simply exclude non-80-step columns from masking rather than raising.

Patch summary: In cell 11, filter `use_cols` down to only those columns that belong to full 80-length flattened blocks (i.e., the largest prefix of columns divisible by `rows_per_breath`), apply masking to those, and leave any remaining per-breath constant columns untouched. This preserves the existing masking logic and keeps `USE_COLS` unchanged for later cells (k+1).

Updated cells: Only cell 11 is modified.

Compatibility notes for cell k+1: `train_df`, `test_df`, and `USE_COLS` remain the same objects and column sets; we only modify values in the time-flattened columns. Cell 12 still select `train_df_to_use[USE_COLS]` / `test_df_to_use[USE_COLS]` successfully.

Assumptions: `USE_COLS` is ordered such that all flattened 80-step columns come first (as produced by `compress_df_fast()`), and any appended per-breath constant columns come last; this matches the construction in cell 9.'

# 9. Code solution

## === cell 0
print("\n... IMPORTS STARTING ...\n")

import os, sys, gc, math, time, random, warnings
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

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

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Could not set TF memory growth:", e)



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

train_df["breath_step"] = (
    cudf.Series(cupy.arange(len(train_df), dtype=cupy.int32)) % ROWS_PER_BREATH
).astype("uint8")
test_df["breath_step"] = (
    cudf.Series(cupy.arange(len(test_df), dtype=cupy.int32)) % ROWS_PER_BREATH
).astype("uint8")

POSSIBLE_PRESSURES = train_df.pressure.unique().sort_values().to_pandas().values
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


def add_features_cudf(
    df,
    U_IN_N_FORWARD=3,
    U_IN_N_BACKWARD=3,
    U_OUT_N_FORWARD=1,
    U_OUT_N_BACKWARD=1,
    use_rc=True,
):
    print("\n... Add general features (GPU/cuDF) ...\n")
    df["uin_auc"] = df["time_step"] * df["u_in"]
    df["uin_auc"] = df.groupby("breath_id")["uin_auc"].cumsum()
    df["uin_csum"] = df.groupby("breath_id")["u_in"].cumsum()
    df["cross3"] = df["time_step"] * df["u_in"]
    df["cross3_sqd_1"] = df["time_step"] * (df["u_in"] ** 2)
    df["cross3_sqd_2"] = (df["time_step"] ** 2) * df["u_in"]
    df["cross3_cubed_1"] = df["time_step"] * (df["u_in"] ** 3)
    df["cross3_cubed_2"] = (df["time_step"] ** 3) * df["u_in"]

    df["insp_mask_raw"] = (df["u_out"] == 0).astype("uint8")
    df["insp_mask"] = df["insp_mask_raw"].astype("uint8")

    print("\t... Add lag and advance UIN features ...")
    gb_uin = df.groupby("breath_id")["u_in"]
    for i in range(1, U_IN_N_BACKWARD + 1):
        df[f"u_in_{i}_back"] = gb_uin.shift(i).fillna(0)
    for i in range(1, U_IN_N_FORWARD + 1):
        df[f"u_in_{i}_forw"] = gb_uin.shift(-i).fillna(0)

    print("\t... Add lag and advance UOUT features ...")
    gb_uout = df.groupby("breath_id")["u_out"]
    for i in range(1, U_OUT_N_BACKWARD + 1):
        df[f"u_out_{i}_back"] = gb_uout.shift(i).fillna(0)
    for i in range(1, U_OUT_N_FORWARD + 1):
        df[f"u_out_{i}_forw"] = gb_uout.shift(-i).fillna(0)

    print("\t... Add UIN and UOUT `diff` features ...")
    for i in range(1, U_IN_N_BACKWARD + 1):
        df[f"u_in_diff_{i}_back"] = df["u_in"] - df[f"u_in_{i}_back"]
    for i in range(1, U_OUT_N_BACKWARD + 1):
        df[f"u_out_diff_{i}_back"] = df["u_out"] - df[f"u_out_{i}_back"]
    for i in range(1, U_IN_N_FORWARD + 1):
        df[f"u_in_diff_{i}_forw"] = df["u_in"] - df[f"u_in_{i}_forw"]
    for i in range(1, U_OUT_N_FORWARD + 1):
        df[f"u_out_diff_{i}_forw"] = df["u_out"] - df[f"u_out_{i}_forw"]

    print("\t... Add categorical features ...")
    if use_rc:
        df["_R"] = df["R"].astype("str")
        df["_C"] = df["C"].astype("str")
        dR = cudf.get_dummies(df["_R"], prefix="_R")
        dC = cudf.get_dummies(df["_C"], prefix="_C")
        df = df.drop(columns=["_R", "_C"])
        df = df.join(dR).join(dC)

    print("\t... Reset dtypes for lower memory usage ...")
    for c in df.columns:
        if c in ["u_out", "breath_step", "insp_mask", "insp_mask_raw"]:
            df[c] = df[c].astype("uint8")
        else:
            if str(df[c].dtype) == "float64":
                df[c] = df[c].astype("float32")

    gc.collect()
    gc.collect()
    return df




## === cell 6
print("\n... ADDING FEATURES TO TRAIN DATAFRAME ...\n")
train_df = add_features_cudf(train_df, use_rc=True)

print("\n... ADDING FEATURES TO TEST DATAFRAME ...\n")
test_df = add_features_cudf(test_df, use_rc=True)

display(train_df.head(3))
display(test_df.head(3))



## === cell 7
for _df in (test_df, train_df):
    _df["breath_duration"] = _df.groupby("breath_id")["time_step"].transform("max")
    _df["exhale_steps"] = _df.groupby("breath_id")["u_out"].transform("sum")
    _df["inhale_steps"] = 80 - _df["exhale_steps"]

display(train_df.head())
display(test_df.head())



## === cell 8
RS = cuml.preprocessing.RobustScaler()
FEATURE_COLS = [
    _c for _c in train_df.columns if _c not in ["id", "breath_id", "pressure"]
]

SCALE_COLS = [c for c in FEATURE_COLS if c not in ["R", "C", "insp_mask_raw"]]

train_df[SCALE_COLS] = RS.fit_transform(train_df[SCALE_COLS])
test_df[SCALE_COLS] = RS.transform(test_df[SCALE_COLS])

display(train_df.head())
display(test_df.head())




## === cell 9
def compress_df_fast(df):
    df = df.sort_values(["breath_id", "breath_step"])

    flatten_cols = list(
        set(
            [_c for _c in df.columns if "_back" in _c]
            + [_c for _c in df.columns if "_forw" in _c]
            + [_c for _c in df.columns if "cross" in _c]
            + [_c for _c in df.columns if "u_in" in _c]
            + [_c for _c in df.columns if "u_out" in _c]
            + [_c for _c in df.columns if "uin" in _c]
            + [_c for _c in df.columns if "uout" in _c]
            + ["breath_step", "time_step", "insp_mask", "insp_mask_raw"]
        )
    )

    n_breaths = int(len(df) // ROWS_PER_BREATH)
    breath_id = df["breath_id"].to_pandas().values[::ROWS_PER_BREATH].astype(np.int64)

    X = df[flatten_cols].to_cupy()  # shape (n_breaths*80, n_features)
    X = X.reshape((n_breaths, ROWS_PER_BREATH, X.shape[1]))  # (B,80,F)
    X = cupy.transpose(X, (0, 2, 1)).reshape((n_breaths, -1))  # (B,F*80)

    out = cudf.DataFrame({"breath_id": breath_id})
    nF = len(flatten_cols)
    colnames = []
    for j in range(nF):
        prefix = f"{chr(97 + j)}_"
        colnames.extend([f"{prefix}{i}" for i in range(ROWS_PER_BREATH)])

    def _cudf_from_cupy_2d(arr_2d, columns):
        return cudf.from_pandas(pd.DataFrame(cupy.asnumpy(arr_2d), columns=columns))

    out[colnames] = _cudf_from_cupy_2d(X, colnames)

    if "pressure" in df.columns:
        y = df["pressure"].to_cupy().reshape((n_breaths, ROWS_PER_BREATH))
        zcols = [f"z_{i}" for i in range(ROWS_PER_BREATH)]
        out[zcols] = _cudf_from_cupy_2d(y, zcols)

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
    first_rows = df.iloc[::ROWS_PER_BREATH]
    for _rc in REPEAT_COLS:
        if _rc in df.columns:
            out[_rc] = first_rows[_rc].to_pandas().values

    return out


train_df = compress_df_fast(train_df)
test_df = compress_df_fast(test_df)

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
def apply_insp_mask_on_flattened_features(df_flat, use_cols, rows_per_breath=80):
    """
    Score-relevant fix:
    - Use the preserved binary mask block (preferring insp_mask_raw if present) so expiratory steps are truly zeroed.
    - Only apply masking to time-varying flattened blocks; do NOT mask repeated per-breath constants (R/C/duration/etc),
      otherwise we can destroy useful signals and hurt neighbor selection.

    Bug fix:
    - USE_COLS may include trailing per-breath constant columns (not repeated 80 times), so len(USE_COLS) may not be
      divisible by rows_per_breath. We mask only the largest prefix that forms complete 80-step flattened blocks and
      leave the remaining columns untouched.
    """
    cols_all = list(use_cols)
    n_steps = rows_per_breath

    n_full = (len(cols_all) // n_steps) * n_steps
    cols = cols_all[:n_full]  # only full flattened blocks (multiple of 80)
    if n_full == 0:
        return df_flat  # nothing to mask safely

    nF = len(cols) // n_steps

    mask_block_cols = None
    for j in range(nF):
        block_cols = cols[j * n_steps : (j + 1) * n_steps]
        joined = " ".join(block_cols)
        if "insp_mask_raw" in joined:
            mask_block_cols = block_cols
            break
    if mask_block_cols is None:
        for j in range(nF):
            block_cols = cols[j * n_steps : (j + 1) * n_steps]
            joined = " ".join(block_cols)
            if "insp_mask" in joined:
                mask_block_cols = block_cols
                break

    if mask_block_cols is None:
        sample = df_flat[cols].head(1024).to_pandas()
        best_j = None
        best_score = -1
        for j in range(nF):
            block = sample.iloc[:, j * n_steps : (j + 1) * n_steps].values
            close01 = np.mean(
                (np.abs(block - 0.0) < 1e-6) | (np.abs(block - 1.0) < 1e-6)
            )
            var = np.std(block)
            score = close01 + 0.01 * var
            if score > best_score:
                best_score = score
                best_j = j
        mask_block_cols = cols[best_j * n_steps : (best_j + 1) * n_steps]

    M = df_flat[mask_block_cols].to_cupy().astype(cupy.float32)  # (B,80)

    sampleB = min(2048, len(df_flat))
    sample = df_flat[cols].head(sampleB).to_pandas()

    constant_blocks = set()
    for j in range(nF):
        block = sample.iloc[:, j * n_steps : (j + 1) * n_steps].values.astype(
            np.float32
        )
        ms = float(np.mean(np.std(block, axis=1)))
        if ms < 1e-8:
            constant_blocks.add(j)

    for j in range(nF):
        block_cols = cols[j * n_steps : (j + 1) * n_steps]
        if block_cols == mask_block_cols:
            continue
        if j in constant_blocks:
            continue
        Xb = df_flat[block_cols].to_cupy().astype(cupy.float32)
        Xb = Xb * M
        df_flat[block_cols] = cudf.from_pandas(
            pd.DataFrame(cupy.asnumpy(Xb), columns=block_cols)
        )

    return df_flat


train_df = apply_insp_mask_on_flattened_features(train_df, USE_COLS, ROWS_PER_BREATH)
test_df = apply_insp_mask_on_flattened_features(test_df, USE_COLS, ROWS_PER_BREATH)


## === cell 12
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
        distances = distances.values  # cupy
        indices = indices.values  # cupy

        nn_pressures = (
            train_df_to_use[PRESSURE_COLS].iloc[indices.reshape(-1)].to_cupy()
        )
        nn_pressures = nn_pressures.reshape(
            (len(test_df_to_use), NN_TO_USE, ROWS_PER_BREATH)
        )  # (B,K,80)

        w = 1.0 / (distances + EPS)  # (B,K)
        w_sum = cupy.sum(w, axis=1, keepdims=True)  # (B,1)
        mean_pred = cupy.sum(nn_pressures * w[:, :, None], axis=1) / w_sum  # (B,80)
        median_pred = cupy.median(nn_pressures, axis=1)  # (B,80)

        breath_ids.append(
            test_df_to_use["breath_id"].to_pandas().values.astype(np.int64)
        )
        mean_test_pressures.append(cupy.asnumpy(mean_pred).astype(np.float32))
        median_test_pressures.append(cupy.asnumpy(median_pred).astype(np.float32))

breath_ids = (
    np.concatenate(breath_ids, axis=0)
    if len(breath_ids)
    else np.array([], dtype=np.int64)
)
mean_test_pressures = (
    np.concatenate(mean_test_pressures, axis=0)
    if len(mean_test_pressures)
    else np.empty((0, ROWS_PER_BREATH), dtype=np.float32)
)
median_test_pressures = (
    np.concatenate(median_test_pressures, axis=0)
    if len(median_test_pressures)
    else np.empty((0, ROWS_PER_BREATH), dtype=np.float32)
)

print(
    "Predicted breaths:", len(breath_ids), "Expected test breaths:", int(len(test_df))
)



## === cell 13
test_meta = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv"), usecols=["id", "breath_id", "u_out"]
)
sub = ss_df.copy()

sub["breath_id"] = test_meta["breath_id"].values
sub["u_out"] = test_meta["u_out"].values

order = np.argsort(breath_ids)
breath_ids_sorted = breath_ids[order]
mean_sorted = mean_test_pressures[order]
median_sorted = median_test_pressures[order]

idx = pd.Index(breath_ids_sorted)
row_idx = idx.get_indexer(sub["breath_id"].values)  # (n_rows,)

within_breath_pos = (
    test_meta.groupby("breath_id").cumcount().values.astype(np.int64)
)  # 0..79

mean_pred = mean_sorted[row_idx, within_breath_pos].astype(np.float32)
median_pred = median_sorted[row_idx, within_breath_pos].astype(np.float32)

sub["mean_pressure"] = mean_pred
sub["median_pressure"] = median_pred
sub["pressure"] = (sub["mean_pressure"] + sub["median_pressure"]) / 2.0

sub["_insp_pressure"] = sub["pressure"].where(sub["u_out"] == 0, np.nan)
sub["_insp_ffill"] = sub.groupby("breath_id")["_insp_pressure"].ffill()
sub["_insp_bfill"] = sub.groupby("breath_id")["_insp_pressure"].bfill()
sub["_hold_pressure"] = sub["_insp_ffill"].fillna(sub["_insp_bfill"]).fillna(0.0)
sub.loc[sub["u_out"] == 1, "pressure"] = sub.loc[
    sub["u_out"] == 1, "_hold_pressure"
].values

sub["pressure"] = POSSIBLE_PRESSURES[
    np.abs(POSSIBLE_PRESSURES[None, :] - sub["pressure"].values[:, None]).argmin(axis=1)
].astype(np.float32)

sub = sub.drop(
    columns=["_insp_pressure", "_insp_ffill", "_insp_bfill", "_hold_pressure"]
)

display(sub.head())



## === cell 14
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



## === cell 15
chk = pd.read_csv("/kaggle/working/submission.csv")
print("Rows:", len(chk), "Cols:", chk.columns.tolist())
print("NaNs:", chk.isna().sum().to_dict())
print("Pressure stats:", chk["pressure"].describe())
print("ID min/max:", chk["id"].min(), chk["id"].max())
