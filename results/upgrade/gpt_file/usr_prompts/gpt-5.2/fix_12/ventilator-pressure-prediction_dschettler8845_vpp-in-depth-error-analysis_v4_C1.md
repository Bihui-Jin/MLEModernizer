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

geopandas==0.14.4
google-api-python-client==2.177.0
imageio==2.37.0
imageio-ffmpeg==0.6.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

0.1608598290068039

# 6. Current score

10.86378

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'I fix the two blockers preventing any submission from being written: (1) the protobuf/TensorFlow “MessageFactory GetPrototype” crash during imports by forcing the Python protobuf implementation, and (2) the missing pretrained model directory by switching to the correct available competition input path and adding a safe fallback if no fold models exist. I also make model-loading robust to both SavedModel directories and single-file `.h5/.keras` formats (without changing inference semantics when models are present). Finally, I ensure the script always writes a valid `submission.csv` with the required `id,pressure` columns even if it has to fall back to a naive baseline.'
- What this solution (achieved 10.86378) has done: 'You’re crashing on TensorFlow import due to an old protobuf API call (“MessageFactory.GetPrototype”) that gets triggered in this environment; setting the protobuf env var inside the notebook is too late unless it’s done before *any* protobuf-related import. I move the environment setup to the very top (and also force the pure-Python protobuf runtime) so TensorFlow imports cleanly and the pipeline runs end-to-end. I also fix a small feature-engineering bug where forward `u_in` diffs were accidentally overwriting the backward diff columns (this is score-relevant but doesn’t change the model/training approach). Finally, I keep submission writing unchanged but ensure IDs align and the file is always produced.'
- What this solution (achieved 10.86378) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment variables to the very top of the script *before any TensorFlow/protobuf import can occur*, and I also pin `protobuf` to use the Python implementation plus disable C++ acceleration via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` early. Then I correct a feature-engineering leakage/bug: `measured_volume` currently uses a global `.diff()` which crosses breath boundaries; I make it a per-`breath_id` diff (same feature intent, but correct), which should substantially improve MAE toward your target without changing the model architecture or inference approach. Finally, I make the SavedModel wrapper input shape deterministic by setting the exact `N_FEATURES` dimension, avoiding `None`-shape issues across Keras/TFSMLayer, and keep the submission writing format unchanged.'
- What this solution (achieved 10.86378) has done: 'I fix the immediate TensorFlow/protobuf import crash by moving the protobuf environment variables to the very top *before any other import* and also forcing the pure-Python protobuf runtime via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, which is required in this Kaggle image. Then I make two small, score-relevant feature fixes that preserve the intended feature semantics: compute `measured_volume` as a per-breath cumulative integral (instead of resetting each row) and ensure it doesn’t bleed across breaths, and remove duplicated feature assignments that were redundant. Finally, I keep the same inference/ensembling logic but add a safe pressure post-processing step (round-to-known-pressure grid) which matches the competition’s discrete pressure levels and typically reduces MAE without changing the model architecture.'
- What this solution (achieved 10.86378) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment variables to the very top of the script (before any other import can indirectly load protobuf) and by additionally forcing the pure-Python protobuf runtime via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus disabling the C++ implementation. Then, to improve MAE toward your target without changing the model/inference approach, I fix the feature bug where `measured_pressure` was accidentally adding `p_0` twice (a large constant offset that can explain the very poor score). Finally, I keep the same model loading/ensembling/submission logic, ensuring a valid `submission.csv` is always written with `id,pressure`.'
- What this solution (achieved 10.86378) has done: 'I fix the TensorFlow import crash by setting the protobuf environment variables before any TensorFlow/protobuf-related imports and (critically) forcing `protobuf<5` compatibility by using the Python implementation early; this resolves the `MessageFactory.GetPrototype` AttributeError in this Kaggle image. Then I correct a major feature bug in `measured_pressure` where `p_0` is being added as a constant offset (your current defaults make it +1 cmH2O everywhere), which severely hurts MAE; I set the default `p_0=0` to match the intended physics-based feature while keeping the rest of the feature pipeline intact. Finally, I keep model loading/inference the same but ensure the submission is always written as `submission.csv` with `id,pressure` and that test IDs align even with padding.'
- What this solution (achieved 10.86378) has done: 'I fix the TensorFlow/protobuf import crash by moving the protobuf environment variables to the very top of the script and also forcing `protobuf` to use the pure-Python backend before any other import can indirectly load it. Then I keep your model loading/inference logic intact, but make the fallback submission path always match the `test.csv` row order (to avoid any potential ID/padding misalignment). Finally, I ensure the submission file is always written as `submission.csv` with exactly the required `id,pressure` columns.'
- What this solution (achieved 10.86378) has done: 'You’re failing immediately on `import tensorflow as tf` due to an incompatibility between TensorFlow 2.18 and the installed `protobuf==6.x`, which triggers `MessageFactory.GetPrototype` errors; setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` alone doesn’t fix this in all cases. The minimal robust fix in a Kaggle offline environment is to ensure a TF-compatible protobuf (<5) is present by installing a compatible wheel at runtime before importing TensorFlow. I add a tiny “preflight” cell that (a) sets the protobuf env vars *before any protobuf/tensorflow import*, and (b) safely pip-installs `protobuf==4.25.3` if the current protobuf major version is too new, then proceeds with the rest of your pipeline unchanged. This should unblock execution and allow your existing model inference/ensembling + pressure snapping to run, which is necessary to move MAE drastically toward your target (your current 10.86 is consistent with “no model ran / fallback baseline”).'
- What this solution (achieved 10.86378) has done: 'Your current MAE (10.86) strongly suggests the script is not actually using the intended pretrained fold models (or is feeding them mis-shaped/mis-ordered inputs), so it falls back to a crude baseline that scores terribly. I make the smallest score-relevant fixes: (1) point `MODEL_DIR` only to locations that actually contain `best_model_fold_*` assets and explicitly search for those models; (2) ensure the `id` column in the submission comes from `test.csv` (not `sample_submission.csv`) so predictions align 1:1 with test rows; and (3) keep the exact same feature pipeline and snapping, but add a safety check that model outputs are reshaped to `(n_rows,)` correctly even if the SavedModel returns `(batch, 80, 1)` or `(batch, 80)`. These changes preserve your core approach (feature engineering + RobustScaler + fold ensemble + pressure snapping) while removing the main reason you’re stuck at baseline-level MAE, which should move you substantially toward the 0.16 target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP", "1")

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    _pb_major = int(_pb_ver.split(".")[0])
except Exception:
    _pb_major = None

if _pb_major is None or _pb_major >= 5:
    import sys
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )

import gc
import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:

    def display(x):
        print(x)




## === cell 1
print("\n... IMPORTS STARTING ...\n")
print("\n\tVERSION INFORMATION")

import random
import matplotlib

import tensorflow as tf

print(f"\t\t– TENSORFLOW VERSION: {tf.__version__}")
print(f"\t\t– MATPLOTLIB VERSION: {matplotlib.__version__}")

try:
    import tensorflow_addons as tfa  # noqa: F401

    print(f"\t\t– TENSORFLOW ADDONS VERSION: {tfa.__version__}")
except Exception as e:
    tfa = None
    print(f"\t\t– TENSORFLOW ADDONS IMPORT FAILED (ignored): {type(e).__name__}: {e}")

pd.options.mode.chained_assignment = None

import sklearn

print(f"\t\t– SKLEARN VERSION: {sklearn.__version__}")
from sklearn.preprocessing import RobustScaler

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm


def seed_it_all(seed=7):
    """Attempt to be reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


print("\n\n... IMPORTS COMPLETE ...\n")
print("\n... SEEDING FOR DETERMINISTIC BEHAVIOUR ...\n")
seed_it_all()



## === cell 2
print(f"\n... ACCELERATOR SETUP STARTING ...\n")

try:
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()
except Exception:
    TPU = None

if TPU:
    print(f"\n... RUNNING ON TPU - {TPU.master()}...")
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    strategy = tf.distribute.experimental.TPUStrategy(TPU)
else:
    print(f"\n... RUNNING ON CPU/GPU ...")
    strategy = tf.distribute.get_strategy()

N_REPLICAS = strategy.num_replicas_in_sync
print(f"... # OF REPLICAS: {N_REPLICAS} ...\n")
print(f"\n... ACCELERATOR SETUP COMPLTED ...\n")



## === cell 3
print("\n... DATA ACCESS SETUP STARTED ...\n")

DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"

CANDIDATE_MODEL_DIRS = [
    "/kaggle/input/vpp-synthetic-adventure-lstm-w-sofia-features",
    "/kaggle/input/ventilator-pressure-prediction/ventilator-pressure-prediction",
    "/kaggle/input/ventilator-pressure-prediction",
]


def _dir_has_any_fold_model(d):
    if not os.path.exists(d):
        return False
    for i in range(6):
        base = os.path.join(d, f"best_model_fold_{i}")
        if (
            os.path.exists(base)
            or os.path.exists(base + ".h5")
            or os.path.exists(base + ".keras")
        ):
            return True
    return False


MODEL_DIR = next((p for p in CANDIDATE_MODEL_DIRS if _dir_has_any_fold_model(p)), None)
if MODEL_DIR is None:
    MODEL_DIR = next(
        (p for p in CANDIDATE_MODEL_DIRS if os.path.exists(p)), CANDIDATE_MODEL_DIRS[-1]
    )

print(f"\n... DATA DIRECTORY PATH IS:\n\t--> {DATA_DIR}")
print(f"\n... MODEL DIRECTORY PATH IS:\n\t--> {MODEL_DIR}")

print(f"\n... IMMEDIATE CONTENTS OF DATA DIRECTORY IS:")
for file in tf.io.gfile.glob(os.path.join(DATA_DIR, "*")):
    print(f"\t--> {file}")

print("\n\n... DATA ACCESS SETUP COMPLETED ...\n")



## === cell 4
N_FOLDS = 6
ROWS_PER_BREATH = 80
REPLICA_BATCH_SIZE = 64
OVERALL_BATCH_SIZE = N_REPLICAS * REPLICA_BATCH_SIZE

print("\n... BASIC DATA SETUP STARTING ...\n\n")

print("\n... TRAIN DATAFRAME ...\n")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
train_df = pd.read_csv(TRAIN_CSV)
display(train_df.head())

print("\n... TEST DATAFRAME ..\n")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
test_df = pd.read_csv(TEST_CSV)
display(test_df.head())

print("\n... SAMPLE SUBMISSION DATAFRAME ..\n")
SS_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
ss_df = pd.read_csv(SS_CSV)
display(ss_df.head())

print("\n... VAL DATAFRAMES (optional) ...\n")
fold_val_df_map = {}
for i in range(N_FOLDS):
    p = os.path.join(MODEL_DIR, f"fold_{i}_val.csv")
    if os.path.exists(p):
        fold_val_df_map[f"fold_{i}"] = pd.read_csv(p)

if len(fold_val_df_map) > 0:
    display(next(iter(fold_val_df_map.values())).head())
else:
    print(
        "\t--> No fold_{i}_val.csv files found; continuing without validation fold data."
    )

N_TRAIN_BREATHS = train_df["breath_id"].nunique()
N_TEST_BREATHS = test_df["breath_id"].nunique()

print(f"\n... N_TRAIN_BREATHS: {N_TRAIN_BREATHS}")
print(f"... N_TEST_BREATHS: {N_TEST_BREATHS}")

POSSIBLE_PRESSURES = sorted(np.array(train_df.pressure.value_counts().keys()))
APPROX_PRESSURE_DELTA_STEP = 0.0703021454512

print("\n\n... BASIC DATA SETUP FINISHING ...\n")



## === cell 5
print(f"\n... XLA OPTIMIZATIONS STARTING ...\n")
print(f"\n... CONFIGURE JIT (JUST IN TIME) COMPILATION ...\n")
tf.config.optimizer.set_jit(True)
print(f"\n... XLA OPTIMIZATIONS COMPLETED ...\n")




## === cell 6
def flatten_l_o_l(nested_list):
    """Flatten a list of lists."""
    return [item for sublist in nested_list for item in sublist]


def add_features(
    df,
    U_IN_N_FORWARD=7,
    U_IN_N_BACKWARD=7,
    U_OUT_N_FORWARD=2,
    U_OUT_N_BACKWARD=2,
    V_0=1,
    p_0=0,
    r_0=1,
    use_rc=True,
):
    df = df.copy()

    dt = df.groupby("breath_id")["time_step"].diff().fillna(0).astype("float32")
    df["measured_volume"] = (df["u_in"].astype("float32") * dt).groupby(
        df["breath_id"]
    ).cumsum() + float(V_0)

    r_t = (3 * df["measured_volume"] / 4 / np.pi) ** (1 / 3)

    df["measured_pressure"] = (
        float(p_0) + (1 - (r_t / float(r_0)) ** 6) * 1 / (r_t * float(r_0) ** 2)
    ).fillna(0)

    print("\n... Add general features ...\n")
    df["uin_auc"] = df["time_step"] * df["u_in"]
    df["uin_auc"] = df.groupby("breath_id")["uin_auc"].cumsum()
    df["uin_csum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
    df["breath_id__u_in__max"] = df.groupby(["breath_id"])["u_in"].transform("max")
    df["breath_id__u_in__diffmax"] = df["breath_id__u_in__max"] - df["u_in"]
    df["breath_id__u_in__diffmean"] = (
        df.groupby(["breath_id"])["u_in"].transform("mean") - df["u_in"]
    )

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
        df[f"u_in_diff_{i}_forw"] = df["u_in"] - df[f"u_in_{i}_forw"]
    for i in range(1, U_OUT_N_FORWARD + 1):
        df[f"u_out_diff_{i}_forw"] = df["u_out"] - df[f"u_out_{i}_forw"]

    print("\t... Add categorical features ...")
    if use_rc:
        df["R_C"] = df["R"].astype(str) + "_" + df["C"].astype(str)
        df["R"] = df["R"] / 50
        df["C"] = df["C"] / 50
        df = pd.get_dummies(df)

    print("\t... Reset dtypes for lower memory usage ...")
    for c in df.columns:
        if c == "u_out":
            df[c] = df[c].astype("uint8")
        elif df[c].dtype == "float64":
            df[c] = df[c].astype("float32")

    gc.collect()
    gc.collect()
    return df


def load_model(model_path, n_features):
    """
    Make model loading robust across formats:
    - SavedModel directory -> use TFSMLayer wrapper for inference (Keras 3 safe).
    - .keras/.h5 -> load_model works.
    """
    tf.keras.backend.clear_session()
    gc.collect()
    gc.collect()

    if (
        tf.io.gfile.exists(model_path)
        and tf.io.gfile.isfile(model_path)
        and (model_path.endswith(".keras") or model_path.endswith(".h5"))
    ):
        with strategy.scope():
            return tf.keras.models.load_model(model_path, compile=False)

    if not tf.io.gfile.exists(model_path):
        raise OSError(f"Model path does not exist: {model_path}")

    endpoints = None
    try:
        endpoints = list(tf.saved_model.load(model_path).signatures.keys())
    except Exception:
        endpoints = None

    call_endpoint = "serving_default"
    if endpoints and call_endpoint not in endpoints:
        call_endpoint = endpoints[0]

    with strategy.scope():
        tfsml = tf.keras.layers.TFSMLayer(model_path, call_endpoint=call_endpoint)
        inp = tf.keras.Input(
            shape=(ROWS_PER_BREATH, n_features), dtype=tf.float32, name="x"
        )
        out = tfsml(inp)
        if isinstance(out, dict):
            out = out["output_0"] if "output_0" in out else out[next(iter(out.keys()))]
        model = tf.keras.Model(inputs=inp, outputs=out)
    return model


def _flatten_preds_to_rows(preds, n_rows):
    """
    Score-relevant robustness: different SavedModel/Keras exports can return:
    - (n_breaths, 80) or (n_breaths, 80, 1) or already flat.
    We convert to a flat per-row vector aligned with original row order.
    """
    p = np.asarray(preds)
    if p.ndim == 3 and p.shape[-1] == 1:
        p = p[..., 0]
    if p.ndim == 2 and p.shape[1] == ROWS_PER_BREATH:
        p = p.reshape(-1)
    else:
        p = p.reshape(-1)
    return p[:n_rows]




## === cell 7
print("\n... ADDING FEATURES TO TRAIN DATAFRAME ...\n")
train_df = add_features(train_df, use_rc=True)

print("\n... ADDING FEATURES TO TEST DATAFRAME ...\n")
test_df = add_features(test_df, use_rc=True)

train_cols = set(train_df.columns)
test_cols = set(test_df.columns)
missing_in_test = sorted(list(train_cols - test_cols))
missing_in_train = sorted(list(test_cols - train_cols))

for c in missing_in_test:
    test_df[c] = 0
for c in missing_in_train:
    train_df[c] = 0

test_df = test_df[train_df.columns]

LABEL_NAMES = ["pressure"]
GROUPBY_NAMES = ["breath_id"]
IGNORE_NAMES = ["id"]
FEATURE_NAMES = [
    x for x in train_df.columns if x not in LABEL_NAMES + GROUPBY_NAMES + IGNORE_NAMES
]
N_FEATURES = len(FEATURE_NAMES)

RS = RobustScaler()
RS.fit(train_df[FEATURE_NAMES].to_numpy())

display(train_df.head(3))
display(test_df.head(3))



## === cell 8
N_TEST = len(test_df)
mod = N_TEST % (OVERALL_BATCH_SIZE * ROWS_PER_BREATH)
TEST_PADDING = (OVERALL_BATCH_SIZE * ROWS_PER_BREATH - mod) if mod != 0 else 0

if TEST_PADDING > 0:
    test_df_pad = pd.concat(
        [test_df, test_df.iloc[:TEST_PADDING].reset_index(drop=True)],
        axis=0,
        ignore_index=True,
    )
else:
    test_df_pad = test_df

sub_test_x_np = RS.transform(test_df_pad[FEATURE_NAMES].to_numpy()).astype(np.float32)
test_ds = tf.data.Dataset.from_tensor_slices(sub_test_x_np)
test_ds = (
    test_ds.batch(ROWS_PER_BREATH, drop_remainder=True)
    .cache()
    .batch(OVERALL_BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)



## === cell 9
fold_pred_map = {f"fold_{i}": dict(val=None, test=None) for i in range(N_FOLDS)}

found_any_model = False
for i in tqdm(range(N_FOLDS), total=N_FOLDS):
    print(f"\n\n\n... STARTING FOLD {i} ...\n")

    model_path = os.path.join(MODEL_DIR, f"best_model_fold_{i}")
    candidate_paths = [
        model_path,
        model_path + "/",
        model_path + ".keras",
        model_path + ".h5",
    ]

    existing_path = next((p for p in candidate_paths if tf.io.gfile.exists(p)), None)
    if existing_path is None:
        print(f"\t--> MODEL FOR FOLD {i} NOT FOUND at {model_path} (skipping fold).")
        continue

    found_any_model = True
    print(f"\t--> LOADING MODEL FROM FOLD {i}: {existing_path}")
    model = load_model(existing_path, n_features=N_FEATURES)

    if f"fold_{i}" in fold_val_df_map:
        _tmp_val_df = fold_val_df_map[f"fold_{i}"].copy()
        _tmp_val_df = add_features(_tmp_val_df, use_rc=True)

        val_cols = set(_tmp_val_df.columns)
        for c in set(train_df.columns) - val_cols:
            _tmp_val_df[c] = 0
        extra = sorted(list(val_cols - set(train_df.columns)))
        if len(extra) > 0:
            _tmp_val_df = _tmp_val_df.drop(columns=extra)
        _tmp_val_df = _tmp_val_df[train_df.columns]

        _tmp_n_val = len(_tmp_val_df)
        modv = _tmp_n_val % (OVERALL_BATCH_SIZE * ROWS_PER_BREATH)
        _val_test_padding = (
            (OVERALL_BATCH_SIZE * ROWS_PER_BREATH - modv) if modv != 0 else 0
        )

        if _val_test_padding > 0:
            _tmp_val_df_pad = pd.concat(
                [
                    _tmp_val_df,
                    _tmp_val_df.iloc[:_val_test_padding].reset_index(drop=True),
                ],
                axis=0,
                ignore_index=True,
            )
        else:
            _tmp_val_df_pad = _tmp_val_df

        _sub_val_x_np = RS.transform(_tmp_val_df_pad[FEATURE_NAMES].to_numpy()).astype(
            np.float32
        )
        _tmp_val_ds = tf.data.Dataset.from_tensor_slices(_sub_val_x_np)
        _tmp_val_ds = (
            _tmp_val_ds.batch(ROWS_PER_BREATH, drop_remainder=True)
            .cache()
            .batch(OVERALL_BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        print(f"\t--> FOLD {i} MODEL INFER ON FOLD VAL DATASET AND SAVE PREDS ...")
        _val_preds = model.predict(_tmp_val_ds, verbose=0)
        fold_pred_map[f"fold_{i}"]["val"] = _flatten_preds_to_rows(
            _val_preds, _tmp_n_val
        )
        fold_val_df_map[f"fold_{i}"][f"fold_{i}_pressure"] = fold_pred_map[f"fold_{i}"][
            "val"
        ]

        del _tmp_val_df, _tmp_val_df_pad, _sub_val_x_np, _tmp_val_ds, _val_preds

    print(f"\t--> FOLD {i} MODEL INFER ON TEST DATASET AND SAVE PREDS ...")
    _test_preds = model.predict(test_ds, verbose=0)
    fold_pred_map[f"fold_{i}"]["test"] = _flatten_preds_to_rows(_test_preds, N_TEST)

    print(f"\t--> FOLD {i} MODEL PREDS BEING ADDED TO SAMPLE SUBMISSION DATAFRAME ...")
    ss_df[f"fold_{i}_pressure"] = fold_pred_map[f"fold_{i}"]["test"]

    tf.keras.backend.clear_session()
    del model, _test_preds
    gc.collect()
    gc.collect()

if not found_any_model:
    print(
        "\nWARNING: No fold models were found; using a baseline constant pressure prediction.\n"
    )
    baseline_pressure = float(train_df["pressure"].median())
    ss_df = test_df[["id"]].copy()
    ss_df["pressure"] = baseline_pressure



## === cell 10
if (
    "id" not in ss_df.columns
    or len(ss_df) != len(test_df)
    or not np.array_equal(ss_df["id"].to_numpy(), test_df["id"].to_numpy())
):
    ss_df = ss_df.copy()
    ss_df["id"] = test_df["id"].to_numpy()

fold_cols = [x for x in ss_df.columns if "fold_" in x and x.endswith("_pressure")]
if len(fold_cols) > 0:
    ss_df["pressure"] = ss_df[fold_cols].median(axis=1)

possible_pressures = np.asarray(POSSIBLE_PRESSURES, dtype=np.float32)
pred = ss_df["pressure"].to_numpy(dtype=np.float32)
idx = np.searchsorted(possible_pressures, pred, side="left")
idx = np.clip(idx, 0, len(possible_pressures) - 1)
idx0 = np.clip(idx - 1, 0, len(possible_pressures) - 1)
left = possible_pressures[idx0]
right = possible_pressures[idx]
choose_right = np.abs(right - pred) < np.abs(left - pred)
snapped = np.where(choose_right, right, left).astype(np.float32)
ss_df["pressure"] = snapped

out_path = "submission.csv"
ss_df[["id", "pressure"]].to_csv(out_path, index=False)

print(f"\nWrote submission to: {out_path}")
display(ss_df.head(10))



## === cell 11
if len(fold_val_df_map) == 0:
    print("\nNo fold validation data available; skipping fold MAE reporting.\n")
else:
    for i in range(N_FOLDS):
        k = f"fold_{i}"
        if k not in fold_val_df_map:
            continue
        tmp_df = fold_val_df_map[k].copy()
        if "pressure" in tmp_df.columns and f"{k}_pressure" in tmp_df.columns:
            tmp_df["int_step"] = (tmp_df.id - 1) % 80
            tmp_df["mae"] = (tmp_df[f"{k}_pressure"] - tmp_df["pressure"]).abs()
            fold_val_df_map[k] = tmp_df
            print(f"\nFOLD {i} MAE: {tmp_df.mae.mean()}\n")
            display(tmp_df.head())
