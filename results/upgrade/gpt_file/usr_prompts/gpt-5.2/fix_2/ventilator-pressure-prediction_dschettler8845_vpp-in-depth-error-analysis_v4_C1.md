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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
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

import tensorflow as tf

print(f"\t\t– TENSORFLOW VERSION: {tf.__version__}")

try:
    import tensorflow_addons as tfa

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

from kaggle_datasets import KaggleDatasets
import random
import matplotlib

print(f"\t\t– MATPLOTLIB VERSION: {matplotlib.__version__}")


def seed_it_all(seed=7):
    """Attempt to be reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


print("\n\n... IMPORTS COMPLETE ...\n")

print("\n... SEEDING FOR DETERMINISTIC BEHAVIOUR ...\n")
seed_it_all()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

if TPU:
    DATA_DIR = KaggleDatasets().get_gcs_path("ventilator-pressure-prediction")
    save_locally = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
else:
    DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"
    save_locally = None

print(f"\n... DATA DIRECTORY PATH IS:\n\t--> {DATA_DIR}")

print(f"\n... IMMEDIATE CONTENTS OF DATA DIRECTORY IS:")
for file in tf.io.gfile.glob(os.path.join(DATA_DIR, "*")):
    print(f"\t--> {file}")

print("\n\n... DATA ACCESS SETUP COMPLETED ...\n")



## === cell 4
MODEL_DIR = "/kaggle/input/vpp-synthetic-adventure-lstm-w-sofia-features"
N_FOLDS = 6
ROWS_PER_BREATH = 80

print("\n... BASIC DATA SETUP STARTING ...\n\n")

print("\n... TRAIN DATAFRAME ...\n")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
train_df = pd.read_csv(TRAIN_CSV)
display(train_df.head())

print("\n... VAL DATAFRAMES ...\n")
fold_val_df_map = {
    f"fold_{i}": pd.read_csv(os.path.join(MODEL_DIR, f"fold_{i}_val.csv"))
    for i in range(N_FOLDS)
}
display(fold_val_df_map["fold_0"].head())

print("\n... TEST DATAFRAME ..\n")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
test_df = pd.read_csv(TEST_CSV)
display(test_df.head())

N_TRAIN_BREATHS = len(train_df.groupby("breath_id").count())
N_TEST_BREATHS = len(test_df.breath_id.value_counts())

print("\n... SAMPLE SUBMISSION DATAFRAME ..\n")
SS_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
ss_df = pd.read_csv(SS_CSV)
display(ss_df.head())

print("\n... SETTING OTHER VARIABLES ..\n")
POSSIBLE_PRESSURES = sorted(np.array(train_df.pressure.value_counts().keys()))
APPROX_PRESSURE_DELTA_STEP = 0.0703021454512

REPLICA_BATCH_SIZE = 64
OVERALL_BATCH_SIZE = N_REPLICAS * REPLICA_BATCH_SIZE

print("\n\n... BASIC DATA SETUP FINISHING ...\n")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3868550128.py in <cell line: 0>()
     12 
     13 print("\n... VAL DATAFRAMES ...\n")
---> 14 fold_val_df_map = {
     15     f"fold_{i}": pd.read_csv(os.path.join(MODEL_DIR, f"fold_{i}_val.csv"))
     16     for i in range(N_FOLDS)

/tmp/ipykernel_55/3868550128.py in <dictcomp>(.0)
     13 print("\n... VAL DATAFRAMES ...\n")
     14 fold_val_df_map = {
---> 15     f"fold_{i}": pd.read_csv(os.path.join(MODEL_DIR, f"fold_{i}_val.csv"))
     16     for i in range(N_FOLDS)
     17 }

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/vpp-synthetic-adventure-lstm-w-sofia-features/fold_0_val.csv'

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
    p_0=1,
    r_0=1,
    use_rc=True,
):
    df = df.copy()

    df["measured_volume"] = (V_0 + df["u_in"] * df["time_step"].diff()).fillna(0) + V_0

    r_t = (3 * df["measured_volume"] / 4 / np.pi) ** (1 / 3)
    df["measured_pressure"] = (
        p_0 + (1 - (r_t / r_0) ** 6) * 1 / (r_t * r_0**2)
    ).fillna(0) + p_0

    print("\n... Add general features ...\n")
    df["uin_auc"] = df["time_step"] * df["u_in"]
    df["uin_auc"] = df.groupby("breath_id")["uin_auc"].cumsum()
    df["uin_csum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
    df["breath_id__u_in__max"] = df.groupby(["breath_id"])["u_in"].transform("max")
    df["breath_id__u_in__diffmax"] = (
        df.groupby(["breath_id"])["u_in"].transform("max") - df["u_in"]
    )
    df["breath_id__u_in__diffmean"] = (
        df.groupby(["breath_id"])["u_in"].transform("mean") - df["u_in"]
    )
    df["breath_id__u_in__diffmax"] = (
        df.groupby(["breath_id"])["u_in"].transform("max") - df["u_in"]
    )
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
        df[f"u_in_diff_{i}_back"] = df["u_in"] - df[f"u_in_{i}_forw"]
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


def load_model(model_path):
    tf.keras.backend.clear_session()
    gc.collect()
    gc.collect()

    with strategy.scope():
        if TPU:
            load_locally = tf.saved_model.LoadOptions(
                experimental_io_device="/job:localhost"
            )
            _model = tf.keras.models.load_model(model_path, options=load_locally)
        else:
            _model = tf.keras.models.load_model(model_path)
        _model.compile("adam", loss="mae", weighted_metrics=["mae"])
    return _model




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



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1477816476.py in <cell line: 0>()
      3 
      4 print("\n... ADDING FEATURES TO TEST DATAFRAME ...\n")
----> 5 test_df = add_features(test_df, use_rc=True)
      6 
      7 # BUGFIX: ensure one-hot encoded columns match between train and test (and val later),

NameError: name 'test_df' is not defined

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

sub_test_x_np = RS.transform(test_df_pad[FEATURE_NAMES].to_numpy())
test_ds = tf.data.Dataset.from_tensor_slices(sub_test_x_np)
test_ds = (
    test_ds.batch(ROWS_PER_BREATH, drop_remainder=True)
    .cache()
    .batch(OVERALL_BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/834109038.py in <cell line: 0>()
      1 # Get testing data (keep original padding logic, but handle exact divisibility)
----> 2 N_TEST = len(test_df)
      3 mod = N_TEST % (OVERALL_BATCH_SIZE * ROWS_PER_BREATH)
      4 TEST_PADDING = (OVERALL_BATCH_SIZE * ROWS_PER_BREATH - mod) if mod != 0 else 0
      5 

NameError: name 'test_df' is not defined

## === cell 9
fold_pred_map = {f"fold_{i}": dict(val=None, test=None) for i in range(N_FOLDS)}

for i in tqdm(range(N_FOLDS), total=N_FOLDS):
    print(f"\n\n\n... STARTING FOLD {i} ...\n")

    print(f"\t--> LOADING MODEL FROM FOLD {i} ...")
    model = load_model(os.path.join(MODEL_DIR, f"best_model_fold_{i}"))

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
            [_tmp_val_df, _tmp_val_df.iloc[:_val_test_padding].reset_index(drop=True)],
            axis=0,
            ignore_index=True,
        )
    else:
        _tmp_val_df_pad = _tmp_val_df

    _sub_val_x_np = RS.transform(_tmp_val_df_pad[FEATURE_NAMES].to_numpy())
    _tmp_val_ds = tf.data.Dataset.from_tensor_slices(_sub_val_x_np)
    _tmp_val_ds = (
        _tmp_val_ds.batch(ROWS_PER_BREATH, drop_remainder=True)
        .cache()
        .batch(OVERALL_BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"\t--> FOLD {i} MODEL INFER ON FOLD VAL DATASET AND SAVE PREDS ...")
    fold_pred_map[f"fold_{i}"]["val"] = model.predict(_tmp_val_ds, verbose=0).reshape(
        -1
    )[:_tmp_n_val]

    print(f"\t--> FOLD {i} MODEL INFER ON TEST DATASET AND SAVE PREDS ...")
    fold_pred_map[f"fold_{i}"]["test"] = model.predict(test_ds, verbose=0).reshape(-1)[
        :N_TEST
    ]

    print(f"\t--> FOLD {i} MODEL PREDS BEING ADDED TO FOLD VAL DATAFRAMES ...")
    fold_val_df_map[f"fold_{i}"][f"fold_{i}_pressure"] = fold_pred_map[f"fold_{i}"][
        "val"
    ]

    print(f"\t--> FOLD {i} MODEL PREDS BEING ADDED TO SAMPLE SUBMISSION DATAFRAME ...")
    ss_df[f"fold_{i}_pressure"] = fold_pred_map[f"fold_{i}"]["test"]

    tf.keras.backend.clear_session()
    del model, _tmp_val_df, _tmp_val_df_pad, _sub_val_x_np, _tmp_val_ds
    gc.collect()
    gc.collect()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/710373412.py in <cell line: 0>()
      5 
      6     print(f"\t--> LOADING MODEL FROM FOLD {i} ...")
----> 7     model = load_model(os.path.join(MODEL_DIR, f"best_model_fold_{i}"))
      8 
      9     _tmp_val_df = fold_val_df_map[f"fold_{i}"].copy()

/tmp/ipykernel_55/2364769819.py in load_model(model_path)
    102             _model = tf.keras.models.load_model(model_path, options=load_locally)
    103         else:
--> 104             _model = tf.keras.models.load_model(model_path)
    105         _model.compile("adam", loss="mae", weighted_metrics=["mae"])
    106     return _model

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    204         )
    205     else:
--> 206         raise ValueError(
    207             f"File format not supported: filepath={filepath}. "
    208             "Keras 3 only supports V3 `.keras` files and "

ValueError: File format not supported: filepath=/kaggle/input/vpp-synthetic-adventure-lstm-w-sofia-features/best_model_fold_0. Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension). Note that the legacy SavedModel format is not supported by `load_model()` in Keras 3. In order to reload a TensorFlow SavedModel as an inference-only layer in Keras 3, use `keras.layers.TFSMLayer(/kaggle/input/vpp-synthetic-adventure-lstm-w-sofia-features/best_model_fold_0, call_endpoint='serving_default')` (note that your `call_endpoint` might have a different name).

## === cell 10
ss_df["pressure"] = ss_df[[x for x in ss_df.columns if "fold" in x]].median(axis=1)
ss_df[["id", "pressure"]].to_csv("submission.csv", index=False)

print("\nWrote submission to: submission.csv")
display(ss_df.head(10))



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2706301260.py in <cell line: 0>()
      1 # Keep original ensembling logic (median over folds)
----> 2 ss_df["pressure"] = ss_df[[x for x in ss_df.columns if "fold" in x]].median(axis=1)
      3 ss_df[["id", "pressure"]].to_csv("submission.csv", index=False)
      4 
      5 print("\nWrote submission to: submission.csv")

NameError: name 'ss_df' is not defined

## === cell 11
for i in range(N_FOLDS):
    tmp_df = fold_val_df_map[f"fold_{i}"].copy()
    if "pressure" in tmp_df.columns and f"fold_{i}_pressure" in tmp_df.columns:
        tmp_df["int_step"] = (tmp_df.id - 1) % 80
        tmp_df["mae"] = (tmp_df[f"fold_{i}_pressure"] - tmp_df["pressure"]).abs()
        fold_val_df_map[f"fold_{i}"] = tmp_df
        print(f"\nFOLD {i} MAE: {fold_val_df_map[f'fold_{i}'].mae.mean()}\n")
        display(tmp_df.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2648135053.py in <cell line: 0>()
      1 # Optional fold MAE reporting (runs only if fold val dfs contain true pressure)
      2 for i in range(N_FOLDS):
----> 3     tmp_df = fold_val_df_map[f"fold_{i}"].copy()
      4     if "pressure" in tmp_df.columns and f"fold_{i}_pressure" in tmp_df.columns:
      5         tmp_df["int_step"] = (tmp_df.id - 1) % 80

NameError: name 'fold_val_df_map' is not defined
