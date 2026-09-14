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

0.6658

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
print("\n... IMPORTS STARTING ...\n")

!pip install weightedstats
import weightedstats as ws

print("\n\tVERSION INFORMATION")
import tensorflow as tf; print(f"\t\t– TENSORFLOW VERSION: {tf.__version__}");
import tensorflow_addons as tfa; print(f"\t\t– TENSORFLOW ADDONS VERSION: {tfa.__version__}");
import pandas as pd; pd.options.mode.chained_assignment = None;
import numpy as np; print(f"\t\t– NUMPY VERSION: {np.__version__}");
import sklearn; print(f"\t\t– SKLEARN VERSION: {sklearn.__version__}");
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.model_selection import GroupKFold;

from kaggle_datasets import KaggleDatasets
from collections import Counter
from datetime import datetime
from glob import glob
import warnings
import requests
import imageio
import IPython
import sklearn
import urllib
import zipfile
import pickle
import random
import shutil
import string
import math
import time
import gzip
import ast
import sys
import io
import os
import gc
import re

from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm; tqdm.pandas();
import plotly.express as px
import seaborn as sns
from PIL import Image
import matplotlib; print(f"\t\t– MATPLOTLIB VERSION: {matplotlib.__version__}");
import plotly
import PIL
import cv2


def seed_it_all(seed=7):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    
print("\n\n... IMPORTS COMPLETE ...\n")
    
print("\n... SEEDING FOR DETERMINISTIC BEHAVIOUR ...\n")
seed_it_all()


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    if tf.config.experimental.list_physical_devices('GPU'):
        print(f"\n ... RUNNING ON GPU ...\n")
        import cudf, cuml, cupy
        from numba import cuda
        from cuml.neighbors import NearestNeighbors
    else:
        print(f"\n ... RUNNING ON CPU ...\n")
    

N_REPLICAS = strategy.num_replicas_in_sync
    
print(f"... # OF REPLICAS: {N_REPLICAS} ...\n")

print(f"\n... ACCELERATOR SETUP COMPLTED ...\n")


## === cell 2
print("\n... DATA ACCESS SETUP STARTED ...\n")

if TPU:
    DATA_DIR = KaggleDatasets().get_gcs_path('ventilator-pressure-prediction')
else:
    DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"
    
print(f"\n... DATA DIRECTORY PATH IS:\n\t--> {DATA_DIR}")

print(f"\n... IMMEDIATE CONTENTS OF DATA DIRECTORY IS:")
for file in tf.io.gfile.glob(os.path.join(DATA_DIR, "*")): print(f"\t--> {file}")

    
print("\n\n... DATA ACCESS SETUP COMPLETED ...\n")


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2624807020.py in <cell line: 0>()
     11 
     12 print(f"\n... IMMEDIATE CONTENTS OF DATA DIRECTORY IS:")
---> 13 for file in tf.io.gfile.glob(os.path.join(DATA_DIR, "*")): print(f"\t--> {file}")
     14 
     15 

NameError: name 'os' is not defined

## === cell 3
print("\n... BASIC DATA SETUP STARTING ...\n\n")

DO_CLUSTERING=True

print("\n... TRAIN DATAFRAME ..\n")
if DO_CLUSTERING:
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    train_df = cudf.read_csv(TRAIN_CSV)
else:
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    train_df = pd.read_csv(TRAIN_CSV)
display(train_df)

print("\n... TEST DATAFRAME ..\n")
if DO_CLUSTERING:
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")
    test_df = cudf.read_csv(TEST_CSV)
else:
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")
    test_df = pd.read_csv(TEST_CSV)
display(test_df)

print("\n... SAMPLE SUBMISSION DATAFRAME ..\n")
SS_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
ss_df = pd.read_csv(SS_CSV)
display(ss_df)


N_TRAIN_BREATHS = len(train_df.groupby("breath_id").count())
N_TEST_BREATHS = len(test_df.breath_id.value_counts())
ROWS_PER_BREATH = 80

test_df["breath_step"] = np.arange(ROWS_PER_BREATH).tolist()*(len(test_df)//ROWS_PER_BREATH)
train_df["breath_step"] = np.arange(ROWS_PER_BREATH).tolist()*(len(train_df)//ROWS_PER_BREATH)

POSSIBLE_PRESSURES = train_df.pressure.unique().sort_values().values
PRESSURE_DELTA_STEP = float((POSSIBLE_PRESSURES[1:]-POSSIBLE_PRESSURES[:-1]).mean())

print("\n\n... BASIC DATA SETUP FINISHING ...\n")


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4065402127.py in <cell line: 0>()
      6 print("\n... TRAIN DATAFRAME ..\n")
      7 if DO_CLUSTERING:
----> 8     TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
      9     train_df = cudf.read_csv(TRAIN_CSV)
     10 else:

NameError: name 'os' is not defined

## === cell 4
print(f"\n... XLA OPTIMIZATIONS STARTING ...\n")

print(f"\n... CONFIGURE JIT (JUST IN TIME) COMPILATION ...\n")
tf.config.optimizer.set_jit(True)

print(f"\n... XLA OPTIMIZATIONS COMPLETED ...\n")


## === cell 5
def flatten_l_o_l(nested_list):
    """ Flatten a list of lists """
    return [item for sublist in nested_list for item in sublist]


def compute_weighted_median(_values, _weights, redux_factor=1):
    """ Compute a weighted median using knn distances
    
    Args:
        values (): TBD
        weights (): TBD
    
    Returns:
        The weighted median
    """
    return np.median(flatten_l_o_l([[_v,]*int(_w//redux_factor) for _v,_w in zip(_values,_weights)]))


def add_features(df, 
                 U_IN_N_FORWARD=3,
                 U_IN_N_BACKWARD=3,
                 U_OUT_N_FORWARD=1, 
                 U_OUT_N_BACKWARD=1, 
                 use_rc=True):
    """ TBD """
    
    print("\n... Add general features ...\n")
    df['uin_auc'] = df['time_step'] * df['u_in']
    df['uin_auc'] = df.groupby('breath_id')['uin_auc'].cumsum()
    df['uin_csum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['cross3']= df['time_step']*df['u_in']
    df['cross3_sqd_1']= df['time_step']*df['u_in']**2
    df['cross3_sqd_2']= df['time_step']**2*df['u_in']
    df['cross3_cubed_1']= df['time_step']*df['u_in']**3
    df['cross3_cubed_2']= df['time_step']**3*df['u_in']

    
    print("\t... Add lag and advance UIN features ...")
    for i in range(1, U_IN_N_BACKWARD+1):
        df[f'u_in_{i}_back'] = df.groupby("breath_id")['u_in'].shift(i).fillna(0)
    for i in range(1, U_IN_N_FORWARD+1):
        df[f'u_in_{i}_forw'] = df.groupby("breath_id")['u_in'].shift(-i).fillna(0)
    
    print("\t... Add lag and advance UOUT features ...")
    for i in range(1, U_OUT_N_BACKWARD+1):
        df[f'u_out_{i}_back'] = df.groupby("breath_id")['u_out'].shift(i).fillna(0)
    for i in range(1, U_OUT_N_FORWARD+1):
        df[f'u_out_{i}_forw'] = df.groupby("breath_id")['u_out'].shift(-i).fillna(0)
    
    print("\t... Add UIN and UOUT `diff` features ...")
    for i in range(1, U_IN_N_BACKWARD+1):
        df[f'u_in_diff_{i}_back'] = df['u_in'] - df[f'u_in_{i}_back']
    for i in range(1, U_OUT_N_BACKWARD+1):
        df[f'u_out_diff_{i}_back'] = df['u_out'] - df[f'u_out_{i}_back']
    for i in range(1, U_IN_N_FORWARD+1):
        df[f'u_in_diff_{i}_back'] = df['u_in'] - df[f'u_in_{i}_forw']
    for i in range(1, U_OUT_N_FORWARD+1):
        df[f'u_out_diff_{i}_forw'] = df['u_out'] - df[f'u_out_{i}_forw']
    
    print("\t... Add categorical features ...")
    if use_rc:
        df['_R'] = df['R'].astype(str)
        df['_C'] = df['C'].astype(str)
        df['R'] = df['R']/50
        df['C'] = df['C']/50
        df = pd.get_dummies(df,)
    
    print("\t... Reset dtypes for lower memory usage ...")
    for c in df.columns:
        if c in ["u_out", "breath_step"]:
            df[c] = df[c].astype("uint8")
        elif df[c].dtype=="float64":
            df[c] = df[c].astype("float32")
    
    gc.collect(); gc.collect();
    
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


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3954684499.py in <cell line: 0>()
      1 print("\n... ADDING FEATURES TO TRAIN DATAFRAME ...\n")
----> 2 train_df = add_features(train_df.to_pandas(), use_rc=True)
      3 train_df = cudf.from_pandas(train_df)
      4 
      5 print("\n... ADDING FEATURES TO TEST DATAFRAME ...\n")

NameError: name 'train_df' is not defined

## === cell 7
test_df = test_df.to_pandas()
test_df["breath_duration"] = test_df.groupby('breath_id')[['time_step']].transform('max')
test_df["exhale_steps"] = test_df.groupby('breath_id')[['u_out']].transform('sum')
test_df["inhale_steps"] = 80-test_df.groupby('breath_id')[['u_out']].transform('sum')
test_df = cudf.from_pandas(test_df)

train_df = train_df.to_pandas()
train_df["breath_duration"] = train_df.groupby('breath_id')[['time_step']].transform('max')
train_df["exhale_steps"] = train_df.groupby('breath_id')[['u_out']].transform('sum')
train_df["inhale_steps"] = 80-train_df.groupby('breath_id')[['u_out']].transform('sum')
train_df = cudf.from_pandas(train_df)

display(train_df)
display(test_df)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/6777106.py in <cell line: 0>()
----> 1 test_df = test_df.to_pandas()
      2 test_df["breath_duration"] = test_df.groupby('breath_id')[['time_step']].transform('max')
      3 test_df["exhale_steps"] = test_df.groupby('breath_id')[['u_out']].transform('sum')
      4 test_df["inhale_steps"] = 80-test_df.groupby('breath_id')[['u_out']].transform('sum')
      5 test_df = cudf.from_pandas(test_df)

NameError: name 'test_df' is not defined

## === cell 8
RS = cuml.preprocessing.RobustScaler()
FEATURE_COLS = [_c for _c in train_df.columns if _c not in ['id', 'breath_id', 'pressure']]
train_df[FEATURE_COLS] = RS.fit_transform(train_df[FEATURE_COLS])
test_df[FEATURE_COLS] = RS.transform(test_df[FEATURE_COLS])

display(train_df)
display(test_df)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2722140971.py in <cell line: 0>()
      1 RS = cuml.preprocessing.RobustScaler()
----> 2 FEATURE_COLS = [_c for _c in train_df.columns if _c not in ['id', 'breath_id', 'pressure']]
      3 train_df[FEATURE_COLS] = RS.fit_transform(train_df[FEATURE_COLS])
      4 test_df[FEATURE_COLS] = RS.transform(test_df[FEATURE_COLS])
      5 

NameError: name 'train_df' is not defined

## === cell 9
def compress_df(df):
    cdf = df.groupby('breath_id').collect().reset_index()
    
    flatten_cols = list(set([_c for _c in cdf.columns if "_back" in _c]+\
                   [_c for _c in cdf.columns if "_forw" in _c]+\
                   [_c for _c in cdf.columns if "cross" in _c]+\
                   [_c for _c in cdf.columns if "u_in" in _c]+\
                   [_c for _c in cdf.columns if "u_out" in _c]+\
                   [_c for _c in cdf.columns if "uin" in _c]+\
                   [_c for _c in cdf.columns if "uout" in _c]+\
                   ["breath_step", "time_step"]))
    
    for j, _c in enumerate(flatten_cols):
        for i in range(ROWS_PER_BREATH): cdf[f'{chr(97+j)}_{i}'] = cdf[_c].list.get(i)
    
    if "pressure" in cdf.columns:
        for i in range(ROWS_PER_BREATH): cdf[f'z_{i}'] = cdf["pressure"].list.get(i)
        flatten_cols.append("pressure")
    cdf.drop(columns=flatten_cols+["id",], inplace=True)
    
    REPEAT_COLS = list(set([_c for _c in cdf.columns if "R" in _c]+\
                   [_c for _c in cdf.columns if "C" in _c]+\
                   ["breath_duration", "exhale_steps", "inshale_steps"]))
    REPEAT_COLS = ["R", "C", "_R_20", "_R_5", "_R_50", "_C_10", "_C_20", "_C_50", "breath_duration", "exhale_steps", "inhale_steps"]
    for _rc in REPEAT_COLS:
        cdf[_rc] = cdf[_rc].list.get(0)
    return cdf

train_df = compress_df(train_df)
test_df = compress_df(test_df)

display(train_df)
display(test_df)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3133895730.py in <cell line: 0>()
     28     return cdf
     29 
---> 30 train_df = compress_df(train_df)
     31 test_df = compress_df(test_df)
     32 

NameError: name 'train_df' is not defined

## === cell 10
PRESSURE_COLS = [f"z_{i}" for i in range(80)]
USE_COLS = [_c for _c in train_df.columns if _c not in ["breath_id", "id"]+[f"z_{i}" for i in range(80)]]
BLEND_NEIGHBORS = 100
RESTRICT_RC=True


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2202993967.py in <cell line: 0>()
      1 PRESSURE_COLS = [f"z_{i}" for i in range(80)]
----> 2 USE_COLS = [_c for _c in train_df.columns if _c not in ["breath_id", "id"]+[f"z_{i}" for i in range(80)]]
      3 BLEND_NEIGHBORS = 100
      4 RESTRICT_RC=True

NameError: name 'train_df' is not defined

## === cell 11
BREATH_ID=1
gt_breath = train_df.loc[train_df.breath_id==BREATH_ID].to_pandas()
BREATH_R = gt_breath.R.values[0]
BREATH_C = gt_breath.C.values[0]

df_to_use = train_df.copy()[(train_df.R==BREATH_R)&(train_df.C==BREATH_C)].reset_index(drop=True) if RESTRICT_RC else train_df.copy()

model = NearestNeighbors(n_neighbors=BLEND_NEIGHBORS, metric="l1",)
model.fit(df_to_use[USE_COLS])

distances, indices = model.kneighbors(gt_breath[USE_COLS])
distances, indices = distances[0, 1:], indices[0, 1:] #discard GT

gt_breath_pressure = gt_breath[PRESSURE_COLS].squeeze().values
nn_pressure_df = df_to_use.iloc[indices][PRESSURE_COLS]
nn_pressure_df.insert(loc=0, name="nn_distance", value=distances)
nn_pressures = nn_pressure_df[PRESSURE_COLS].values

for UP_TO in [BLEND_NEIGHBORS-1, 75, 50, 25, 12, 5, 2, 1]:
    print("\n\n\n")
    plt.figure(figsize=(20, 8))
    for i in range(UP_TO):
        if i==UP_TO-1:
            plt.plot(nn_pressures[i].get(), color="lightblue", linewidth=1, label="NEIGHBORS")
        else:
            plt.plot(nn_pressures[i].get(), color="lightblue")
    plt.plot(gt_breath_pressure, color="blue", linewidth=3, label="GROUND TRUTH")
    
    mean_pressure = nn_pressures[:UP_TO].mean(axis=0).get()
    mean_mae = np.abs(gt_breath_pressure-mean_pressure).mean()
    plt.plot(mean_pressure, color="deepskyblue", linewidth=3, label="NEIGHBOR MEAN")
    
    median_pressure = np.median(nn_pressures[:UP_TO].get(),axis=0)
    median_mae = np.abs(gt_breath_pressure-median_pressure).mean()
    plt.plot(median_pressure, color="mediumspringgreen", linewidth=3, label="NEIGHBOR MEDIAN")
    
    plt.title(f"{UP_TO} NEAREST NEIGHBORS VS GROUND TRUTH\n\nMEAN MAE = {mean_mae:.5f}\nMEDIAN MAE = {median_mae:.5f}", fontweight="bold")
    plt.xlabel("Step In Breath", fontweight="bold")
    plt.ylabel("Pressure (mm/H2O)", fontweight="bold")

    plt.legend(loc="upper right", prop={"size":14})
    plt.grid(which="both")
    
    plt.tight_layout()
    plt.show()
    
neighbor_means = []
neighbor_medians = []
for UP_TO in tqdm(range(1, BLEND_NEIGHBORS), total=BLEND_NEIGHBORS-1):
    mean_pressure = nn_pressures[:UP_TO].mean(axis=0).get()
    neighbor_means.append(np.abs(gt_breath_pressure-mean_pressure).mean())
    
    median_pressure = np.median(nn_pressures[:UP_TO].get(),axis=0)
    neighbor_medians.append(np.abs(gt_breath_pressure-median_pressure).mean())
    
    
    
plt.figure(figsize=(20, 8))

MIN_MAE = min(min(neighbor_means), min(neighbor_medians))

plt.plot(neighbor_means, color="deepskyblue", linewidth=3, label="NEIGHBOR MEAN")
plt.fill_between(x=np.arange(BLEND_NEIGHBORS-1), y1=neighbor_means, y2=MIN_MAE, color="deepskyblue", alpha=0.25)

plt.plot(neighbor_medians, color="mediumspringgreen", linewidth=3, label="NEIGHBOR MEDIAN")
plt.fill_between(x=np.arange(BLEND_NEIGHBORS-1), y1=neighbor_medians, y2=MIN_MAE, color="mediumspringgreen", alpha=0.25)

IDEAL_NUMBER_OF_NEIGHBORS = np.argmin(neighbor_medians) if min(neighbor_means)>min(neighbor_medians) else np.argmin(neighbor_means)
BLEND_STYLE = "MEDIAN" if min(neighbor_means)>min(neighbor_medians) else "MEAN"
plt.axvline(x=IDEAL_NUMBER_OF_NEIGHBORS, color='blue', linestyle="dotted")

plt.xlabel("Number of Neighbors", fontweight="bold")
plt.ylabel("MAE When Compared to Ground Truth", fontweight="bold")
plt.title(f"NEAREST NEIGHBOR MEAN v. MEDIAN FOR DIFFERENT AMOUNTS OF NEIGHBORS", fontweight="bold")
plt.legend(loc="upper right", prop={"size":14})
plt.grid(which="both")

plt.tight_layout()
plt.show()

print(f"\n\n... THE IDEAL NUMBER OF NEIGHBORS IS {IDEAL_NUMBER_OF_NEIGHBORS}, USING `{BLEND_STYLE}` WITH AN MAE OF {MIN_MAE} ...\n\n")


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4204395470.py in <cell line: 0>()
      1 # Explore For Breath 1
      2 BREATH_ID=1
----> 3 gt_breath = train_df.loc[train_df.breath_id==BREATH_ID].to_pandas()
      4 BREATH_R = gt_breath.R.values[0]
      5 BREATH_C = gt_breath.C.values[0]

NameError: name 'train_df' is not defined

## === cell 12
BREATH_ID=28

gt_breath = train_df.loc[train_df.breath_id==BREATH_ID].to_pandas()
BREATH_R = gt_breath.R.values[0]
BREATH_C = gt_breath.C.values[0]

df_to_use = train_df.copy()[(train_df.R==BREATH_R)&(train_df.C==BREATH_C)].reset_index(drop=True) if RESTRICT_RC else train_df.copy()

model = NearestNeighbors(n_neighbors=BLEND_NEIGHBORS, metric="l1",)
model.fit(df_to_use[USE_COLS])

distances, indices = model.kneighbors(gt_breath[USE_COLS])
distances, indices = distances[0, 1:], indices[0, 1:] #discard GT

gt_breath_pressure = gt_breath[PRESSURE_COLS].squeeze().values
nn_pressure_df = df_to_use.iloc[indices][PRESSURE_COLS]
nn_pressure_df.insert(loc=0, name="nn_distance", value=distances)
nn_pressures = nn_pressure_df[PRESSURE_COLS].values

for UP_TO in [BLEND_NEIGHBORS-1, 75, 50, 25, 12, 5, 2, 1]:
    print("\n\n\n")
    plt.figure(figsize=(20, 8))
    for i in range(UP_TO):
        if i==UP_TO-1:
            plt.plot(nn_pressures[i].get(), color="lightblue", linewidth=1, label="NEIGHBORS")
        else:
            plt.plot(nn_pressures[i].get(), color="lightblue")
    plt.plot(gt_breath_pressure, color="blue", linewidth=3, label="GROUND TRUTH")
    
    mean_pressure = nn_pressures[:UP_TO].mean(axis=0).get()
    mean_mae = np.abs(gt_breath_pressure-mean_pressure).mean()
    plt.plot(mean_pressure, color="deepskyblue", linewidth=3, label="NEIGHBOR MEAN")
    
    median_pressure = np.median(nn_pressures[:UP_TO].get(),axis=0)
    median_mae = np.abs(gt_breath_pressure-median_pressure).mean()
    plt.plot(median_pressure, color="mediumspringgreen", linewidth=3, label="NEIGHBOR MEDIAN")
    
    plt.title(f"{UP_TO} NEAREST NEIGHBORS VS GROUND TRUTH\n\nMEAN MAE = {mean_mae:.5f}\nMEDIAN MAE = {median_mae:.5f}", fontweight="bold")
    plt.xlabel("Step In Breath", fontweight="bold")
    plt.ylabel("Pressure (mm/H2O)", fontweight="bold")

    plt.legend(loc="upper right", prop={"size":14})
    plt.grid(which="both")
    
    plt.tight_layout()
    plt.show()
    
neighbor_means = []
neighbor_medians = []
for UP_TO in tqdm(range(1, BLEND_NEIGHBORS), total=BLEND_NEIGHBORS-1):
    mean_pressure = nn_pressures[:UP_TO].mean(axis=0).get()
    neighbor_means.append(np.abs(gt_breath_pressure-mean_pressure).mean())
    
    median_pressure = np.median(nn_pressures[:UP_TO].get(),axis=0)
    neighbor_medians.append(np.abs(gt_breath_pressure-median_pressure).mean())
    
    
    
plt.figure(figsize=(20, 8))

MIN_MAE = min(min(neighbor_means), min(neighbor_medians))

plt.plot(neighbor_means, color="deepskyblue", linewidth=3, label="NEIGHBOR MEAN")
plt.fill_between(x=np.arange(BLEND_NEIGHBORS-1), y1=neighbor_means, y2=MIN_MAE, color="deepskyblue", alpha=0.25)

plt.plot(neighbor_medians, color="mediumspringgreen", linewidth=3, label="NEIGHBOR MEDIAN")
plt.fill_between(x=np.arange(BLEND_NEIGHBORS-1), y1=neighbor_medians, y2=MIN_MAE, color="mediumspringgreen", alpha=0.25)

IDEAL_NUMBER_OF_NEIGHBORS = np.argmin(neighbor_medians) if min(neighbor_means)>min(neighbor_medians) else np.argmin(neighbor_means)
BLEND_STYLE = "MEDIAN" if min(neighbor_means)>min(neighbor_medians) else "MEAN"
plt.axvline(x=IDEAL_NUMBER_OF_NEIGHBORS, color='blue', linestyle="dotted")

plt.xlabel("Number of Neighbors", fontweight="bold")
plt.ylabel("MAE When Compared to Ground Truth", fontweight="bold")
plt.title(f"NEAREST NEIGHBOR MEAN v. MEDIAN FOR DIFFERENT AMOUNTS OF NEIGHBORS", fontweight="bold")
plt.legend(loc="upper right", prop={"size":14})
plt.grid(which="both")

plt.tight_layout()
plt.show()

print(f"\n\n... THE IDEAL NUMBER OF NEIGHBORS IS {IDEAL_NUMBER_OF_NEIGHBORS}, USING `{BLEND_STYLE}` WITH AN MAE OF {MIN_MAE} ...\n\n")


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2816162916.py in <cell line: 0>()
      2 BREATH_ID=28
      3 
----> 4 gt_breath = train_df.loc[train_df.breath_id==BREATH_ID].to_pandas()
      5 BREATH_R = gt_breath.R.values[0]
      6 BREATH_C = gt_breath.C.values[0]

NameError: name 'train_df' is not defined

## === cell 13
BREATH_ID=87

gt_breath = train_df.loc[train_df.breath_id==BREATH_ID].to_pandas()
BREATH_R = gt_breath.R.values[0]
BREATH_C = gt_breath.C.values[0]

df_to_use = train_df.copy()[(train_df.R==BREATH_R)&(train_df.C==BREATH_C)].reset_index(drop=True) if RESTRICT_RC else train_df.copy()

model = NearestNeighbors(n_neighbors=BLEND_NEIGHBORS, metric="l1",)
model.fit(df_to_use[USE_COLS])

distances, indices = model.kneighbors(gt_breath[USE_COLS])
distances, indices = distances[0, 1:], indices[0, 1:] #discard GT

gt_breath_pressure = gt_breath[PRESSURE_COLS].squeeze().values
nn_pressure_df = df_to_use.iloc[indices][PRESSURE_COLS]
nn_pressure_df.insert(loc=0, name="nn_distance", value=distances)
nn_pressures = nn_pressure_df[PRESSURE_COLS].values

for UP_TO in [BLEND_NEIGHBORS-1, 75, 50, 25, 12, 5, 2, 1]:
    print("\n\n\n")
    plt.figure(figsize=(20, 8))
    for i in range(UP_TO):
        if i==UP_TO-1:
            plt.plot(nn_pressures[i].get(), color="lightblue", linewidth=1, label="NEIGHBORS")
        else:
            plt.plot(nn_pressures[i].get(), color="lightblue")
    plt.plot(gt_breath_pressure, color="blue", linewidth=3, label="GROUND TRUTH")
    
    mean_pressure = nn_pressures[:UP_TO].mean(axis=0).get()
    mean_mae = np.abs(gt_breath_pressure-mean_pressure).mean()
    plt.plot(mean_pressure, color="deepskyblue", linewidth=3, label="NEIGHBOR MEAN")
    
    median_pressure = np.median(nn_pressures[:UP_TO].get(),axis=0)
    median_mae = np.abs(gt_breath_pressure-median_pressure).mean()
    plt.plot(median_pressure, color="mediumspringgreen", linewidth=3, label="NEIGHBOR MEDIAN")
    
    plt.title(f"{UP_TO} NEAREST NEIGHBORS VS GROUND TRUTH\n\nMEAN MAE = {mean_mae:.5f}\nMEDIAN MAE = {median_mae:.5f}", fontweight="bold")
    plt.xlabel("Step In Breath", fontweight="bold")
    plt.ylabel("Pressure (mm/H2O)", fontweight="bold")

    plt.legend(loc="upper right", prop={"size":14})
    plt.grid(which="both")
    
    plt.tight_layout()
    plt.show()
    
neighbor_means = []
neighbor_medians = []
for UP_TO in tqdm(range(1, BLEND_NEIGHBORS), total=BLEND_NEIGHBORS-1):
    mean_pressure = nn_pressures[:UP_TO].mean(axis=0).get()
    neighbor_means.append(np.abs(gt_breath_pressure-mean_pressure).mean())
    
    median_pressure = np.median(nn_pressures[:UP_TO].get(),axis=0)
    neighbor_medians.append(np.abs(gt_breath_pressure-median_pressure).mean())
    
    
    
plt.figure(figsize=(20, 8))

MIN_MAE = min(min(neighbor_means), min(neighbor_medians))

plt.plot(neighbor_means, color="deepskyblue", linewidth=3, label="NEIGHBOR MEAN")
plt.fill_between(x=np.arange(BLEND_NEIGHBORS-1), y1=neighbor_means, y2=MIN_MAE, color="deepskyblue", alpha=0.25)

plt.plot(neighbor_medians, color="mediumspringgreen", linewidth=3, label="NEIGHBOR MEDIAN")
plt.fill_between(x=np.arange(BLEND_NEIGHBORS-1), y1=neighbor_medians, y2=MIN_MAE, color="mediumspringgreen", alpha=0.25)

IDEAL_NUMBER_OF_NEIGHBORS = np.argmin(neighbor_medians) if min(neighbor_means)>min(neighbor_medians) else np.argmin(neighbor_means)
BLEND_STYLE = "MEDIAN" if min(neighbor_means)>min(neighbor_medians) else "MEAN"
plt.axvline(x=IDEAL_NUMBER_OF_NEIGHBORS, color='blue', linestyle="dotted")

plt.xlabel("Number of Neighbors", fontweight="bold")
plt.ylabel("MAE When Compared to Ground Truth", fontweight="bold")
plt.title(f"NEAREST NEIGHBOR MEAN v. MEDIAN FOR DIFFERENT AMOUNTS OF NEIGHBORS", fontweight="bold")
plt.legend(loc="upper right", prop={"size":14})
plt.grid(which="both")

plt.tight_layout()
plt.show()

print(f"\n\n... THE IDEAL NUMBER OF NEIGHBORS IS {IDEAL_NUMBER_OF_NEIGHBORS}, USING `{BLEND_STYLE}` WITH AN MAE OF {MIN_MAE} ...\n\n")


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4034167887.py in <cell line: 0>()
      2 BREATH_ID=87
      3 
----> 4 gt_breath = train_df.loc[train_df.breath_id==BREATH_ID].to_pandas()
      5 BREATH_R = gt_breath.R.values[0]
      6 BREATH_C = gt_breath.C.values[0]

NameError: name 'train_df' is not defined

## === cell 14
BREATH_ID=101

gt_breath = train_df.loc[train_df.breath_id==BREATH_ID].to_pandas()
BREATH_R = gt_breath.R.values[0]
BREATH_C = gt_breath.C.values[0]

df_to_use = train_df.copy()[(train_df.R==BREATH_R)&(train_df.C==BREATH_C)].reset_index(drop=True) if RESTRICT_RC else train_df.copy()

model = NearestNeighbors(n_neighbors=BLEND_NEIGHBORS, metric="l1",)
model.fit(df_to_use[USE_COLS])

distances, indices = model.kneighbors(gt_breath[USE_COLS])
distances, indices = distances[0, 1:], indices[0, 1:] #discard GT

gt_breath_pressure = gt_breath[PRESSURE_COLS].squeeze().values
nn_pressure_df = df_to_use.iloc[indices][PRESSURE_COLS]
nn_pressure_df.insert(loc=0, name="nn_distance", value=distances)
nn_pressures = nn_pressure_df[PRESSURE_COLS].values

for UP_TO in [BLEND_NEIGHBORS-1, 75, 50, 25, 12, 5, 2, 1]:
    print("\n\n\n")
    plt.figure(figsize=(20, 8))
    for i in range(UP_TO):
        if i==UP_TO-1:
            plt.plot(nn_pressures[i].get(), color="lightblue", linewidth=1, label="NEIGHBORS")
        else:
            plt.plot(nn_pressures[i].get(), color="lightblue")
    plt.plot(gt_breath_pressure, color="blue", linewidth=3, label="GROUND TRUTH")
    
    mean_pressure = nn_pressures[:UP_TO].mean(axis=0).get()
    mean_mae = np.abs(gt_breath_pressure-mean_pressure).mean()
    plt.plot(mean_pressure, color="deepskyblue", linewidth=3, label="NEIGHBOR MEAN")
    
    median_pressure = np.median(nn_pressures[:UP_TO].get(),axis=0)
    median_mae = np.abs(gt_breath_pressure-median_pressure).mean()
    plt.plot(median_pressure, color="mediumspringgreen", linewidth=3, label="NEIGHBOR MEDIAN")
    
    plt.title(f"{UP_TO} NEAREST NEIGHBORS VS GROUND TRUTH\n\nMEAN MAE = {mean_mae:.5f}\nMEDIAN MAE = {median_mae:.5f}", fontweight="bold")
    plt.xlabel("Step In Breath", fontweight="bold")
    plt.ylabel("Pressure (mm/H2O)", fontweight="bold")

    plt.legend(loc="upper right", prop={"size":14})
    plt.grid(which="both")
    
    plt.tight_layout()
    plt.show()
    
neighbor_means = []
neighbor_medians = []
for UP_TO in tqdm(range(1, BLEND_NEIGHBORS), total=BLEND_NEIGHBORS-1):
    mean_pressure = nn_pressures[:UP_TO].mean(axis=0).get()
    neighbor_means.append(np.abs(gt_breath_pressure-mean_pressure).mean())
    
    median_pressure = np.median(nn_pressures[:UP_TO].get(),axis=0)
    neighbor_medians.append(np.abs(gt_breath_pressure-median_pressure).mean())
    
    
    
plt.figure(figsize=(20, 8))

MIN_MAE = min(min(neighbor_means), min(neighbor_medians))

plt.plot(neighbor_means, color="deepskyblue", linewidth=3, label="NEIGHBOR MEAN")
plt.fill_between(x=np.arange(BLEND_NEIGHBORS-1), y1=neighbor_means, y2=MIN_MAE, color="deepskyblue", alpha=0.25)

plt.plot(neighbor_medians, color="mediumspringgreen", linewidth=3, label="NEIGHBOR MEDIAN")
plt.fill_between(x=np.arange(BLEND_NEIGHBORS-1), y1=neighbor_medians, y2=MIN_MAE, color="mediumspringgreen", alpha=0.25)

IDEAL_NUMBER_OF_NEIGHBORS = np.argmin(neighbor_medians) if min(neighbor_means)>min(neighbor_medians) else np.argmin(neighbor_means)
BLEND_STYLE = "MEDIAN" if min(neighbor_means)>min(neighbor_medians) else "MEAN"
plt.axvline(x=IDEAL_NUMBER_OF_NEIGHBORS, color='blue', linestyle="dotted")

plt.xlabel("Number of Neighbors", fontweight="bold")
plt.ylabel("MAE When Compared to Ground Truth", fontweight="bold")
plt.title(f"NEAREST NEIGHBOR MEAN v. MEDIAN FOR DIFFERENT AMOUNTS OF NEIGHBORS", fontweight="bold")
plt.legend(loc="upper right", prop={"size":14})
plt.grid(which="both")

plt.tight_layout()
plt.show()

print(f"\n\n... THE IDEAL NUMBER OF NEIGHBORS IS {IDEAL_NUMBER_OF_NEIGHBORS}, USING `{BLEND_STYLE}` WITH AN MAE OF {MIN_MAE} ...\n\n")


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3494875555.py in <cell line: 0>()
      2 BREATH_ID=101
      3 
----> 4 gt_breath = train_df.loc[train_df.breath_id==BREATH_ID].to_pandas()
      5 BREATH_R = gt_breath.R.values[0]
      6 BREATH_C = gt_breath.C.values[0]

NameError: name 'train_df' is not defined

## === cell 15
MAX_DISTANCE = 5000
IDEAL_NEIGHBOR_MAX = 25
CALCULATE_WEIGHTED = True

global_nn_means, global_nn_medians = [], []
global_wtd_nn_means, global_wtd_nn_medians = [], []

for BREATH_ID in tqdm(sorted(train_df.breath_id.values.get())[:20], total=20,):
    gt_breath = train_df.loc[train_df.breath_id==BREATH_ID].to_pandas()
    BREATH_R = gt_breath.R.values[0]
    BREATH_C = gt_breath.C.values[0]
    df_to_use = train_df.copy()[(train_df.R==BREATH_R)&(train_df.C==BREATH_C)].reset_index(drop=True) if RESTRICT_RC else train_df.copy()

    model = NearestNeighbors(n_neighbors=IDEAL_NEIGHBOR_MAX, metric="l1",)
    model.fit(df_to_use[USE_COLS])

    distances, indices = model.kneighbors(gt_breath[USE_COLS])
    distances, indices = distances[0, 1:], indices[0, 1:] #discard GT

    gt_breath_pressure = gt_breath[PRESSURE_COLS].squeeze().values
    nn_pressure_df = df_to_use.iloc[indices][PRESSURE_COLS]
    nn_pressure_df.insert(loc=0, name="nn_distance", value=distances)
    nn_pressures = nn_pressure_df[PRESSURE_COLS].values

    neighbor_means, neighbor_medians = [], []
    wtd_neighbor_means, wtd_neighbor_medians = [], []
    
    for UP_TO in range(1, IDEAL_NEIGHBOR_MAX):
        
        mean_pressure = nn_pressures[:UP_TO].mean(axis=0).get()
        median_pressure = np.median(nn_pressures[:UP_TO].get(),axis=0)
        neighbor_means.append(np.abs(gt_breath_pressure-mean_pressure).mean())
        neighbor_medians.append(np.abs(gt_breath_pressure-median_pressure).mean())
        
        if CALCULATE_WEIGHTED:
            wtd_mean_pressure = np.average(nn_pressures[:UP_TO], weights=MAX_DISTANCE-np.clip(distances[:UP_TO], 0, MAX_DISTANCE), axis=0).get()
            wtd_median_pressure = np.array([compute_weighted_median(_values=nn_pressures[:UP_TO, i].get(), _weights=(MAX_DISTANCE-np.clip(distances[:UP_TO], 0, MAX_DISTANCE))**0.5, redux_factor=3) for i in range(80)])
            
            wtd_neighbor_means.append(np.abs(gt_breath_pressure-mean_pressure).mean())
            wtd_neighbor_medians.append(np.abs(gt_breath_pressure-wtd_median_pressure).mean())
            
    global_nn_means.append(neighbor_means)
    global_nn_medians.append(neighbor_medians)
    
    if CALCULATE_WEIGHTED:
        global_wtd_nn_means.append(wtd_neighbor_means)
        global_wtd_nn_medians.append(wtd_neighbor_medians)
    
global_nn_mean_reduced = np.array(global_nn_means).mean(axis=0)
global_nn_median_reduced = np.array(global_nn_medians).mean(axis=0)

if CALCULATE_WEIGHTED:
    global_wtd_nn_mean_reduced = np.array(global_wtd_nn_means).mean(axis=0)
    global_wtd_nn_median_reduced = np.array(global_wtd_nn_medians).mean(axis=0)

plt.figure(figsize=(20, 8))

plt.plot(global_nn_mean_reduced, color="deepskyblue", linewidth=3, label="NEIGHBOR MEAN")
plt.plot(global_nn_median_reduced, color="mediumspringgreen", linewidth=3, label="NEIGHBOR MEDIAN")
plt.fill_between(x=np.arange(IDEAL_NEIGHBOR_MAX-1), y1=global_nn_mean_reduced, y2=global_nn_median_reduced, color="lightblue", alpha=0.5)

if CALCULATE_WEIGHTED:
    plt.plot(global_wtd_nn_mean_reduced, color="orangered", linewidth=3, label="NEIGHBOR WTD MEAN")
    plt.plot(global_wtd_nn_median_reduced, color="hotpink", linewidth=3, label="NEIGHBOR WTD MEDIAN")
    plt.fill_between(x=np.arange(IDEAL_NEIGHBOR_MAX-1), y1=global_wtd_nn_mean_reduced, y2=global_wtd_nn_median_reduced, color="lightcoral", alpha=0.5)

plt.xticks(np.arange(IDEAL_NEIGHBOR_MAX-1))
plt.xlabel("Number of Neighbors", fontweight="bold")
plt.ylabel("MAE Averaged Across First 100 Training Breaths", fontweight="bold")
plt.title(f"NEAREST NEIGHBOR MEAN v. MEDIAN FOR DIFFERENT AMOUNTS OF NEIGHBORS", fontweight="bold")
plt.legend(loc="upper right", prop={"size":14})
plt.grid(which="both")

plt.tight_layout()
plt.show()


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2429123368.py in <cell line: 0>()
      6 global_wtd_nn_means, global_wtd_nn_medians = [], []
      7 
----> 8 for BREATH_ID in tqdm(sorted(train_df.breath_id.values.get())[:20], total=20,):
      9     gt_breath = train_df.loc[train_df.breath_id==BREATH_ID].to_pandas()
     10     BREATH_R = gt_breath.R.values[0]

NameError: name 'tqdm' is not defined

## === cell 17
NN_TO_USE = 10
breath_ids, mean_test_pressures, median_test_pressures = [], [], []
for _R in test_df.R.unique().to_pandas():
    for _C in test_df.C.unique().to_pandas():
        test_df_to_use = test_df[(test_df.R==_R)&(test_df.C==_C)].reset_index(drop=True)
        train_df_to_use = train_df[(train_df.R==_R)&(train_df.C==_C)].reset_index(drop=True)
        model = NearestNeighbors(n_neighbors=NN_TO_USE, metric="l1")
        model.fit(train_df_to_use[USE_COLS])
        distances, indices = model.kneighbors(test_df_to_use[USE_COLS])
        distances, indices = distances.values, indices.values
        for i, BREATH_ID in tqdm(enumerate(sorted(test_df_to_use.breath_id.values.get())), total=len(test_df_to_use)):
            breath_ids.append(BREATH_ID)
            _distances, _indices = distances[i], indices[i]
            nn_pressures = train_df_to_use.iloc[_indices][PRESSURE_COLS].values.get()         
            mean_test_pressures.append(np.average(nn_pressures, axis=0, weights=_distances.get()))
            median_test_pressures.append(np.median(nn_pressures,axis=0))


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/144245238.py in <cell line: 0>()
      1 NN_TO_USE = 10
      2 breath_ids, mean_test_pressures, median_test_pressures = [], [], []
----> 3 for _R in test_df.R.unique().to_pandas():
      4     for _C in test_df.C.unique().to_pandas():
      5         test_df_to_use = test_df[(test_df.R==_R)&(test_df.C==_C)].reset_index(drop=True)

NameError: name 'test_df' is not defined

## === cell 18
ss_df = pd.merge(left=ss_df, right=pd.read_csv("../input/ventilator-pressure-prediction/test.csv")[["id", "breath_id"]], on="id")
ss_df["median_pressure"] = ss_df["pressure"].copy()
ss_df["mean_pressure"] = ss_df["pressure"].copy()

sort_indices = np.argsort(breath_ids)
ss_df["mean_pressure"] = np.array(mean_test_pressures)[sort_indices].reshape(-1)
ss_df["median_pressure"] = np.array(median_test_pressures)[sort_indices].reshape(-1)
ss_df["pressure"] = (ss_df["mean_pressure"]+ss_df["median_pressure"])/2
ss_df


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2500402839.py in <cell line: 0>()
----> 1 ss_df = pd.merge(left=ss_df, right=pd.read_csv("../input/ventilator-pressure-prediction/test.csv")[["id", "breath_id"]], on="id")
      2 ss_df["median_pressure"] = ss_df["pressure"].copy()
      3 ss_df["mean_pressure"] = ss_df["pressure"].copy()
      4 
      5 sort_indices = np.argsort(breath_ids)

NameError: name 'pd' is not defined

## === cell 19
for x in ["median", "mean", None]:
    p_name = "pressure" if not x else f"{x}_pressure"
    tmp_csv_df = ss_df[["id", p_name]]
    tmp_csv_df.columns = ["id", "pressure"]
    tmp_csv_df.to_csv(f"{p_name}_submission.csv", index=False)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1721306669.py in <cell line: 0>()
      1 for x in ["median", "mean", None]:
      2     p_name = "pressure" if not x else f"{x}_pressure"
----> 3     tmp_csv_df = ss_df[["id", p_name]]
      4     tmp_csv_df.columns = ["id", "pressure"]
      5     tmp_csv_df.to_csv(f"{p_name}_submission.csv", index=False)

NameError: name 'ss_df' is not defined

## === cell 20
BLEND_NEIGHBORS = 1000

model = NearestNeighbors(n_neighbors=BLEND_NEIGHBORS, metric="l1",)
model.fit(train_df[USE_COLS])

for i in range(BLEND_NEIGHBORS):
    distances, indices = model.kneighbors(gt_breath[USE_COLS])
    distances, indices = distances[0, 1:], indices[0, 1:] #discard GT
    break


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2340025472.py in <cell line: 0>()
      3 
      4 model = NearestNeighbors(n_neighbors=BLEND_NEIGHBORS, metric="l1",)
----> 5 model.fit(train_df[USE_COLS])
      6 
      7 # Causes OOM... on hold for now

NameError: name 'train_df' is not defined

## === cell 21
pd.read_csv("median_pressure_submission.csv")


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2320260288.py in <cell line: 0>()
----> 1 pd.read_csv("median_pressure_submission.csv")

NameError: name 'pd' is not defined
