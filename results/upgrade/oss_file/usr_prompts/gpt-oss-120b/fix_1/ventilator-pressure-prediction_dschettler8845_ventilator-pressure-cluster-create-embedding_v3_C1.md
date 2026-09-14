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

0.8938

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
print("\n... IMPORTS STARTING ...\n")
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

def add_features(df, use_rc=True):
    """ TBD """
    df['uin_auc'] = df['time_step'] * df['u_in']
    df['uin_auc'] = df.groupby('breath_id')['uin_auc'].cumsum()
    df['uin_csum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['uin_1_back'] = df.groupby("breath_id")['u_in'].shift(1).fillna(0)
    df['uout_1_back'] = df.groupby("breath_id")['u_out'].shift(1).fillna(0)
    df['uout_2_back'] = df.groupby("breath_id")['u_out'].shift(2).fillna(0)
    df['uin_2_back'] = df.groupby("breath_id")['u_in'].shift(2).fillna(0)
    df['uin_3_back'] = df.groupby("breath_id")['u_in'].shift(3).fillna(0)
    df['uin_5_back'] = df.groupby("breath_id")['u_in'].shift(5).fillna(0)
    df['uin_7_back'] = df.groupby("breath_id")['u_in'].shift(7).fillna(0)
    df['uout_1_forw'] = df.groupby("breath_id")['u_out'].shift(-1).fillna(0)
    df['uout_2_forw'] = df.groupby("breath_id")['u_out'].shift(-2).fillna(0)
    df['uin_1_forw'] = df.groupby("breath_id")['u_in'].shift(-1).fillna(0)
    df['uin_2_forw'] = df.groupby("breath_id")['u_in'].shift(-2).fillna(0)
    df['uin_3_forw'] = df.groupby("breath_id")['u_in'].shift(-3).fillna(0)
    df['uin_5_forw'] = df.groupby("breath_id")['u_in'].shift(-5).fillna(0)
    df['uin_7_forw'] = df.groupby("breath_id")['u_in'].shift(-7).fillna(0)
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    
    df['u_in_diff1'] = df['u_in'] - df['uin_1_back']
    df['u_out_diff1'] = df['u_out'] - df['uout_1_back']
    df['u_in_diff2'] = df['u_in'] - df['uin_2_back']
    df['u_out_diff2'] = df['u_out'] - df['uout_2_back']
    
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    
    df['u_in_diff3'] = df['u_in'] - df['uin_3_back']
    df['u_in_diff5'] = df['u_in'] - df['uin_5_back']
    df['cross3']= df['time_step']*df['u_in']
    df['cross3_sqd_1']= df['time_step']*df['u_in']**2
    df['cross3_sqd_2']= df['time_step']**2*df['u_in']
    
    if use_rc:
        df['R'] = df['R'].astype(str)
        df['C'] = df['C'].astype(str)
        df = pd.get_dummies(df,)
        
    return df


## === cell 6
print("\n... Here are the TRAIN counts for different combinations of R and C values.   NOTE: Counts are by BREATH (80 rows) ...\n")
display(("R"+train_df.R.astype(str)+"_C"+train_df.C.astype(str)).value_counts()//80)

print("\n\n\n... Here are the TEST counts for different combinations of R and C values.   NOTE: Counts are by BREATH (80 rows) ...\n")
display(("R"+test_df.R.astype(str)+"_C"+test_df.C.astype(str)).value_counts()//80)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/294010004.py in <cell line: 0>()
      1 print("\n... Here are the TRAIN counts for different combinations of R and C values.   NOTE: Counts are by BREATH (80 rows) ...\n")
----> 2 display(("R"+train_df.R.astype(str)+"_C"+train_df.C.astype(str)).value_counts()//80)
      3 
      4 print("\n\n\n... Here are the TEST counts for different combinations of R and C values.   NOTE: Counts are by BREATH (80 rows) ...\n")
      5 display(("R"+test_df.R.astype(str)+"_C"+test_df.C.astype(str)).value_counts()//80)

NameError: name 'train_df' is not defined

## === cell 7
def create_compressed_df(df):
    """ Function to 'compress' the original dataframe structure into one-row=one-breath format
    
    Args:
        df (cudf.Dataframe): A dataframe to perform the compression operation on (test or train)
    
    Returns:
        Updated dataframe
    """
    
    additional_info = {
        "inhale_uin_max":df[df.u_out==0].groupby('breath_id')[['u_in']].agg('max'), 
        "exhale_uin_max":df[df.u_out==1].groupby('breath_id')[['u_in']].agg('max'), 
        "inhale_uin_min":df[df.u_out==0].groupby('breath_id')[['u_in']].agg('min'), 
        "exhale_uin_min":df[df.u_out==1].groupby('breath_id')[['u_in']].agg('min'), 
        "breath_duration":df.groupby('breath_id')[['time_step']].agg('max'), 
        "exhale_steps":df.groupby('breath_id')[['u_out']].agg('sum'), 
        "inhale_steps":80-df.groupby('breath_id')[['u_out']].agg('sum'),
    }

    cdf = df.groupby('breath_id').collect().reset_index()
    for i in range(ROWS_PER_BREATH): cdf[f'x_ui_{i}'] = cdf.u_in.list.get(i)
    for i in range(ROWS_PER_BREATH): cdf[f'y_uo_{i}'] = 1-cdf.u_out.list.get(i)
    if "pressure" in cdf.columns:
        for i in range(ROWS_PER_BREATH): cdf[f'z_pr_{i}'] = cdf.pressure.list.get(i)
        
    cdf.R = cdf.R.list.get(0)
    cdf.C = cdf.C.list.get(0)
    
    if "pressure" in cdf.columns:
        cdf = cdf.drop(columns=["id","time_step","u_in","u_out","pressure","breath_step"],axis=1)
    else:
        cdf = cdf.drop(columns=["time_step","u_in","u_out","breath_step"],axis=1)
        cdf["id"] = cdf["id"].list.get(i)
    
    for c_name, c_ser in additional_info.items():
        cdf = cdf.merge(c_ser, on="breath_id", how="left").rename({c_ser.columns[0]:c_name}, axis=1)
        
    cdf = cdf.sort_values('breath_id').reset_index(drop=True)
    
    return cdf

train_df = create_compressed_df(train_df)
test_df = create_compressed_df(test_df)

print("\n\n... TRAIN DATAFRAME POST-COMPRESSION ...\n")
display(train_df)

print("\n... TRAIN DATAFRAME COLUMNS ...\n")
print(train_df.columns.tolist())

print("\n\n\n... TEST DATAFRAME POST-COMPRESSION ...\n")
display(test_df)

print("\n... TEST DATAFRAME COLUMNS (NO PRESSURE) ...\n")
print(test_df.columns.tolist())


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2944620571.py in <cell line: 0>()
     47     return cdf
     48 
---> 49 train_df = create_compressed_df(train_df)
     50 test_df = create_compressed_df(test_df)
     51 

NameError: name 'train_df' is not defined

## === cell 8

DEFAULT_X = np.arange(ROWS_PER_BREATH)
PRESSURE_COLORS = px.colors.sequential.Rainbow
RC_CONFIGS = [
    "R50_C10", "R50_C20", "R50_C50",
    "R5_C10", "R5_C20", "R5_C50",
    "R20_C50", "R20_C20", "R20_C10"
]
P_COLOR_MAP = {RC_CONFIG:PRESSURE_COLOR for RC_CONFIG,PRESSURE_COLOR in zip(RC_CONFIGS, PRESSURE_COLORS)}

IGNORE_COLS = ["breath_id", "R", "C"]
TIMESTEPS_TO_USE = range(ROWS_PER_BREATH)

UIN_COLS = [f"x_ui_{i}" for i in TIMESTEPS_TO_USE]
UOUT_COLS = [f"y_uo_{i}" for i in TIMESTEPS_TO_USE]
PRESSURE_COLS = [f"z_pr_{i}" for i in TIMESTEPS_TO_USE]
ADDITIONAL_FEATURE_COLS = [
    'inhale_uin_max', 'exhale_uin_max', 'inhale_uin_min', 
    'exhale_uin_min', 'breath_duration', 'exhale_steps', 'inhale_steps'
]

USE_COLS = UIN_COLS+ADDITIONAL_FEATURE_COLS

NEIGHBORS = 120

model = NearestNeighbors(n_neighbors=NEIGHBORS, metric="euclidean",)

model.fit(train_df[USE_COLS])


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/670913235.py in <cell line: 0>()
      4 
      5 # SETUP
----> 6 DEFAULT_X = np.arange(ROWS_PER_BREATH)
      7 PRESSURE_COLORS = px.colors.sequential.Rainbow
      8 RC_CONFIGS = [

NameError: name 'np' is not defined

## === cell 9
BREATH_IDS = [1,2,3,28,87,101] # Similar to cdeotte
for BREATH_ID in BREATH_IDS:
    original_breath = train_df.loc[train_df.breath_id==BREATH_ID].to_pandas()
    distances, indices = model.kneighbors(original_breath[USE_COLS])
    distances, indices, original_breath = distances[0], indices[0], original_breath.squeeze()
    original_breath_config = "R"+original_breath["R"].astype(int).astype(str)+"_C"+original_breath["C"].astype(int).astype(str)
    print("distances shape:",distances.shape, "\nindices shape:",indices.shape, "\n\n")

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=DEFAULT_X, 
                             y=original_breath[UIN_COLS],
                             line=dict(color='purple'), name='U_IN',
                             text=f"ORIGINAL U_IN"))
    fig.add_trace(go.Scatter(x=DEFAULT_X, 
                             y=original_breath[PRESSURE_COLS],
                             line=dict(color='black', width=3), name='PRESSURE',
                             text=f"ORIGINAL PRESSURE"))
    fig.add_vrect(x0=0, x1=original_breath.squeeze()["inhale_steps"], line=dict(color='grey', dash='dash'), annotation_text="INHALE", annotation_position="top",fillcolor="black", opacity=0.1)
    fig.update_layout(title=f'ORIGINAL U_IN/PRESSURE PLOT FOR BREATH_ID={BREATH_ID}',
                      xaxis_title='Time Step',
                      yaxis_title='Unit Different For Respective Lines',
                      legend=dict(
                          yanchor="top",
                          y=0.99,
                          xanchor="right",
                          x=0.995,
                          font=dict(size=11)))
    fig.show()

    fig = go.Figure()
    colors_used=[]; add_to_legend=True
    for i, (__d, __idx) in enumerate(zip(distances, indices)):
        if i==0:
            continue

        knn_breath = train_df.iloc[__idx].to_pandas().squeeze()
        breath_config = "R"+knn_breath["R"].astype(int).astype(str)+"_C"+knn_breath["C"].astype(int).astype(str)
        breath_color = P_COLOR_MAP[breath_config]

        if breath_color not in colors_used: colors_used.append(breath_color); add_to_legend=True;
        fig.add_trace(go.Scatter(x=DEFAULT_X, 
                                 y=knn_breath[UIN_COLS], legendgroup="U_IN", showlegend=True if i==1 else False, name="U_IN", line=dict(color='purple'), 
                                 text=f"NEIGHBOR U_IN#{i} Closest Breath - Distance Of {__d}", visible='legendonly'))
        fig.add_trace(go.Scatter(x=DEFAULT_X, 
                                 y=knn_breath[PRESSURE_COLS], legendgroup=f"NEIGHBOR PRESSURE ({breath_config})", showlegend=add_to_legend, name="" if not add_to_legend else f"NEIGHBOR PRESSURE ({breath_config})", line=dict(color="grey") if original_breath_config==breath_config else dict(color=breath_color), 
                                 text=f"NEIGHBOR PRESSURERC Config={breath_config}#{i} Closest Breath - Distance Of {__d}",))
        add_to_legend=False

    fig.add_trace(go.Scatter(x=DEFAULT_X, 
                             y=original_breath[PRESSURE_COLS],
                             name=f"ORIGINAL PRESSURE ({original_breath_config})",
                             line=dict(color='black', dash='dash', width=3),))

    fig.update_layout(title=f'{NEIGHBORS} NEAREST NEIGHBOR U_IN/PRESSURE PLOT FOR BREATH_ID={BREATH_ID}',
                      xaxis_title='Time Step',
                      yaxis_title='Unit Different For Respective Lines',
                      legend=dict(
                          yanchor="top",
                          y=0.99,
                          xanchor="right",
                          x=0.995,
                          font=dict(size=11)))
    fig.show()


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3142303828.py in <cell line: 0>()
      1 BREATH_IDS = [1,2,3,28,87,101] # Similar to cdeotte
      2 for BREATH_ID in BREATH_IDS:
----> 3     original_breath = train_df.loc[train_df.breath_id==BREATH_ID].to_pandas()
      4     distances, indices = model.kneighbors(original_breath[USE_COLS])
      5     distances, indices, original_breath = distances[0], indices[0], original_breath.squeeze()

NameError: name 'train_df' is not defined

## === cell 10
test_df.head(3)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2127995872.py in <cell line: 0>()
----> 1 test_df.head(3)

NameError: name 'test_df' is not defined

## === cell 12
instead of blind median... do weighted median?


## === cell 13
all_pred = []
all_test_id = []

NEIGHBORS=50
model = NearestNeighbors(n_neighbors=NEIGHBORS, metric="euclidean",)

for r in tqdm([5,20,50], total=3):
    for c in tqdm([10,20,50], total=3):
        sub_test_df = test_df[(test_df.R==r) & (test_df.C==c)]
        sub_train_df = train_df[(train_df.R==r) & (train_df.C==c)]
        
        model.fit(sub_train_df[USE_COLS])
        distances, indices = model.kneighbors(sub_test_df[USE_COLS])
        
        for i, _id in tqdm(enumerate(sub_test_df.id.values.get()), total=len(sub_test_df)):
            ss_df.loc[(ss_df.id>=_id) & (ss_df.id<(_id+ROWS_PER_BREATH)), "pressure"] = sub_train_df[PRESSURE_COLS].iloc[indices.iloc[i]].to_pandas().median().values


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1559740555.py in <cell line: 0>()
      6 model = NearestNeighbors(n_neighbors=NEIGHBORS, metric="euclidean",)
      7 
----> 8 for r in tqdm([5,20,50], total=3):
      9     for c in tqdm([10,20,50], total=3):
     10         # Get RC SubDataFrames

NameError: name 'tqdm' is not defined

## === cell 14
ss_df.to_csv("submission.csv", index=False)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3694584876.py in <cell line: 0>()
----> 1 ss_df.to_csv("submission.csv", index=False)

NameError: name 'ss_df' is not defined

## === cell 15
VAL_SPLIT_NTH = 8
REPLICA_BATCH_SIZE=64
OVERALL_BATCH_SIZE=N_REPLICAS*REPLICA_BATCH_SIZE
N_EPOCHS = 250
INIT_LR = 0.00125
LR_DECAY_RATE = 4


## === cell 16
print("\n... ADDING FEATURES TO TRAIN AND TEST DATAFRAMES ...\n")
train_df["R_C"] = "R"+train_df.R.astype(str)+"_C"+train_df.C.astype(str)
train_df = add_features(train_df, use_rc=False)
test_df["R_C"] = "R"+test_df.R.astype(str)+"_C"+test_df.C.astype(str)
test_df = add_features(test_df, use_rc=False)

print("\n... CREATING SUB DATAFRAME MAP FOR TRAIN ...\n")
train_sub_df_map = {
    unq_rc:train_df[train_df.R_C==unq_rc].reset_index(drop=True) \
    for unq_rc in train_df["R_C"].unique()
}

print("\n... CREATING SUB DATAFRAME MAP FOR TEST ...\n")
test_sub_df_map = {
    unq_rc:test_df[test_df.R_C==unq_rc].reset_index(drop=True) \
    for unq_rc in test_df["R_C"].unique()
}

LABEL_NAMES = ["pressure",]
GROUPBY_NAMES = ["breath_id"]
IGNORE_NAMES = ["id", "R", "C", "R_C", "index"]
FEATURE_NAMES = [x for x in train_df.columns if x not in LABEL_NAMES+GROUPBY_NAMES+IGNORE_NAMES]
N_FEATURES = len(FEATURE_NAMES)

RS = RobustScaler()
RS.fit(train_df[FEATURE_NAMES].to_numpy())

del train_df
del test_df
gc.collect(); gc.collect(); gc.collect();

train_sub_df_map["R20_C10"]


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4054984438.py in <cell line: 0>()
      1 print("\n... ADDING FEATURES TO TRAIN AND TEST DATAFRAMES ...\n")
----> 2 train_df["R_C"] = "R"+train_df.R.astype(str)+"_C"+train_df.C.astype(str)
      3 train_df = add_features(train_df, use_rc=False)
      4 test_df["R_C"] = "R"+test_df.R.astype(str)+"_C"+test_df.C.astype(str)
      5 test_df = add_features(test_df, use_rc=False)

NameError: name 'train_df' is not defined

## === cell 17
train_ds_map = {k:{} for k in train_sub_df_map.keys()}
val_ds_map = {k:{} for k in train_sub_df_map.keys()}
test_ds_map = {k:{} for k in test_sub_df_map.keys()}

for k,df in tqdm(train_sub_df_map.items()):
    for train_indices, val_indices in GroupKFold(n_splits=VAL_SPLIT_NTH).split(
        X=df[FEATURE_NAMES], y=df[LABEL_NAMES], groups=df[GROUPBY_NAMES]
    ):
        sub_train_df = df.iloc[val_indices].reset_index(drop=True)
        sub_val_df = df.iloc[train_indices].reset_index(drop=True)
        
        sub_train_x_np = sub_train_df[FEATURE_NAMES].to_numpy()
        sub_train_x_np = RS.fit_transform(sub_train_x_np).reshape(-1, ROWS_PER_BREATH, N_FEATURES)
        sub_train_y_np = sub_train_df[LABEL_NAMES].to_numpy().reshape((-1, ROWS_PER_BREATH, 1))
        train_ds_map[k]["x"] = sub_train_x_np
        train_ds_map[k]["y"] = sub_train_y_np
        
        sub_val_x_np   = sub_val_df[FEATURE_NAMES].to_numpy()
        sub_val_x_np = RS.transform(sub_val_x_np).reshape(-1, ROWS_PER_BREATH, N_FEATURES)
        sub_val_y_np = sub_val_df[LABEL_NAMES].to_numpy().reshape((-1, ROWS_PER_BREATH, 1))
        val_ds_map[k]["x"] = sub_val_x_np
        val_ds_map[k]["y"] = sub_val_y_np
        
        break
        
for k,df in tqdm(test_sub_df_map.items()):
    sub_test_x_np  = df[FEATURE_NAMES].to_numpy()
    sub_test_id = df["id"].to_numpy()
    test_ds_map[k]["x"] = sub_test_x_np 
    test_ds_map[k]["id"] = sub_test_id


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2368509501.py in <cell line: 0>()
----> 1 train_ds_map = {k:{} for k in train_sub_df_map.keys()}
      2 val_ds_map = {k:{} for k in train_sub_df_map.keys()}
      3 test_ds_map = {k:{} for k in test_sub_df_map.keys()}
      4 
      5 for k,df in tqdm(train_sub_df_map.items()):

NameError: name 'train_sub_df_map' is not defined

## === cell 18
def get_model():
    _inputs = tf.keras.layers.Input(shape=(ROWS_PER_BREATH, N_FEATURES))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(768, return_sequences=True))(_inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(768, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(384, return_sequences=True))(x)    
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(192, activation="selu", name="head_layer")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    _outputs = tf.keras.layers.Dense(1, name="classification_layer")(x)
    return tf.keras.Model(inputs=_inputs, outputs=_outputs)

save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')


## === cell 19
histories = {k:None for k in train_sub_df_map.keys()}
models = []
with strategy.scope():
    for rc_key, history in histories.items():
        print(f"\n\n\n... STARTING TRAINING FOR {rc_key} ...\n\n\n")
        cb_list = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=40, verbose=1, mode="min", restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.888, patience=9, verbose=1),
        ]
        model = get_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0011), loss="mae")
        history[rc_key] = model.fit(train_ds_map[rc_key]["x"], train_ds_map[rc_key]["y"], validation_data=(val_ds_map[rc_key]["x"], val_ds_map[rc_key]["y"]), epochs=N_EPOCHS, batch_size=OVERALL_BATCH_SIZE, callbacks=cb_list)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1041111970.py in <cell line: 0>()
----> 1 histories = {k:None for k in train_sub_df_map.keys()}
      2 models = []
      3 with strategy.scope():
      4     for rc_key, history in histories.items():
      5         print(f"\n\n\n... STARTING TRAINING FOR {rc_key} ...\n\n\n")

NameError: name 'train_sub_df_map' is not defined

## === cell 20
model.evaluate(train_x_np, train_y_np, batch_size=OVERALL_BATCH_SIZE)
model.evaluate(val_x_np, val_y_np, batch_size=OVERALL_BATCH_SIZE)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/463419023.py in <cell line: 0>()
----> 1 model.evaluate(train_x_np, train_y_np, batch_size=OVERALL_BATCH_SIZE)
      2 model.evaluate(val_x_np, val_y_np, batch_size=OVERALL_BATCH_SIZE)

base.pyx in cuml.internals.base.UniversalBase.__getattr__()

base.pyx in cuml.internals.base.UniversalBase.__getattr__()

base.pyx in cuml.internals.base.Base.__getattr__()

AttributeError: evaluate

## === cell 21
def get_feature_model(full_model, feature_layer_name="head_layer"):
    """ Get feature extractor model """
    _inputs = full_model.inputs
    _outputs = full_model.get_layer(feature_layer_name).output
    return tf.keras.Model(inputs=_inputs, outputs=[_outputs])

feature_model = get_feature_model(model)
feature_model.summary()


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1006891729.py in <cell line: 0>()
      5     return tf.keras.Model(inputs=_inputs, outputs=[_outputs])
      6 
----> 7 feature_model = get_feature_model(model)
      8 feature_model.summary()

/tmp/ipykernel_11/1006891729.py in get_feature_model(full_model, feature_layer_name)
      1 def get_feature_model(full_model, feature_layer_name="head_layer"):
      2     """ Get feature extractor model """
----> 3     _inputs = full_model.inputs
      4     _outputs = full_model.get_layer(feature_layer_name).output
      5     return tf.keras.Model(inputs=_inputs, outputs=[_outputs])

base.pyx in cuml.internals.base.UniversalBase.__getattr__()

base.pyx in cuml.internals.base.UniversalBase.__getattr__()

base.pyx in cuml.internals.base.Base.__getattr__()

AttributeError: inputs

## === cell 22
train_preds = model.predict(train_x_np)
val_preds = model.predict(val_x_np)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3764001462.py in <cell line: 0>()
----> 1 train_preds = model.predict(train_x_np)
      2 val_preds = model.predict(val_x_np)

base.pyx in cuml.internals.base.UniversalBase.__getattr__()

base.pyx in cuml.internals.base.UniversalBase.__getattr__()

base.pyx in cuml.internals.base.Base.__getattr__()

AttributeError: predict

## === cell 23
val_df_1.val_id


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/361012546.py in <cell line: 0>()
----> 1 val_df_1.val_id

NameError: name 'val_df_1' is not defined

## === cell 24
val_df_1[val_df_1.mae>0.2].groupby("breath_id").first().C_50.hist()
plt.show()

val_df_1.groupby("breath_id").first().C_50.hist()
plt.show()


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4266054123.py in <cell line: 0>()
----> 1 val_df_1[val_df_1.mae>0.2].groupby("breath_id").first().C_50.hist()
      2 plt.show()
      3 
      4 val_df_1.groupby("breath_id").first().C_50.hist()
      5 plt.show()

NameError: name 'val_df_1' is not defined

## === cell 25
val_df_1["val_id"] = (np.arange(len(val_df_1))/80).astype(int)
val_df_1["pred_pressure"] = val_preds.reshape(-1)
val_df_1["mae"] = val_df_1["val_id"].apply(lambda x: val_mae[x])

for idx, breath in val_df_1[val_df_1.mae>0.25].groupby("breath_id"):
    fig = px.line(x=range(80), y=[breath.u_in, breath.pressure, breath.pred_pressure])
    fig.show()
    if idx>1000:
        break


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1863085450.py in <cell line: 0>()
----> 1 val_df_1["val_id"] = (np.arange(len(val_df_1))/80).astype(int)
      2 val_df_1["pred_pressure"] = val_preds.reshape(-1)
      3 val_df_1["mae"] = val_df_1["val_id"].apply(lambda x: val_mae[x])
      4 
      5 for idx, breath in val_df_1[val_df_1.mae>0.25].groupby("breath_id"):

NameError: name 'np' is not defined

## === cell 26
train_mae = [np.mean(np.abs(ex-train_y_np[i])) for i, ex in tqdm(enumerate(train_preds))]
val_mae = [np.mean(np.abs(ex-val_y_np[i])) for i, ex in tqdm(enumerate(val_preds))]


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4104805458.py in <cell line: 0>()
----> 1 train_mae = [np.mean(np.abs(ex-train_y_np[i])) for i, ex in tqdm(enumerate(train_preds))]
      2 val_mae = [np.mean(np.abs(ex-val_y_np[i])) for i, ex in tqdm(enumerate(val_preds))]

NameError: name 'tqdm' is not defined

## === cell 27
train_preds.shape


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1383573296.py in <cell line: 0>()
----> 1 train_preds.shape

NameError: name 'train_preds' is not defined

## === cell 29
preds = model.predict(test_ds)
preds = preds[:-int(ROWS_TO_PAD/80)]
print(preds.shape)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3326380399.py in <cell line: 0>()
      1 # GET PREDS
----> 2 preds = model.predict(test_ds)
      3 preds = preds[:-int(ROWS_TO_PAD/80)]
      4 print(preds.shape)

base.pyx in cuml.internals.base.UniversalBase.__getattr__()

base.pyx in cuml.internals.base.UniversalBase.__getattr__()

base.pyx in cuml.internals.base.Base.__getattr__()

AttributeError: predict

## === cell 30

tmp_df = pd.DataFrame()
tmp_df["possible_pressures"] = POSSIBLE_PRESSURES
tmp_df["delta"] = tmp_df["possible_pressures"].shift(1)-tmp_df["possible_pressures"]
tmp_df


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3967026374.py in <cell line: 0>()
      1 # plt.scatter(range(950), sorted(POSSIBLE_PRESSURES))
      2 
----> 3 tmp_df = pd.DataFrame()
      4 tmp_df["possible_pressures"] = POSSIBLE_PRESSURES
      5 tmp_df["delta"] = tmp_df["possible_pressures"].shift(1)-tmp_df["possible_pressures"]

NameError: name 'pd' is not defined

## === cell 31
ss_df["pressure"] = tf.reshape(preds, (-1,))
ss_df.to_csv("./submission.csv", index=False)
ss_df


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4255199085.py in <cell line: 0>()
----> 1 ss_df["pressure"] = tf.reshape(preds, (-1,))
      2 # ss_df["pressure"] = ((len(POSSIBLE_PRESSURES)-1)*(ss_df["pressure"]-ss_df["pressure"].min())/(ss_df["pressure"].max()-ss_df["pressure"].min())).round().astype(int)
      3 # ss_df["pressure"] = ss_df["pressure"].apply(lambda x: POSSIBLE_PRESSURES[x])
      4 ss_df.to_csv("./submission.csv", index=False)
      5 ss_df

NameError: name 'preds' is not defined

## === cell 32
model.save("./best_model", options=save_locally)


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3931228387.py in <cell line: 0>()
----> 1 model.save("./best_model", options=save_locally)

base.pyx in cuml.internals.base.UniversalBase.__getattr__()

base.pyx in cuml.internals.base.UniversalBase.__getattr__()

base.pyx in cuml.internals.base.Base.__getattr__()

AttributeError: save
