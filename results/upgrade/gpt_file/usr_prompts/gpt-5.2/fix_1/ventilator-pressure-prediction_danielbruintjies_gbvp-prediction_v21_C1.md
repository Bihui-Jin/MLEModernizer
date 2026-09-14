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

0.1491524544004632

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 4
import os 
import time
import numpy as np
import pandas as pd
import math
import random
import gc

from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
from numpy.random import seed
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
from sklearn.model_selection import KFold, GroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error as mae
from scipy import stats
from scipy.stats import pearsonr
import scipy as sc

import lightgbm as lgb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tqdm.keras import TqdmCallback
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
get_custom_objects().update({'swish': Activation(swish)})

from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.matlib

import pickle
from pickle import dump
from pickle import load

import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_columns', 300)
sns.set_style("darkgrid")
sns.set_palette("dark")
gc.enable()

def set_seed(seeed):
    seed(seeed)
    tf.random.set_seed(seeed)
    os.environ['PYTHONHASHSEED'] = str(seeed)
    np.random.seed(seeed)
    random.seed(seeed)

start_time = time.time()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
def connect_to_tpu(tpu_address: str = None):
    if tpu_address is not None:  # When using GCP
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_address)
        if tpu_address not in ("", "local"):
            tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
        print("Running on TPU ", cluster_resolver.master())
        print("REPLICAS: ", strategy.num_replicas_in_sync)
        return cluster_resolver, strategy
    else:                           # When using Colab or Kaggle
        try:
            cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
            strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
            print("Running on TPU ", cluster_resolver.master())
            print("REPLICAS: ", strategy.num_replicas_in_sync)
            return cluster_resolver, strategy
        except:
            print("WARNING: No TPU detected.")
            mirrored_strategy = tf.distribute.MirroredStrategy()
            return None, mirrored_strategy
cluster_resolver, strategy = connect_to_tpu()

## === cell 6
DEBUG = False
TRAIN_MODEL = True

## === cell 9
train = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
test = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')
submission = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')




if DEBUG:
    train = train[:80*1000]
    test = test[:80*100]
    submission = submission[:80*100]

## === cell 10
print('TRAIN\n')
display(train)
print('\n\nTEST\n')
display(test)

## === cell 12
print(f'Length of TRAIN dataset: {len(train)}')
print(f'Length of TEST dataset: {len(test)}')

print(f'Number of breaths in train dataset: {train["breath_id"].nunique()}')
print(f'Number of breaths in test dataset: {test["breath_id"].nunique()}')
print(f'The number of observations for each breath: {train["breath_id"].value_counts().reset_index()["breath_id"].unique()[0]}')

## === cell 13
display(test[test['breath_id']==0])

## === cell 15
train_gf = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')

plt.title('Histogram of Train Pressures',size=14)
plt.hist(train_gf.sample(100_000).pressure.values,bins=100)
plt.show()
print('Max pressure =',train_gf.pressure.max(), 'Min pressure =',train_gf.pressure.min())

## === cell 16
all_pressure = np.sort( train_gf.pressure.unique())
del train_gf
print('The first 25 unique pressures...')
PRESSURE_MIN = all_pressure[0].item()
PRESSURE_MAX = all_pressure[-1].item()
all_pressure[:25]

## === cell 17
print('The differences between first 25 pressures...')
PRESSURE_STEP = ( all_pressure[1] - all_pressure[0] ).item()
all_pressure[1:26] - all_pressure[:25]

## === cell 20
train["log_u_in"] = np.log1p(train.u_in)
test["log_u_in"] = np.log1p(test.u_in)

train["time_step_class"] = pd.qcut(train.time_step, q=80, labels=range(0,80))
test["time_step_class"] = pd.qcut(test.time_step, q=80, labels=range(0,80))

piv = train.pivot_table(index="breath_id", columns="time_step_class", values="log_u_in", fill_value=0, aggfunc="mean")
piv_test = test.pivot_table(index="breath_id", columns="time_step_class", values="log_u_in", fill_value=0, aggfunc="mean")

piv.head()

## === cell 21
pca = PCA(n_components=2, random_state=42)
pca.fit(piv)

plt.plot(pca.explained_variance_ratio_.cumsum())
plt.grid()
plt.xlabel("n_components")
plt.ylabel("explained_variance_ratio_")
plt.xticks([0, 1])
plt.show()

## === cell 22
train_pca = pca.transform(piv)
test_pca = pca.transform(piv_test)

train_pca = pd.DataFrame(train_pca, columns=["c"+str(c) for c in range(2)], index=piv.index)
test_pca = pd.DataFrame(test_pca, columns=["c"+str(c) for c in range(2)], index=piv_test.index)

train_pca.head()

## === cell 23
sns.scatterplot(data=train_pca, x="c0", y="c1")
plt.show()

## === cell 24
km = KMeans(n_clusters=5, 
            random_state=42,
            max_iter=200,
            init="k-means++", 
            tol=0.0001)
y_km = km.fit_predict(train_pca)
y_km_test = km.predict(test_pca)

## === cell 25
train_pca["cluster"] = y_km
test_pca["cluster"] = y_km_test

center = km.cluster_centers_

sns.scatterplot(data=train_pca, x="c0", y="c1", hue="cluster")
plt.plot(center[0, 0], center[0, 1], "bo", c="r")
plt.plot(center[1, 0], center[1, 1], "bo", c="r")
plt.plot(center[2, 0], center[2, 1], "bo", c="r")
plt.plot(center[3, 0], center[3, 1], "bo", c="r")
plt.plot(center[4, 0], center[4, 1], "bo", c="r")


plt.show()

## === cell 26

'''
Let's try how it was isolated using the information obtained from the cluster
'''

train_pca["breath_id"] = train_pca.index 
train_pca.drop(["c0", "c1"], axis=1, inplace=True)
train_pca = train_pca.reset_index(drop=True)
train = pd.merge(train, train_pca, how="left", on="breath_id")

test_pca["breath_id"] = test_pca.index 
test_pca.drop(["c0", "c1"], axis=1, inplace=True)
test_pca = test_pca.reset_index(drop=True)
test = pd.merge(test, test_pca, how="left", on="breath_id")

def find_cluster_r_c(df):
    fig, ax = plt.subplots(5, 2, figsize=(15, 10))
    for c in range(5):
        for r_c in range(2):
            x = df.loc[df.cluster == c, "R" if r_c == 0 else "C" ]
            sns.countplot(x, ax=ax[c][r_c])
            ax[c][r_c].set_title(f"Cluster={c}")
    plt.tight_layout()
    
    
def find_cluster_transition(df, is_train=True):
    fig, ax = plt.subplots(5, 5, figsize=(15, 10))
    for c in range(5):
        x = df.loc[df.cluster == c]
        breath = x.breath_id.unique()
        for n in range(5):
            if is_train:
                xx = x.loc[x.breath_id == breath[n], ["time_step", "u_in", "u_out", "pressure"]]
            else:
                xx = x.loc[x.breath_id == breath[n], ["time_step", "u_in", "u_out"]]
            xx.set_index("time_step").plot(ax=ax[c][n])
            ax[c][n].set_title(f"breath_id={breath[n]}")
            ax[c][n].set_xticks([])
            
            if n == 0:
                ax[c][n].set_ylabel(f"Cluster={c}")
    plt.tight_layout()

## === cell 27
sns.countplot(train['cluster'])

## === cell 28
sns.countplot(test.cluster)

## === cell 29

'''
Some clusters show a single attribute.
It is more strongly reflected in clusters that do not exist in a straight line.
'''

find_cluster_r_c(train)

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item()

KeyError: 0

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1788108955.py in <cell line: 0>()
      4 '''
      5 
----> 6 find_cluster_r_c(train)

/tmp/ipykernel_11/1269151288.py in find_cluster_r_c(df)
     20         for r_c in range(2):
     21             x = df.loc[df.cluster == c, "R" if r_c == 0 else "C" ]
---> 22             sns.countplot(x, ax=ax[c][r_c])
     23             ax[c][r_c].set_title(f"Cluster={c}")
     24     plt.tight_layout()

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in countplot(data, x, y, hue, order, hue_order, orient, color, palette, saturation, width, dodge, ax, **kwargs)
   2941         raise ValueError("Cannot pass values for both `x` and `y`")
   2942 
-> 2943     plotter = _CountPlotter(
   2944         x, y, hue, data, order, hue_order,
   2945         estimator, errorbar, n_boot, units, seed,

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in __init__(self, x, y, hue, data, order, hue_order, estimator, errorbar, n_boot, units, seed, orient, color, palette, saturation, width, errcolor, errwidth, capsize, dodge)
   1528                  errcolor, errwidth, capsize, dodge):
   1529         """Initialize the plotter."""
-> 1530         self.establish_variables(x, y, hue, data, orient,
   1531                                  order, hue_order, units)
   1532         self.establish_colors(color, palette, saturation)

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in establish_variables(self, x, y, hue, data, orient, order, hue_order, units)
    484                 if hasattr(data, "shape"):
    485                     if len(data.shape) == 1:
--> 486                         if np.isscalar(data[0]):
    487                             plot_data = [data]
    488                         else:

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __getitem__(self, key)
   1119 
   1120         elif key_is_scalar:
-> 1121             return self._get_value(key)
   1122 
   1123         # Convert generator to list before going through hashable part

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1235 
   1236         # Similar to Index.get_value, but we do not fall back to positional
-> 1237         loc = self.index.get_loc(label)
   1238 
   1239         if is_integer(loc):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 0

## === cell 30
find_cluster_r_c(test)

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item()

KeyError: 0

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3210862479.py in <cell line: 0>()
----> 1 find_cluster_r_c(test)

/tmp/ipykernel_11/1269151288.py in find_cluster_r_c(df)
     20         for r_c in range(2):
     21             x = df.loc[df.cluster == c, "R" if r_c == 0 else "C" ]
---> 22             sns.countplot(x, ax=ax[c][r_c])
     23             ax[c][r_c].set_title(f"Cluster={c}")
     24     plt.tight_layout()

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in countplot(data, x, y, hue, order, hue_order, orient, color, palette, saturation, width, dodge, ax, **kwargs)
   2941         raise ValueError("Cannot pass values for both `x` and `y`")
   2942 
-> 2943     plotter = _CountPlotter(
   2944         x, y, hue, data, order, hue_order,
   2945         estimator, errorbar, n_boot, units, seed,

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in __init__(self, x, y, hue, data, order, hue_order, estimator, errorbar, n_boot, units, seed, orient, color, palette, saturation, width, errcolor, errwidth, capsize, dodge)
   1528                  errcolor, errwidth, capsize, dodge):
   1529         """Initialize the plotter."""
-> 1530         self.establish_variables(x, y, hue, data, orient,
   1531                                  order, hue_order, units)
   1532         self.establish_colors(color, palette, saturation)

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in establish_variables(self, x, y, hue, data, orient, order, hue_order, units)
    484                 if hasattr(data, "shape"):
    485                     if len(data.shape) == 1:
--> 486                         if np.isscalar(data[0]):
    487                             plot_data = [data]
    488                         else:

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __getitem__(self, key)
   1119 
   1120         elif key_is_scalar:
-> 1121             return self._get_value(key)
   1122 
   1123         # Convert generator to list before going through hashable part

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1235 
   1236         # Similar to Index.get_value, but we do not fall back to positional
-> 1237         loc = self.index.get_loc(label)
   1238 
   1239         if is_integer(loc):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 0

## === cell 31
'''
From the time-series data distribution,
we confirmed a sharp rise in u_in following 1.0 second. 
After that, it decreases smoothly.
'''

find_cluster_transition(train)

## === cell 32
find_cluster_transition(test, False)

## === cell 33
train.drop(["time_step_class", "log_u_in"], axis=1, inplace=True)
test.drop(["time_step_class", "log_u_in"], axis=1, inplace=True)

## === cell 34
display(test)
print(test.shape)
display(train)
print(train.shape)

## === cell 35
%%time

def add_features(dff):
    df = dff.copy()
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    
    df['u_in_lag1'] =  df.groupby('breath_id')['u_in'].shift(1)
    df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    df['u_in_lag8'] = df.groupby('breath_id')['u_in'].shift(8)
    df['u_in_lag_back8'] = df.groupby('breath_id')['u_in'].shift(-8)
    df['u_out_lag_back8'] = df.groupby('breath_id')['u_out'].shift(-8)
    df['time_step_diff3'] = df.groupby('breath_id')['time_step'].diff(3)
    df['u_in_pct'] = df.groupby('breath_id')['u_in'].pct_change()
    df['u_in_pct10'] = df.groupby('breath_id')['u_in'].pct_change(10)
    df['u_in_rolling8'] = list(df.groupby('breath_id')['u_in'].rolling(window=8).mean().fillna(0))
    df['u_out_rolling8'] = list(df.groupby('breath_id')['u_out'].rolling(window=8).mean().fillna(0))
    df['u_in_rolling4'] = list(df.groupby('breath_id')['u_in'].rolling(window=4).mean().fillna(0))
    df['u_in_expanding5'] = list(df.groupby('breath_id')['u_in'].expanding(5).mean().fillna(0))
    df['u_in_expanding2'] = list(df.groupby('breath_id')['u_in'].expanding(2).mean().fillna(0))
    df = df.fillna(0)
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']   
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')  
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']  
    
    df['u_in_change']= df.groupby('breath_id')['u_in'].diff(-1)
    df['delta_time']=df.groupby('breath_id')['time_step'].diff(-1)
    df['area_u_in']=df['u_in']*df['delta_time']
    df['area_u_in_abs']=df['u_in_change']*df['delta_time']
    df['uin_in_time']=df['u_in_change']/df['delta_time']
    
    
    cvc = df['cluster'].value_counts()
    df['cvc'] = df['cluster'].apply(lambda x: cvc[x])
    df['cluster__u_in__diffmean'] = df.groupby(['cluster'])['u_in'].transform('mean') - df['u_in']
    
    df['mean_RC'] = df.groupby(['R', 'C'])['u_in'].transform('mean') - df['u_in']
    df['max_RC'] = df.groupby(['R', 'C'])['u_in'].transform('max') - df['u_in']
    df['mean_RC_u_in_rolling8_mean'] = list(df.groupby('breath_id')['mean_RC'].rolling(window=8).mean().fillna(0))
    df['max_RC_u_in_rolling8_max'] = list(df.groupby('breath_id')['max_RC'].rolling(window=8).mean().fillna(0))
    
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    df['cluster'] = df['cluster'].astype(str)
    df = pd.get_dummies(df)
    df = df.fillna(0)
    return df

train_ = add_features(train)
test_ = add_features(test)

## === cell 36
display(train_.head())
print(train_.shape)
display(test_)
print(test_.shape)

## === cell 37
def getDuplicateColumns(df):
  
    duplicateColumnNames = set()
      
    for x in range(df.shape[1]):
          
        col = df.iloc[:, x]
          
        for y in range(x + 1, df.shape[1]):
              
            otherCol = df.iloc[:, y]
              
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
                  
    return list(duplicateColumnNames)

## === cell 38
dc = getDuplicateColumns(train_)
dc

## === cell 39
targets = train_[['pressure']].to_numpy().reshape(-1, 80)
train_.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)
test_ = test_.drop(['id', 'breath_id'], axis=1)

train_.replace([np.inf, -np.inf], 0,inplace=True)
test_.replace([np.inf, -np.inf], 0,inplace=True)

## === cell 40
for col in [c for c in train_.columns if train_[c].dtype == "float64"]:
    train_[col] = train_[col].astype('float32')

## === cell 41
RS = RobustScaler()
train_ = RS.fit_transform(train_)
test_ = RS.transform(test_)

## === cell 42
train_ = train_.reshape(-1, 80, train_.shape[-1])
test_ = test_.reshape(-1, 80, train_.shape[-1])

## === cell 43
np.savez_compressed('gbvpp_reshaped_tt', a=train_, b=test_)

## === cell 45
set_seed(23)
    
BATCH_SIZE = 1024
NUM_FOLDS = 10
EPOCHS = 300

if DEBUG:
    EPOCHS = 3
    test_ = test_[:80*100]
    NUM_FOLDS = 2
    
def fit_lstm(train_,test_):
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)
    test_preds = []
    train_preds = train[['id', 'breath_id', 'pressure']]
    train_preds.loc[:, 'modified_breath_id'] = [i for i in range(len(train_)) for _ in range(80)]
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(train_, targets)):
        K.clear_session()
        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = train_[train_idx], train_[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]
        
        checkpoint_filepath = f"folds{fold}.hdf5"
        if TRAIN_MODEL:
            model = keras.models.Sequential([
                    keras.layers.Input(shape=train_.shape[-2:]),
                    keras.layers.Bidirectional(keras.layers.LSTM(1024, return_sequences=True)),
                    keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)),
                    keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
                    keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
                    keras.layers.Dense(128, activation='selu'),
                    keras.layers.Dense(1),
                ])
            model.compile(optimizer="adam", loss="mae")
            if fold==0:
                print(model.summary())

            lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=0)
            es = EarlyStopping(monitor="val_loss", patience=60, verbose=2, mode="min", restore_best_weights=True)
            sv = ModelCheckpoint(
                checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True,
                save_weights_only=False, mode='auto', save_freq='epoch',
                options=None)
            
            model.fit(X_train, y_train, validation_data=(X_valid, y_valid), 
                      epochs=EPOCHS,
                      batch_size=BATCH_SIZE, 
                      verbose=0,
                      callbacks=[lr, es, sv, TqdmCallback(verbose=0)])
        else:
            model = keras.models.load_model('../input/finetune-of-tensorflow-bidirectional-lstm/'+ checkpoint_filepath)
            
        
        test_preds.append(model.predict(test_, batch_size=BATCH_SIZE, verbose=2).ravel())
        train_preds.loc[train_preds.loc[:, 'modified_breath_id'].isin(test_idx), 'pressure'] = model.predict(X_valid, 
                                                                                                             batch_size=BATCH_SIZE, 
                                                                                                             verbose=2).ravel()
        
        del X_train, X_valid
        gc.collect()
    return test_preds, train_preds

## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2544222574.py in <cell line: 0>()
----> 1 set_seed(23)
      2 
      3 BATCH_SIZE = 1024
      4 NUM_FOLDS = 10
      5 EPOCHS = 300

NameError: name 'set_seed' is not defined

## === cell 46
gc.collect()

## === cell 47
%%time
set_seed(23)

test_preds, train_preds = fit_lstm(train_,test_)

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'set_seed' is not defined

## === cell 50
submission["pressure"] = sum(test_preds)/NUM_FOLDS
test["pred1"] = sum(test_preds)/NUM_FOLDS
submission.to_csv('submission_mean.csv', index=False)

submission["pressure"] = np.median(np.vstack(test_preds),axis=0)
test["pred1"] = np.median(np.vstack(test_preds),axis=0)
submission.to_csv('submission_median.csv', index=False)

submission["pressure"] =\
    np.round( (submission.pressure - PRESSURE_MIN)/PRESSURE_STEP ) * PRESSURE_STEP + PRESSURE_MIN
submission.pressure = np.clip(submission.pressure, PRESSURE_MIN, PRESSURE_MAX)
submission.to_csv('submission_median_round.csv', index=False)

test["pred1"] =\
    np.round( (test.pred1 - PRESSURE_MIN)/PRESSURE_STEP ) * PRESSURE_STEP + PRESSURE_MIN
test.pred1 = np.clip(test.pred1, PRESSURE_MIN, PRESSURE_MAX)


fea_names = ['id', 'pressure']

train_preds[fea_names].to_csv('train_p2.csv', index=False)
test[['id','pred1']].to_csv('test_p2.csv', index=False)

## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2813070821.py in <cell line: 0>()
      1 # ENSEMBLE FOLDS WITH MEAN
----> 2 submission["pressure"] = sum(test_preds)/NUM_FOLDS
      3 test["pred1"] = sum(test_preds)/NUM_FOLDS
      4 submission.to_csv('submission_mean.csv', index=False)
      5 

NameError: name 'test_preds' is not defined

## === cell 51
!wget 'https://www.kaggleusercontent.com/kf/76508760/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..8HziFSp37IaUybW4dC0GwA.tDQZdRFxTi-lIYrbq80NT81Da7azs9P-1Lfi4_Ov9F6D3K1CMk2mR4N1EaPMLEg93cKt0bQAz4RY-tZK8_JDcf_kkIz69p1-yAwrCPwde5RcH_VvBrRrRuPE6Y18pkeBXtI-i5gMfpO2sTWsfTWip2McBEYsPRlN7PLs70LL2KsLC_hH3tbMaodrzmLSIT4q3EOG-LAIdImfNKCjczNm6N43t68Ms9XhQj-rCbBM7DzdGp9GXCyhTupTjlLHnLKAdlIE-MS-dDA6DN2j21v5Oe1-9NZFvqo0U5RNG2OAEaIlY81z8ffljg46SqEaq_d1TJqsJoIme89AMRZVCesfhiEbZmbmMD6lFpan_37OJMFseV2t5GnOQsKnp4TtF6SXj0heK_8w2Zp0v3pTfr4EBIfzae-keS7hJty637u4lZhrSU-_RV3vYVTk8hJT3osg_3vteSTXFKSoet3GgIW4XeR96B_QEyCIVcnLOZSl_RqLfCigZaU3_lg9CetccqUbhO0A6FFV2r3rEXpzWeNs76hrc-emDr6KT03k1R5q2QGN0LxTS83FyedD8meLFz_D1C3xAxotze4O9ar33VE9gORneKFqxB6DOdo4nYZOYELz0oRdndSOPTRE_QlTHBFyUjp60-LFv2dF70HFGIVdduSHSWUNZf1edaUvdGWvtOZgbuKXYiAlRkGU7IfUq5es.KDunPjsTcqgAeeuiNfprLw/rwb%20125%20loops.csv'
!wget 'https://www.kaggleusercontent.com/kf/76430756/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..ziSs7kEObr-2cRzPIvwdBA.sN6Qqp-DXw0H0OUoMFiqSu90PtbqXc9Nb0zdcoPXJ4HYDBolPs6GUeAwHpRdbPZ-d9eUN1EsgDnBXsQ3U-00SUgd1-69087Sw3yJypMfrPVdpC8AyuKGfIl3S4ugVZitedGbsK5dMSeC5ViCESKvoXPVIeoxb-eaxVM9-QJg5zntQ9twodRpsOreTU94hgWYUlOY7rKWFVhO7a3Z8FsCvlORUDUaKPjBVyJ6i6fA4bvkYZu14UNiCXUk9mTtvtZDz5joLFFiX-2pfRcVSaRtw-15NcgaLRDUmwySmoFS_AcnN4Ep-Dk8eWKuAZ3iphUw__6ER53reSMjE7sDqigwG0MkEdSmjcoPICErb3ff30NdYY0OoAQtguUxkHxHyCF6zKlvAyW85qZPhooxauyAsbjMbFm4dnK4YX0ywfvrbcystZ5_iXSYXh7iybOlR0aOQzeKZujrW53dK7Ta_SbqzsH11-K6DPDkTAWTnz_f_OpGb-df6aoBktCIPAk7SoXhO1h3ETIyZfcgJE35zT6dQRHdeinESDyWewwTs4z8v8Wrf6oaUYs0ou6k7vpyeYmQ8tarIS0n4VgqZQGH2D-GJrN6z2qpMJtrgfzhr5zqILWDXv-Fv4S2eyBbmMQth2vvWHCU7EdfAWSLM2mcYpBP5n6QVaFqXK3B8uscAFD89HQ.ERJ9P2QryBLtg5ED6i36zA/kSUB.csv'

## === cell 52
!wget 'https://www.kaggleusercontent.com/kf/76520248/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..dwlcsegPuxZ-AZkCO2Tc-A._4GNAe2MEgCOPDezQ54QQA7SlmWOAWH7zNRxYdKtkfmoDczTdK2Bk6AE4YJgcE5LUMIi711WdRGWwoj9323PimCygg0gvotp5I6aREKcbqB1W37LGA3xJL9w4Vb2GLRvUtu3zqZ7pweQ70SWo4Et0ZPd-7-yuBxyBvm_uOMcmS0YKkmI7n_7BeHan-F0ujDNPiVekDovBKNcQZn3rBH6bOC2DFaazo_u8ljXAkdyZCBnPHH1gM_4hM9r_WA_-r79B74-TbRGKqT3I0v6cWkr1j2dqxWTRpQlhe16sKx40J1NXiXRdfd0TdzrFKt8NspmwjzuYlqNjEzUDsJUwRcTcd-qOaNPUAid1XYnjMJKLAAmxTt3kWFuIBmaNFy2TqbG9_-gyG455FIkWosT1BellL74LpOzTxbBYjthF9QhDqJKm8TazvAdVPyOFOiPDI-Iy9bXPZwvETc_NPlTHovCixtgmdxp1eo76jlpi8VKsZeSm1HCAh_v4SeNZgDt0s4Olpmj9KnY2gSdKp7NmjMe7vEqDf7dQVh0bpbrLXscJuLCT9tgcoQruqzX8f-AVDloaCTVbrv35HLLOUyz5SrdY16mEHFd2XJ3sq2shssvg2N5g-9nioN3brCNU7iyJ5_4CiwRmFl2nhtbCVU6_SSWXt3eO653Is-2mfo9slz_3U4uZCD_fQXouxEoYKEXZbE-.58LYIN9kkZtfu7SkkMJg5A/submission_median_round_mean.csv'

## === cell 53
!wget 'https://www.kaggleusercontent.com/kf/76519770/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..AUJIdMPPJK4JmPcyd9bDIA.yH7slyyMqYSL244KYc4xD9BR-74n4QQk6A8Z6krKf-YlP1Ex6kYgHJmhxsj0nmdaOYcFr3oRa9GzL7AyApWJ1v14-UAgjhDxu0r47e9ylKHQnfzov8u16irZWj1YtwoSRaPZYB7ejihABe3bfobkSKSKU6AaqEyPllXi0zOMpw1uh3Ov6mWtftjuOuax_BTnzqhgqB-AJ0X5cnHNdYtczCDODgae7YwKOKY3glJrd4ndoiU_kKwAC7ykBlQu1bfh3cV38SEa_ND8vZ4RDCIxKfJEjnv0C05Q4zBrXcpZ3obfGP7M7n-_hQY5dKXWXc0dSsRzqwoWNNOk-1tGnwW_-5VfFQyOaOQiMYu3e8LYrfd4v1gTD-nDrKIwk6s_aBgxCy7mR57J_GBi6fv36DjgXJwRyYVJv1lfqzFEwuDYP4m01BqBIZW-v5Lsq60th20VvqJEbiY2EN4hnxCzOh-wYFZl_Tsx96sJctQI3aKqlJOfRvHdtEiC45aV_7pqdDU6ywvRnsT4aEJGNj1t4XKvaxigAdi3xi93Tk1QUWKjNmZu5qrVrhxxEVELCTV2TDyY_oaGCsAJLtnC4l-lJ6QnqesNTxQK7lxHId0yt3KPTrDNiJQG3enyAo0B57b36W-h0KlrY-LcbeO8YPsVc7A7zvQGLrZ_MaGwJcoghHFfere6HsiwrtWLQ0IioftJJp31.IctGWj_5dE_VuxBLsh5vwQ/submission_median_round.csv'

## === cell 54
f8 = pd.read_csv('./submission_median_round.csv')
f3 =  pd.read_csv('./kSUB.csv')
f2 = pd.read_csv('./rwb 125 loops.csv')
f0 = pd.read_csv('./submission_median_round_mean.csv')

## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/922318237.py in <cell line: 0>()
----> 1 f8 = pd.read_csv('./submission_median_round.csv')
      2 f3 =  pd.read_csv('./kSUB.csv')
      3 f2 = pd.read_csv('./rwb 125 loops.csv')
      4 f0 = pd.read_csv('./submission_median_round_mean.csv')

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

FileNotFoundError: [Errno 2] No such file or directory: './submission_median_round.csv'

## === cell 55
k = f8.copy()

k['f8'] = f8.pressure 
k['f3'] = f3.pressure
k['f2'] = f2.pressure
k['f0'] = f0.pressure

k['pressure'] = np.median((k[['f8','f3','f2', 'f0']].values), axis=1)
k["pressure"] =\
    np.round( (k.pressure - PRESSURE_MIN)/PRESSURE_STEP ) * PRESSURE_STEP + PRESSURE_MIN
k.pressure = np.clip(k.pressure, PRESSURE_MIN, PRESSURE_MAX)

fea_names = ['id', 'pressure']

k[fea_names].to_csv('kOSUB.csv', index=False)

## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/354848408.py in <cell line: 0>()
----> 1 k = f8.copy()
      2 
      3 k['f8'] = f8.pressure
      4 k['f3'] = f3.pressure
      5 k['f2'] = f2.pressure

NameError: name 'f8' is not defined

## === cell 58
plt.title('Histogram of Test Pressures',size=14)
plt.hist(submission.sample(10_000).pressure.values, bins=100)
plt.show()
print('Max pressure =',submission.pressure.max(), 'Min pressure =',submission.pressure.min())

## === cell 59
all_pressure = np.sort( submission.pressure.unique() )
print('The differences between first 25 test pressures...')
all_pressure[1:26] - all_pressure[:25]
