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

0.1490965840003578

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 4
!wget 'https://www.kaggleusercontent.com/kf/77128042/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..aeS67vbZ55J63RQ1l_qcsw.Eyu5140cdK2X09Hm3ZyzzVKfeOfpa91UweC0S3arRQ3MjYKmtKc0wIqmszZMsOI3uc5dZq5eQiEBCi02K9W68qt_JjWy4EZgU1syzMy_Yg2Q32Rsh5wf7rGBy-n1OdWTrTZaYW-d3dkgZGWrqpY3VoXhVYSiDhVNrCho8suiHVnGW367ZKxitUo9KPt33Wwf6Op8Zv7yzR6BIwFXM8Zva-BulrxvLQ7IRtor4P3YL2XuWlyOquOtnnSV6PWGPRft5f7XxhT-EyNCOUxaPxACVz4Ghy71zpjE1JzytXYBILKBcZorvMnLBRe3meBALOuz7QBTRXYSxWVxCR2YpUZjg05EI99rcnj_ZXhLSzzpfhhIS9ApPmyuVzozts45ci6rgkwUB6TWuFfsXWkF6eRcLHWLUu1GaYShnGt7lsDJ7yOdrtp-thJXtClVcppnP-e2jbF95cielXe78P9C-iUPu_OEaQbpiwVeSmfEBsDw9z6zDzwfh5V5gzQHRZmC5laMsrvMs6FUs-vz51urnULRF1KHMmEyXflDaAHqa4J5ibziYatWJnyimPYauSU46OcrAGDfJ4356FGmJCzcFHgpsI814wINuA4PPZbH1jmVUz5SgS07W5ksa7WDKEQ-2mOmuC3lDsCCtcjbTqx1XOclrw.r_sIlZdlM2yUdJevyuau2A/submission_median_round.csv'

## === cell 5
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

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
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

## === cell 7
DEBUG = False
TRAIN_MODEL = True

## === cell 10
train = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
test = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')
submission = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')

test_bilstm = pd.read_csv('../input/gbvpp-v2-0144/test_p2.csv')
train_bilstm = pd.read_csv('../input/gbvpp-v2-0144/train_p2.csv')

train['bilstm_pred'] = train_bilstm['pressure']
test['bilstm_pred'] = test_bilstm['pred1']

del train_bilstm,test_bilstm
gc.collect()

if DEBUG:
    train = train[:80*1000]
    test = test[:80*100]
    submission = submission[:80*100]

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3037071948.py in <cell line: 0>()
      3 submission = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')
      4 
----> 5 test_bilstm = pd.read_csv('../input/gbvpp-v2-0144/test_p2.csv')
      6 train_bilstm = pd.read_csv('../input/gbvpp-v2-0144/train_p2.csv')
      7 

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/gbvpp-v2-0144/test_p2.csv'

## === cell 11
print('TRAIN\n')
display(train)
print('\n\nTEST\n')
display(test)

## === cell 13
print(f'Length of TRAIN dataset: {len(train)}')
print(f'Length of TEST dataset: {len(test)}')

print(f'Number of breaths in train dataset: {train["breath_id"].nunique()}')
print(f'Number of breaths in test dataset: {test["breath_id"].nunique()}')
print(f'The number of observations for each breath: {train["breath_id"].value_counts().reset_index()["breath_id"].unique()[0]}')

## === cell 14
display(test[test['breath_id']==0])

## === cell 16
train_gf = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')

plt.title('Histogram of Train Pressures',size=14)
plt.hist(train_gf.sample(100_000).pressure.values,bins=100)
plt.show()
print('Max pressure =',train_gf.pressure.max(), 'Min pressure =',train_gf.pressure.min())

## === cell 17
all_pressure = np.sort( train_gf.pressure.unique())
del train_gf
print('The first 25 unique pressures...')
PRESSURE_MIN = all_pressure[0].item()
PRESSURE_MAX = all_pressure[-1].item()
all_pressure[:25]

## === cell 18
print('The differences between first 25 pressures...')
PRESSURE_STEP = ( all_pressure[1] - all_pressure[0] ).item()
all_pressure[1:26] - all_pressure[:25]

## === cell 21
train["log_u_in"] = np.log1p(train.u_in)
test["log_u_in"] = np.log1p(test.u_in)

train["time_step_class"] = pd.qcut(train.time_step, q=80, labels=range(0,80))
test["time_step_class"] = pd.qcut(test.time_step, q=80, labels=range(0,80))

piv = train.pivot_table(index="breath_id", columns="time_step_class", values="log_u_in", fill_value=0, aggfunc="mean")
piv_test = test.pivot_table(index="breath_id", columns="time_step_class", values="log_u_in", fill_value=0, aggfunc="mean")

piv.head()

## === cell 22
pca = PCA(n_components=2, random_state=42)
pca.fit(piv)

plt.plot(pca.explained_variance_ratio_.cumsum())
plt.grid()
plt.xlabel("n_components")
plt.ylabel("explained_variance_ratio_")
plt.xticks([0, 1])
plt.show()

## === cell 23
train_pca = pca.transform(piv)
test_pca = pca.transform(piv_test)

train_pca = pd.DataFrame(train_pca, columns=["c"+str(c) for c in range(2)], index=piv.index)
test_pca = pd.DataFrame(test_pca, columns=["c"+str(c) for c in range(2)], index=piv_test.index)

train_pca.head()

## === cell 24
sns.scatterplot(data=train_pca, x="c0", y="c1")
plt.show()

## === cell 25
km = KMeans(n_clusters=4, 
            random_state=42,
            max_iter=200,
            init="k-means++", 
            tol=0.0001)
y_km = km.fit_predict(train_pca)
y_km_test = km.predict(test_pca)

## === cell 26
train_pca["cluster"] = y_km
test_pca["cluster"] = y_km_test

center = km.cluster_centers_

sns.scatterplot(data=train_pca, x="c0", y="c1", hue="cluster")
plt.plot(center[0, 0], center[0, 1], "bo", c="r")
plt.plot(center[1, 0], center[1, 1], "bo", c="r")


plt.show()

## === cell 27

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
    fig, ax = plt.subplots(4, 2, figsize=(15, 10))
    for c in range(4):
        for r_c in range(2):
            x = df.loc[df.cluster == c, "R" if r_c == 0 else "C" ]
            sns.countplot(x, ax=ax[c][r_c])
            ax[c][r_c].set_title(f"Cluster={c}")
    plt.tight_layout()
    
    
def find_cluster_transition(df, is_train=True):
    fig, ax = plt.subplots(4, 4, figsize=(15, 10))
    for c in range(4):
        x = df.loc[df.cluster == c]
        breath = x.breath_id.unique()
        for n in range(4):
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

## === cell 28
sns.countplot(train['cluster'])

## === cell 29
sns.countplot(test.cluster)

## === cell 30

'''
Some clusters show a single attribute.
It is more strongly reflected in clusters that do not exist in a straight line.
'''



## === cell 31
find_cluster_r_c(test)

## --- ERROR in cell 31, traceback:
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

/tmp/ipykernel_11/526442625.py in find_cluster_r_c(df)
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

## === cell 33
'''
From the time-series data distribution,
we confirmed a sharp rise in u_in following 1.0 second. 
After that, it decreases smoothly.
'''

find_cluster_transition(train)

## === cell 34
find_cluster_transition(test, False)

## === cell 35
train.drop(["time_step_class", "log_u_in"], axis=1, inplace=True)
test.drop(["time_step_class", "log_u_in"], axis=1, inplace=True)

## === cell 36
display(test)
print(test.shape)
display(train)
print(train.shape)

## === cell 37
dff = train[train['breath_id'].isin([51397,1,125749,3,560,3928])].copy()

## === cell 38
fig, ax = plt.subplots(6,1, figsize=(20, 25))
i = 0
for breath_id in dff['breath_id'].unique():
    pd_df= dff[dff['breath_id']==breath_id]
    pd_df.plot(x = 'time_step', y = 'u_in', c='b', title=f"Breath ID {breath_id}, R {pd_df['R'][:1].values}, C {pd_df['C'][:1].values}",ax= ax[i])
    pd_df.plot(x = 'time_step', y = 'pressure', c='purple', ax= ax[i])
    i+=1

## === cell 39
dx = train[train['breath_id']==49906]['time_step'].diff().mean()
y = list(train[train['breath_id']==49906]['u_in'])
area = np.trapz(y, dx=dx)
print("area =", area)

## === cell 40
%%time

remove = ['u_in_lag1']
def get_area(y):
    area = np.trapz(y, dx=dx)
    return area

def log_return(series):
    return np.log1p(series).diff()

def realized_volatility(series):
    return np.sqrt(np.sum(series**2))

def slope_expiratory(df):  
    time_at_u_out = df.iloc[0,1]
    u_in = df[df.iloc[:,0]>=time_at_u_out].iloc[:,2]
    u_in = np.where(u_in > 0, u_in, 10**-10)
    time_steps = df[df.iloc[:,0]>=time_at_u_out].iloc[:,0]
    if u_in.size==0 or time_steps.size==0:
        slope = np.nan 
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_steps, u_in)
    k = pd.DataFrame()
    k['x'] =[slope for i in range(df.shape[0])]
    return k['x']

def realized_volatility_inspiratory(df):
    time_at_u_out = df.iloc[0,1]
    series = df[df.iloc[:,0]<time_at_u_out].iloc[:,2]
    x = realized_volatility(series)
    k = pd.DataFrame()
    k['x'] =[x for i in range(df.shape[0])]
    return k['x']

def absolute_sum_of_changes(x):
    return np.sum(np.abs(np.diff(x)))

def std_inspiratory(df):
    time_at_u_out = df.iloc[0,1]
    series = df[df.iloc[:,0]<time_at_u_out].iloc[:,2]
    x = series.std()
    k = pd.DataFrame()
    k['x'] =[x for i in range(df.shape[0])]
    return k['x']

def range_ratio(x):
    mean_median_difference = np.abs(np.mean(x) - np.median(x))
    max_min_difference = np.max(x) - np.min(x)
    if max_min_difference == 0:
        return np.nan
    else:
        return mean_median_difference / max_min_difference
    
def range_ratio_inspiratory(df):
    time_at_u_out = df.iloc[0,1]
    series = df[df.iloc[:,0]<time_at_u_out].iloc[:,2]
    x = range_ratio(series)
    k = pd.DataFrame()
    k['x'] =[x for i in range(df.shape[0])]
    return k['x']

def variation_coefficient(x):
    mean = np.mean(x)
    if mean != 0:
        return np.std(x) / mean
    else:
        return np.nan
    
def variation_coefficient_inspiratory(df):
    time_at_u_out = df.iloc[0,1]
    series = df[df.iloc[:,0]<time_at_u_out].iloc[:,2]
    x = variation_coefficient(series)
    k = pd.DataFrame()
    k['x'] =[x for i in range(df.shape[0])]
    return k['x']

def variation_coefficient_expiratory(df):
    time_at_u_out = df.iloc[0,1]
    series = df[df.iloc[:,0]>=time_at_u_out].iloc[:,2]
    x = variation_coefficient(series)
    k = pd.DataFrame()
    k['x'] =[x for i in range(df.shape[0])]
    return k['x']
 
def add_features(dff):
    s = time.time()
    df = dff.copy()
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    
    true_areas = {}
    for bid, df_ in tqdm(df[['breath_id', 'time_step', 'u_in']].groupby('breath_id')):
        dx = df_[df_['breath_id']==bid]['time_step'].diff().mean()
        true_areas[bid] = df_['u_in'].expanding().apply(get_area).values
    
    print((time.time() - s)/60, ' minutes elapsed.')
    df['true_area'] = np.concatenate(list(true_areas.values()))
    
    df = df.reset_index().drop('index', axis=1)
    times_at_u_out = {}
    areas_at_u_out = {}
    for idx,row in tqdm(df.iterrows()):
        if (row['time_step'] < 0.95) or (row['time_step'] > 1.2):
            pass
        else:
            zero = df.iloc[idx-1]['u_out']
            one = row['u_out']
            bid = row['breath_id']
            if zero==0 and one==1:
                areas_at_u_out[bid] = row['true_area']
                times_at_u_out[bid] = row['time_step']
    
    print((time.time() - s)/60, ' minutes elapsed.')
    temp = pd.DataFrame()
    temp['breath_id'] = times_at_u_out.keys()
    temp['time_at_u_out'] = times_at_u_out.values()
    temp['area_at_u_out'] = areas_at_u_out.values()
    df = df.merge(temp, how='left', on='breath_id')
    
    df['true_area_abs_sum_changes'] = df.groupby("breath_id")["true_area"].transform(absolute_sum_of_changes)
    df['true_area_max'] = df.groupby('breath_id')['true_area'].transform("max")
    
    df["u_in_log"] = df.groupby("breath_id")["u_in"].apply(log_return)
    
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df["u_in_std"] = df.groupby("breath_id")["u_in"].transform("std")
    df["u_in_mean"] = df.groupby("breath_id")["u_in"].transform("mean")
    
    for col in [std_inspiratory, range_ratio_inspiratory, variation_coefficient_inspiratory,
                slope_expiratory, variation_coefficient_expiratory]:
        df['u_in_'+col.__name__] = np.concatenate(list(df.groupby("breath_id")[['time_step',
                                                                                'time_at_u_out', 
                                                                                'u_in']].apply(col).values))
    
    for col in [realized_volatility_inspiratory, range_ratio_inspiratory]:
        df['u_in_log_'+col.__name__] = np.concatenate(list(df.groupby("breath_id")[['time_step',
                                                                                'time_at_u_out', 
                                                                                'u_in_log']].apply(col).values))
    
    df['u_in_lag1'] =  df.groupby('breath_id')['u_in'].shift(1)
    df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    df['time_step_diff3'] = df.groupby('breath_id')['time_step'].diff(3)
    
    df['true_area_lag1'] =  df.groupby('breath_id')['true_area'].shift(1)
    df['true_area_lag2'] = df.groupby('breath_id')['true_area'].shift(2)
    df['true_area_lag_back2'] = df.groupby('breath_id')['true_area'].shift(-2)
    df['true_area_lag3'] = df.groupby('breath_id')['true_area'].shift(3)
    
    df['u_in_pct'] = df.groupby('breath_id')['u_in'].pct_change()
    print((time.time() - s)/60, ' minutes elapsed.')
    df['time_step_diff2'] = df.groupby('breath_id')['time_step'].diff(2)
    df['time_step_diff4'] = df.groupby('breath_id')['time_step'].diff(4)
    df['time_step_diff5'] = df.groupby('breath_id')['time_step'].diff(5)
    df['time_step_diff6'] = df.groupby('breath_id')['time_step'].diff(6)
    df['u_in_expanding10'] = list(df.groupby('breath_id')['u_in'].expanding(10).mean().fillna(0))    
    df['true_area_expanding15'] = list(df.groupby('breath_id')['true_area'].expanding(15).std().fillna(0)) 
    df['true_area_expanding10'] = list(df.groupby('breath_id')['true_area'].expanding(10).mean().fillna(0))
    print((time.time() - s)/60, ' minutes elapsed after expanding.')
    df['u_in_rolling3max'] = list(df.groupby('breath_id')['u_in'].rolling(window=3).max().fillna(0))
    df['u_in_rolling4max'] = list(df.groupby('breath_id')['u_in'].rolling(window=4).max().fillna(0))
    df['u_in_rolling5max'] = list(df.groupby('breath_id')['u_in'].rolling(window=5).max().fillna(0))
    df['u_in_rolling4'] = list(df.groupby('breath_id')['u_in'].rolling(window=4).mean().fillna(0))
    df['u_in_rolling5'] = list(df.groupby('breath_id')['u_in'].rolling(window=5).mean().fillna(0))
    df['u_out_rolling8'] = list(df.groupby('breath_id')['u_out'].rolling(window=8).mean().fillna(0))
    
    print((time.time() - s)/60, ' minutes elapsed after rolling.')
    print('Number of Features:', df.shape[1])
    df = df.fillna(0)
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']  
   

    
    df['bilstm_pred_lag1'] = df.groupby('breath_id')['bilstm_pred'].shift(1)
    df['bilstm_pred_lag2'] = df.groupby('breath_id')['bilstm_pred'].shift(2)
    df['bilstm_pred_lag_back1'] = df.groupby('breath_id')['bilstm_pred'].shift(-1)
    df['bilstm_pred_lag_back2'] = df.groupby('breath_id')['bilstm_pred'].shift(-2)
    
    
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    df['cluster'] = df['cluster'].astype(str)
    df = pd.get_dummies(df, drop_first=True)
    
    print('Number of Features:', df.shape[1])
    df.drop(remove,axis=1, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df = df.fillna(0)
    
    return df
train_ = add_features(train)
test_ = add_features(test)

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _reindex_for_setitem(value, index)
  12686     try:
> 12687         reindexed_value = value.reindex(index)._values
  12688     except ValueError as err:

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in reindex(self, index, axis, method, copy, level, fill_value, limit, tolerance)
   5152     ) -> Series:
-> 5153         return super().reindex(
   5154             index=index,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in reindex(self, labels, index, columns, axis, method, copy, level, fill_value, limit, tolerance)
   5609         # perform the reindex on the axes
-> 5610         return self._reindex_axes(
   5611             axes, level, limit, tolerance, method, fill_value, copy

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _reindex_axes(self, axes, level, limit, tolerance, method, fill_value, copy)
   5632             ax = self._get_axis(a)
-> 5633             new_index, indexer = ax.reindex(
   5634                 labels, level=level, limit=limit, tolerance=tolerance, method=method

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in reindex(self, target, method, level, limit, tolerance)
   4432 
-> 4433         target = self._wrap_reindex_result(target, indexer, preserve_names)
   4434         return target, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py in _wrap_reindex_result(self, target, indexer, preserve_names)
   2716                 try:
-> 2717                     target = MultiIndex.from_tuples(target)
   2718                 except TypeError:

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py in new_meth(self_or_cls, *args, **kwargs)
    221 
--> 222         return meth(self_or_cls, *args, **kwargs)
    223 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py in from_tuples(cls, tuples, sortorder, names)
    616 
--> 617             arrays = list(lib.tuples_to_object_array(tuples).T)
    618         elif isinstance(tuples, list):

lib.pyx in pandas._libs.lib.tuples_to_object_array()

ValueError: Buffer dtype mismatch, expected 'Python object' but got 'long'

The above exception was the direct cause of the following exception:

TypeError                                 Traceback (most recent call last)
<timed exec> in <module>

<timed exec> in add_features(dff)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5261             if not isinstance(value, Series):
   5262                 value = Series(value)
-> 5263             return _reindex_for_setitem(value, self.index)
   5264 
   5265         if is_list_like(value):

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _reindex_for_setitem(value, index)
  12692             raise err
  12693 
> 12694         raise TypeError(
  12695             "incompatible index of inserted column with frame index"
  12696         ) from err

TypeError: incompatible index of inserted column with frame index

## === cell 41
display(train_.head())
print(train_.shape)
display(test_)
print(test_.shape)

## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1325789722.py in <cell line: 0>()
----> 1 display(train_.head())
      2 print(train_.shape)
      3 display(test_)
      4 print(test_.shape)

NameError: name 'train_' is not defined

## === cell 45
def takeout_u_out(df):
    df.loc[df['time_step']>=df['time_at_u_out']] = np.nan
    return df

## === cell 46
train_ = pd.read_pickle('./train_v4.0.pkl')
test_ = pd.read_pickle('./test_v.0.pkl')

## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1911643249.py in <cell line: 0>()
----> 1 train_ = pd.read_pickle('./train_v4.0.pkl')
      2 test_ = pd.read_pickle('./test_v.0.pkl')

/usr/local/lib/python3.11/dist-packages/pandas/io/pickle.py in read_pickle(filepath_or_buffer, compression, storage_options)
    183     """
    184     excs_to_catch = (AttributeError, ImportError, ModuleNotFoundError, TypeError)
--> 185     with get_handle(
    186         filepath_or_buffer,
    187         "rb",

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    880         else:
    881             # Binary mode
--> 882             handle = open(handle, ioargs.mode)
    883         handles.append(handle)
    884 

FileNotFoundError: [Errno 2] No such file or directory: './train_v4.0.pkl'

## === cell 47
%%time
train_ = takeout_u_out(train_)
test_ = takeout_u_out(test_)

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'train_' is not defined

## === cell 48
def getDuplicateColumns(df):
  
    duplicateColumnNames = set()
      
    for x in range(df.shape[1]):
          
        col = df.iloc[:, x]
          
        for y in range(x + 1, df.shape[1]):
              
            otherCol = df.iloc[:, y]
              
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
                  
    return list(duplicateColumnNames)

## === cell 49
dc = getDuplicateColumns(train_)
dc

## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3700397094.py in <cell line: 0>()
----> 1 dc = getDuplicateColumns(train_)
      2 dc

NameError: name 'train_' is not defined

## === cell 50
targets = train_[['pressure']].to_numpy().reshape(-1, 80)
train_.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)
test_ = test_.drop(['id', 'breath_id'], axis=1)

train_.replace([np.inf, -np.inf], 0,inplace=True)
test_.replace([np.inf, -np.inf], 0,inplace=True)

## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3704379470.py in <cell line: 0>()
----> 1 targets = train_[['pressure']].to_numpy().reshape(-1, 80)
      2 train_.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)
      3 test_ = test_.drop(['id', 'breath_id'], axis=1)
      4 
      5 train_.replace([np.inf, -np.inf], 0,inplace=True)

NameError: name 'train_' is not defined

## === cell 51
for col in [c for c in train_.columns if train_[c].dtype == "float64"]:
    train_[col] = train_[col].astype('float32')

## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/558083098.py in <cell line: 0>()
----> 1 for col in [c for c in train_.columns if train_[c].dtype == "float64"]:
      2     train_[col] = train_[col].astype('float32')

NameError: name 'train_' is not defined

## === cell 52
gc.collect()

## === cell 53
RS = RobustScaler()
train_ = RS.fit_transform(train_)
test_ = RS.transform(test_)

## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1874192806.py in <cell line: 0>()
      1 RS = RobustScaler()
----> 2 train_ = RS.fit_transform(train_)
      3 test_ = RS.transform(test_)

NameError: name 'train_' is not defined

## === cell 54
train_ = train_.reshape(-1, 80, train_.shape[-1])
test_ = test_.reshape(-1, 80, train_.shape[-1])

## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4050034781.py in <cell line: 0>()
----> 1 train_ = train_.reshape(-1, 80, train_.shape[-1])
      2 test_ = test_.reshape(-1, 80, train_.shape[-1])

NameError: name 'train_' is not defined

## === cell 55
np.savez_compressed('gbvpp_reshaped_tt_v4.1', a=train_, b=test_, c=targets)

## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3223395649.py in <cell line: 0>()
----> 1 np.savez_compressed('gbvpp_reshaped_tt_v4.1', a=train_, b=test_, c=targets)

NameError: name 'train_' is not defined

## === cell 56
gc.collect()

## === cell 57
"""train = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
test = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')
submission = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')
    
loaded = np.load('../input/gbvp-prediction/gbvpp_reshaped_tt_v2.npz')
train_ = loaded['a']
test_ = loaded['b']

if DEBUG:
    train = train[:80*50]
    test = test[:80*10]
    train_ = train_[:50]
    test_ = test_[:10]
targets = train[['pressure']].to_numpy().reshape(-1, 80)"""

## === cell 58
"""set_seed(23)
    
BATCH_SIZE = 512
NUM_FOLDS = 10
EPOCHS = 300

if DEBUG:
    EPOCHS = 3
    NUM_FOLDS = 3
    
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
        log_filepath = f"fold{fold}_training.log"
        
        if TRAIN_MODEL:
            #with strategy.scope():
            model = keras.models.Sequential([
                        keras.layers.Input(shape=train_.shape[-2:]),
                        keras.layers.Bidirectional(keras.layers.LSTM(1024, return_sequences=True)),
                        SpatialDropout1D(0.05),
                        keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)),
                        keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
                        SpatialDropout1D(0.0125),
                        keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
                        keras.layers.Dense(128, activation='selu'),
                        keras.layers.Dense(1),
                    ])
            model.compile(optimizer="adam", loss="mae")
            if fold==0:
                print(model.summary())

            lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=0)
            es = EarlyStopping(monitor="val_loss", patience=30, verbose=1, mode="min", restore_best_weights=True)
            sv = ModelCheckpoint(
                checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                save_weights_only=False, mode='auto', save_freq='epoch',
                options=None)
            csv_logger = CSVLogger(log_filepath)
            
            model.fit(X_train, y_train, validation_data=(X_valid, y_valid), 
                      epochs=EPOCHS,
                      batch_size=BATCH_SIZE, 
                      verbose=1,
                      callbacks=[lr, es, sv, TqdmCallback(verbose=0),csv_logger])
        else:
            model = keras.models.load_model(''+ checkpoint_filepath)
            
        
        test_preds.append(model.predict(test_, batch_size=BATCH_SIZE, verbose=2).ravel())
        train_preds.loc[train_preds.loc[:, 'modified_breath_id'].isin(test_idx), 'pressure'] = model.predict(X_valid, 
                                                                                                             batch_size=BATCH_SIZE, 
                                                                                                             verbose=2).ravel()
        
        del X_train, X_valid, y_train, y_valid, model
        gc.collect()
    return test_preds, train_preds
    
test_preds, train_preds = fit_lstm(train_,test_)"""
