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

No external packages required in the script and installed.

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
Diagnosis: The crash happens during imports in cell 0 with `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`, which is a known incompatibility between TensorFlow/Keras and newer `protobuf` versions. The error is triggered when importing `keras`/`tensorflow` before forcing protobuf to use the pure-Python implementation. Setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and its version) **before** importing TensorFlow/Keras avoids the problematic C++ implementation path that calls `GetPrototype`.  
Patch summary: Add the protobuf environment-variable workaround at the very top of cell 0 (before any TF/Keras imports) to prevent the protobuf API mismatch crash, leaving all model/training logic unchanged.  
Updated cells: Only cell 0 is modified.  
Compatibility notes for cell k+1: No variables, functions, or interfaces used by cell 1 change; `tf` is still imported and available for `tf.random.set_seed`.  
Assumptions: TensorFlow/Keras are present but bundled against an older protobuf API than the runtime `protobuf` package; using the Python protobuf implementation is acceptable and deterministic for this notebook.

```python
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import time
from datetime import datetime

start_time = time.time()

%matplotlib inline

import warnings
import numpy as np 
from numpy.random import seed
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import metrics
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix, precision_score,recall_score, f1_score, classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from category_encoders import MEstimateEncoder

sns.set(style='white', context='notebook', palette='deep', rc={'figure.figsize':(10,8)})
print("loaded ...")
```

## --- ERROR in cell 0, traceback:
[0;36m  File [0;32m"/tmp/ipykernel_11/2913922379.py"[0;36m, line [0;32m1[0m
[0;31m    Diagnosis: The crash happens during imports in cell 0 with `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`, which is a known incompatibility between TensorFlow/Keras and newer `protobuf` versions. The error is triggered when importing `keras`/`tensorflow` before forcing protobuf to use the pure-Python implementation. Setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and its version) **before** importing TensorFlow/Keras avoids the problematic C++ implementation path that calls `GetPrototype`.[0m
[0m                   ^[0m
[0;31mSyntaxError[0m[0;31m:[0m invalid syntax


## === cell 1
def set_seed(sd):
    seed(sd)
    np.random.seed(sd)
    tf.random.set_seed(sd)
    os.environ['PYTHONHASHSEED'] = str(sd)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
RandomSeed = 13
set_seed(RandomSeed)
