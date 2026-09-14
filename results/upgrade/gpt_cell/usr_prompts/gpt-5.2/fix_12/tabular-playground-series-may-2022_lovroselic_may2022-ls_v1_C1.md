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
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import time
from datetime import datetime

start_time = time.time()

get_ipython().run_line_magic("matplotlib", "inline")

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
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    plot_confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    accuracy_score,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from category_encoders import MEstimateEncoder
except ModuleNotFoundError:
    MEstimateEncoder = None

sns.set(
    style="white", context="notebook", palette="deep", rc={"figure.figsize": (10, 8)}
)
print("loaded ...")


## --- ERROR in cell 0, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mImportError[0m                               Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2019220243.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     22[0m [0;32mimport[0m [0mseaborn[0m [0;32mas[0m [0msns[0m[0;34m[0m[0;34m[0m[0m
[1;32m     23[0m [0;34m[0m[0m
[0;32m---> 24[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mmodels[0m [0;32mimport[0m [0mSequential[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     25[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mlayers[0m [0;32mimport[0m [0mDense[0m[0;34m,[0m [0mInput[0m[0;34m,[0m [0mDropout[0m[0;34m,[0m [0mBatchNormalization[0m[0;34m[0m[0;34m[0m[0m
[1;32m     26[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mcallbacks[0m [0;32mimport[0m [0mEarlyStopping[0m[0;34m,[0m [0mModelCheckpoint[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;31m# DO NOT EDIT. Generated by api_gen.sh[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mDTypePolicy[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mFloatDTypePolicy[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mFunction[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mInitializer[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/api/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mactivations[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mapplications[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mbackend[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/api/activations/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      5[0m """
[1;32m      6[0m [0;34m[0m[0m
[0;32m----> 7[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mdeserialize[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mget[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mserialize[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/__init__.py[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mactivations[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mapplications[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mbackend[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mconstraints[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mdatasets[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/activations/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;32mimport[0m [0mtypes[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;34m[0m[0m
[0;32m----> 3[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mcelu[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0melu[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mexponential[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/activations/activations.py[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mbackend[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mops[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapi_export[0m [0;32mimport[0m [0mkeras_export[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      8[0m [0;34m[0m[0m
[1;32m      9[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapi_export[0m [0;32mimport[0m [0mkeras_export[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mdtypes[0m [0;32mimport[0m [0mresult_type[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     11[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mkeras_tensor[0m [0;32mimport[0m [0mKerasTensor[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mkeras_tensor[0m [0;32mimport[0m [0many_symbolic_tensors[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m [0;32mimport[0m [0mbackend_utils[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mdtypes[0m [0;32mimport[0m [0mresult_type[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mvariables[0m [0;32mimport[0m [0mAutocastScope[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mvariables[0m [0;32mimport[0m [0mVariable[0m [0;32mas[0m [0mKerasVariable[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mvariables[0m [0;32mimport[0m [0mget_autocast_scope[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/dtypes.py[0m in [0;36m<module>[0;34m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapi_export[0m [0;32mimport[0m [0mkeras_export[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m [0;32mimport[0m [0mconfig[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mvariables[0m [0;32mimport[0m [0mstandardize_dtype[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m [0mBOOL_TYPES[0m [0;34m=[0m [0;34m([0m[0;34m"bool"[0m[0;34m,[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py[0m in [0;36m<module>[0;34m[0m
[1;32m      9[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mstateless_scope[0m [0;32mimport[0m [0mget_stateless_scope[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mstateless_scope[0m [0;32mimport[0m [0min_stateless_scope[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 11[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mmodule_utils[0m [0;32mimport[0m [0mtensorflow[0m [0;32mas[0m [0mtf[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     12[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mnaming[0m [0;32mimport[0m [0mauto_name[0m[0;34m[0m[0;34m[0m[0m
[1;32m     13[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/__init__.py[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0maudio_dataset_utils[0m [0;32mimport[0m [0maudio_dataset_from_directory[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mdataset_utils[0m [0;32mimport[0m [0msplit_dataset[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mfile_utils[0m [0;32mimport[0m [0mget_file[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mimage_dataset_utils[0m [0;32mimport[0m [0mimage_dataset_from_directory[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mimage_utils[0m [0;32mimport[0m [0marray_to_img[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/audio_dataset_utils.py[0m in [0;36m<module>[0;34m[0m
[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapi_export[0m [0;32mimport[0m [0mkeras_export[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mdataset_utils[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mmodule_utils[0m [0;32mimport[0m [0mtensorflow[0m [0;32mas[0m [0mtf[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mmodule_utils[0m [0;32mimport[0m [0mtensorflow_io[0m [0;32mas[0m [0mtfio[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/dataset_utils.py[0m in [0;36m<module>[0;34m[0m
[1;32m      7[0m [0;32mimport[0m [0mnumpy[0m [0;32mas[0m [0mnp[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m [0;34m[0m[0m
[0;32m----> 9[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mtree[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     10[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapi_export[0m [0;32mimport[0m [0mkeras_export[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mio_utils[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/tree/__init__.py[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m[0;34m.[0m[0mtree_api[0m [0;32mimport[0m [0massert_same_paths[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m[0;34m.[0m[0mtree_api[0m [0;32mimport[0m [0massert_same_structure[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m[0;34m.[0m[0mtree_api[0m [0;32mimport[0m [0mflatten[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m[0;34m.[0m[0mtree_api[0m [0;32mimport[0m [0mflatten_with_path[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m[0;34m.[0m[0mtree_api[0m [0;32mimport[0m [0mis_nested[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/tree/tree_api.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m [0;32mif[0m [0moptree[0m[0;34m.[0m[0mavailable[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 8[0;31m     [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m [0;32mimport[0m [0moptree_impl[0m [0;32mas[0m [0mtree_impl[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32melif[0m [0mdmtree[0m[0;34m.[0m[0mavailable[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m     [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m [0;32mimport[0m [0mdmtree_impl[0m [0;32mas[0m [0mtree_impl[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/tree/optree_impl.py[0m in [0;36m<module>[0;34m[0m
[1;32m     11[0m [0;31m# Register backend-specific node classes[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m [0;32mif[0m [0mbackend[0m[0;34m([0m[0;34m)[0m [0;34m==[0m [0;34m"tensorflow"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 13[0;31m     [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mtrackable[0m[0;34m.[0m[0mdata_structures[0m [0;32mimport[0m [0mListWrapper[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     14[0m     [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mtrackable[0m[0;34m.[0m[0mdata_structures[0m [0;32mimport[0m [0m_DictWrapper[0m[0;34m[0m[0;34m[0m[0m
[1;32m     15[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     47[0m [0m_tf2[0m[0;34m.[0m[0menable[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     48[0m [0;34m[0m[0m
[0;32m---> 49[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m [0;32mimport[0m [0m__internal__[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     50[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m [0;32mimport[0m [0m__operators__[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m [0;32mimport[0m [0maudio[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;32mimport[0m [0msys[0m [0;32mas[0m [0m_sys[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mautograph[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mdecorator[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mdispatch[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;32mimport[0m [0msys[0m [0;32mas[0m [0m_sys[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mag_ctx[0m [0;32mimport[0m [0mcontrol_status_ctx[0m [0;31m# line: 34[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mimpl[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mtf_convert[0m [0;31m# line: 493[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py[0m in [0;36m<module>[0;34m[0m
[1;32m     19[0m [0;32mimport[0m [0mthreading[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m [0;34m[0m[0m
[0;32m---> 21[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mag_logging[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     22[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mutil[0m[0;34m.[0m[0mtf_export[0m [0;32mimport[0m [0mtf_export[0m[0;34m[0m[0;34m[0m[0m
[1;32m     23[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     15[0m [0;34m"""Utility module that contains APIs usable in the generated code."""[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m [0;34m[0m[0m
[0;32m---> 17[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mcontext_managers[0m [0;32mimport[0m [0mcontrol_dependency_on_returns[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     18[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mmisc[0m [0;32mimport[0m [0malias_tensors[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mtensor_list[0m [0;32mimport[0m [0mdynamic_list_append[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py[0m in [0;36m<module>[0;34m[0m
[1;32m     17[0m [0;32mimport[0m [0mcontextlib[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m [0;34m[0m[0m
[0;32m---> 19[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mops[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     20[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mops[0m [0;32mimport[0m [0mtensor_array_ops[0m[0;34m[0m[0;34m[0m[0m
[1;32m     21[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py[0m in [0;36m<module>[0;34m[0m
[1;32m     31[0m [0;34m[0m[0m
[1;32m     32[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mmessage[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 33[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mattr_value_pb2[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     34[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mfull_type_pb2[0m[0;34m[0m[0;34m[0m[0m
[1;32m     35[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mfunction_pb2[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py[0m in [0;36m<module>[0;34m[0m
[1;32m      3[0m [0;31m# source: tensorflow/core/framework/attr_value.proto[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m"""Generated protocol buffer code."""[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0minternal[0m [0;32mimport[0m [0mbuilder[0m [0;32mas[0m [0m_builder[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mdescriptor[0m [0;32mas[0m [0m_descriptor[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mdescriptor_pool[0m [0;32mas[0m [0m_descriptor_pool[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py[0m in [0;36m<module>[0;34m[0m
[1;32m     16[0m [0;34m[0m[0m
[1;32m     17[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0minternal[0m [0;32mimport[0m [0menum_type_wrapper[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 18[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0minternal[0m [0;32mimport[0m [0mpython_message[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     19[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mmessage[0m [0;32mas[0m [0m_message[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mreflection[0m [0;32mas[0m [0m_reflection[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py[0m in [0;36m<module>[0;34m[0m
[1;32m     36[0m [0;32mimport[0m [0mweakref[0m[0;34m[0m[0;34m[0m[0m
[1;32m     37[0m [0;34m[0m[0m
[0;32m---> 38[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mdescriptor[0m [0;32mas[0m [0mdescriptor_mod[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     39[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mmessage[0m [0;32mas[0m [0mmessage_mod[0m[0;34m[0m[0;34m[0m[0m
[1;32m     40[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mtext_format[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py[0m in [0;36m<module>[0;34m[0m
[1;32m     27[0m   [0;31m# TODO: Remove this import after fix api_implementation[0m[0;34m[0m[0;34m[0m[0m
[1;32m     28[0m   [0;32mif[0m [0m_message[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 29[0;31m     [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0mpyext[0m [0;32mimport[0m [0m_message[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     30[0m   [0m_USE_C_DESCRIPTORS[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m     31[0m [0;34m[0m[0m

[0;31mImportError[0m: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 1
def set_seed(sd):
    seed(sd)
    np.random.seed(sd)
    tf.random.set_seed(sd)
    os.environ['PYTHONHASHSEED'] = str(sd)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
RandomSeed = 13
set_seed(RandomSeed)
