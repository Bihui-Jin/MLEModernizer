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

3.8

# 2. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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
tf_keras==2.18.0

# 3. Data file paths

```
/
    kaggle/
        data/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 4. Code solution

## === cell 0
!pip install -q efficientnet


## === cell 1
import os
import sys
import subprocess

try:
    import google.protobuf as _pb

    _pb_ver = getattr(_pb, "__version__", "0")
    if _pb_ver and int(_pb_ver.split(".")[0]) >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        import importlib

        importlib.invalidate_caches()
        for _m in list(sys.modules):
            if _m.startswith("google.protobuf"):
                del sys.modules[_m]
except Exception:
    pass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow.keras.layers as l

from keras.regularizers import l2
from keras.optimizers import Adam

import tensorflow.keras.applications.efficientnet as efn

from sklearn.model_selection import train_test_split


## === cell 2
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

    print(tpu.master())
    print(tpu_strategy.num_replicas_in_sync)
except Exception as e:
    tpu = None
    tpu_strategy = tf.distribute.get_strategy()
    print("TPU not available; using default strategy:", type(tpu_strategy).__name__)
    print("num_replicas_in_sync:", tpu_strategy.num_replicas_in_sync)


## === cell 3
AUTO = tf.data.experimental.AUTOTUNE
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False

try:
    from kaggle_datasets import KaggleDatasets  # type: ignore

    gcs_path = KaggleDatasets().get_gcs_path()
except Exception:
    gcs_path = None


## === cell 4
sample = pd.read_csv("/kaggle/input/alaska2-image-steganalysis/sample_submission.csv")
BATCH_SIZE = 32 * tpu_strategy.num_replicas_in_sync # batch size in tpu
EPOCHS = 1


dir_name = ['Test', 'JUNIWARD', 'JMiPOD', 'Cover', 'UERD']
df = pd.DataFrame({})
lists = []
cate = []

for dir_ in dir_name:
    list_ = os.listdir("/kaggle/input/alaska2-image-steganalysis/"+dir_+"/")
    lists = lists+list_
    cate_ = np.tile(dir_,len(list_))
    cate = np.concatenate([cate,cate_])
    
df["cate"] = cate
df["name"] = lists


## === cell 5
df["path"] = [str(os.path.join(gcs_path,cate,name)) for cate, name in zip(df["cate"], df["name"])]

def cate_label(x):
    if x["cate"] == "Cover":
        res = 0
    else:
        res = 1
    return res

Test_df = df.query("cate=='Test'").sort_values(by="name")
Train_df = df.query("cate!='Test'")

Train_df["labled"] = df.apply(cate_label, axis=1)


## --- ERROR in cell 5, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3446625581.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m#add path to df[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mdf[0m[0;34m[[0m[0;34m"path"[0m[0;34m][0m [0;34m=[0m [0;34m[[0m[0mstr[0m[0;34m([0m[0mos[0m[0;34m.[0m[0mpath[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mgcs_path[0m[0;34m,[0m[0mcate[0m[0;34m,[0m[0mname[0m[0;34m)[0m[0;34m)[0m [0;32mfor[0m [0mcate[0m[0;34m,[0m [0mname[0m [0;32min[0m [0mzip[0m[0;34m([0m[0mdf[0m[0;34m[[0m[0;34m"cate"[0m[0;34m][0m[0;34m,[0m [0mdf[0m[0;34m[[0m[0;34m"name"[0m[0;34m][0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0;34m[0m[0m
[1;32m      4[0m [0;31m#Labeling func[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mdef[0m [0mcate_label[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3446625581.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m      1[0m [0;31m#add path to df[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mdf[0m[0;34m[[0m[0;34m"path"[0m[0;34m][0m [0;34m=[0m [0;34m[[0m[0mstr[0m[0;34m([0m[0mos[0m[0;34m.[0m[0mpath[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mgcs_path[0m[0;34m,[0m[0mcate[0m[0;34m,[0m[0mname[0m[0;34m)[0m[0;34m)[0m [0;32mfor[0m [0mcate[0m[0;34m,[0m [0mname[0m [0;32min[0m [0mzip[0m[0;34m([0m[0mdf[0m[0;34m[[0m[0;34m"cate"[0m[0;34m][0m[0;34m,[0m [0mdf[0m[0;34m[[0m[0;34m"name"[0m[0;34m][0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0;34m[0m[0m
[1;32m      4[0m [0;31m#Labeling func[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mdef[0m [0mcate_label[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/posixpath.py[0m in [0;36mjoin[0;34m(a, *p)[0m

[0;31mTypeError[0m: expected str, bytes or os.PathLike object, not NoneType

## === cell 6
print("Training set: \n",Train_df["cate"].value_counts())

print('\n', Train_df["path"].head())
print('\n',Train_df["labled"].head())
print('\n',Test_df["path"].head())
