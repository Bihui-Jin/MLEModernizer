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
tqdm==4.67.1

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        input/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        working/
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
```

-> data/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/stanford-covid-vaccine/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> data/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> input/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import sys
import subprocess

try:
    import google.protobuf as _pb  # noqa: F401
    from packaging.version import Version
    import google.protobuf

    _pb_ver = Version(google.protobuf.__version__)
    if _pb_ver.major >= 6:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<6"]
        )
        import importlib

        importlib.invalidate_caches()
        for _m in list(sys.modules):
            if _m.startswith("google.protobuf"):
                del sys.modules[_m]
except Exception:
    pass

import json
import tensorflow as tf
from matplotlib import pyplot as plt


## === cell 2
os.chdir('/kaggle/')
os.getcwd()


## === cell 3
train_data = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines = True)
test_data = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines = True)
submission_format = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv', encoding = 'utf-8-sig')


## === cell 4
train_data.head()


## === cell 5
train_data.shape


## === cell 6
train_data.groupby(['SN_filter']).size()


## === cell 7
test_data.head()


## === cell 8
test_data.shape


## === cell 9
submission_format.head()


## === cell 10
print(train_data.shape)
print(test_data.shape)
print(submission_format.shape)


## === cell 11
print('Training data:\n',train_data['seq_scored'].value_counts())
print('Test data:\n',test_data['seq_scored'].value_counts())
len(train_data['reactivity'].iloc[0])


## === cell 12
len(train_data['sequence'].iloc[0])


## === cell 13
flag = False
for i in range(0,len(train_data)):
    if(([x<0 for x in train_data['reactivity_error'].iloc[i]].count(True) > 0) |
       ([x<0 for x in train_data['deg_error_Mg_pH10'].iloc[i]].count(True) > 0) |
       ([x<0 for x in train_data['deg_error_pH10'].iloc[i]].count(True) > 0) |
       ([x<0 for x in train_data['deg_error_Mg_50C'].iloc[i]].count(True) > 0) |
       ([x<0 for x in train_data['deg_error_50C'].iloc[i]].count(True) > 0)):
        flag = True
print(flag)


## === cell 14
min_reactivity_value = min(train_data['reactivity'].iloc[0])
min_deg_Mg_pH10_value = min(train_data['deg_Mg_pH10'].iloc[0])
min_deg_pH10_value = min(train_data['deg_pH10'].iloc[0])
min_deg_Mg_50C_value = min(train_data['deg_Mg_50C'].iloc[0])
min_deg_Mg_50C_value = min(train_data['deg_50C'].iloc[0])

for i in range(0,len(train_data)):
    if(min(train_data['reactivity'].iloc[i]) < min_reactivity_value):
        min_reactivity_value = min(train_data['reactivity'].iloc[i])   

    if(min(train_data['deg_Mg_pH10'].iloc[i]) < min_deg_Mg_pH10_value):
        min_deg_Mg_pH10_value = min(train_data['deg_Mg_pH10'].iloc[i])

    if(min(train_data['deg_pH10'].iloc[i]) < min_deg_pH10_value):
        min_deg_pH10_value = min(train_data['deg_pH10'].iloc[i])

    if(min(train_data['deg_Mg_50C'].iloc[i]) < min_deg_Mg_50C_value):
        min_deg_Mg_50C_value = min(train_data['deg_Mg_50C'].iloc[i])

    if(min(train_data['deg_50C'].iloc[i]) < min_deg_Mg_50C_value):
        min_deg_Mg_50C_value = min(train_data['deg_50C'].iloc[i])
        
print(min_reactivity_value, min_deg_Mg_pH10_value, min_deg_pH10_value, min_deg_Mg_50C_value, min_deg_Mg_50C_value)


## === cell 15
train_data.columns


## === cell 16
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


## === cell 17
token2int


## === cell 18
def read_bpps_sum(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").sum(axis=1))
    return bpps_arr

def read_bpps_max(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").max(axis=1))
    return bpps_arr

def read_bpps_nb(df):
    bpps_nb_mean = 0.077522
    bpps_nb_std = 0.08914
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps = np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy")
        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
        bpps_arr.append(bpps_nb)
    return bpps_arr 

os.chdir("/kaggle/working/")
train_data['bpps_sum'] = read_bpps_sum(train_data)
test_data['bpps_sum'] = read_bpps_sum(test_data)
train_data['bpps_max'] = read_bpps_max(train_data)
test_data['bpps_max'] = read_bpps_max(test_data)
train_data['bpps_nb'] = read_bpps_nb(train_data)
test_data['bpps_nb'] = read_bpps_nb(test_data)

train_data.head()


## --- ERROR in cell 18, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3180844256.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     24[0m [0;34m[0m[0m
[1;32m     25[0m [0mos[0m[0;34m.[0m[0mchdir[0m[0;34m([0m[0;34m"/kaggle/working/"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 26[0;31m [0mtrain_data[0m[0;34m[[0m[0;34m'bpps_sum'[0m[0;34m][0m [0;34m=[0m [0mread_bpps_sum[0m[0;34m([0m[0mtrain_data[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     27[0m [0mtest_data[0m[0;34m[[0m[0;34m'bpps_sum'[0m[0;34m][0m [0;34m=[0m [0mread_bpps_sum[0m[0;34m([0m[0mtest_data[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     28[0m [0mtrain_data[0m[0;34m[[0m[0;34m'bpps_max'[0m[0;34m][0m [0;34m=[0m [0mread_bpps_max[0m[0;34m([0m[0mtrain_data[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3180844256.py[0m in [0;36mread_bpps_sum[0;34m(df)[0m
[1;32m      2[0m     [0mbpps_arr[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m     [0;32mfor[0m [0mmol_id[0m [0;32min[0m [0mdf[0m[0;34m.[0m[0mid[0m[0;34m.[0m[0mto_list[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m         [0mbpps_arr[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mload[0m[0;34m([0m[0;34mf"../input/stanford-covid-vaccine/bpps/{mol_id}.npy"[0m[0;34m)[0m[0;34m.[0m[0msum[0m[0;34m([0m[0maxis[0m[0;34m=[0m[0;36m1[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m     [0;32mreturn[0m [0mbpps_arr[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py[0m in [0;36mload[0;34m(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)[0m
[1;32m    425[0m             [0mown_fid[0m [0;34m=[0m [0;32mFalse[0m[0;34m[0m[0;34m[0m[0m
[1;32m    426[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 427[0;31m             [0mfid[0m [0;34m=[0m [0mstack[0m[0;34m.[0m[0menter_context[0m[0;34m([0m[0mopen[0m[0;34m([0m[0mos_fspath[0m[0;34m([0m[0mfile[0m[0;34m)[0m[0;34m,[0m [0;34m"rb"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    428[0m             [0mown_fid[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m    429[0m [0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '../input/stanford-covid-vaccine/bpps/id_001f94081.npy'

## === cell 19
def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    base_fea = np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )
    
    bpps_sum_fea = np.array(df['bpps_sum'].to_list())[:,:,np.newaxis]
    bpps_max_fea = np.array(df['bpps_max'].to_list())[:,:,np.newaxis]
    bpps_nb_fea = np.array(df['bpps_nb'].to_list())[:,:,np.newaxis]
    
    return np.concatenate([base_fea,bpps_sum_fea,bpps_max_fea,bpps_nb_fea], 2)
