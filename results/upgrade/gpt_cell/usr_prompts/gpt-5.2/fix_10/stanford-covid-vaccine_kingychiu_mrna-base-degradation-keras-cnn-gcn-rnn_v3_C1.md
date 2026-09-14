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

3.9

# 2. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
import sys
import subprocess

try:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )
except Exception:
    pass

import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

from sklearn.model_selection import train_test_split

try:
    from keras.utils.vis_utils import plot_model
except Exception:
    plot_model = None

import tensorflow.keras.layers as L
import keras.backend as K
import tensorflow as tf


## === cell 1
!pip install spektral -q


## === cell 2
try:
    from spektral.layers import GraphConv  # older Spektral
except ImportError:
    from spektral.layers import GCNConv as GraphConv  # newer Spektral


## === cell 3
train_json_path = "/kaggle/input/stanford-covid-vaccine/train.json"
test_json_path = "/kaggle/input/stanford-covid-vaccine/test.json"
sample_sub_path = "/kaggle/input/stanford-covid-vaccine/sample_submission.csv"

output_path = "./"
bpps_path = "/kaggle/input/stanford-covid-vaccine/bpps"

train_df = pd.read_json(train_json_path, lines=True)
test_df = pd.read_json(test_json_path, lines=True)

public_df = test_df.query("seq_length == 107").copy()
private_df = test_df.query("seq_length == 130").copy()


## === cell 4
train_df.shape


## === cell 5
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
train_df[pred_cols].head()


## === cell 6
train_y = np.array(train_df[pred_cols].values.tolist()).transpose((0, 2, 1))
train_y.shape


## === cell 7
train_df[["id", "sequence", "structure", "predicted_loop_type"]].head()


## === cell 8
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
sequence_token2int = {x: i for i, x in enumerate("AGUC")}
structure_token2int = {
    ".": 0,
    "(": 1,
    ")": 2,
}
loop_token2int = {x: i for i, x in enumerate("SMIBHEX")}
token2int_map = {
    "sequence": sequence_token2int,
    "structure": structure_token2int,
    "predicted_loop_type": loop_token2int,
}
sequence_columns = ["sequence", "structure", "predicted_loop_type"]


def to_seq(df):
    if df is None or len(df) == 0:
        return np.zeros((0, 0, len(sequence_columns)), dtype=np.int32)

    cols_tok = [
        np.array(
            df[col].apply(lambda s: [token2int[x] for x in s]).values.tolist(),
            dtype=np.int32,
        )
        for col in sequence_columns
    ]  # each is (n, L)
    arr = np.stack(cols_tok, axis=1)  # (n, 3, L)
    return np.transpose(arr, (0, 2, 1))  # (n, L, 3)


train = to_seq(train_df)
public = to_seq(public_df)
private = to_seq(private_df)

train.shape, public.shape, private.shape


## === cell 9
def to_one_hot(df):
    if df is None or len(df) == 0:
        return np.zeros((0, 0, 14), dtype=np.float32)

    cols_idx = []
    for col in sequence_columns:
        idx = np.stack(
            df[col]
            .apply(lambda seq: [token2int_map[col][x] for x in seq])
            .values.tolist(),
            axis=0,
        ).astype(
            np.int32
        )  # (n, L)
        cols_idx.append(idx)

    temp = np.stack(cols_idx, axis=2)  # (n, L, 3)

    ohe_1 = tf.keras.utils.to_categorical(temp[:, :, 0], 4)
    ohe_2 = tf.keras.utils.to_categorical(temp[:, :, 1], 3)
    ohe_3 = tf.keras.utils.to_categorical(temp[:, :, 2], 7)
    return np.concatenate([ohe_1, ohe_2, ohe_3], axis=2)


train_ohe = to_one_hot(train_df)
public_ohe = to_one_hot(public_df)
private_ohe = to_one_hot(private_df)

train_ohe.shape, public_ohe.shape, private_ohe.shape


## === cell 10
def get_adjacency_matrix(inps):
    As = []
    for row in range(0, inps.shape[0]):
        A = np.zeros((inps.shape[1], inps.shape[1]))
        stack = []
        opened_so_far = []

        for seqpos in range(0, inps.shape[1]):
            if inps[row, seqpos, 1] == 0:
                stack.append(seqpos)
                opened_so_far.append(seqpos)
            elif inps[row, seqpos, 1] == 1:
                openpos = stack.pop()
                A[openpos, seqpos] = 1
                A[seqpos, openpos] = 1
        As.append(A)
    return np.array(As)

train_adj = get_adjacency_matrix(train)
public_adj = get_adjacency_matrix(public)
private_adj = get_adjacency_matrix(private)

train_adj.shape, public_adj.shape, private_adj.shape


## === cell 11
train_adj.mean(), public_adj.mean(), private_adj.mean()


## === cell 12
def get_bpps(mRNA_ids, length_map=None):
    candidate_dirs = [
        bpps_path,
        "/kaggle/input/stanford-covid-vaccine/stanford-covid-vaccine/bpps",
        "/kaggle/data/stanford-covid-vaccine/bpps",
        "/kaggle/data/stanford-covid-vaccine/stanford-covid-vaccine/bpps",
    ]

    if length_map is None:
        length_map = {}

    bpps = []
    for mRNA_id in tqdm(mRNA_ids):
        arr = None
        for d in candidate_dirs:
            fp = os.path.join(d, f"{mRNA_id}.npy")
            if os.path.exists(fp):
                arr = np.load(fp)
                break

        if arr is None:
            L = int(length_map.get(mRNA_id, 0))
            arr = np.zeros((L, L), dtype=np.float32)

        bpps.append(arr)

    return np.array(bpps)


train_len_map = dict(zip(train_df["id"].values, train_df["seq_length"].values))
public_len_map = dict(zip(public_df["id"].values, public_df["seq_length"].values))
private_len_map = dict(zip(private_df["id"].values, private_df["seq_length"].values))

train_bpps = get_bpps(train_df.id.values, length_map=train_len_map)
public_bpps = get_bpps(public_df.id.values, length_map=public_len_map)
private_bpps = get_bpps(private_df.id.values, length_map=private_len_map)

train_bpps.shape, public_bpps.shape, private_bpps.shape


## === cell 13
train_bpps.mean(), public_bpps.mean(), private_bpps.mean() 


## === cell 14
train_bpps_stats = [train_bpps.mean(axis=2), train_bpps.max(axis=2)]
public_bpps_stats = [public_bpps.mean(axis=2), public_bpps.max(axis=2)]
private_bpps_stats = [private_bpps.mean(axis=2), private_bpps.max(axis=2)]


## --- ERROR in cell 14, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAxisError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2701602045.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0mtrain_bpps_stats[0m [0;34m=[0m [0;34m[[0m[0mtrain_bpps[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0maxis[0m[0;34m=[0m[0;36m2[0m[0;34m)[0m[0;34m,[0m [0mtrain_bpps[0m[0;34m.[0m[0mmax[0m[0;34m([0m[0maxis[0m[0;34m=[0m[0;36m2[0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0mpublic_bpps_stats[0m [0;34m=[0m [0;34m[[0m[0mpublic_bpps[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0maxis[0m[0;34m=[0m[0;36m2[0m[0;34m)[0m[0;34m,[0m [0mpublic_bpps[0m[0;34m.[0m[0mmax[0m[0;34m([0m[0maxis[0m[0;34m=[0m[0;36m2[0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m [0mprivate_bpps_stats[0m [0;34m=[0m [0;34m[[0m[0mprivate_bpps[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0maxis[0m[0;34m=[0m[0;36m2[0m[0;34m)[0m[0;34m,[0m [0mprivate_bpps[0m[0;34m.[0m[0mmax[0m[0;34m([0m[0maxis[0m[0;34m=[0m[0;36m2[0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/numpy/core/_methods.py[0m in [0;36m_mean[0;34m(a, axis, dtype, out, keepdims, where)[0m
[1;32m    104[0m     [0mis_float16_result[0m [0;34m=[0m [0;32mFalse[0m[0;34m[0m[0;34m[0m[0m
[1;32m    105[0m [0;34m[0m[0m
[0;32m--> 106[0;31m     [0mrcount[0m [0;34m=[0m [0m_count_reduce_items[0m[0;34m([0m[0marr[0m[0;34m,[0m [0maxis[0m[0;34m,[0m [0mkeepdims[0m[0;34m=[0m[0mkeepdims[0m[0;34m,[0m [0mwhere[0m[0;34m=[0m[0mwhere[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    107[0m     [0;32mif[0m [0mrcount[0m [0;34m==[0m [0;36m0[0m [0;32mif[0m [0mwhere[0m [0;32mis[0m [0;32mTrue[0m [0;32melse[0m [0mumr_any[0m[0;34m([0m[0mrcount[0m [0;34m==[0m [0;36m0[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    108[0m         [0mwarnings[0m[0;34m.[0m[0mwarn[0m[0;34m([0m[0;34m"Mean of empty slice."[0m[0;34m,[0m [0mRuntimeWarning[0m[0;34m,[0m [0mstacklevel[0m[0;34m=[0m[0;36m2[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/core/_methods.py[0m in [0;36m_count_reduce_items[0;34m(arr, axis, keepdims, where)[0m
[1;32m     75[0m         [0mitems[0m [0;34m=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m     76[0m         [0;32mfor[0m [0max[0m [0;32min[0m [0maxis[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 77[0;31m             [0mitems[0m [0;34m*=[0m [0marr[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0mmu[0m[0;34m.[0m[0mnormalize_axis_index[0m[0;34m([0m[0max[0m[0;34m,[0m [0marr[0m[0;34m.[0m[0mndim[0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     78[0m         [0mitems[0m [0;34m=[0m [0mnt[0m[0;34m.[0m[0mintp[0m[0;34m([0m[0mitems[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     79[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mAxisError[0m: axis 2 is out of bounds for array of dimension 1

## === cell 15
train_bpps_stats = np.concatenate([stats[:,:,None] for stats in train_bpps_stats], axis=2)
public_bpps_stats = np.concatenate([stats[:,:,None] for stats in public_bpps_stats], axis=2)
private_bpps_stats = np.concatenate([stats[:,:,None] for stats in private_bpps_stats], axis=2)

train_bpps_stats.shape, public_bpps_stats.shape, private_bpps_stats.shape
