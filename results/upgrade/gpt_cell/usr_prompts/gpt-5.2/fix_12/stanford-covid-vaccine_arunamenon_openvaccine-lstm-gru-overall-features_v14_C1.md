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
plotly==5.24.1
plotly-express==0.4.1
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
import json
import os

import sys
import subprocess

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver  # type: ignore
except Exception:
    _pb_ver = None


def _major(ver):
    try:
        return int(str(ver).split(".", 1)[0])
    except Exception:
        return None


if _pb_ver is None or _major(_pb_ver) >= 6:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf>=5.28.0,<6"]
    )

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
train_data.head()


## === cell 16
for i in range(0,len(train_data)):
    num_time_steps = len(train_data['reactivity'].iloc[i])
    for j in range(num_time_steps):
        train_data['reactivity'][i][j] =  train_data['reactivity'][i][j] - train_data['reactivity_error'][i][j]
        train_data['deg_Mg_pH10'][i][j] =  train_data['deg_Mg_pH10'][i][j] - train_data['deg_error_Mg_pH10'][i][j]
        train_data['deg_pH10'][i][j] =  train_data['deg_pH10'][i][j] - train_data['deg_error_pH10'][i][j]
        train_data['deg_Mg_50C'][i][j] =  train_data['deg_Mg_50C'][i][j] - train_data['deg_error_Mg_50C'][i][j]
        train_data['deg_50C'][i][j] =  train_data['deg_50C'][i][j] - train_data['deg_error_50C'][i][j]


## === cell 17
train_data.columns


## === cell 18
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


## === cell 19
token2int


## === cell 20
def read_bpps_sum(df):
    bpps_arr = []
    for mol_id, seq_len in zip(df.id.to_list(), df.seq_length.to_list()):
        paths_to_try = [
            f"/kaggle/input/stanford-covid-vaccine/bpps/{mol_id}.npy",
            f"/kaggle/data/stanford-covid-vaccine/bpps/{mol_id}.npy",
            f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy",
        ]
        bpps_path = next((p for p in paths_to_try if os.path.exists(p)), None)
        if bpps_path is None:
            bpps_arr.append(np.zeros(int(seq_len), dtype=np.float32))
        else:
            bpps_arr.append(np.load(bpps_path).sum(axis=1))
    return bpps_arr


def read_bpps_max(df):
    bpps_arr = []
    for mol_id, seq_len in zip(df.id.to_list(), df.seq_length.to_list()):
        paths_to_try = [
            f"/kaggle/input/stanford-covid-vaccine/bpps/{mol_id}.npy",
            f"/kaggle/data/stanford-covid-vaccine/bpps/{mol_id}.npy",
            f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy",
        ]
        bpps_path = next((p for p in paths_to_try if os.path.exists(p)), None)
        if bpps_path is None:
            bpps_arr.append(np.zeros(int(seq_len), dtype=np.float32))
        else:
            bpps_arr.append(np.load(bpps_path).max(axis=1))
    return bpps_arr


def read_bpps_nb(df):
    bpps_nb_mean = 0.077522
    bpps_nb_std = 0.08914
    bpps_arr = []
    for mol_id, seq_len in zip(df.id.to_list(), df.seq_length.to_list()):
        paths_to_try = [
            f"/kaggle/input/stanford-covid-vaccine/bpps/{mol_id}.npy",
            f"/kaggle/data/stanford-covid-vaccine/bpps/{mol_id}.npy",
            f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy",
        ]
        bpps_path = next((p for p in paths_to_try if os.path.exists(p)), None)
        if bpps_path is None:
            bpps_arr.append(np.zeros(int(seq_len), dtype=np.float32))
        else:
            bpps = np.load(bpps_path)
            bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
            bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
            bpps_arr.append(bpps_nb)
    return bpps_arr


os.chdir("/kaggle/working/")
train_data["bpps_sum"] = read_bpps_sum(train_data)
test_data["bpps_sum"] = read_bpps_sum(test_data)
train_data["bpps_max"] = read_bpps_max(train_data)
test_data["bpps_max"] = read_bpps_max(test_data)
train_data["bpps_nb"] = read_bpps_nb(train_data)
test_data["bpps_nb"] = read_bpps_nb(test_data)

train_data.head()


## === cell 21
import plotly.express as px
from collections import Counter as count

def get_bases(data):
    bases = []

    for j in range(len(data)):
        counts = dict(count(data.iloc[j]['sequence']))
        bases.append((
            counts['A'] / 107,
            counts['G'] / 107,
            counts['C'] / 107,
            counts['U'] / 107
        ))

    bases = pd.DataFrame(bases, columns=['A_percent', 'G_percent', 'C_percent', 'U_percent'])
    return bases


## === cell 22
def get_pairs_rate(data):
    pairs_rate = []

    for j in range(len(data)):
        res = dict(count(data.iloc[j]['structure']))
        pairs_rate.append(res['('] / 53.5)

    pairs_rate = pd.DataFrame(pairs_rate, columns=['pairs_rate'])
    return pairs_rate


## === cell 23
def get_pairs(data):
    pairs = []
    all_partners = []
    for j in range(len(data)):
        partners = [-1 for i in range(130)]
        pairs_dict = {}
        queue = []
        for i in range(0, len(data.iloc[j]['structure'])):
            if data.iloc[j]['structure'][i] == '(':
                queue.append(i)
            if data.iloc[j]['structure'][i] == ')':
                first = queue.pop()
                try:
                    pairs_dict[(data.iloc[j]['sequence'][first], data.iloc[j]['sequence'][i])] += 1
                except:
                    pairs_dict[(data.iloc[j]['sequence'][first], data.iloc[j]['sequence'][i])] = 1

                partners[first] = i
                partners[i] = first

        all_partners.append(partners)

        pairs_num = 0
        pairs_unique = [('U', 'G'), ('C', 'G'), ('U', 'A'), ('G', 'C'), ('A', 'U'), ('G', 'U')]
        for item in pairs_dict:
            pairs_num += pairs_dict[item]
        add_tuple = list()
        for item in pairs_unique:
            try:
                add_tuple.append(pairs_dict[item]/pairs_num)
            except:
                add_tuple.append(0)
        pairs.append(add_tuple)

    pairs = pd.DataFrame(pairs, columns=['U-G', 'C-G', 'U-A', 'G-C', 'A-U', 'G-U'])
    return pairs


## === cell 24
def get_loops(data):
    loops = []
    for j in range(len(data)):
        counts = dict(count(data.iloc[j]['predicted_loop_type']))
        available = ['E', 'S', 'H', 'B', 'X', 'I', 'M']
        row = []
        for item in available:
            try:
                row.append(counts[item] / 107)
            except:
                row.append(0)
        loops.append(row)

    loops = pd.DataFrame(loops, columns=available)
    return loops


## === cell 25
from tqdm.notebook import tqdm

def get_structure_adj(train):
    
    Ss = []
    for i in tqdm(range(len(train))):
        seq_length = train["seq_length"].iloc[i]
        structure = train["structure"].iloc[i]
        sequence = train["sequence"].iloc[i]

        cue = []
        a_structures = {
            ("A", "U") : np.zeros([seq_length, seq_length]),
            ("C", "G") : np.zeros([seq_length, seq_length]),
            ("U", "G") : np.zeros([seq_length, seq_length]),
            ("U", "A") : np.zeros([seq_length, seq_length]),
            ("G", "C") : np.zeros([seq_length, seq_length]),
            ("G", "U") : np.zeros([seq_length, seq_length]),
        }
        a_structure = np.zeros([seq_length, seq_length])
        for j in range(seq_length):
            if structure[j] == "(":
                cue.append(j)
            elif structure[j] == ")":
                start = cue.pop()
                a_structures[(sequence[start], sequence[j])][start, j] = 1
                a_structures[(sequence[j], sequence[start])][j, start] = 1
        
        a_strc = np.stack([a for a in a_structures.values()], axis = 2)
        a_strc = np.sum(a_strc, axis = 2, keepdims = True)
        Ss.append(a_strc)
    
    Ss = np.array(Ss)
    print(Ss.shape)
    return Ss


## === cell 26
def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type'], seq_length = 107):
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

    Ss = get_structure_adj(df)
    Ss = Ss.sum(axis = 1)
    
    data = np.concatenate([base_fea,bpps_sum_fea,bpps_max_fea,bpps_nb_fea, Ss], 2)
    
    array_data = (np.reshape(([([list(df['A_percent'])[0]] * seq_length)]),(1,seq_length,1)))
    for i in range(1,len(df)):
        array_data_i = (np.reshape(([([list(df['A_percent'])[i]] * seq_length)]),(1,seq_length,1)))
        array_data = np.concatenate([array_data,array_data_i], axis = 0)

    data = np.concatenate([data, array_data], 2)

    for col in ['G_percent','C_percent','U_percent','U-G','C-G','U-A','G-C','A-U','G-U',
                'E','S','H','B','X','I','M','pairs_rate']:
        array_data = (np.reshape(([([list(df[col])[0]] * seq_length)]),(1,seq_length,1)))
        for i in range(1,len(df)):
            array_data_i = (np.reshape(([([list(df[col])[i]] * seq_length)]),(1,seq_length,1)))
            array_data = np.concatenate([array_data,array_data_i], axis = 0)

        data = np.concatenate([data, array_data], 2)

    return data


## === cell 27
bases = get_bases(train_data)
pairs = get_pairs(train_data)
loops = get_loops(train_data)
pairs_rate = get_pairs_rate(train_data)
train_data = pd.concat([train_data, bases, pairs, loops, pairs_rate], axis=1)

bases = get_bases(test_data)
pairs = get_pairs(test_data)
loops = get_loops(test_data)
pairs_rate = get_pairs_rate(test_data)
test_data = pd.concat([test_data, bases, pairs, loops, pairs_rate], axis=1)


## === cell 28
train_inputs = preprocess_inputs(train_data.loc[train_data['signal_to_noise'] > 1], seq_length = 107)
train_labels = np.array(train_data.loc[train_data['signal_to_noise'] > 1][target_cols].values.tolist()).transpose((0, 2, 1))


## === cell 29
from keras.losses import mean_squared_error

def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(mean_squared_error(y_true, y_pred))

def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)

def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
                                tf.keras.layers.LSTM(hidden_dim,
                                dropout=dropout,
                                return_sequences=True,
                                kernel_initializer = 'orthogonal'))

def gru_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
                                tf.keras.layers.GRU(hidden_dim,
                                dropout=dropout,
                                return_sequences=True,
                                kernel_initializer='orthogonal'))

def build_model(n_layers = 2, seq_len = 107, num_features = 25, embed_dim = 200, sp_dropout = 0.2, hidden_dim = 256, dropout = 0.5, pred_len = 68, gru_flag = False):
    
    inputs = tf.keras.layers.Input(shape=(seq_len, num_features))
    categorical_feats = inputs[:, :, :3]
    numerical_feats = inputs[:, :, 3:7]
    overall_gene_feats = inputs[:, :, 7:]

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(categorical_feats)
   
    reshaped = tf.reshape(embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
    
    reshaped = tf.keras.layers.concatenate([reshaped, numerical_feats], axis=2)
    
    spatial_dropout = tf.keras.layers.SpatialDropout1D(sp_dropout)(reshaped)
        
    normalized_layer_1 = tf.keras.layers.BatchNormalization()(spatial_dropout)
    
    if gru_flag:
        for x in range(n_layers):
            normalized_layer_1 = gru_layer(hidden_dim, dropout)(normalized_layer_1)
        normalized_layer_2 = tf.keras.layers.BatchNormalization()(normalized_layer_1)
    else:
        for x in range(n_layers):
            normalized_layer_1 = lstm_layer(hidden_dim, dropout)(normalized_layer_1)
        normalized_layer_2 = tf.keras.layers.BatchNormalization()(normalized_layer_1)
    
    concat_layer = tf.keras.layers.concatenate([normalized_layer_2, overall_gene_feats], axis=2)
    
    dense_layer = tf.keras.layers.Dense(50, activation='linear')(concat_layer)
    normalized_layer_3 = tf.keras.layers.BatchNormalization()(dense_layer)
    dropout_layer = tf.keras.layers.Dropout(sp_dropout)(normalized_layer_3)
    
    truncated = dropout_layer[:, :pred_len]

    out = tf.keras.layers.Dense(5, activation='linear')(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.optimizers.Adam()

    model.compile(optimizer = adam, loss = MCRMSE)
    
    return model


## === cell 30
EPOCHS = 60
BATCH_SIZE = 32


def _build_model_keras3_safe(*args, **kwargs):
    model = build_model(*args, **kwargs)
    return model


def build_model(
    n_layers=2,
    seq_len=107,
    num_features=25,
    embed_dim=200,
    sp_dropout=0.2,
    hidden_dim=256,
    dropout=0.5,
    pred_len=68,
    gru_flag=False,
):

    inputs = tf.keras.layers.Input(shape=(seq_len, num_features))
    categorical_feats = inputs[:, :, :3]
    numerical_feats = inputs[:, :, 3:7]
    overall_gene_feats = inputs[:, :, 7:]

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        categorical_feats
    )

    reshaped = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)

    reshaped = tf.keras.layers.concatenate([reshaped, numerical_feats], axis=2)

    spatial_dropout = tf.keras.layers.SpatialDropout1D(sp_dropout)(reshaped)

    normalized_layer_1 = tf.keras.layers.BatchNormalization()(spatial_dropout)

    if gru_flag:
        for x in range(n_layers):
            normalized_layer_1 = gru_layer(hidden_dim, dropout)(normalized_layer_1)
        normalized_layer_2 = tf.keras.layers.BatchNormalization()(normalized_layer_1)
    else:
        for x in range(n_layers):
            normalized_layer_1 = lstm_layer(hidden_dim, dropout)(normalized_layer_1)
        normalized_layer_2 = tf.keras.layers.BatchNormalization()(normalized_layer_1)

    concat_layer = tf.keras.layers.concatenate(
        [normalized_layer_2, overall_gene_feats], axis=2
    )

    dense_layer = tf.keras.layers.Dense(50, activation="linear")(concat_layer)
    normalized_layer_3 = tf.keras.layers.BatchNormalization()(dense_layer)
    dropout_layer = tf.keras.layers.Dropout(sp_dropout)(normalized_layer_3)

    truncated = dropout_layer[:, :pred_len]

    out = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.optimizers.Adam()

    model.compile(optimizer=adam, loss=MCRMSE)

    return model


model_GRU_on_train_data = build_model(gru_flag=True)
model_GRU_on_train_data.summary()
model_GRU_callback = tf.keras.callbacks.ModelCheckpoint(f"GRU model.h5")

history_GRU = model_GRU_on_train_data.fit(
    train_inputs,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
    callbacks=[model_GRU_callback],
)


## === cell 31
EPOCHS = 60
BATCH_SIZE = 32

model_LSTM_on_train_data = build_model(gru_flag = False)
model_LSTM_on_train_data.summary()
model_LSTM_callback = tf.keras.callbacks.ModelCheckpoint(f'LSTM model.h5')

history_LSTM = model_LSTM_on_train_data.fit(train_inputs, train_labels,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  verbose = 2,
                  callbacks=[model_LSTM_callback])  


## === cell 32
print(f" LSTM loss: {min(history_LSTM.history['loss'])}")
print(f" GRU loss: {min(history_GRU.history['loss'])}")

fig, ax = plt.subplots(1, 1, figsize = (20, 10))

ax.plot(history_LSTM.history['loss'])
ax.plot(history_GRU.history['loss'])

ax.set_title('Model - LSTM vs GRU')

ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
plt.legend(loc="upper right")
plt.show()


## === cell 33
def preprocess_inputs(
    df, cols=["sequence", "structure", "predicted_loop_type"], seq_length=107
):
    n = len(df)

    encoded_cols = []
    for col in cols:
        arr = np.empty((n, seq_length), dtype=np.int32)
        for i, seq in enumerate(df[col].tolist()):
            arr[i, :] = [token2int[x] for x in seq]
        encoded_cols.append(arr)

    base_fea = np.stack(encoded_cols, axis=2)  # (n, seq_length, 3)

    bpps_sum_fea = np.array(df["bpps_sum"].to_list())[:, :, np.newaxis]
    bpps_max_fea = np.array(df["bpps_max"].to_list())[:, :, np.newaxis]
    bpps_nb_fea = np.array(df["bpps_nb"].to_list())[:, :, np.newaxis]

    Ss = get_structure_adj(df)
    Ss = Ss.sum(axis=1)

    data = np.concatenate([base_fea, bpps_sum_fea, bpps_max_fea, bpps_nb_fea, Ss], 2)

    array_data = np.reshape(
        ([([list(df["A_percent"])[0]] * seq_length)]), (1, seq_length, 1)
    )
    for i in range(1, len(df)):
        array_data_i = np.reshape(
            ([([list(df["A_percent"])[i]] * seq_length)]), (1, seq_length, 1)
        )
        array_data = np.concatenate([array_data, array_data_i], axis=0)

    data = np.concatenate([data, array_data], 2)

    for col in [
        "G_percent",
        "C_percent",
        "U_percent",
        "U-G",
        "C-G",
        "U-A",
        "G-C",
        "A-U",
        "G-U",
        "E",
        "S",
        "H",
        "B",
        "X",
        "I",
        "M",
        "pairs_rate",
    ]:
        array_data = np.reshape(
            ([([list(df[col])[0]] * seq_length)]), (1, seq_length, 1)
        )
        for i in range(1, len(df)):
            array_data_i = np.reshape(
                ([([list(df[col])[i]] * seq_length)]), (1, seq_length, 1)
            )
            array_data = np.concatenate([array_data, array_data_i], axis=0)

        data = np.concatenate([data, array_data], 2)

    return data


## === cell 34
public_df = test_data.loc[test_data["seq_length"] == 107].reset_index(drop=True)
private_df = test_data.loc[test_data["seq_length"] == 130].reset_index(drop=True)

public_inputs = (
    preprocess_inputs(public_df, seq_length=107)
    if len(public_df) > 0
    else np.zeros((0, 107, 25), dtype=np.float32)
)
private_inputs = (
    preprocess_inputs(private_df, seq_length=130)
    if len(private_df) > 0
    else np.zeros((0, 130, 25), dtype=np.float32)
)

model_LSTM_on_test_data_public = build_model(seq_len=107, pred_len=107, gru_flag=False)
model_LSTM_on_test_data_public.load_weights(f"LSTM model.h5")
pred_test_data_public_LSTM = model_LSTM_on_test_data_public.predict(public_inputs)

model_GRU_on_test_data_public = build_model(seq_len=107, pred_len=107, gru_flag=True)
model_GRU_on_test_data_public.load_weights(f"GRU model.h5")
pred_test_data_public_GRU = model_GRU_on_test_data_public.predict(public_inputs)

if len(private_df) > 0:
    model_LSTM_on_test_data_private = build_model(
        seq_len=130, pred_len=130, gru_flag=False
    )
    model_LSTM_on_test_data_private.load_weights(f"LSTM model.h5")
    pred_test_data_private_LSTM = model_LSTM_on_test_data_private.predict(
        private_inputs
    )

    model_GRU_on_test_data_private = build_model(
        seq_len=130, pred_len=130, gru_flag=True
    )
    model_GRU_on_test_data_private.load_weights(f"GRU model.h5")
    pred_test_data_private_GRU = model_GRU_on_test_data_private.predict(private_inputs)
else:
    pred_test_data_private_LSTM = np.zeros((0, 130, 5), dtype=np.float32)
    pred_test_data_private_GRU = np.zeros((0, 130, 5), dtype=np.float32)


## === cell 35
model_LSTM_on_test_data_private = build_model(seq_len=130, pred_len=130, gru_flag = False)
model_LSTM_on_test_data_private.load_weights(f'LSTM model.h5')
pred_test_data_private_LSTM = model_LSTM_on_test_data_private.predict(private_inputs)

model_GRU_on_test_data_private = build_model(seq_len=130, pred_len=130, gru_flag = True)
model_GRU_on_test_data_private.load_weights(f'GRU model.h5')
pred_test_data_private_GRU = model_GRU_on_test_data_private.predict(private_inputs)


## --- ERROR in cell 35, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4113270568.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0mmodel_LSTM_on_test_data_private[0m [0;34m=[0m [0mbuild_model[0m[0;34m([0m[0mseq_len[0m[0;34m=[0m[0;36m130[0m[0;34m,[0m [0mpred_len[0m[0;34m=[0m[0;36m130[0m[0;34m,[0m [0mgru_flag[0m [0;34m=[0m [0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0mmodel_LSTM_on_test_data_private[0m[0;34m.[0m[0mload_weights[0m[0;34m([0m[0;34mf'LSTM model.h5'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m [0mpred_test_data_private_LSTM[0m [0;34m=[0m [0mmodel_LSTM_on_test_data_private[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mprivate_inputs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m [0mmodel_GRU_on_test_data_private[0m [0;34m=[0m [0mbuild_model[0m[0;34m([0m[0mseq_len[0m[0;34m=[0m[0;36m130[0m[0;34m,[0m [0mpred_len[0m[0;34m=[0m[0;36m130[0m[0;34m,[0m [0mgru_flag[0m [0;34m=[0m [0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    120[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m             [0;31m# `keras.config.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/progbar.py[0m in [0;36mupdate[0;34m(self, current, values, finalize)[0m
[1;32m    117[0m [0;34m[0m[0m
[1;32m    118[0m             [0;32mif[0m [0mself[0m[0;34m.[0m[0mtarget[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 119[0;31m                 [0mnumdigits[0m [0;34m=[0m [0mint[0m[0;34m([0m[0mmath[0m[0;34m.[0m[0mlog10[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mtarget[0m[0;34m)[0m[0;34m)[0m [0;34m+[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    120[0m                 [0mbar[0m [0;34m=[0m [0;34m([0m[0;34m"%"[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mnumdigits[0m[0;34m)[0m [0;34m+[0m [0;34m"d/%d"[0m[0;34m)[0m [0;34m%[0m [0;34m([0m[0mcurrent[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mtarget[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m                 [0mbar[0m [0;34m=[0m [0;34mf"\x1b[1m{bar}\x1b[0m "[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: math domain error

## === cell 36
def format_predictions(public_preds, private_preds):
    preds = []
    
    for df, preds_ in [(public_df, public_preds), (private_df, private_preds)]:
        for i, uid in enumerate(df.id):
            single_pred = preds_[i]

            single_df = pd.DataFrame(single_pred, columns=target_cols)
            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

            preds.append(single_df)
    return pd.concat(preds).reset_index(drop = True)
