# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Predict likely degradation rates at each base of an RNA molecule.

## Metric
Mean columnwise root mean squared error:

$\textrm{MCRMSE} = \frac{1}{N_{t}}\sum_{j=1}^{N_{t}}\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{ij} - \hat{y}_{ij})^2}$

where $N_{t}$ is the number of scored ground truth target columns, and $y$ and $\hat{y}$ are the actual and predicted values, respectively.

There are multiple ground truth values provided in the training data. While the submission format requires all 5 to be predicted, only the following are scored: reactivity, deg_Mg_pH10, and deg_Mg_50C.

## Submission Formats
For each sample `id` in the test set, you must predict targets for *each* sequence position (`seqpos`), one per row. If the length of the `sequence` of an `id` is, e.g., 107, then you should make 107 predictions. Positions greater than the `seq_scored` value of a sample are not scored, but still need a value in the solution file.

```csv
id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C
id_d190610e8_0,0.1,0.3,0.2,0.5,0.4
id_d190610e8_1,0.3,0.2,0.5,0.4,0.2
id_d190610e8_2,0.5,0.4,0.2,0.1,0.2
etc.
```

## Dataset 
- **train.json** - the training data
- **test.json** - the test set, without any columns associated with the ground truth.
- **sample_submission.csv** - a sample submission file in the correct format

#### Columns
- `id` - An arbitrary identifier for each sample.
- `seq_scored` - (68 in Train and Public Test, 68 in Private Test) Integer value denoting the number of positions used in scoring with predicted values. This should match the length of `reactivity`, `deg_*` and `*_error_*` columns.
- `seq_length` - (107 in Train and Public Test, 107 in Private Test) Integer values, denotes the length of `sequence`.
- `sequence` - (1x107 string in Train and Public Test, 107 in Private Test) Describes the RNA sequence, a combination of `A`, `G`, `U`, and `C` for each sample. Should be 107 characters long, and the first 68 bases should correspond to the 68 positions specified in `seq_scored` (note: indexed starting at 0).
- `structure` - (1x107 string in Train and Public Test, 107 in Private Test) An array of `(`, `)`, and `.` characters that describe whether a base is estimated to be paired or unpaired. Paired bases are denoted by opening and closing parentheses e.g. (....) means that base 0 is paired to base 5, and bases 1-4 are unpaired.
- `reactivity` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likely secondary structure of the RNA sample.
- `deg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high pH (pH 10).
- `deg_Mg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium in high pH (pH 10).
- `deg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high temperature (50 degrees Celsius).
- `deg_Mg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium at high temperature (50 degrees Celsius).
- `*_error_*` - An array of floating point numbers, should have the same length as the corresponding `reactivity` or `deg_*` columns, calculated errors in experimental values obtained in `reactivity` and `deg_*` columns.
- `predicted_loop_type` - (1x107 string) Describes the structural context (also referred to as 'loop type')of each character in `sequence`. Loop types assigned by bpRNA from Vienna RNAfold 2 structure. From the bpRNA_documentation: S: paired "Stem" M: Multiloop I: Internal loop B: Bulge H: Hairpin loop E: dangling End X: eXternal loop
    - `S/N filter` Indicates if the sample passed filters described below in `Additional Notes`.

#### Additional Notes
At the beginning of the competition, Stanford scientists have data on 2400 RNA sequences of length 107. For technical reasons, measurements cannot be carried out on the final bases of these RNA sequences, so we have experimental data (ground truth) in 5 conditions for the first 68 bases.

We have split out 240 of these 2400 sequences for a public test set to allow for continuous evaluation through the competition, on the public leaderboard. These sequences, in `test.json`, have been additionally filtered based on three criteria detailed below to ensure that this subset is not dominated by any large cluster of RNA molecules with poor data, which might bias the public leaderboard. The remaining 2160 sequences for which we have data are in `train.json`.

For our final and most important scoring (the Private Leaderbooard), Stanford scientists are carrying out measurements on 240 new RNAs. For these data, we expect to have measurements for the first 68 bases, again missing the ends of the RNA. These sequences constitute the 240 sequences in `test.json`.

For those interested in how the sequences in `test.json` were filtered, here were the steps to ensure a diverse and high quality test set for public leaderboard scoring:

1. Minimum value across all 5 conditions must be greater than -0.5.
2. Mean signal/noise across all 5 conditions must be greater than 1.0. [Signal/noise is defined as mean( measurement value over 68 nts )/mean( statistical error in measurement value over 68 nts)]
3. To help ensure sequence diversity, the resulting sequences were clustered into clusters with less than 50% sequence similarity, and the 240 test set sequences were chosen from clusters with 3 or fewer members. That is, any sequence in the test set should be sequence similar to at most 2 other sequences.

Note that these filters have not been applied to the 2160 RNAs in the public training data `train.json` -- some of those measurements have negative values or poor signal-to-noise, or some RNA sequences have near-identical sequences in that set. But we are providing all those data in case competitors can squeeze out more signal.

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

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

# 5. Target score

0.3733857541732854

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU, Dropout
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import os
import copy
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from torch.utils.data import Dataset,TensorDataset, DataLoader,RandomSampler
import time,datetime
import tensorflow as tf
import keras.backend as K
import tensorflow.keras.layers as L

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def allocate_gpu_memory(gpu_number=0):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    if physical_devices:
        try:
            print("Found {} GPU(s)".format(len(physical_devices)))
            tf.config.set_visible_devices(physical_devices[gpu_number], 'GPU')
            tf.config.experimental.set_memory_growth(physical_devices[gpu_number], True)
            print("#{} GPU memory is allocated".format(gpu_number))
        except RuntimeError as e:
            print(e)
    else:
        print("Not enough GPU hardware devices available")
allocate_gpu_memory()

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

def rmse(y_actual, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)
    return K.sqrt(mse)

def mcrmse(y_actual, y_pred, num_scored=len(pred_cols)):
    score = 0
    for i in range(num_scored):
        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / num_scored
    return score

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


train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)

def read_bpps_sum(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").max(axis=1))
    return bpps_arr

def read_bpps_max(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").sum(axis=1))
    return bpps_arr

def read_bpps_nb(df):
    bpps_nb_mean = 0.077522 # mean of bpps_nb across all training data
    bpps_nb_std = 0.08914   # std of bpps_nb across all training data
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps = np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy")
        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
        bpps_arr.append(bpps_nb)
    return bpps_arr

train['bpps_sum'] = read_bpps_sum(train)
test['bpps_sum'] = read_bpps_sum(test)
train['bpps_max'] = read_bpps_max(train)
test['bpps_max'] = read_bpps_max(test)
train['bpps_nb'] = read_bpps_nb(train)
test['bpps_nb'] = read_bpps_nb(test)

from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=200, random_state=110).fit(preprocess_inputs(train)[:,:,0])
train['cluster_id'] = kmeans_model.labels_

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1594245588.py in <cell line: 0>()
     56     return bpps_arr
     57 
---> 58 train['bpps_sum'] = read_bpps_sum(train)
     59 test['bpps_sum'] = read_bpps_sum(test)
     60 train['bpps_max'] = read_bpps_max(train)

/tmp/ipykernel_11/1594245588.py in read_bpps_sum(df)
     34     bpps_arr = []
     35     for mol_id in df.id.to_list():
---> 36         bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").max(axis=1))
     37     return bpps_arr
     38 

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '../input/stanford-covid-vaccine/bpps/id_001f94081.npy'

## === cell 4
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer = 'orthogonal'))

def lstm_layer(hidden_dim, dropout):
    return L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer = 'orthogonal'))

def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=256, type=0):
    inputs = L.Input(shape=(seq_len, 6))
    
    categorical_feat_dim = 3
    categorical_fea = inputs[:, :, :categorical_feat_dim]
    numerical_fea = inputs[:, :, 3:]

    embed = L.Embedding(input_dim=len(token2int), output_dim=embed_dim)(categorical_fea)
    reshaped = tf.reshape(embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
    reshaped = L.concatenate([reshaped, numerical_fea], axis=2)
    
    if type == 0:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    elif type == 1:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    elif type == 2:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
    elif type == 3:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
    
    truncated = hidden[:, :pred_len]
    out = L.Dense(5, activation='linear')(truncated)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(tf.keras.optimizers.Adam(), loss=mcrmse)
    return model

keras_model = build_model()
keras_model.load_weights('../input/gru-lstm-with-feature-engineering-and-augmentation/modelGRU_LSTM1_cv0.h5')
keras_model.layers

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1956427182.py in <cell line: 0>()
     36     return model
     37 
---> 38 keras_model = build_model()
     39 keras_model.load_weights('../input/gru-lstm-with-feature-engineering-and-augmentation/modelGRU_LSTM1_cv0.h5')
     40 keras_model.layers

/tmp/ipykernel_11/1956427182.py in build_model(seq_len, pred_len, dropout, embed_dim, hidden_dim, type)
     14 
     15     embed = L.Embedding(input_dim=len(token2int), output_dim=embed_dim)(categorical_fea)
---> 16     reshaped = tf.reshape(embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
     17     reshaped = L.concatenate([reshaped, numerical_fea], axis=2)
     18 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
     86   def wrapper(*args, **kwargs):
     87     if not ops.is_auto_dtype_conversion_enabled():
---> 88       return op(*args, **kwargs)
     89     bound_arguments = signature.bind(*args, **kwargs)
     90     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __tf_tensor__(self, dtype, name)
    136 
    137     def __tf_tensor__(self, dtype=None, name=None):
--> 138         raise ValueError(
    139             "A KerasTensor cannot be used as input to a TensorFlow function. "
    140             "A KerasTensor is a symbolic placeholder for a shape and dtype, "

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```


## === cell 5
device = torch.device('cuda:%s'%0 if torch.cuda.is_available() else 'cpu')
def Init_params(shape,w=None,b=None):
    if w is None:
        w = torch.nn.Parameter(torch.empty(*shape))
        nn.init.xavier_uniform_(w)
    else:
        w = torch.nn.Parameter(w)
    if b is None:
        b = torch.nn.Parameter(torch.zeros(shape[1]))
    else:
        b = torch.nn.Parameter(b)
    return w,b

class GRU(nn.Module):
    def __init__(self,input_dim,hidden_dim,w_i=None,b_i=None,w_h=None,b_h=None):
        super(GRU, self).__init__()
        self.w_i,self.b_i = Init_params([input_dim,3*hidden_dim],w_i,b_i)
        self.w_h,self.b_h = Init_params([hidden_dim,3*hidden_dim],w_h,b_h)
        self.hd = hidden_dim
    def forward(self,x):
        hidden = torch.zeros((x.shape[0], self.hd)).to(device)
        output = []
        for i in range(x.shape[1]):
            x_z = torch.matmul(x[:,i,:],self.w_i[:,:self.hd]) + self.b_i[:self.hd]
            x_r = torch.matmul(x[:,i,:],self.w_i[:,self.hd:2*self.hd]) + self.b_i[self.hd:2*self.hd]
            x_n = torch.matmul(x[:,i,:],self.w_i[:,2*self.hd:]) + self.b_i[2*self.hd:]

            h_z = torch.matmul(hidden,self.w_h[:,:self.hd]) + self.b_h[:self.hd]
            h_r = torch.matmul(hidden,self.w_h[:,self.hd:2*self.hd]) + self.b_h[self.hd:2*self.hd]
            h_n = torch.matmul(hidden,self.w_h[:,2*self.hd:]) + self.b_h[2*self.hd:]

            z = torch.sigmoid(x_z+h_z)
            r = torch.sigmoid(x_r+h_r)
            n = torch.tanh(x_n+r*h_n)
            h = (1-z)*n + z*hidden
            hidden = h
            output.append(h.unsqueeze(1))
        return torch.cat(output,1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        num_target=5
        w = torch.Tensor(keras_model.layers[2].get_weights()[0])
        self.cate_emb = nn.Embedding.from_pretrained(w,freeze=False)
        self.gru = GRU(100*3+3, 256, torch.Tensor(keras_model.layers[-4].get_weights()[0]).contiguous(),
                       torch.Tensor(keras_model.layers[-4].get_weights()[2][0]).contiguous(),
                       torch.Tensor(keras_model.layers[-4].get_weights()[1]).contiguous(),
                       torch.Tensor(keras_model.layers[-4].get_weights()[2][1]).contiguous())
        self.reverse_gru = GRU(100*3+3, 256, torch.Tensor(keras_model.layers[-4].get_weights()[3]).contiguous(),
                       torch.Tensor(keras_model.layers[-4].get_weights()[5][0]).contiguous(),
                       torch.Tensor(keras_model.layers[-4].get_weights()[4]).contiguous(),
                       torch.Tensor(keras_model.layers[-4].get_weights()[5][1]).contiguous())
        self.gru1 = GRU(512, 256, torch.Tensor(keras_model.layers[-3].get_weights()[0]).contiguous(),
                       torch.Tensor(keras_model.layers[-3].get_weights()[2][0]).contiguous(),
                       torch.Tensor(keras_model.layers[-3].get_weights()[1]).contiguous(),
                       torch.Tensor(keras_model.layers[-3].get_weights()[2][1]).contiguous())
        self.reverse_gru1 = GRU(512, 256, torch.Tensor(keras_model.layers[-3].get_weights()[3]).contiguous(),
                       torch.Tensor(keras_model.layers[-3].get_weights()[5][0]).contiguous(),
                       torch.Tensor(keras_model.layers[-3].get_weights()[4]).contiguous(),
                       torch.Tensor(keras_model.layers[-3].get_weights()[5][1]).contiguous())
        self.predict = nn.Linear(512,num_target)
        for i,(n,p) in enumerate(self.predict.named_parameters()):
            if i == 0:
                p.data = torch.nn.Parameter(torch.Tensor(keras_model.layers[-1].get_weights()[0].T).contiguous())
            if i == 1:
                p.data = torch.nn.Parameter(torch.Tensor(keras_model.layers[-1].get_weights()[1]).contiguous())

    def forward(self, cateX,contX):
        cate_x = self.cate_emb(cateX).view(cateX.shape[0],cateX.shape[1],-1)
        sequence = torch.cat([cate_x,contX],-1)
        x = self.gru(sequence)
        reverse_x = torch.flip(self.reverse_gru(torch.flip(sequence,[1])),[1])
        sequence = torch.cat([x,reverse_x],-1)
        x = self.gru1(sequence)
        reverse_x = torch.flip(self.reverse_gru1(torch.flip(sequence,[1])),[1])
        x = torch.cat([x,reverse_x],-1)
        x = F.dropout(x,0.5,training=self.training)
        predict = self.predict(x)
        return predict
pytorch_model = Net()
pytorch_model.to(device)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/984181111.py in <cell line: 0>()
     79         predict = self.predict(x)
     80         return predict
---> 81 pytorch_model = Net()
     82 pytorch_model.to(device)

/tmp/ipykernel_11/984181111.py in __init__(self)
     42         super(Net, self).__init__()
     43         num_target=5
---> 44         w = torch.Tensor(keras_model.layers[2].get_weights()[0])
     45         self.cate_emb = nn.Embedding.from_pretrained(w,freeze=False)
     46         self.gru = GRU(100*3+3, 256, torch.Tensor(keras_model.layers[-4].get_weights()[0]).contiguous(),

NameError: name 'keras_model' is not defined

## === cell 6
x = preprocess_inputs(train[:1])
cate_x = torch.LongTensor(x[:,:,:3]).to(device)
cont_x = torch.Tensor(x[:,:,3:]).to(device)
y = np.array(train[:1][pred_cols].values.tolist()).transpose((0, 2, 1))

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'bpps_sum'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3467524336.py in <cell line: 0>()
      1 # check 1 samples
----> 2 x = preprocess_inputs(train[:1])
      3 cate_x = torch.LongTensor(x[:,:,:3]).to(device)
      4 cont_x = torch.Tensor(x[:,:,3:]).to(device)
      5 y = np.array(train[:1][pred_cols].values.tolist()).transpose((0, 2, 1))

/tmp/ipykernel_11/1594245588.py in preprocess_inputs(df, cols)
     22         (0, 2, 1)
     23     )
---> 24     bpps_sum_fea = np.array(df['bpps_sum'].to_list())[:,:,np.newaxis]
     25     bpps_max_fea = np.array(df['bpps_max'].to_list())[:,:,np.newaxis]
     26     bpps_nb_fea = np.array(df['bpps_nb'].to_list())[:,:,np.newaxis]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'bpps_sum'

## === cell 7
keras_y = keras_model.predict(x)
keras_y

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3588972569.py in <cell line: 0>()
----> 1 keras_y = keras_model.predict(x)
      2 keras_y

NameError: name 'keras_model' is not defined

## === cell 8
pytorch_model.eval()
pytorch_y = pytorch_model(cate_x,cont_x).detach().cpu().numpy()
pytorch_y

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2805656226.py in <cell line: 0>()
----> 1 pytorch_model.eval()
      2 pytorch_y = pytorch_model(cate_x,cont_x).detach().cpu().numpy()
      3 pytorch_y

NameError: name 'pytorch_model' is not defined

## === cell 9
np.mean(np.abs(keras_y-pytorch_y[:,:68,:]))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/686428071.py in <cell line: 0>()
      1 # output difference between Keras and Pytorch
----> 2 np.mean(np.abs(keras_y-pytorch_y[:,:68,:]))

NameError: name 'keras_y' is not defined

## === cell 10

gkf = GroupKFold(n_splits=5)
keras_predict = []
pytorch_predict = []
targets = []

for fold, (train_index, valid_index) in enumerate(gkf.split(train,  train['reactivity'], train['cluster_id'])):
    keras_model.load_weights('../input/gru-lstm-with-feature-engineering-and-augmentation/modelGRU_LSTM1_cv%s.h5'%fold)
    t_valid = train.iloc[valid_index]
    t_valid = t_valid[t_valid['SN_filter'] == 1]
    valid_x = preprocess_inputs(t_valid)
    valid_count = valid_x.shape[0]
    valid_cate_x = torch.LongTensor(valid_x[:,:,:3])
    valid_cont_x = torch.Tensor(valid_x[:,:,3:])
    valid_y = torch.Tensor(np.array(t_valid[pred_cols].values.tolist()).transpose((0, 2, 1)))

    valid_data = TensorDataset(valid_cate_x,valid_cont_x,valid_y)
    valid_data_loader = DataLoader(dataset=valid_data,shuffle=False,batch_size=32,num_workers=1)
    valid_y = valid_y.numpy()
    targets.append(valid_y)
    
    keras_predict.append(keras_model.predict(valid_x))
    
    pytorch_model = Net()
    pytorch_model.to(device)
    
    pytorch_model.eval()
 
    all_pred = []

    for data in valid_data_loader:
        cate_x,cont_x,y = [x.to(device) for x in data]
        outputs = pytorch_model(cate_x,cont_x)
        all_pred.append(outputs.detach().cpu().numpy())
    all_pred = np.concatenate(all_pred,0)[:,:68,:]
    pytorch_predict.append(all_pred)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'cluster_id'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/738063056.py in <cell line: 0>()
      6 targets = []
      7 
----> 8 for fold, (train_index, valid_index) in enumerate(gkf.split(train,  train['reactivity'], train['cluster_id'])):
      9     keras_model.load_weights('../input/gru-lstm-with-feature-engineering-and-augmentation/modelGRU_LSTM1_cv%s.h5'%fold)
     10     t_valid = train.iloc[valid_index]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'cluster_id'

## === cell 11
for i in range(5):
    print('fold %s output difference between Keras and Pytorch:'%i,np.mean(np.abs(keras_predict[i]-pytorch_predict[i])))

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/2840318351.py in <cell line: 0>()
      1 for i in range(5):
----> 2     print('fold %s output difference between Keras and Pytorch:'%i,np.mean(np.abs(keras_predict[i]-pytorch_predict[i])))

IndexError: list index out of range

## === cell 13
def Metric(target,pred):
    metric = 0
    for i in range(target.shape[-1]):
        metric += (np.sqrt(np.mean((target[:,:,i]-pred[:,:,i])**2))/target.shape[-1])
    return metric

## === cell 14
for i in range(5):
    print('fold %s'%i,'|','metric of keras outputs:%.6f'%Metric(targets[i],keras_predict[i]),'|','metric of pytorch outputs:%.6f'%Metric(targets[i],pytorch_predict[i]))

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/1342495439.py in <cell line: 0>()
      1 for i in range(5):
----> 2     print('fold %s'%i,'|','metric of keras outputs:%.6f'%Metric(targets[i],keras_predict[i]),'|','metric of pytorch outputs:%.6f'%Metric(targets[i],pytorch_predict[i]))

IndexError: list index out of range

## === cell 16

def rmse(y_actual, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)
    return K.sqrt(mse)

def mcrmse(y_actual, y_pred, num_scored=5):
    score = 0
    for i in range(num_scored):
        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / num_scored
    return score

for i in range(5):
    print('fold %s'%i,mcrmse(targets[i],keras_predict[i]))

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3844450982.py in <cell line: 0>()
     12 
     13 for i in range(5):
---> 14     print('fold %s'%i,mcrmse(targets[i],keras_predict[i]))

IndexError: list index out of range

## === cell 18
for i in range(5):
    print('fold %s'%i,K.mean(mcrmse(targets[i],keras_predict[i])))

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2091138171.py in <cell line: 0>()
      1 # let's check the average of keras loss fuction outputs
      2 for i in range(5):
----> 3     print('fold %s'%i,K.mean(mcrmse(targets[i],keras_predict[i])))

AttributeError: module 'keras.backend' has no attribute 'mean'

## === cell 20

def Pred(df):
    test_x = preprocess_inputs(df)
    test_cate_x = torch.LongTensor(test_x[:,:,:3])
    test_cont_x = torch.Tensor(test_x[:,:,3:])
    test_data = TensorDataset(test_cate_x,test_cont_x)
    test_data_loader = DataLoader(dataset=test_data,shuffle=False,batch_size=64,num_workers=1)
    all_id = []
    for i,row in df.iterrows():
        for j in range(row['seq_length']):
            all_id.append(row['id']+'_%s'%j)

    all_id = np.array(all_id).reshape(-1,1)
    all_pred = np.zeros(len(all_id)*5).reshape(len(all_id),5)
    for fold in range(5):
        keras_model.load_weights('../input/gru-lstm-with-feature-engineering-and-augmentation/modelGRU_LSTM1_cv%s.h5'%fold)
        model = Net()
        model.to(device)
        model.eval()
        t_all_pred = []
        for data in test_data_loader:
            cate_x,cont_x = [x.to(device) for x in data]
            outputs = model(cate_x,cont_x)
            t_all_pred.append(outputs.detach().cpu().numpy())
        t_all_pred = np.concatenate(t_all_pred,0)
        all_pred += t_all_pred.reshape(-1,5)
    all_pred /= 5
    sub = pd.DataFrame(all_pred,columns=['reactivity','deg_Mg_pH10','deg_pH10','deg_Mg_50C','deg_50C'])
    sub['id_seqpos'] = all_id
    return sub
public_sub = Pred(test.loc[test['seq_length']==107])
private_sub = Pred(test.loc[test['seq_length']==130])
pytorch_sub = pd.concat([public_sub,private_sub]).reset_index(drop=True)
pytorch_sub = pytorch_sub[['id_seqpos']+pred_cols]

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'bpps_sum'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2405914407.py in <cell line: 0>()
     31     sub['id_seqpos'] = all_id
     32     return sub
---> 33 public_sub = Pred(test.loc[test['seq_length']==107])
     34 private_sub = Pred(test.loc[test['seq_length']==130])
     35 pytorch_sub = pd.concat([public_sub,private_sub]).reset_index(drop=True)

/tmp/ipykernel_11/2405914407.py in Pred(df)
      3 # predict test
      4 def Pred(df):
----> 5     test_x = preprocess_inputs(df)
      6     test_cate_x = torch.LongTensor(test_x[:,:,:3])
      7     test_cont_x = torch.Tensor(test_x[:,:,3:])

/tmp/ipykernel_11/1594245588.py in preprocess_inputs(df, cols)
     22         (0, 2, 1)
     23     )
---> 24     bpps_sum_fea = np.array(df['bpps_sum'].to_list())[:,:,np.newaxis]
     25     bpps_max_fea = np.array(df['bpps_max'].to_list())[:,:,np.newaxis]
     26     bpps_nb_fea = np.array(df['bpps_nb'].to_list())[:,:,np.newaxis]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'bpps_sum'

## === cell 21
pytorch_sub = pytorch_sub.sort_values(by=['id_seqpos']).reset_index(drop=True)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2731235935.py in <cell line: 0>()
----> 1 pytorch_sub = pytorch_sub.sort_values(by=['id_seqpos']).reset_index(drop=True)

NameError: name 'pytorch_sub' is not defined

## === cell 22
keras_sub = pd.read_csv('../input/gru-lstm-with-feature-engineering-and-augmentation/submission.csv')

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3102556453.py in <cell line: 0>()
----> 1 keras_sub = pd.read_csv('../input/gru-lstm-with-feature-engineering-and-augmentation/submission.csv')

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/gru-lstm-with-feature-engineering-and-augmentation/submission.csv'

## === cell 23
keras_sub = keras_sub.sort_values(by=['id_seqpos']).reset_index(drop=True)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/903524414.py in <cell line: 0>()
----> 1 keras_sub = keras_sub.sort_values(by=['id_seqpos']).reset_index(drop=True)

NameError: name 'keras_sub' is not defined

## === cell 24
keras_sub.head()

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4148739219.py in <cell line: 0>()
----> 1 keras_sub.head()

NameError: name 'keras_sub' is not defined

## === cell 25
pytorch_sub.head()

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/907471660.py in <cell line: 0>()
----> 1 pytorch_sub.head()

NameError: name 'pytorch_sub' is not defined

## === cell 26
np.mean(np.abs(keras_sub[pred_cols].values-pytorch_sub[pred_cols].values))

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3963782787.py in <cell line: 0>()
      1 # submission difference between Keras and Pytorch
----> 2 np.mean(np.abs(keras_sub[pred_cols].values-pytorch_sub[pred_cols].values))

NameError: name 'keras_sub' is not defined

## === cell 27
pytorch_sub.to_csv('./submission.csv',index=False)

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2500541886.py in <cell line: 0>()
----> 1 pytorch_sub.to_csv('./submission.csv',index=False)

NameError: name 'pytorch_sub' is not defined
