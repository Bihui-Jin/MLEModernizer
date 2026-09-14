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

3.9

# 3. Installed packages

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

0.36425

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from keras.utils.vis_utils import plot_model

import tensorflow.keras.layers as L
import keras.backend as K
import tensorflow as tf


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
!pip install spektral -q


## === cell 2
from spektral.layers import GraphConv


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3274357752.py in <cell line: 0>()
----> 1 from spektral.layers import GraphConv

ImportError: cannot import name 'GraphConv' from 'spektral.layers' (/usr/local/lib/python3.11/dist-packages/spektral/layers/__init__.py)

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
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}
sequence_token2int = {x:i for i, x in enumerate('AGUC')}
structure_token2int = {
    '.': 0,
    '(': 1,
    ')': 2,
}
loop_token2int = {x:i for i, x in enumerate('SMIBHEX')}
token2int_map = {
    "sequence": sequence_token2int,
    "structure": structure_token2int,
    "predicted_loop_type": loop_token2int
}
sequence_columns = ["sequence", "structure", "predicted_loop_type"]

def to_seq(df):
    return np.transpose(
        np.array(
            df[sequence_columns]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )

train = to_seq(train_df)
public = to_seq(public_df)
private = to_seq(private_df)

train.shape, public.shape, private.shape


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/226011126.py in <cell line: 0>()
     28 train = to_seq(train_df)
     29 public = to_seq(public_df)
---> 30 private = to_seq(private_df)
     31 
     32 train.shape, public.shape, private.shape

/tmp/ipykernel_11/226011126.py in to_seq(df)
     16 
     17 def to_seq(df):
---> 18     return np.transpose(
     19         np.array(
     20             df[sequence_columns]

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in transpose(a, axes)
    653 
    654     """
--> 655     return _wrapfunc(a, 'transpose', axes)
    656 
    657 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in _wrapfunc(obj, method, *args, **kwds)
     57 
     58     try:
---> 59         return bound(*args, **kwds)
     60     except TypeError:
     61         # A TypeError occurs if the object does have such a method in its

ValueError: axes don't match array

## === cell 9
def to_one_hot(df):
    temp = np.transpose(
        np.array([
            df[col]
            .apply(lambda seq: [token2int_map[col][x] for x in seq])
            .values
            .tolist()
            for col in sequence_columns
        ]),
        (1, 2, 0)
    )
    ohe_1 = tf.keras.utils.to_categorical(temp[:,:,0], 4)
    ohe_2 = tf.keras.utils.to_categorical(temp[:,:,1], 3)
    ohe_3 = tf.keras.utils.to_categorical(temp[:,:,2], 7)
    return np.concatenate([ohe_1, ohe_2, ohe_3], axis=2)

train_ohe = to_one_hot(train_df)
public_ohe = to_one_hot(public_df)
private_ohe = to_one_hot(private_df)

train_ohe.shape, public_ohe.shape, private_ohe.shape


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/722701318.py in <cell line: 0>()
     15     return np.concatenate([ohe_1, ohe_2, ohe_3], axis=2)
     16 
---> 17 train_ohe = to_one_hot(train_df)
     18 public_ohe = to_one_hot(public_df)
     19 private_ohe = to_one_hot(private_df)

/tmp/ipykernel_11/722701318.py in to_one_hot(df)
     10         (1, 2, 0)
     11     )
---> 12     ohe_1 = tf.keras.utils.to_categorical(temp[:,:,0], 4)
     13     ohe_2 = tf.keras.utils.to_categorical(temp[:,:,1], 3)
     14     ohe_3 = tf.keras.utils.to_categorical(temp[:,:,2], 7)

NameError: name 'tf' is not defined

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


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2096409058.py in <cell line: 0>()
     20 train_adj = get_adjacency_matrix(train)
     21 public_adj = get_adjacency_matrix(public)
---> 22 private_adj = get_adjacency_matrix(private)
     23 
     24 train_adj.shape, public_adj.shape, private_adj.shape

NameError: name 'private' is not defined

## === cell 11
train_adj.mean(), public_adj.mean(), private_adj.mean()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3219059323.py in <cell line: 0>()
----> 1 train_adj.mean(), public_adj.mean(), private_adj.mean()

NameError: name 'private_adj' is not defined

## === cell 12
def get_bpps(mRNA_ids):
    bpps = []
    for mRNA_id in tqdm(mRNA_ids):
        bpps.append(
            np.load(f"{bpps_path}/{mRNA_id}.npy"),
        )
    return np.array(bpps)


train_bpps = get_bpps(train_df.id.values)
public_bpps = get_bpps(public_df.id.values)
private_bpps = get_bpps(private_df.id.values)

train_bpps.shape, public_bpps.shape, private_bpps.shape 


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2251112320.py in <cell line: 0>()
      8 
      9 
---> 10 train_bpps = get_bpps(train_df.id.values)
     11 public_bpps = get_bpps(public_df.id.values)
     12 private_bpps = get_bpps(private_df.id.values)

/tmp/ipykernel_11/2251112320.py in get_bpps(mRNA_ids)
      3     for mRNA_id in tqdm(mRNA_ids):
      4         bpps.append(
----> 5             np.load(f"{bpps_path}/{mRNA_id}.npy"),
      6         )
      7     return np.array(bpps)

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/stanford-covid-vaccine/bpps/id_001f94081.npy'

## === cell 13
train_bpps.mean(), public_bpps.mean(), private_bpps.mean() 


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294274073.py in <cell line: 0>()
----> 1 train_bpps.mean(), public_bpps.mean(), private_bpps.mean()

NameError: name 'train_bpps' is not defined

## === cell 14
train_bpps_stats = [train_bpps.mean(axis=2), train_bpps.max(axis=2)]
public_bpps_stats = [public_bpps.mean(axis=2), public_bpps.max(axis=2)]
private_bpps_stats = [private_bpps.mean(axis=2), private_bpps.max(axis=2)]


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2701602045.py in <cell line: 0>()
----> 1 train_bpps_stats = [train_bpps.mean(axis=2), train_bpps.max(axis=2)]
      2 public_bpps_stats = [public_bpps.mean(axis=2), public_bpps.max(axis=2)]
      3 private_bpps_stats = [private_bpps.mean(axis=2), private_bpps.max(axis=2)]

NameError: name 'train_bpps' is not defined

## === cell 15
train_bpps_stats = np.concatenate([stats[:,:,None] for stats in train_bpps_stats], axis=2)
public_bpps_stats = np.concatenate([stats[:,:,None] for stats in public_bpps_stats], axis=2)
private_bpps_stats = np.concatenate([stats[:,:,None] for stats in private_bpps_stats], axis=2)

train_bpps_stats.shape, public_bpps_stats.shape, private_bpps_stats.shape


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1458181739.py in <cell line: 0>()
----> 1 train_bpps_stats = np.concatenate([stats[:,:,None] for stats in train_bpps_stats], axis=2)
      2 public_bpps_stats = np.concatenate([stats[:,:,None] for stats in public_bpps_stats], axis=2)
      3 private_bpps_stats = np.concatenate([stats[:,:,None] for stats in private_bpps_stats], axis=2)
      4 
      5 train_bpps_stats.shape, public_bpps_stats.shape, private_bpps_stats.shape

NameError: name 'train_bpps_stats' is not defined

## === cell 16
scored_seq_length = 68

def rmse(y_actual, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)
    return K.sqrt(mse)

def mcrmse(y_actual, y_pred):
    score = 0
    for i in range(y_actual.shape[2]):
        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / y_actual.shape[2]
    return score
    

def build_model(input_seq_len=107, output_seq_len=scored_seq_length):
    
    def _bi_gru_block(x, hidden_dim, dropout):
        gru = L.Bidirectional(
            L.GRU(hidden_dim, 
                  dropout=dropout,
                  return_sequences=True,
                 ),
        )(x)
        return gru

    def _conv_block(x, adj_m, bpp_m, conv_filters, graph_channels):
        conv = L.Conv1D(
            conv_filters, 5,
            padding='same',
            activation='tanh',
        )(x)
        
        gcn_1 = GraphConv(
            graph_channels,
            activation='tanh',
        )([conv, adj_m])
        
        gcn_2 = GraphConv(
            graph_channels,
            activation='tanh',
        )([conv, bpp_m])

        conv = L.Concatenate()([conv, gcn_1, gcn_2])
        conv = L.Activation("relu")(conv)
        conv = L.SpatialDropout1D(0.1)(conv)
        
        return conv
    
    one_hot_encoding_inputs = L.Input(shape=(input_seq_len, 14), name="onehot")
    adj_matrix_inputs = L.Input((input_seq_len, input_seq_len), name="adjmatrix")
    base_pair_proba_inputs = L.Input((input_seq_len, input_seq_len), name="pairproba")
    base_pair_proba_stats_inputs = L.Input(shape=(input_seq_len, 2), name="pairprobastats")
    
    merged_inputs = L.Concatenate()([one_hot_encoding_inputs, base_pair_proba_stats_inputs])
    
    hidden = _conv_block(merged_inputs, adj_matrix_inputs, base_pair_proba_inputs, 512, 80)
    hidden = _bi_gru_block(hidden, 256, 0.5)
    hidden = _conv_block(hidden, adj_matrix_inputs, base_pair_proba_inputs, 512, 80)
    hidden = _bi_gru_block(hidden, 256, 0.5)
    
    out = hidden[:, :output_seq_len]
    out = L.Dense(5, activation='linear')(out)
    
    model = tf.keras.Model(
        inputs=[
            one_hot_encoding_inputs,
            adj_matrix_inputs,
            base_pair_proba_inputs,
            base_pair_proba_stats_inputs,
        ],
        outputs=out,
    )

    return model

model = build_model()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/498254948.py in <cell line: 0>()
     81     return model
     82 
---> 83 model = build_model()
     84 plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

/tmp/ipykernel_11/498254948.py in build_model(input_seq_len, output_seq_len)
     50 
     51     # inputs
---> 52     one_hot_encoding_inputs = L.Input(shape=(input_seq_len, 14), name="onehot")
     53     # adjacency matrix about seq. connectivity
     54     adj_matrix_inputs = L.Input((input_seq_len, input_seq_len), name="adjmatrix")

NameError: name 'L' is not defined

## === cell 17
split_results  = train_test_split(
    train_ohe,
    train_adj,
    train_bpps,
    train_bpps_stats,
    train_y,
    train_df.signal_to_noise,
    train_df.SN_filter,
    test_size=0.1,
    random_state=42,
)

[a.shape for a in split_results]


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4198383193.py in <cell line: 0>()
      1 split_results  = train_test_split(
----> 2     train_ohe,
      3     train_adj,
      4     train_bpps,
      5     train_bpps_stats,

NameError: name 'train_ohe' is not defined

## === cell 18
trn_ohe, val_ohe, trn_adj, val_adj, trn_bpps, val_bpps, trn_bpps_stats, val_bpps_stats, trn_y, val_y, trn_snr, val_snr, trn_snf, val_snf = split_results


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/744178267.py in <cell line: 0>()
----> 1 trn_ohe, val_ohe, trn_adj, val_adj, trn_bpps, val_bpps, trn_bpps_stats, val_bpps_stats, trn_y, val_y, trn_snr, val_snr, trn_snf, val_snf = split_results

NameError: name 'split_results' is not defined

## === cell 19
model = build_model()
model.compile(tf.keras.optimizers.Adam(), loss=mcrmse)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3915808096.py in <cell line: 0>()
----> 1 model = build_model()
      2 model.compile(tf.keras.optimizers.Adam(), loss=mcrmse)

/tmp/ipykernel_11/498254948.py in build_model(input_seq_len, output_seq_len)
     50 
     51     # inputs
---> 52     one_hot_encoding_inputs = L.Input(shape=(input_seq_len, 14), name="onehot")
     53     # adjacency matrix about seq. connectivity
     54     adj_matrix_inputs = L.Input((input_seq_len, input_seq_len), name="adjmatrix")

NameError: name 'L' is not defined

## === cell 20
trn_inputs = [trn_ohe, trn_adj, trn_bpps, trn_bpps_stats]
val_inputs = [val_ohe, val_adj, val_bpps, val_bpps_stats]


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/457595822.py in <cell line: 0>()
----> 1 trn_inputs = [trn_ohe, trn_adj, trn_bpps, trn_bpps_stats]
      2 val_inputs = [val_ohe, val_adj, val_bpps, val_bpps_stats]

NameError: name 'trn_ohe' is not defined

## === cell 21
val_mask = np.where((val_snf==1))
val_inputs = [val_input[val_mask] for val_input in val_inputs]
val_y = val_y[val_mask]


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1235600028.py in <cell line: 0>()
      1 # only validate on data with sn_filter = 1
----> 2 val_mask = np.where((val_snf==1))
      3 val_inputs = [val_input[val_mask] for val_input in val_inputs]
      4 val_y = val_y[val_mask]

NameError: name 'val_snf' is not defined

## === cell 22
sample_weight = np.log(trn_snr+1.11)/2


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1176129051.py in <cell line: 0>()
----> 1 sample_weight = np.log(trn_snr+1.11)/2

NameError: name 'trn_snr' is not defined

## === cell 23
history = model.fit(
    trn_inputs, trn_y,
    validation_data = (val_inputs, val_y),
    batch_size=64,
    epochs=300,
    sample_weight=sample_weight,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(verbose=1, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint(f'model.h5',save_best_only=True, verbose=0, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(
            patience=20, 
            monitor='val_loss',
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        ),
    ],
    verbose=2
)
print(f"Min validation loss history={min(history.history['val_loss'])}")


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/285360310.py in <cell line: 0>()
----> 1 history = model.fit(
      2     trn_inputs, trn_y,
      3     validation_data = (val_inputs, val_y),
      4     batch_size=64,
      5     epochs=300,

NameError: name 'model' is not defined

## === cell 24
model.load_weights(f'model.h5')


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2412593705.py in <cell line: 0>()
----> 1 model.load_weights(f'model.h5')

NameError: name 'model' is not defined

## === cell 25
val_preds = model.predict(val_inputs)
tf.reduce_mean(mcrmse(val_y, val_preds))


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2405777327.py in <cell line: 0>()
----> 1 val_preds = model.predict(val_inputs)
      2 tf.reduce_mean(mcrmse(val_y, val_preds))

NameError: name 'model' is not defined

## === cell 26
model_public = build_model(107, 107)
model_private = build_model(130, 130)

model_public.load_weights(f'model.h5')
model_private.load_weights(f'model.h5')


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2548369782.py in <cell line: 0>()
----> 1 model_public = build_model(107, 107)
      2 model_private = build_model(130, 130)
      3 
      4 model_public.load_weights(f'model.h5')
      5 model_private.load_weights(f'model.h5')

/tmp/ipykernel_11/498254948.py in build_model(input_seq_len, output_seq_len)
     50 
     51     # inputs
---> 52     one_hot_encoding_inputs = L.Input(shape=(input_seq_len, 14), name="onehot")
     53     # adjacency matrix about seq. connectivity
     54     adj_matrix_inputs = L.Input((input_seq_len, input_seq_len), name="adjmatrix")

NameError: name 'L' is not defined

## === cell 27
public_inputs = [public_ohe, public_adj, public_bpps, public_bpps_stats,]
private_inputs = [private_ohe, private_adj, private_bpps, private_bpps_stats,]


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/794575319.py in <cell line: 0>()
----> 1 public_inputs = [public_ohe, public_adj, public_bpps, public_bpps_stats,]
      2 private_inputs = [private_ohe, private_adj, private_bpps, private_bpps_stats,]

NameError: name 'public_ohe' is not defined

## === cell 28
test_preds = [model_public.predict(public_inputs), model_private.predict(private_inputs)]
test_dfs = [public_df, private_df]

test_preds[0].shape, test_preds[1].shape


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2113852727.py in <cell line: 0>()
----> 1 test_preds = [model_public.predict(public_inputs), model_private.predict(private_inputs)]
      2 test_dfs = [public_df, private_df]
      3 
      4 test_preds[0].shape, test_preds[1].shape

NameError: name 'model_public' is not defined

## === cell 29
preds_ls = []
for df, preds in zip(test_dfs, test_preds):
    for i, uid in tqdm(enumerate(df.id)):
        single_pred = preds[i]
        single_df = pd.DataFrame(single_pred, columns=pred_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]
        preds_ls.append(single_df)
preds_df = pd.concat(preds_ls).groupby('id_seqpos').mean().reset_index()

test_df.shape, preds_df.shape


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2399794238.py in <cell line: 0>()
      1 preds_ls = []
----> 2 for df, preds in zip(test_dfs, test_preds):
      3     for i, uid in tqdm(enumerate(df.id)):
      4         single_pred = preds[i]
      5         single_df = pd.DataFrame(single_pred, columns=pred_cols)

NameError: name 'test_dfs' is not defined

## === cell 30
submission = preds_df[['id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']]
submission.to_csv(f'submission.csv', index=False)
print(f'wrote to submission.csv')


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/307211097.py in <cell line: 0>()
----> 1 submission = preds_df[['id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']]
      2 submission.to_csv(f'submission.csv', index=False)
      3 print(f'wrote to submission.csv')

NameError: name 'preds_df' is not defined

## === cell 31
submission.shape, pd.read_csv(sample_sub_path).shape, test_df.shape


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1171586144.py in <cell line: 0>()
----> 1 submission.shape, pd.read_csv(sample_sub_path).shape, test_df.shape

NameError: name 'submission' is not defined
